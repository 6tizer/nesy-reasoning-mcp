"""Foreground worker for queued Auto-Ingest candidate reviews."""

from __future__ import annotations

import signal
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime, timedelta
from typing import Any

import anyio

from nesy_reasoning_mcp.auto_ingest.openai_agents import (
    ChatCompletionRunner,
    LLMRuntimeOptions,
    OpenAICompatibleProviderConfig,
    ReviewerModelConfig,
    review_gate_and_write_candidate_batch,
)
from nesy_reasoning_mcp.auto_ingest.runner_types import AgentRunner
from nesy_reasoning_mcp.auto_ingest.scheduler import (
    DEFAULT_SCHEDULE_MAX_RETRIES,
    DEFAULT_SCHEDULE_RETRY_BACKOFF_SECONDS,
)
from nesy_reasoning_mcp.auto_ingest.schemas import (
    CandidateRelationBatch,
    ConversationTurnJobStatus,
    GateAction,
    IngestionInput,
    ReviewDecision,
    ReviewQueueRecord,
    ReviewQueueStatus,
    ReviewVotingPolicy,
)
from nesy_reasoning_mcp.schemas import Diagnostic
from nesy_reasoning_mcp.storage.common import _utc_now_iso
from nesy_reasoning_mcp.storage.protocol import RelationStoreProtocol

DEFAULT_REVIEW_WORKER_POLL_SECONDS = 10.0
DEFAULT_REVIEW_WORKER_CLAIM_LIMIT = 20

ProgressCallback = Callable[[dict[str, Any]], None]


class ReviewWorkerRuntimeError(RuntimeError):
    """Raised when review helper reports provider/runtime failure diagnostics."""


@dataclass(frozen=True)
class ReviewWorkerConfig:
    """Runtime controls for one foreground review worker."""

    poll_seconds: float = DEFAULT_REVIEW_WORKER_POLL_SECONDS
    claim_limit: int = DEFAULT_REVIEW_WORKER_CLAIM_LIMIT
    max_retries: int = DEFAULT_SCHEDULE_MAX_RETRIES  # Extra retries after the first failed attempt.
    retry_backoff_seconds: int = DEFAULT_SCHEDULE_RETRY_BACKOFF_SECONDS
    model: str | None = None
    reviewer_models: list[str] | None = None
    reviewer_configs: list[ReviewerModelConfig] | None = None
    high_priority_reviewer_models: list[str] | None = None
    voting_policy: ReviewVotingPolicy = ReviewVotingPolicy.RISK_TIERED
    provider_config: OpenAICompatibleProviderConfig | None = None
    disable_tracing: bool = False
    runtime_options: LLMRuntimeOptions = field(default_factory=LLMRuntimeOptions)
    auto_write: bool = False
    min_write_confidence: float = 0.85
    allow_single_reviewer_write: bool = False

    def __post_init__(self) -> None:
        if self.poll_seconds <= 0:
            raise ValueError("poll_seconds must be greater than 0")
        if self.claim_limit < 0:
            raise ValueError("claim_limit must be non-negative")
        if self.max_retries < 0:
            raise ValueError("max_retries must be non-negative")
        if self.retry_backoff_seconds < 0:
            raise ValueError("retry_backoff_seconds must be non-negative")
        if not 0 <= self.min_write_confidence <= 1:
            raise ValueError("min_write_confidence must be between 0 and 1")
        if self.auto_write and not self.allow_single_reviewer_write:
            reviewer_ids = [
                *[model.strip() for model in self.reviewer_models or [] if model.strip()],
                *[
                    (config.reviewer_id or config.model or "").strip()
                    for config in self.reviewer_configs or []
                    if (config.reviewer_id or config.model or "").strip()
                ],
            ]
            reviewer_count = len(dict.fromkeys(reviewer_ids))
            if reviewer_count < 2:
                raise ValueError(
                    "auto review queue writes require multiple reviewers unless explicitly allowed"
                )
        object.__setattr__(self, "voting_policy", ReviewVotingPolicy(self.voting_policy))


@dataclass(frozen=True)
class ReviewWorkerResult:
    """Structured result from one or more review worker iterations."""

    claimed_record_ids: list[str] = field(default_factory=list)
    reviewed_record_ids: list[str] = field(default_factory=list)
    committed_record_ids: list[str] = field(default_factory=list)
    resolved_record_ids: list[str] = field(default_factory=list)
    pending_record_ids: list[str] = field(default_factory=list)
    failed_record_ids: list[str] = field(default_factory=list)
    done_job_ids: list[str] = field(default_factory=list)
    failed_job_ids: list[str] = field(default_factory=list)
    diagnostics: list[Diagnostic] = field(default_factory=list)
    iterations: int = 1
    interrupted: bool = False

    def merged(self, other: ReviewWorkerResult) -> ReviewWorkerResult:
        """Return a result containing this result followed by another result."""
        return ReviewWorkerResult(
            claimed_record_ids=[*self.claimed_record_ids, *other.claimed_record_ids],
            reviewed_record_ids=[*self.reviewed_record_ids, *other.reviewed_record_ids],
            committed_record_ids=[*self.committed_record_ids, *other.committed_record_ids],
            resolved_record_ids=[*self.resolved_record_ids, *other.resolved_record_ids],
            pending_record_ids=[*self.pending_record_ids, *other.pending_record_ids],
            failed_record_ids=[*self.failed_record_ids, *other.failed_record_ids],
            done_job_ids=[*self.done_job_ids, *other.done_job_ids],
            failed_job_ids=[*self.failed_job_ids, *other.failed_job_ids],
            diagnostics=[*self.diagnostics, *other.diagnostics],
            iterations=self.iterations + other.iterations,
            interrupted=self.interrupted or other.interrupted,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable review worker result."""
        return {
            "status": "interrupted" if self.interrupted else "ok",
            "claimed_record_ids": self.claimed_record_ids,
            "reviewed_record_ids": self.reviewed_record_ids,
            "committed_record_ids": self.committed_record_ids,
            "resolved_record_ids": self.resolved_record_ids,
            "pending_record_ids": self.pending_record_ids,
            "failed_record_ids": self.failed_record_ids,
            "done_job_ids": self.done_job_ids,
            "failed_job_ids": self.failed_job_ids,
            "diagnostics": [
                diagnostic.model_dump(mode="json", exclude_none=True)
                for diagnostic in self.diagnostics
            ],
            "iterations": self.iterations,
        }


async def process_review_queue_once(
    store: RelationStoreProtocol,
    *,
    config: ReviewWorkerConfig | None = None,
    env: Mapping[str, str] | None = None,
    run_agent: AgentRunner | None = None,
    run_chat_completion: ChatCompletionRunner | None = None,
    progress_callback: ProgressCallback | None = None,
) -> ReviewWorkerResult:
    """Claim and process one batch of due review queue records."""
    review_config = config or ReviewWorkerConfig()
    claimed = store.claim_pending_review_queue_records(limit=review_config.claim_limit)
    result = ReviewWorkerResult(claimed_record_ids=[record.id for record in claimed])
    for group in _group_review_records(claimed):
        group_result = await _process_review_group(
            group,
            store,
            config=review_config,
            env=env,
            run_agent=run_agent,
            run_chat_completion=run_chat_completion,
            progress_callback=progress_callback,
        )
        result = result.merged(group_result)
    return result


async def run_review_worker(
    store: RelationStoreProtocol,
    *,
    config: ReviewWorkerConfig | None = None,
    max_records: int | None = None,
    env: Mapping[str, str] | None = None,
    run_agent: AgentRunner | None = None,
    run_chat_completion: ChatCompletionRunner | None = None,
    progress_callback: ProgressCallback | None = None,
) -> ReviewWorkerResult:
    """Run the foreground review worker until stopped or max_records is reached."""
    review_config = config or ReviewWorkerConfig()
    if max_records is not None and max_records < 0:
        raise ValueError("max_records must be non-negative")
    total = ReviewWorkerResult(iterations=0)
    stop_signals: list[str] = []
    async with anyio.create_task_group() as task_group:
        task_group.start_soon(_watch_review_worker_shutdown, task_group.cancel_scope, stop_signals)
        while max_records is None or len(total.reviewed_record_ids) < max_records:
            iteration_config = review_config
            if max_records is not None:
                remaining = max_records - len(total.reviewed_record_ids)
                iteration_config = replace(
                    review_config,
                    claim_limit=min(review_config.claim_limit, remaining),
                )
            batch = await process_review_queue_once(
                store,
                config=iteration_config,
                env=env,
                run_agent=run_agent,
                run_chat_completion=run_chat_completion,
                progress_callback=progress_callback,
            )
            total = total.merged(batch)
            if max_records is not None and len(total.reviewed_record_ids) >= max_records:
                break
            if max_records is not None and not batch.claimed_record_ids:
                break
            await anyio.sleep(review_config.poll_seconds)
        task_group.cancel_scope.cancel()
    if stop_signals:
        total = ReviewWorkerResult(
            claimed_record_ids=total.claimed_record_ids,
            reviewed_record_ids=total.reviewed_record_ids,
            committed_record_ids=total.committed_record_ids,
            resolved_record_ids=total.resolved_record_ids,
            pending_record_ids=total.pending_record_ids,
            failed_record_ids=total.failed_record_ids,
            done_job_ids=total.done_job_ids,
            failed_job_ids=total.failed_job_ids,
            diagnostics=total.diagnostics,
            iterations=total.iterations,
            interrupted=True,
        )
    return total


async def _watch_review_worker_shutdown(
    cancel_scope: anyio.CancelScope,
    stop_signals: list[str],
) -> None:
    try:
        with anyio.open_signal_receiver(signal.SIGINT, signal.SIGTERM) as signals:
            async for signum in signals:
                stop_signals.append(signal.Signals(signum).name)
                cancel_scope.cancel()
                return
    except NotImplementedError:
        return


async def _process_review_group(
    records: list[ReviewQueueRecord],
    store: RelationStoreProtocol,
    *,
    config: ReviewWorkerConfig,
    env: Mapping[str, str] | None,
    run_agent: AgentRunner | None,
    run_chat_completion: ChatCompletionRunner | None,
    progress_callback: ProgressCallback | None,
) -> ReviewWorkerResult:
    record_ids = [record.id for record in records]
    source_job_ids = _source_job_ids(records)
    try:
        _emit(
            progress_callback,
            {"event": "started", "stage": "review_worker", "record_ids": record_ids},
        )
        report = await review_gate_and_write_candidate_batch(
            _ingestion_input_for_records(records),
            CandidateRelationBatch(candidates=[record.candidate for record in records]),
            store=store,
            model=config.model,
            reviewer_models=config.reviewer_models,
            reviewer_configs=config.reviewer_configs,
            voting_policy=config.voting_policy,
            high_priority_reviewer_models=config.high_priority_reviewer_models,
            env=env,
            run_agent=run_agent,
            run_chat_completion=run_chat_completion,
            auto_write=config.auto_write,
            min_write_confidence=config.min_write_confidence,
            provider_config=config.provider_config,
            disable_tracing=config.disable_tracing,
            runtime_options=config.runtime_options,
            progress_callback=progress_callback,
            canonicalize_preview=config.auto_write,
            enqueue_queued_records=False,
        )
        _raise_for_runtime_diagnostics(report.diagnostics)
        _raise_for_write_failure(report)
        updates = _updated_records_from_report(records, report, config)
        store.update_review_queue_records(updates, expected_status=ReviewQueueStatus.REVIEWING)
        done_job_ids, failed_job_ids = _writeback_source_jobs_for_records(
            store,
            updates,
            failed=False,
        )
        _emit(
            progress_callback,
            {
                "event": "done",
                "stage": "review_worker",
                "record_ids": record_ids,
                "job_ids": source_job_ids,
            },
        )
        return ReviewWorkerResult(
            reviewed_record_ids=record_ids,
            committed_record_ids=[
                record.id for record in updates if record.status == ReviewQueueStatus.COMMITTED
            ],
            resolved_record_ids=[
                record.id for record in updates if record.status == ReviewQueueStatus.RESOLVED
            ],
            pending_record_ids=[
                record.id for record in updates if record.status == ReviewQueueStatus.PENDING
            ],
            failed_record_ids=[
                record.id for record in updates if record.status == ReviewQueueStatus.FAILED
            ],
            done_job_ids=done_job_ids,
            failed_job_ids=failed_job_ids,
            diagnostics=report.diagnostics,
            iterations=0,
        )
    except Exception as exc:
        failed_records, retrying_records, diagnostic = _failure_updates(records, config, exc)
        store.update_review_queue_records(
            [*failed_records, *retrying_records],
            expected_status=ReviewQueueStatus.REVIEWING,
        )
        done_job_ids, failed_job_ids = _writeback_source_jobs_for_records(
            store,
            failed_records,
            failed=True,
        )
        _emit(
            progress_callback,
            {"event": "error", "stage": "review_worker", "record_ids": record_ids},
        )
        return ReviewWorkerResult(
            reviewed_record_ids=record_ids,
            pending_record_ids=[record.id for record in retrying_records],
            failed_record_ids=[record.id for record in failed_records],
            done_job_ids=done_job_ids,
            failed_job_ids=failed_job_ids,
            diagnostics=[diagnostic],
            iterations=0,
        )


def _updated_records_from_report(
    records: list[ReviewQueueRecord],
    report: Any,
    config: ReviewWorkerConfig,
) -> list[ReviewQueueRecord]:
    reviews_by_candidate = _reviews_by_candidate(report)
    gates_by_candidate = {gate.candidate_id: gate for gate in report.gate_results}
    relation_ids_by_candidate = _written_relation_ids_by_candidate(report)
    updated_at = datetime.now(UTC)
    timestamp = updated_at.isoformat()
    retry_at = (updated_at + timedelta(seconds=config.retry_backoff_seconds)).isoformat()
    updates: list[ReviewQueueRecord] = []
    for record in records:
        gate = gates_by_candidate.get(record.candidate.id) or record.gate_result
        review = reviews_by_candidate.get(record.candidate.id) or record.review
        if gate.action == GateAction.AUTO_WRITE:
            status = ReviewQueueStatus.COMMITTED
            committed_relation_ids = relation_ids_by_candidate.get(record.candidate.id, [])
            resolution: dict[str, Any] = {}
            next_retry_at = None
        elif gate.action == GateAction.REJECT:
            status = ReviewQueueStatus.RESOLVED
            committed_relation_ids = []
            resolution = {"reason": "auto_rejected", "resolved_at": timestamp}
            next_retry_at = None
        else:
            status = ReviewQueueStatus.PENDING
            committed_relation_ids = []
            resolution = {}
            next_retry_at = retry_at
        updates.append(
            record.model_copy(
                deep=True,
                update={
                    "status": status,
                    "review": review,
                    "gate_result": gate,
                    "diagnostics": [
                        *record.diagnostics,
                        *_diagnostics_for_candidate(report.diagnostics, record.candidate.id),
                    ],
                    "committed_relation_ids": committed_relation_ids,
                    "resolution": resolution,
                    "next_retry_at": next_retry_at,
                    "updated_at": timestamp,
                },
            )
        )
    return updates


def _failure_updates(
    records: list[ReviewQueueRecord],
    config: ReviewWorkerConfig,
    exc: Exception,
) -> tuple[list[ReviewQueueRecord], list[ReviewQueueRecord], Diagnostic]:
    timestamp = _utc_now_iso()
    retry_at = (datetime.now(UTC) + timedelta(seconds=config.retry_backoff_seconds)).isoformat()
    failed: list[ReviewQueueRecord] = []
    retrying: list[ReviewQueueRecord] = []
    diagnostic = Diagnostic(
        level="error",
        code="REVIEW_WORKER_RECORD_FAILED",
        message=f"review queue processing failed: {exc.__class__.__name__}",
        related_ids=[record.id for record in records],
    )
    for record in records:
        exhausted = record.attempt_count > config.max_retries
        update = record.model_copy(
            deep=True,
            update={
                "status": ReviewQueueStatus.FAILED if exhausted else ReviewQueueStatus.PENDING,
                "diagnostics": [*record.diagnostics, diagnostic],
                "next_retry_at": None if exhausted else retry_at,
                "updated_at": timestamp,
            },
        )
        if exhausted:
            failed.append(update)
        else:
            retrying.append(update)
    return failed, retrying, diagnostic


def _raise_for_runtime_diagnostics(diagnostics: list[Diagnostic]) -> None:
    if any(
        diagnostic.code in {"LLM_RUNTIME_ERROR", "LLM_RUNTIME_TIMEOUT"}
        for diagnostic in diagnostics
    ):
        raise ReviewWorkerRuntimeError("review helper reported runtime failure")


def _raise_for_write_failure(report: Any) -> None:
    write_result = report.metadata.get("write_result", {})
    if isinstance(write_result, dict) and write_result.get("status") == "error":
        raise ReviewWorkerRuntimeError("review helper reported write failure")
    approved_count = len(report.approved_relations)
    if approved_count and len(report.written_relation_ids) < approved_count:
        raise ReviewWorkerRuntimeError("review helper did not return written relation ids")


def _writeback_source_jobs_for_records(
    store: RelationStoreProtocol,
    records: list[ReviewQueueRecord],
    *,
    failed: bool,
) -> tuple[list[str], list[str]]:
    done_job_ids: list[str] = []
    failed_job_ids: list[str] = []
    source_job_ids = _source_job_ids(records)
    related_by_job_id: dict[str, list[ReviewQueueRecord]] = {
        job_id: [] for job_id in source_job_ids
    }
    if not failed:
        pending_job_ids = set(source_job_ids)
        for record in store.list_review_queue():
            for job_id in pending_job_ids.intersection(record.source_job_ids):
                related_by_job_id[job_id].append(record)
    for job_id in source_job_ids:
        if failed:
            updated = store.update_ingestion_job_status(
                job_id,
                ConversationTurnJobStatus.FAILED,
                expected_status=ConversationTurnJobStatus.REVIEWING,
            )
            if updated is not None:
                failed_job_ids.append(job_id)
            continue
        related = related_by_job_id[job_id]
        if related and all(
            record.status in {ReviewQueueStatus.COMMITTED, ReviewQueueStatus.RESOLVED}
            for record in related
        ):
            updated = store.update_ingestion_job_status(
                job_id,
                ConversationTurnJobStatus.DONE,
                expected_status=ConversationTurnJobStatus.REVIEWING,
            )
            if updated is not None:
                done_job_ids.append(job_id)
    return done_job_ids, failed_job_ids


def _group_review_records(records: list[ReviewQueueRecord]) -> list[list[ReviewQueueRecord]]:
    groups: dict[tuple[str, tuple[str, ...]], list[ReviewQueueRecord]] = {}
    for record in records:
        key = (record.run_id, tuple(record.source_job_ids))
        groups.setdefault(key, []).append(record)
    return list(groups.values())


def _ingestion_input_for_records(records: list[ReviewQueueRecord]) -> IngestionInput:
    evidence = [evidence for record in records for evidence in record.candidate.evidence]
    context_metadata: dict[str, Any] = {}
    for record in records:
        context_metadata.update(record.context_metadata)
    return IngestionInput(
        task="review queued conversation ingestion candidates",
        evidence=evidence,
        propositions=[proposition for record in records for proposition in record.propositions],
        context_metadata=context_metadata,
        metadata={"review_queue_record_ids": [record.id for record in records]},
    )


def _reviews_by_candidate(report: Any) -> dict[str, ReviewDecision]:
    aggregation = report.metadata.get("review_aggregation", {})
    aggregate_reviews = (
        aggregation.get("aggregate_reviews") if isinstance(aggregation, dict) else []
    )
    reviews: list[ReviewDecision] = []
    for item in aggregate_reviews if isinstance(aggregate_reviews, list) else []:
        if isinstance(item, dict):
            reviews.append(ReviewDecision.model_validate(item))
    if not reviews:
        reviews = list(report.reviews)
    return {review.candidate_id: review for review in reviews}


def _written_relation_ids_by_candidate(report: Any) -> dict[str, list[str]]:
    relation_ids: dict[str, list[str]] = {}
    for relation, relation_id in zip(
        report.approved_relations,
        report.written_relation_ids,
        strict=False,
    ):
        candidate_id = relation.provenance.get("candidate_id")
        if candidate_id:
            relation_ids.setdefault(str(candidate_id), []).append(relation_id)
    return relation_ids


def _diagnostics_for_candidate(
    diagnostics: list[Diagnostic],
    candidate_id: str,
) -> list[Diagnostic]:
    return [diagnostic for diagnostic in diagnostics if candidate_id in diagnostic.related_ids]


def _source_job_ids(records: list[ReviewQueueRecord]) -> list[str]:
    return list(dict.fromkeys(job_id for record in records for job_id in record.source_job_ids))


def _emit(progress_callback: ProgressCallback | None, event: dict[str, Any]) -> None:
    if progress_callback is not None:
        progress_callback(event)
