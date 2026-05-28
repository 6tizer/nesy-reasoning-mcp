"""Foreground worker for queued Auto-Ingest conversation turns."""

from __future__ import annotations

import json
import os
import signal
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import anyio

from nesy_reasoning_mcp.auto_ingest.extraction import (
    ChatCompletionRunner,
    ExtractionModelConfig,
    build_transcript_context_window,
    extract_candidate_relations_with_context_metadata,
)
from nesy_reasoning_mcp.auto_ingest.nesy_facts import extract_nesy_facts
from nesy_reasoning_mcp.auto_ingest.schemas import (
    CandidateRelation,
    CandidateRelationBatch,
    ConversationTurnJob,
    ConversationTurnJobStatus,
    EvidenceRecord,
    GateAction,
    GateResult,
    ReviewQueueRecord,
)
from nesy_reasoning_mcp.auto_ingest.semantic_dedupe import semantic_duplicate_concerns
from nesy_reasoning_mcp.schemas import Diagnostic, PropositionRecord, RelationInput
from nesy_reasoning_mcp.storage.common import _utc_now_iso
from nesy_reasoning_mcp.storage.protocol import RelationStoreProtocol

DEFAULT_INGEST_WORKER_POLL_SECONDS = 10.0
DEFAULT_INGEST_WORKER_QUEUE_MAX_DEPTH = 50
DEFAULT_INGEST_WORKER_CLAIM_LIMIT = 5
DEFAULT_INGEST_WORKER_MAX_MERGE_JOBS = 3

ProgressCallback = Callable[[dict[str, Any]], None]


@dataclass(frozen=True)
class IngestionWorkerConfig:
    """Runtime controls for one foreground ingestion worker."""

    poll_seconds: float = DEFAULT_INGEST_WORKER_POLL_SECONDS
    queue_max_depth: int | None = None
    claim_limit: int = DEFAULT_INGEST_WORKER_CLAIM_LIMIT
    max_merge_jobs: int = DEFAULT_INGEST_WORKER_MAX_MERGE_JOBS
    extraction_config: ExtractionModelConfig = field(default_factory=ExtractionModelConfig)

    def __post_init__(self) -> None:
        if self.poll_seconds <= 0:
            raise ValueError("poll_seconds must be greater than 0")
        if self.queue_max_depth is not None and self.queue_max_depth < 0:
            raise ValueError("queue_max_depth must be non-negative")
        if self.claim_limit < 0:
            raise ValueError("claim_limit must be non-negative")
        if self.max_merge_jobs <= 0:
            raise ValueError("max_merge_jobs must be greater than 0")

    def resolved_queue_max_depth(self, env: Mapping[str, str] | None = None) -> int:
        """Return configured queue depth, honoring the worker environment default."""
        if self.queue_max_depth is not None:
            return self.queue_max_depth
        runtime_env = os.environ if env is None else env
        raw = runtime_env.get("NESY_INGEST_QUEUE_MAX_DEPTH")
        if raw is None:
            return DEFAULT_INGEST_WORKER_QUEUE_MAX_DEPTH
        try:
            value = int(raw)
        except ValueError:
            return DEFAULT_INGEST_WORKER_QUEUE_MAX_DEPTH
        return value if value >= 0 else DEFAULT_INGEST_WORKER_QUEUE_MAX_DEPTH


@dataclass(frozen=True)
class IngestionWorkerResult:
    """Structured result from one or more ingestion worker iterations."""

    claimed_job_ids: list[str] = field(default_factory=list)
    processed_job_ids: list[str] = field(default_factory=list)
    queued_record_ids: list[str] = field(default_factory=list)
    dropped_job_ids: list[str] = field(default_factory=list)
    failed_job_ids: list[str] = field(default_factory=list)
    diagnostics: list[Diagnostic] = field(default_factory=list)
    iterations: int = 1
    interrupted: bool = False

    def merged(self, other: IngestionWorkerResult) -> IngestionWorkerResult:
        """Return a result containing this result followed by another result."""
        return IngestionWorkerResult(
            claimed_job_ids=[*self.claimed_job_ids, *other.claimed_job_ids],
            processed_job_ids=[*self.processed_job_ids, *other.processed_job_ids],
            queued_record_ids=[*self.queued_record_ids, *other.queued_record_ids],
            dropped_job_ids=[*self.dropped_job_ids, *other.dropped_job_ids],
            failed_job_ids=[*self.failed_job_ids, *other.failed_job_ids],
            diagnostics=[*self.diagnostics, *other.diagnostics],
            iterations=self.iterations + other.iterations,
            interrupted=self.interrupted or other.interrupted,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable worker result."""
        return {
            "status": "interrupted" if self.interrupted else "ok",
            "claimed_job_ids": self.claimed_job_ids,
            "processed_job_ids": self.processed_job_ids,
            "queued_record_ids": self.queued_record_ids,
            "dropped_job_ids": self.dropped_job_ids,
            "failed_job_ids": self.failed_job_ids,
            "diagnostics": [
                diagnostic.model_dump(mode="json", exclude_none=True)
                for diagnostic in self.diagnostics
            ],
            "iterations": self.iterations,
        }


async def process_ingestion_queue_once(
    store: RelationStoreProtocol,
    *,
    config: IngestionWorkerConfig | None = None,
    env: Mapping[str, str] | None = None,
    run_chat_completion: ChatCompletionRunner | None = None,
    progress_callback: ProgressCallback | None = None,
) -> IngestionWorkerResult:
    """Claim and process one batch of queued conversation turn jobs."""
    worker_config = config or IngestionWorkerConfig()
    diagnostics: list[Diagnostic] = []
    dropped = store.drop_pending_ingestion_jobs_over_depth(
        worker_config.resolved_queue_max_depth(env)
    )
    if dropped:
        diagnostics.append(
            Diagnostic(
                level="warning",
                code="INGESTION_QUEUE_BACKPRESSURE_DROP",
                message="Dropped pending conversation turn jobs over queue depth",
                related_ids=[record.job_id for record in dropped],
            )
        )
    claimed = store.claim_pending_ingestion_jobs(limit=worker_config.claim_limit)
    groups = _merge_claimed_jobs(claimed, max_merge_jobs=worker_config.max_merge_jobs)
    result = IngestionWorkerResult(
        claimed_job_ids=[record.job_id for record in claimed],
        dropped_job_ids=[record.job_id for record in dropped],
        diagnostics=diagnostics,
    )
    for group in groups:
        group_result = await _process_job_group(
            group,
            store,
            config=worker_config,
            env=env,
            run_chat_completion=run_chat_completion,
            progress_callback=progress_callback,
        )
        result = result.merged(group_result)
    return result


async def run_ingestion_worker(
    store: RelationStoreProtocol,
    *,
    config: IngestionWorkerConfig | None = None,
    max_jobs: int | None = None,
    env: Mapping[str, str] | None = None,
    run_chat_completion: ChatCompletionRunner | None = None,
    progress_callback: ProgressCallback | None = None,
) -> IngestionWorkerResult:
    """Run the foreground ingestion worker until stopped or max_jobs is reached."""
    worker_config = config or IngestionWorkerConfig()
    if max_jobs is not None and max_jobs < 0:
        raise ValueError("max_jobs must be non-negative")
    total = IngestionWorkerResult(iterations=0)
    stop_signals: list[str] = []
    async with anyio.create_task_group() as task_group:
        task_group.start_soon(_watch_worker_shutdown, task_group.cancel_scope, stop_signals)
        while max_jobs is None or len(total.processed_job_ids) < max_jobs:
            iteration_config = worker_config
            if max_jobs is not None:
                remaining = max_jobs - len(total.processed_job_ids)
                iteration_config = IngestionWorkerConfig(
                    poll_seconds=worker_config.poll_seconds,
                    queue_max_depth=worker_config.queue_max_depth,
                    claim_limit=min(worker_config.claim_limit, remaining),
                    max_merge_jobs=worker_config.max_merge_jobs,
                    extraction_config=worker_config.extraction_config,
                )
            batch = await process_ingestion_queue_once(
                store,
                config=iteration_config,
                env=env,
                run_chat_completion=run_chat_completion,
                progress_callback=progress_callback,
            )
            total = total.merged(batch)
            if max_jobs is not None and len(total.processed_job_ids) >= max_jobs:
                break
            if max_jobs is not None and not batch.claimed_job_ids:
                break
            await anyio.sleep(worker_config.poll_seconds)
        task_group.cancel_scope.cancel()
    if stop_signals:
        total = IngestionWorkerResult(
            claimed_job_ids=total.claimed_job_ids,
            processed_job_ids=total.processed_job_ids,
            queued_record_ids=total.queued_record_ids,
            dropped_job_ids=total.dropped_job_ids,
            failed_job_ids=total.failed_job_ids,
            diagnostics=total.diagnostics,
            iterations=total.iterations,
            interrupted=True,
        )
    return total


async def _watch_worker_shutdown(
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


async def _process_job_group(
    jobs: list[ConversationTurnJob],
    store: RelationStoreProtocol,
    *,
    config: IngestionWorkerConfig,
    env: Mapping[str, str] | None,
    run_chat_completion: ChatCompletionRunner | None,
    progress_callback: ProgressCallback | None,
) -> IngestionWorkerResult:
    job_ids = [job.job_id for job in jobs]
    try:
        _emit(
            progress_callback, {"event": "started", "stage": "ingestion_worker", "job_ids": job_ids}
        )
        batch, propositions, diagnostics = await _candidate_batch_for_jobs(
            jobs,
            config=config,
            env=env,
            run_chat_completion=run_chat_completion,
        )
        records = _review_queue_records_for_candidates(
            batch.candidates,
            jobs=jobs,
            store=store,
            propositions=propositions,
            diagnostics=diagnostics,
        )
        queued_record_ids: list[str] = []
        if records:
            stored, _updated = store.enqueue_review_queue(records)
            queued_record_ids = [record.id for record in stored]
        final_status = (
            ConversationTurnJobStatus.REVIEWING
            if queued_record_ids
            else ConversationTurnJobStatus.DONE
        )
        for job in jobs:
            store.update_ingestion_job_status(
                job.job_id,
                final_status,
                expected_status=ConversationTurnJobStatus.EXTRACTING,
            )
        _emit(
            progress_callback,
            {
                "event": "done",
                "stage": "ingestion_worker",
                "job_ids": job_ids,
                "queued_count": len(queued_record_ids),
            },
        )
        return IngestionWorkerResult(
            processed_job_ids=job_ids,
            queued_record_ids=queued_record_ids,
            diagnostics=diagnostics,
            iterations=0,
        )
    except Exception as exc:
        diagnostic = Diagnostic(
            level="error",
            code="INGESTION_WORKER_JOB_FAILED",
            message=f"conversation ingestion job failed: {exc.__class__.__name__}",
            related_ids=job_ids,
        )
        for job in jobs:
            store.update_ingestion_job_status(
                job.job_id,
                ConversationTurnJobStatus.FAILED,
                expected_status=ConversationTurnJobStatus.EXTRACTING,
            )
        _emit(
            progress_callback,
            {"event": "error", "stage": "ingestion_worker", "job_ids": job_ids},
        )
        return IngestionWorkerResult(
            processed_job_ids=job_ids,
            failed_job_ids=job_ids,
            diagnostics=[diagnostic],
            iterations=0,
        )


async def _candidate_batch_for_jobs(
    jobs: list[ConversationTurnJob],
    *,
    config: IngestionWorkerConfig,
    env: Mapping[str, str] | None,
    run_chat_completion: ChatCompletionRunner | None,
) -> tuple[CandidateRelationBatch, list[PropositionRecord], list[Diagnostic]]:
    if any(job.skip_extraction for job in jobs):
        return _fast_path_candidate_batch(jobs)
    context = build_transcript_context_window(jobs[0].transcript_path)
    result = await extract_candidate_relations_with_context_metadata(
        context,
        config=config.extraction_config,
        env=env,
        run_chat_completion=run_chat_completion,
    )
    diagnostics = [
        *result.diagnostics,
        *(
            [
                Diagnostic(
                    level="info",
                    code="TRANSCRIPT_CONTEXT_COMPACTION_RECOMMENDED",
                    message="Extraction context was truncated before LLM extraction",
                    related_ids=[job.job_id for job in jobs],
                )
            ]
            if result.compaction_recommended
            else []
        ),
    ]
    return result.candidate_batch, [], diagnostics


def _fast_path_candidate_batch(
    jobs: list[ConversationTurnJob],
) -> tuple[CandidateRelationBatch, list[PropositionRecord], list[Diagnostic]]:
    message = _latest_assistant_message(Path(jobs[0].transcript_path))
    payload = extract_nesy_facts(message)
    if payload is None:
        raise ValueError("skip_extraction job did not contain NESY_FACTS")
    evidence = EvidenceRecord(
        url="conversation://current-transcript",
        span=message,
        source_type="conversation_transcript",
        metadata={"job_ids": [job.job_id for job in jobs], "fast_path": "nesy_facts"},
    )
    candidates: list[CandidateRelation] = []
    for index, item in enumerate(payload["relations"]):
        relation = RelationInput.model_validate(item)
        candidates.append(
            CandidateRelation(
                id=relation.id or f"cand_{jobs[0].job_id}_{index}",
                source=relation.source,
                source_id=relation.source_id,
                target=relation.target,
                target_id=relation.target_id,
                relation_type=relation.relation_type,
                confidence=relation.confidence,
                context_id=relation.context_id,
                store_id=relation.store_id,
                evidence=[evidence],
                metadata={
                    **relation.metadata,
                    "provenance": relation.provenance or {},
                    "fast_path": "nesy_facts",
                    "job_ids": [job.job_id for job in jobs],
                },
            )
        )
    propositions = [PropositionRecord.model_validate(item) for item in payload["propositions"]]
    return CandidateRelationBatch(candidates=candidates), propositions, []


def _review_queue_records_for_candidates(
    candidates: list[CandidateRelation],
    *,
    jobs: list[ConversationTurnJob],
    store: RelationStoreProtocol,
    propositions: list[PropositionRecord],
    diagnostics: list[Diagnostic],
) -> list[ReviewQueueRecord]:
    concerns = semantic_duplicate_concerns(
        relations=[candidate.to_relation_input() for candidate in candidates],
        existing_relations=store.list_relations(),
        propositions=[*store.list_propositions(), *propositions],
    )
    run_id = f"turn_ing_{jobs[0].job_id}"
    generated_at = _utc_now_iso()
    records: list[ReviewQueueRecord] = []
    for candidate, concern in zip(candidates, concerns, strict=True):
        metadata: dict[str, Any] = {
            "job_ids": [job.job_id for job in jobs],
            "session_id": jobs[0].session_id,
            "transcript_path": jobs[0].transcript_path,
        }
        reasons = ["conversation ingestion worker queued candidate"]
        if concern is not None:
            metadata["semantic_duplicate"] = concern.to_metadata()
            reasons.append("likely semantic duplicate relation requires human review")
        records.append(
            ReviewQueueRecord(
                run_id=run_id,
                run_metadata={
                    "generated_at": generated_at,
                    "mode": "conversation_worker",
                    "metadata": metadata,
                    "diagnostic_count": len(diagnostics),
                },
                candidate=candidate,
                review=None,
                gate_result=GateResult(
                    candidate_id=candidate.id,
                    action=GateAction.QUEUE,
                    reasons=reasons,
                    metadata=metadata,
                ),
                diagnostics=[
                    diagnostic
                    for diagnostic in diagnostics
                    if not diagnostic.related_ids or candidate.id in diagnostic.related_ids
                ],
                propositions=propositions,
                context_metadata=metadata,
            )
        )
    return records


def _merge_claimed_jobs(
    jobs: list[ConversationTurnJob],
    *,
    max_merge_jobs: int,
) -> list[list[ConversationTurnJob]]:
    groups: list[list[ConversationTurnJob]] = []
    for job in jobs:
        if not groups or not _can_merge(groups[-1], job, max_merge_jobs=max_merge_jobs):
            groups.append([job])
        else:
            groups[-1].append(job)
    return groups


def _can_merge(
    group: list[ConversationTurnJob],
    job: ConversationTurnJob,
    *,
    max_merge_jobs: int,
) -> bool:
    previous = group[-1]
    if len(group) >= max_merge_jobs:
        return False
    if previous.session_id != job.session_id or previous.transcript_path != job.transcript_path:
        return False
    if previous.skip_extraction != job.skip_extraction:
        return False
    if previous.turn_index is None and job.turn_index is None:
        return True
    if previous.turn_index is None or job.turn_index is None:
        return False
    return job.turn_index == previous.turn_index + 1


def _latest_assistant_message(path: Path) -> str:
    latest: str | None = None
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        role, content = _payload_role_content(payload)
        if role == "assistant":
            text = _content_to_text(content).strip()
            if text:
                latest = text
    if latest is None:
        raise ValueError("transcript does not contain an assistant message")
    return latest


def _payload_role_content(payload: Any) -> tuple[str | None, Any]:
    if not isinstance(payload, Mapping):
        return None, None
    role = payload.get("role")
    content = payload.get("content")
    message = payload.get("message")
    if (not isinstance(role, str) or content is None) and isinstance(message, Mapping):
        role = message.get("role")
        content = message.get("content")
    return role if isinstance(role, str) else None, content


def _content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, Mapping):
        text = content.get("text")
        if text is None:
            text = content.get("content")
        return _content_to_text(text)
    if isinstance(content, list):
        return "\n".join(text for item in content if (text := _content_to_text(item).strip()))
    return ""


def _emit(progress_callback: ProgressCallback | None, event: dict[str, Any]) -> None:
    if progress_callback is not None:
        progress_callback(event)
