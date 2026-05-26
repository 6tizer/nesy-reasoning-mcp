"""OpenAI Agents SDK orchestration for dry-run candidate ingestion."""

from __future__ import annotations

import json
import os
import re
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from inspect import Parameter, signature
from time import perf_counter
from types import MappingProxyType
from typing import Any, TypeVar

import anyio
from pydantic import BaseModel, ValidationError

from nesy_reasoning_mcp.auto_ingest.canonicalization import (
    PropositionCanonicalizationBatch,
    PropositionCanonicalizationResult,
    canonicalization_prompt,
    canonicalize_candidate_relations,
)
from nesy_reasoning_mcp.auto_ingest.gate import run_dry_run_gate
from nesy_reasoning_mcp.auto_ingest.providers import ProviderStructuredOutputMode
from nesy_reasoning_mcp.auto_ingest.review_queue import queued_records_from_report
from nesy_reasoning_mcp.auto_ingest.review_voting import aggregate_review_decisions
from nesy_reasoning_mcp.auto_ingest.schemas import (
    CandidateRelation,
    CandidateRelationBatch,
    IngestionInput,
    IngestionMode,
    IngestionReport,
    ReviewDecision,
    ReviewDecisionBatch,
    ReviewDecisionValue,
    ReviewVotingPolicy,
)
from nesy_reasoning_mcp.auto_ingest.text import dedupe_non_empty_text
from nesy_reasoning_mcp.auto_ingest.writer import write_approved_relations
from nesy_reasoning_mcp.normalization import normalized_implication_preview
from nesy_reasoning_mcp.schemas import DEFAULT_STORE_ID, Diagnostic, PropositionRecord
from nesy_reasoning_mcp.store import RelationStoreProtocol

AgentRunner = Callable[..., Awaitable[Any]]
ChatCompletionRunner = Callable[..., Awaitable[Any]]
ProgressCallback = Callable[[dict[str, Any]], None]
OutputBatch = TypeVar("OutputBatch", bound=BaseModel)

_CANONICALIZATION_TOKEN_RE = re.compile(r"[a-z0-9]+")
_CANONICALIZATION_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "be",
        "by",
        "for",
        "from",
        "in",
        "is",
        "of",
        "on",
        "or",
        "the",
        "to",
        "with",
    }
)


@dataclass(frozen=True)
class LLMRuntimeOptions:
    """Timeout, token, and progress controls for ingestion LLM calls."""

    extractor_timeout_seconds: float = 180
    high_priority_reviewer_timeout_seconds: float = 180
    reviewer_timeout_seconds: float = 120
    extractor_max_tokens: int = 4096
    reviewer_max_tokens: int = 2048
    progress_mode: str = "auto"

    def __post_init__(self) -> None:
        for name in (
            "extractor_timeout_seconds",
            "high_priority_reviewer_timeout_seconds",
            "reviewer_timeout_seconds",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be greater than 0")
        for name in ("extractor_max_tokens", "reviewer_max_tokens"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be greater than 0")
        if self.progress_mode not in {"auto", "off"}:
            raise ValueError("progress_mode must be auto or off")


@dataclass(frozen=True)
class OpenAICompatibleProviderConfig:
    """Configuration for OpenAI-compatible Chat Completions providers."""

    base_url: str
    api_key_env: str
    default_headers: Mapping[str, str] = field(default_factory=dict)
    disable_tracing: bool = True
    structured_output_mode: ProviderStructuredOutputMode = ProviderStructuredOutputMode.AGENT_SCHEMA
    reasoning_effort: str | None = None
    extra_body: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "default_headers",
            MappingProxyType(dict(self.default_headers)),
        )
        object.__setattr__(
            self,
            "structured_output_mode",
            ProviderStructuredOutputMode(self.structured_output_mode),
        )
        object.__setattr__(self, "extra_body", MappingProxyType(dict(self.extra_body)))


@dataclass(frozen=True)
class ReviewerModelConfig:
    """Resolved model/provider settings for one reviewer call."""

    reviewer_id: str | None
    model: str | None
    provider_name: str | None = None
    provider_config: OpenAICompatibleProviderConfig | None = None


class OpenAIAgentsDryRunError(ValueError):
    """Raised when a live Agent SDK dry-run cannot start safely."""


class _LLMRuntimeStageError(OpenAIAgentsDryRunError):
    """Raised when one traced LLM stage fails before producing structured output."""

    def __init__(
        self,
        message: str,
        *,
        code: str,
        status: str,
        stage: str,
        label: str,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.status = status
        self.stage = stage
        self.label = label


@dataclass(frozen=True)
class _ReviewerCallResult:
    config: ReviewerModelConfig
    reviews: list[ReviewDecision]
    diagnostics: list[Diagnostic]
    failed: bool = False


async def run_openai_agents_dry_run(
    ingestion_input: IngestionInput,
    *,
    store: RelationStoreProtocol,
    model: str | None = None,
    reviewer_models: list[str] | None = None,
    reviewer_configs: list[ReviewerModelConfig] | None = None,
    voting_policy: ReviewVotingPolicy = ReviewVotingPolicy.RISK_TIERED,
    high_priority_reviewer_models: list[str] | None = None,
    env: Mapping[str, str] | None = None,
    run_agent: AgentRunner | None = None,
    run_chat_completion: ChatCompletionRunner | None = None,
    provider_config: OpenAICompatibleProviderConfig | None = None,
    disable_tracing: bool = False,
    runtime_options: LLMRuntimeOptions | None = None,
    progress_callback: ProgressCallback | None = None,
    canonicalize_preview: bool = False,
) -> IngestionReport:
    """Extract, review, and gate candidate relations without persistent writes."""
    return await run_openai_agents_ingestion(
        ingestion_input,
        store=store,
        model=model,
        reviewer_models=reviewer_models,
        reviewer_configs=reviewer_configs,
        voting_policy=voting_policy,
        high_priority_reviewer_models=high_priority_reviewer_models,
        env=env,
        run_agent=run_agent,
        run_chat_completion=run_chat_completion,
        provider_config=provider_config,
        disable_tracing=disable_tracing,
        runtime_options=runtime_options,
        progress_callback=progress_callback,
        canonicalize_preview=canonicalize_preview,
        auto_write=False,
    )


async def run_openai_agents_ingestion(
    ingestion_input: IngestionInput,
    *,
    store: RelationStoreProtocol,
    model: str | None = None,
    reviewer_models: list[str] | None = None,
    reviewer_configs: list[ReviewerModelConfig] | None = None,
    voting_policy: ReviewVotingPolicy = ReviewVotingPolicy.RISK_TIERED,
    high_priority_reviewer_models: list[str] | None = None,
    env: Mapping[str, str] | None = None,
    run_agent: AgentRunner | None = None,
    run_chat_completion: ChatCompletionRunner | None = None,
    auto_write: bool = False,
    min_write_confidence: float = 0.85,
    provider_config: OpenAICompatibleProviderConfig | None = None,
    disable_tracing: bool = False,
    runtime_options: LLMRuntimeOptions | None = None,
    progress_callback: ProgressCallback | None = None,
    canonicalize_preview: bool = False,
) -> IngestionReport:
    """Extract, review, gate, and optionally write approved candidate relations."""
    if not 0 <= min_write_confidence <= 1:
        raise OpenAIAgentsDryRunError("min_write_confidence must be between 0 and 1")
    runtime_options = runtime_options or LLMRuntimeOptions()
    effective_progress_callback = (
        progress_callback if runtime_options.progress_mode != "off" else None
    )
    runtime_trace: list[dict[str, Any]] = []
    runtime_env = env if env is not None else os.environ
    base_model_name = model or runtime_env.get("OPENAI_DEFAULT_MODEL")
    voting_policy = ReviewVotingPolicy(voting_policy)
    resolved_reviewers = _reviewer_model_configs(
        reviewer_models=reviewer_models,
        reviewer_configs=reviewer_configs,
        default_model=base_model_name,
        default_provider_config=provider_config,
    )
    high_priority_models = _dedupe_model_names(high_priority_reviewer_models or [])
    use_json_object_provider = _uses_json_object_provider(provider_config)
    tracing_disabled = disable_tracing or (
        provider_config is not None and provider_config.disable_tracing
    )
    if run_agent is None and provider_config is None and not runtime_env.get("OPENAI_API_KEY"):
        raise OpenAIAgentsDryRunError(
            "OPENAI_API_KEY is required for live OpenAI Agents SDK ingestion"
        )

    try:
        if use_json_object_provider:
            candidate_batch = await _run_stage_with_trace(
                _json_stage_runner(
                    run_chat_completion=run_chat_completion,
                    model=base_model_name,
                    provider_config=provider_config,
                    env=runtime_env,
                    instructions=_EXTRACTOR_INSTRUCTIONS,
                    prompt=_extraction_prompt(ingestion_input),
                    output_type=CandidateRelationBatch,
                    label="extractor",
                    max_tokens=runtime_options.extractor_max_tokens,
                ),
                stage="extractor",
                label=_runtime_label("extractor", None, base_model_name),
                provider=_provider_name(provider_config),
                model=base_model_name,
                reviewer_id=None,
                timeout_seconds=runtime_options.extractor_timeout_seconds,
                runtime_trace=runtime_trace,
                progress_callback=effective_progress_callback,
            )
            agent_model = None
        else:
            agent_model = _agent_model(
                base_model_name,
                provider_config,
                runtime_env,
            )
            extractor = _build_agent(
                name="NeSy relation extractor",
                instructions=_EXTRACTOR_INSTRUCTIONS,
                output_type=CandidateRelationBatch,
                model=agent_model,
            )
            extraction_output = await _run_stage_with_trace(
                lambda: _run_agent_with_optional_runner(
                    run_agent,
                    extractor,
                    _extraction_prompt(ingestion_input),
                    tracing_disabled=tracing_disabled,
                ),
                stage="extractor",
                label=_runtime_label("extractor", None, base_model_name),
                provider=_provider_name(provider_config),
                model=base_model_name,
                reviewer_id=None,
                timeout_seconds=runtime_options.extractor_timeout_seconds,
                runtime_trace=runtime_trace,
                progress_callback=effective_progress_callback,
            )
            candidate_batch = _coerce_candidate_batch(extraction_output)
    except _LLMRuntimeStageError as exc:
        return _runtime_error_report(
            ingestion_input=ingestion_input,
            auto_write=auto_write,
            provider_config=provider_config,
            tracing_disabled=tracing_disabled,
            runtime_trace=runtime_trace,
            diagnostic=_diagnostic_from_stage_error(exc),
        )

    canonicalization_result: PropositionCanonicalizationResult | None = None
    run_propositions = [*ingestion_input.propositions]
    if (auto_write or canonicalize_preview) and candidate_batch.candidates:
        try:
            canonicalization_result = await _run_auto_write_canonicalization(
                ingestion_input=ingestion_input,
                candidate_batch=candidate_batch,
                store=store,
                use_json_object_provider=use_json_object_provider,
                run_chat_completion=run_chat_completion,
                run_agent=run_agent,
                provider_config=provider_config,
                runtime_env=runtime_env,
                base_model_name=base_model_name,
                agent_model=agent_model,
                tracing_disabled=tracing_disabled,
                runtime_options=runtime_options,
                runtime_trace=runtime_trace,
                progress_callback=effective_progress_callback,
            )
        except _LLMRuntimeStageError as exc:
            return _runtime_error_report(
                ingestion_input=ingestion_input,
                auto_write=auto_write,
                provider_config=provider_config,
                tracing_disabled=tracing_disabled,
                runtime_trace=runtime_trace,
                diagnostic=_diagnostic_from_stage_error(exc),
            )
        if any(diagnostic.level == "error" for diagnostic in canonicalization_result.diagnostics):
            return _canonicalization_error_report(
                ingestion_input=ingestion_input,
                candidate_batch=candidate_batch,
                auto_write=auto_write,
                provider_config=provider_config,
                tracing_disabled=tracing_disabled,
                runtime_trace=runtime_trace,
                canonicalization_result=canonicalization_result,
            )
        if auto_write:
            proposition_import_diagnostics = _validate_canonical_proposition_import(
                canonicalization_result.propositions,
                store,
            )
            if proposition_import_diagnostics:
                canonicalization_result = PropositionCanonicalizationResult(
                    candidates=canonicalization_result.candidates,
                    propositions=canonicalization_result.propositions,
                    diagnostics=[
                        *canonicalization_result.diagnostics,
                        *proposition_import_diagnostics,
                    ],
                    metadata={
                        **canonicalization_result.metadata,
                        "diagnostic_count": (
                            canonicalization_result.metadata.get("diagnostic_count", 0)
                            + len(proposition_import_diagnostics)
                        ),
                    },
                )
                return _canonicalization_error_report(
                    ingestion_input=ingestion_input,
                    candidate_batch=candidate_batch,
                    auto_write=auto_write,
                    provider_config=provider_config,
                    tracing_disabled=tracing_disabled,
                    runtime_trace=runtime_trace,
                    canonicalization_result=canonicalization_result,
                )
        candidate_batch = CandidateRelationBatch(candidates=canonicalization_result.candidates)
        run_propositions = [
            *run_propositions,
            *canonicalization_result.propositions,
        ]

    reviewer_results = await _run_reviewer_calls(
        ingestion_input=ingestion_input,
        candidate_batch=candidate_batch,
        reviewers=resolved_reviewers,
        high_priority_models=high_priority_models,
        voting_policy=voting_policy,
        runtime_options=runtime_options,
        runtime_env=runtime_env,
        run_agent=run_agent,
        run_chat_completion=run_chat_completion,
        provider_config=provider_config,
        base_model_name=base_model_name,
        agent_model=agent_model,
        tracing_disabled=tracing_disabled,
        runtime_trace=runtime_trace,
        progress_callback=effective_progress_callback,
    )
    reviews = [
        review
        for result in reviewer_results
        for review in _reviews_with_model(result.reviews, result.config.reviewer_id)
    ]
    runtime_diagnostics = [
        diagnostic for result in reviewer_results for diagnostic in result.diagnostics
    ]
    reviewer_runtime_failed = any(result.failed for result in reviewer_results)

    aggregation = aggregate_review_decisions(
        candidates=candidate_batch.candidates,
        reviews=reviews,
        policy=voting_policy,
        high_priority_reviewer_models=high_priority_models,
        expected_reviewer_models=[
            reviewer_config.reviewer_id
            for reviewer_config in resolved_reviewers
            if reviewer_config.reviewer_id is not None
        ],
    )

    gate_reviews = aggregation.gate_reviews
    if auto_write and reviewer_runtime_failed:
        gate_reviews = _runtime_guard_reviews(candidate_batch.candidates)

    _emit_progress(effective_progress_callback, {"event": "started", "stage": "gate"})
    gate_results, approved_relations, diagnostics, reasoning = await run_dry_run_gate(
        candidates=candidate_batch.candidates,
        reviews=gate_reviews,
        store=store,
        min_write_confidence=min_write_confidence if auto_write else 0.0,
        write_enabled=auto_write,
        propositions=run_propositions,
    )
    queued_count = sum(1 for result in gate_results if result.action.value == "queue")
    _emit_progress(
        effective_progress_callback,
        {
            "event": "done",
            "stage": "gate",
            "status": "ok",
            "queued_count": queued_count,
            "approved_count": len(approved_relations),
        },
    )
    diagnostics = [*runtime_diagnostics, *aggregation.diagnostics, *diagnostics]
    written_relation_ids: list[str] = []
    write_result: dict[str, Any] = {}
    if auto_write and approved_relations:
        written_relation_ids, write_diagnostics, write_result = await write_approved_relations(
            relations=approved_relations,
            store=store,
        )
        diagnostics = [*diagnostics, *write_diagnostics]
        if (
            canonicalization_result is not None
            and canonicalization_result.propositions
            and _structured_write_succeeded(write_result)
        ):
            store.import_records(
                [],
                [],
                propositions=canonicalization_result.propositions,
                mode="append",
                store_id=DEFAULT_STORE_ID,
            )

    report = IngestionReport(
        mode=IngestionMode.WRITE if auto_write else IngestionMode.DRY_RUN,
        candidates=candidate_batch.candidates,
        reviews=aggregation.audit_reviews,
        gate_results=gate_results,
        approved_relations=approved_relations,
        written_relation_ids=written_relation_ids,
        diagnostics=diagnostics,
        metadata={
            "task": ingestion_input.task,
            "question": ingestion_input.question,
            "input_metadata": ingestion_input.metadata,
            "evidence_count": len(ingestion_input.evidence),
            "url_count": len(ingestion_input.urls),
            "reasoning": reasoning,
            "review_aggregation": {
                **aggregation.metadata,
                "aggregate_reviews": [
                    review.model_dump(mode="json", exclude_none=True)
                    for review in aggregation.gate_reviews
                ],
            },
            "write_result": write_result,
            "proposition_canonicalization": (
                canonicalization_result.metadata if canonicalization_result else {}
            ),
            "provider": _provider_metadata(provider_config, tracing_disabled),
            "reviewer_providers": _reviewer_provider_metadata(resolved_reviewers),
            "runtime_trace": runtime_trace,
        },
    )
    if auto_write:
        queued_records = queued_records_from_report(
            report,
            propositions=run_propositions,
            context_metadata=ingestion_input.context_metadata,
        )
        if queued_records:
            stored_records, _updated = store.enqueue_review_queue(queued_records)
            record_ids = [record.id for record in stored_records]
            store.record_audit(
                event_type="review_queue",
                tool_name="auto_ingest.enqueue_review_queue",
                arguments={"run_id": report.run_id, "queue_record_ids": record_ids},
                result_status="ok",
                metadata={"queued_count": len(record_ids)},
            )
            report = report.model_copy(
                update={
                    "metadata": {
                        **report.metadata,
                        "review_queue_record_ids": record_ids,
                    }
                }
            )
    return report


def _json_stage_runner(
    *,
    run_chat_completion: ChatCompletionRunner | None,
    model: str | None,
    provider_config: OpenAICompatibleProviderConfig | None,
    env: Mapping[str, str],
    instructions: str,
    prompt: str,
    output_type: type[OutputBatch],
    label: str,
    max_tokens: int,
) -> Callable[[], Awaitable[OutputBatch]]:
    async def run() -> OutputBatch:
        return await _run_json_object_completion(
            run_chat_completion,
            model=model,
            provider_config=provider_config,
            env=env,
            instructions=instructions,
            prompt=prompt,
            output_type=output_type,
            label=label,
            max_tokens=max_tokens,
        )

    return run


async def _run_auto_write_canonicalization(
    *,
    ingestion_input: IngestionInput,
    candidate_batch: CandidateRelationBatch,
    store: RelationStoreProtocol,
    use_json_object_provider: bool,
    run_chat_completion: ChatCompletionRunner | None,
    run_agent: AgentRunner | None,
    provider_config: OpenAICompatibleProviderConfig | None,
    runtime_env: Mapping[str, str],
    base_model_name: str | None,
    agent_model: Any,
    tracing_disabled: bool,
    runtime_options: LLMRuntimeOptions,
    runtime_trace: list[dict[str, Any]],
    progress_callback: ProgressCallback | None,
) -> PropositionCanonicalizationResult:
    known_propositions = _canonicalization_known_propositions(ingestion_input, store)
    if not known_propositions:
        result = canonicalize_candidate_relations(
            candidates=candidate_batch.candidates,
            known_propositions=[],
        )
        _record_canonicalizer_skip(
            runtime_trace=runtime_trace,
            progress_callback=progress_callback,
            provider_config=provider_config,
            model=base_model_name,
            reason="no_known_propositions",
        )
        return _canonicalization_result_with_llm_status(
            result,
            status="skipped",
            reason="no_known_propositions",
        )
    llm_required, prefilter_reason = _canonicalization_llm_prefilter(
        candidate_batch.candidates,
        known_propositions,
    )
    if not llm_required:
        result = canonicalize_candidate_relations(
            candidates=candidate_batch.candidates,
            known_propositions=known_propositions,
        )
        _record_canonicalizer_skip(
            runtime_trace=runtime_trace,
            progress_callback=progress_callback,
            provider_config=provider_config,
            model=base_model_name,
            reason=prefilter_reason,
        )
        return _canonicalization_result_with_llm_status(
            result,
            status="skipped",
            reason=prefilter_reason,
        )
    prompt = canonicalization_prompt(ingestion_input, candidate_batch, known_propositions)
    if use_json_object_provider:
        canonicalization_batch = await _run_stage_with_trace(
            _json_stage_runner(
                run_chat_completion=run_chat_completion,
                model=base_model_name,
                provider_config=provider_config,
                env=runtime_env,
                instructions=_CANONICALIZER_INSTRUCTIONS,
                prompt=prompt,
                output_type=PropositionCanonicalizationBatch,
                label="canonicalizer",
                max_tokens=runtime_options.reviewer_max_tokens,
            ),
            stage="canonicalizer",
            label=_runtime_label("canonicalizer", None, base_model_name),
            provider=_provider_name(provider_config),
            model=base_model_name,
            reviewer_id=None,
            timeout_seconds=runtime_options.reviewer_timeout_seconds,
            runtime_trace=runtime_trace,
            progress_callback=progress_callback,
        )
    else:
        canonicalizer = _build_agent(
            name="NeSy proposition canonicalizer",
            instructions=_CANONICALIZER_INSTRUCTIONS,
            output_type=PropositionCanonicalizationBatch,
            model=agent_model,
        )
        canonicalization_output = await _run_stage_with_trace(
            lambda: _run_agent_with_optional_runner(
                run_agent,
                canonicalizer,
                prompt,
                tracing_disabled=tracing_disabled,
            ),
            stage="canonicalizer",
            label=_runtime_label("canonicalizer", None, base_model_name),
            provider=_provider_name(provider_config),
            model=base_model_name,
            reviewer_id=None,
            timeout_seconds=runtime_options.reviewer_timeout_seconds,
            runtime_trace=runtime_trace,
            progress_callback=progress_callback,
        )
        canonicalization_batch = _coerce_canonicalization_batch(canonicalization_output)
    result = canonicalize_candidate_relations(
        candidates=candidate_batch.candidates,
        known_propositions=known_propositions,
        canonicalization=canonicalization_batch,
    )
    return _canonicalization_result_with_llm_status(
        result,
        status="executed",
        reason=prefilter_reason,
    )


def _canonicalization_llm_prefilter(
    candidates: list[CandidateRelation],
    known_propositions: list[PropositionRecord],
) -> tuple[bool, str]:
    exact_terms = {
        term
        for proposition in known_propositions
        for term in (proposition.id, proposition.label, *proposition.aliases)
        if term
    }
    known_token_sets = []
    for term in exact_terms:
        tokens = _canonicalization_tokens(term)
        if tokens:
            known_token_sets.append(tokens)
    exact_match_seen = False
    for candidate in candidates:
        for text in (candidate.source, candidate.target):
            if text in exact_terms:
                exact_match_seen = True
                continue
            endpoint_tokens = _canonicalization_tokens(text)
            if any(endpoint_tokens & known_tokens for known_tokens in known_token_sets):
                return True, "likely_overlap"
    return False, "exact_match_only" if exact_match_seen else "no_likely_overlap"


def _canonicalization_tokens(text: str) -> set[str]:
    tokens: set[str] = set()
    for token in _CANONICALIZATION_TOKEN_RE.findall(text.casefold()):
        if len(token) < 3 or token in _CANONICALIZATION_STOPWORDS:
            continue
        tokens.add(token)
        tokens.update(_canonicalization_stems(token))
    return tokens


def _canonicalization_stems(token: str) -> set[str]:
    stems: set[str] = set()
    for suffix in ("ing", "ed", "es", "s"):
        if token.endswith(suffix) and len(token) > len(suffix) + 3:
            stems.add(token[: -len(suffix)])
    return stems


def _record_canonicalizer_skip(
    *,
    runtime_trace: list[dict[str, Any]],
    progress_callback: ProgressCallback | None,
    provider_config: OpenAICompatibleProviderConfig | None,
    model: str | None,
    reason: str,
) -> None:
    error_code = f"CANONICALIZER_PREFILTER_{reason.upper()}"
    label = _runtime_label("canonicalizer", None, model)
    _record_runtime_trace(
        runtime_trace,
        stage="canonicalizer",
        label=label,
        provider=_provider_name(provider_config),
        model=model,
        reviewer_id=None,
        started_at=_utc_now(),
        duration_ms=0,
        status="skipped",
        error_code=error_code,
    )
    _emit_progress(
        progress_callback,
        {
            "event": "skipped",
            "stage": "canonicalizer",
            "label": label,
            "status": "skipped",
            "reason": reason,
            "error_code": error_code,
        },
    )


def _canonicalization_result_with_llm_status(
    result: PropositionCanonicalizationResult,
    *,
    status: str,
    reason: str,
) -> PropositionCanonicalizationResult:
    return PropositionCanonicalizationResult(
        candidates=result.candidates,
        propositions=result.propositions,
        diagnostics=result.diagnostics,
        metadata={
            **result.metadata,
            "llm_canonicalizer": {
                "status": status,
                "reason": reason,
            },
        },
    )


async def _run_reviewer_calls(
    *,
    ingestion_input: IngestionInput,
    candidate_batch: CandidateRelationBatch,
    reviewers: list[ReviewerModelConfig],
    high_priority_models: list[str],
    voting_policy: ReviewVotingPolicy,
    runtime_options: LLMRuntimeOptions,
    runtime_env: Mapping[str, str],
    run_agent: AgentRunner | None,
    run_chat_completion: ChatCompletionRunner | None,
    provider_config: OpenAICompatibleProviderConfig | None,
    base_model_name: str | None,
    agent_model: Any,
    tracing_disabled: bool,
    runtime_trace: list[dict[str, Any]],
    progress_callback: ProgressCallback | None,
) -> list[_ReviewerCallResult]:
    high_priority_reviewers = [
        reviewer for reviewer in reviewers if reviewer.reviewer_id in high_priority_models
    ]
    normal_reviewers = [
        reviewer for reviewer in reviewers if reviewer.reviewer_id not in high_priority_models
    ]
    if voting_policy == ReviewVotingPolicy.RISK_TIERED and high_priority_reviewers:
        high_priority_results = await _run_reviewers_parallel(
            ingestion_input=ingestion_input,
            candidate_batch=candidate_batch,
            reviewers=high_priority_reviewers,
            timeout_seconds=runtime_options.high_priority_reviewer_timeout_seconds,
            runtime_options=runtime_options,
            runtime_env=runtime_env,
            run_agent=run_agent,
            run_chat_completion=run_chat_completion,
            provider_config=provider_config,
            base_model_name=base_model_name,
            agent_model=agent_model,
            tracing_disabled=tracing_disabled,
            runtime_trace=runtime_trace,
            progress_callback=progress_callback,
        )
        if _high_priority_should_short_circuit(
            candidate_batch=candidate_batch,
            high_priority_results=high_priority_results,
            high_priority_models=high_priority_models,
        ):
            skipped_results = [
                _skipped_reviewer_result(reviewer, candidate_batch) for reviewer in normal_reviewers
            ]
            for result in skipped_results:
                _record_runtime_trace(
                    runtime_trace,
                    stage="reviewer",
                    label=_reviewer_agent_name(result.config.reviewer_id),
                    provider=_provider_name(result.config.provider_config),
                    model=result.config.model,
                    reviewer_id=result.config.reviewer_id,
                    started_at=_utc_now(),
                    duration_ms=0,
                    status="skipped",
                    error_code="HIGH_PRIORITY_REVIEWER_SHORT_CIRCUIT",
                    review_count=0,
                )
                _emit_progress(
                    progress_callback,
                    {
                        "event": "skipped",
                        "stage": "reviewer",
                        "label": _reviewer_agent_name(result.config.reviewer_id),
                        "reviewer_id": result.config.reviewer_id,
                        "status": "skipped",
                        "error_code": "HIGH_PRIORITY_REVIEWER_SHORT_CIRCUIT",
                    },
                )
            return [*high_priority_results, *skipped_results]
        normal_results = await _run_reviewers_parallel(
            ingestion_input=ingestion_input,
            candidate_batch=candidate_batch,
            reviewers=normal_reviewers,
            timeout_seconds=runtime_options.reviewer_timeout_seconds,
            runtime_options=runtime_options,
            runtime_env=runtime_env,
            run_agent=run_agent,
            run_chat_completion=run_chat_completion,
            provider_config=provider_config,
            base_model_name=base_model_name,
            agent_model=agent_model,
            tracing_disabled=tracing_disabled,
            runtime_trace=runtime_trace,
            progress_callback=progress_callback,
        )
        return [*high_priority_results, *normal_results]

    return await _run_reviewers_parallel(
        ingestion_input=ingestion_input,
        candidate_batch=candidate_batch,
        reviewers=reviewers,
        timeout_seconds=runtime_options.reviewer_timeout_seconds,
        runtime_options=runtime_options,
        runtime_env=runtime_env,
        run_agent=run_agent,
        run_chat_completion=run_chat_completion,
        provider_config=provider_config,
        base_model_name=base_model_name,
        agent_model=agent_model,
        tracing_disabled=tracing_disabled,
        runtime_trace=runtime_trace,
        progress_callback=progress_callback,
    )


async def _run_reviewers_parallel(
    *,
    ingestion_input: IngestionInput,
    candidate_batch: CandidateRelationBatch,
    reviewers: list[ReviewerModelConfig],
    timeout_seconds: float,
    runtime_options: LLMRuntimeOptions,
    runtime_env: Mapping[str, str],
    run_agent: AgentRunner | None,
    run_chat_completion: ChatCompletionRunner | None,
    provider_config: OpenAICompatibleProviderConfig | None,
    base_model_name: str | None,
    agent_model: Any,
    tracing_disabled: bool,
    runtime_trace: list[dict[str, Any]],
    progress_callback: ProgressCallback | None,
) -> list[_ReviewerCallResult]:
    results: list[_ReviewerCallResult | None] = [None] * len(reviewers)

    async def run_one(index: int, reviewer_config: ReviewerModelConfig) -> None:
        results[index] = await _run_one_reviewer(
            ingestion_input=ingestion_input,
            candidate_batch=candidate_batch,
            reviewer_config=reviewer_config,
            timeout_seconds=timeout_seconds,
            runtime_options=runtime_options,
            runtime_env=runtime_env,
            run_agent=run_agent,
            run_chat_completion=run_chat_completion,
            provider_config=provider_config,
            base_model_name=base_model_name,
            agent_model=agent_model,
            tracing_disabled=tracing_disabled,
            runtime_trace=runtime_trace,
            progress_callback=progress_callback,
        )

    async with anyio.create_task_group() as task_group:
        for index, reviewer_config in enumerate(reviewers):
            task_group.start_soon(run_one, index, reviewer_config)
    return [result for result in results if result is not None]


async def _run_one_reviewer(
    *,
    ingestion_input: IngestionInput,
    candidate_batch: CandidateRelationBatch,
    reviewer_config: ReviewerModelConfig,
    timeout_seconds: float,
    runtime_options: LLMRuntimeOptions,
    runtime_env: Mapping[str, str],
    run_agent: AgentRunner | None,
    run_chat_completion: ChatCompletionRunner | None,
    provider_config: OpenAICompatibleProviderConfig | None,
    base_model_name: str | None,
    agent_model: Any,
    tracing_disabled: bool,
    runtime_trace: list[dict[str, Any]],
    progress_callback: ProgressCallback | None,
) -> _ReviewerCallResult:
    reviewer_provider_config = reviewer_config.provider_config
    label = _reviewer_agent_name(reviewer_config.reviewer_id)
    try:
        if _uses_json_object_provider(reviewer_provider_config):
            review_batch = await _run_stage_with_trace(
                _json_stage_runner(
                    run_chat_completion=run_chat_completion,
                    model=reviewer_config.model,
                    provider_config=reviewer_provider_config,
                    env=runtime_env,
                    instructions=_REVIEWER_INSTRUCTIONS,
                    prompt=_review_prompt(ingestion_input, candidate_batch),
                    output_type=ReviewDecisionBatch,
                    label=label,
                    max_tokens=runtime_options.reviewer_max_tokens,
                ),
                stage="reviewer",
                label=label,
                provider=_provider_name(reviewer_provider_config),
                model=reviewer_config.model,
                reviewer_id=reviewer_config.reviewer_id,
                timeout_seconds=timeout_seconds,
                runtime_trace=runtime_trace,
                progress_callback=progress_callback,
            )
        else:
            reviewer_agent_model = (
                agent_model
                if (
                    reviewer_provider_config == provider_config
                    and reviewer_config.model == base_model_name
                )
                else _agent_model(reviewer_config.model, reviewer_provider_config, runtime_env)
            )
            reviewer = _build_agent(
                name=label,
                instructions=_REVIEWER_INSTRUCTIONS,
                output_type=ReviewDecisionBatch,
                model=reviewer_agent_model,
            )
            review_output = await _run_stage_with_trace(
                lambda: _run_agent_with_optional_runner(
                    run_agent,
                    reviewer,
                    _review_prompt(ingestion_input, candidate_batch),
                    tracing_disabled=tracing_disabled,
                ),
                stage="reviewer",
                label=label,
                provider=_provider_name(reviewer_provider_config),
                model=reviewer_config.model,
                reviewer_id=reviewer_config.reviewer_id,
                timeout_seconds=timeout_seconds,
                runtime_trace=runtime_trace,
                progress_callback=progress_callback,
            )
            review_batch = _coerce_review_batch(review_output)
    except _LLMRuntimeStageError as exc:
        return _failed_reviewer_result(reviewer_config, candidate_batch, exc)
    return _ReviewerCallResult(
        config=reviewer_config,
        reviews=review_batch.reviews,
        diagnostics=[],
    )


async def _run_stage_with_trace(
    operation: Callable[[], Awaitable[Any]],
    *,
    stage: str,
    label: str,
    provider: str,
    model: str | None,
    reviewer_id: str | None,
    timeout_seconds: float,
    runtime_trace: list[dict[str, Any]],
    progress_callback: ProgressCallback | None,
) -> Any:
    started_at = _utc_now()
    started_perf = perf_counter()
    cancelled_exc_class = anyio.get_cancelled_exc_class()
    _emit_progress(
        progress_callback,
        {
            "event": "started",
            "stage": stage,
            "label": label,
            "provider": provider,
            "model": model,
            "reviewer_id": reviewer_id,
        },
    )
    try:
        with anyio.fail_after(timeout_seconds):
            result = await operation()
    except TimeoutError as exc:
        duration_ms = _elapsed_ms(started_perf)
        _record_runtime_trace(
            runtime_trace,
            stage=stage,
            label=label,
            provider=provider,
            model=model,
            reviewer_id=reviewer_id,
            started_at=started_at,
            duration_ms=duration_ms,
            status="timeout",
            error_code="LLM_RUNTIME_TIMEOUT",
        )
        _emit_progress(
            progress_callback,
            {
                "event": "timeout",
                "stage": stage,
                "label": label,
                "provider": provider,
                "model": model,
                "reviewer_id": reviewer_id,
                "status": "timeout",
                "duration_ms": duration_ms,
                "timeout_seconds": timeout_seconds,
            },
        )
        raise _LLMRuntimeStageError(
            f"{label} timed out after {timeout_seconds:g}s",
            code="LLM_RUNTIME_TIMEOUT",
            status="timeout",
            stage=stage,
            label=label,
        ) from exc
    except cancelled_exc_class:
        raise
    except Exception as exc:
        duration_ms = _elapsed_ms(started_perf)
        _record_runtime_trace(
            runtime_trace,
            stage=stage,
            label=label,
            provider=provider,
            model=model,
            reviewer_id=reviewer_id,
            started_at=started_at,
            duration_ms=duration_ms,
            status="error",
            error_code=exc.__class__.__name__,
        )
        _emit_progress(
            progress_callback,
            {
                "event": "error",
                "stage": stage,
                "label": label,
                "provider": provider,
                "model": model,
                "reviewer_id": reviewer_id,
                "status": "error",
                "duration_ms": duration_ms,
                "error_code": exc.__class__.__name__,
            },
        )
        raise _LLMRuntimeStageError(
            f"{label} failed before producing structured output",
            code=exc.__class__.__name__,
            status="error",
            stage=stage,
            label=label,
        ) from exc
    duration_ms = _elapsed_ms(started_perf)
    counts = _output_counts(result)
    candidate_count = counts.get("candidate_count")
    review_count = counts.get("review_count")
    _record_runtime_trace(
        runtime_trace,
        stage=stage,
        label=label,
        provider=provider,
        model=model,
        reviewer_id=reviewer_id,
        started_at=started_at,
        duration_ms=duration_ms,
        status="ok",
        candidate_count=candidate_count,
        review_count=review_count,
    )
    _emit_progress(
        progress_callback,
        {
            "event": "done",
            "stage": stage,
            "label": label,
            "provider": provider,
            "model": model,
            "reviewer_id": reviewer_id,
            "status": "ok",
            "duration_ms": duration_ms,
            **counts,
        },
    )
    return result


def _high_priority_should_short_circuit(
    *,
    candidate_batch: CandidateRelationBatch,
    high_priority_results: list[_ReviewerCallResult],
    high_priority_models: list[str],
) -> bool:
    if any(result.failed for result in high_priority_results):
        return True
    reviews = [
        review
        for result in high_priority_results
        for review in _reviews_with_model(result.reviews, result.config.reviewer_id)
    ]
    reviews_by_candidate_model = {
        (review.candidate_id, review.reviewer_model): review for review in reviews
    }
    for candidate in candidate_batch.candidates:
        for reviewer_model in high_priority_models:
            review = reviews_by_candidate_model.get((candidate.id, reviewer_model))
            if review is None:
                return True
            if review.decision is not ReviewDecisionValue.APPROVE:
                return True
            if review.normalized_implication_supported is not True:
                return True
    return False


def _failed_reviewer_result(
    reviewer_config: ReviewerModelConfig,
    candidate_batch: CandidateRelationBatch,
    error: _LLMRuntimeStageError,
) -> _ReviewerCallResult:
    return _ReviewerCallResult(
        config=reviewer_config,
        reviews=[
            ReviewDecision(
                candidate_id=candidate.id,
                decision=ReviewDecisionValue.NEEDS_HUMAN,
                reasons=[f"{reviewer_config.reviewer_id or 'reviewer'} runtime {error.status}"],
                reviewer_model=reviewer_config.reviewer_id,
                metadata={
                    "synthetic_vote": f"reviewer_{error.status}",
                    "error_code": error.code,
                },
            )
            for candidate in candidate_batch.candidates
        ],
        diagnostics=[_diagnostic_from_stage_error(error)],
        failed=True,
    )


def _skipped_reviewer_result(
    reviewer_config: ReviewerModelConfig,
    candidate_batch: CandidateRelationBatch,
) -> _ReviewerCallResult:
    return _ReviewerCallResult(
        config=reviewer_config,
        reviews=[
            ReviewDecision(
                candidate_id=candidate.id,
                decision=ReviewDecisionValue.NEEDS_HUMAN,
                reasons=["high-priority reviewer short-circuited lower-priority review"],
                reviewer_model=reviewer_config.reviewer_id,
                metadata={"synthetic_vote": "reviewer_skipped"},
            )
            for candidate in candidate_batch.candidates
        ],
        diagnostics=[],
        failed=False,
    )


def _diagnostic_from_stage_error(error: _LLMRuntimeStageError) -> Diagnostic:
    code = "LLM_RUNTIME_TIMEOUT" if error.status == "timeout" else "LLM_RUNTIME_ERROR"
    return Diagnostic(
        level="error",
        code=code,
        message=f"{error.stage} {error.status}: {error.label}",
    )


def _runtime_error_report(
    *,
    ingestion_input: IngestionInput,
    auto_write: bool,
    provider_config: OpenAICompatibleProviderConfig | None,
    tracing_disabled: bool,
    runtime_trace: list[dict[str, Any]],
    diagnostic: Diagnostic,
) -> IngestionReport:
    return IngestionReport(
        mode=IngestionMode.WRITE if auto_write else IngestionMode.DRY_RUN,
        diagnostics=[diagnostic],
        metadata={
            "task": ingestion_input.task,
            "question": ingestion_input.question,
            "input_metadata": ingestion_input.metadata,
            "evidence_count": len(ingestion_input.evidence),
            "url_count": len(ingestion_input.urls),
            "provider": _provider_metadata(provider_config, tracing_disabled),
            "runtime_trace": runtime_trace,
        },
    )


def _canonicalization_error_report(
    *,
    ingestion_input: IngestionInput,
    candidate_batch: CandidateRelationBatch,
    auto_write: bool,
    provider_config: OpenAICompatibleProviderConfig | None,
    tracing_disabled: bool,
    runtime_trace: list[dict[str, Any]],
    canonicalization_result: PropositionCanonicalizationResult,
) -> IngestionReport:
    return IngestionReport(
        mode=IngestionMode.WRITE if auto_write else IngestionMode.DRY_RUN,
        candidates=candidate_batch.candidates,
        diagnostics=canonicalization_result.diagnostics,
        metadata={
            "task": ingestion_input.task,
            "question": ingestion_input.question,
            "input_metadata": ingestion_input.metadata,
            "evidence_count": len(ingestion_input.evidence),
            "url_count": len(ingestion_input.urls),
            "provider": _provider_metadata(provider_config, tracing_disabled),
            "runtime_trace": runtime_trace,
            "proposition_canonicalization": canonicalization_result.metadata,
        },
    )


def _runtime_guard_reviews(candidates: list[Any]) -> list[ReviewDecision]:
    return [
        ReviewDecision(
            candidate_id=candidate.id,
            decision=ReviewDecisionValue.NEEDS_HUMAN,
            reasons=["reviewer runtime failure blocked auto-write"],
            reviewer_model="runtime:guard",
            metadata={"runtime_guard": "reviewer_runtime_failure"},
        )
        for candidate in candidates
    ]


def _record_runtime_trace(
    runtime_trace: list[dict[str, Any]],
    *,
    stage: str,
    label: str,
    provider: str,
    model: str | None,
    reviewer_id: str | None,
    started_at: str,
    duration_ms: int,
    status: str,
    error_code: str | None = None,
    candidate_count: int | None = None,
    review_count: int | None = None,
) -> None:
    finished_at = _utc_now()
    item: dict[str, Any] = {
        "stage": stage,
        "label": label,
        "provider": provider,
        "model": model,
        "reviewer_id": reviewer_id,
        "started_at": started_at,
        "finished_at": finished_at,
        "duration_ms": duration_ms,
        "status": status,
    }
    if error_code:
        item["error_code"] = error_code
    if candidate_count is not None:
        item["candidate_count"] = candidate_count
    if review_count is not None:
        item["review_count"] = review_count
    runtime_trace.append(item)


def _emit_progress(
    progress_callback: ProgressCallback | None,
    event: dict[str, Any],
) -> None:
    if progress_callback is not None:
        progress_callback({key: value for key, value in event.items() if value is not None})


def _output_counts(output: Any) -> dict[str, int]:
    if isinstance(output, CandidateRelationBatch):
        return {"candidate_count": len(output.candidates)}
    if isinstance(output, ReviewDecisionBatch):
        return {"review_count": len(output.reviews)}
    if isinstance(output, PropositionCanonicalizationBatch):
        return {"proposition_count": len(output.propositions)}
    return {}


def _elapsed_ms(started_perf: float) -> int:
    return int((perf_counter() - started_perf) * 1000)


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _provider_name(provider_config: OpenAICompatibleProviderConfig | None) -> str:
    if provider_config is None:
        return "openai"
    if "deepseek" in provider_config.base_url:
        return "deepseek"
    if "moonshot" in provider_config.base_url:
        return "kimi"
    if "openrouter" in provider_config.base_url:
        return "openrouter"
    return "openai_compatible"


def _runtime_label(stage: str, reviewer_id: str | None, model: str | None) -> str:
    if reviewer_id:
        return _reviewer_agent_name(reviewer_id)
    if model:
        return f"{stage} ({model})"
    return stage


def _build_agent(
    *,
    name: str,
    instructions: str,
    output_type: type[Any],
    model: Any,
) -> Any:
    from agents import Agent, AgentOutputSchema

    kwargs: dict[str, Any] = {
        "name": name,
        "instructions": instructions,
        "output_type": AgentOutputSchema(output_type, strict_json_schema=False),
    }
    if model:
        kwargs["model"] = model
    return Agent(**kwargs)


def _uses_json_object_provider(
    provider_config: OpenAICompatibleProviderConfig | None,
) -> bool:
    return (
        provider_config is not None
        and provider_config.structured_output_mode is ProviderStructuredOutputMode.JSON_OBJECT
    )


async def _run_json_object_completion(
    run_chat_completion: ChatCompletionRunner | None,
    *,
    model: str | None,
    provider_config: OpenAICompatibleProviderConfig | None,
    env: Mapping[str, str],
    instructions: str,
    prompt: str,
    output_type: type[OutputBatch],
    label: str,
    max_tokens: int,
) -> OutputBatch:
    if provider_config is None:
        raise OpenAIAgentsDryRunError("provider_config is required for JSON Object mode")
    if not model:
        raise OpenAIAgentsDryRunError(
            "--model or OPENAI_DEFAULT_MODEL is required for provider JSON Object mode"
        )
    api_key = env.get(provider_config.api_key_env)
    if not api_key:
        raise OpenAIAgentsDryRunError(
            f"{provider_config.api_key_env} is required for OpenAI-compatible provider"
        )
    request_kwargs = _json_object_request_kwargs(
        model=model,
        provider_config=provider_config,
        instructions=instructions,
        prompt=prompt,
        output_type=output_type,
        max_tokens=max_tokens,
    )
    if run_chat_completion is not None:
        response = await run_chat_completion(
            provider_config=provider_config,
            **request_kwargs,
        )
    else:
        response = await _run_openai_compatible_json_object_completion(
            api_key=api_key,
            provider_config=provider_config,
            request_kwargs=request_kwargs,
        )
    if isinstance(response, output_type):
        return response
    content = _json_object_response_content(response)
    if not content or not content.strip():
        raise OpenAIAgentsDryRunError(f"{label} returned empty JSON Object content")
    try:
        return output_type.model_validate_json(content)
    except ValidationError as exc:
        raise OpenAIAgentsDryRunError(
            f"{label} returned JSON that does not match {output_type.__name__}"
        ) from exc


def _json_object_request_kwargs(
    *,
    model: str,
    provider_config: OpenAICompatibleProviderConfig,
    instructions: str,
    prompt: str,
    output_type: type[BaseModel],
    max_tokens: int,
) -> dict[str, Any]:
    request_kwargs: dict[str, Any] = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": _json_object_system_prompt(instructions, output_type),
            },
            {"role": "user", "content": prompt},
        ],
        "response_format": {"type": "json_object"},
        "stream": False,
        "max_tokens": max_tokens,
    }
    if provider_config.reasoning_effort:
        request_kwargs["reasoning_effort"] = provider_config.reasoning_effort
    if provider_config.extra_body:
        request_kwargs["extra_body"] = dict(provider_config.extra_body)
    return request_kwargs


def _json_object_system_prompt(
    instructions: str,
    output_type: type[BaseModel],
) -> str:
    schema_json = json.dumps(output_type.model_json_schema(), ensure_ascii=False)
    example_json = json.dumps(_json_object_example(output_type), ensure_ascii=False)
    return (
        f"{instructions}\n"
        "Return only one valid JSON object. Do not include markdown fences, prose, "
        "or comments.\n"
        "The JSON object must satisfy this JSON schema:\n"
        f"{schema_json}\n"
        "Example JSON output:\n"
        f"{example_json}"
    )


def _json_object_example(output_type: type[BaseModel]) -> dict[str, object]:
    if output_type is CandidateRelationBatch:
        return {"candidates": []}
    if output_type is PropositionCanonicalizationBatch:
        return {"propositions": []}
    if output_type is ReviewDecisionBatch:
        return {"reviews": []}
    return {}


async def _run_openai_compatible_json_object_completion(
    *,
    api_key: str,
    provider_config: OpenAICompatibleProviderConfig,
    request_kwargs: Mapping[str, Any],
) -> Any:
    from openai import AsyncOpenAI

    client = AsyncOpenAI(
        api_key=api_key,
        base_url=provider_config.base_url,
        default_headers=provider_config.default_headers or None,
    )
    return await client.chat.completions.create(**dict(request_kwargs))


def _json_object_response_content(response: Any) -> str | None:
    if isinstance(response, str):
        return response
    if isinstance(response, Mapping):
        choices = response.get("choices")
        if choices is None:
            raise OpenAIAgentsDryRunError("provider JSON Object response did not include choices")
        return _json_object_content_from_choices(choices)
    return _json_object_content_from_choices(getattr(response, "choices", None))


def _json_object_content_from_choices(choices: Any) -> str | None:
    if not choices:
        return None
    choice = choices[0]
    message = (
        choice.get("message") if isinstance(choice, Mapping) else getattr(choice, "message", None)
    )
    if message is None:
        return None
    content = (
        message.get("content")
        if isinstance(message, Mapping)
        else getattr(message, "content", None)
    )
    return content if isinstance(content, str) else None


def _agent_model(
    model: str | None,
    provider_config: OpenAICompatibleProviderConfig | None,
    env: Mapping[str, str],
) -> Any:
    if provider_config is None:
        return model
    if not model:
        raise OpenAIAgentsDryRunError(
            "--model or OPENAI_DEFAULT_MODEL is required when --base-url is set"
        )
    api_key = env.get(provider_config.api_key_env)
    if not api_key:
        raise OpenAIAgentsDryRunError(
            f"{provider_config.api_key_env} is required for OpenAI-compatible provider"
        )
    return _openai_compatible_model(model, api_key, provider_config)


def _openai_compatible_model(
    model: str,
    api_key: str,
    provider_config: OpenAICompatibleProviderConfig,
) -> Any:
    from agents.models.openai_chatcompletions import OpenAIChatCompletionsModel
    from openai import AsyncOpenAI

    client = AsyncOpenAI(
        api_key=api_key,
        base_url=provider_config.base_url,
        default_headers=provider_config.default_headers or None,
    )
    return OpenAIChatCompletionsModel(model=model, openai_client=client)


async def _run_agent_with_optional_runner(
    run_agent: AgentRunner | None,
    agent: Any,
    prompt: str,
    *,
    tracing_disabled: bool,
) -> Any:
    if run_agent is not None:
        if _runner_accepts_tracing_disabled(run_agent):
            return await run_agent(agent, prompt, tracing_disabled=tracing_disabled)
        return await run_agent(agent, prompt)
    return await _run_agent(agent, prompt, tracing_disabled=tracing_disabled)


def _runner_accepts_tracing_disabled(run_agent: AgentRunner) -> bool:
    try:
        parameters = signature(run_agent).parameters
    except (TypeError, ValueError):
        return False
    return any(
        parameter.kind is Parameter.VAR_KEYWORD or name == "tracing_disabled"
        for name, parameter in parameters.items()
    )


def _provider_metadata(
    provider_config: OpenAICompatibleProviderConfig | None,
    disable_tracing: bool,
) -> dict[str, Any]:
    if provider_config is None:
        return {"type": "openai", "tracing_disabled": disable_tracing}
    thinking = provider_config.extra_body.get("thinking")
    thinking_type = (
        thinking.get("type")
        if isinstance(thinking, Mapping) and isinstance(thinking.get("type"), str)
        else None
    )
    metadata: dict[str, Any] = {
        "type": "openai_compatible",
        "header_keys": sorted(provider_config.default_headers),
        "tracing_disabled": disable_tracing or provider_config.disable_tracing,
        "structured_output_mode": provider_config.structured_output_mode.value,
    }
    if provider_config.reasoning_effort:
        metadata["reasoning_effort"] = provider_config.reasoning_effort
    reasoning = _safe_reasoning_metadata(provider_config.extra_body.get("reasoning"))
    if reasoning:
        metadata["reasoning"] = reasoning
    if thinking_type:
        metadata["thinking"] = {"type": thinking_type}
    return metadata


def _reviewer_provider_metadata(
    reviewer_configs: list[ReviewerModelConfig],
) -> list[dict[str, Any]]:
    metadata: list[dict[str, Any]] = []
    for reviewer_config in reviewer_configs:
        item: dict[str, Any] = {
            "reviewer_id": reviewer_config.reviewer_id,
            "model": reviewer_config.model,
            "provider": reviewer_config.provider_name or "run",
        }
        if reviewer_config.provider_config is None:
            item["type"] = "openai"
        else:
            item.update(
                {
                    "type": "openai_compatible",
                    "structured_output_mode": (
                        reviewer_config.provider_config.structured_output_mode.value
                    ),
                    "header_keys": sorted(reviewer_config.provider_config.default_headers),
                }
            )
            reasoning = _safe_reasoning_metadata(
                reviewer_config.provider_config.extra_body.get("reasoning")
            )
            if reasoning:
                item["reasoning"] = reasoning
        metadata.append(item)
    return metadata


def _safe_reasoning_metadata(value: object) -> dict[str, object]:
    if not isinstance(value, Mapping):
        return {}
    metadata: dict[str, object] = {}
    for key in ("effort", "exclude", "enabled", "max_tokens"):
        item = value.get(key)
        if isinstance(item, (str, bool, int)):
            metadata[key] = item
    return metadata


def _reviewer_model_configs(
    *,
    reviewer_models: list[str] | None,
    reviewer_configs: list[ReviewerModelConfig] | None,
    default_model: str | None,
    default_provider_config: OpenAICompatibleProviderConfig | None,
) -> list[ReviewerModelConfig]:
    resolved: list[ReviewerModelConfig] = []
    seen: set[str] = set()
    for model in _dedupe_model_names(reviewer_models or []):
        if model in seen:
            continue
        seen.add(model)
        resolved.append(
            ReviewerModelConfig(
                reviewer_id=model,
                model=model,
                provider_name=None,
                provider_config=default_provider_config,
            )
        )
    for reviewer_config in reviewer_configs or []:
        if reviewer_config.reviewer_id is not None and reviewer_config.reviewer_id in seen:
            continue
        if reviewer_config.reviewer_id is not None:
            seen.add(reviewer_config.reviewer_id)
        resolved.append(reviewer_config)
    if resolved:
        return resolved
    return [
        ReviewerModelConfig(
            reviewer_id=default_model,
            model=default_model,
            provider_name=None,
            provider_config=default_provider_config,
        )
    ]


def _dedupe_model_names(values: list[str]) -> list[str]:
    return dedupe_non_empty_text(values)


def _reviewer_agent_name(reviewer_model_name: str | None) -> str:
    if reviewer_model_name is None:
        return "NeSy relation reviewer"
    return f"NeSy relation reviewer ({reviewer_model_name})"


def _reviews_with_model(
    reviews: list[ReviewDecision],
    reviewer_model_name: str | None,
) -> list[ReviewDecision]:
    if reviewer_model_name is None:
        return reviews
    normalized: list[ReviewDecision] = []
    for review in reviews:
        metadata = dict(review.metadata)
        if review.reviewer_model and review.reviewer_model != reviewer_model_name:
            metadata["reported_reviewer_model"] = review.reviewer_model
        normalized.append(
            review.model_copy(
                update={
                    "reviewer_model": reviewer_model_name,
                    "metadata": metadata,
                }
            )
        )
    return normalized


async def _run_agent(agent: Any, prompt: str, *, tracing_disabled: bool = False) -> Any:
    from agents import RunConfig, Runner

    run_config = RunConfig(tracing_disabled=True) if tracing_disabled else None
    result = await Runner.run(agent, prompt, run_config=run_config)
    return result.final_output


def _coerce_candidate_batch(output: Any) -> CandidateRelationBatch:
    if isinstance(output, CandidateRelationBatch):
        return output
    if isinstance(output, list):
        return CandidateRelationBatch(candidates=output)
    return CandidateRelationBatch.model_validate(output)


def _coerce_review_batch(output: Any) -> ReviewDecisionBatch:
    if isinstance(output, ReviewDecisionBatch):
        return output
    if isinstance(output, list):
        return ReviewDecisionBatch(reviews=output)
    return ReviewDecisionBatch.model_validate(output)


def _coerce_canonicalization_batch(output: Any) -> PropositionCanonicalizationBatch:
    if isinstance(output, PropositionCanonicalizationBatch):
        return output
    if isinstance(output, list):
        return PropositionCanonicalizationBatch(propositions=output)
    return PropositionCanonicalizationBatch.model_validate(output)


def _canonicalization_known_propositions(
    ingestion_input: IngestionInput,
    store: RelationStoreProtocol,
) -> list[PropositionRecord]:
    merged: dict[str, PropositionRecord] = {}
    for proposition in [*store.list_propositions(), *ingestion_input.propositions]:
        current = merged.get(proposition.id)
        if current is None:
            merged[proposition.id] = proposition
            continue
        aliases = dedupe_non_empty_text([*current.aliases, proposition.label, *proposition.aliases])
        aliases = [alias for alias in aliases if alias not in {current.id, current.label}]
        merged[proposition.id] = current.model_copy(
            update={
                "aliases": aliases,
                "metadata": {**current.metadata, **proposition.metadata},
            }
        )
    return list(merged.values())


def _structured_write_succeeded(write_result: dict[str, Any]) -> bool:
    return write_result.get("status") in {"ok", "warning"}


def _validate_canonical_proposition_import(
    propositions: list[PropositionRecord],
    store: RelationStoreProtocol,
) -> list[Diagnostic]:
    if not propositions:
        return []
    try:
        store.import_records(
            [],
            [],
            propositions=propositions,
            mode="append",
            store_id=DEFAULT_STORE_ID,
            dry_run=True,
        )
    except ValueError as exc:
        return [
            Diagnostic(
                level="error",
                code="PROPOSITION_CANONICALIZATION_IMPORT_INVALID",
                message=str(exc),
            )
        ]
    return []


def _extraction_prompt(ingestion_input: IngestionInput) -> str:
    return (
        "Extract only evidence-supported logical relations.\n"
        f"{_RELATION_DIRECTION_RULES}\n"
        "Return no candidate when the evidence only shows topical similarity, "
        "correlation, weak possibility, or unsupported speculation.\n\n"
        f"Input JSON:\n{_input_json(ingestion_input)}"
    )


def _review_prompt(
    ingestion_input: IngestionInput,
    candidate_batch: CandidateRelationBatch,
) -> str:
    payload = {
        "input": ingestion_input.model_dump(mode="json", exclude_none=True),
        "candidates": _candidate_review_payload(candidate_batch),
    }
    return (
        "Review each candidate relation against the evidence.\n"
        f"{_RELATION_DIRECTION_RULES}\n"
        "Use approve only when source text directly supports the final relation. "
        "Set normalized_implication_supported=true only when evidence directly "
        "supports every normalized implication edge for the final relation type. "
        "Use reject for unsupported claims and needs_human for ambiguous claims.\n\n"
        f"Review payload JSON:\n{json.dumps(payload, ensure_ascii=False)}"
    )


def _candidate_review_payload(candidate_batch: CandidateRelationBatch) -> list[dict[str, Any]]:
    return [
        {
            **candidate.model_dump(mode="json", exclude_none=True),
            "normalized_implications": {
                "candidate_relation_type": candidate.relation_type.value,
                "edges": normalized_implication_preview(
                    candidate.source,
                    candidate.target,
                    candidate.relation_type,
                ),
            },
        }
        for candidate in candidate_batch.candidates
    ]


def _input_json(ingestion_input: IngestionInput) -> str:
    return json.dumps(
        ingestion_input.model_dump(mode="json", exclude_none=True), ensure_ascii=False
    )


_EXTRACTOR_INSTRUCTIONS = """\
You extract candidate symbolic relations for NeSy Reasoning MCP.
Only emit sufficient, necessary, or equivalent relations directly supported by evidence.
Relation direction rules are strict: sufficient(A, B)=A -> B; necessary(A, B)=B -> A.
Equivalent(A, B)=A -> B and B -> A.
Each candidate must cite at least one provided EvidenceRecord.
Do not turn "may improve", correlation, topical similarity, or vague support into a relation.
"""

_REVIEWER_INSTRUCTIONS = """\
You review candidate symbolic relations for NeSy Reasoning MCP.
For approve or downgrade, provide final_relation_type and final_confidence.
For approve or downgrade, set normalized_implication_supported=true only when evidence directly
supports every normalized implication edge for the final relation type.
Prefer reject or needs_human when evidence is weak, ambiguous, or missing.
Do not approve claims that would require external knowledge not present in the evidence.
"""

_CANONICALIZER_INSTRUCTIONS = """\
You canonicalize candidate proposition labels into stable graph nodes.
Return one proposition group per real proposition and cover every endpoint_ref exactly once.
Reuse known proposition ids only when the endpoint means the same proposition.
Do not merge eligibility, possibility, permission, readiness, or capability with the actual event.
"""

_RELATION_DIRECTION_RULES = (
    "Relation direction rules: sufficient(A, B)=A -> B; "
    "necessary(A, B)=B -> A; equivalent(A, B)=A -> B and B -> A."
)
