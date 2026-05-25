"""OpenAI Agents SDK orchestration for dry-run candidate ingestion."""

from __future__ import annotations

import json
import os
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from inspect import Parameter, signature
from types import MappingProxyType
from typing import Any

from nesy_reasoning_mcp.auto_ingest.gate import run_dry_run_gate
from nesy_reasoning_mcp.auto_ingest.review_queue import queued_records_from_report
from nesy_reasoning_mcp.auto_ingest.review_voting import aggregate_review_decisions
from nesy_reasoning_mcp.auto_ingest.schemas import (
    CandidateRelationBatch,
    IngestionInput,
    IngestionMode,
    IngestionReport,
    ReviewDecision,
    ReviewDecisionBatch,
    ReviewVotingPolicy,
)
from nesy_reasoning_mcp.auto_ingest.writer import write_approved_relations
from nesy_reasoning_mcp.store import RelationStoreProtocol

AgentRunner = Callable[..., Awaitable[Any]]


@dataclass(frozen=True)
class OpenAICompatibleProviderConfig:
    """Configuration for OpenAI-compatible Chat Completions providers."""

    base_url: str
    api_key_env: str
    default_headers: Mapping[str, str] = field(default_factory=dict)
    disable_tracing: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "default_headers",
            MappingProxyType(dict(self.default_headers)),
        )


class OpenAIAgentsDryRunError(ValueError):
    """Raised when a live Agent SDK dry-run cannot start safely."""


async def run_openai_agents_dry_run(
    ingestion_input: IngestionInput,
    *,
    store: RelationStoreProtocol,
    model: str | None = None,
    reviewer_models: list[str] | None = None,
    voting_policy: ReviewVotingPolicy = ReviewVotingPolicy.RISK_TIERED,
    high_priority_reviewer_models: list[str] | None = None,
    env: Mapping[str, str] | None = None,
    run_agent: AgentRunner | None = None,
    provider_config: OpenAICompatibleProviderConfig | None = None,
    disable_tracing: bool = False,
) -> IngestionReport:
    """Extract, review, and gate candidate relations without persistent writes."""
    return await run_openai_agents_ingestion(
        ingestion_input,
        store=store,
        model=model,
        reviewer_models=reviewer_models,
        voting_policy=voting_policy,
        high_priority_reviewer_models=high_priority_reviewer_models,
        env=env,
        run_agent=run_agent,
        provider_config=provider_config,
        disable_tracing=disable_tracing,
        auto_write=False,
    )


async def run_openai_agents_ingestion(
    ingestion_input: IngestionInput,
    *,
    store: RelationStoreProtocol,
    model: str | None = None,
    reviewer_models: list[str] | None = None,
    voting_policy: ReviewVotingPolicy = ReviewVotingPolicy.RISK_TIERED,
    high_priority_reviewer_models: list[str] | None = None,
    env: Mapping[str, str] | None = None,
    run_agent: AgentRunner | None = None,
    auto_write: bool = False,
    min_write_confidence: float = 0.85,
    provider_config: OpenAICompatibleProviderConfig | None = None,
    disable_tracing: bool = False,
) -> IngestionReport:
    """Extract, review, gate, and optionally write approved candidate relations."""
    if not 0 <= min_write_confidence <= 1:
        raise OpenAIAgentsDryRunError("min_write_confidence must be between 0 and 1")
    runtime_env = env if env is not None else os.environ
    base_model_name = model or runtime_env.get("OPENAI_DEFAULT_MODEL")
    voting_policy = ReviewVotingPolicy(voting_policy)
    reviewer_model_names = _reviewer_model_names(reviewer_models, base_model_name)
    high_priority_models = _dedupe_model_names(high_priority_reviewer_models or [])
    agent_model = _agent_model(
        base_model_name,
        provider_config,
        runtime_env,
    )
    tracing_disabled = disable_tracing or (
        provider_config is not None and provider_config.disable_tracing
    )
    if run_agent is None and provider_config is None and not runtime_env.get("OPENAI_API_KEY"):
        raise OpenAIAgentsDryRunError(
            "OPENAI_API_KEY is required for live OpenAI Agents SDK ingestion"
        )

    extractor = _build_agent(
        name="NeSy relation extractor",
        instructions=_EXTRACTOR_INSTRUCTIONS,
        output_type=CandidateRelationBatch,
        model=agent_model,
    )
    extraction_output = await _run_agent_with_optional_runner(
        run_agent,
        extractor,
        _extraction_prompt(ingestion_input),
        tracing_disabled=tracing_disabled,
    )
    candidate_batch = _coerce_candidate_batch(extraction_output)

    reviews: list[ReviewDecision] = []
    for reviewer_model_name in reviewer_model_names:
        reviewer_agent_model = (
            agent_model
            if reviewer_model_name == base_model_name
            else _agent_model(reviewer_model_name, provider_config, runtime_env)
        )
        reviewer = _build_agent(
            name=_reviewer_agent_name(reviewer_model_name),
            instructions=_REVIEWER_INSTRUCTIONS,
            output_type=ReviewDecisionBatch,
            model=reviewer_agent_model,
        )
        review_output = await _run_agent_with_optional_runner(
            run_agent,
            reviewer,
            _review_prompt(ingestion_input, candidate_batch),
            tracing_disabled=tracing_disabled,
        )
        review_batch = _coerce_review_batch(review_output)
        reviews.extend(_reviews_with_model(review_batch.reviews, reviewer_model_name))

    aggregation = aggregate_review_decisions(
        candidates=candidate_batch.candidates,
        reviews=reviews,
        policy=voting_policy,
        high_priority_reviewer_models=high_priority_models,
        expected_reviewer_models=[
            reviewer_model_name
            for reviewer_model_name in reviewer_model_names
            if reviewer_model_name is not None
        ],
    )

    gate_results, approved_relations, diagnostics, reasoning = await run_dry_run_gate(
        candidates=candidate_batch.candidates,
        reviews=aggregation.gate_reviews,
        store=store,
        min_write_confidence=min_write_confidence if auto_write else 0.0,
        write_enabled=auto_write,
    )
    diagnostics = [*aggregation.diagnostics, *diagnostics]
    written_relation_ids: list[str] = []
    write_result: dict[str, Any] = {}
    if auto_write and approved_relations:
        written_relation_ids, write_diagnostics, write_result = await write_approved_relations(
            relations=approved_relations,
            store=store,
        )
        diagnostics = [*diagnostics, *write_diagnostics]

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
            "provider": _provider_metadata(provider_config, tracing_disabled),
        },
    )
    if auto_write:
        queued_records = queued_records_from_report(
            report,
            propositions=ingestion_input.propositions,
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
    return {
        "type": "openai_compatible",
        "header_keys": sorted(provider_config.default_headers),
        "tracing_disabled": disable_tracing or provider_config.disable_tracing,
    }


def _reviewer_model_names(
    reviewer_models: list[str] | None,
    default_model: str | None,
) -> list[str | None]:
    normalized = _dedupe_model_names(reviewer_models or [])
    if normalized:
        reviewer_names: list[str | None] = [*normalized]
        return reviewer_names
    return [default_model]


def _dedupe_model_names(values: list[str]) -> list[str]:
    stripped = [value.strip() for value in values]
    return list(dict.fromkeys(value for value in stripped if value))


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


def _extraction_prompt(ingestion_input: IngestionInput) -> str:
    return (
        "Extract only evidence-supported logical relations.\n"
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
        "candidates": candidate_batch.model_dump(mode="json", exclude_none=True),
    }
    return (
        "Review each candidate relation against the evidence.\n"
        "Use approve only when source text directly supports the final relation. "
        "Use reject for unsupported claims and needs_human for ambiguous claims.\n\n"
        f"Review payload JSON:\n{json.dumps(payload, ensure_ascii=False)}"
    )


def _input_json(ingestion_input: IngestionInput) -> str:
    return json.dumps(
        ingestion_input.model_dump(mode="json", exclude_none=True), ensure_ascii=False
    )


_EXTRACTOR_INSTRUCTIONS = """\
You extract candidate symbolic relations for NeSy Reasoning MCP.
Only emit sufficient, necessary, or equivalent relations directly supported by evidence.
Each candidate must cite at least one provided EvidenceRecord.
Do not turn "may improve", correlation, topical similarity, or vague support into a relation.
"""

_REVIEWER_INSTRUCTIONS = """\
You review candidate symbolic relations for NeSy Reasoning MCP.
For approve or downgrade, provide final_relation_type and final_confidence.
Prefer reject or needs_human when evidence is weak, ambiguous, or missing.
Do not approve claims that would require external knowledge not present in the evidence.
"""
