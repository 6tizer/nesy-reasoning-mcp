"""OpenAI Agents SDK orchestration for dry-run candidate ingestion."""

from __future__ import annotations

import json
import os
from collections.abc import Awaitable, Callable, Mapping
from typing import Any

from nesy_reasoning_mcp.auto_ingest.gate import run_dry_run_gate
from nesy_reasoning_mcp.auto_ingest.schemas import (
    CandidateRelationBatch,
    IngestionInput,
    IngestionMode,
    IngestionReport,
    ReviewDecisionBatch,
)
from nesy_reasoning_mcp.auto_ingest.writer import write_approved_relations
from nesy_reasoning_mcp.store import RelationStoreProtocol

AgentRunner = Callable[[Any, str], Awaitable[Any]]


class OpenAIAgentsDryRunError(ValueError):
    """Raised when a live Agent SDK dry-run cannot start safely."""


async def run_openai_agents_dry_run(
    ingestion_input: IngestionInput,
    *,
    store: RelationStoreProtocol,
    model: str | None = None,
    env: Mapping[str, str] | None = None,
    run_agent: AgentRunner | None = None,
) -> IngestionReport:
    """Extract, review, and gate candidate relations without persistent writes."""
    return await run_openai_agents_ingestion(
        ingestion_input,
        store=store,
        model=model,
        env=env,
        run_agent=run_agent,
        auto_write=False,
    )


async def run_openai_agents_ingestion(
    ingestion_input: IngestionInput,
    *,
    store: RelationStoreProtocol,
    model: str | None = None,
    env: Mapping[str, str] | None = None,
    run_agent: AgentRunner | None = None,
    auto_write: bool = False,
    min_write_confidence: float = 0.85,
) -> IngestionReport:
    """Extract, review, gate, and optionally write approved candidate relations."""
    if not 0 <= min_write_confidence <= 1:
        raise OpenAIAgentsDryRunError("min_write_confidence must be between 0 and 1")
    runtime_env = env if env is not None else os.environ
    if run_agent is None and not runtime_env.get("OPENAI_API_KEY"):
        raise OpenAIAgentsDryRunError(
            "OPENAI_API_KEY is required for live OpenAI Agents SDK ingestion"
        )

    extractor = _build_agent(
        name="NeSy relation extractor",
        instructions=_EXTRACTOR_INSTRUCTIONS,
        output_type=CandidateRelationBatch,
        model=model or runtime_env.get("OPENAI_DEFAULT_MODEL"),
    )
    extraction_output = await (run_agent or _run_agent)(
        extractor,
        _extraction_prompt(ingestion_input),
    )
    candidate_batch = _coerce_candidate_batch(extraction_output)

    reviewer = _build_agent(
        name="NeSy relation reviewer",
        instructions=_REVIEWER_INSTRUCTIONS,
        output_type=ReviewDecisionBatch,
        model=model or runtime_env.get("OPENAI_DEFAULT_MODEL"),
    )
    review_output = await (run_agent or _run_agent)(
        reviewer,
        _review_prompt(ingestion_input, candidate_batch),
    )
    review_batch = _coerce_review_batch(review_output)

    gate_results, approved_relations, diagnostics, reasoning = await run_dry_run_gate(
        candidates=candidate_batch.candidates,
        reviews=review_batch.reviews,
        store=store,
        min_write_confidence=min_write_confidence if auto_write else 0.0,
        write_enabled=auto_write,
    )
    written_relation_ids: list[str] = []
    write_result: dict[str, Any] = {}
    if auto_write and approved_relations:
        written_relation_ids, write_diagnostics, write_result = await write_approved_relations(
            relations=approved_relations,
            store=store,
        )
        diagnostics = [*diagnostics, *write_diagnostics]

    return IngestionReport(
        mode=IngestionMode.WRITE if auto_write else IngestionMode.DRY_RUN,
        candidates=candidate_batch.candidates,
        reviews=review_batch.reviews,
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
            "write_result": write_result,
        },
    )


def _build_agent(
    *,
    name: str,
    instructions: str,
    output_type: type[Any],
    model: str | None,
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


async def _run_agent(agent: Any, prompt: str) -> Any:
    from agents import Runner

    result = await Runner.run(agent, prompt)
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
