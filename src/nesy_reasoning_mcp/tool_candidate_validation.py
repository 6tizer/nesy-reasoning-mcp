"""Candidate relation validation helper for Agent SDK orchestrators."""

from __future__ import annotations

from typing import Any

from nesy_reasoning_mcp.auto_ingest.gate import run_dry_run_gate
from nesy_reasoning_mcp.auto_ingest.review_voting import aggregate_review_decisions
from nesy_reasoning_mcp.auto_ingest.schemas import (
    GateAction,
    GateResult,
    ValidateCandidateRelationsInput,
)
from nesy_reasoning_mcp.schemas import CheckContradictionsInput, ContradictionMode, Diagnostic
from nesy_reasoning_mcp.store import RelationStoreProtocol
from nesy_reasoning_mcp.tool_relations import check_contradictions


async def validate_candidate_relations(
    arguments: dict[str, Any],
    store: RelationStoreProtocol,
) -> dict[str, Any]:
    """Validate reviewed candidate relations without writing durable graph state."""
    payload = ValidateCandidateRelationsInput.model_validate(arguments)
    aggregation = aggregate_review_decisions(
        candidates=payload.candidates,
        reviews=payload.reviews,
        policy=payload.voting_policy,
        high_priority_reviewer_models=payload.high_priority_reviewer_models,
    )
    gate_results, approved_relations, diagnostics, candidate_reasoning = await run_dry_run_gate(
        candidates=payload.candidates,
        reviews=aggregation.gate_reviews,
        store=store,
        min_write_confidence=payload.min_write_confidence,
        write_enabled=True,
        propositions=payload.propositions,
        include_soft=payload.include_soft,
        max_depth=payload.max_depth,
        min_confidence=payload.min_confidence,
    )
    diagnostics = [*aggregation.diagnostics, *diagnostics]

    combined_reasoning: dict[str, Any] = {}
    if approved_relations:
        combined_reasoning = await check_contradictions(
            CheckContradictionsInput(
                facts=approved_relations,
                propositions=payload.propositions,
                mode=ContradictionMode.COMBINED,
                include_soft=payload.include_soft,
                max_depth=payload.max_depth,
                min_confidence=payload.min_confidence,
            ).model_dump(mode="json"),
            store,
        )
        diagnostics = [*diagnostics, *_diagnostics_from_structured(combined_reasoning)]
        if combined_reasoning.get("status") == "error":
            gate_results = _queue_gate_approved(
                gate_results,
                reason="combined contradiction check failed",
            )
            approved_relations = []
        elif _has_hard_contradiction(combined_reasoning):
            gate_results = _queue_gate_approved(
                gate_results,
                reason="hard contradiction found against current graph",
            )
            approved_relations = []

    queued_count = sum(1 for item in gate_results if item.action == GateAction.QUEUE)
    rejected_count = sum(1 for item in gate_results if item.action == GateAction.REJECT)
    status = _status(diagnostics, queued_count, rejected_count)
    reasoning = {
        "candidate_set": candidate_reasoning,
        "combined": combined_reasoning,
    }
    return {
        "status": status,
        "persisted": False,
        "candidate_count": len(payload.candidates),
        "approved_count": len(approved_relations),
        "queued_count": queued_count,
        "rejected_count": rejected_count,
        "gate_results": [item.model_dump(mode="json") for item in gate_results],
        "approved_relations": [
            relation.model_dump(mode="json", exclude_none=True) for relation in approved_relations
        ],
        "review_aggregation": aggregation.metadata,
        "diagnostics": [item.model_dump(mode="json") for item in diagnostics],
        "reasoning": reasoning,
        "graph_stats": _graph_stats(reasoning, store),
        "trace": _trace(candidate_reasoning, combined_reasoning),
    }


def _queue_gate_approved(gate_results: list[GateResult], *, reason: str) -> list[GateResult]:
    queued: list[GateResult] = []
    for item in gate_results:
        if item.action == GateAction.AUTO_WRITE:
            queued.append(
                item.model_copy(
                    update={
                        "action": GateAction.QUEUE,
                        "reasons": [reason],
                    }
                )
            )
        else:
            queued.append(item)
    return queued


def _has_hard_contradiction(reasoning: dict[str, Any]) -> bool:
    return any(
        item.get("severity") == "hard"
        for item in reasoning.get("contradictions", [])
        if isinstance(item, dict)
    )


def _diagnostics_from_structured(structured: dict[str, Any]) -> list[Diagnostic]:
    diagnostics: list[Diagnostic] = []
    for item in structured.get("diagnostics", []):
        if isinstance(item, dict):
            diagnostics.append(Diagnostic.model_validate(item))
    return diagnostics


def _status(diagnostics: list[Diagnostic], queued_count: int, rejected_count: int) -> str:
    if any(item.level == "error" for item in diagnostics):
        return "error"
    if queued_count or rejected_count:
        return "warning"
    return "ok"


def _graph_stats(reasoning: dict[str, Any], store: RelationStoreProtocol) -> dict[str, Any]:
    combined = reasoning.get("combined", {})
    if isinstance(combined, dict) and isinstance(combined.get("graph_stats"), dict):
        return combined["graph_stats"]
    candidate_set = reasoning.get("candidate_set", {})
    if isinstance(candidate_set, dict) and isinstance(candidate_set.get("graph_stats"), dict):
        return candidate_set["graph_stats"]
    return store.graph_stats().model_dump(mode="json")


def _trace(candidate_reasoning: dict[str, Any], combined_reasoning: dict[str, Any]) -> list[str]:
    trace = ["Validated candidate relations without persisting graph state."]
    for item in candidate_reasoning.get("trace", []):
        if isinstance(item, str):
            trace.append(item)
    for item in combined_reasoning.get("trace", []):
        if isinstance(item, str):
            trace.append(item)
    return trace
