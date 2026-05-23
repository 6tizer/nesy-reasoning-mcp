"""Dry-run deterministic gate for Agent SDK candidate ingestion."""

from __future__ import annotations

from typing import Any

from nesy_reasoning_mcp.auto_ingest.policy import DRY_RUN_TOOL_ALLOWLIST
from nesy_reasoning_mcp.auto_ingest.schemas import (
    CandidateRelation,
    GateAction,
    GateResult,
    ReviewDecision,
    ReviewDecisionValue,
)
from nesy_reasoning_mcp.schemas import Diagnostic, RelationInput
from nesy_reasoning_mcp.store import RelationStoreProtocol
from nesy_reasoning_mcp.tool_names import REASON_OVER_RELATIONS
from nesy_reasoning_mcp.tool_registry import call_tool


async def run_dry_run_gate(
    *,
    candidates: list[CandidateRelation],
    reviews: list[ReviewDecision],
    store: RelationStoreProtocol,
    min_write_confidence: float = 0.0,
) -> tuple[list[GateResult], list[RelationInput], list[Diagnostic], dict[str, Any]]:
    """Gate reviewed candidates without calling any persistent write tool."""
    if REASON_OVER_RELATIONS not in DRY_RUN_TOOL_ALLOWLIST:
        raise RuntimeError("dry-run policy does not allow ephemeral reasoning")

    review_by_id = {review.candidate_id: review for review in reviews}
    gate_results: list[GateResult] = []
    approved_candidates: list[tuple[CandidateRelation, ReviewDecision]] = []

    for candidate in candidates:
        review = review_by_id.get(candidate.id)
        if review is None:
            gate_results.append(
                GateResult(
                    candidate_id=candidate.id,
                    action=GateAction.QUEUE,
                    reasons=["missing reviewer decision"],
                )
            )
            continue
        if review.decision == ReviewDecisionValue.REJECT:
            gate_results.append(
                GateResult(
                    candidate_id=candidate.id,
                    action=GateAction.REJECT,
                    reasons=review.reasons or ["reviewer rejected candidate"],
                )
            )
            continue
        if review.decision in {
            ReviewDecisionValue.DOWNGRADE,
            ReviewDecisionValue.NEEDS_HUMAN,
        }:
            gate_results.append(
                GateResult(
                    candidate_id=candidate.id,
                    action=GateAction.QUEUE,
                    reasons=review.reasons or [f"reviewer decision: {review.decision}"],
                )
            )
            continue
        final_confidence = (
            review.final_confidence if review.final_confidence is not None else candidate.confidence
        )
        if final_confidence < min_write_confidence:
            gate_results.append(
                GateResult(
                    candidate_id=candidate.id,
                    action=GateAction.QUEUE,
                    reasons=[
                        f"final confidence {final_confidence:.3f} below write threshold "
                        f"{min_write_confidence:.3f}"
                    ],
                    metadata={"review_reasons": review.reasons},
                )
            )
            continue
        approved_candidates.append((candidate, review))

    approved_relations = [
        _relation_from_review(candidate, review) for candidate, review in approved_candidates
    ]
    diagnostics: list[Diagnostic] = []
    reasoning: dict[str, Any] = {}
    hard_contradiction = False
    if approved_relations:
        reasoning = await _check_approved_relations(approved_relations, store)
        reasoning_failed = reasoning.get("status") == "error"
        hard_contradiction = _has_hard_contradiction(reasoning)
        diagnostics = _diagnostics_from_reasoning(reasoning)
    else:
        reasoning_failed = False

    for candidate, review in approved_candidates:
        if reasoning_failed:
            gate_results.append(
                GateResult(
                    candidate_id=candidate.id,
                    action=GateAction.QUEUE,
                    reasons=["dry-run reasoning failed"],
                    metadata={"review_reasons": review.reasons},
                )
            )
            continue
        if hard_contradiction:
            gate_results.append(
                GateResult(
                    candidate_id=candidate.id,
                    action=GateAction.QUEUE,
                    reasons=["hard contradiction found in dry-run reasoning"],
                    metadata={"review_reasons": review.reasons},
                )
            )
            continue
        gate_results.append(
            GateResult(
                candidate_id=candidate.id,
                action=GateAction.AUTO_WRITE,
                reasons=["dry-run approved; no persistent write performed"],
                metadata={"review_reasons": review.reasons},
            )
        )

    durable_approved = [] if hard_contradiction or reasoning_failed else approved_relations
    return gate_results, durable_approved, diagnostics, reasoning


def _relation_from_review(candidate: CandidateRelation, review: ReviewDecision) -> RelationInput:
    reviewed = candidate.model_copy(
        update={
            "relation_type": review.final_relation_type or candidate.relation_type,
            "confidence": review.final_confidence
            if review.final_confidence is not None
            else candidate.confidence,
        }
    )
    relation = reviewed.to_relation_input()
    provenance = dict(relation.provenance or {})
    provenance["review"] = {
        "decision": review.decision,
        "reasons": review.reasons,
        "risk_flags": review.risk_flags,
        "reviewer_model": review.reviewer_model,
        "metadata": review.metadata,
    }
    return relation.model_copy(update={"provenance": provenance})


async def _check_approved_relations(
    relations: list[RelationInput],
    store: RelationStoreProtocol,
) -> dict[str, Any]:
    arguments = {
        "relations": [
            relation.model_dump(mode="json", exclude_none=True) for relation in relations
        ],
        "query": {
            "mode": "check_contradictions",
            "include_soft": False,
        },
    }
    result = await call_tool(REASON_OVER_RELATIONS, arguments, store)
    structured = result.structuredContent or {}
    if result.isError:
        diagnostics = structured.get("diagnostics", [])
        return {
            "status": "error",
            "diagnostics": diagnostics,
            "result": {},
        }
    return dict(structured)


def _has_hard_contradiction(reasoning: dict[str, Any]) -> bool:
    result = reasoning.get("result", {})
    if not isinstance(result, dict):
        return False
    if result.get("has_contradictions") is not True:
        return False
    return any(
        item.get("severity") == "hard"
        for item in result.get("contradictions", [])
        if isinstance(item, dict)
    )


def _diagnostics_from_reasoning(reasoning: dict[str, Any]) -> list[Diagnostic]:
    diagnostics: list[Diagnostic] = []
    for item in reasoning.get("diagnostics", []):
        if isinstance(item, dict):
            diagnostics.append(Diagnostic.model_validate(item))
    return diagnostics
