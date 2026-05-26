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
from nesy_reasoning_mcp.auto_ingest.semantic_dedupe import (
    SemanticDuplicateConcern,
    semantic_duplicate_concerns,
)
from nesy_reasoning_mcp.normalization import normalized_implication_preview
from nesy_reasoning_mcp.schemas import Diagnostic, PropositionRecord, RelationInput, RelationType
from nesy_reasoning_mcp.store import RelationStoreProtocol
from nesy_reasoning_mcp.tool_names import REASON_OVER_RELATIONS
from nesy_reasoning_mcp.tool_registry import call_tool


async def run_dry_run_gate(
    *,
    candidates: list[CandidateRelation],
    reviews: list[ReviewDecision],
    store: RelationStoreProtocol,
    min_write_confidence: float = 0.0,
    write_enabled: bool = False,
    propositions: list[PropositionRecord] | None = None,
    include_soft: bool = False,
    max_depth: int = 8,
    min_confidence: float = 0.0,
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
        if (
            review.decision
            in {
                ReviewDecisionValue.APPROVE,
                ReviewDecisionValue.DOWNGRADE,
            }
            and review.normalized_implication_supported is not True
        ):
            gate_results.append(
                GateResult(
                    candidate_id=candidate.id,
                    action=GateAction.QUEUE,
                    reasons=["normalized implication support was not confirmed"],
                    metadata={
                        "review_reasons": review.reasons,
                        "normalized_implications": _normalized_implication_metadata(
                            candidate,
                            review,
                        ),
                    },
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

    approved_relation_items = [
        (candidate, review, _relation_from_review(candidate, review))
        for candidate, review in approved_candidates
    ]
    approved_relations = [relation for _candidate, _review, relation in approved_relation_items]
    diagnostics: list[Diagnostic] = []
    reasoning: dict[str, Any] = {}
    hard_contradiction = False
    if approved_relations:
        reasoning = await _check_approved_relations(
            approved_relations,
            store,
            propositions=propositions or [],
            include_soft=include_soft,
            max_depth=max_depth,
            min_confidence=min_confidence,
        )
        reasoning_failed = reasoning.get("status") == "error"
        hard_contradiction = _has_hard_contradiction(reasoning)
        diagnostics = _diagnostics_from_reasoning(reasoning)
    else:
        reasoning_failed = False

    semantic_duplicates = _semantic_duplicate_concerns_by_candidate(
        approved_relation_items,
        store=store,
        propositions=propositions or [],
        enabled=write_enabled and not reasoning_failed and not hard_contradiction,
    )
    diagnostics.extend(
        _semantic_duplicate_diagnostics(semantic_duplicates, approved_relation_items)
    )

    for candidate, review, _relation in approved_relation_items:
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
        semantic_duplicate = semantic_duplicates.get(candidate.id)
        if semantic_duplicate is not None:
            gate_results.append(
                GateResult(
                    candidate_id=candidate.id,
                    action=GateAction.QUEUE,
                    reasons=["likely semantic duplicate relation requires human review"],
                    metadata={
                        "review_reasons": review.reasons,
                        "semantic_duplicate": semantic_duplicate.to_metadata(),
                    },
                )
            )
            continue
        gate_results.append(
            GateResult(
                candidate_id=candidate.id,
                action=GateAction.AUTO_WRITE,
                reasons=[
                    "write approved; persistent assertion may proceed"
                    if write_enabled
                    else "dry-run approved; no persistent write performed"
                ],
                metadata={"review_reasons": review.reasons},
            )
        )

    durable_approved = (
        []
        if hard_contradiction or reasoning_failed
        else [
            relation
            for candidate, _review, relation in approved_relation_items
            if candidate.id not in semantic_duplicates
        ]
    )
    return gate_results, durable_approved, diagnostics, reasoning


def _semantic_duplicate_concerns_by_candidate(
    approved_relation_items: list[tuple[CandidateRelation, ReviewDecision, RelationInput]],
    *,
    store: RelationStoreProtocol,
    propositions: list[PropositionRecord],
    enabled: bool,
) -> dict[str, SemanticDuplicateConcern]:
    if not enabled or not approved_relation_items:
        return {}
    concerns = semantic_duplicate_concerns(
        relations=[relation for _candidate, _review, relation in approved_relation_items],
        existing_relations=store.list_relations(),
        propositions=[*store.list_propositions(), *propositions],
    )
    return {
        candidate.id: concern
        for (candidate, _review, _relation), concern in zip(
            approved_relation_items,
            concerns,
            strict=True,
        )
        if concern is not None
    }


def _semantic_duplicate_diagnostics(
    semantic_duplicates: dict[str, SemanticDuplicateConcern],
    approved_relation_items: list[tuple[CandidateRelation, ReviewDecision, RelationInput]],
) -> list[Diagnostic]:
    candidates = {
        candidate.id: candidate for candidate, _review, _relation in approved_relation_items
    }
    diagnostics: list[Diagnostic] = []
    for candidate_id, concern in semantic_duplicates.items():
        candidate = candidates[candidate_id]
        diagnostics.append(
            Diagnostic(
                level="warning",
                code="SEMANTIC_DUPLICATE_CANDIDATE",
                message="Candidate is likely a semantic duplicate of existing relation(s).",
                related_ids=[candidate.id, *concern.existing_relation_ids],
            )
        )
    return diagnostics


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
        "normalized_implication_supported": review.normalized_implication_supported,
        "metadata": review.metadata,
    }
    return relation.model_copy(update={"provenance": provenance})


def _normalized_implication_metadata(
    candidate: CandidateRelation,
    review: ReviewDecision,
) -> dict[str, Any]:
    relation_type = RelationType(review.final_relation_type or candidate.relation_type)
    return {
        "relation_type": relation_type.value,
        "edges": normalized_implication_preview(
            candidate.source,
            candidate.target,
            relation_type,
        ),
    }


async def _check_approved_relations(
    relations: list[RelationInput],
    store: RelationStoreProtocol,
    *,
    propositions: list[PropositionRecord],
    include_soft: bool,
    max_depth: int,
    min_confidence: float,
) -> dict[str, Any]:
    arguments = {
        "relations": [
            relation.model_dump(mode="json", exclude_none=True) for relation in relations
        ],
        "propositions": [
            proposition.model_dump(mode="json", exclude_none=True) for proposition in propositions
        ],
        "max_depth": max_depth,
        "min_confidence": min_confidence,
        "query": {
            "mode": "check_contradictions",
            "include_soft": include_soft,
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
