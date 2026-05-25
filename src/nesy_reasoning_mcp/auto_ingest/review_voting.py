"""Review decision aggregation for multi-reviewer ingestion."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any

from nesy_reasoning_mcp.auto_ingest.schemas import (
    CandidateRelation,
    ReviewDecision,
    ReviewDecisionValue,
    ReviewVotingPolicy,
)
from nesy_reasoning_mcp.schemas import Diagnostic, RelationType


@dataclass(frozen=True)
class ReviewAggregationResult:
    """Aggregated reviewer decisions and audit metadata."""

    gate_reviews: list[ReviewDecision]
    audit_reviews: list[ReviewDecision]
    metadata: dict[str, Any]
    diagnostics: list[Diagnostic]


def aggregate_review_decisions(
    *,
    candidates: list[CandidateRelation],
    reviews: list[ReviewDecision],
    policy: ReviewVotingPolicy = ReviewVotingPolicy.RISK_TIERED,
    high_priority_reviewer_models: list[str] | None = None,
    expected_reviewer_models: list[str] | None = None,
) -> ReviewAggregationResult:
    """Aggregate reviewer decisions while preserving individual audit votes."""
    high_priority_models = _dedupe_text(high_priority_reviewer_models or [])
    expected_models = _dedupe_text(expected_reviewer_models or [])
    candidate_ids = {candidate.id for candidate in candidates}
    reviews_by_candidate: dict[str, list[ReviewDecision]] = defaultdict(list)
    diagnostics: list[Diagnostic] = []
    for review in reviews:
        if review.candidate_id not in candidate_ids:
            diagnostics.append(
                Diagnostic(
                    level="warning",
                    code="UNKNOWN_REVIEW_CANDIDATE",
                    message=f"review references unknown candidate {review.candidate_id}",
                    related_ids=[review.candidate_id],
                )
            )
            continue
        reviews_by_candidate[review.candidate_id].append(review)

    gate_reviews: list[ReviewDecision] = []
    audit_reviews: list[ReviewDecision] = []
    candidate_metadata: list[dict[str, Any]] = []
    for candidate in candidates:
        candidate_reviews = [*reviews_by_candidate.get(candidate.id, [])]
        candidate_reviews.extend(
            _missing_reviewer_votes(
                candidate_id=candidate.id,
                reviews=candidate_reviews,
                expected_reviewer_models=expected_models,
                high_priority_reviewer_models=high_priority_models,
            )
        )
        audit_reviews.extend(candidate_reviews)
        gate_review = _aggregate_candidate_reviews(
            candidate_id=candidate.id,
            reviews=candidate_reviews,
            policy=policy,
            high_priority_reviewer_models=high_priority_models,
        )
        if gate_review is not None:
            gate_reviews.append(gate_review)
        candidate_metadata.append(
            _candidate_metadata(
                candidate_id=candidate.id,
                reviews=candidate_reviews,
                gate_review=gate_review,
                policy=policy,
            )
        )

    return ReviewAggregationResult(
        gate_reviews=gate_reviews,
        audit_reviews=audit_reviews,
        metadata={
            "policy": policy.value,
            "high_priority_reviewer_models": high_priority_models,
            "expected_reviewer_models": expected_models,
            "candidate_count": len(candidates),
            "review_count": len(audit_reviews),
            "candidates": candidate_metadata,
        },
        diagnostics=diagnostics,
    )


def _aggregate_candidate_reviews(
    *,
    candidate_id: str,
    reviews: list[ReviewDecision],
    policy: ReviewVotingPolicy,
    high_priority_reviewer_models: list[str],
) -> ReviewDecision | None:
    if not reviews:
        return None
    if len(reviews) == 1 and not _has_missing_high_priority_vote(
        reviews, high_priority_reviewer_models
    ):
        return reviews[0]
    if policy == ReviewVotingPolicy.UNANIMOUS:
        return _aggregate_unanimous(candidate_id, reviews, policy)
    if policy == ReviewVotingPolicy.MAJORITY:
        return _aggregate_majority(candidate_id, reviews, policy)
    return _aggregate_risk_tiered(
        candidate_id,
        reviews,
        policy,
        high_priority_reviewer_models,
    )


def _aggregate_unanimous(
    candidate_id: str,
    reviews: list[ReviewDecision],
    policy: ReviewVotingPolicy,
) -> ReviewDecision:
    if any(review.decision == ReviewDecisionValue.REJECT for review in reviews):
        return _aggregate_review(
            candidate_id=candidate_id,
            decision=ReviewDecisionValue.REJECT,
            reason="unanimous policy rejected because at least one reviewer rejected",
            reviews=reviews,
            policy=policy,
        )
    approvals = [review for review in reviews if review.decision == ReviewDecisionValue.APPROVE]
    relation_type, selected = _single_majority_relation_type(approvals, required=len(reviews))
    if len(approvals) == len(reviews) and relation_type is not None:
        return _aggregate_review(
            candidate_id=candidate_id,
            decision=ReviewDecisionValue.APPROVE,
            reason="unanimous policy approved all reviewer decisions",
            reviews=reviews,
            policy=policy,
            selected_reviews=selected,
            relation_type=relation_type,
        )
    return _aggregate_review(
        candidate_id=candidate_id,
        decision=ReviewDecisionValue.NEEDS_HUMAN,
        reason="unanimous policy queued reviewer disagreement",
        reviews=reviews,
        policy=policy,
    )


def _aggregate_majority(
    candidate_id: str,
    reviews: list[ReviewDecision],
    policy: ReviewVotingPolicy,
) -> ReviewDecision:
    approvals = [review for review in reviews if review.decision == ReviewDecisionValue.APPROVE]
    relation_type, selected = _single_majority_relation_type(
        approvals, required=(len(reviews) // 2) + 1
    )
    if relation_type is not None:
        return _aggregate_review(
            candidate_id=candidate_id,
            decision=ReviewDecisionValue.APPROVE,
            reason="majority policy approved strict reviewer majority",
            reviews=reviews,
            policy=policy,
            selected_reviews=selected,
            relation_type=relation_type,
        )
    reject_count = sum(1 for review in reviews if review.decision == ReviewDecisionValue.REJECT)
    if reject_count > len(reviews) / 2:
        return _aggregate_review(
            candidate_id=candidate_id,
            decision=ReviewDecisionValue.REJECT,
            reason="majority policy rejected strict reviewer majority",
            reviews=reviews,
            policy=policy,
        )
    return _aggregate_review(
        candidate_id=candidate_id,
        decision=ReviewDecisionValue.NEEDS_HUMAN,
        reason="majority policy queued reviewer disagreement",
        reviews=reviews,
        policy=policy,
    )


def _aggregate_risk_tiered(
    candidate_id: str,
    reviews: list[ReviewDecision],
    policy: ReviewVotingPolicy,
    high_priority_reviewer_models: list[str],
) -> ReviewDecision:
    high_priority_reviews = [
        review for review in reviews if review.reviewer_model in high_priority_reviewer_models
    ]
    for review in high_priority_reviews:
        if review.decision == ReviewDecisionValue.REJECT:
            return _aggregate_review(
                candidate_id=candidate_id,
                decision=ReviewDecisionValue.REJECT,
                reason="risk_tiered policy rejected high-priority reviewer veto",
                reviews=reviews,
                policy=policy,
            )
    for review in high_priority_reviews:
        if review.decision in {
            ReviewDecisionValue.DOWNGRADE,
            ReviewDecisionValue.NEEDS_HUMAN,
        }:
            return _aggregate_review(
                candidate_id=candidate_id,
                decision=ReviewDecisionValue.NEEDS_HUMAN,
                reason="risk_tiered policy queued high-priority reviewer concern",
                reviews=reviews,
                policy=policy,
            )

    high_priority_approvals = [
        review for review in high_priority_reviews if review.decision == ReviewDecisionValue.APPROVE
    ]
    high_priority_type = _single_relation_type(high_priority_approvals)
    if high_priority_approvals and high_priority_type is None:
        return _aggregate_review(
            candidate_id=candidate_id,
            decision=ReviewDecisionValue.NEEDS_HUMAN,
            reason="risk_tiered policy queued high-priority relation type disagreement",
            reviews=reviews,
            policy=policy,
        )

    approvals = [review for review in reviews if review.decision == ReviewDecisionValue.APPROVE]
    relation_type, selected = _single_majority_relation_type(
        approvals, required=(len(reviews) // 2) + 1
    )
    if relation_type is not None and (
        high_priority_type is None or relation_type == high_priority_type
    ):
        return _aggregate_review(
            candidate_id=candidate_id,
            decision=ReviewDecisionValue.APPROVE,
            reason="risk_tiered policy approved strict reviewer majority",
            reviews=reviews,
            policy=policy,
            selected_reviews=selected,
            relation_type=relation_type,
        )
    return _aggregate_review(
        candidate_id=candidate_id,
        decision=ReviewDecisionValue.NEEDS_HUMAN,
        reason="risk_tiered policy queued without strict safe majority",
        reviews=reviews,
        policy=policy,
    )


def _aggregate_review(
    *,
    candidate_id: str,
    decision: ReviewDecisionValue,
    reason: str,
    reviews: list[ReviewDecision],
    policy: ReviewVotingPolicy,
    selected_reviews: list[ReviewDecision] | None = None,
    relation_type: RelationType | None = None,
) -> ReviewDecision:
    selected = selected_reviews or []
    kwargs: dict[str, Any] = {
        "candidate_id": candidate_id,
        "decision": decision,
        "reasons": [reason],
        "risk_flags": _unique_text(flag for review in reviews for flag in review.risk_flags),
        "reviewer_model": f"aggregate:{policy.value}",
        "metadata": {
            "review_aggregation": {
                "policy": policy.value,
                "review_count": len(reviews),
                "votes": [_review_vote_summary(review) for review in reviews],
                "selected_reviewers": [
                    review.reviewer_model for review in selected if review.reviewer_model
                ],
            }
        },
    }
    if decision in {ReviewDecisionValue.APPROVE, ReviewDecisionValue.DOWNGRADE}:
        kwargs["final_relation_type"] = relation_type
        kwargs["final_confidence"] = min(
            review.final_confidence for review in selected if review.final_confidence is not None
        )
    return ReviewDecision(**kwargs)


def _candidate_metadata(
    *,
    candidate_id: str,
    reviews: list[ReviewDecision],
    gate_review: ReviewDecision | None,
    policy: ReviewVotingPolicy,
) -> dict[str, Any]:
    return {
        "candidate_id": candidate_id,
        "policy": policy.value,
        "review_count": len(reviews),
        "agreement": _reviews_agree(reviews),
        "aggregate_decision": gate_review.decision.value if gate_review else None,
        "aggregate_reviewer_model": gate_review.reviewer_model if gate_review else None,
        "votes": [_review_vote_summary(review) for review in reviews],
    }


def _review_vote_summary(review: ReviewDecision) -> dict[str, Any]:
    return review.model_dump(
        mode="json",
        exclude_none=True,
        exclude={"metadata"},
    )


def _reviews_agree(reviews: list[ReviewDecision]) -> bool:
    if len(reviews) <= 1:
        return True
    decisions = {review.decision for review in reviews}
    relation_types = {
        review.final_relation_type
        for review in reviews
        if review.decision == ReviewDecisionValue.APPROVE
    }
    return decisions == {ReviewDecisionValue.APPROVE} and len(relation_types) == 1


def _single_majority_relation_type(
    approvals: list[ReviewDecision],
    *,
    required: int,
) -> tuple[RelationType | None, list[ReviewDecision]]:
    counts = Counter(review.final_relation_type for review in approvals)
    if not counts:
        return None, []
    relation_type, count = counts.most_common(1)[0]
    if relation_type is None or count < required:
        return None, []
    selected = [review for review in approvals if review.final_relation_type == relation_type]
    return relation_type, selected


def _single_relation_type(reviews: list[ReviewDecision]) -> RelationType | None:
    relation_types = {review.final_relation_type for review in reviews}
    if len(relation_types) != 1:
        return None
    return next(iter(relation_types))


def _missing_reviewer_votes(
    *,
    candidate_id: str,
    reviews: list[ReviewDecision],
    expected_reviewer_models: list[str],
    high_priority_reviewer_models: list[str],
) -> list[ReviewDecision]:
    present_models = {
        review.reviewer_model for review in reviews if review.reviewer_model is not None
    }
    missing_models = [
        model
        for model in _dedupe_text([*expected_reviewer_models, *high_priority_reviewer_models])
        if model not in present_models
    ]
    return [
        ReviewDecision(
            candidate_id=candidate_id,
            decision=ReviewDecisionValue.NEEDS_HUMAN,
            reasons=[f"missing reviewer decision from {model}"],
            reviewer_model=model,
            metadata={"synthetic_vote": "missing_reviewer_decision"},
        )
        for model in missing_models
    ]


def _has_missing_high_priority_vote(
    reviews: list[ReviewDecision],
    high_priority_reviewer_models: list[str],
) -> bool:
    present_models = {
        review.reviewer_model for review in reviews if review.reviewer_model is not None
    }
    return any(model not in present_models for model in high_priority_reviewer_models)


def _dedupe_text(values: list[str]) -> list[str]:
    stripped = [value.strip() for value in values]
    return list(dict.fromkeys(value for value in stripped if value))


def _unique_text(values: Any) -> list[str]:
    return list(dict.fromkeys(value for value in values if isinstance(value, str) and value))
