from nesy_reasoning_mcp.auto_ingest import (
    CandidateRelation,
    EvidenceRecord,
    ReviewDecision,
    ReviewDecisionValue,
    ReviewVotingPolicy,
)
from nesy_reasoning_mcp.auto_ingest.review_voting import aggregate_review_decisions
from nesy_reasoning_mcp.schemas import RelationType


def _candidate(candidate_id: str = "candidate-1") -> CandidateRelation:
    return CandidateRelation(
        id=candidate_id,
        source="A",
        target="B",
        relation_type="sufficient",
        confidence=0.9,
        evidence=[EvidenceRecord(url="https://example.com/source", span="A enables B.")],
    )


def _review(
    candidate_id: str = "candidate-1",
    *,
    decision: ReviewDecisionValue = ReviewDecisionValue.APPROVE,
    reviewer_model: str = "reviewer-a",
    relation_type: str = "sufficient",
    confidence: float = 0.9,
) -> ReviewDecision:
    kwargs = {
        "candidate_id": candidate_id,
        "decision": decision,
        "reasons": [f"{reviewer_model} reason"],
        "reviewer_model": reviewer_model,
    }
    if decision in {ReviewDecisionValue.APPROVE, ReviewDecisionValue.DOWNGRADE}:
        kwargs["final_relation_type"] = relation_type
        kwargs["final_confidence"] = confidence
        kwargs["normalized_implication_supported"] = True
    return ReviewDecision(**kwargs)


def test_single_reviewer_passes_through_original_review() -> None:
    candidate = _candidate()
    review = _review()

    result = aggregate_review_decisions(candidates=[candidate], reviews=[review])

    assert result.gate_reviews == [review]
    assert result.audit_reviews == [review]
    assert result.metadata["candidates"][0]["agreement"] is True


def test_unanimous_approves_all_matching_reviews_with_min_confidence() -> None:
    candidate = _candidate()

    result = aggregate_review_decisions(
        candidates=[candidate],
        reviews=[
            _review(reviewer_model="reviewer-a", confidence=0.9),
            _review(reviewer_model="reviewer-b", confidence=0.86),
        ],
        policy=ReviewVotingPolicy.UNANIMOUS,
    )

    gate_review = result.gate_reviews[0]
    assert gate_review.decision == ReviewDecisionValue.APPROVE
    assert gate_review.final_confidence == 0.86
    assert gate_review.reviewer_model == "aggregate:unanimous"


def test_unanimous_queues_disagreement_and_rejects_any_reject() -> None:
    candidate = _candidate()
    queued = aggregate_review_decisions(
        candidates=[candidate],
        reviews=[
            _review(reviewer_model="reviewer-a"),
            _review(reviewer_model="reviewer-b", relation_type="necessary"),
        ],
        policy=ReviewVotingPolicy.UNANIMOUS,
    )
    rejected = aggregate_review_decisions(
        candidates=[candidate],
        reviews=[
            _review(reviewer_model="reviewer-a"),
            _review(reviewer_model="reviewer-b", decision=ReviewDecisionValue.REJECT),
        ],
        policy=ReviewVotingPolicy.UNANIMOUS,
    )

    assert queued.gate_reviews[0].decision == ReviewDecisionValue.NEEDS_HUMAN
    assert rejected.gate_reviews[0].decision == ReviewDecisionValue.REJECT


def test_majority_approves_strict_matching_majority_and_queues_tie() -> None:
    candidate = _candidate()
    approved = aggregate_review_decisions(
        candidates=[candidate],
        reviews=[
            _review(reviewer_model="reviewer-a", confidence=0.91),
            _review(reviewer_model="reviewer-b", confidence=0.87),
            _review(reviewer_model="reviewer-c", decision=ReviewDecisionValue.NEEDS_HUMAN),
        ],
        policy=ReviewVotingPolicy.MAJORITY,
    )
    tied = aggregate_review_decisions(
        candidates=[candidate],
        reviews=[
            _review(reviewer_model="reviewer-a"),
            _review(reviewer_model="reviewer-b", decision=ReviewDecisionValue.NEEDS_HUMAN),
        ],
        policy=ReviewVotingPolicy.MAJORITY,
    )

    assert approved.gate_reviews[0].decision == ReviewDecisionValue.APPROVE
    assert approved.gate_reviews[0].final_confidence == 0.87
    assert tied.gate_reviews[0].decision == ReviewDecisionValue.NEEDS_HUMAN


def test_majority_rejects_strict_reject_majority() -> None:
    candidate = _candidate()

    result = aggregate_review_decisions(
        candidates=[candidate],
        reviews=[
            _review(reviewer_model="reviewer-a", decision=ReviewDecisionValue.REJECT),
            _review(reviewer_model="reviewer-b", decision=ReviewDecisionValue.REJECT),
            _review(reviewer_model="reviewer-c"),
        ],
        policy=ReviewVotingPolicy.MAJORITY,
    )

    assert result.gate_reviews[0].decision == ReviewDecisionValue.REJECT


def test_risk_tiered_honors_high_priority_veto_and_missing_vote() -> None:
    candidate = _candidate()
    rejected = aggregate_review_decisions(
        candidates=[candidate],
        reviews=[
            _review(reviewer_model="senior", decision=ReviewDecisionValue.REJECT),
            _review(reviewer_model="reviewer-a"),
            _review(reviewer_model="reviewer-b"),
        ],
        high_priority_reviewer_models=["senior"],
    )
    missing = aggregate_review_decisions(
        candidates=[candidate],
        reviews=[_review(reviewer_model="reviewer-a"), _review(reviewer_model="reviewer-b")],
        high_priority_reviewer_models=["senior"],
    )

    assert rejected.gate_reviews[0].decision == ReviewDecisionValue.REJECT
    assert missing.gate_reviews[0].decision == ReviewDecisionValue.NEEDS_HUMAN
    assert any(review.reviewer_model == "senior" for review in missing.audit_reviews)


def test_risk_tiered_queues_high_priority_downgrade() -> None:
    candidate = _candidate()

    result = aggregate_review_decisions(
        candidates=[candidate],
        reviews=[
            _review(reviewer_model="senior", decision=ReviewDecisionValue.DOWNGRADE),
            _review(reviewer_model="reviewer-a"),
            _review(reviewer_model="reviewer-b"),
        ],
        high_priority_reviewer_models=["senior"],
    )

    assert result.gate_reviews[0].decision == ReviewDecisionValue.NEEDS_HUMAN
    assert "high-priority reviewer concern" in result.gate_reviews[0].reasons[0]


def test_positive_aggregate_without_confidence_queues_instead_of_crashing() -> None:
    candidate = _candidate()
    first = ReviewDecision.model_construct(
        candidate_id=candidate.id,
        decision=ReviewDecisionValue.APPROVE,
        final_relation_type=RelationType.SUFFICIENT,
        final_confidence=None,
        reasons=["corrupt reviewer output"],
        risk_flags=[],
        reviewer_model="reviewer-a",
        metadata={},
    )
    second = ReviewDecision.model_construct(
        candidate_id=candidate.id,
        decision=ReviewDecisionValue.APPROVE,
        final_relation_type=RelationType.SUFFICIENT,
        final_confidence=None,
        reasons=["corrupt reviewer output"],
        risk_flags=[],
        reviewer_model="reviewer-b",
        metadata={},
    )

    result = aggregate_review_decisions(
        candidates=[candidate],
        reviews=[first, second],
        policy=ReviewVotingPolicy.MAJORITY,
    )

    assert result.gate_reviews[0].decision == ReviewDecisionValue.NEEDS_HUMAN
    assert "missing final relation info" in result.gate_reviews[0].reasons[0]


def test_positive_aggregate_without_normalized_implication_support_queues() -> None:
    candidate = _candidate()

    result = aggregate_review_decisions(
        candidates=[candidate],
        reviews=[
            _review(reviewer_model="reviewer-a"),
            _review(reviewer_model="reviewer-b").model_copy(
                update={"normalized_implication_supported": None}
            ),
        ],
        policy=ReviewVotingPolicy.MAJORITY,
    )

    assert result.gate_reviews[0].decision == ReviewDecisionValue.NEEDS_HUMAN
    assert "missing normalized implication support" in result.gate_reviews[0].reasons[0]
