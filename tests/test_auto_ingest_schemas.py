import pytest
from pydantic import ValidationError

from nesy_reasoning_mcp.auto_ingest import (
    DRY_RUN_TOOL_ALLOWLIST,
    WRITE_MODE_TOOL_ALLOWLIST,
    CandidateRelation,
    CandidateRelationBatch,
    EvidenceRecord,
    GateAction,
    GateResult,
    IngestionInput,
    IngestionReport,
    ReviewDecision,
    ReviewDecisionBatch,
    ReviewDecisionValue,
    ReviewQueueRecord,
    ReviewQueueStatus,
    ReviewVotingPolicy,
    ValidateCandidateRelationsInput,
)
from nesy_reasoning_mcp.auto_ingest.review_queue import queued_records_from_report
from nesy_reasoning_mcp.schemas import Diagnostic
from nesy_reasoning_mcp.tool_names import ASSERT_RELATIONS, LOAD_RELATIONS, REASON_OVER_RELATIONS


def _evidence() -> EvidenceRecord:
    return EvidenceRecord(
        url="https://example.com/docs",
        span="A explicitly enables B under condition C.",
        source_type="official_docs",
    )


def test_candidate_review_gate_and_report_serialize() -> None:
    candidate = CandidateRelation(
        id="candidate-1",
        source="A",
        target="B",
        relation_type="sufficient",
        confidence=0.86,
        evidence=[_evidence()],
        metadata={"claim_strength": "explicit"},
    )
    review = ReviewDecision(
        candidate_id=candidate.id,
        decision=ReviewDecisionValue.APPROVE,
        final_relation_type="sufficient",
        final_confidence=0.86,
        reasons=["Evidence explicitly supports the implication."],
    )
    gate = GateResult(
        candidate_id=candidate.id,
        action=GateAction.AUTO_WRITE,
        reasons=["Reviewer approved and confidence meets threshold."],
    )
    relation = candidate.to_relation_input()
    report = IngestionReport(
        run_id="run-1",
        candidates=[candidate],
        reviews=[review],
        gate_results=[gate],
        approved_relations=[relation],
        written_relation_ids=["rel-1"],
    )

    dumped = report.model_dump(mode="json")

    assert dumped["mode"] == "dry_run"
    assert dumped["candidates"][0]["source"] == "A"
    assert dumped["approved_relations"][0]["provenance"]["candidate_id"] == "candidate-1"
    assert dumped["approved_relations"][0]["provenance"]["evidence"][0]["url"].startswith(
        "https://"
    )
    assert dumped["written_relation_ids"] == ["rel-1"]


def test_review_queue_record_round_trips_candidate_review_and_gate() -> None:
    candidate = CandidateRelation(
        id="candidate-1",
        source="A",
        target="B",
        relation_type="sufficient",
        evidence=[_evidence()],
    )
    review = ReviewDecision(
        candidate_id=candidate.id,
        decision=ReviewDecisionValue.APPROVE,
        final_relation_type="sufficient",
        final_confidence=0.9,
        reasons=["Evidence supports the relation."],
    )
    gate = GateResult(candidate_id=candidate.id, action=GateAction.QUEUE)

    record = ReviewQueueRecord(
        id="queue-1",
        run_id="run-1",
        candidate=candidate,
        review=review,
        gate_result=gate,
        run_metadata={"task": "demo"},
    )
    reloaded = ReviewQueueRecord.model_validate(record.model_dump(mode="json"))

    assert reloaded.status == ReviewQueueStatus.PENDING
    assert reloaded.created_at == reloaded.updated_at
    assert reloaded.candidate.evidence[0].url == "https://example.com/docs"
    assert reloaded.review is not None
    assert reloaded.review.reasons == ["Evidence supports the relation."]
    assert reloaded.gate_result.action == GateAction.QUEUE


def test_review_queue_record_rejects_mismatched_candidate_ids() -> None:
    candidate = CandidateRelation(
        id="candidate-1",
        source="A",
        target="B",
        relation_type="sufficient",
        evidence=[_evidence()],
    )

    with pytest.raises(ValidationError):
        ReviewQueueRecord(
            run_id="run-1",
            candidate=candidate,
            gate_result=GateResult(candidate_id="other", action=GateAction.QUEUE),
        )


def test_review_queue_record_requires_queue_gate_action() -> None:
    candidate = CandidateRelation(
        id="candidate-1",
        source="A",
        target="B",
        relation_type="sufficient",
        evidence=[_evidence()],
    )

    with pytest.raises(ValidationError):
        ReviewQueueRecord(
            run_id="run-1",
            candidate=candidate,
            gate_result=GateResult(candidate_id=candidate.id, action=GateAction.AUTO_WRITE),
        )


def test_queued_records_keep_candidate_diagnostics_without_repeating_all_run_diagnostics() -> None:
    candidate = CandidateRelation(
        id="candidate-1",
        source="A",
        target="B",
        relation_type="sufficient",
        evidence=[_evidence()],
    )
    report = IngestionReport(
        run_id="run-1",
        candidates=[candidate],
        reviews=[],
        gate_results=[GateResult(candidate_id=candidate.id, action=GateAction.QUEUE)],
        diagnostics=[
            Diagnostic(
                level="warning",
                code="CANDIDATE_LOW_CONFIDENCE",
                message="candidate-specific",
                related_ids=[candidate.id],
            ),
            Diagnostic(
                level="warning",
                code="RUN_LEVEL",
                message="run-level",
            ),
        ],
    )

    records = queued_records_from_report(report, propositions=[], context_metadata={})

    assert records[0].run_metadata["diagnostic_count"] == 2
    assert [diagnostic.code for diagnostic in records[0].diagnostics] == [
        "CANDIDATE_LOW_CONFIDENCE"
    ]


def test_queued_records_keep_first_duplicate_aggregate_review() -> None:
    candidate = CandidateRelation(
        id="candidate-1",
        source="A",
        target="B",
        relation_type="sufficient",
        evidence=[_evidence()],
    )
    first_review = ReviewDecision(
        candidate_id=candidate.id,
        decision=ReviewDecisionValue.NEEDS_HUMAN,
        reasons=["first aggregate review"],
    )
    second_review = ReviewDecision(
        candidate_id=candidate.id,
        decision=ReviewDecisionValue.NEEDS_HUMAN,
        reasons=["second aggregate review"],
    )
    report = IngestionReport(
        run_id="run-1",
        candidates=[candidate],
        gate_results=[GateResult(candidate_id=candidate.id, action=GateAction.QUEUE)],
        metadata={
            "review_aggregation": {
                "aggregate_reviews": [
                    first_review.model_dump(mode="json"),
                    second_review.model_dump(mode="json"),
                ]
            }
        },
    )

    records = queued_records_from_report(report, propositions=[], context_metadata={})

    assert records[0].review is not None
    assert records[0].review.reasons == ["first aggregate review"]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("url", " "),
        ("span", ""),
    ],
)
def test_evidence_requires_non_empty_url_and_span(field: str, value: str) -> None:
    data = {
        "url": "https://example.com/docs",
        "span": "A enables B.",
    }
    data[field] = value

    with pytest.raises(ValidationError):
        EvidenceRecord.model_validate(data)


def test_candidate_rejects_confidence_out_of_range() -> None:
    with pytest.raises(ValidationError):
        CandidateRelation(
            source="A",
            target="B",
            relation_type="sufficient",
            confidence=1.1,
            evidence=[_evidence()],
        )


def test_candidate_rejects_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        CandidateRelation.model_validate(
            {
                "source": "A",
                "target": "B",
                "relation_type": "sufficient",
                "evidence": [{"url": "https://example.com/docs", "span": "A enables B."}],
                "unexpected": True,
            }
        )


@pytest.mark.parametrize(
    "decision",
    [ReviewDecisionValue.APPROVE, ReviewDecisionValue.DOWNGRADE],
)
def test_positive_review_decisions_require_final_relation_info(
    decision: ReviewDecisionValue,
) -> None:
    with pytest.raises(ValidationError):
        ReviewDecision(candidate_id="candidate-1", decision=decision)


def test_reject_review_does_not_require_final_relation_info() -> None:
    review = ReviewDecision(
        candidate_id="candidate-1",
        decision=ReviewDecisionValue.REJECT,
        reasons=["Evidence only supports correlation."],
    )

    assert review.final_relation_type is None
    assert review.final_confidence is None


def test_review_decision_normalized_implication_support_is_optional_for_compatibility() -> None:
    legacy = ReviewDecision(
        candidate_id="candidate-1",
        decision=ReviewDecisionValue.APPROVE,
        final_relation_type="sufficient",
        final_confidence=0.9,
    )
    confirmed = legacy.model_copy(update={"normalized_implication_supported": True})

    assert legacy.normalized_implication_supported is None
    assert confirmed.model_dump(mode="json")["normalized_implication_supported"] is True


def test_tool_allowlists_keep_write_tools_out_of_dry_run() -> None:
    assert REASON_OVER_RELATIONS in DRY_RUN_TOOL_ALLOWLIST
    assert ASSERT_RELATIONS not in DRY_RUN_TOOL_ALLOWLIST
    assert LOAD_RELATIONS not in DRY_RUN_TOOL_ALLOWLIST
    assert ASSERT_RELATIONS in WRITE_MODE_TOOL_ALLOWLIST
    assert LOAD_RELATIONS not in WRITE_MODE_TOOL_ALLOWLIST


def test_ingestion_input_and_agent_batches_are_strict() -> None:
    candidate = CandidateRelation(
        source="A",
        target="B",
        relation_type="sufficient",
        evidence=[_evidence()],
    )
    review = ReviewDecision(
        candidate_id=candidate.id,
        decision=ReviewDecisionValue.APPROVE,
        final_relation_type="sufficient",
        final_confidence=0.9,
    )

    ingestion_input = IngestionInput(
        evidence=[_evidence()],
        urls=[" https://example.com/more "],
        task=" Extract relations ",
    )

    assert ingestion_input.urls == ["https://example.com/more"]
    assert CandidateRelationBatch(candidates=[candidate]).candidates[0].source == "A"
    assert ReviewDecisionBatch(reviews=[review]).reviews[0].candidate_id == candidate.id

    with pytest.raises(ValidationError):
        IngestionInput.model_validate({"evidence": [], "unknown": True})


def test_validate_candidate_relations_input_accepts_voting_policy() -> None:
    candidate = CandidateRelation(
        id="candidate-1",
        source="A",
        target="B",
        relation_type="sufficient",
        evidence=[_evidence()],
    )
    payload = ValidateCandidateRelationsInput(
        candidates=[candidate],
        voting_policy="majority",
        high_priority_reviewer_models=[" gpt-4.1 ", "gpt-4.1"],
    )

    assert payload.voting_policy == ReviewVotingPolicy.MAJORITY
    assert payload.high_priority_reviewer_models == ["gpt-4.1"]
