from types import SimpleNamespace
from typing import Any

import pytest

from nesy_reasoning_mcp.auto_ingest import (
    CandidateRelation,
    EvidenceRecord,
    IngestionInput,
    ReviewDecision,
    ReviewDecisionValue,
    openai_agents,
    writer,
)
from nesy_reasoning_mcp.auto_ingest.openai_agents import (
    OpenAIAgentsDryRunError,
    run_openai_agents_ingestion,
)
from nesy_reasoning_mcp.schemas import Diagnostic
from nesy_reasoning_mcp.store import RelationStore


def _evidence() -> EvidenceRecord:
    return EvidenceRecord(url="https://example.com/source", span="A explicitly enables B.")


def _candidate(
    *,
    candidate_id: str = "candidate-1",
    source: str = "A",
    target: str = "B",
    confidence: float = 0.92,
) -> CandidateRelation:
    return CandidateRelation(
        id=candidate_id,
        source=source,
        target=target,
        relation_type="sufficient",
        confidence=confidence,
        evidence=[_evidence()],
    )


def _review(
    candidate: CandidateRelation,
    *,
    decision: ReviewDecisionValue = ReviewDecisionValue.APPROVE,
    confidence: float = 0.92,
) -> ReviewDecision:
    kwargs: dict[str, Any] = {
        "candidate_id": candidate.id,
        "decision": decision,
        "reasons": ["Evidence directly supports the relation."],
        "risk_flags": ["needs periodical review"],
        "reviewer_model": "gpt-test",
    }
    if decision in {ReviewDecisionValue.APPROVE, ReviewDecisionValue.DOWNGRADE}:
        kwargs.update(
            {
                "final_relation_type": "sufficient",
                "final_confidence": confidence,
            }
        )
    return ReviewDecision(**kwargs)


def _mock_agents(
    monkeypatch: pytest.MonkeyPatch, *pairs: tuple[CandidateRelation, ReviewDecision]
) -> None:
    def fake_agent(**kwargs: Any) -> SimpleNamespace:
        return SimpleNamespace(output_type=kwargs["output_type"])

    async def fake_run_agent(agent: Any, prompt: str, *, tracing_disabled: bool = False) -> Any:
        assert tracing_disabled is False
        if agent.output_type is openai_agents.CandidateRelationBatch:
            return {"candidates": [candidate.model_dump(mode="json") for candidate, _ in pairs]}
        return {"reviews": [review.model_dump(mode="json") for _, review in pairs]}

    monkeypatch.setattr(openai_agents, "_build_agent", fake_agent)
    monkeypatch.setattr(openai_agents, "_run_agent", fake_run_agent)


async def test_default_dry_run_does_not_write(monkeypatch: pytest.MonkeyPatch) -> None:
    candidate = _candidate()
    _mock_agents(monkeypatch, (candidate, _review(candidate)))
    store = RelationStore()

    report = await run_openai_agents_ingestion(
        IngestionInput(evidence=[_evidence()]),
        store=store,
        env={"OPENAI_API_KEY": "test"},
    )

    assert report.mode == "dry_run"
    assert report.written_relation_ids == []
    assert report.approved_relations[0].source == "A"
    assert store.list_review_queue() == []
    assert store.list_relations() == []


async def test_auto_write_persists_gate_approved_relations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate = _candidate()
    _mock_agents(monkeypatch, (candidate, _review(candidate)))
    store = RelationStore()

    report = await run_openai_agents_ingestion(
        IngestionInput(evidence=[_evidence()]),
        store=store,
        env={"OPENAI_API_KEY": "test"},
        auto_write=True,
        min_write_confidence=0.85,
    )

    stored = store.list_relations()
    assert report.mode == "write"
    assert len(report.written_relation_ids) == 1
    assert report.written_relation_ids == [stored[0].id]
    assert report.gate_results[0].reasons == ["write approved; persistent assertion may proceed"]
    assert stored[0].source == "A"
    assert stored[0].provenance["candidate_id"] == candidate.id
    assert stored[0].provenance["review"]["reviewer_model"] == "gpt-test"


async def test_auto_write_queues_low_confidence_without_writing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate = _candidate(confidence=0.9)
    _mock_agents(monkeypatch, (candidate, _review(candidate, confidence=0.7)))
    store = RelationStore()

    report = await run_openai_agents_ingestion(
        IngestionInput(evidence=[_evidence()]),
        store=store,
        env={"OPENAI_API_KEY": "test"},
        auto_write=True,
        min_write_confidence=0.85,
    )

    assert report.gate_results[0].action == "queue"
    assert "below write threshold" in report.gate_results[0].reasons[0]
    assert len(store.list_review_queue()) == 1
    assert store.list_review_queue()[0].candidate.evidence[0].url == "https://example.com/source"
    assert report.metadata["review_queue_record_ids"] == [store.list_review_queue()[0].id]
    assert report.approved_relations == []
    assert report.written_relation_ids == []
    assert store.list_relations() == []


async def test_ingestion_rejects_invalid_write_threshold() -> None:
    with pytest.raises(OpenAIAgentsDryRunError, match="min_write_confidence"):
        await run_openai_agents_ingestion(
            IngestionInput(evidence=[_evidence()]),
            store=RelationStore(),
            env={"OPENAI_API_KEY": "test"},
            auto_write=True,
            min_write_confidence=1.5,
        )


@pytest.mark.parametrize(
    "decision",
    [
        ReviewDecisionValue.REJECT,
        ReviewDecisionValue.DOWNGRADE,
        ReviewDecisionValue.NEEDS_HUMAN,
    ],
)
async def test_auto_write_blocks_non_approved_reviews(
    monkeypatch: pytest.MonkeyPatch,
    decision: ReviewDecisionValue,
) -> None:
    candidate = _candidate()
    _mock_agents(monkeypatch, (candidate, _review(candidate, decision=decision)))
    store = RelationStore()

    report = await run_openai_agents_ingestion(
        IngestionInput(evidence=[_evidence()]),
        store=store,
        env={"OPENAI_API_KEY": "test"},
        auto_write=True,
    )

    assert report.written_relation_ids == []
    assert store.list_relations() == []


async def test_auto_write_queues_hard_contradictions_without_writing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = _candidate(candidate_id="candidate-1", source="A", target="B")
    second = _candidate(candidate_id="candidate-2", source="A", target="not B")
    _mock_agents(monkeypatch, (first, _review(first)), (second, _review(second)))
    store = RelationStore()

    report = await run_openai_agents_ingestion(
        IngestionInput(evidence=[_evidence()]),
        store=store,
        env={"OPENAI_API_KEY": "test"},
        auto_write=True,
    )

    assert {item.action for item in report.gate_results} == {"queue"}
    assert len(store.list_review_queue()) == 2
    assert report.approved_relations == []
    assert report.written_relation_ids == []
    assert store.list_relations() == []


async def test_auto_write_reports_assert_failure_without_fake_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate = _candidate()
    _mock_agents(monkeypatch, (candidate, _review(candidate)))
    store = RelationStore()

    async def fake_write_approved_relations(
        *args: Any, **kwargs: Any
    ) -> tuple[
        list[str],
        list[Diagnostic],
        dict[str, Any],
    ]:
        return (
            [],
            [
                Diagnostic(
                    level="error",
                    code="ASSERT_FAILED",
                    message="assert_relations failed",
                )
            ],
            {"status": "error"},
        )

    monkeypatch.setattr(writer, "write_approved_relations", fake_write_approved_relations)
    monkeypatch.setattr(
        openai_agents,
        "write_approved_relations",
        fake_write_approved_relations,
    )

    report = await run_openai_agents_ingestion(
        IngestionInput(evidence=[_evidence()]),
        store=store,
        env={"OPENAI_API_KEY": "test"},
        auto_write=True,
    )

    assert report.approved_relations
    assert report.written_relation_ids == []
    assert report.diagnostics[0].code == "ASSERT_FAILED"
    assert store.list_relations() == []
