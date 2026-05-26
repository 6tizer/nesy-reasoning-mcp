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
from nesy_reasoning_mcp.config import NesyConfig, StorageConfig
from nesy_reasoning_mcp.schemas import Diagnostic, PropositionRecord, RelationInput
from nesy_reasoning_mcp.store import JsonRelationStore, RelationStore, SqliteRelationStore


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
                "normalized_implication_supported": True,
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


def _store_for_backend(tmp_path: Any, backend: str) -> Any:
    if backend == "memory":
        return RelationStore()
    if backend == "json":
        return JsonRelationStore(
            NesyConfig(
                storage=StorageConfig(
                    backend="json",
                    json_path=str(tmp_path / "relations.json"),
                )
            )
        )
    return SqliteRelationStore(
        NesyConfig(
            storage=StorageConfig(
                backend="sqlite",
                sqlite_path=str(tmp_path / "nesy.db"),
            )
        )
    )


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


async def test_dry_run_does_not_run_canonicalizer_with_existing_propositions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate = _candidate(target="release is auto-deployed")
    store = RelationStore()
    store.import_records(
        [],
        [],
        propositions=[PropositionRecord(id="auto_deploy", label="auto-deploy")],
        mode="append",
        store_id="default",
    )

    def fake_agent(**kwargs: Any) -> SimpleNamespace:
        return SimpleNamespace(output_type=kwargs["output_type"])

    async def fake_run_agent(agent: Any, prompt: str, *, tracing_disabled: bool = False) -> Any:
        if agent.output_type is openai_agents.PropositionCanonicalizationBatch:
            raise AssertionError("canonicalizer should not run in ordinary dry-run")
        if agent.output_type is openai_agents.CandidateRelationBatch:
            return {"candidates": [candidate.model_dump(mode="json")]}
        return {"reviews": [_review(candidate).model_dump(mode="json")]}

    monkeypatch.setattr(openai_agents, "_build_agent", fake_agent)
    monkeypatch.setattr(openai_agents, "_run_agent", fake_run_agent)

    report = await run_openai_agents_ingestion(
        IngestionInput(evidence=[_evidence()]),
        store=store,
        env={"OPENAI_API_KEY": "test"},
    )

    assert report.mode == "dry_run"
    assert report.written_relation_ids == []
    assert store.list_relations() == []


async def test_dry_run_canonicalize_preview_runs_without_writing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate = _candidate(target="release is auto-deployed")
    store = RelationStore()
    store.import_records(
        [],
        [],
        propositions=[PropositionRecord(id="auto_deploy", label="auto-deploy")],
        mode="append",
        store_id="default",
    )
    calls: list[str] = []

    def fake_agent(**kwargs: Any) -> SimpleNamespace:
        return SimpleNamespace(output_type=kwargs["output_type"])

    async def fake_run_agent(agent: Any, prompt: str, *, tracing_disabled: bool = False) -> Any:
        if agent.output_type is openai_agents.CandidateRelationBatch:
            calls.append("extractor")
            return {"candidates": [candidate.model_dump(mode="json")]}
        if agent.output_type is openai_agents.PropositionCanonicalizationBatch:
            calls.append("canonicalizer")
            return {
                "propositions": [
                    {
                        "endpoint_refs": [f"{candidate.id}:source"],
                        "canonical_label": candidate.source,
                    },
                    {
                        "endpoint_refs": [f"{candidate.id}:target"],
                        "canonical_label": "auto-deploy",
                        "canonical_id": "auto_deploy",
                        "aliases": ["release is auto-deployed"],
                    },
                ]
            }
        calls.append("reviewer")
        return {"reviews": [_review(candidate).model_dump(mode="json")]}

    monkeypatch.setattr(openai_agents, "_build_agent", fake_agent)
    monkeypatch.setattr(openai_agents, "_run_agent", fake_run_agent)

    report = await run_openai_agents_ingestion(
        IngestionInput(evidence=[_evidence()]),
        store=store,
        env={"OPENAI_API_KEY": "test"},
        canonicalize_preview=True,
    )

    assert calls == ["extractor", "canonicalizer", "reviewer"]
    assert report.mode == "dry_run"
    assert report.candidates[0].target_id == "auto_deploy"
    assert report.metadata["proposition_canonicalization"]["mode"] == "llm_assisted"
    assert report.metadata["proposition_canonicalization"]["llm_canonicalizer"] == {
        "status": "executed",
        "reason": "likely_overlap",
    }
    assert report.written_relation_ids == []
    assert store.list_relations() == []


async def test_auto_write_skips_canonicalizer_without_likely_overlap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate = _candidate(source="schema validates", target="cache warms")
    store = RelationStore()
    store.import_records(
        [],
        [],
        propositions=[PropositionRecord(id="auto_deploy", label="auto-deploy")],
        mode="append",
        store_id="default",
    )
    progress_events: list[dict[str, Any]] = []

    def fake_agent(**kwargs: Any) -> SimpleNamespace:
        return SimpleNamespace(output_type=kwargs["output_type"])

    async def fake_run_agent(agent: Any, prompt: str, *, tracing_disabled: bool = False) -> Any:
        if agent.output_type is openai_agents.CandidateRelationBatch:
            return {"candidates": [candidate.model_dump(mode="json")]}
        if agent.output_type is openai_agents.PropositionCanonicalizationBatch:
            raise AssertionError("canonicalizer should be skipped when there is no overlap")
        return {"reviews": [_review(candidate).model_dump(mode="json")]}

    monkeypatch.setattr(openai_agents, "_build_agent", fake_agent)
    monkeypatch.setattr(openai_agents, "_run_agent", fake_run_agent)

    report = await run_openai_agents_ingestion(
        IngestionInput(evidence=[_evidence()]),
        store=store,
        env={"OPENAI_API_KEY": "test"},
        auto_write=True,
        progress_callback=progress_events.append,
    )

    assert report.mode == "write"
    assert report.metadata["proposition_canonicalization"]["mode"] == "deterministic"
    assert report.metadata["proposition_canonicalization"]["llm_canonicalizer"] == {
        "status": "skipped",
        "reason": "no_likely_overlap",
    }
    assert any(
        item["stage"] == "canonicalizer" and item["status"] == "skipped"
        for item in report.metadata["runtime_trace"]
    )
    assert any(
        item["stage"] == "canonicalizer" and item["event"] == "skipped" for item in progress_events
    )
    assert len(store.list_relations()) == 1


async def test_auto_write_exact_known_alias_canonicalizes_without_llm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate = _candidate(source="CI passes", target="release is auto-deployed")
    store = RelationStore()
    store.import_records(
        [],
        [],
        propositions=[
            PropositionRecord(
                id="auto_deploy",
                label="auto-deploy",
                aliases=["release is auto-deployed"],
            )
        ],
        mode="append",
        store_id="default",
    )

    def fake_agent(**kwargs: Any) -> SimpleNamespace:
        return SimpleNamespace(output_type=kwargs["output_type"])

    async def fake_run_agent(agent: Any, prompt: str, *, tracing_disabled: bool = False) -> Any:
        if agent.output_type is openai_agents.CandidateRelationBatch:
            return {"candidates": [candidate.model_dump(mode="json")]}
        if agent.output_type is openai_agents.PropositionCanonicalizationBatch:
            raise AssertionError("canonicalizer should be skipped for exact alias match")
        return {"reviews": [_review(candidate).model_dump(mode="json")]}

    monkeypatch.setattr(openai_agents, "_build_agent", fake_agent)
    monkeypatch.setattr(openai_agents, "_run_agent", fake_run_agent)

    report = await run_openai_agents_ingestion(
        IngestionInput(evidence=[_evidence()]),
        store=store,
        env={"OPENAI_API_KEY": "test"},
        auto_write=True,
    )

    assert report.candidates[0].target_id == "auto_deploy"
    assert report.metadata["proposition_canonicalization"]["mode"] == "deterministic"
    assert report.metadata["proposition_canonicalization"]["llm_canonicalizer"] == {
        "status": "skipped",
        "reason": "exact_match_only",
    }


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
    assert store.list_propositions()


@pytest.mark.parametrize("backend", ["memory", "json", "sqlite"])
async def test_write_approved_relations_deduplicates_existing_relations(
    tmp_path: Any,
    backend: str,
) -> None:
    store = _store_for_backend(tmp_path, backend)
    relation = RelationInput(source="A", target="B", relation_type="sufficient")

    first_ids, first_diagnostics, first_result = await writer.write_approved_relations(
        relations=[relation],
        store=store,
    )
    second_ids, second_diagnostics, second_result = await writer.write_approved_relations(
        relations=[relation],
        store=store,
    )

    assert first_diagnostics == []
    assert second_diagnostics == []
    assert first_ids == second_ids
    assert len(store.list_relations()) == 1
    assert first_result["deduplicated_count"] == 0
    assert second_result["deduplicated_count"] == 1
    assert second_result["deduplicated_relation_ids"] == first_ids


async def test_write_approved_relations_deduplicates_batch_duplicates() -> None:
    store = RelationStore()
    relation = RelationInput(source="A", target="B", relation_type="sufficient")

    relation_ids, diagnostics, result = await writer.write_approved_relations(
        relations=[relation, relation],
        store=store,
    )

    assert diagnostics == []
    assert len(store.list_relations()) == 1
    assert relation_ids == [store.list_relations()[0].id, store.list_relations()[0].id]
    assert result["deduplicated_count"] == 1


async def test_write_approved_relations_deduplicates_legacy_label_match() -> None:
    store = RelationStore()
    existing, _updated = store.assert_relations(
        [RelationInput(source="A", target="B", relation_type="sufficient")],
        mode="append",
    )

    relation_ids, diagnostics, result = await writer.write_approved_relations(
        relations=[
            RelationInput(
                source="A",
                source_id="prop_a",
                target="B",
                target_id="prop_b",
                relation_type="sufficient",
            )
        ],
        store=store,
    )

    assert diagnostics == []
    assert relation_ids == [existing[0].id]
    assert len(store.list_relations()) == 1
    assert result["deduplicated_count"] == 1


async def test_auto_write_canonicalizes_semantic_duplicate_to_existing_proposition(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = RelationStore()
    current: dict[str, CandidateRelation] = {}

    def fake_agent(**kwargs: Any) -> SimpleNamespace:
        return SimpleNamespace(output_type=kwargs["output_type"])

    async def fake_run_agent(agent: Any, prompt: str, *, tracing_disabled: bool = False) -> Any:
        candidate = current["candidate"]
        if agent.output_type is openai_agents.CandidateRelationBatch:
            return {"candidates": [candidate.model_dump(mode="json")]}
        if agent.output_type is openai_agents.PropositionCanonicalizationBatch:
            propositions = {item.label: item for item in store.list_propositions()}
            target_label = (
                "auto-deploy"
                if candidate.target == "release is auto-deployed"
                else "eligible for auto-deploy"
            )
            target_id = propositions["auto-deploy"].id if target_label == "auto-deploy" else None
            target_aliases = ["release is auto-deployed"] if target_label == "auto-deploy" else []
            return {
                "propositions": [
                    {
                        "endpoint_refs": [f"{candidate.id}:source"],
                        "canonical_label": "CI passes",
                        "canonical_id": propositions["CI passes"].id,
                    },
                    {
                        "endpoint_refs": [f"{candidate.id}:target"],
                        "canonical_label": target_label,
                        "canonical_id": target_id,
                        "aliases": target_aliases,
                    },
                ]
            }
        return {
            "reviews": [
                _review(candidate).model_dump(mode="json", exclude_none=True),
            ]
        }

    monkeypatch.setattr(openai_agents, "_build_agent", fake_agent)
    monkeypatch.setattr(openai_agents, "_run_agent", fake_run_agent)

    current["candidate"] = _candidate(
        candidate_id="candidate-1",
        source="CI passes",
        target="auto-deploy",
    )
    first_report = await run_openai_agents_ingestion(
        IngestionInput(evidence=[_evidence()]),
        store=store,
        env={"OPENAI_API_KEY": "test"},
        auto_write=True,
    )

    current["candidate"] = _candidate(
        candidate_id="candidate-2",
        source="CI passes",
        target="release is auto-deployed",
    )
    second_report = await run_openai_agents_ingestion(
        IngestionInput(evidence=[_evidence()]),
        store=store,
        env={"OPENAI_API_KEY": "test"},
        auto_write=True,
    )

    current["candidate"] = _candidate(
        candidate_id="candidate-3",
        source="CI passes",
        target="eligible for auto-deploy",
    )
    third_report = await run_openai_agents_ingestion(
        IngestionInput(evidence=[_evidence()]),
        store=store,
        env={"OPENAI_API_KEY": "test"},
        auto_write=True,
    )

    target_proposition = next(
        proposition
        for proposition in store.list_propositions()
        if proposition.label == "auto-deploy"
    )
    assert second_report.written_relation_ids == first_report.written_relation_ids
    assert second_report.metadata["write_result"]["deduplicated_count"] == 1
    assert "release is auto-deployed" in target_proposition.aliases
    assert len(store.list_relations()) == 2
    assert third_report.written_relation_ids != first_report.written_relation_ids
    assert {relation.target for relation in store.list_relations()} == {
        "auto-deploy",
        "eligible for auto-deploy",
    }


async def test_auto_write_canonicalizer_error_does_not_write(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = RelationStore()
    store.import_records(
        [],
        [],
        propositions=[PropositionRecord(id="auto_deploy", label="auto-deploy")],
        mode="append",
        store_id="default",
    )
    candidate = _candidate(source="CI passes", target="release is auto-deployed")

    def fake_agent(**kwargs: Any) -> SimpleNamespace:
        return SimpleNamespace(output_type=kwargs["output_type"])

    async def fake_run_agent(agent: Any, prompt: str, *, tracing_disabled: bool = False) -> Any:
        if agent.output_type is openai_agents.CandidateRelationBatch:
            return {"candidates": [candidate.model_dump(mode="json")]}
        if agent.output_type is openai_agents.PropositionCanonicalizationBatch:
            raise RuntimeError("canonicalizer failed")
        return {"reviews": [_review(candidate).model_dump(mode="json")]}

    monkeypatch.setattr(openai_agents, "_build_agent", fake_agent)
    monkeypatch.setattr(openai_agents, "_run_agent", fake_run_agent)

    report = await run_openai_agents_ingestion(
        IngestionInput(evidence=[_evidence()]),
        store=store,
        env={"OPENAI_API_KEY": "test"},
        auto_write=True,
    )

    assert report.diagnostics[0].code == "LLM_RUNTIME_ERROR"
    assert report.gate_results == []
    assert report.written_relation_ids == []
    assert store.list_relations() == []


async def test_auto_write_canonicalizer_import_conflict_does_not_write(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = RelationStore()
    store.import_records(
        [],
        [],
        propositions=[PropositionRecord(id="known", label="known proposition")],
        mode="append",
        store_id="default",
    )
    candidate = _candidate(source="source proposition", target="target proposition")

    def fake_agent(**kwargs: Any) -> SimpleNamespace:
        return SimpleNamespace(output_type=kwargs["output_type"])

    async def fake_run_agent(agent: Any, prompt: str, *, tracing_disabled: bool = False) -> Any:
        if agent.output_type is openai_agents.CandidateRelationBatch:
            return {"candidates": [candidate.model_dump(mode="json")]}
        if agent.output_type is openai_agents.PropositionCanonicalizationBatch:
            return {
                "propositions": [
                    {
                        "endpoint_refs": [f"{candidate.id}:source"],
                        "canonical_label": "source proposition",
                        "aliases": ["shared alias"],
                    },
                    {
                        "endpoint_refs": [f"{candidate.id}:target"],
                        "canonical_label": "target proposition",
                        "aliases": ["shared alias"],
                    },
                ]
            }
        return {"reviews": [_review(candidate).model_dump(mode="json")]}

    monkeypatch.setattr(openai_agents, "_build_agent", fake_agent)
    monkeypatch.setattr(openai_agents, "_run_agent", fake_run_agent)

    report = await run_openai_agents_ingestion(
        IngestionInput(evidence=[_evidence()]),
        store=store,
        env={"OPENAI_API_KEY": "test"},
        auto_write=True,
    )

    assert report.diagnostics[0].code == "PROPOSITION_CANONICALIZATION_IMPORT_INVALID"
    assert report.gate_results == []
    assert report.written_relation_ids == []
    assert store.list_relations() == []


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


async def test_auto_write_queues_voting_disagreement_with_audit_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate = _candidate()

    def fake_agent(**kwargs: Any) -> SimpleNamespace:
        return SimpleNamespace(output_type=kwargs["output_type"], model=kwargs["model"])

    async def fake_run_agent(agent: Any, prompt: str, *, tracing_disabled: bool = False) -> Any:
        if agent.output_type is openai_agents.CandidateRelationBatch:
            return {"candidates": [candidate.model_dump(mode="json")]}
        decision = (
            ReviewDecisionValue.APPROVE
            if agent.model == "reviewer-a"
            else ReviewDecisionValue.NEEDS_HUMAN
        )
        review = _review(candidate, decision=decision)
        return {"reviews": [review.model_dump(mode="json", exclude_none=True)]}

    monkeypatch.setattr(openai_agents, "_build_agent", fake_agent)
    monkeypatch.setattr(openai_agents, "_run_agent", fake_run_agent)
    store = RelationStore()

    report = await run_openai_agents_ingestion(
        IngestionInput(evidence=[_evidence()]),
        store=store,
        model="extractor-model",
        reviewer_models=["reviewer-a", "reviewer-b"],
        voting_policy=openai_agents.ReviewVotingPolicy.MAJORITY,
        env={"OPENAI_API_KEY": "test"},
        auto_write=True,
    )

    queued = store.list_review_queue()
    assert report.gate_results[0].action == "queue"
    assert len(queued) == 1
    assert queued[0].review is not None
    assert queued[0].review.reviewer_model == "aggregate:majority"
    assert queued[0].run_metadata["metadata"]["review_aggregation"]["policy"] == "majority"
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
