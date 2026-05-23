import argparse
import json
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from nesy_reasoning_mcp.auto_ingest import (
    CandidateRelation,
    EvidenceRecord,
    IngestionInput,
    IngestionReport,
    ReviewDecision,
    ReviewDecisionValue,
    openai_agents,
)
from nesy_reasoning_mcp.auto_ingest import cli as ingest_cli
from nesy_reasoning_mcp.auto_ingest.fetcher import fetch_url_evidence
from nesy_reasoning_mcp.auto_ingest.gate import run_dry_run_gate
from nesy_reasoning_mcp.auto_ingest.openai_agents import (
    OpenAIAgentsDryRunError,
    run_openai_agents_dry_run,
)
from nesy_reasoning_mcp.store import RelationStore


def _evidence(span: str = "A cannot run unless B is configured.") -> EvidenceRecord:
    return EvidenceRecord(url="https://example.com/source", span=span)


def _candidate(
    *,
    candidate_id: str = "candidate-1",
    source: str = "A",
    target: str = "B",
) -> CandidateRelation:
    return CandidateRelation(
        id=candidate_id,
        source=source,
        target=target,
        relation_type="sufficient",
        confidence=0.9,
        evidence=[_evidence()],
    )


def _approval(candidate: CandidateRelation) -> ReviewDecision:
    return ReviewDecision(
        candidate_id=candidate.id,
        decision=ReviewDecisionValue.APPROVE,
        final_relation_type="sufficient",
        final_confidence=0.9,
        reasons=["Evidence is explicit."],
    )


def test_fetch_url_evidence_allows_only_http_https(monkeypatch: pytest.MonkeyPatch) -> None:
    class Response:
        headers = {"content-type": "text/plain; charset=utf-8"}

        def __enter__(self) -> "Response":
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def read(self, size: int) -> bytes:
            assert size == 6
            return b"abcdef"

    def fake_urlopen(req: Any, timeout: float) -> Response:
        assert req.full_url == "https://example.com/source"
        assert timeout == 3
        return Response()

    monkeypatch.setattr("nesy_reasoning_mcp.auto_ingest.fetcher.request.urlopen", fake_urlopen)

    record = fetch_url_evidence("https://example.com/source", timeout_seconds=3, max_bytes=5)

    assert record.span == "abcde"
    assert record.metadata["truncated"] is True
    with pytest.raises(ValueError):
        fetch_url_evidence("file:///tmp/source.txt")
    with pytest.raises(ValueError):
        fetch_url_evidence("http://localhost/source")
    with pytest.raises(ValueError):
        fetch_url_evidence("http://127.0.0.1/source")


async def test_openai_agents_dry_run_requires_api_key_without_mock_runner() -> None:
    with pytest.raises(OpenAIAgentsDryRunError):
        await run_openai_agents_dry_run(
            IngestionInput(evidence=[_evidence()]),
            store=RelationStore(),
            env={},
        )


async def test_openai_agents_dry_run_maps_runner_outputs_to_report(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate = _candidate()
    review = _approval(candidate)

    def fake_agent(**kwargs: Any) -> SimpleNamespace:
        return SimpleNamespace(output_type=kwargs["output_type"])

    async def fake_run_agent(agent: Any, prompt: str) -> Any:
        assert "EvidenceRecord" not in prompt
        if agent.output_type is openai_agents.CandidateRelationBatch:
            return {"candidates": [candidate.model_dump(mode="json")]}
        return {"reviews": [review.model_dump(mode="json")]}

    monkeypatch.setattr(openai_agents, "_build_agent", fake_agent)

    report = await run_openai_agents_dry_run(
        IngestionInput(evidence=[_evidence()], task="extract dependencies"),
        store=RelationStore(),
        env={"OPENAI_API_KEY": "test"},
        run_agent=fake_run_agent,
    )

    assert report.mode == "dry_run"
    assert report.candidates == [candidate]
    assert report.reviews == [review]
    assert report.approved_relations[0].source == "A"
    assert report.gate_results[0].action == "auto_write"


async def test_dry_run_gate_never_writes_and_queues_hard_contradictions() -> None:
    store = RelationStore()
    first = _candidate(candidate_id="candidate-1", source="A", target="B")
    second = _candidate(candidate_id="candidate-2", source="A", target="not B")
    reviews = [_approval(first), _approval(second)]

    gate_results, approved_relations, diagnostics, reasoning = await run_dry_run_gate(
        candidates=[first, second],
        reviews=reviews,
        store=store,
    )

    assert {item.action for item in gate_results} == {"queue"}
    assert approved_relations == []
    assert diagnostics == []
    assert reasoning["result"]["has_contradictions"] is True
    assert store.list_relations() == []


async def test_dry_run_gate_approved_path_reports_without_writing() -> None:
    store = RelationStore()
    candidate = _candidate()

    gate_results, approved_relations, _, _ = await run_dry_run_gate(
        candidates=[candidate],
        reviews=[_approval(candidate)],
        store=store,
    )

    assert gate_results[0].action == "auto_write"
    assert approved_relations[0].source == "A"
    assert store.list_relations() == []


def test_cli_agent_dry_run_json_output_uses_same_runtime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_path = tmp_path / "input.json"
    input_path.write_text(
        json.dumps({"evidence": [_evidence().model_dump(mode="json")]}),
        encoding="utf-8",
    )

    async def fake_run(
        ingestion_input: IngestionInput,
        *,
        store: Any,
        model: str | None = None,
    ) -> IngestionReport:
        assert ingestion_input.evidence[0].url == "https://example.com/source"
        assert store.list_relations() == []
        assert model == "gpt-test"
        return IngestionReport(candidates=[_candidate()])

    monkeypatch.setattr(ingest_cli, "run_openai_agents_dry_run", fake_run)
    stdout = StringIO()
    stderr = StringIO()
    args = argparse.Namespace(
        input=str(input_path),
        url=[],
        task=None,
        question=None,
        model="gpt-test",
        format="json",
        output=None,
        timeout_seconds=1.0,
        max_url_bytes=1000,
    )

    exit_code = ingest_cli.run_agent_dry_run_cli(args, stdout=stdout, stderr=stderr)
    payload = json.loads(stdout.getvalue())

    assert exit_code == 0
    assert payload["mode"] == "dry_run"
    assert payload["candidates"][0]["source"] == "A"
    assert stderr.getvalue() == ""


def test_cli_agent_dry_run_invalid_input_returns_nonzero(tmp_path: Path) -> None:
    input_path = tmp_path / "input.json"
    input_path.write_text("[]", encoding="utf-8")
    args = argparse.Namespace(
        input=str(input_path),
        url=[],
        task=None,
        question=None,
        model=None,
        format="json",
        output=None,
        timeout_seconds=1.0,
        max_url_bytes=1000,
    )
    stderr = StringIO()

    exit_code = ingest_cli.run_agent_dry_run_cli(args, stdout=StringIO(), stderr=stderr)

    assert exit_code == 2
    assert "input JSON must be an object" in stderr.getvalue()
