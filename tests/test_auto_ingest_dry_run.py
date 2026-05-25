import argparse
import json
from io import StringIO
from pathlib import Path
from types import MappingProxyType, SimpleNamespace
from typing import Any

import pytest

from nesy_reasoning_mcp.auto_ingest import (
    CandidateRelation,
    EvidenceRecord,
    IngestionInput,
    IngestionReport,
    ReviewDecision,
    ReviewDecisionValue,
    fetcher,
    openai_agents,
)
from nesy_reasoning_mcp.auto_ingest import cli as ingest_cli
from nesy_reasoning_mcp.auto_ingest import gate as gate_module
from nesy_reasoning_mcp.auto_ingest.fetcher import fetch_url_evidence
from nesy_reasoning_mcp.auto_ingest.gate import run_dry_run_gate
from nesy_reasoning_mcp.auto_ingest.openai_agents import (
    OpenAIAgentsDryRunError,
    OpenAICompatibleProviderConfig,
    run_openai_agents_dry_run,
)
from nesy_reasoning_mcp.auto_ingest.providers import (
    PROVIDER_REGISTRY,
    get_provider_entry,
    list_provider_entries,
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


def test_openai_agents_output_schema_accepts_ingestion_batches() -> None:
    agent = openai_agents._build_agent(
        name="test extractor",
        instructions="test",
        output_type=openai_agents.CandidateRelationBatch,
        model=None,
    )

    assert agent.output_type.output_type is openai_agents.CandidateRelationBatch
    assert agent.output_type._strict_json_schema is False


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

    class Opener:
        def open(self, req: Any, timeout: float) -> Response:
            return fake_open(req, timeout)

    def fake_open(req: Any, timeout: float) -> Response:
        assert req.full_url == "https://example.com/source"
        assert timeout == 3
        return Response()

    monkeypatch.setattr(
        fetcher, "getaddrinfo", lambda *args: [(None, None, None, None, ("93.184.216.34", 443))]
    )
    monkeypatch.setattr(fetcher.request, "build_opener", lambda *args: Opener())

    record = fetch_url_evidence("https://example.com/source", timeout_seconds=3, max_bytes=5)

    assert record.span == "abcde"
    assert record.metadata["truncated"] is True
    with pytest.raises(ValueError):
        fetch_url_evidence("file:///tmp/source.txt")
    with pytest.raises(ValueError):
        fetch_url_evidence("http://localhost/source")
    with pytest.raises(ValueError):
        fetch_url_evidence("http://127.0.0.1/source")


def test_fetch_url_evidence_rejects_private_dns_resolution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_build_opener(*args: Any) -> Any:
        raise AssertionError("private DNS target must be rejected before fetch")

    monkeypatch.setattr(
        fetcher, "getaddrinfo", lambda *args: [(None, None, None, None, ("10.0.0.5", 443))]
    )
    monkeypatch.setattr(fetcher.request, "build_opener", fail_build_opener)

    with pytest.raises(ValueError, match="resolved URL host is local or private"):
        fetch_url_evidence("https://example.com/source")


def test_fetch_url_evidence_revalidates_redirect_targets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        fetcher, "getaddrinfo", lambda *args: [(None, None, None, None, ("93.184.216.34", 443))]
    )
    handler = fetcher._SafeRedirectHandler()

    with pytest.raises(ValueError, match="local URLs are not supported"):
        handler.redirect_request(None, None, 302, "Found", {}, "http://127.0.0.1/source")


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
    assert report.written_relation_ids == []
    assert report.gate_results[0].action == "auto_write"
    assert report.gate_results[0].reasons == ["dry-run approved; no persistent write performed"]


async def test_openai_compatible_provider_uses_env_key_headers_and_disables_tracing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate = _candidate()
    review = _approval(candidate)
    captured: dict[str, Any] = {}

    def fake_model(
        model: str,
        api_key: str,
        provider_config: OpenAICompatibleProviderConfig,
    ) -> str:
        captured["model"] = model
        captured["api_key"] = api_key
        captured["base_url"] = provider_config.base_url
        captured["headers"] = provider_config.default_headers
        return "provider-model"

    def fake_agent(**kwargs: Any) -> SimpleNamespace:
        captured.setdefault("agent_models", []).append(kwargs["model"])
        return SimpleNamespace(output_type=kwargs["output_type"])

    async def fake_run_agent(agent: Any, prompt: str, *, tracing_disabled: bool = False) -> Any:
        captured.setdefault("tracing_disabled", []).append(tracing_disabled)
        if agent.output_type is openai_agents.CandidateRelationBatch:
            return {"candidates": [candidate.model_dump(mode="json")]}
        return {"reviews": [review.model_dump(mode="json")]}

    monkeypatch.setattr(openai_agents, "_openai_compatible_model", fake_model)
    monkeypatch.setattr(openai_agents, "_build_agent", fake_agent)
    monkeypatch.setattr(openai_agents, "_run_agent", fake_run_agent)

    report = await run_openai_agents_dry_run(
        IngestionInput(evidence=[_evidence()]),
        store=RelationStore(),
        model="deepseek-v4-pro",
        env={"DEEPSEEK_API_KEY": "secret"},
        provider_config=OpenAICompatibleProviderConfig(
            base_url="https://api.deepseek.com",
            api_key_env="DEEPSEEK_API_KEY",
            default_headers={"X-Test": "yes"},
        ),
    )

    assert captured["model"] == "deepseek-v4-pro"
    assert captured["api_key"] == "secret"
    assert captured["base_url"] == "https://api.deepseek.com"
    assert captured["headers"] == {"X-Test": "yes"}
    assert captured["tracing_disabled"] == [True, True]
    assert captured["agent_models"] == ["provider-model", "provider-model"]
    assert report.metadata["provider"] == {
        "type": "openai_compatible",
        "header_keys": ["X-Test"],
        "tracing_disabled": True,
    }


async def test_custom_runner_can_receive_tracing_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate = _candidate()
    review = _approval(candidate)
    captured: list[bool] = []

    def fake_agent(**kwargs: Any) -> SimpleNamespace:
        return SimpleNamespace(output_type=kwargs["output_type"])

    async def fake_run_agent(agent: Any, prompt: str, *, tracing_disabled: bool) -> Any:
        captured.append(tracing_disabled)
        if agent.output_type is openai_agents.CandidateRelationBatch:
            return {"candidates": [candidate.model_dump(mode="json")]}
        return {"reviews": [review.model_dump(mode="json")]}

    monkeypatch.setattr(openai_agents, "_build_agent", fake_agent)

    await run_openai_agents_dry_run(
        IngestionInput(evidence=[_evidence()]),
        store=RelationStore(),
        env={"OPENAI_API_KEY": "test"},
        run_agent=fake_run_agent,
        disable_tracing=True,
    )

    assert captured == [True, True]


def test_openai_compatible_provider_config_headers_are_read_only() -> None:
    config = OpenAICompatibleProviderConfig(
        base_url="https://api.deepseek.com",
        api_key_env="DEEPSEEK_API_KEY",
        default_headers={"X-Test": "yes"},
    )

    assert isinstance(config.default_headers, MappingProxyType)
    with pytest.raises(TypeError):
        config.default_headers["X-Test"] = "no"  # type: ignore[index]


def test_provider_registry_contains_static_shortcuts_without_secrets() -> None:
    assert set(PROVIDER_REGISTRY) == {"deepseek", "kimi", "openrouter"}
    assert isinstance(PROVIDER_REGISTRY, MappingProxyType)
    entries = list_provider_entries()
    assert [entry.name for entry in entries] == ["deepseek", "kimi", "openrouter"]
    assert get_provider_entry("deepseek").base_url == "https://api.deepseek.com"
    assert get_provider_entry("DeepSeek").base_url == "https://api.deepseek.com"
    assert get_provider_entry("kimi").api_key_env == "MOONSHOT_API_KEY"
    assert get_provider_entry("openrouter").default_model is None
    assert get_provider_entry("openrouter").notes
    rendered = ingest_cli._render_provider_list()
    assert "provider\tbase_url\tapi_key_env\tdefault_model\tdocs_url\tnotes" in rendered
    assert "DEEPSEEK_API_KEY" in rendered
    assert "MOONSHOT_API_KEY" in rendered
    assert "OPENROUTER_API_KEY" in rendered
    assert "OpenRouter requires an explicit model" in rendered
    assert "secret" not in rendered.lower()


def test_provider_registry_lookup_error_lists_supported_providers() -> None:
    with pytest.raises(ValueError, match="supported providers: deepseek, kimi, openrouter"):
        get_provider_entry("unknown")


async def test_openai_compatible_provider_requires_model_and_env_key() -> None:
    provider_config = OpenAICompatibleProviderConfig(
        base_url="https://api.deepseek.com",
        api_key_env="DEEPSEEK_API_KEY",
    )
    with pytest.raises(OpenAIAgentsDryRunError, match="--model or OPENAI_DEFAULT_MODEL"):
        await run_openai_agents_dry_run(
            IngestionInput(evidence=[_evidence()]),
            store=RelationStore(),
            env={"DEEPSEEK_API_KEY": "secret"},
            provider_config=provider_config,
        )
    with pytest.raises(OpenAIAgentsDryRunError, match="DEEPSEEK_API_KEY is required"):
        await run_openai_agents_dry_run(
            IngestionInput(evidence=[_evidence()]),
            store=RelationStore(),
            model="deepseek-v4-pro",
            env={},
            provider_config=provider_config,
        )


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


async def test_dry_run_gate_queues_when_reasoning_tool_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = RelationStore()
    candidate = _candidate()

    async def fake_call_tool(*args: Any, **kwargs: Any) -> SimpleNamespace:
        return SimpleNamespace(
            isError=True,
            structuredContent={
                "status": "error",
                "diagnostics": [
                    {
                        "level": "error",
                        "code": "DRY_RUN_FAILED",
                        "message": "ephemeral reasoning failed",
                        "related_ids": [],
                    }
                ],
            },
        )

    monkeypatch.setattr(gate_module, "call_tool", fake_call_tool)

    gate_results, approved_relations, diagnostics, reasoning = await run_dry_run_gate(
        candidates=[candidate],
        reviews=[_approval(candidate)],
        store=store,
    )

    assert gate_results[0].action == "queue"
    assert gate_results[0].reasons == ["dry-run reasoning failed"]
    assert approved_relations == []
    assert diagnostics[0].code == "DRY_RUN_FAILED"
    assert reasoning["status"] == "error"
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
        auto_write: bool = False,
        min_write_confidence: float = 0.85,
        provider_config: OpenAICompatibleProviderConfig | None = None,
        disable_tracing: bool = False,
    ) -> IngestionReport:
        assert ingestion_input.evidence[0].url == "https://example.com/source"
        assert store.list_relations() == []
        assert model == "gpt-test"
        assert auto_write is False
        assert min_write_confidence == 0.85
        assert provider_config is None
        assert disable_tracing is False
        return IngestionReport(candidates=[_candidate()])

    monkeypatch.setattr(ingest_cli, "run_openai_agents_ingestion", fake_run)
    stdout = StringIO()
    stderr = StringIO()
    args = argparse.Namespace(
        input=str(input_path),
        url=[],
        task=None,
        question=None,
        model="gpt-test",
        auto_write=False,
        min_write_confidence=0.85,
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


def test_cli_disable_tracing_default_openai_path(
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
        auto_write: bool = False,
        min_write_confidence: float = 0.85,
        provider_config: OpenAICompatibleProviderConfig | None = None,
        disable_tracing: bool = False,
    ) -> IngestionReport:
        assert ingestion_input.evidence
        assert store.list_relations() == []
        assert model is None
        assert auto_write is False
        assert min_write_confidence == 0.85
        assert provider_config is None
        assert disable_tracing is True
        return IngestionReport(candidates=[_candidate()])

    monkeypatch.setattr(ingest_cli, "run_openai_agents_ingestion", fake_run)
    args = argparse.Namespace(
        input=str(input_path),
        url=[],
        task=None,
        question=None,
        model=None,
        base_url=None,
        api_key_env=None,
        provider_header=[],
        disable_tracing=True,
        auto_write=False,
        min_write_confidence=0.85,
        format="json",
        output=None,
        timeout_seconds=1.0,
        max_url_bytes=1000,
    )

    exit_code = ingest_cli.run_agent_dry_run_cli(args, stdout=StringIO(), stderr=StringIO())

    assert exit_code == 0


def test_script_wrapper_accepts_auto_write_arguments(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_path = tmp_path / "input.json"
    output_path = tmp_path / "report.json"
    input_path.write_text(
        json.dumps({"evidence": [_evidence().model_dump(mode="json")]}),
        encoding="utf-8",
    )

    async def fake_run(
        ingestion_input: IngestionInput,
        *,
        store: Any,
        model: str | None = None,
        auto_write: bool = False,
        min_write_confidence: float = 0.85,
        provider_config: OpenAICompatibleProviderConfig | None = None,
        disable_tracing: bool = False,
    ) -> IngestionReport:
        assert ingestion_input.evidence
        assert store.list_relations() == []
        assert model is None
        assert auto_write is True
        assert min_write_confidence == 0.91
        assert provider_config is None
        assert disable_tracing is False
        return IngestionReport(written_relation_ids=["rel-1"])

    monkeypatch.setattr(ingest_cli, "run_openai_agents_ingestion", fake_run)

    exit_code = ingest_cli.main(
        [
            "--input",
            str(input_path),
            "--auto-write",
            "--min-write-confidence",
            "0.91",
            "--output",
            str(output_path),
        ]
    )
    payload = json.loads(output_path.read_text(encoding="utf-8"))

    assert exit_code == 0
    assert payload["written_relation_ids"] == ["rel-1"]


def test_cli_passes_openai_compatible_provider_config(
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
        auto_write: bool = False,
        min_write_confidence: float = 0.85,
        provider_config: OpenAICompatibleProviderConfig | None = None,
        disable_tracing: bool = False,
    ) -> IngestionReport:
        assert ingestion_input.evidence
        assert store.list_relations() == []
        assert model == "deepseek-v4-pro"
        assert auto_write is False
        assert min_write_confidence == 0.85
        assert provider_config == OpenAICompatibleProviderConfig(
            base_url="https://api.deepseek.com",
            api_key_env="DEEPSEEK_API_KEY",
            default_headers={"X-Test": "yes"},
            disable_tracing=True,
        )
        assert disable_tracing is False
        return IngestionReport(candidates=[_candidate()])

    monkeypatch.setattr(ingest_cli, "run_openai_agents_ingestion", fake_run)
    stdout = StringIO()
    stderr = StringIO()
    args = argparse.Namespace(
        input=str(input_path),
        url=[],
        task=None,
        question=None,
        model="deepseek-v4-pro",
        base_url="https://api.deepseek.com",
        api_key_env="DEEPSEEK_API_KEY",
        provider_header=["X-Test=yes"],
        disable_tracing=False,
        auto_write=False,
        min_write_confidence=0.85,
        format="json",
        output=None,
        timeout_seconds=1.0,
        max_url_bytes=1000,
    )

    exit_code = ingest_cli.run_agent_dry_run_cli(args, stdout=stdout, stderr=stderr)

    assert exit_code == 0
    assert stderr.getvalue() == ""


def test_cli_list_providers_does_not_require_input_or_api_key() -> None:
    stdout = StringIO()
    stderr = StringIO()
    args = argparse.Namespace(list_providers=True)

    exit_code = ingest_cli.run_agent_dry_run_cli(args, stdout=stdout, stderr=stderr)

    assert exit_code == 0
    assert stderr.getvalue() == ""
    rendered = stdout.getvalue()
    assert "deepseek\thttps://api.deepseek.com\tDEEPSEEK_API_KEY\tdeepseek-v4-pro" in rendered
    assert "kimi\thttps://api.moonshot.cn/v1\tMOONSHOT_API_KEY\tkimi-k2.6" in rendered
    assert "openrouter\thttps://openrouter.ai/api/v1\tOPENROUTER_API_KEY\t-" in rendered
    assert "OpenRouter requires an explicit model" in rendered
    assert "secret" not in rendered.lower()


@pytest.mark.parametrize(
    ("provider_name", "expected_base_url", "expected_api_key_env", "expected_model"),
    [
        ("deepseek", "https://api.deepseek.com", "DEEPSEEK_API_KEY", "deepseek-v4-pro"),
        ("kimi", "https://api.moonshot.cn/v1", "MOONSHOT_API_KEY", "kimi-k2.6"),
    ],
)
def test_cli_provider_shortcuts_fill_config_and_default_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    provider_name: str,
    expected_base_url: str,
    expected_api_key_env: str,
    expected_model: str,
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
        auto_write: bool = False,
        min_write_confidence: float = 0.85,
        provider_config: OpenAICompatibleProviderConfig | None = None,
        disable_tracing: bool = False,
    ) -> IngestionReport:
        assert ingestion_input.evidence
        assert store.list_relations() == []
        assert model == expected_model
        assert auto_write is False
        assert min_write_confidence == 0.85
        assert provider_config == OpenAICompatibleProviderConfig(
            base_url=expected_base_url,
            api_key_env=expected_api_key_env,
            disable_tracing=True,
        )
        assert disable_tracing is False
        return IngestionReport(candidates=[_candidate()])

    monkeypatch.setattr(ingest_cli, "run_openai_agents_ingestion", fake_run)
    args = argparse.Namespace(
        input=str(input_path),
        url=[],
        task=None,
        question=None,
        model=None,
        provider=provider_name,
        list_providers=False,
        base_url=None,
        api_key_env=None,
        provider_header=[],
        disable_tracing=False,
        auto_write=False,
        min_write_confidence=0.85,
        format="json",
        output=None,
        timeout_seconds=1.0,
        max_url_bytes=1000,
    )

    exit_code = ingest_cli.run_agent_dry_run_cli(args, stdout=StringIO(), stderr=StringIO())

    assert exit_code == 0


def test_cli_provider_explicit_flags_override_registry(
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
        auto_write: bool = False,
        min_write_confidence: float = 0.85,
        provider_config: OpenAICompatibleProviderConfig | None = None,
        disable_tracing: bool = False,
    ) -> IngestionReport:
        assert ingestion_input.evidence
        assert store.list_relations() == []
        assert model == "custom-model"
        assert provider_config == OpenAICompatibleProviderConfig(
            base_url="https://custom.example/v1",
            api_key_env="CUSTOM_API_KEY",
            default_headers={"X-Test": "yes"},
            disable_tracing=True,
        )
        assert auto_write is False
        assert min_write_confidence == 0.85
        assert disable_tracing is False
        return IngestionReport(candidates=[_candidate()])

    monkeypatch.setattr(ingest_cli, "run_openai_agents_ingestion", fake_run)
    args = argparse.Namespace(
        input=str(input_path),
        url=[],
        task=None,
        question=None,
        model="custom-model",
        provider="deepseek",
        list_providers=False,
        base_url="https://custom.example/v1",
        api_key_env="CUSTOM_API_KEY",
        provider_header=["X-Test=yes"],
        disable_tracing=False,
        auto_write=False,
        min_write_confidence=0.85,
        format="json",
        output=None,
        timeout_seconds=1.0,
        max_url_bytes=1000,
    )

    exit_code = ingest_cli.run_agent_dry_run_cli(args, stdout=StringIO(), stderr=StringIO())

    assert exit_code == 0


def test_cli_openrouter_provider_accepts_headers_with_model(
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
        auto_write: bool = False,
        min_write_confidence: float = 0.85,
        provider_config: OpenAICompatibleProviderConfig | None = None,
        disable_tracing: bool = False,
    ) -> IngestionReport:
        assert ingestion_input.evidence
        assert store.list_relations() == []
        assert model == "openai/gpt-latest"
        assert provider_config == OpenAICompatibleProviderConfig(
            base_url="https://openrouter.ai/api/v1",
            api_key_env="OPENROUTER_API_KEY",
            default_headers={
                "HTTP-Referer": "https://github.com/6tizer/nesy-reasoning-mcp",
                "X-OpenRouter-Title": "NeSy Reasoning MCP",
            },
            disable_tracing=True,
        )
        assert auto_write is False
        assert min_write_confidence == 0.85
        assert disable_tracing is False
        return IngestionReport(candidates=[_candidate()])

    monkeypatch.setattr(ingest_cli, "run_openai_agents_ingestion", fake_run)
    args = argparse.Namespace(
        input=str(input_path),
        url=[],
        task=None,
        question=None,
        model="openai/gpt-latest",
        provider="openrouter",
        list_providers=False,
        base_url=None,
        api_key_env=None,
        provider_header=[
            "HTTP-Referer=https://github.com/6tizer/nesy-reasoning-mcp",
            "X-OpenRouter-Title=NeSy Reasoning MCP",
        ],
        disable_tracing=False,
        auto_write=False,
        min_write_confidence=0.85,
        format="json",
        output=None,
        timeout_seconds=1.0,
        max_url_bytes=1000,
    )

    exit_code = ingest_cli.run_agent_dry_run_cli(args, stdout=StringIO(), stderr=StringIO())

    assert exit_code == 0


def test_cli_provider_unknown_returns_clear_error(tmp_path: Path) -> None:
    input_path = tmp_path / "input.json"
    input_path.write_text(
        json.dumps({"evidence": [_evidence().model_dump(mode="json")]}),
        encoding="utf-8",
    )
    stderr = StringIO()

    exit_code = ingest_cli.run_agent_dry_run_cli(
        argparse.Namespace(
            input=str(input_path),
            url=[],
            task=None,
            question=None,
            model=None,
            provider="unknown",
            list_providers=False,
            base_url=None,
            api_key_env=None,
            provider_header=[],
            disable_tracing=False,
            auto_write=False,
            min_write_confidence=0.85,
            format="json",
            output=None,
            timeout_seconds=1.0,
            max_url_bytes=1000,
        ),
        stdout=StringIO(),
        stderr=stderr,
    )

    assert exit_code == 2
    assert "unknown provider 'unknown'" in stderr.getvalue()
    assert "deepseek, kimi, openrouter" in stderr.getvalue()
    assert "--list-providers" in stderr.getvalue()


def test_cli_openrouter_provider_requires_explicit_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OPENAI_DEFAULT_MODEL", raising=False)
    input_path = tmp_path / "input.json"
    input_path.write_text(
        json.dumps({"evidence": [_evidence().model_dump(mode="json")]}),
        encoding="utf-8",
    )
    stderr = StringIO()

    exit_code = ingest_cli.run_agent_dry_run_cli(
        argparse.Namespace(
            input=str(input_path),
            url=[],
            task=None,
            question=None,
            model=None,
            provider="openrouter",
            list_providers=False,
            base_url=None,
            api_key_env=None,
            provider_header=[],
            disable_tracing=False,
            auto_write=False,
            min_write_confidence=0.85,
            format="json",
            output=None,
            timeout_seconds=1.0,
            max_url_bytes=1000,
        ),
        stdout=StringIO(),
        stderr=stderr,
    )

    assert exit_code == 2
    assert "provider 'openrouter' requires --model or OPENAI_DEFAULT_MODEL" in stderr.getvalue()


@pytest.mark.parametrize(
    ("args_update", "message"),
    [
        ({"base_url": "https://api.deepseek.com", "api_key_env": None}, "--api-key-env"),
        ({"base_url": None, "api_key_env": "DEEPSEEK_API_KEY"}, "--api-key-env requires"),
        (
            {"base_url": None, "api_key_env": None, "provider_header": ["X-Test=yes"]},
            "--provider-header requires",
        ),
        ({"base_url": "https://api.deepseek.com", "provider_header": ["bad"]}, "KEY=VALUE"),
        ({"base_url": "https://api.deepseek.com", "provider_header": ["X-Test="]}, "non-empty"),
        ({"base_url": "http://api.deepseek.com"}, "https URL"),
        (
            {"base_url": "https://api.deepseek.com", "provider_header": ["Bad Header=yes"]},
            "HTTP header token",
        ),
        (
            {"base_url": "https://api.deepseek.com", "provider_header": ["X-Test=bad\nvalue"]},
            "must not contain newlines",
        ),
    ],
)
def test_cli_rejects_invalid_provider_config(
    tmp_path: Path,
    args_update: dict[str, Any],
    message: str,
) -> None:
    input_path = tmp_path / "input.json"
    input_path.write_text(
        json.dumps({"evidence": [_evidence().model_dump(mode="json")]}),
        encoding="utf-8",
    )
    values = {
        "input": str(input_path),
        "url": [],
        "task": None,
        "question": None,
        "model": "provider-model",
        "provider": None,
        "list_providers": False,
        "base_url": "https://api.deepseek.com",
        "api_key_env": "DEEPSEEK_API_KEY",
        "provider_header": [],
        "disable_tracing": False,
        "auto_write": False,
        "min_write_confidence": 0.85,
        "format": "json",
        "output": None,
        "timeout_seconds": 1.0,
        "max_url_bytes": 1000,
    }
    values.update(args_update)
    stderr = StringIO()

    exit_code = ingest_cli.run_agent_dry_run_cli(
        argparse.Namespace(**values),
        stdout=StringIO(),
        stderr=stderr,
    )

    assert exit_code == 2
    assert message in stderr.getvalue()


def test_cli_agent_dry_run_invalid_input_returns_nonzero(tmp_path: Path) -> None:
    input_path = tmp_path / "input.json"
    input_path.write_text("[]", encoding="utf-8")
    args = argparse.Namespace(
        input=str(input_path),
        url=[],
        task=None,
        question=None,
        model=None,
        provider=None,
        list_providers=False,
        auto_write=False,
        min_write_confidence=0.85,
        format="json",
        output=None,
        timeout_seconds=1.0,
        max_url_bytes=1000,
    )
    stderr = StringIO()

    exit_code = ingest_cli.run_agent_dry_run_cli(args, stdout=StringIO(), stderr=stderr)

    assert exit_code == 2
    assert "input JSON must be an object" in stderr.getvalue()
