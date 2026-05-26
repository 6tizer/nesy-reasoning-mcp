import argparse
import json
from io import StringIO
from pathlib import Path
from types import MappingProxyType, SimpleNamespace
from typing import Any

import anyio
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
    LLMRuntimeOptions,
    OpenAIAgentsDryRunError,
    OpenAICompatibleProviderConfig,
    ReviewerModelConfig,
    run_openai_agents_dry_run,
)
from nesy_reasoning_mcp.auto_ingest.providers import (
    PROVIDER_REGISTRY,
    ProviderStructuredOutputMode,
    get_provider_entry,
    list_provider_entries,
)
from nesy_reasoning_mcp.schemas import Diagnostic, PropositionRecord
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
        normalized_implication_supported=True,
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


def test_reviewer_prompt_includes_normalized_implication_rules_and_preview() -> None:
    candidate = CandidateRelation(
        id="candidate-necessary",
        source="A",
        target="B",
        relation_type="necessary",
        confidence=0.9,
        evidence=[_evidence("B cannot happen without A.")],
    )

    prompt = openai_agents._review_prompt(
        IngestionInput(evidence=[_evidence()]),
        openai_agents.CandidateRelationBatch(candidates=[candidate]),
    )

    assert "necessary(A, B)=B -> A" in prompt
    assert '"normalized_implications"' in prompt
    assert '"antecedent": "B"' in prompt
    assert '"consequent": "A"' in prompt
    assert "normalized_implication_supported=true" in prompt


async def test_openai_agents_dry_run_runs_multiple_reviewers_and_reports_voting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate = _candidate()
    captured_reviewers: list[str] = []

    def fake_agent(**kwargs: Any) -> SimpleNamespace:
        return SimpleNamespace(
            name=kwargs["name"],
            model=kwargs["model"],
            output_type=kwargs["output_type"],
        )

    async def fake_run_agent(agent: Any, prompt: str) -> Any:
        if agent.output_type is openai_agents.CandidateRelationBatch:
            return {"candidates": [candidate.model_dump(mode="json")]}
        captured_reviewers.append(agent.model)
        decision = (
            ReviewDecisionValue.NEEDS_HUMAN
            if agent.model == "reviewer-c"
            else ReviewDecisionValue.APPROVE
        )
        review = ReviewDecision(
            candidate_id=candidate.id,
            decision=decision,
            final_relation_type="sufficient" if decision == ReviewDecisionValue.APPROVE else None,
            final_confidence=0.88 if decision == ReviewDecisionValue.APPROVE else None,
            normalized_implication_supported=True
            if decision == ReviewDecisionValue.APPROVE
            else None,
            reasons=[f"{agent.model} reviewed"],
            reviewer_model="model-reported-by-agent",
        )
        return {"reviews": [review.model_dump(mode="json", exclude_none=True)]}

    monkeypatch.setattr(openai_agents, "_build_agent", fake_agent)

    report = await run_openai_agents_dry_run(
        IngestionInput(evidence=[_evidence()]),
        store=RelationStore(),
        model="extractor-model",
        reviewer_models=["reviewer-a", "reviewer-b", "reviewer-c"],
        voting_policy=openai_agents.ReviewVotingPolicy.MAJORITY,
        env={"OPENAI_API_KEY": "test"},
        run_agent=fake_run_agent,
    )

    assert captured_reviewers == ["reviewer-a", "reviewer-b", "reviewer-c"]
    assert [review.reviewer_model for review in report.reviews] == [
        "reviewer-a",
        "reviewer-b",
        "reviewer-c",
    ]
    assert report.metadata["review_aggregation"]["policy"] == "majority"
    assert report.metadata["review_aggregation"]["candidates"][0]["review_count"] == 3
    assert report.gate_results[0].action == "auto_write"
    assert report.approved_relations[0].confidence == 0.88
    assert report.reviews[0].metadata["reported_reviewer_model"] == "model-reported-by-agent"


async def test_json_object_reviewers_run_in_parallel() -> None:
    candidate = _candidate()
    active_reviewers = 0
    max_active_reviewers = 0

    async def fake_chat_completion(**kwargs: Any) -> str:
        nonlocal active_reviewers, max_active_reviewers
        if kwargs["model"] == "extractor-model":
            return json.dumps({"candidates": [candidate.model_dump(mode="json")]})
        active_reviewers += 1
        max_active_reviewers = max(max_active_reviewers, active_reviewers)
        await anyio.sleep(0.02)
        active_reviewers -= 1
        review = _approval(candidate).model_copy(
            update={"reasons": [f"{kwargs['model']} approved"]}
        )
        return json.dumps({"reviews": [review.model_dump(mode="json")]})

    await run_openai_agents_dry_run(
        IngestionInput(evidence=[_evidence()]),
        store=RelationStore(),
        model="extractor-model",
        reviewer_models=["reviewer-a", "reviewer-b"],
        voting_policy=openai_agents.ReviewVotingPolicy.MAJORITY,
        env={"DEEPSEEK_API_KEY": "secret"},
        provider_config=OpenAICompatibleProviderConfig(
            base_url="https://api.deepseek.com",
            api_key_env="DEEPSEEK_API_KEY",
            structured_output_mode=ProviderStructuredOutputMode.JSON_OBJECT,
            reasoning_effort="high",
            extra_body={"thinking": {"type": "enabled"}},
        ),
        run_chat_completion=fake_chat_completion,
    )

    assert max_active_reviewers == 2


async def test_risk_tiered_high_priority_concern_skips_normal_reviewers() -> None:
    candidate = _candidate()
    called_models: list[str] = []

    async def fake_chat_completion(**kwargs: Any) -> str:
        called_models.append(kwargs["model"])
        if kwargs["model"] == "extractor-model":
            return json.dumps({"candidates": [candidate.model_dump(mode="json")]})
        if kwargs["model"] == "high-priority":
            review = ReviewDecision(
                candidate_id=candidate.id,
                decision=ReviewDecisionValue.NEEDS_HUMAN,
                reasons=["needs human"],
            )
            return json.dumps({"reviews": [review.model_dump(mode="json", exclude_none=True)]})
        raise AssertionError("normal reviewer should be skipped")

    report = await run_openai_agents_dry_run(
        IngestionInput(evidence=[_evidence()]),
        store=RelationStore(),
        model="extractor-model",
        reviewer_models=["high-priority", "normal-reviewer"],
        high_priority_reviewer_models=["high-priority"],
        env={"DEEPSEEK_API_KEY": "secret"},
        provider_config=OpenAICompatibleProviderConfig(
            base_url="https://api.deepseek.com",
            api_key_env="DEEPSEEK_API_KEY",
            structured_output_mode=ProviderStructuredOutputMode.JSON_OBJECT,
            reasoning_effort="high",
            extra_body={"thinking": {"type": "enabled"}},
        ),
        run_chat_completion=fake_chat_completion,
    )

    assert called_models == ["extractor-model", "high-priority"]
    assert report.gate_results[0].action == "queue"
    assert report.metadata["runtime_trace"][-1]["status"] == "skipped"
    assert any(
        review.metadata.get("synthetic_vote") == "reviewer_skipped" for review in report.reviews
    )


async def test_reviewer_timeout_queues_auto_write_without_graph_write() -> None:
    store = RelationStore()
    candidate = _candidate()

    async def fake_chat_completion(**kwargs: Any) -> str:
        if kwargs["model"] == "extractor-model":
            return json.dumps({"candidates": [candidate.model_dump(mode="json")]})
        if kwargs["model"] == "slow-reviewer":
            await anyio.sleep(1)
        review = _approval(candidate)
        return json.dumps({"reviews": [review.model_dump(mode="json")]})

    report = await openai_agents.run_openai_agents_ingestion(
        IngestionInput(evidence=[_evidence()]),
        store=store,
        model="extractor-model",
        reviewer_models=["fast-reviewer", "slow-reviewer"],
        voting_policy=openai_agents.ReviewVotingPolicy.MAJORITY,
        env={"DEEPSEEK_API_KEY": "secret"},
        provider_config=OpenAICompatibleProviderConfig(
            base_url="https://api.deepseek.com",
            api_key_env="DEEPSEEK_API_KEY",
            structured_output_mode=ProviderStructuredOutputMode.JSON_OBJECT,
            reasoning_effort="high",
            extra_body={"thinking": {"type": "enabled"}},
        ),
        run_chat_completion=fake_chat_completion,
        runtime_options=LLMRuntimeOptions(reviewer_timeout_seconds=0.01),
        auto_write=True,
    )

    assert report.diagnostics[0].code == "LLM_RUNTIME_TIMEOUT"
    assert report.gate_results[0].action == "queue"
    assert store.list_relations() == []


async def test_extractor_timeout_returns_diagnostic_report_without_graph_write() -> None:
    store = RelationStore()

    async def fake_chat_completion(**kwargs: Any) -> str:
        await anyio.sleep(1)
        return "{}"

    report = await openai_agents.run_openai_agents_ingestion(
        IngestionInput(evidence=[_evidence()]),
        store=store,
        model="extractor-model",
        env={"DEEPSEEK_API_KEY": "secret"},
        provider_config=OpenAICompatibleProviderConfig(
            base_url="https://api.deepseek.com",
            api_key_env="DEEPSEEK_API_KEY",
            structured_output_mode=ProviderStructuredOutputMode.JSON_OBJECT,
            reasoning_effort="high",
            extra_body={"thinking": {"type": "enabled"}},
        ),
        run_chat_completion=fake_chat_completion,
        runtime_options=LLMRuntimeOptions(extractor_timeout_seconds=0.01),
        auto_write=True,
    )

    assert report.diagnostics[0].code == "LLM_RUNTIME_TIMEOUT"
    assert report.metadata["runtime_trace"][0]["stage"] == "extractor"
    assert report.metadata["runtime_trace"][0]["status"] == "timeout"
    assert store.list_relations() == []


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
        "structured_output_mode": "agent_schema",
    }


async def test_deepseek_json_object_provider_uses_chat_completions_json_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate = _candidate()
    review = _approval(candidate)
    captured: list[dict[str, Any]] = []

    def fail_build_agent(**kwargs: Any) -> None:
        raise AssertionError("DeepSeek JSON Object mode must bypass AgentOutputSchema")

    async def fake_chat_completion(**kwargs: Any) -> str:
        captured.append(kwargs)
        assert kwargs["response_format"] == {"type": "json_object"}
        assert kwargs["reasoning_effort"] == "high"
        assert kwargs["extra_body"] == {"thinking": {"type": "enabled"}}
        assert kwargs["provider_config"].api_key_env == "DEEPSEEK_API_KEY"
        assert "secret" not in json.dumps(kwargs, default=str)
        assert "JSON object" in kwargs["messages"][0]["content"]
        if len(captured) == 1:
            return json.dumps({"candidates": [candidate.model_dump(mode="json")]})
        return json.dumps({"reviews": [review.model_dump(mode="json")]})

    monkeypatch.setattr(openai_agents, "_build_agent", fail_build_agent)

    report = await run_openai_agents_dry_run(
        IngestionInput(evidence=[_evidence()]),
        store=RelationStore(),
        model="deepseek-v4-pro",
        reviewer_models=["deepseek-v4-pro"],
        env={"DEEPSEEK_API_KEY": "secret"},
        provider_config=OpenAICompatibleProviderConfig(
            base_url="https://api.deepseek.com",
            api_key_env="DEEPSEEK_API_KEY",
            structured_output_mode=ProviderStructuredOutputMode.JSON_OBJECT,
            reasoning_effort="high",
            extra_body={"thinking": {"type": "enabled"}},
        ),
        run_chat_completion=fake_chat_completion,
    )

    assert [request["model"] for request in captured] == [
        "deepseek-v4-pro",
        "deepseek-v4-pro",
    ]
    assert report.candidates == [candidate]
    assert report.reviews == [review.model_copy(update={"reviewer_model": "deepseek-v4-pro"})]
    assert report.gate_results[0].action == "auto_write"
    assert report.metadata["provider"] == {
        "type": "openai_compatible",
        "header_keys": [],
        "tracing_disabled": True,
        "structured_output_mode": "json_object",
        "reasoning_effort": "high",
        "thinking": {"type": "enabled"},
    }


async def test_kimi_json_object_provider_uses_chat_completions_json_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate = _candidate()
    review = _approval(candidate)
    captured: list[dict[str, Any]] = []

    def fail_build_agent(**kwargs: Any) -> None:
        raise AssertionError("Kimi JSON Object mode must bypass AgentOutputSchema")

    async def fake_chat_completion(**kwargs: Any) -> str:
        captured.append(kwargs)
        assert kwargs["response_format"] == {"type": "json_object"}
        assert "reasoning_effort" not in kwargs
        assert kwargs["extra_body"] == {"thinking": {"type": "enabled"}}
        assert kwargs["provider_config"].api_key_env == "MOONSHOT_API_KEY"
        assert "secret" not in json.dumps(kwargs, default=str)
        assert "JSON object" in kwargs["messages"][0]["content"]
        if len(captured) == 1:
            return json.dumps({"candidates": [candidate.model_dump(mode="json")]})
        return json.dumps({"reviews": [review.model_dump(mode="json")]})

    monkeypatch.setattr(openai_agents, "_build_agent", fail_build_agent)

    report = await run_openai_agents_dry_run(
        IngestionInput(evidence=[_evidence()]),
        store=RelationStore(),
        model="kimi-k2.6",
        reviewer_models=["kimi-k2.6"],
        env={"MOONSHOT_API_KEY": "secret"},
        provider_config=OpenAICompatibleProviderConfig(
            base_url="https://api.moonshot.cn/v1",
            api_key_env="MOONSHOT_API_KEY",
            structured_output_mode=ProviderStructuredOutputMode.JSON_OBJECT,
            extra_body={"thinking": {"type": "enabled"}},
        ),
        run_chat_completion=fake_chat_completion,
    )

    assert [request["model"] for request in captured] == [
        "kimi-k2.6",
        "kimi-k2.6",
    ]
    assert report.candidates == [candidate]
    assert report.reviews == [review.model_copy(update={"reviewer_model": "kimi-k2.6"})]
    assert report.gate_results[0].action == "auto_write"
    assert report.metadata["provider"] == {
        "type": "openai_compatible",
        "header_keys": [],
        "tracing_disabled": True,
        "structured_output_mode": "json_object",
        "thinking": {"type": "enabled"},
    }


async def test_openrouter_json_object_provider_uses_chat_completions_json_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate = _candidate()
    review = _approval(candidate)
    captured: list[dict[str, Any]] = []

    def fail_build_agent(**kwargs: Any) -> None:
        raise AssertionError("OpenRouter JSON Object mode must bypass AgentOutputSchema")

    async def fake_chat_completion(**kwargs: Any) -> str:
        captured.append(kwargs)
        assert kwargs["response_format"] == {"type": "json_object"}
        assert "reasoning_effort" not in kwargs
        assert kwargs["extra_body"] == {"reasoning": {"effort": "medium", "exclude": True}}
        assert kwargs["max_tokens"] in {4096, 2048}
        assert kwargs["provider_config"].api_key_env == "OPENROUTER_API_KEY"
        assert kwargs["provider_config"].default_headers == {
            "HTTP-Referer": "https://github.com/6tizer/nesy-reasoning-mcp",
            "X-OpenRouter-Title": "NeSy Reasoning MCP",
        }
        assert "secret" not in json.dumps(kwargs, default=str)
        assert "JSON object" in kwargs["messages"][0]["content"]
        if len(captured) == 1:
            return json.dumps({"candidates": [candidate.model_dump(mode="json")]})
        return json.dumps({"reviews": [review.model_dump(mode="json")]})

    monkeypatch.setattr(openai_agents, "_build_agent", fail_build_agent)

    report = await run_openai_agents_dry_run(
        IngestionInput(evidence=[_evidence()]),
        store=RelationStore(),
        model="qwen/qwen3.7-max",
        reviewer_models=["qwen/qwen3.7-max"],
        env={"OPENROUTER_API_KEY": "secret"},
        provider_config=OpenAICompatibleProviderConfig(
            base_url="https://openrouter.ai/api/v1",
            api_key_env="OPENROUTER_API_KEY",
            default_headers={
                "HTTP-Referer": "https://github.com/6tizer/nesy-reasoning-mcp",
                "X-OpenRouter-Title": "NeSy Reasoning MCP",
            },
            structured_output_mode=ProviderStructuredOutputMode.JSON_OBJECT,
            extra_body={"reasoning": {"effort": "medium", "exclude": True}},
        ),
        run_chat_completion=fake_chat_completion,
    )

    assert [request["model"] for request in captured] == [
        "qwen/qwen3.7-max",
        "qwen/qwen3.7-max",
    ]
    assert report.candidates == [candidate]
    assert report.reviews == [review.model_copy(update={"reviewer_model": "qwen/qwen3.7-max"})]
    assert report.gate_results[0].action == "auto_write"
    assert report.metadata["provider"] == {
        "type": "openai_compatible",
        "header_keys": ["HTTP-Referer", "X-OpenRouter-Title"],
        "tracing_disabled": True,
        "structured_output_mode": "json_object",
        "reasoning": {"effort": "medium", "exclude": True},
    }


async def test_cross_provider_reviewers_use_their_own_provider_configs() -> None:
    candidate = _candidate()
    review = _approval(candidate)
    captured_api_key_envs: list[str] = []
    captured_models: list[str] = []

    async def fake_chat_completion(**kwargs: Any) -> str:
        provider_config = kwargs["provider_config"]
        captured_api_key_envs.append(provider_config.api_key_env)
        captured_models.append(kwargs["model"])
        if len(captured_api_key_envs) == 1:
            return json.dumps({"candidates": [candidate.model_dump(mode="json")]})
        return json.dumps({"reviews": [review.model_dump(mode="json")]})

    report = await run_openai_agents_dry_run(
        IngestionInput(evidence=[_evidence()]),
        store=RelationStore(),
        model="deepseek-v4-pro",
        provider_config=OpenAICompatibleProviderConfig(
            base_url="https://api.deepseek.com",
            api_key_env="DEEPSEEK_API_KEY",
            structured_output_mode=ProviderStructuredOutputMode.JSON_OBJECT,
            reasoning_effort="high",
            extra_body={"thinking": {"type": "enabled"}},
        ),
        reviewer_configs=[
            ReviewerModelConfig(
                reviewer_id="kimi:kimi-k2.6",
                model="kimi-k2.6",
                provider_name="kimi",
                provider_config=OpenAICompatibleProviderConfig(
                    base_url="https://api.moonshot.cn/v1",
                    api_key_env="MOONSHOT_API_KEY",
                    structured_output_mode=ProviderStructuredOutputMode.JSON_OBJECT,
                    extra_body={"thinking": {"type": "enabled"}},
                ),
            ),
            ReviewerModelConfig(
                reviewer_id="openrouter:qwen/qwen3.7-max",
                model="qwen/qwen3.7-max",
                provider_name="openrouter",
                provider_config=OpenAICompatibleProviderConfig(
                    base_url="https://openrouter.ai/api/v1",
                    api_key_env="OPENROUTER_API_KEY",
                    structured_output_mode=ProviderStructuredOutputMode.JSON_OBJECT,
                ),
            ),
        ],
        high_priority_reviewer_models=["kimi:kimi-k2.6"],
        env={
            "DEEPSEEK_API_KEY": "secret",
            "MOONSHOT_API_KEY": "secret",
            "OPENROUTER_API_KEY": "secret",
        },
        run_chat_completion=fake_chat_completion,
    )

    assert captured_api_key_envs == [
        "DEEPSEEK_API_KEY",
        "MOONSHOT_API_KEY",
        "OPENROUTER_API_KEY",
    ]
    assert captured_models == ["deepseek-v4-pro", "kimi-k2.6", "qwen/qwen3.7-max"]
    assert [review.reviewer_model for review in report.reviews] == [
        "kimi:kimi-k2.6",
        "openrouter:qwen/qwen3.7-max",
    ]
    assert report.metadata["review_aggregation"]["high_priority_reviewer_models"] == [
        "kimi:kimi-k2.6"
    ]
    assert report.metadata["reviewer_providers"] == [
        {
            "reviewer_id": "kimi:kimi-k2.6",
            "model": "kimi-k2.6",
            "provider": "kimi",
            "type": "openai_compatible",
            "structured_output_mode": "json_object",
            "header_keys": [],
        },
        {
            "reviewer_id": "openrouter:qwen/qwen3.7-max",
            "model": "qwen/qwen3.7-max",
            "provider": "openrouter",
            "type": "openai_compatible",
            "structured_output_mode": "json_object",
            "header_keys": [],
        },
    ]


async def test_deepseek_json_object_provider_runs_multiple_reviewers() -> None:
    candidate = _candidate()
    captured_models: list[str] = []

    async def fake_chat_completion(**kwargs: Any) -> str:
        captured_models.append(kwargs["model"])
        if len(captured_models) == 1:
            return json.dumps({"candidates": [candidate.model_dump(mode="json")]})
        review = ReviewDecision(
            candidate_id=candidate.id,
            decision=ReviewDecisionValue.APPROVE,
            final_relation_type="sufficient",
            final_confidence=0.9,
            normalized_implication_supported=True,
            reasons=[f"{kwargs['model']} approved"],
        )
        return json.dumps({"reviews": [review.model_dump(mode="json")]})

    report = await run_openai_agents_dry_run(
        IngestionInput(evidence=[_evidence()]),
        store=RelationStore(),
        model="extractor-model",
        reviewer_models=["reviewer-a", "reviewer-b"],
        voting_policy=openai_agents.ReviewVotingPolicy.MAJORITY,
        env={"DEEPSEEK_API_KEY": "secret"},
        provider_config=OpenAICompatibleProviderConfig(
            base_url="https://api.deepseek.com",
            api_key_env="DEEPSEEK_API_KEY",
            structured_output_mode=ProviderStructuredOutputMode.JSON_OBJECT,
            reasoning_effort="high",
            extra_body={"thinking": {"type": "enabled"}},
        ),
        run_chat_completion=fake_chat_completion,
    )

    assert captured_models == ["extractor-model", "reviewer-a", "reviewer-b"]
    assert [review.reviewer_model for review in report.reviews] == ["reviewer-a", "reviewer-b"]
    assert report.metadata["review_aggregation"]["candidates"][0]["review_count"] == 2
    assert report.gate_results[0].action == "auto_write"


async def test_json_object_provider_failure_happens_before_write() -> None:
    store = RelationStore()

    async def fake_chat_completion(**kwargs: Any) -> str:
        return ""

    report = await openai_agents.run_openai_agents_ingestion(
        IngestionInput(evidence=[_evidence()]),
        store=store,
        model="deepseek-v4-pro",
        env={"DEEPSEEK_API_KEY": "secret"},
        provider_config=OpenAICompatibleProviderConfig(
            base_url="https://api.deepseek.com",
            api_key_env="DEEPSEEK_API_KEY",
            structured_output_mode=ProviderStructuredOutputMode.JSON_OBJECT,
            reasoning_effort="high",
            extra_body={"thinking": {"type": "enabled"}},
        ),
        run_chat_completion=fake_chat_completion,
        auto_write=True,
    )

    assert report.diagnostics[0].code == "LLM_RUNTIME_ERROR"
    assert report.metadata["runtime_trace"][0]["status"] == "error"
    assert store.list_relations() == []
    assert store.list_review_queue() == []


async def test_json_object_auto_write_runs_canonicalizer_before_reviewer() -> None:
    store = RelationStore()
    store.import_records(
        [],
        [],
        propositions=[PropositionRecord(id="auto_deploy", label="auto-deploy")],
        mode="append",
        store_id="default",
    )
    candidate = _candidate(target="release is auto-deployed")
    calls: list[str] = []

    async def fake_chat_completion(**kwargs: Any) -> str:
        assert kwargs["response_format"] == {"type": "json_object"}
        if len(calls) == 0:
            calls.append("extractor")
            return json.dumps({"candidates": [candidate.model_dump(mode="json")]})
        if len(calls) == 1:
            calls.append("canonicalizer")
            assert "Canonicalization payload JSON" in kwargs["messages"][1]["content"]
            return json.dumps(
                {
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
            )
        calls.append("reviewer")
        review = ReviewDecision(
            candidate_id=candidate.id,
            decision=ReviewDecisionValue.APPROVE,
            final_relation_type="sufficient",
            final_confidence=0.9,
            normalized_implication_supported=True,
            reasons=["approved"],
        )
        return json.dumps({"reviews": [review.model_dump(mode="json")]})

    report = await openai_agents.run_openai_agents_ingestion(
        IngestionInput(evidence=[_evidence()]),
        store=store,
        model="deepseek-v4-pro",
        env={"DEEPSEEK_API_KEY": "secret"},
        provider_config=OpenAICompatibleProviderConfig(
            base_url="https://api.deepseek.com",
            api_key_env="DEEPSEEK_API_KEY",
            structured_output_mode=ProviderStructuredOutputMode.JSON_OBJECT,
            reasoning_effort="high",
            extra_body={"thinking": {"type": "enabled"}},
        ),
        run_chat_completion=fake_chat_completion,
        auto_write=True,
    )

    assert calls == ["extractor", "canonicalizer", "reviewer"]
    assert report.metadata["runtime_trace"][1]["stage"] == "canonicalizer"
    assert store.list_relations()[0].target_id == "auto_deploy"
    assert "release is auto-deployed" in store.list_propositions()[0].aliases


async def test_json_object_provider_rejects_mapping_without_choices_before_write() -> None:
    store = RelationStore()

    async def fake_chat_completion(**kwargs: Any) -> dict[str, Any]:
        return {"id": "response-id", "usage": {"total_tokens": 10}}

    report = await openai_agents.run_openai_agents_ingestion(
        IngestionInput(evidence=[_evidence()]),
        store=store,
        model="deepseek-v4-pro",
        env={"DEEPSEEK_API_KEY": "secret"},
        provider_config=OpenAICompatibleProviderConfig(
            base_url="https://api.deepseek.com",
            api_key_env="DEEPSEEK_API_KEY",
            structured_output_mode=ProviderStructuredOutputMode.JSON_OBJECT,
            reasoning_effort="high",
            extra_body={"thinking": {"type": "enabled"}},
        ),
        run_chat_completion=fake_chat_completion,
        auto_write=True,
    )

    assert report.diagnostics[0].code == "LLM_RUNTIME_ERROR"
    assert report.metadata["runtime_trace"][0]["error_code"] == "OpenAIAgentsDryRunError"
    assert store.list_relations() == []
    assert store.list_review_queue() == []


async def test_cross_provider_reviewer_failure_happens_before_write_or_queue() -> None:
    store = RelationStore()
    candidate = _candidate()
    calls = 0

    async def fake_chat_completion(**kwargs: Any) -> str:
        nonlocal calls
        calls += 1
        if calls == 1:
            return json.dumps({"candidates": [candidate.model_dump(mode="json")]})
        raise OpenAIAgentsDryRunError("reviewer provider failed")

    report = await openai_agents.run_openai_agents_ingestion(
        IngestionInput(evidence=[_evidence()]),
        store=store,
        model="deepseek-v4-pro",
        reviewer_configs=[
            ReviewerModelConfig(
                reviewer_id="kimi:kimi-k2.6",
                model="kimi-k2.6",
                provider_name="kimi",
                provider_config=OpenAICompatibleProviderConfig(
                    base_url="https://api.moonshot.cn/v1",
                    api_key_env="MOONSHOT_API_KEY",
                    structured_output_mode=ProviderStructuredOutputMode.JSON_OBJECT,
                    extra_body={"thinking": {"type": "enabled"}},
                ),
            )
        ],
        env={"DEEPSEEK_API_KEY": "secret", "MOONSHOT_API_KEY": "secret"},
        provider_config=OpenAICompatibleProviderConfig(
            base_url="https://api.deepseek.com",
            api_key_env="DEEPSEEK_API_KEY",
            structured_output_mode=ProviderStructuredOutputMode.JSON_OBJECT,
            reasoning_effort="high",
            extra_body={"thinking": {"type": "enabled"}},
        ),
        run_chat_completion=fake_chat_completion,
        auto_write=True,
    )

    assert calls == 2
    assert report.diagnostics[0].code == "LLM_RUNTIME_ERROR"
    assert report.gate_results[0].action == "queue"
    assert store.list_relations() == []
    assert len(store.list_review_queue()) == 1


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
        extra_body={"thinking": {"type": "enabled"}},
    )

    assert isinstance(config.default_headers, MappingProxyType)
    assert isinstance(config.extra_body, MappingProxyType)
    with pytest.raises(TypeError):
        config.default_headers["X-Test"] = "no"  # type: ignore[index]
    with pytest.raises(TypeError):
        config.extra_body["thinking"] = {"type": "disabled"}  # type: ignore[index]


def test_provider_registry_contains_static_shortcuts_without_secrets() -> None:
    assert set(PROVIDER_REGISTRY) == {"deepseek", "kimi", "openrouter"}
    assert isinstance(PROVIDER_REGISTRY, MappingProxyType)
    entries = list_provider_entries()
    assert [entry.name for entry in entries] == ["deepseek", "kimi", "openrouter"]
    assert get_provider_entry("deepseek").base_url == "https://api.deepseek.com"
    assert get_provider_entry("DeepSeek").base_url == "https://api.deepseek.com"
    assert (
        get_provider_entry("deepseek").structured_output_mode
        is ProviderStructuredOutputMode.JSON_OBJECT
    )
    assert get_provider_entry("deepseek").reasoning_effort == "high"
    assert get_provider_entry("deepseek").supported_models == (
        "deepseek-v4-pro",
        "deepseek-v4-flash",
    )
    assert get_provider_entry("deepseek").extra_body == {"thinking": {"type": "enabled"}}
    assert (
        get_provider_entry("kimi").structured_output_mode
        is ProviderStructuredOutputMode.JSON_OBJECT
    )
    assert get_provider_entry("kimi").api_key_env == "MOONSHOT_API_KEY"
    assert get_provider_entry("kimi").reasoning_effort is None
    assert get_provider_entry("kimi").extra_body == {"thinking": {"type": "enabled"}}
    assert get_provider_entry("openrouter").default_model is None
    assert (
        get_provider_entry("openrouter").structured_output_mode
        is ProviderStructuredOutputMode.JSON_OBJECT
    )
    assert get_provider_entry("openrouter").reasoning_effort is None
    assert get_provider_entry("openrouter").extra_body == {
        "reasoning": {"effort": "medium", "exclude": True}
    }
    assert get_provider_entry("openrouter").notes
    rendered = ingest_cli._render_provider_list()
    assert (
        "provider\tbase_url\tapi_key_env\tdefault_model\tstructured_output_mode"
        "\tsupported_models\treasoning_effort\tdocs_url\tnotes"
    ) in rendered
    assert "DEEPSEEK_API_KEY" in rendered
    assert "MOONSHOT_API_KEY" in rendered
    assert "OPENROUTER_API_KEY" in rendered
    assert "deepseek-v4-pro\tjson_object\tdeepseek-v4-pro,deepseek-v4-flash\thigh" in rendered
    assert "OpenRouter uses JSON Object mode" in rendered
    assert "requires an explicit model" in rendered
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


async def test_dry_run_gate_queues_approval_without_normalized_implication_confirmation() -> None:
    store = RelationStore()
    candidate = CandidateRelation(
        id="candidate-necessary",
        source="A",
        target="B",
        relation_type="necessary",
        confidence=0.9,
        evidence=[_evidence("B cannot happen without A.")],
    )
    review = _approval(candidate).model_copy(
        update={
            "final_relation_type": "necessary",
            "normalized_implication_supported": None,
        }
    )

    gate_results, approved_relations, _, _ = await run_dry_run_gate(
        candidates=[candidate],
        reviews=[review],
        store=store,
    )

    assert gate_results[0].action == "queue"
    assert gate_results[0].reasons == ["normalized implication support was not confirmed"]
    assert gate_results[0].metadata["normalized_implications"] == {
        "relation_type": "necessary",
        "edges": [{"antecedent": "B", "consequent": "A"}],
    }
    assert approved_relations == []
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
        **kwargs: Any,
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


def test_cli_agent_dry_run_passes_canonicalize_preview(
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
        canonicalize_preview: bool = False,
        **kwargs: Any,
    ) -> IngestionReport:
        assert ingestion_input.evidence
        assert store.list_relations() == []
        assert canonicalize_preview is True
        return IngestionReport(candidates=[_candidate()])

    monkeypatch.setattr(ingest_cli, "run_openai_agents_ingestion", fake_run)
    args = argparse.Namespace(
        input=str(input_path),
        url=[],
        task=None,
        question=None,
        model="gpt-test",
        auto_write=False,
        canonicalize_preview=True,
        min_write_confidence=0.85,
        format="json",
        output=None,
        timeout_seconds=1.0,
        max_url_bytes=1000,
    )

    exit_code = ingest_cli.run_agent_dry_run_cli(args, stdout=StringIO(), stderr=StringIO())

    assert exit_code == 0


def test_cli_agent_dry_run_passes_voting_options(
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
        **kwargs: Any,
    ) -> IngestionReport:
        assert ingestion_input.evidence
        assert store.list_relations() == []
        assert model == "extractor-model"
        assert kwargs["reviewer_models"] == ["reviewer-a", "reviewer-b"]
        assert kwargs["voting_policy"] == openai_agents.ReviewVotingPolicy.MAJORITY
        assert kwargs["high_priority_reviewer_models"] == ["reviewer-a"]
        assert auto_write is False
        assert min_write_confidence == 0.85
        assert provider_config is None
        assert disable_tracing is False
        return IngestionReport(candidates=[_candidate()])

    monkeypatch.setattr(ingest_cli, "run_openai_agents_ingestion", fake_run)
    args = argparse.Namespace(
        input=str(input_path),
        url=[],
        task=None,
        question=None,
        model="extractor-model",
        reviewer_models=["reviewer-a", "reviewer-b"],
        voting_policy="majority",
        high_priority_reviewer_models=["reviewer-a"],
        auto_write=False,
        min_write_confidence=0.85,
        format="json",
        output=None,
        timeout_seconds=1.0,
        max_url_bytes=1000,
    )

    exit_code = ingest_cli.run_agent_dry_run_cli(args, stdout=StringIO(), stderr=StringIO())

    assert exit_code == 0


def test_cli_agent_dry_run_passes_provider_qualified_reviewers(
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
        provider_config: OpenAICompatibleProviderConfig | None = None,
        **kwargs: Any,
    ) -> IngestionReport:
        assert ingestion_input.evidence
        assert store.list_relations() == []
        assert model == "deepseek-v4-pro"
        assert provider_config == OpenAICompatibleProviderConfig(
            base_url="https://api.deepseek.com",
            api_key_env="DEEPSEEK_API_KEY",
            disable_tracing=True,
            structured_output_mode=ProviderStructuredOutputMode.JSON_OBJECT,
            reasoning_effort="high",
            extra_body={"thinking": {"type": "enabled"}},
        )
        reviewer_configs = kwargs["reviewer_configs"]
        assert [config.reviewer_id for config in reviewer_configs] == [
            "kimi:kimi-k2.6",
            "openrouter:qwen/qwen3.7-max",
        ]
        assert [config.model for config in reviewer_configs] == [
            "kimi-k2.6",
            "qwen/qwen3.7-max",
        ]
        assert [config.provider_name for config in reviewer_configs] == ["kimi", "openrouter"]
        assert [
            config.provider_config.api_key_env
            for config in reviewer_configs
            if config.provider_config is not None
        ] == ["MOONSHOT_API_KEY", "OPENROUTER_API_KEY"]
        assert kwargs["high_priority_reviewer_models"] == [
            "legacy-senior",
            "deepseek:deepseek-v4-pro",
        ]
        return IngestionReport(candidates=[_candidate()])

    monkeypatch.setattr(ingest_cli, "run_openai_agents_ingestion", fake_run)
    args = argparse.Namespace(
        input=str(input_path),
        url=[],
        task=None,
        question=None,
        model=None,
        provider="deepseek",
        list_providers=False,
        base_url=None,
        api_key_env=None,
        provider_header=[],
        provider_thinking=None,
        provider_reasoning_effort=None,
        disable_tracing=False,
        reviewer_models=["legacy-reviewer"],
        reviewers=["kimi:kimi-k2.6", "openrouter:qwen/qwen3.7-max"],
        voting_policy="risk_tiered",
        high_priority_reviewer_models=["legacy-senior"],
        high_priority_reviewers=["deepseek:deepseek-v4-pro"],
        auto_write=False,
        min_write_confidence=0.85,
        format="json",
        output=None,
        timeout_seconds=1.0,
        max_url_bytes=1000,
    )

    exit_code = ingest_cli.run_agent_dry_run_cli(args, stdout=StringIO(), stderr=StringIO())

    assert exit_code == 0


def test_cli_agent_dry_run_search_query_adds_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_path = tmp_path / "input.json"
    input_path.write_text(
        json.dumps({"evidence": [_evidence().model_dump(mode="json")]}),
        encoding="utf-8",
    )
    search_record = EvidenceRecord(
        url="https://search.example/source",
        title="Search Source",
        span="Search evidence.",
        source_type="search",
        metadata={"provider": "exa"},
    )

    def fake_search(options: ingest_cli.SearchRetrievalOptions) -> ingest_cli.SearchRetrievalResult:
        assert options.queries == ["search query"]
        assert options.limit == 2
        assert options.include_domains == ["example.com"]
        return ingest_cli.SearchRetrievalResult(
            evidence=[search_record],
            diagnostics=[],
            metadata={"provider": "exa", "accepted_count": 1},
        )

    async def fake_run(
        ingestion_input: IngestionInput,
        *,
        store: Any,
        **kwargs: Any,
    ) -> IngestionReport:
        assert [record.url for record in ingestion_input.evidence] == [
            "https://example.com/source",
            "https://search.example/source",
        ]
        return IngestionReport(candidates=[_candidate()])

    monkeypatch.setattr(ingest_cli, "retrieve_search_evidence", fake_search)
    monkeypatch.setattr(ingest_cli, "run_openai_agents_ingestion", fake_run)
    stdout = StringIO()
    args = argparse.Namespace(
        input=str(input_path),
        url=[],
        task=None,
        question=None,
        model=None,
        search_queries=["search query"],
        search_provider="exa",
        search_limit=2,
        search_timeout_seconds=3.0,
        search_include_domains=["example.com"],
        search_exclude_domains=[],
        search_api_key_env="EXA_API_KEY",
        auto_write=False,
        min_write_confidence=0.85,
        format="json",
        output=None,
        timeout_seconds=1.0,
        max_url_bytes=1000,
    )

    exit_code = ingest_cli.run_agent_dry_run_cli(args, stdout=stdout, stderr=StringIO())
    payload = json.loads(stdout.getvalue())

    assert exit_code == 0
    assert payload["metadata"]["search_retrieval"] == {"provider": "exa", "accepted_count": 1}


def test_cli_agent_dry_run_retrieval_input_adds_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_path = tmp_path / "input.json"
    input_path.write_text(
        json.dumps({"evidence": [_evidence().model_dump(mode="json")]}),
        encoding="utf-8",
    )
    retrieval_path = tmp_path / "retrieval.json"
    retrieval_path.write_text(
        json.dumps(
            {
                "retriever_name": "graph-rag",
                "run_id": "retrieval-run-1",
                "evidence": [
                    {
                        "span": "Retrieved evidence.",
                        "original_url": "https://retrieval.example/source",
                        "source_document_id": "doc-1",
                        "chunk_id": "chunk-1",
                        "score": 0.8,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    async def fake_run(
        ingestion_input: IngestionInput,
        *,
        store: Any,
        **kwargs: Any,
    ) -> IngestionReport:
        assert [record.url for record in ingestion_input.evidence] == [
            "https://example.com/source",
            "https://retrieval.example/source",
        ]
        assert ingestion_input.evidence[1].metadata["retrieval"]["retriever_name"] == "graph-rag"
        return IngestionReport(candidates=[_candidate()])

    monkeypatch.setattr(ingest_cli, "run_openai_agents_ingestion", fake_run)
    stdout = StringIO()
    args = argparse.Namespace(
        input=str(input_path),
        retrieval_input=str(retrieval_path),
        url=[],
        task=None,
        question=None,
        model=None,
        search_queries=[],
        auto_write=False,
        min_write_confidence=0.85,
        format="json",
        output=None,
        timeout_seconds=1.0,
        max_url_bytes=1000,
    )

    exit_code = ingest_cli.run_agent_dry_run_cli(args, stdout=stdout, stderr=StringIO())
    payload = json.loads(stdout.getvalue())

    assert exit_code == 0
    assert payload["metadata"]["external_retrieval"]["retriever_name"] == "graph-rag"
    assert payload["metadata"]["external_retrieval"]["accepted_evidence_count"] == 1


def test_cli_agent_dry_run_retrieval_provenance_error_short_circuits(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    retrieval_path = tmp_path / "retrieval.json"
    retrieval_path.write_text(
        json.dumps({"evidence": [{"span": "Retrieved evidence without source."}]}),
        encoding="utf-8",
    )

    async def fail_run(*args: Any, **kwargs: Any) -> IngestionReport:
        raise AssertionError("invalid retrieval evidence must not call ingestion runtime")

    monkeypatch.setattr(ingest_cli, "run_openai_agents_ingestion", fail_run)
    stdout = StringIO()
    args = argparse.Namespace(
        input=None,
        retrieval_input=str(retrieval_path),
        url=[],
        task=None,
        question=None,
        model=None,
        search_queries=[],
        auto_write=True,
        min_write_confidence=0.85,
        format="json",
        output=None,
        timeout_seconds=1.0,
        max_url_bytes=1000,
    )

    exit_code = ingest_cli.run_agent_dry_run_cli(args, stdout=stdout, stderr=StringIO())
    payload = json.loads(stdout.getvalue())

    assert exit_code == 0
    assert payload["mode"] == "write"
    assert payload["written_relation_ids"] == []
    assert payload["diagnostics"][0]["code"] == "RETRIEVAL_PROVENANCE_MISSING"
    assert payload["metadata"]["external_retrieval"]["diagnostic_count"] == 1


def test_cli_agent_dry_run_retrieval_input_size_limit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    retrieval_path = tmp_path / "retrieval.json"
    retrieval_path.write_text(json.dumps({"evidence": []}), encoding="utf-8")
    monkeypatch.setattr(ingest_cli, "MAX_EXTERNAL_RETRIEVAL_INPUT_BYTES", 8)
    stderr = StringIO()
    args = argparse.Namespace(
        input=None,
        retrieval_input=str(retrieval_path),
        url=[],
        task=None,
        question=None,
        model=None,
        search_queries=[],
        auto_write=False,
        min_write_confidence=0.85,
        format="json",
        output=None,
        timeout_seconds=1.0,
        max_url_bytes=1000,
    )

    exit_code = ingest_cli.run_agent_dry_run_cli(args, stdout=StringIO(), stderr=stderr)

    assert exit_code == 2
    assert "retrieval input JSON exceeds" in stderr.getvalue()


def test_cli_agent_dry_run_search_failure_short_circuits_auto_write(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_path = tmp_path / "input.json"
    input_path.write_text(
        json.dumps({"evidence": [_evidence().model_dump(mode="json")]}),
        encoding="utf-8",
    )

    def fake_search(options: ingest_cli.SearchRetrievalOptions) -> ingest_cli.SearchRetrievalResult:
        return ingest_cli.SearchRetrievalResult(
            evidence=[],
            diagnostics=[
                Diagnostic(
                    level="error",
                    code="EXA_SEARCH_REQUEST_FAILED",
                    message="Exa search request failed: TimeoutError",
                )
            ],
            metadata={"provider": "exa", "diagnostic_count": 1},
        )

    async def fail_run(*args: Any, **kwargs: Any) -> IngestionReport:
        raise AssertionError("search failure must not call ingestion runtime")

    monkeypatch.setattr(ingest_cli, "retrieve_search_evidence", fake_search)
    monkeypatch.setattr(ingest_cli, "run_openai_agents_ingestion", fail_run)
    stdout = StringIO()
    args = argparse.Namespace(
        input=str(input_path),
        url=[],
        task=None,
        question=None,
        model=None,
        search_queries=["search query"],
        search_provider="exa",
        search_limit=2,
        search_timeout_seconds=3.0,
        search_include_domains=[],
        search_exclude_domains=[],
        search_api_key_env="EXA_API_KEY",
        auto_write=True,
        min_write_confidence=0.85,
        format="json",
        output=None,
        timeout_seconds=1.0,
        max_url_bytes=1000,
    )

    exit_code = ingest_cli.run_agent_dry_run_cli(args, stdout=stdout, stderr=StringIO())
    payload = json.loads(stdout.getvalue())

    assert exit_code == 0
    assert payload["mode"] == "write"
    assert payload["written_relation_ids"] == []
    assert payload["diagnostics"][0]["code"] == "EXA_SEARCH_REQUEST_FAILED"
    assert payload["metadata"]["auto_write_requested"] is True


def test_cli_agent_dry_run_empty_search_results_return_diagnostic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_search(options: ingest_cli.SearchRetrievalOptions) -> ingest_cli.SearchRetrievalResult:
        return ingest_cli.SearchRetrievalResult(
            evidence=[],
            diagnostics=[
                Diagnostic(
                    level="warning",
                    code="SEARCH_NO_ACCEPTED_RESULTS",
                    message="no Exa search results accepted for query: search query",
                )
            ],
            metadata={"provider": "exa", "accepted_count": 0},
        )

    async def fail_run(*args: Any, **kwargs: Any) -> IngestionReport:
        raise AssertionError("empty evidence must not call ingestion runtime")

    monkeypatch.setattr(ingest_cli, "retrieve_search_evidence", fake_search)
    monkeypatch.setattr(ingest_cli, "run_openai_agents_ingestion", fail_run)
    stdout = StringIO()
    args = argparse.Namespace(
        input=None,
        url=[],
        task=None,
        question=None,
        model=None,
        search_queries=["search query"],
        search_provider="exa",
        search_limit=2,
        search_timeout_seconds=3.0,
        search_include_domains=[],
        search_exclude_domains=[],
        search_api_key_env="EXA_API_KEY",
        auto_write=False,
        min_write_confidence=0.85,
        format="json",
        output=None,
        timeout_seconds=1.0,
        max_url_bytes=1000,
    )

    exit_code = ingest_cli.run_agent_dry_run_cli(args, stdout=stdout, stderr=StringIO())
    payload = json.loads(stdout.getvalue())

    assert exit_code == 0
    assert [diagnostic["code"] for diagnostic in payload["diagnostics"]] == [
        "SEARCH_NO_ACCEPTED_RESULTS",
        "INGESTION_EVIDENCE_MISSING",
    ]


def test_cli_agent_dry_run_crawl_adds_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_path = tmp_path / "input.json"
    input_path.write_text(
        json.dumps({"evidence": [_evidence().model_dump(mode="json")]}),
        encoding="utf-8",
    )
    crawled = EvidenceRecord(
        url="https://example.com/crawled",
        span="Crawled evidence.",
        source_type="crawl",
        metadata={"crawl_depth": 0},
    )

    def fake_crawl(options: ingest_cli.CrawlOptions) -> ingest_cli.CrawlResult:
        assert options.seed_urls == ["https://example.com/seed"]
        assert options.max_depth == 1
        assert options.allow_domains == ["docs.example.com"]
        return ingest_cli.CrawlResult(
            evidence=[crawled],
            diagnostics=[
                Diagnostic(
                    level="info",
                    code="CRAWL_URL_DUPLICATE",
                    message="skipped duplicate",
                )
            ],
            metadata={"accepted_count": 1},
        )

    def fail_fetch(*args: Any, **kwargs: Any) -> list[EvidenceRecord]:
        raise AssertionError("--crawl must not also run explicit URL fetch")

    async def fake_run(
        ingestion_input: IngestionInput,
        *,
        store: Any,
        **kwargs: Any,
    ) -> IngestionReport:
        assert [record.url for record in ingestion_input.evidence] == [
            "https://example.com/source",
            "https://example.com/crawled",
        ]
        assert kwargs["auto_write"] is True
        return IngestionReport(candidates=[_candidate()])

    monkeypatch.setattr(ingest_cli, "crawl_url_evidence", fake_crawl)
    monkeypatch.setattr(ingest_cli, "fetch_url_evidence_many", fail_fetch)
    monkeypatch.setattr(ingest_cli, "run_openai_agents_ingestion", fake_run)
    stdout = StringIO()
    args = argparse.Namespace(
        input=str(input_path),
        url=["https://example.com/seed"],
        task=None,
        question=None,
        model=None,
        search_queries=[],
        crawl=True,
        crawl_max_depth=1,
        crawl_max_pages=10,
        crawl_max_page_bytes=1000,
        crawl_max_total_bytes=5000,
        crawl_timeout_seconds=3.0,
        crawl_allow_domains=["docs.example.com"],
        auto_write=True,
        min_write_confidence=0.85,
        format="json",
        output=None,
        timeout_seconds=1.0,
        max_url_bytes=1000,
    )

    exit_code = ingest_cli.run_agent_dry_run_cli(args, stdout=stdout, stderr=StringIO())
    payload = json.loads(stdout.getvalue())

    assert exit_code == 0
    assert payload["metadata"]["crawl_retrieval"] == {"accepted_count": 1}
    assert payload["diagnostics"][0]["code"] == "CRAWL_URL_DUPLICATE"


def test_cli_agent_dry_run_crawl_requires_seed() -> None:
    stderr = StringIO()
    args = argparse.Namespace(
        input=None,
        url=[],
        task=None,
        question=None,
        model=None,
        search_queries=[],
        crawl=True,
        auto_write=False,
        min_write_confidence=0.85,
        format="json",
        output=None,
        timeout_seconds=1.0,
        max_url_bytes=1000,
    )

    exit_code = ingest_cli.run_agent_dry_run_cli(args, stdout=StringIO(), stderr=stderr)

    assert exit_code == 2
    assert "--crawl requires at least one --url or input urls" in stderr.getvalue()


def test_cli_agent_dry_run_empty_crawl_results_return_diagnostic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_crawl(options: ingest_cli.CrawlOptions) -> ingest_cli.CrawlResult:
        return ingest_cli.CrawlResult(
            evidence=[],
            diagnostics=[
                Diagnostic(
                    level="warning",
                    code="CRAWL_FETCH_FAILED",
                    message="could not fetch crawl URL",
                )
            ],
            metadata={"accepted_count": 0},
        )

    async def fail_run(*args: Any, **kwargs: Any) -> IngestionReport:
        raise AssertionError("empty crawl evidence must not call ingestion runtime")

    monkeypatch.setattr(ingest_cli, "crawl_url_evidence", fake_crawl)
    monkeypatch.setattr(ingest_cli, "run_openai_agents_ingestion", fail_run)
    stdout = StringIO()
    args = argparse.Namespace(
        input=None,
        url=["https://example.com/seed"],
        task=None,
        question=None,
        model=None,
        search_queries=[],
        crawl=True,
        crawl_max_depth=1,
        crawl_max_pages=10,
        crawl_max_page_bytes=1000,
        crawl_max_total_bytes=5000,
        crawl_timeout_seconds=3.0,
        crawl_allow_domains=[],
        auto_write=False,
        min_write_confidence=0.85,
        format="json",
        output=None,
        timeout_seconds=1.0,
        max_url_bytes=1000,
    )

    exit_code = ingest_cli.run_agent_dry_run_cli(args, stdout=stdout, stderr=StringIO())
    payload = json.loads(stdout.getvalue())

    assert exit_code == 0
    assert payload["metadata"]["crawl_retrieval"] == {"accepted_count": 0}
    assert [diagnostic["code"] for diagnostic in payload["diagnostics"]] == [
        "CRAWL_FETCH_FAILED",
        "INGESTION_EVIDENCE_MISSING",
    ]


def test_cli_retrieval_validate_uses_existing_candidate_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    retrieval_path = tmp_path / "retrieval.json"
    candidate = _candidate()
    retrieval_path.write_text(
        json.dumps(
            {
                "retriever_name": "graph-rag",
                "candidates": [
                    {
                        "candidate": candidate.model_dump(mode="json"),
                        "source_document_id": "doc-1",
                        "chunk_id": "chunk-1",
                    }
                ],
                "reviews": [_approval(candidate).model_dump(mode="json")],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(ingest_cli, "create_relation_store", lambda config: RelationStore())
    stdout = StringIO()
    args = argparse.Namespace(
        retrieval_command="validate",
        input=str(retrieval_path),
        min_write_confidence=0.85,
        voting_policy="risk_tiered",
        high_priority_reviewer_models=[],
        format="json",
    )

    exit_code = ingest_cli.run_retrieval_cli(args, stdout=stdout, stderr=StringIO())
    payload = json.loads(stdout.getvalue())

    assert exit_code == 0
    assert payload["status"] == "ok"
    assert payload["persisted"] is False
    assert payload["approved_count"] == 1
    assert payload["external_retrieval"]["candidate_count"] == 1


def test_cli_retrieval_validate_queues_missing_candidate_provenance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    retrieval_path = tmp_path / "retrieval.json"
    candidate = _candidate()
    retrieval_path.write_text(
        json.dumps(
            {
                "retriever_name": "graph-rag",
                "candidates": [{"candidate": candidate.model_dump(mode="json")}],
                "reviews": [_approval(candidate).model_dump(mode="json")],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(ingest_cli, "create_relation_store", lambda config: RelationStore())
    stdout = StringIO()
    args = argparse.Namespace(
        retrieval_command="validate",
        input=str(retrieval_path),
        min_write_confidence=0.85,
        voting_policy="risk_tiered",
        high_priority_reviewer_models=[],
        format="json",
    )

    exit_code = ingest_cli.run_retrieval_cli(args, stdout=stdout, stderr=StringIO())
    payload = json.loads(stdout.getvalue())

    assert exit_code == 0
    assert payload["status"] == "warning"
    assert payload["approved_count"] == 0
    assert payload["approved_relations"] == []
    assert payload["queued_count"] == 1
    assert payload["gate_results"][0]["action"] == "queue"
    assert payload["diagnostics"][-1]["code"] == "RETRIEVAL_PROVENANCE_MISSING"


def test_cli_retrieval_validate_reports_orphan_review(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    retrieval_path = tmp_path / "retrieval.json"
    candidate = _candidate()
    retrieval_path.write_text(
        json.dumps(
            {
                "retriever_name": "graph-rag",
                "candidates": [
                    {
                        "candidate": candidate.model_dump(mode="json"),
                        "source_document_id": "doc-1",
                    }
                ],
                "reviews": [
                    {
                        "candidate_id": "missing-candidate",
                        "decision": "needs_human",
                        "reasons": ["No matching candidate."],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(ingest_cli, "create_relation_store", lambda config: RelationStore())
    stdout = StringIO()
    args = argparse.Namespace(
        retrieval_command="validate",
        input=str(retrieval_path),
        min_write_confidence=0.85,
        voting_policy="risk_tiered",
        high_priority_reviewer_models=[],
        format="json",
    )

    exit_code = ingest_cli.run_retrieval_cli(args, stdout=stdout, stderr=StringIO())
    payload = json.loads(stdout.getvalue())

    assert exit_code == 0
    assert payload["status"] == "warning"
    assert "RETRIEVAL_ORPHAN_REVIEW" in [
        diagnostic["code"] for diagnostic in payload["diagnostics"]
    ]
    assert payload["external_retrieval"]["orphan_review_candidate_ids"] == ["missing-candidate"]


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
        **kwargs: Any,
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
        provider_thinking=None,
        provider_reasoning_effort=None,
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
        **kwargs: Any,
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
        **kwargs: Any,
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
        provider_thinking=None,
        provider_reasoning_effort=None,
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
    assert (
        "deepseek\thttps://api.deepseek.com\tDEEPSEEK_API_KEY\tdeepseek-v4-pro"
        "\tjson_object\tdeepseek-v4-pro,deepseek-v4-flash\thigh"
    ) in rendered
    assert (
        "kimi\thttps://api.moonshot.cn/v1\tMOONSHOT_API_KEY\tkimi-k2.6\tjson_object\t-\t-"
    ) in rendered
    assert (
        "openrouter\thttps://openrouter.ai/api/v1\tOPENROUTER_API_KEY\t-\tjson_object\t-\t-"
    ) in rendered
    assert "OpenRouter uses JSON Object mode" in rendered
    assert "requires an explicit model" in rendered
    assert "secret" not in rendered.lower()


def test_cli_progress_writes_to_stderr_without_polluting_stdout(
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
        progress_callback: Any = None,
        **kwargs: Any,
    ) -> IngestionReport:
        assert ingestion_input.evidence
        assert progress_callback is not None
        progress_callback({"event": "started", "stage": "extractor", "label": "extractor"})
        progress_callback(
            {
                "event": "done",
                "stage": "extractor",
                "label": "extractor",
                "duration_ms": 1200,
                "candidate_count": 1,
            }
        )
        return IngestionReport(candidates=[_candidate()])

    monkeypatch.setattr(ingest_cli, "run_openai_agents_ingestion", fake_run)
    stdout = StringIO()
    stderr = StringIO()
    args = argparse.Namespace(
        input=str(input_path),
        retrieval_input=None,
        url=[],
        task=None,
        question=None,
        model=None,
        reviewer_models=[],
        reviewers=[],
        voting_policy="risk_tiered",
        high_priority_reviewer_models=[],
        high_priority_reviewers=[],
        provider=None,
        list_providers=False,
        base_url=None,
        api_key_env=None,
        provider_header=[],
        provider_thinking=None,
        provider_reasoning_effort=None,
        extractor_timeout_seconds=180,
        high_priority_reviewer_timeout_seconds=180,
        reviewer_timeout_seconds=120,
        extractor_max_tokens=4096,
        reviewer_max_tokens=2048,
        progress="auto",
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
    assert json.loads(stdout.getvalue())["candidates"]
    assert "[extractor] extractor started" in stderr.getvalue()
    assert "[extractor] extractor done in 1.2s candidates=1" in stderr.getvalue()


def test_parse_provider_qualified_reviewer_specs() -> None:
    assert ingest_cli._parse_provider_reviewer_spec("deepseek:deepseek-v4-pro")[1:] == (
        "deepseek-v4-pro",
        "deepseek:deepseek-v4-pro",
    )
    assert ingest_cli._parse_provider_reviewer_spec("kimi:kimi-k2.6")[1:] == (
        "kimi-k2.6",
        "kimi:kimi-k2.6",
    )
    assert ingest_cli._parse_provider_reviewer_spec("openrouter:qwen/qwen3.7-max")[1:] == (
        "qwen/qwen3.7-max",
        "openrouter:qwen/qwen3.7-max",
    )
    assert ingest_cli._parse_provider_reviewer_spec("openrouter:test:model:v1")[1:] == (
        "test:model:v1",
        "openrouter:test:model:v1",
    )
    with pytest.raises(ValueError, match="PROVIDER:MODEL"):
        ingest_cli._parse_provider_reviewer_spec("deepseek-v4-pro")
    with pytest.raises(ValueError, match="provider and model"):
        ingest_cli._parse_provider_reviewer_spec("deepseek:")
    with pytest.raises(ValueError, match="unknown provider"):
        ingest_cli._parse_provider_reviewer_spec("unknown:model")


@pytest.mark.parametrize(
    (
        "provider_name",
        "expected_base_url",
        "expected_api_key_env",
        "expected_model",
        "expected_output_mode",
        "expected_reasoning_effort",
        "expected_extra_body",
    ),
    [
        (
            "deepseek",
            "https://api.deepseek.com",
            "DEEPSEEK_API_KEY",
            "deepseek-v4-pro",
            ProviderStructuredOutputMode.JSON_OBJECT,
            "high",
            {"thinking": {"type": "enabled"}},
        ),
        (
            "kimi",
            "https://api.moonshot.cn/v1",
            "MOONSHOT_API_KEY",
            "kimi-k2.6",
            ProviderStructuredOutputMode.JSON_OBJECT,
            None,
            {"thinking": {"type": "enabled"}},
        ),
    ],
)
def test_cli_provider_shortcuts_fill_config_and_default_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    provider_name: str,
    expected_base_url: str,
    expected_api_key_env: str,
    expected_model: str,
    expected_output_mode: ProviderStructuredOutputMode,
    expected_reasoning_effort: str | None,
    expected_extra_body: dict[str, Any],
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
        **kwargs: Any,
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
            structured_output_mode=expected_output_mode,
            reasoning_effort=expected_reasoning_effort,
            extra_body=expected_extra_body,
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
        provider_thinking=None,
        provider_reasoning_effort=None,
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
        **kwargs: Any,
    ) -> IngestionReport:
        assert ingestion_input.evidence
        assert store.list_relations() == []
        assert model == "custom-model"
        assert provider_config == OpenAICompatibleProviderConfig(
            base_url="https://custom.example/v1",
            api_key_env="CUSTOM_API_KEY",
            default_headers={"X-Test": "yes"},
            disable_tracing=True,
            structured_output_mode=ProviderStructuredOutputMode.JSON_OBJECT,
            reasoning_effort="high",
            extra_body={"thinking": {"type": "enabled"}},
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
        provider_thinking=None,
        provider_reasoning_effort=None,
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


def test_cli_deepseek_provider_thinking_overrides_registry(
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
        provider_config: OpenAICompatibleProviderConfig | None = None,
        **kwargs: Any,
    ) -> IngestionReport:
        assert ingestion_input.evidence
        assert model == "deepseek-v4-pro"
        assert provider_config == OpenAICompatibleProviderConfig(
            base_url="https://api.deepseek.com",
            api_key_env="DEEPSEEK_API_KEY",
            disable_tracing=True,
            structured_output_mode=ProviderStructuredOutputMode.JSON_OBJECT,
            reasoning_effort="max",
            extra_body={"thinking": {"type": "disabled"}},
        )
        return IngestionReport(candidates=[_candidate()])

    monkeypatch.setattr(ingest_cli, "run_openai_agents_ingestion", fake_run)
    args = argparse.Namespace(
        input=str(input_path),
        url=[],
        task=None,
        question=None,
        model=None,
        provider="deepseek",
        list_providers=False,
        base_url=None,
        api_key_env=None,
        provider_header=[],
        provider_thinking="disabled",
        provider_reasoning_effort="max",
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


def test_cli_kimi_provider_thinking_overrides_registry(
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
        provider_config: OpenAICompatibleProviderConfig | None = None,
        **kwargs: Any,
    ) -> IngestionReport:
        assert ingestion_input.evidence
        assert model == "kimi-k2.6"
        assert provider_config == OpenAICompatibleProviderConfig(
            base_url="https://api.moonshot.cn/v1",
            api_key_env="MOONSHOT_API_KEY",
            disable_tracing=True,
            structured_output_mode=ProviderStructuredOutputMode.JSON_OBJECT,
            extra_body={"thinking": {"type": "disabled"}},
        )
        return IngestionReport(candidates=[_candidate()])

    monkeypatch.setattr(ingest_cli, "run_openai_agents_ingestion", fake_run)
    args = argparse.Namespace(
        input=str(input_path),
        url=[],
        task=None,
        question=None,
        model=None,
        provider="kimi",
        list_providers=False,
        base_url=None,
        api_key_env=None,
        provider_header=[],
        provider_thinking="disabled",
        provider_reasoning_effort=None,
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


def test_cli_deepseek_provider_accepts_flash_model(
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
        provider_config: OpenAICompatibleProviderConfig | None = None,
        **kwargs: Any,
    ) -> IngestionReport:
        assert ingestion_input.evidence
        assert model == "deepseek-v4-flash"
        assert provider_config == OpenAICompatibleProviderConfig(
            base_url="https://api.deepseek.com",
            api_key_env="DEEPSEEK_API_KEY",
            disable_tracing=True,
            structured_output_mode=ProviderStructuredOutputMode.JSON_OBJECT,
            reasoning_effort="high",
            extra_body={"thinking": {"type": "enabled"}},
        )
        return IngestionReport(candidates=[_candidate()])

    monkeypatch.setattr(ingest_cli, "run_openai_agents_ingestion", fake_run)
    args = argparse.Namespace(
        input=str(input_path),
        url=[],
        task=None,
        question=None,
        model="deepseek-v4-flash",
        provider="deepseek",
        list_providers=False,
        base_url=None,
        api_key_env=None,
        provider_header=[],
        provider_thinking=None,
        provider_reasoning_effort=None,
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
        **kwargs: Any,
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
            structured_output_mode=ProviderStructuredOutputMode.JSON_OBJECT,
            extra_body={"reasoning": {"effort": "medium", "exclude": True}},
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
        provider_thinking=None,
        provider_reasoning_effort=None,
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
            provider_thinking=None,
            provider_reasoning_effort=None,
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
            provider_thinking=None,
            provider_reasoning_effort=None,
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
        ({"provider": "openrouter", "provider_thinking": "disabled"}, "not supported"),
        ({"provider": "openrouter", "provider_reasoning_effort": "high"}, "not supported"),
        ({"provider": "kimi", "provider_reasoning_effort": "max"}, "not supported"),
        ({"reviewers": ["deepseek-v4-pro"]}, "PROVIDER:MODEL"),
        ({"reviewers": ["unknown:model"]}, "unknown provider"),
        ({"high_priority_reviewers": [":model"]}, "provider and model"),
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
        "provider_thinking": None,
        "provider_reasoning_effort": None,
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
