"""RED tests for Agent SDK handoff chain (issue #95).

All tests should FAIL until the handoff implementation is merged.

Expected changes:
  - _build_agent() gains an optional `handoffs` parameter
  - Default scenario (no reviewer_models): extractor handoffs to reviewer
    inside a single Runner.run(), returning ReviewDecisionBatch
  - Multi-reviewer scenario: unchanged manual orchestration
  - _run_agent() result may be ReviewDecisionBatch (handoff chain output)
  - JSON_OBJECT fallback for non-Agent-SDK providers is preserved
"""

from __future__ import annotations

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
)
from nesy_reasoning_mcp.auto_ingest.openai_agents import (
    CandidateRelationBatch,
    OpenAICompatibleProviderConfig,
    ReviewDecisionBatch,
    _build_agent,
    _coerce_review_batch,
    run_openai_agents_dry_run,
)
from nesy_reasoning_mcp.auto_ingest.providers import ProviderStructuredOutputMode
from nesy_reasoning_mcp.schemas import RelationType
from nesy_reasoning_mcp.store import RelationStore

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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
        relation_type=RelationType.SUFFICIENT,
        confidence=0.9,
        evidence=[_evidence()],
    )


def _approval(candidate: CandidateRelation) -> ReviewDecision:
    return ReviewDecision(
        candidate_id=candidate.id,
        decision=ReviewDecisionValue.APPROVE,
        final_relation_type=RelationType.SUFFICIENT,
        final_confidence=0.9,
        normalized_implication_supported=True,
        reasons=["Evidence is explicit."],
    )


# ---------------------------------------------------------------------------
# 1. _build_agent accepts handoffs parameter
# ---------------------------------------------------------------------------


class TestBuildAgentHandoffsParam:
    """_build_agent should accept an optional `handoffs` list."""

    def test_build_agent_accepts_handoffs_empty_list(self) -> None:
        """Passing handoffs=[] must not raise."""
        agent = _build_agent(
            name="test extractor",
            instructions="extract relations",
            output_type=CandidateRelationBatch,
            model=None,
            handoffs=[],
        )
        assert agent.name == "test extractor"

    def test_build_agent_accepts_handoffs_with_agent(self) -> None:
        """Passing handoffs=[Agent] must wire up the handoff."""
        from agents import Agent

        reviewer = Agent(
            name="default reviewer",
            instructions="review candidates",
        )
        agent = _build_agent(
            name="extractor",
            instructions="extract",
            output_type=CandidateRelationBatch,
            model=None,
            handoffs=[reviewer],
        )
        assert len(agent.handoffs) == 1

    def test_build_agent_no_handoffs_backward_compat(self) -> None:
        """Calling without handoffs kwarg must still work (backward compat)."""
        agent = _build_agent(
            name="legacy extractor",
            instructions="extract",
            output_type=CandidateRelationBatch,
            model=None,
        )
        assert agent.name == "legacy extractor"


# ---------------------------------------------------------------------------
# 2. Default scenario: handoff chain single Runner.run()
# ---------------------------------------------------------------------------


class TestHandoffDefaultScenario:
    """Default scenario (no reviewer_models): extractor handoffs to
    reviewer inside a single Runner.run(), returning ReviewDecisionBatch."""

    @pytest.mark.anyio
    async def test_default_scenario_single_run_agent_call(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Default scenario should call _run_agent once (handoff chain
        handles extract->review internally), not twice."""
        call_count = 0

        async def fake_run_agent(agent: Any, prompt: str, **kwargs: Any) -> Any:
            nonlocal call_count
            call_count += 1
            # Handoff chain returns ReviewDecisionBatch from the reviewer
            return ReviewDecisionBatch(reviews=[_approval(_candidate())])

        monkeypatch.setattr(openai_agents, "_run_agent", fake_run_agent)

        def fake_build_agent(**kw: Any) -> SimpleNamespace:
            return SimpleNamespace(output_type=kw["output_type"])

        monkeypatch.setattr(openai_agents, "_build_agent", fake_build_agent)

        report = await run_openai_agents_dry_run(
            IngestionInput(evidence=[_evidence()], task="extract dependencies"),
            store=RelationStore(),
            env={"OPENAI_API_KEY": "test"},
        )
        assert call_count == 1
        assert report.mode == "dry_run"

    @pytest.mark.anyio
    async def test_default_scenario_returns_review_decisions(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When handoff is active, the pipeline should receive
        ReviewDecisionBatch (not CandidateRelationBatch) from the
        single Runner.run() call."""
        candidate = _candidate()
        approval = _approval(candidate)

        async def fake_run_agent(agent: Any, prompt: str, **kwargs: Any) -> Any:
            return ReviewDecisionBatch(reviews=[approval])

        monkeypatch.setattr(openai_agents, "_run_agent", fake_run_agent)
        monkeypatch.setattr(
            openai_agents,
            "_build_agent",
            lambda **kw: SimpleNamespace(output_type=kw["output_type"]),
        )

        report = await run_openai_agents_dry_run(
            IngestionInput(evidence=[_evidence()], task="extract dependencies"),
            store=RelationStore(),
            env={"OPENAI_API_KEY": "test"},
        )
        # Reviews from the handoff chain should appear in the report
        assert len(report.reviews) >= 1
        assert report.reviews[0].decision == ReviewDecisionValue.APPROVE


# ---------------------------------------------------------------------------
# 3. Structured JSON output coercion still works
# ---------------------------------------------------------------------------


class TestHandoffCoercion:
    """Handoff results should coerce to the expected Pydantic types."""

    def test_coerce_review_batch_from_dict(self) -> None:
        """Dict output from handoff chain coerces to ReviewDecisionBatch."""
        approval = _approval(_candidate())
        raw = {"reviews": [approval.model_dump(mode="json")]}
        batch = _coerce_review_batch(raw)
        assert isinstance(batch, ReviewDecisionBatch)
        assert len(batch.reviews) == 1

    def test_coerce_review_batch_from_list(self) -> None:
        """List output coerces to ReviewDecisionBatch."""
        approval = _approval(_candidate())
        raw = [approval.model_dump(mode="json")]
        batch = _coerce_review_batch(raw)
        assert isinstance(batch, ReviewDecisionBatch)
        assert len(batch.reviews) == 1

    def test_coerce_review_batch_passthrough(self) -> None:
        """Already-typed ReviewDecisionBatch passes through unchanged."""
        batch = ReviewDecisionBatch(reviews=[_approval(_candidate())])
        assert _coerce_review_batch(batch) is batch


# ---------------------------------------------------------------------------
# 4. Multi-reviewer scenario: manual orchestration preserved
# ---------------------------------------------------------------------------


class TestMultiReviewerNoHandoff:
    """When reviewer_models is set, pipeline must still use manual
    orchestration (multiple _run_agent calls), not handoff."""

    @pytest.mark.anyio
    async def test_multi_reviewer_calls_run_agent_multiple_times(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """With reviewer_models=['m1','m2'], _run_agent should be called
        for extraction + once per reviewer (3 total), not once via handoff."""
        call_count = 0
        candidate = _candidate()
        approval = _approval(candidate)

        async def counting_run_agent(agent: Any, prompt: str, **kwargs: Any) -> Any:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return CandidateRelationBatch(candidates=[candidate])
            return ReviewDecisionBatch(reviews=[approval])

        monkeypatch.setattr(openai_agents, "_run_agent", counting_run_agent)
        monkeypatch.setattr(
            openai_agents,
            "_build_agent",
            lambda **kw: SimpleNamespace(output_type=kw["output_type"]),
        )

        report = await run_openai_agents_dry_run(
            IngestionInput(evidence=[_evidence()], task="extract dependencies"),
            store=RelationStore(),
            env={"OPENAI_API_KEY": "test"},
            reviewer_models=["m1", "m2"],
        )
        # 1 extraction + 2 reviewers = 3 calls
        assert call_count == 3
        assert report.mode == "dry_run"


# ---------------------------------------------------------------------------
# 5. Error propagation in handoff chain
# ---------------------------------------------------------------------------


class TestHandoffErrorPropagation:
    """Errors inside the handoff chain should surface correctly."""

    @pytest.mark.anyio
    async def test_handoff_chain_error_propagates(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """If Runner.run raises inside the handoff chain, the error
        should propagate as a diagnostic (not silently swallowed)."""

        async def failing_run_agent(agent: Any, prompt: str, **kwargs: Any) -> Any:
            raise RuntimeError("handoff chain LLM failure")

        monkeypatch.setattr(openai_agents, "_run_agent", failing_run_agent)
        monkeypatch.setattr(
            openai_agents,
            "_build_agent",
            lambda **kw: SimpleNamespace(output_type=kw["output_type"]),
        )

        report = await run_openai_agents_dry_run(
            IngestionInput(evidence=[_evidence()], task="extract dependencies"),
            store=RelationStore(),
            env={"OPENAI_API_KEY": "test"},
        )
        # Pipeline should capture the error as a diagnostic, not crash
        assert any(d.level == "error" for d in report.diagnostics), (
            f"Expected error diagnostic, got: {report.diagnostics}"
        )

    @pytest.mark.anyio
    async def test_handoff_chain_malformed_output_handled(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """If the handoff chain returns an unrecognizable type,
        coercion should raise a validation error."""

        async def bad_output_run_agent(agent: Any, prompt: str, **kwargs: Any) -> Any:
            return {"unexpected_key": "garbage"}

        monkeypatch.setattr(openai_agents, "_run_agent", bad_output_run_agent)
        monkeypatch.setattr(
            openai_agents,
            "_build_agent",
            lambda **kw: SimpleNamespace(output_type=kw["output_type"]),
        )

        # Default scenario: _run_agent returns ReviewDecisionBatch from handoff.
        # If it returns garbage, _coerce_review_batch should fail.
        with pytest.raises((ValueError, TypeError)):
            _coerce_review_batch({"unexpected_key": "garbage"})


# ---------------------------------------------------------------------------
# 6. JSON_OBJECT fallback preserved
# ---------------------------------------------------------------------------


class TestJsonObjectFallback:
    """Providers that don't support Agent SDK should still use the
    JSON_OBJECT completion path (no handoff)."""

    @pytest.mark.anyio
    async def test_json_object_provider_skips_handoff(self) -> None:
        """When provider_config uses JSON_OBJECT mode, the pipeline
        should use _run_json_object_completion, not Agent SDK handoff."""
        provider_config = OpenAICompatibleProviderConfig(
            base_url="https://api.example.com/v1",
            api_key_env="TEST_KEY",
            structured_output_mode=ProviderStructuredOutputMode.JSON_OBJECT,
        )

        approval = _approval(_candidate())

        call_log: list[str] = []

        async def fake_chat_completion(
            *,
            model: str,
            messages: list[Any],
            response_format: Any,
            **kwargs: Any,
        ) -> Any:
            call_log.append("chat_completion")
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(
                            content=ReviewDecisionBatch(reviews=[approval]).model_dump_json()
                        )
                    )
                ]
            )

        report = await run_openai_agents_dry_run(
            IngestionInput(evidence=[_evidence()], task="extract dependencies"),
            store=RelationStore(),
            env={"TEST_KEY": "sk-test", "OPENAI_DEFAULT_MODEL": "gpt-4o"},
            provider_config=provider_config,
            run_chat_completion=fake_chat_completion,
        )
        # Should have used chat completion, not Agent SDK Runner.run
        assert "chat_completion" in call_log
        assert report.mode == "dry_run"
