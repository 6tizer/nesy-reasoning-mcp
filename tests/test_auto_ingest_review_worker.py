from pathlib import Path
from types import SimpleNamespace
from typing import Any

import anyio
import pytest

from nesy_reasoning_mcp.auto_ingest import (
    CandidateRelation,
    ConversationTurnJobStatus,
    EvidenceRecord,
    GateAction,
    GateResult,
    IngestionWorkerConfig,
    ReviewDecision,
    ReviewDecisionValue,
    ReviewQueueRecord,
    ReviewQueueStatus,
    ReviewVotingPolicy,
    openai_agents,
    process_ingestion_queue_once,
    run_ingestion_worker,
)
from nesy_reasoning_mcp.auto_ingest.openai_agents import (
    OpenAICompatibleProviderConfig,
    ReviewerModelConfig,
)
from nesy_reasoning_mcp.auto_ingest.providers import ProviderStructuredOutputMode
from nesy_reasoning_mcp.auto_ingest.review_worker import (
    ReviewWorkerConfig,
    process_review_queue_once,
)
from nesy_reasoning_mcp.schemas import RelationType
from nesy_reasoning_mcp.store import RelationStore


def _candidate(
    candidate_id: str = "cand-1",
    *,
    source: str = "A",
    target: str = "B",
) -> CandidateRelation:
    return CandidateRelation(
        id=candidate_id,
        source=source,
        target=target,
        relation_type=RelationType.SUFFICIENT,
        confidence=0.95,
        evidence=[EvidenceRecord(url="conversation://test", span=f"{source} enables {target}")],
    )


def _provider_config() -> OpenAICompatibleProviderConfig:
    return OpenAICompatibleProviderConfig(
        base_url="https://example.com/v1",
        api_key_env="EXAMPLE_API_KEY",
        structured_output_mode=ProviderStructuredOutputMode.JSON_OBJECT,
    )


def _worker_config(**kwargs: Any) -> IngestionWorkerConfig:
    return IngestionWorkerConfig(
        poll_seconds=0.01,
        **kwargs,
    )


def _turn_job(
    *,
    transcript_path: str,
    skip_extraction: bool = False,
) -> Any:
    from nesy_reasoning_mcp.auto_ingest import ConversationTurnJob

    return ConversationTurnJob(
        job_id="turn-1",
        session_id="session-1",
        transcript_path=transcript_path,
        turn_index=1,
        skip_extraction=skip_extraction,
    )


def _write_transcript(path: Path, messages: list[dict[str, Any]]) -> None:
    import json

    path.write_text("\n".join(json.dumps(message) for message in messages), encoding="utf-8")


def _review_record(
    record_id: str = "queue-1",
    *,
    candidate: CandidateRelation | None = None,
    source_job_ids: list[str] | None = None,
) -> ReviewQueueRecord:
    item = candidate or _candidate()
    return ReviewQueueRecord(
        id=record_id,
        run_id="run-1",
        candidate=item,
        gate_result=GateResult(candidate_id=item.id, action=GateAction.QUEUE),
        source_job_ids=source_job_ids or ["turn-1"],
    )


def _review_config(**kwargs: Any) -> ReviewWorkerConfig:
    return ReviewWorkerConfig(
        poll_seconds=0.01,
        reviewer_models=["reviewer-a", "reviewer-b"],
        model="reviewer",
        allow_single_reviewer_write=True,
        **kwargs,
    )


async def _review_agent(
    agent: Any,
    prompt: str,
    *,
    tracing_disabled: bool = False,
) -> dict[str, Any]:
    assert agent.output_type is openai_agents.ReviewDecisionBatch
    return {
        "reviews": [
            ReviewDecision(
                candidate_id="cand-1",
                decision=ReviewDecisionValue.APPROVE,
                final_relation_type=RelationType.SUFFICIENT,
                final_confidence=0.95,
                normalized_implication_supported=True,
                reasons=["supported"],
            ).model_dump(mode="json")
        ]
    }


def _fake_agent(**kwargs: Any) -> SimpleNamespace:
    return SimpleNamespace(output_type=kwargs["output_type"], name=kwargs["name"])


def test_review_worker_write_safety_counts_deduped_effective_reviewers() -> None:
    with pytest.raises(ValueError, match="multiple reviewers"):
        ReviewWorkerConfig(auto_write=True, reviewer_models=["reviewer-a", " reviewer-a "])
    with pytest.raises(ValueError, match="multiple reviewers"):
        ReviewWorkerConfig(auto_write=True, reviewer_models=["", "  "])

    config = ReviewWorkerConfig(
        auto_write=True,
        reviewer_models=["reviewer-a"],
        reviewer_configs=[
            ReviewerModelConfig(
                reviewer_id="provider:reviewer-b",
                model="reviewer-b",
                provider_name="provider",
            )
        ],
    )

    assert config.auto_write is True


@pytest.mark.asyncio
async def test_review_worker_commits_record_and_marks_source_job_done(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = RelationStore()
    store.enqueue_ingestion_jobs([_turn_job(transcript_path="/tmp/transcript.jsonl")])
    store.claim_pending_ingestion_jobs()
    store.update_ingestion_job_status(
        "turn-1",
        ConversationTurnJobStatus.REVIEWING,
        expected_status=ConversationTurnJobStatus.EXTRACTING,
    )
    store.enqueue_review_queue([_review_record()])
    monkeypatch.setattr(openai_agents, "_build_agent", _fake_agent)

    result = await process_review_queue_once(
        store,
        config=_review_config(auto_write=True),
        env={"EXAMPLE_API_KEY": "test-key"},
        run_agent=_review_agent,
    )

    record = store.list_review_queue()[0]
    assert result.committed_record_ids == ["queue-1"]
    assert result.done_job_ids == ["turn-1"]
    assert record.status == ReviewQueueStatus.COMMITTED
    assert record.gate_result.action == GateAction.AUTO_WRITE
    assert store.list_ingestion_jobs()[0].status == ConversationTurnJobStatus.DONE
    assert store.list_relations()[0].source == "A"


@pytest.mark.asyncio
async def test_review_worker_keeps_job_reviewing_when_gate_stays_queue(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = RelationStore()
    store.enqueue_ingestion_jobs([_turn_job(transcript_path="/tmp/transcript.jsonl")])
    store.claim_pending_ingestion_jobs()
    store.update_ingestion_job_status(
        "turn-1",
        ConversationTurnJobStatus.REVIEWING,
        expected_status=ConversationTurnJobStatus.EXTRACTING,
    )
    store.enqueue_review_queue([_review_record()])

    async def needs_human(
        agent: Any, prompt: str, *, tracing_disabled: bool = False
    ) -> dict[str, Any]:
        return {
            "reviews": [
                ReviewDecision(
                    candidate_id="cand-1",
                    decision=ReviewDecisionValue.NEEDS_HUMAN,
                    reasons=["ambiguous"],
                ).model_dump(mode="json")
            ]
        }

    monkeypatch.setattr(openai_agents, "_build_agent", _fake_agent)

    result = await process_review_queue_once(
        store,
        config=_review_config(auto_write=True),
        env={"EXAMPLE_API_KEY": "test-key"},
        run_agent=needs_human,
    )

    record = store.list_review_queue()[0]
    assert result.pending_record_ids == ["queue-1"]
    assert result.done_job_ids == []
    assert record.status == ReviewQueueStatus.PENDING
    assert record.attempt_count == 1
    assert record.next_retry_at is not None
    assert store.claim_pending_review_queue_records(now=record.updated_at) == []
    assert store.list_ingestion_jobs()[0].status == ConversationTurnJobStatus.REVIEWING


@pytest.mark.asyncio
async def test_review_worker_resolves_rejected_candidate_and_marks_job_done(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = RelationStore()
    store.enqueue_ingestion_jobs([_turn_job(transcript_path="/tmp/transcript.jsonl")])
    store.claim_pending_ingestion_jobs()
    store.update_ingestion_job_status(
        "turn-1",
        ConversationTurnJobStatus.REVIEWING,
        expected_status=ConversationTurnJobStatus.EXTRACTING,
    )
    store.enqueue_review_queue([_review_record()])

    async def reject(agent: Any, prompt: str, *, tracing_disabled: bool = False) -> dict[str, Any]:
        return {
            "reviews": [
                ReviewDecision(
                    candidate_id="cand-1",
                    decision=ReviewDecisionValue.REJECT,
                    reasons=["unsupported"],
                ).model_dump(mode="json")
            ]
        }

    monkeypatch.setattr(openai_agents, "_build_agent", _fake_agent)

    result = await process_review_queue_once(
        store,
        config=_review_config(auto_write=True, voting_policy=ReviewVotingPolicy.MAJORITY),
        env={"EXAMPLE_API_KEY": "test-key"},
        run_agent=reject,
    )

    record = store.list_review_queue()[0]
    assert result.resolved_record_ids == ["queue-1"]
    assert result.done_job_ids == ["turn-1"]
    assert record.status == ReviewQueueStatus.RESOLVED
    assert record.gate_result.action == GateAction.REJECT
    assert store.list_ingestion_jobs()[0].status == ConversationTurnJobStatus.DONE


@pytest.mark.asyncio
async def test_review_worker_retries_then_fails_source_job_after_exhaustion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = RelationStore()
    store.enqueue_ingestion_jobs([_turn_job(transcript_path="/tmp/transcript.jsonl")])
    store.claim_pending_ingestion_jobs()
    store.update_ingestion_job_status(
        "turn-1",
        ConversationTurnJobStatus.REVIEWING,
        expected_status=ConversationTurnJobStatus.EXTRACTING,
    )
    store.enqueue_review_queue([_review_record()])
    monkeypatch.setattr(openai_agents, "_build_agent", _fake_agent)

    async def fail(agent: Any, prompt: str, *, tracing_disabled: bool = False) -> dict[str, Any]:
        raise RuntimeError("secret key")

    first = await process_review_queue_once(
        store,
        config=_review_config(auto_write=True, max_retries=1, retry_backoff_seconds=0),
        env={"EXAMPLE_API_KEY": "test-key"},
        run_agent=fail,
    )
    await anyio.sleep(0.001)
    second = await process_review_queue_once(
        store,
        config=_review_config(auto_write=True, max_retries=1, retry_backoff_seconds=0),
        env={"EXAMPLE_API_KEY": "test-key"},
        run_agent=fail,
    )

    record = store.list_review_queue()[0]
    assert first.pending_record_ids == ["queue-1"]
    assert second.failed_record_ids == ["queue-1"]
    assert "secret" not in second.diagnostics[0].message
    assert record.status == ReviewQueueStatus.FAILED
    assert record.attempt_count == 2
    assert store.list_ingestion_jobs()[0].status == ConversationTurnJobStatus.FAILED


@pytest.mark.asyncio
async def test_review_worker_retries_write_failure_without_fake_commit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = RelationStore()
    store.enqueue_ingestion_jobs([_turn_job(transcript_path="/tmp/transcript.jsonl")])
    store.claim_pending_ingestion_jobs()
    store.update_ingestion_job_status(
        "turn-1",
        ConversationTurnJobStatus.REVIEWING,
        expected_status=ConversationTurnJobStatus.EXTRACTING,
    )
    store.enqueue_review_queue([_review_record()])
    monkeypatch.setattr(openai_agents, "_build_agent", _fake_agent)

    def fail_assert_relations(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("write failed with secret path")

    monkeypatch.setattr(store, "assert_relations", fail_assert_relations)

    result = await process_review_queue_once(
        store,
        config=_review_config(auto_write=True, max_retries=1, retry_backoff_seconds=0),
        env={"EXAMPLE_API_KEY": "test-key"},
        run_agent=_review_agent,
    )

    record = store.list_review_queue()[0]
    assert result.committed_record_ids == []
    assert result.pending_record_ids == ["queue-1"]
    assert record.status == ReviewQueueStatus.PENDING
    assert record.committed_relation_ids == []
    assert store.list_ingestion_jobs()[0].status == ConversationTurnJobStatus.REVIEWING


@pytest.mark.asyncio
async def test_ingestion_nesy_facts_fast_path_flows_into_review_worker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transcript = tmp_path / "transcript.jsonl"
    _write_transcript(
        transcript,
        [
            {
                "role": "assistant",
                "content": (
                    'Done.\nNESY_FACTS:\n[{"id":"cand-1","source":"A","target":"B",'
                    '"relation_type":"sufficient"}]'
                ),
            }
        ],
    )
    store = RelationStore()
    store.enqueue_ingestion_jobs([_turn_job(transcript_path=str(transcript), skip_extraction=True)])
    monkeypatch.setattr(openai_agents, "_build_agent", _fake_agent)

    ingest = await process_ingestion_queue_once(
        store,
        config=_worker_config(),
        env={"EXAMPLE_API_KEY": "test-key"},
        run_chat_completion=None,
    )
    review = await process_review_queue_once(
        store,
        config=_review_config(auto_write=True),
        env={"EXAMPLE_API_KEY": "test-key"},
        run_agent=_review_agent,
    )

    assert len(ingest.queued_record_ids) == 1
    assert review.committed_record_ids == ingest.queued_record_ids
    assert store.list_ingestion_jobs()[0].status == ConversationTurnJobStatus.DONE
    assert store.list_relations()[0].source == "A"


@pytest.mark.asyncio
async def test_ingestion_worker_runs_review_queue_handoff(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transcript = tmp_path / "transcript.jsonl"
    _write_transcript(
        transcript,
        [
            {
                "role": "assistant",
                "content": (
                    'Done.\nNESY_FACTS:\n[{"id":"cand-1","source":"A","target":"B",'
                    '"relation_type":"sufficient"}]'
                ),
            }
        ],
    )
    store = RelationStore()
    store.enqueue_ingestion_jobs([_turn_job(transcript_path=str(transcript), skip_extraction=True)])
    monkeypatch.setattr(openai_agents, "_build_agent", _fake_agent)

    result = await run_ingestion_worker(
        store,
        config=IngestionWorkerConfig(
            poll_seconds=0.01,
            claim_limit=1,
            review_config=_review_config(auto_write=True),
        ),
        max_jobs=1,
        env={"EXAMPLE_API_KEY": "test-key"},
        run_agent=_review_agent,
    )

    assert result.processed_job_ids == ["turn-1"]
    assert len(result.queued_record_ids) == 1
    assert result.committed_record_ids == result.queued_record_ids
    assert store.list_ingestion_jobs()[0].status == ConversationTurnJobStatus.DONE
