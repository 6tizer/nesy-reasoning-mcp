import json
from pathlib import Path
from typing import Any

import pytest

from nesy_reasoning_mcp.auto_ingest import (
    CandidateRelation,
    CandidateRelationBatch,
    ConversationTurnJob,
    ConversationTurnJobStatus,
    EvidenceRecord,
    ExtractionModelConfig,
    IngestionWorkerConfig,
    process_ingestion_queue_once,
    run_ingestion_worker,
)
from nesy_reasoning_mcp.auto_ingest.openai_agents import OpenAICompatibleProviderConfig
from nesy_reasoning_mcp.auto_ingest.providers import ProviderStructuredOutputMode
from nesy_reasoning_mcp.schemas import RelationInput, RelationType
from nesy_reasoning_mcp.store import RelationStore


def _provider_config() -> OpenAICompatibleProviderConfig:
    return OpenAICompatibleProviderConfig(
        base_url="https://example.com/v1",
        api_key_env="EXAMPLE_API_KEY",
        structured_output_mode=ProviderStructuredOutputMode.JSON_OBJECT,
    )


def _worker_config(**kwargs: Any) -> IngestionWorkerConfig:
    extraction_config = ExtractionModelConfig(
        model="extractor",
        provider_config=_provider_config(),
        timeout_seconds=5,
    )
    return IngestionWorkerConfig(
        poll_seconds=0.01,
        extraction_config=extraction_config,
        **kwargs,
    )


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
        evidence=[EvidenceRecord(url="conversation://test", span=f"{source} enables {target}")],
    )


def _turn_job(
    job_id: str = "turn-1",
    *,
    transcript_path: str,
    session_id: str = "session-1",
    turn_index: int | None = 1,
    priority: int = 0,
    skip_extraction: bool = False,
) -> ConversationTurnJob:
    return ConversationTurnJob(
        job_id=job_id,
        session_id=session_id,
        transcript_path=transcript_path,
        turn_index=turn_index,
        priority=priority,
        skip_extraction=skip_extraction,
    )


def _write_transcript(path: Path, messages: list[dict[str, Any]]) -> None:
    path.write_text("\n".join(json.dumps(message) for message in messages), encoding="utf-8")


@pytest.mark.asyncio
async def test_worker_processes_normal_job_into_review_queue(tmp_path: Path) -> None:
    transcript = tmp_path / "transcript.jsonl"
    _write_transcript(
        transcript,
        [
            {"role": "user", "content": "A requires B because the workflow depends on it."},
            {"role": "assistant", "content": "A enables B."},
        ],
    )
    store = RelationStore()
    store.enqueue_ingestion_jobs([_turn_job(transcript_path=str(transcript))])

    async def fake_completion(**_kwargs: Any) -> CandidateRelationBatch:
        return CandidateRelationBatch(candidates=[_candidate()])

    result = await process_ingestion_queue_once(
        store,
        config=_worker_config(),
        env={"EXAMPLE_API_KEY": "test-key"},
        run_chat_completion=fake_completion,
    )

    jobs = store.list_ingestion_jobs()
    queued = store.list_review_queue()
    assert result.processed_job_ids == ["turn-1"]
    assert len(queued) == 1
    assert queued[0].candidate.id == "cand-1"
    assert jobs[0].status == ConversationTurnJobStatus.REVIEWING


@pytest.mark.asyncio
async def test_worker_marks_empty_candidate_job_done(tmp_path: Path) -> None:
    transcript = tmp_path / "transcript.jsonl"
    _write_transcript(transcript, [{"role": "assistant", "content": "No durable relation."}])
    store = RelationStore()
    store.enqueue_ingestion_jobs([_turn_job(transcript_path=str(transcript))])

    async def fake_completion(**_kwargs: Any) -> CandidateRelationBatch:
        return CandidateRelationBatch(candidates=[])

    result = await process_ingestion_queue_once(
        store,
        config=_worker_config(),
        env={"EXAMPLE_API_KEY": "test-key"},
        run_chat_completion=fake_completion,
    )

    assert result.processed_job_ids == ["turn-1"]
    assert store.list_review_queue() == []
    assert store.list_ingestion_jobs()[0].status == ConversationTurnJobStatus.DONE


@pytest.mark.asyncio
async def test_worker_marks_extraction_error_failed(tmp_path: Path) -> None:
    transcript = tmp_path / "transcript.jsonl"
    _write_transcript(transcript, [{"role": "assistant", "content": "A causes B."}])
    store = RelationStore()
    store.enqueue_ingestion_jobs([_turn_job(transcript_path=str(transcript))])

    async def fail_completion(**_kwargs: Any) -> CandidateRelationBatch:
        raise RuntimeError("secret path")

    result = await process_ingestion_queue_once(
        store,
        config=_worker_config(),
        env={"EXAMPLE_API_KEY": "test-key"},
        run_chat_completion=fail_completion,
    )

    assert result.failed_job_ids == ["turn-1"]
    assert result.diagnostics[0].message.endswith("RuntimeError")
    assert "secret" not in result.diagnostics[0].message
    assert store.list_ingestion_jobs()[0].status == ConversationTurnJobStatus.FAILED


@pytest.mark.asyncio
async def test_worker_nesy_facts_fast_path_skips_llm(tmp_path: Path) -> None:
    transcript = tmp_path / "transcript.jsonl"
    _write_transcript(
        transcript,
        [
            {
                "role": "assistant",
                "content": (
                    'Done.\nNESY_FACTS:\n[{"source":"A","target":"B","relation_type":"sufficient"}]'
                ),
            }
        ],
    )
    store = RelationStore()
    store.enqueue_ingestion_jobs([_turn_job(transcript_path=str(transcript), skip_extraction=True)])

    async def fail_if_called(**_kwargs: Any) -> CandidateRelationBatch:
        raise AssertionError("fast path must not call LLM")

    result = await process_ingestion_queue_once(
        store,
        config=_worker_config(),
        env={"EXAMPLE_API_KEY": "test-key"},
        run_chat_completion=fail_if_called,
    )

    queued = store.list_review_queue()
    assert result.queued_record_ids == [queued[0].id]
    assert queued[0].candidate.source == "A"
    assert queued[0].candidate.metadata["fast_path"] == "nesy_facts"
    assert store.list_ingestion_jobs()[0].status == ConversationTurnJobStatus.REVIEWING


@pytest.mark.asyncio
async def test_worker_nesy_facts_xml_fast_path(tmp_path: Path) -> None:
    transcript = tmp_path / "transcript.jsonl"
    _write_transcript(
        transcript,
        [
            {
                "role": "assistant",
                "content": (
                    '<NESY_FACTS>[{"source":"A","target":"B",'
                    '"relation_type":"necessary"}]</NESY_FACTS>'
                ),
            }
        ],
    )
    store = RelationStore()
    store.enqueue_ingestion_jobs([_turn_job(transcript_path=str(transcript), skip_extraction=True)])

    result = await process_ingestion_queue_once(
        store,
        config=_worker_config(),
        env={"EXAMPLE_API_KEY": "test-key"},
        run_chat_completion=None,
    )

    assert result.failed_job_ids == []
    assert store.list_review_queue()[0].candidate.relation_type == RelationType.NECESSARY


@pytest.mark.asyncio
async def test_worker_adds_semantic_duplicate_metadata(tmp_path: Path) -> None:
    transcript = tmp_path / "transcript.jsonl"
    _write_transcript(transcript, [{"role": "assistant", "content": "Login enables checkout."}])
    store = RelationStore()
    store.assert_relations(
        [
            RelationInput(
                source="user login required",
                target="submit checkout order",
                relation_type=RelationType.SUFFICIENT,
            )
        ]
    )
    store.enqueue_ingestion_jobs([_turn_job(transcript_path=str(transcript))])

    async def fake_completion(**_kwargs: Any) -> CandidateRelationBatch:
        return CandidateRelationBatch(
            candidates=[
                _candidate(
                    source="login required",
                    target="checkout order",
                )
            ]
        )

    await process_ingestion_queue_once(
        store,
        config=_worker_config(),
        env={"EXAMPLE_API_KEY": "test-key"},
        run_chat_completion=fake_completion,
    )

    metadata = store.list_review_queue()[0].gate_result.metadata
    assert metadata["semantic_duplicate"]["reason"] == "likely_semantic_duplicate"


@pytest.mark.asyncio
async def test_worker_merges_adjacent_turns_for_one_extraction(tmp_path: Path) -> None:
    transcript = tmp_path / "transcript.jsonl"
    _write_transcript(transcript, [{"role": "assistant", "content": "A implies B."}])
    store = RelationStore()
    store.enqueue_ingestion_jobs(
        [
            _turn_job("turn-1", transcript_path=str(transcript), turn_index=1),
            _turn_job("turn-2", transcript_path=str(transcript), turn_index=2),
        ]
    )
    calls = 0

    async def fake_completion(**_kwargs: Any) -> CandidateRelationBatch:
        nonlocal calls
        calls += 1
        return CandidateRelationBatch(candidates=[_candidate()])

    result = await process_ingestion_queue_once(
        store,
        config=_worker_config(claim_limit=2),
        env={"EXAMPLE_API_KEY": "test-key"},
        run_chat_completion=fake_completion,
    )

    assert calls == 1
    assert result.processed_job_ids == ["turn-1", "turn-2"]
    assert [job.status for job in store.list_ingestion_jobs()] == [
        ConversationTurnJobStatus.REVIEWING,
        ConversationTurnJobStatus.REVIEWING,
    ]


@pytest.mark.asyncio
async def test_worker_backpressure_drops_over_depth(tmp_path: Path) -> None:
    transcript = tmp_path / "transcript.jsonl"
    _write_transcript(transcript, [{"role": "assistant", "content": "A implies B."}])
    store = RelationStore()
    store.enqueue_ingestion_jobs(
        [
            _turn_job("turn-old", transcript_path=str(transcript)).model_copy(
                update={"enqueued_at": "2026-01-01T00:00:01+00:00"}
            ),
            _turn_job("turn-new", transcript_path=str(transcript)).model_copy(
                update={"enqueued_at": "2026-01-01T00:00:02+00:00"}
            ),
        ]
    )

    result = await process_ingestion_queue_once(
        store,
        config=_worker_config(claim_limit=0),
        env={"NESY_INGEST_QUEUE_MAX_DEPTH": "1", "EXAMPLE_API_KEY": "test-key"},
    )

    assert result.dropped_job_ids == ["turn-old"]
    assert result.diagnostics[0].code == "INGESTION_QUEUE_BACKPRESSURE_DROP"
    assert [job.job_id for job in store.list_ingestion_jobs()] == ["turn-new"]


@pytest.mark.asyncio
async def test_worker_max_jobs_limits_claim_size(tmp_path: Path) -> None:
    transcript = tmp_path / "transcript.jsonl"
    _write_transcript(transcript, [{"role": "assistant", "content": "A implies B."}])
    store = RelationStore()
    store.enqueue_ingestion_jobs(
        [
            _turn_job("turn-1", transcript_path=str(transcript)),
            _turn_job("turn-2", transcript_path=str(transcript)),
        ]
    )

    async def fake_completion(**_kwargs: Any) -> CandidateRelationBatch:
        return CandidateRelationBatch(candidates=[])

    result = await run_ingestion_worker(
        store,
        config=_worker_config(claim_limit=5),
        max_jobs=1,
        env={"EXAMPLE_API_KEY": "test-key"},
        run_chat_completion=fake_completion,
    )

    assert result.processed_job_ids == ["turn-1"]
    assert {job.job_id: job.status for job in store.list_ingestion_jobs()} == {
        "turn-1": ConversationTurnJobStatus.DONE,
        "turn-2": ConversationTurnJobStatus.PENDING,
    }
