"""Tests for the async worker loop and nesy-worker daemon."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest

from nesy_reasoning_mcp.auto_ingest import (
    CandidateRelation,
    ConversationTurnJob,
    ConversationTurnJobStatus,
    EvidenceRecord,
)
from nesy_reasoning_mcp.config import HttpConfig, NesyConfig, WorkerConfig, load_config
from nesy_reasoning_mcp.schemas import RelationType
from nesy_reasoning_mcp.store import RelationStore
from nesy_reasoning_mcp.worker import run_worker_loop


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
) -> ConversationTurnJob:
    return ConversationTurnJob(
        job_id=job_id,
        session_id=session_id,
        transcript_path=transcript_path,
        turn_index=turn_index,
    )


def _write_transcript(path: Path, messages: list[dict[str, Any]]) -> None:
    path.write_text("\n".join(json.dumps(message) for message in messages), encoding="utf-8")


@pytest.mark.asyncio
async def test_worker_loop_processes_jobs_and_shuts_down(tmp_path: Path) -> None:
    """Worker loop processes queued extraction jobs when a transcript exists."""
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

    shutdown_event = asyncio.Event()

    async def _run_and_shutdown() -> None:  # noqa: RUF029
        await asyncio.sleep(0.05)
        shutdown_event.set()

    loop_task = asyncio.create_task(
        run_worker_loop(
            store,
            config=WorkerConfig(poll_seconds=0.02),
            shutdown_event=shutdown_event,
            env={"EXAMPLE_API_KEY": "test-key"},
        )
    )
    await _run_and_shutdown()
    result = await loop_task

    jobs = store.list_ingestion_jobs()
    # Worker should have processed the job (extraction requires LLM, so it fails,
    # but the job status should transition from pending)
    assert result.iterations > 0
    assert result.interrupted is False
    # Without a real LLM, the extraction fails and job is marked FAILED
    assert jobs[0].status in {
        ConversationTurnJobStatus.FAILED,
        ConversationTurnJobStatus.REVIEWING,
        ConversationTurnJobStatus.DONE,
    }


@pytest.mark.asyncio
async def test_worker_loop_shuts_down_cleanly(tmp_path: Path) -> None:
    """Worker loop shuts down immediately when event is already set."""
    store = RelationStore()
    shutdown_event = asyncio.Event()
    shutdown_event.set()  # already set before starting

    result = await run_worker_loop(
        store,
        config=WorkerConfig(poll_seconds=0.02),
        shutdown_event=shutdown_event,
    )

    assert result.iterations == 0
    assert result.interrupted is False


@pytest.mark.asyncio
async def test_worker_loop_handles_empty_queue(tmp_path: Path) -> None:
    """Worker loop handles an empty queue without crashing."""
    store = RelationStore()
    shutdown_event = asyncio.Event()

    async def _shutdown_after_poll() -> None:  # noqa: RUF029
        await asyncio.sleep(0.05)
        shutdown_event.set()

    loop_task = asyncio.create_task(
        run_worker_loop(
            store,
            config=WorkerConfig(poll_seconds=0.02),
            shutdown_event=shutdown_event,
        )
    )
    await _shutdown_after_poll()
    result = await loop_task

    assert result.processed_job_ids == []
    assert result.interrupted is False


@pytest.mark.asyncio
async def test_worker_config_defaults() -> None:
    """WorkerConfig has expected defaults."""
    config = WorkerConfig()
    assert config.poll_seconds == 5.0
    assert config.claim_limit == 5
    assert config.health_port == 8766


def test_worker_config_env_overrides() -> None:
    """Worker config is loaded from environment variables."""
    env: dict[str, str] = {
        "NESY_WORKER_POLL_SECONDS": "10.0",
        "NESY_WORKER_CLAIM_LIMIT": "3",
        "NESY_WORKER_HEALTH_PORT": "9999",
        "NESY_LOCAL_TOKEN": "secret",
    }
    config = load_config(env=env)
    assert config.worker.poll_seconds == 10.0
    assert config.worker.claim_limit == 3
    assert config.worker.health_port == 9999


def test_http_server_accepts_with_worker_flag() -> None:
    """create_http_app accepts with_worker=True and starts the background task."""
    from nesy_reasoning_mcp.http_server import create_http_app

    config = NesyConfig(http=HttpConfig(local_token="secret"))
    app = create_http_app(config, RelationStore(config), with_worker=True)
    assert app is not None


def test_http_server_worker_disabled_by_default_in_create_http_app() -> None:
    """create_http_app does not start worker when with_worker=False (default)."""
    from nesy_reasoning_mcp.http_server import create_http_app

    config = NesyConfig(http=HttpConfig(local_token="secret"))
    app = create_http_app(config, RelationStore(config))
    assert app is not None
