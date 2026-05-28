import json
from datetime import UTC, datetime, timedelta

import pytest

from nesy_reasoning_mcp.auto_ingest import ConversationTurnJob, ConversationTurnJobStatus
from nesy_reasoning_mcp.config import NesyConfig, StorageConfig
from nesy_reasoning_mcp.schemas import RelationRecord, RelationType
from nesy_reasoning_mcp.store import RelationStore, SqliteRelationStore
from nesy_reasoning_mcp.tool_queue_status import queue_status_snapshot
from nesy_reasoning_mcp.tools import QUEUE_STATUS, call_tool


def _turn_job(
    job_id: str,
    *,
    status: ConversationTurnJobStatus = ConversationTurnJobStatus.PENDING,
    updated_at: str = "2026-01-01T00:00:00+00:00",
) -> ConversationTurnJob:
    return ConversationTurnJob(
        job_id=job_id,
        session_id="session-1",
        transcript_path="/tmp/transcript.jsonl",
        status=status,
        updated_at=updated_at,
    )


@pytest.mark.asyncio
async def test_queue_status_empty_queue_mirrors_content() -> None:
    store = RelationStore()

    result = await call_tool(QUEUE_STATUS, {}, store)

    assert result.isError is False
    assert json.loads(result.content[0].text) == result.structuredContent
    assert result.structuredContent["pending"] == 0
    assert result.structuredContent["extracting"] == 0
    assert result.structuredContent["reviewing"] == 0
    assert result.structuredContent["done_last_24h"] == 0
    assert result.structuredContent["failed_last_24h"] == 0
    assert result.structuredContent["in_flight_total"] == 0
    assert result.structuredContent["last_write_at"] is None
    assert result.structuredContent["last_write_relation_count"] == 0


def test_queue_status_snapshot_counts_jobs_and_latest_write() -> None:
    now = datetime(2026, 1, 2, tzinfo=UTC)
    store = RelationStore()
    recent = (now - timedelta(hours=1)).isoformat()
    old = (now - timedelta(days=2)).isoformat()
    latest_write = "2026-01-01T12:00:00+00:00"
    store.enqueue_ingestion_jobs(
        [
            _turn_job("turn-pending", status=ConversationTurnJobStatus.PENDING),
            _turn_job("turn-extracting", status=ConversationTurnJobStatus.EXTRACTING),
            _turn_job("turn-reviewing", status=ConversationTurnJobStatus.REVIEWING),
            _turn_job("turn-done-recent", status=ConversationTurnJobStatus.DONE, updated_at=recent),
            _turn_job("turn-done-old", status=ConversationTurnJobStatus.DONE, updated_at=old),
            _turn_job(
                "turn-failed-recent",
                status=ConversationTurnJobStatus.FAILED,
                updated_at=recent,
            ),
            _turn_job("turn-failed-old", status=ConversationTurnJobStatus.FAILED, updated_at=old),
        ]
    )
    store.import_records(
        [
            RelationRecord(
                id="rel-old",
                source="A",
                target="B",
                relation_type=RelationType.SUFFICIENT,
                created_at="2026-01-01T00:00:00+00:00",
            ),
            RelationRecord(
                id="rel-new-1",
                source="C",
                target="D",
                relation_type=RelationType.SUFFICIENT,
                created_at=latest_write,
            ),
            RelationRecord(
                id="rel-new-2",
                source="E",
                target="F",
                relation_type=RelationType.SUFFICIENT,
                created_at=latest_write,
            ),
        ],
        [],
        mode="append",
        store_id="default",
    )

    snapshot = queue_status_snapshot(store, now=now)

    assert snapshot == {
        "pending": 1,
        "extracting": 1,
        "reviewing": 1,
        "done_last_24h": 1,
        "failed_last_24h": 1,
        "in_flight_total": 3,
        "last_write_at": latest_write,
        "last_write_relation_count": 2,
    }


@pytest.mark.asyncio
async def test_queue_status_reads_live_sqlite_state(tmp_path) -> None:
    config = NesyConfig(
        storage=StorageConfig(backend="sqlite", sqlite_path=str(tmp_path / "nesy.db"))
    )
    writer = SqliteRelationStore(config)
    writer.enqueue_ingestion_jobs(
        [
            _turn_job("turn-pending", status=ConversationTurnJobStatus.PENDING),
            _turn_job("turn-reviewing", status=ConversationTurnJobStatus.REVIEWING),
        ]
    )
    reader = SqliteRelationStore(config)

    result = await call_tool(QUEUE_STATUS, {}, reader)

    assert result.structuredContent["pending"] == 1
    assert result.structuredContent["reviewing"] == 1
    assert result.structuredContent["in_flight_total"] == 2
