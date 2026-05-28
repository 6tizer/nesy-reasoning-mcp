"""Queue status tool handler for Auto-Ingest visibility."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

from nesy_reasoning_mcp.auto_ingest.schemas import ConversationTurnJobStatus
from nesy_reasoning_mcp.schemas import QueueStatusInput, RelationRecord
from nesy_reasoning_mcp.store import RelationStoreProtocol
from nesy_reasoning_mcp.time_utils import parse_datetime_value


async def queue_status(
    arguments: dict[str, Any],
    store: RelationStoreProtocol,
) -> dict[str, Any]:
    """Handle `nesy.queue_status`."""
    QueueStatusInput.model_validate(arguments)
    snapshot = queue_status_snapshot(store)
    return {
        "status": "ok",
        **snapshot,
        "diagnostics": [],
        "trace": [
            ("Counted Auto-Ingest conversation turn jobs and latest durable relation write.")
        ],
        "graph_stats": store.graph_stats().model_dump(mode="json"),
    }


def queue_status_snapshot(
    store: RelationStoreProtocol,
    *,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Return a read-only Auto-Ingest queue status snapshot."""
    threshold = (now or datetime.now(UTC)) - timedelta(hours=24)
    jobs = store.list_ingestion_jobs()
    pending = sum(1 for job in jobs if job.status == ConversationTurnJobStatus.PENDING)
    extracting = sum(1 for job in jobs if job.status == ConversationTurnJobStatus.EXTRACTING)
    reviewing = sum(1 for job in jobs if job.status == ConversationTurnJobStatus.REVIEWING)
    done_last_24h = sum(
        1
        for job in jobs
        if job.status == ConversationTurnJobStatus.DONE
        and _updated_within(job.updated_at, threshold)
    )
    failed_last_24h = sum(
        1
        for job in jobs
        if job.status == ConversationTurnJobStatus.FAILED
        and _updated_within(job.updated_at, threshold)
    )
    last_write_at, last_write_relation_count = _last_write_summary(store.list_relations())
    return {
        "pending": pending,
        "extracting": extracting,
        "reviewing": reviewing,
        "done_last_24h": done_last_24h,
        "failed_last_24h": failed_last_24h,
        "in_flight_total": pending + extracting + reviewing,
        "last_write_at": last_write_at,
        "last_write_relation_count": last_write_relation_count,
    }


def _last_write_summary(relations: list[RelationRecord]) -> tuple[str | None, int]:
    latest_relation = max(
        (relation for relation in relations if _parse_timestamp(relation.created_at) is not None),
        key=lambda relation: (
            _parse_timestamp(relation.created_at) or datetime.min.replace(tzinfo=UTC)
        ),
        default=None,
    )
    if latest_relation is None:
        return None, 0
    return latest_relation.created_at, sum(
        1 for relation in relations if relation.created_at == latest_relation.created_at
    )


def _updated_within(value: str, threshold: datetime) -> bool:
    parsed = _parse_timestamp(value)
    return parsed is not None and parsed >= threshold


def _parse_timestamp(value: str) -> datetime | None:
    try:
        return parse_datetime_value(value)
    except ValueError:
        return None
