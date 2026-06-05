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
    _ = QueueStatusInput.model_validate(arguments)
    snapshot = queue_status_snapshot(store)
    return {
        "status": "ok",
        **snapshot,
        "diagnostics": [],
        "trace": ["Counted Auto-Ingest conversation turn jobs and latest durable relation write."],
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
    counts: dict[str, int] = {
        "pending": 0,
        "extracting": 0,
        "reviewing": 0,
        "done_last_24h": 0,
        "failed_last_24h": 0,
    }
    for job in jobs:
        if job.status == ConversationTurnJobStatus.PENDING:
            counts["pending"] += 1
        elif job.status == ConversationTurnJobStatus.EXTRACTING:
            counts["extracting"] += 1
        elif job.status == ConversationTurnJobStatus.REVIEWING:
            counts["reviewing"] += 1
        elif job.status == ConversationTurnJobStatus.DONE and _updated_within(
            job.updated_at, threshold
        ):
            counts["done_last_24h"] += 1
        elif job.status == ConversationTurnJobStatus.FAILED and _updated_within(
            job.updated_at, threshold
        ):
            counts["failed_last_24h"] += 1
    pending = counts["pending"]
    extracting = counts["extracting"]
    reviewing = counts["reviewing"]
    done_last_24h = counts["done_last_24h"]
    failed_last_24h = counts["failed_last_24h"]
    in_flight_total = pending + extracting + reviewing
    if in_flight_total > 0:
        last_write_at, last_write_relation_count = _last_write_summary(store.list_relations())
    else:
        last_write_at, last_write_relation_count = None, 0
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
    # NOTE: grouping by exact `created_at` string assumes all relations in a
    # single write batch share the same microsecond timestamp.  This holds in
    # practice (the writer sets `created_at` from `datetime.now(UTC)` once per
    # `import_records` call), but two concurrent writes that happen to land on
    # the same microsecond would be counted as one batch.
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
        # parse_datetime_value raises ValueError for malformed ISO strings.
        # Log a warning so that parsing failures are not silently swallowed
        # and can be investigated if they become frequent.
        import logging

        logger = logging.getLogger(__name__)
        logger.warning("Failed to parse timestamp: %r", value)
        return None
