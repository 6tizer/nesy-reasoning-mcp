"""MCP handlers for persisted ingestion review queue records."""

from __future__ import annotations

import base64
import binascii
import json
from typing import Any

from nesy_reasoning_mcp.auto_ingest.schemas import (
    CommitReviewedRelationsInput,
    ListReviewQueueInput,
    ResolveReviewQueueInput,
    ReviewQueueFilter,
    ReviewQueueRecord,
    ReviewQueueStatus,
)
from nesy_reasoning_mcp.schemas import Diagnostic, PropositionRecord, RelationInput
from nesy_reasoning_mcp.store import RelationStoreProtocol


async def list_review_queue(
    arguments: dict[str, Any],
    store: RelationStoreProtocol,
) -> dict[str, Any]:
    """Handle `nesy.list_review_queue`."""
    payload = ListReviewQueueInput.model_validate(arguments)
    queue_filter, cursor_diagnostic = _filter_with_cursor(payload.filter, payload.cursor)
    if cursor_diagnostic is not None:
        return {
            "status": "error",
            "records": [],
            "total": 0,
            "next_cursor": None,
            "diagnostics": [cursor_diagnostic.model_dump(mode="json")],
            "trace": ["Review queue list rejected invalid cursor."],
            "graph_stats": store.graph_stats().model_dump(mode="json"),
        }
    listed = store.list_review_queue(queue_filter, limit=payload.limit + 1)
    records = listed[: payload.limit]
    next_cursor = _encode_review_queue_cursor(records[-1]) if len(listed) > payload.limit else None
    return {
        "status": "ok",
        "records": [_queue_record_dump(record) for record in records],
        "total": len(records),
        "next_cursor": next_cursor,
        "diagnostics": [],
        "trace": [f"Listed {len(records)} review queue record(s)."],
        "graph_stats": store.graph_stats().model_dump(mode="json"),
    }


async def commit_reviewed_relations(
    arguments: dict[str, Any],
    store: RelationStoreProtocol,
) -> dict[str, Any]:
    """Handle `nesy.commit_reviewed_relations`."""
    from nesy_reasoning_mcp.auto_ingest.writer import write_approved_relations
    from nesy_reasoning_mcp.tool_candidate_validation import validate_candidate_relations

    payload = CommitReviewedRelationsInput.model_validate(arguments)
    records, diagnostics = _selected_pending_records(payload.ids, store)
    if diagnostics:
        return _commit_error(payload.ids, diagnostics, store)

    validation = await validate_candidate_relations(
        {
            "candidates": [
                record.candidate.model_dump(mode="json", exclude_none=True) for record in records
            ],
            "reviews": [
                record.review.model_dump(mode="json", exclude_none=True)
                for record in records
                if record.review is not None
            ],
            "propositions": [
                proposition.model_dump(mode="json", exclude_none=True)
                for proposition in _merged_propositions(records)
            ],
            "min_write_confidence": payload.min_write_confidence,
            "max_depth": payload.max_depth,
            "min_confidence": payload.min_confidence,
            "include_soft": payload.include_soft,
        },
        store,
    )
    if validation.get("status") == "error":
        return _commit_blocked(payload.ids, records, validation, store, status="error")
    if (
        validation.get("queued_count", 0) > 0
        or validation.get("rejected_count", 0) > 0
        or validation.get("approved_count", 0) != len(records)
    ):
        return _commit_blocked(payload.ids, records, validation, store, status="warning")

    approved_relations = [
        RelationInput.model_validate(item) for item in validation.get("approved_relations", [])
    ]
    relation_ids, write_diagnostics, write_result = await write_approved_relations(
        relations=approved_relations,
        store=store,
    )
    if len(relation_ids) != len(records):
        diagnostics = [
            *write_diagnostics,
            Diagnostic(
                level="error",
                code="REVIEW_QUEUE_WRITE_FAILED",
                message="Review queue commit did not write one relation for each selected record.",
                related_ids=payload.ids,
            ),
        ]
        return {
            "status": "error",
            "committed_count": 0,
            "queue_ids": payload.ids,
            "relation_ids": relation_ids,
            "validation": validation,
            "write_result": write_result,
            "diagnostics": [item.model_dump(mode="json") for item in diagnostics],
            "trace": ["Validated queue records, but relation assertion did not fully succeed."],
            "graph_stats": store.graph_stats().model_dump(mode="json"),
        }

    relation_ids_by_record = {
        record.id: [relation_id] for record, relation_id in zip(records, relation_ids, strict=True)
    }
    updated = store.mark_review_queue_committed(payload.ids, relation_ids_by_record)
    return {
        "status": "ok",
        "committed_count": updated,
        "queue_ids": payload.ids,
        "relation_ids": relation_ids,
        "validation": validation,
        "write_result": write_result,
        "diagnostics": [item.model_dump(mode="json") for item in write_diagnostics],
        "trace": [
            f"Committed {updated} review queue record(s) after re-running validation and gate."
        ],
        "graph_stats": store.graph_stats().model_dump(mode="json"),
    }


async def resolve_review_queue(
    arguments: dict[str, Any],
    store: RelationStoreProtocol,
) -> dict[str, Any]:
    """Handle `nesy.resolve_review_queue`."""
    payload = ResolveReviewQueueInput.model_validate(arguments)
    records, diagnostics = _selected_pending_records(payload.ids, store)
    if diagnostics:
        return {
            "status": "error",
            "resolved_count": 0,
            "queue_ids": payload.ids,
            "diagnostics": [item.model_dump(mode="json") for item in diagnostics],
            "trace": ["Review queue resolve rejected before mutation."],
            "graph_stats": store.graph_stats().model_dump(mode="json"),
        }
    resolved = store.resolve_review_queue(
        [record.id for record in records],
        reason=payload.reason,
        metadata=payload.metadata,
    )
    return {
        "status": "ok",
        "resolved_count": resolved,
        "queue_ids": payload.ids,
        "diagnostics": [],
        "trace": [f"Resolved {resolved} review queue record(s) without writing graph memory."],
        "graph_stats": store.graph_stats().model_dump(mode="json"),
    }


def _selected_pending_records(
    ids: list[str],
    store: RelationStoreProtocol,
) -> tuple[list[ReviewQueueRecord], list[Diagnostic]]:
    selected_records = store.list_review_queue(ReviewQueueFilter(ids=ids), limit=len(ids))
    records_by_id = {record.id: record for record in selected_records}
    records: list[ReviewQueueRecord] = []
    diagnostics: list[Diagnostic] = []
    for record_id in ids:
        record = records_by_id.get(record_id)
        if record is None:
            diagnostics.append(
                Diagnostic(
                    level="error",
                    code="REVIEW_QUEUE_RECORD_NOT_FOUND",
                    message=f"Review queue record not found: {record_id}",
                    related_ids=[record_id],
                )
            )
            continue
        if record.status != ReviewQueueStatus.PENDING:
            diagnostics.append(
                Diagnostic(
                    level="error",
                    code="REVIEW_QUEUE_RECORD_NOT_PENDING",
                    message=f"Review queue record is not pending: {record_id}",
                    related_ids=[record_id],
                )
            )
            continue
        records.append(record)
    return records, diagnostics


def _commit_error(
    ids: list[str],
    diagnostics: list[Diagnostic],
    store: RelationStoreProtocol,
) -> dict[str, Any]:
    return {
        "status": "error",
        "committed_count": 0,
        "queue_ids": ids,
        "relation_ids": [],
        "validation": {},
        "write_result": {},
        "diagnostics": [item.model_dump(mode="json") for item in diagnostics],
        "trace": ["Review queue commit rejected before validation."],
        "graph_stats": store.graph_stats().model_dump(mode="json"),
    }


def _commit_blocked(
    ids: list[str],
    records: list[ReviewQueueRecord],
    validation: dict[str, Any],
    store: RelationStoreProtocol,
    *,
    status: str,
) -> dict[str, Any]:
    return {
        "status": status,
        "committed_count": 0,
        "queue_ids": ids,
        "relation_ids": [],
        "validation": validation,
        "write_result": {},
        "diagnostics": validation.get("diagnostics", []),
        "trace": [
            (
                f"Validation did not approve all {len(records)} selected queue record(s); "
                "no graph memory was written. This can happen when graph state changed "
                "after the record was enqueued."
            )
        ],
        "graph_stats": store.graph_stats().model_dump(mode="json"),
    }


def _merged_propositions(records: list[ReviewQueueRecord]) -> list[PropositionRecord]:
    merged: dict[str, PropositionRecord] = {}
    for record in records:
        for proposition in record.propositions:
            merged.setdefault(proposition.id, proposition)
    return list(merged.values())


def _queue_record_dump(record: ReviewQueueRecord) -> dict[str, Any]:
    return record.model_dump(mode="json", exclude_none=True)


def _filter_with_cursor(
    queue_filter: ReviewQueueFilter,
    cursor: str | None,
) -> tuple[ReviewQueueFilter, Diagnostic | None]:
    if cursor is None:
        return queue_filter, None
    decoded = _decode_review_queue_cursor(cursor)
    if decoded is None:
        return queue_filter, Diagnostic(
            level="error",
            code="REVIEW_QUEUE_CURSOR_INVALID",
            message="Review queue cursor is invalid or expired.",
        )
    created_at, record_id = decoded
    return (
        queue_filter.model_copy(
            update={
                "after_created_at": created_at,
                "after_id": record_id,
            }
        ),
        None,
    )


def _encode_review_queue_cursor(record: ReviewQueueRecord) -> str:
    raw = json.dumps(
        {"created_at": record.created_at, "id": record.id},
        separators=(",", ":"),
    ).encode()
    return base64.urlsafe_b64encode(raw).decode().rstrip("=")


def _decode_review_queue_cursor(cursor: str) -> tuple[str, str] | None:
    try:
        padded = cursor + "=" * (-len(cursor) % 4)
        decoded = base64.urlsafe_b64decode(padded.encode()).decode()
        payload = json.loads(decoded)
    except (binascii.Error, UnicodeDecodeError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    created_at = str(payload.get("created_at", "")).strip()
    record_id = str(payload.get("id", "")).strip()
    if not created_at or not record_id:
        return None
    return created_at, record_id
