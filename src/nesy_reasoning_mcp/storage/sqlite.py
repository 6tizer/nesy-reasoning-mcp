"""SQLite relation store backend."""

from __future__ import annotations

import sqlite3
from collections.abc import Callable, Iterable, Mapping
from contextlib import suppress
from datetime import UTC, datetime
from functools import wraps
from pathlib import Path
from threading import RLock
from typing import Any, Concatenate, ParamSpec, TypeVar, cast

from nesy_reasoning_mcp.auto_ingest.scheduler import (
    ScheduledIngestionJob,
    ScheduledIngestionJobFilter,
    ScheduledIngestionJobStatus,
    ScheduledIngestionRun,
    ScheduledIngestionRunFilter,
    ScheduledIngestionState,
)
from nesy_reasoning_mcp.auto_ingest.schemas import (
    ConversationTurnJob,
    ConversationTurnJobFilter,
    ConversationTurnJobStatus,
    ReviewQueueFilter,
    ReviewQueueRecord,
    ReviewQueueStatus,
)
from nesy_reasoning_mcp.config import NesyConfig
from nesy_reasoning_mcp.normalization import normalize_relation_edges
from nesy_reasoning_mcp.schemas import (
    CanonicalImplicationEdge,
    ExclusiveGroupInput,
    ExclusiveGroupRecord,
    GraphStats,
    IndependenceRecord,
    PropositionRecord,
    RelationFilter,
    RelationInput,
    RelationRecord,
)
from nesy_reasoning_mcp.storage.audit import AuditEntry, _input_hash
from nesy_reasoning_mcp.storage.common import (
    _apply_assert_relations_mode,
    _dumps,
    _group_for_store,
    _group_matches_scope,
    _independence_for_store,
    _independence_from_row,
    _independence_matches_scope,
    _loads,
    _matches_filter,
    _merge_context_metadata,
    _merge_import,
    _merge_propositions,
    _normalize_relation_identities,
    _relation_for_store,
    _relation_from_row,
    _utc_now_iso,
    graph_stats_for,
)

P = ParamSpec("P")
R = TypeVar("R")

_SINGLE_KEY_SYNC_COLUMNS = {
    "relations": {"id"},
    "independence_records": {"id"},
    "context_metadata": {"context_id"},
    "propositions": {"id"},
}
_SYNC_TEMP_TABLES = {
    "desired_relation_ids",
    "desired_independence_ids",
    "desired_context_ids",
    "desired_proposition_ids",
}


def _locked(
    method: Callable[Concatenate[SqliteRelationStore, P], R],
) -> Callable[Concatenate[SqliteRelationStore, P], R]:
    """Serialize access to the shared SQLite connection."""

    @wraps(method)
    def wrapper(self: SqliteRelationStore, *args: P.args, **kwargs: P.kwargs) -> R:
        with self._lock:
            return method(self, *args, **kwargs)

    return cast(Any, wrapper)


def _relation_filter_sql(relation_filter: RelationFilter | None) -> tuple[str, list[Any]]:
    if relation_filter is None:
        return "", []
    clauses: list[str] = []
    params: list[Any] = []
    if relation_filter.source is not None:
        clauses.append("source = ?")
        params.append(relation_filter.source)
    if relation_filter.target is not None:
        clauses.append("target = ?")
        params.append(relation_filter.target)
    if relation_filter.relation_type is not None:
        clauses.append("relation_type = ?")
        params.append(relation_filter.relation_type.value)
    if relation_filter.context_id is not None:
        clauses.append("context_id = ?")
        params.append(relation_filter.context_id)
    if relation_filter.store_id is not None:
        clauses.append("store_id = ?")
        params.append(relation_filter.store_id)
    if relation_filter.domain is not None:
        clauses.append("json_extract(metadata_json, '$.domain') = ?")
        params.append(relation_filter.domain)
    return (f"WHERE {' AND '.join(clauses)}", params) if clauses else ("", params)


def _review_queue_filter_sql(queue_filter: ReviewQueueFilter | None) -> tuple[str, list[Any]]:
    if queue_filter is None:
        return "", []
    clauses: list[str] = []
    params: list[Any] = []
    if queue_filter.ids:
        clauses.append(f"id IN ({','.join('?' for _ in queue_filter.ids)})")
        params.extend(queue_filter.ids)
    if queue_filter.status is not None:
        clauses.append("status = ?")
        params.append(queue_filter.status.value)
    if queue_filter.run_id is not None:
        clauses.append("run_id = ?")
        params.append(queue_filter.run_id)
    if queue_filter.candidate_id is not None:
        clauses.append("candidate_id = ?")
        params.append(queue_filter.candidate_id)
    if queue_filter.store_id is not None:
        clauses.append("store_id = ?")
        params.append(queue_filter.store_id)
    if queue_filter.context_id is not None:
        clauses.append("context_id = ?")
        params.append(queue_filter.context_id)
    if queue_filter.after_created_at is not None and queue_filter.after_id is not None:
        clauses.append("(created_at > ? OR (created_at = ? AND id > ?))")
        params.extend(
            [
                queue_filter.after_created_at,
                queue_filter.after_created_at,
                queue_filter.after_id,
            ]
        )
    return (f"WHERE {' AND '.join(clauses)}", params) if clauses else ("", params)


def _ingestion_job_filter_sql(
    job_filter: ConversationTurnJobFilter | None,
) -> tuple[str, list[Any]]:
    if job_filter is None:
        return "", []
    clauses: list[str] = []
    params: list[Any] = []
    if job_filter.ids:
        clauses.append(f"job_id IN ({','.join('?' for _ in job_filter.ids)})")
        params.extend(job_filter.ids)
    if job_filter.status is not None:
        clauses.append("status = ?")
        params.append(job_filter.status.value)
    if job_filter.session_id is not None:
        clauses.append("session_id = ?")
        params.append(job_filter.session_id)
    if job_filter.agent_type is not None:
        clauses.append("agent_type = ?")
        params.append(job_filter.agent_type)
    return (f"WHERE {' AND '.join(clauses)}", params) if clauses else ("", params)


def _scheduled_job_filter_sql(
    job_filter: ScheduledIngestionJobFilter | None,
) -> tuple[str, list[Any]]:
    if job_filter is None:
        return "", []
    clauses: list[str] = []
    params: list[Any] = []
    if job_filter.ids:
        clauses.append(f"id IN ({','.join('?' for _ in job_filter.ids)})")
        params.extend(job_filter.ids)
    if job_filter.status is not None:
        clauses.append("status = ?")
        params.append(job_filter.status.value)
    if job_filter.due_before is not None:
        clauses.append("next_run_at IS NOT NULL AND next_run_at <= ?")
        params.append(job_filter.due_before)
    return (f"WHERE {' AND '.join(clauses)}", params) if clauses else ("", params)


def _scheduled_run_filter_sql(
    run_filter: ScheduledIngestionRunFilter | None,
) -> tuple[str, list[Any]]:
    if run_filter is None:
        return "", []
    clauses: list[str] = []
    params: list[Any] = []
    if run_filter.job_id is not None:
        clauses.append("job_id = ?")
        params.append(run_filter.job_id)
    if run_filter.status is not None:
        clauses.append("status = ?")
        params.append(run_filter.status.value)
    return (f"WHERE {' AND '.join(clauses)}", params) if clauses else ("", params)


def _require_unique(values: Iterable[str], label: str) -> None:
    seen: set[str] = set()
    for value in values:
        if value in seen:
            raise ValueError(f"duplicate {label}: {value}")
        seen.add(value)


def _checked_identifier(value: str, allowed: set[str], label: str) -> str:
    if value not in allowed:
        raise ValueError(f"unsupported SQL {label}: {value}")
    return value


def _review_queue_from_row(row: sqlite3.Row) -> ReviewQueueRecord:
    record = ReviewQueueRecord.model_validate(_loads(row["payload_json"], {}))
    expected_columns = {
        "id": record.id,
        "status": record.status.value,
        "run_id": record.run_id,
        "candidate_id": record.candidate.id,
        "context_id": record.candidate.context_id,
        "store_id": record.candidate.store_id,
        "next_retry_at": record.next_retry_at,
        "created_at": record.created_at,
        "updated_at": record.updated_at,
    }
    mismatched = [
        column for column, expected in expected_columns.items() if row[column] != expected
    ]
    if mismatched:
        raise ValueError(
            "review_queue indexed columns do not match payload: " + ", ".join(mismatched)
        )
    return record


def _ingestion_job_from_row(row: sqlite3.Row) -> ConversationTurnJob:
    record = ConversationTurnJob.model_validate(_loads(row["payload_json"], {}))
    expected_columns = {
        "job_id": record.job_id,
        "session_id": record.session_id,
        "transcript_path": record.transcript_path,
        "turn_index": record.turn_index,
        "priority": record.priority,
        "status": record.status.value,
        "agent_type": record.agent_type,
        "skip_extraction": 1 if record.skip_extraction else 0,
        "enqueued_at": record.enqueued_at,
        "updated_at": record.updated_at,
    }
    mismatched = [
        column for column, expected in expected_columns.items() if row[column] != expected
    ]
    if mismatched:
        raise ValueError(
            "ingestion_queue indexed columns do not match payload: " + ", ".join(mismatched)
        )
    return record


def _scheduled_job_from_row(row: sqlite3.Row) -> ScheduledIngestionJob:
    job = ScheduledIngestionJob.model_validate(_loads(row["payload_json"], {}))
    expected_columns = {
        "id": job.id,
        "name": job.name,
        "status": job.status.value,
        "next_run_at": job.state.next_run_at,
        "created_at": job.created_at,
        "updated_at": job.updated_at,
    }
    mismatched = [
        column for column, expected in expected_columns.items() if row[column] != expected
    ]
    if mismatched:
        raise ValueError(
            "scheduled_ingestion_jobs indexed columns do not match payload: "
            + ", ".join(mismatched)
        )
    return job


def _scheduled_run_from_row(row: sqlite3.Row) -> ScheduledIngestionRun:
    run = ScheduledIngestionRun.model_validate(_loads(row["payload_json"], {}))
    expected_columns = {
        "id": run.id,
        "job_id": run.job_id,
        "trigger": run.trigger.value,
        "status": run.status.value,
        "started_at": run.started_at,
        "finished_at": run.finished_at,
    }
    mismatched = [
        column for column, expected in expected_columns.items() if row[column] != expected
    ]
    if mismatched:
        raise ValueError(
            "scheduled_ingestion_runs indexed columns do not match payload: "
            + ", ".join(mismatched)
        )
    return run


class SqliteRelationStore:
    """SQLite source of truth for long-lived relation records."""

    def __init__(self, config: NesyConfig) -> None:
        self.config = config
        sqlite_path = config.storage.sqlite_path or "~/.nesy-reasoning/nesy.db"
        self.path = Path(sqlite_path).expanduser()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = RLock()
        self._conn = sqlite3.connect(self.path, timeout=30, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._configure_connection()
        self._initialize_schema()

    @_locked
    def _configure_connection(self) -> None:
        self._conn.execute("PRAGMA busy_timeout = 30000")
        self._conn.execute("PRAGMA journal_mode = WAL")

    @_locked
    def assert_relations(
        self,
        inputs: Iterable[RelationInput],
        *,
        mode: str = "append",
        dry_run: bool = False,
    ) -> tuple[list[RelationRecord], int]:
        """Add relation records and return added records plus update count."""
        normalized_inputs = _normalize_relation_identities(inputs, self.list_propositions())
        records = [RelationRecord.from_input(item) for item in normalized_inputs]
        # Append uses incremental insert below, so avoid loading a merged full-store view.
        current_relations = [] if mode == "append" else self.list_relations()
        merged, updated = _apply_assert_relations_mode(current_relations, records, mode)

        if dry_run:
            return records, updated

        if mode == "append":
            try:
                self._insert_relations(records)
                self._conn.commit()
            except Exception:
                self._conn.rollback()
                raise
        else:
            try:
                self._replace_all_records(
                    merged,
                    self.list_exclusive_groups(),
                    self.list_independence_records(),
                    self.context_metadata(),
                    self.list_propositions(),
                )
                self._conn.commit()
            except Exception:
                self._conn.rollback()
                raise

        return records, updated

    @_locked
    def assert_exclusive(
        self,
        inputs: Iterable[ExclusiveGroupInput],
    ) -> tuple[list[ExclusiveGroupRecord], int]:
        """Add or replace exclusive groups and return records plus update count."""
        records = [ExclusiveGroupRecord.from_input(item) for item in inputs]
        replace_keys = {(record.group_id, record.context_id, record.store_id) for record in records}
        existing = {
            (group.group_id, group.context_id, group.store_id)
            for group in self.list_exclusive_groups()
        }
        updated = sum(1 for key in replace_keys if key in existing)

        for group_id, context_id, store_id in replace_keys:
            self._conn.execute(
                """
                DELETE FROM exclusive_groups
                WHERE group_id = ? AND context_id = ? AND store_id = ?
                """,
                (group_id, context_id, store_id),
            )
        try:
            self._insert_exclusive_groups(records)
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise
        return records, updated

    @_locked
    def list_relations(
        self,
        relation_filter: RelationFilter | None = None,
        *,
        limit: int | None = None,
        offset: int = 0,
    ) -> list[RelationRecord]:
        """List relation records matching an optional filter."""
        where_sql, params = _relation_filter_sql(relation_filter)
        sql = f"""
            SELECT id, source, source_id, target, target_id, relation_type, polarity,
                   confidence, context_id, store_id, temporal_json, assumptions_json,
                   provenance_json, metadata_json,
                   created_at, updated_at
            FROM relations
            {where_sql}
            ORDER BY created_at, id
            """
        if limit is not None:
            sql += " LIMIT ? OFFSET ?"
            params.extend([limit, offset])
        elif offset:
            sql += " LIMIT -1 OFFSET ?"
            params.append(offset)
        rows = self._conn.execute(sql, params).fetchall()
        return [_relation_from_row(row) for row in rows]

    @_locked
    def list_exclusive_groups(self) -> list[ExclusiveGroupRecord]:
        """List all stored exclusive groups."""
        rows = self._conn.execute(
            """
            SELECT group_id, member, context_id, store_id, scope, member_index,
                   metadata_json, created_at, updated_at
            FROM exclusive_groups
            ORDER BY created_at, group_id, context_id, store_id, member_index
            """
        ).fetchall()
        grouped: dict[tuple[str, str, str], dict[str, Any]] = {}
        for row in rows:
            key = (row["group_id"], row["context_id"], row["store_id"])
            item = grouped.setdefault(
                key,
                {
                    "group_id": row["group_id"],
                    "members": [],
                    "context_id": row["context_id"],
                    "store_id": row["store_id"],
                    "scope": row["scope"],
                    "metadata": _loads(row["metadata_json"], {}),
                    "created_at": row["created_at"],
                    "updated_at": row["updated_at"],
                },
            )
            item["members"].append(row["member"])
        return [ExclusiveGroupRecord(**item) for item in grouped.values()]

    @_locked
    def list_independence_records(self) -> list[IndependenceRecord]:
        """List all stored independence records."""
        rows = self._conn.execute(
            """
            SELECT id, left_value, right_value, relation, confidence, context_id, store_id,
                   metadata_json, created_at, updated_at
            FROM independence_records
            ORDER BY created_at, id
            """
        ).fetchall()
        return [_independence_from_row(row) for row in rows]

    @_locked
    def list_propositions(self) -> list[PropositionRecord]:
        """List all stored proposition records."""
        rows = self._conn.execute(
            """
            SELECT id, label, aliases_json, negates, metadata_json
            FROM propositions
            ORDER BY id
            """
        ).fetchall()
        return [
            PropositionRecord(
                id=row["id"],
                label=row["label"],
                aliases=_loads(row["aliases_json"], []),
                negates=row["negates"],
                metadata=_loads(row["metadata_json"], {}),
            )
            for row in rows
        ]

    @_locked
    def enqueue_ingestion_jobs(
        self,
        records: Iterable[ConversationTurnJob],
    ) -> tuple[list[ConversationTurnJob], int]:
        """Add conversation turn ingestion jobs and return stored records plus update count."""
        incoming = [record.model_copy(deep=True) for record in records]
        queued = list({record.job_id: record for record in incoming}.values())
        if not queued:
            return [], 0
        existing_ids = {
            row["job_id"]
            for row in self._conn.execute(
                f"""
                SELECT job_id FROM ingestion_queue
                WHERE job_id IN ({",".join("?" for _ in queued)})
                """,
                [record.job_id for record in queued],
            ).fetchall()
        }
        try:
            self._upsert_ingestion_jobs(queued)
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise
        return queued, len(existing_ids)

    @_locked
    def list_ingestion_jobs(
        self,
        job_filter: ConversationTurnJobFilter | None = None,
        *,
        limit: int | None = None,
        offset: int = 0,
    ) -> list[ConversationTurnJob]:
        """List conversation turn ingestion jobs matching an optional filter."""
        where_sql, params = _ingestion_job_filter_sql(job_filter)
        sql = f"""
            SELECT *
            FROM ingestion_queue
            {where_sql}
            ORDER BY priority DESC, enqueued_at, job_id
            """
        if limit is not None:
            sql += " LIMIT ? OFFSET ?"
            params.extend([limit, offset])
        elif offset:
            sql += " LIMIT -1 OFFSET ?"
            params.append(offset)
        rows = self._conn.execute(sql, params).fetchall()
        return [_ingestion_job_from_row(row) for row in rows]

    @_locked
    def claim_pending_ingestion_jobs(
        self,
        *,
        limit: int = 1,
    ) -> list[ConversationTurnJob]:
        """Claim pending conversation turn jobs by moving them to extracting."""
        if limit < 0:
            raise ValueError("limit must be non-negative")
        if limit == 0:
            return []
        try:
            self._conn.execute("BEGIN IMMEDIATE")
            rows = self._conn.execute(
                """
                SELECT *
                FROM ingestion_queue
                WHERE status = ?
                ORDER BY priority DESC, enqueued_at, job_id
                LIMIT ?
                """,
                [ConversationTurnJobStatus.PENDING.value, limit],
            ).fetchall()
            timestamp = _utc_now_iso()
            selected = [
                _ingestion_job_from_row(row).model_copy(
                    deep=True,
                    update={
                        "status": ConversationTurnJobStatus.EXTRACTING,
                        "updated_at": timestamp,
                    },
                )
                for row in rows
            ]
            if not selected:
                self._conn.commit()
                return []
            self._update_ingestion_jobs(
                selected,
                expected_status=ConversationTurnJobStatus.PENDING,
            )
            claimed = self._ingestion_jobs_by_ids(
                [record.job_id for record in selected],
                status=ConversationTurnJobStatus.EXTRACTING,
                updated_at=timestamp,
            )
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise
        claimed.sort(key=lambda record: (-record.priority, record.enqueued_at, record.job_id))
        return claimed

    @_locked
    def update_ingestion_job_status(
        self,
        job_id: str,
        status: ConversationTurnJobStatus,
        *,
        expected_status: ConversationTurnJobStatus | None = None,
    ) -> ConversationTurnJob | None:
        """Update one conversation turn job status while preserving enqueue time."""
        try:
            self._conn.execute("BEGIN IMMEDIATE")
            rows = self._conn.execute(
                """
                SELECT *
                FROM ingestion_queue
                WHERE job_id = ?
                """,
                [job_id],
            ).fetchall()
            if not rows:
                self._conn.commit()
                return None
            record = _ingestion_job_from_row(rows[0])
            if expected_status is not None and record.status != expected_status:
                self._conn.commit()
                return None
            updated = record.model_copy(
                deep=True,
                update={"status": status, "updated_at": _utc_now_iso()},
            )
            changed = self._update_ingestion_jobs([updated], expected_status=expected_status)
            if changed == 0:
                self._conn.commit()
                return None
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise
        return updated

    @_locked
    def drop_pending_ingestion_jobs_over_depth(
        self,
        max_pending: int,
    ) -> list[ConversationTurnJob]:
        """Drop pending jobs beyond the configured queue depth."""
        if max_pending < 0:
            raise ValueError("max_pending must be non-negative")
        try:
            self._conn.execute("BEGIN IMMEDIATE")
            rows = self._conn.execute(
                """
                SELECT *
                FROM ingestion_queue
                WHERE status = ?
                ORDER BY priority DESC, enqueued_at DESC, job_id DESC
                """,
                [ConversationTurnJobStatus.PENDING.value],
            ).fetchall()
            pending = [_ingestion_job_from_row(row) for row in rows]
            dropped = pending[max_pending:]
            if not dropped:
                self._conn.commit()
                return []
            self._conn.executemany(
                """
                DELETE FROM ingestion_queue
                WHERE job_id = ? AND status = ?
                """,
                [(record.job_id, ConversationTurnJobStatus.PENDING.value) for record in dropped],
            )
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise
        return dropped

    @_locked
    def enqueue_review_queue(
        self,
        records: Iterable[ReviewQueueRecord],
    ) -> tuple[list[ReviewQueueRecord], int]:
        """Add review queue records and return stored records plus update count."""
        queued = [record.model_copy(deep=True) for record in records]
        if not queued:
            return [], 0
        existing_ids = {
            row["id"]
            for row in self._conn.execute(
                f"""
                SELECT id FROM review_queue
                WHERE id IN ({",".join("?" for _ in queued)})
                """,
                [record.id for record in queued],
            ).fetchall()
        }
        try:
            self._upsert_review_queue_records(queued)
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise
        return queued, len(existing_ids)

    @_locked
    def list_review_queue(
        self,
        queue_filter: ReviewQueueFilter | None = None,
        *,
        limit: int | None = None,
        offset: int = 0,
    ) -> list[ReviewQueueRecord]:
        """List review queue records matching an optional filter."""
        where_sql, params = _review_queue_filter_sql(queue_filter)
        sql = f"""
            SELECT *
            FROM review_queue
            {where_sql}
            ORDER BY created_at, id
            """
        if limit is not None:
            sql += " LIMIT ? OFFSET ?"
            params.extend([limit, offset])
        elif offset:
            sql += " LIMIT -1 OFFSET ?"
            params.append(offset)
        rows = self._conn.execute(sql, params).fetchall()
        return [_review_queue_from_row(row) for row in rows]

    @_locked
    def claim_pending_review_queue_records(
        self,
        *,
        limit: int = 1,
        now: str | None = None,
    ) -> list[ReviewQueueRecord]:
        """Claim due pending review queue records by moving them to reviewing."""
        if limit < 0:
            raise ValueError("limit must be non-negative")
        if limit == 0:
            return []
        due_before = now or _utc_now_iso()
        try:
            self._conn.execute("BEGIN IMMEDIATE")
            rows = self._conn.execute(
                """
                SELECT *
                FROM review_queue
                WHERE status = ?
                  AND (
                    (
                      next_retry_at IS NULL
                      AND COALESCE(json_extract(payload_json, '$.attempt_count'), 0) = 0
                    )
                    OR next_retry_at <= ?
                  )
                ORDER BY COALESCE(next_retry_at, ''), created_at, id
                LIMIT ?
                """,
                [ReviewQueueStatus.PENDING.value, due_before, limit],
            ).fetchall()
            timestamp = _utc_now_iso()
            records = [_review_queue_from_row(row) for row in rows]
            selected = [
                record.model_copy(
                    deep=True,
                    update={
                        "status": ReviewQueueStatus.REVIEWING,
                        "attempt_count": record.attempt_count + 1,
                        "updated_at": timestamp,
                    },
                )
                for record in records
            ]
            if not selected:
                self._conn.commit()
                return []
            self._update_review_queue_records(
                selected,
                expected_status=ReviewQueueStatus.PENDING,
            )
            claimed = self._review_queue_records_by_ids(
                [record.id for record in selected],
                status=ReviewQueueStatus.REVIEWING,
                updated_at=timestamp,
            )
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise
        claimed.sort(key=lambda record: (record.next_retry_at or "", record.created_at, record.id))
        return claimed

    @_locked
    def update_review_queue_records(
        self,
        records: Iterable[ReviewQueueRecord],
        *,
        expected_status: ReviewQueueStatus | None = None,
    ) -> int:
        """Update review queue records with an optional status compare-and-swap guard."""
        try:
            self._conn.execute("BEGIN IMMEDIATE")
            updated = self._update_review_queue_records(
                records,
                expected_status=expected_status,
            )
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise
        return updated

    @_locked
    def mark_review_queue_committed(
        self,
        ids: Iterable[str],
        relation_ids_by_record: Mapping[str, list[str]],
    ) -> int:
        """Mark review queue records as committed and return updated count."""
        records = self._review_queue_records_by_ids(ids)
        if not records:
            return 0
        updated_at = _utc_now_iso()
        updated_records = [
            record.model_copy(
                deep=True,
                update={
                    "status": ReviewQueueStatus.COMMITTED,
                    "updated_at": updated_at,
                    "committed_relation_ids": relation_ids_by_record.get(record.id, []),
                },
            )
            for record in records
        ]
        try:
            self._upsert_review_queue_records(updated_records)
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise
        return len(updated_records)

    @_locked
    def resolve_review_queue(
        self,
        ids: Iterable[str],
        *,
        reason: str,
        metadata: dict[str, Any] | None = None,
    ) -> int:
        """Resolve review queue records without writing graph relations."""
        records = self._review_queue_records_by_ids(ids)
        if not records:
            return 0
        updated_at = _utc_now_iso()
        resolution = {
            "reason": reason,
            "metadata": metadata or {},
            "resolved_at": updated_at,
        }
        updated_records = [
            record.model_copy(
                deep=True,
                update={
                    "status": ReviewQueueStatus.RESOLVED,
                    "updated_at": updated_at,
                    "resolution": resolution,
                },
            )
            for record in records
        ]
        try:
            self._upsert_review_queue_records(updated_records)
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise
        return len(updated_records)

    @_locked
    def upsert_scheduled_ingestion_job(
        self,
        job: ScheduledIngestionJob,
    ) -> tuple[ScheduledIngestionJob, int]:
        """Add or update one scheduled ingestion job."""
        stored = job.model_copy(deep=True, update={"updated_at": _utc_now_iso()})
        existing = self.get_scheduled_ingestion_job(stored.id)
        try:
            self._upsert_scheduled_ingestion_jobs([stored])
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise
        return stored, 1 if existing is not None else 0

    @_locked
    def list_scheduled_ingestion_jobs(
        self,
        job_filter: ScheduledIngestionJobFilter | None = None,
        *,
        limit: int | None = None,
        offset: int = 0,
    ) -> list[ScheduledIngestionJob]:
        """List scheduled ingestion jobs matching an optional filter."""
        where_sql, params = _scheduled_job_filter_sql(job_filter)
        sql = f"""
            SELECT *
            FROM scheduled_ingestion_jobs
            {where_sql}
            ORDER BY next_run_at, created_at, id
            """
        if limit is not None:
            sql += " LIMIT ? OFFSET ?"
            params.extend([limit, offset])
        elif offset:
            sql += " LIMIT -1 OFFSET ?"
            params.append(offset)
        rows = self._conn.execute(sql, params).fetchall()
        return [_scheduled_job_from_row(row) for row in rows]

    @_locked
    def get_scheduled_ingestion_job(self, job_id: str) -> ScheduledIngestionJob | None:
        """Return one scheduled ingestion job by id."""
        rows = self._conn.execute(
            """
            SELECT *
            FROM scheduled_ingestion_jobs
            WHERE id = ?
            """,
            (job_id,),
        ).fetchall()
        if not rows:
            return None
        return _scheduled_job_from_row(rows[0])

    @_locked
    def update_scheduled_ingestion_job_state(
        self,
        job_id: str,
        *,
        state: ScheduledIngestionState,
        status: ScheduledIngestionJobStatus | None = None,
        expected_status: ScheduledIngestionJobStatus | None = None,
    ) -> ScheduledIngestionJob | None:
        """Update mutable scheduled ingestion job state."""
        job = self.get_scheduled_ingestion_job(job_id)
        if job is None:
            return None
        updated = job.model_copy(
            deep=True,
            update={
                "status": status or job.status,
                "state": state,
                "updated_at": _utc_now_iso(),
            },
        )
        try:
            if expected_status is None:
                self._upsert_scheduled_ingestion_jobs([updated])
            else:
                cursor = self._conn.execute(
                    """
                    UPDATE scheduled_ingestion_jobs
                    SET
                        name = ?,
                        status = ?,
                        next_run_at = ?,
                        payload_json = ?,
                        created_at = ?,
                        updated_at = ?
                    WHERE id = ? AND status = ?
                    """,
                    (
                        updated.name,
                        updated.status.value,
                        updated.state.next_run_at,
                        _dumps(updated.model_dump(mode="json", exclude_none=True)),
                        updated.created_at,
                        updated.updated_at,
                        job_id,
                        expected_status.value,
                    ),
                )
                if cursor.rowcount == 0:
                    return None
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise
        return updated

    @_locked
    def append_scheduled_ingestion_run(
        self,
        run: ScheduledIngestionRun,
    ) -> ScheduledIngestionRun:
        """Append or replace one scheduled ingestion run record."""
        stored = run.model_copy(deep=True)
        try:
            self._upsert_scheduled_ingestion_runs([stored])
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise
        return stored

    @_locked
    def list_scheduled_ingestion_runs(
        self,
        run_filter: ScheduledIngestionRunFilter | None = None,
        *,
        limit: int | None = None,
        offset: int = 0,
    ) -> list[ScheduledIngestionRun]:
        """List scheduled ingestion runs matching an optional filter."""
        where_sql, params = _scheduled_run_filter_sql(run_filter)
        sql = f"""
            SELECT *
            FROM scheduled_ingestion_runs
            {where_sql}
            ORDER BY started_at, id
            """
        if limit is not None:
            sql += " LIMIT ? OFFSET ?"
            params.extend([limit, offset])
        elif offset:
            sql += " LIMIT -1 OFFSET ?"
            params.append(offset)
        rows = self._conn.execute(sql, params).fetchall()
        return [_scheduled_run_from_row(row) for row in rows]

    @_locked
    def clear_relations(
        self,
        *,
        scope: str,
        store_id: str,
        context_id: str,
        relation_filter: RelationFilter,
        dry_run: bool,
        include_exclusive_groups: bool = False,
    ) -> tuple[int, int]:
        """Remove relations and optionally exclusive groups by scope."""
        if scope == "all":
            removed = len(self.list_relations())
            removed_groups = len(self.list_exclusive_groups()) if include_exclusive_groups else 0
            if not dry_run:
                self._conn.execute("DELETE FROM relations")
                if include_exclusive_groups:
                    self._conn.execute("DELETE FROM exclusive_groups")
                self._conn.execute("DELETE FROM independence_records")
                self._conn.execute("DELETE FROM propositions")
                self._conn.execute("DELETE FROM context_metadata")
                self._conn.commit()
            return removed, removed_groups

        if scope not in {"store", "context", "filter"}:
            raise ValueError(f"unsupported clear scope: {scope}")

        def predicate(relation: RelationRecord) -> bool:
            if scope == "store":
                return relation.store_id == store_id
            if scope == "context":
                return relation.store_id == store_id and relation.context_id == context_id
            return _matches_filter(relation, relation_filter)

        def group_matches_clear(group: ExclusiveGroupRecord) -> bool:
            return _group_matches_scope(group, scope, store_id, context_id, relation_filter)

        relations = self.list_relations()
        groups = self.list_exclusive_groups()
        independence_records = self.list_independence_records()
        remove_ids = [relation.id for relation in relations if predicate(relation)]
        remove_group_keys = [
            (group.group_id, group.context_id, group.store_id)
            for group in groups
            if group_matches_clear(group)
        ]
        remove_independence_ids = [
            item.id
            for item in independence_records
            if _independence_matches_scope(item, scope, store_id, context_id, relation_filter)
        ]

        if not dry_run:
            self._delete_relation_ids(remove_ids)
            if include_exclusive_groups:
                self._delete_group_keys(remove_group_keys)
            self._delete_independence_ids(remove_independence_ids)
            if scope == "context":
                self._conn.execute(
                    "DELETE FROM context_metadata WHERE context_id = ?",
                    (context_id,),
                )
            self._conn.commit()

        return len(remove_ids), len(remove_group_keys) if include_exclusive_groups else 0

    @_locked
    def implication_edges(
        self,
        relations: Iterable[RelationRecord] | None = None,
    ) -> list[CanonicalImplicationEdge]:
        """Derive canonical implication edges from relation records."""
        selected = list(self.list_relations() if relations is None else relations)
        edges: list[CanonicalImplicationEdge] = []
        for relation in selected:
            edges.extend(normalize_relation_edges(relation))
        return edges

    @_locked
    def graph_stats(self) -> GraphStats:
        """Return statistics for the current graph."""
        relations = self.list_relations()
        return graph_stats_for(
            relations,
            self.implication_edges(relations),
            exclusive_group_count=len(self.list_exclusive_groups()),
        )

    @_locked
    def record_audit(
        self,
        *,
        event_type: str,
        tool_name: str,
        arguments: dict[str, Any],
        result_status: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Record one SQLite audit event."""
        if not self.config.logging.audit_log:
            return
        entry = AuditEntry(
            event_type=event_type,
            tool_name=tool_name,
            input_hash=_input_hash(arguments),
            result_status=result_status,
            metadata=metadata,
        )
        self._conn.execute(
            """
            INSERT INTO audit_log (
                id, event_type, tool_name, input_hash, result_status, created_at, metadata_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                entry.id,
                entry.event_type,
                entry.tool_name,
                entry.input_hash,
                entry.result_status,
                entry.created_at,
                _dumps(entry.metadata),
            ),
        )
        self._conn.commit()

    @_locked
    def list_audit_entries(self) -> list[dict[str, Any]]:
        """List SQLite audit entries."""
        rows = self._conn.execute(
            """
            SELECT id, event_type, tool_name, input_hash, result_status, created_at, metadata_json
            FROM audit_log
            ORDER BY created_at, id
            """
        ).fetchall()
        return [
            AuditEntry(
                entry_id=row["id"],
                event_type=row["event_type"],
                tool_name=row["tool_name"],
                input_hash=row["input_hash"],
                result_status=row["result_status"],
                created_at=row["created_at"],
                metadata=_loads(row["metadata_json"], {}),
            ).to_dict()
            for row in rows
        ]

    @_locked
    def context_metadata(self) -> dict[str, Any]:
        """Return SQLite graph context metadata."""
        rows = self._conn.execute(
            """
            SELECT context_id, metadata_json
            FROM context_metadata
            ORDER BY context_id
            """
        ).fetchall()
        return {row["context_id"]: _loads(row["metadata_json"], {}) for row in rows}

    @_locked
    def import_records(
        self,
        relations: Iterable[RelationRecord],
        exclusive_groups: Iterable[ExclusiveGroupRecord],
        independence_records: Iterable[IndependenceRecord] = (),
        propositions: Iterable[PropositionRecord] = (),
        *,
        mode: str,
        store_id: str,
        context_metadata: dict[str, Any] | None = None,
        dry_run: bool = False,
    ) -> tuple[int, int, int, int]:
        """Import validated records into SQLite."""
        merged_propositions, _updated_propositions = _merge_propositions(
            self.list_propositions(),
            propositions,
        )
        normalized_relations = _normalize_relation_identities(relations, merged_propositions)
        incoming_relations = [
            _relation_for_store(record, store_id) for record in normalized_relations
        ]
        incoming_groups = [_group_for_store(group, store_id) for group in exclusive_groups]
        incoming_independence = [
            _independence_for_store(record, store_id) for record in independence_records
        ]
        (
            merged_relations,
            merged_groups,
            merged_independence,
            updated_relations,
            updated_groups,
        ) = _merge_import(
            self.list_relations(),
            self.list_exclusive_groups(),
            self.list_independence_records(),
            incoming_relations,
            incoming_groups,
            incoming_independence,
            mode=mode,
            store_id=store_id,
        )
        merged_metadata = _merge_context_metadata(
            self.context_metadata(),
            context_metadata or {},
        )
        if not dry_run:
            try:
                self._replace_all_records(
                    merged_relations,
                    merged_groups,
                    merged_independence,
                    merged_metadata,
                    merged_propositions,
                )
                self._conn.commit()
            except Exception:
                self._conn.rollback()
                raise
        return len(incoming_relations), len(incoming_groups), updated_relations, updated_groups

    @_locked
    def _initialize_schema(self) -> None:
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS relations (
              id TEXT PRIMARY KEY,
              source TEXT NOT NULL,
              source_id TEXT,
              target TEXT NOT NULL,
              target_id TEXT,
              relation_type TEXT NOT NULL CHECK (
                relation_type IN ('sufficient','necessary','equivalent')
              ),
              polarity TEXT NOT NULL DEFAULT 'positive',
              confidence REAL NOT NULL DEFAULT 1.0 CHECK (
                confidence >= 0 AND confidence <= 1
              ),
              context_id TEXT NOT NULL DEFAULT 'default',
              store_id TEXT NOT NULL DEFAULT 'default',
              temporal_json TEXT,
              assumptions_json TEXT,
              provenance_json TEXT,
              metadata_json TEXT,
              created_at TEXT NOT NULL,
              updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS exclusive_groups (
              group_id TEXT NOT NULL,
              member TEXT NOT NULL,
              context_id TEXT NOT NULL DEFAULT 'default',
              store_id TEXT NOT NULL DEFAULT 'default',
              scope TEXT NOT NULL DEFAULT 'same_context',
              member_index INTEGER NOT NULL DEFAULT 0,
              metadata_json TEXT,
              created_at TEXT NOT NULL,
              updated_at TEXT NOT NULL,
              PRIMARY KEY (group_id, member, context_id, store_id)
            );

            CREATE TABLE IF NOT EXISTS audit_log (
              id TEXT PRIMARY KEY,
              event_type TEXT NOT NULL,
              tool_name TEXT,
              input_hash TEXT,
              result_status TEXT,
              created_at TEXT NOT NULL,
              metadata_json TEXT
            );

            CREATE TABLE IF NOT EXISTS context_metadata (
              context_id TEXT PRIMARY KEY,
              metadata_json TEXT NOT NULL,
              updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS independence_records (
              id TEXT PRIMARY KEY,
              left_value TEXT NOT NULL,
              right_value TEXT NOT NULL,
              relation TEXT NOT NULL DEFAULT 'independent_of' CHECK (
                relation = 'independent_of'
              ),
              confidence REAL NOT NULL DEFAULT 1.0 CHECK (
                confidence >= 0 AND confidence <= 1
              ),
              context_id TEXT NOT NULL DEFAULT 'default',
              store_id TEXT NOT NULL DEFAULT 'default',
              metadata_json TEXT,
              created_at TEXT NOT NULL,
              updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS propositions (
              id TEXT PRIMARY KEY,
              label TEXT NOT NULL,
              aliases_json TEXT,
              negates TEXT,
              metadata_json TEXT
            );

            CREATE TABLE IF NOT EXISTS review_queue (
              id TEXT PRIMARY KEY,
              status TEXT NOT NULL CHECK (
                status IN ('pending','reviewing','committed','resolved','failed')
              ),
              run_id TEXT NOT NULL,
              candidate_id TEXT NOT NULL,
              context_id TEXT NOT NULL,
              store_id TEXT NOT NULL,
              next_retry_at TEXT,
              payload_json TEXT NOT NULL,
              created_at TEXT NOT NULL,
              updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS ingestion_queue (
              job_id TEXT PRIMARY KEY,
              session_id TEXT NOT NULL,
              transcript_path TEXT NOT NULL,
              turn_index INTEGER,
              priority INTEGER NOT NULL DEFAULT 0,
              status TEXT NOT NULL CHECK (
                status IN ('pending','extracting','reviewing','done','failed')
              ),
              agent_type TEXT,
              skip_extraction INTEGER NOT NULL DEFAULT 0 CHECK (skip_extraction IN (0, 1)),
              payload_json TEXT NOT NULL,
              enqueued_at TEXT NOT NULL,
              updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS scheduled_ingestion_jobs (
              id TEXT PRIMARY KEY,
              name TEXT NOT NULL,
              status TEXT NOT NULL CHECK (status IN ('active','disabled','running','failed')),
              next_run_at TEXT,
              payload_json TEXT NOT NULL,
              created_at TEXT NOT NULL,
              updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS scheduled_ingestion_runs (
              id TEXT PRIMARY KEY,
              job_id TEXT NOT NULL,
              trigger TEXT NOT NULL CHECK (trigger IN ('manual','due','worker')),
              status TEXT NOT NULL CHECK (status IN ('running','succeeded','failed','skipped')),
              started_at TEXT NOT NULL,
              finished_at TEXT,
              payload_json TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_relations_context_store_type
              ON relations (context_id, store_id, relation_type);
            CREATE INDEX IF NOT EXISTS idx_relations_source_target
              ON relations (source, target);
            CREATE INDEX IF NOT EXISTS idx_relations_created_id
              ON relations (created_at, id);
            CREATE INDEX IF NOT EXISTS idx_relations_domain
              ON relations (json_extract(metadata_json, '$.domain'));
            CREATE INDEX IF NOT EXISTS idx_review_queue_status
              ON review_queue (status);
            CREATE INDEX IF NOT EXISTS idx_review_queue_run_id
              ON review_queue (run_id);
            CREATE INDEX IF NOT EXISTS idx_review_queue_candidate_id
              ON review_queue (candidate_id);
            CREATE INDEX IF NOT EXISTS idx_review_queue_context_store
              ON review_queue (context_id, store_id);
            CREATE INDEX IF NOT EXISTS idx_ingestion_queue_status_priority
              ON ingestion_queue (status, priority DESC, enqueued_at, job_id);
            CREATE INDEX IF NOT EXISTS idx_ingestion_queue_session
              ON ingestion_queue (session_id);
            CREATE INDEX IF NOT EXISTS idx_ingestion_queue_agent_type
              ON ingestion_queue (agent_type);
            CREATE INDEX IF NOT EXISTS idx_scheduled_ingestion_jobs_status
              ON scheduled_ingestion_jobs (status);
            CREATE INDEX IF NOT EXISTS idx_scheduled_ingestion_jobs_next_run
              ON scheduled_ingestion_jobs (next_run_at);
            CREATE INDEX IF NOT EXISTS idx_scheduled_ingestion_runs_job_id
              ON scheduled_ingestion_runs (job_id);
            CREATE INDEX IF NOT EXISTS idx_scheduled_ingestion_runs_started
              ON scheduled_ingestion_runs (started_at);
            """
        )
        self._ensure_review_queue_schema()
        self._conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_review_queue_status_retry
              ON review_queue (status, next_retry_at, created_at, id)
            """
        )
        self._ensure_relation_identity_columns()
        self._conn.commit()

    def _ensure_review_queue_schema(self) -> None:
        def table_exists(name: str) -> bool:
            return (
                self._conn.execute(
                    """
                    SELECT 1
                    FROM sqlite_master
                    WHERE type = 'table' AND name = ?
                    """,
                    (name,),
                ).fetchone()
                is not None
            )

        def schema_ready(name: str) -> bool:
            columns = {
                row["name"] for row in self._conn.execute(f"PRAGMA table_info({name})").fetchall()
            }
            row = self._conn.execute(
                """
                SELECT sql
                FROM sqlite_master
                WHERE type = 'table' AND name = ?
                """,
                (name,),
            ).fetchone()
            table_sql = row["sql"] if row is not None else ""
            return "next_retry_at" in columns and "reviewing" in table_sql and "failed" in table_sql

        def drop_indexes() -> None:
            for index_name in (
                "idx_review_queue_status",
                "idx_review_queue_status_retry",
                "idx_review_queue_run_id",
                "idx_review_queue_candidate_id",
                "idx_review_queue_context_store",
            ):
                self._conn.execute(f"DROP INDEX IF EXISTS {index_name}")

        def create_review_queue() -> None:
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS review_queue (
                  id TEXT PRIMARY KEY,
                  status TEXT NOT NULL CHECK (
                    status IN ('pending','reviewing','committed','resolved','failed')
                  ),
                  run_id TEXT NOT NULL,
                  candidate_id TEXT NOT NULL,
                  context_id TEXT NOT NULL,
                  store_id TEXT NOT NULL,
                  next_retry_at TEXT,
                  payload_json TEXT NOT NULL,
                  created_at TEXT NOT NULL,
                  updated_at TEXT NOT NULL
                )
                """
            )

        def copy_legacy_rows() -> None:
            self._conn.execute(
                """
                INSERT OR IGNORE INTO review_queue (
                  id, status, run_id, candidate_id, context_id, store_id,
                  next_retry_at, payload_json, created_at, updated_at
                )
                SELECT
                  id, status, run_id, candidate_id, context_id, store_id,
                  json_extract(payload_json, '$.next_retry_at'), payload_json, created_at,
                  updated_at
                FROM review_queue_legacy
                """
            )

        def create_indexes() -> None:
            self._conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_review_queue_status
                  ON review_queue (status)
                """
            )
            self._conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_review_queue_status_retry
                  ON review_queue (status, next_retry_at, created_at, id)
                """
            )
            self._conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_review_queue_run_id
                  ON review_queue (run_id)
                """
            )
            self._conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_review_queue_candidate_id
                  ON review_queue (candidate_id)
                """
            )
            self._conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_review_queue_context_store
                  ON review_queue (context_id, store_id)
                """
            )

        legacy_exists = table_exists("review_queue_legacy")
        if not legacy_exists and schema_ready("review_queue"):
            return
        self._conn.execute("SAVEPOINT review_queue_schema_migration")
        try:
            drop_indexes()
            if legacy_exists:
                if not schema_ready("review_queue"):
                    self._conn.execute("DROP TABLE IF EXISTS review_queue")
                    create_review_queue()
            else:
                self._conn.execute("ALTER TABLE review_queue RENAME TO review_queue_legacy")
                create_review_queue()
            copy_legacy_rows()
            self._conn.execute("DROP TABLE review_queue_legacy")
            create_indexes()
            self._conn.execute("RELEASE SAVEPOINT review_queue_schema_migration")
        except Exception:
            self._conn.execute("ROLLBACK TO SAVEPOINT review_queue_schema_migration")
            self._conn.execute("RELEASE SAVEPOINT review_queue_schema_migration")
            raise

    def _ensure_relation_identity_columns(self) -> None:
        columns = {
            row["name"] for row in self._conn.execute("PRAGMA table_info(relations)").fetchall()
        }
        if "source_id" not in columns:
            self._conn.execute("ALTER TABLE relations ADD COLUMN source_id TEXT")
        if "target_id" not in columns:
            self._conn.execute("ALTER TABLE relations ADD COLUMN target_id TEXT")

    def _insert_relations(self, records: Iterable[RelationRecord]) -> None:
        self._conn.executemany(
            """
            INSERT INTO relations (
                id, source, source_id, target, target_id, relation_type, polarity,
                confidence, context_id, store_id, temporal_json, assumptions_json,
                provenance_json, metadata_json,
                created_at, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    record.id,
                    record.source,
                    record.source_id,
                    record.target,
                    record.target_id,
                    record.relation_type.value,
                    record.polarity.value,
                    record.confidence,
                    record.context_id,
                    record.store_id,
                    _dumps(record.temporal),
                    _dumps(record.assumptions),
                    _dumps(record.provenance),
                    _dumps(record.metadata),
                    record.created_at,
                    record.updated_at,
                )
                for record in records
            ],
        )

    def _insert_exclusive_groups(self, records: Iterable[ExclusiveGroupRecord]) -> None:
        rows: list[tuple[str, str, str, str, str, int, str | None, str, str]] = []
        for record in records:
            rows.extend(
                (
                    record.group_id,
                    member,
                    record.context_id,
                    record.store_id,
                    record.scope.value,
                    index,
                    _dumps(record.metadata),
                    record.created_at,
                    record.updated_at,
                )
                for index, member in enumerate(record.members)
            )
        self._conn.executemany(
            """
            INSERT INTO exclusive_groups (
                group_id, member, context_id, store_id, scope, member_index,
                metadata_json, created_at, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )

    def _insert_independence_records(self, records: Iterable[IndependenceRecord]) -> None:
        self._conn.executemany(
            """
            INSERT INTO independence_records (
                id, left_value, right_value, relation, confidence, context_id, store_id,
                metadata_json, created_at, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    record.id,
                    record.left,
                    record.right,
                    record.relation,
                    record.confidence,
                    record.context_id,
                    record.store_id,
                    _dumps(record.metadata),
                    record.created_at,
                    record.updated_at,
                )
                for record in records
            ],
        )

    def _insert_propositions(self, records: Iterable[PropositionRecord]) -> None:
        self._conn.executemany(
            """
            INSERT INTO propositions (id, label, aliases_json, negates, metadata_json)
            VALUES (?, ?, ?, ?, ?)
            """,
            [
                (
                    record.id,
                    record.label,
                    _dumps(record.aliases),
                    record.negates,
                    _dumps(record.metadata),
                )
                for record in records
            ],
        )

    def _review_queue_records_by_ids(
        self,
        ids: Iterable[str],
        *,
        status: ReviewQueueStatus | None = None,
        updated_at: str | None = None,
    ) -> list[ReviewQueueRecord]:
        id_list = list(dict.fromkeys(ids))
        if not id_list:
            return []
        params: list[Any] = [*id_list]
        predicates = [f"id IN ({','.join('?' for _ in id_list)})"]
        if status is not None:
            predicates.append("status = ?")
            params.append(status.value)
        if updated_at is not None:
            predicates.append("updated_at = ?")
            params.append(updated_at)
        rows = self._conn.execute(
            f"""
            SELECT *
            FROM review_queue
            WHERE {" AND ".join(predicates)}
            ORDER BY created_at, id
            """,
            params,
        ).fetchall()
        return [_review_queue_from_row(row) for row in rows]

    def _upsert_ingestion_jobs(self, records: Iterable[ConversationTurnJob]) -> None:
        rows = []
        for record in records:
            rows.append(
                (
                    record.job_id,
                    record.session_id,
                    record.transcript_path,
                    record.turn_index,
                    record.priority,
                    record.status.value,
                    record.agent_type,
                    1 if record.skip_extraction else 0,
                    _dumps(record.model_dump(mode="json", exclude_none=True)),
                    record.enqueued_at,
                    record.updated_at,
                )
            )
        self._conn.executemany(
            """
            INSERT INTO ingestion_queue (
                job_id, session_id, transcript_path, turn_index, priority, status,
                agent_type, skip_extraction, payload_json, enqueued_at, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(job_id) DO UPDATE SET
                session_id = excluded.session_id,
                transcript_path = excluded.transcript_path,
                turn_index = excluded.turn_index,
                priority = excluded.priority,
                status = excluded.status,
                agent_type = excluded.agent_type,
                skip_extraction = excluded.skip_extraction,
                payload_json = excluded.payload_json,
                enqueued_at = excluded.enqueued_at,
                updated_at = excluded.updated_at
            """,
            rows,
        )

    def _ingestion_jobs_by_ids(
        self,
        ids: Iterable[str],
        *,
        status: ConversationTurnJobStatus | None = None,
        updated_at: str | None = None,
    ) -> list[ConversationTurnJob]:
        id_list = list(dict.fromkeys(ids))
        if not id_list:
            return []
        params: list[Any] = [*id_list]
        predicates = [f"job_id IN ({','.join('?' for _ in id_list)})"]
        if status is not None:
            predicates.append("status = ?")
            params.append(status.value)
        if updated_at is not None:
            predicates.append("updated_at = ?")
            params.append(updated_at)
        rows = self._conn.execute(
            f"""
            SELECT *
            FROM ingestion_queue
            WHERE {" AND ".join(predicates)}
            ORDER BY priority DESC, enqueued_at, job_id
            """,
            params,
        ).fetchall()
        return [_ingestion_job_from_row(row) for row in rows]

    def _update_ingestion_jobs(
        self,
        records: Iterable[ConversationTurnJob],
        *,
        expected_status: ConversationTurnJobStatus | None = None,
    ) -> int:
        rows = []
        for record in records:
            rows.append(
                (
                    record.session_id,
                    record.transcript_path,
                    record.turn_index,
                    record.priority,
                    record.status.value,
                    record.agent_type,
                    1 if record.skip_extraction else 0,
                    _dumps(record.model_dump(mode="json", exclude_none=True)),
                    record.updated_at,
                    record.job_id,
                    *([expected_status.value] if expected_status is not None else []),
                )
            )
        where_sql = "WHERE job_id = ?"
        if expected_status is not None:
            where_sql += " AND status = ?"
        cursor = self._conn.executemany(
            f"""
            UPDATE ingestion_queue
            SET
                session_id = ?,
                transcript_path = ?,
                turn_index = ?,
                priority = ?,
                status = ?,
                agent_type = ?,
                skip_extraction = ?,
                payload_json = ?,
                updated_at = ?
            {where_sql}
            """,
            rows,
        )
        return cursor.rowcount

    def _upsert_review_queue_records(self, records: Iterable[ReviewQueueRecord]) -> None:
        rows = []
        for record in records:
            rows.append(
                (
                    record.id,
                    record.status.value,
                    record.run_id,
                    record.candidate.id,
                    record.candidate.context_id,
                    record.candidate.store_id,
                    record.next_retry_at,
                    _dumps(record.model_dump(mode="json", exclude_none=True)),
                    record.created_at,
                    record.updated_at,
                )
            )
        self._conn.executemany(
            """
            INSERT INTO review_queue (
                id, status, run_id, candidate_id, context_id, store_id,
                next_retry_at, payload_json, created_at, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                status = excluded.status,
                run_id = excluded.run_id,
                candidate_id = excluded.candidate_id,
                context_id = excluded.context_id,
                store_id = excluded.store_id,
                next_retry_at = excluded.next_retry_at,
                payload_json = excluded.payload_json,
                created_at = excluded.created_at,
                updated_at = excluded.updated_at
            """,
            rows,
        )

    def _update_review_queue_records(
        self,
        records: Iterable[ReviewQueueRecord],
        *,
        expected_status: ReviewQueueStatus | None = None,
    ) -> int:
        rows = []
        for record in records:
            rows.append(
                (
                    record.status.value,
                    record.run_id,
                    record.candidate.id,
                    record.candidate.context_id,
                    record.candidate.store_id,
                    record.next_retry_at,
                    _dumps(record.model_dump(mode="json", exclude_none=True)),
                    record.updated_at,
                    record.id,
                    *([expected_status.value] if expected_status is not None else []),
                )
            )
        where_sql = "WHERE id = ?"
        if expected_status is not None:
            where_sql += " AND status = ?"
        sql = f"""
        UPDATE review_queue
        SET
            status = ?,
            run_id = ?,
            candidate_id = ?,
            context_id = ?,
            store_id = ?,
            next_retry_at = ?,
            payload_json = ?,
            updated_at = ?
        {where_sql}
        """
        return sum(self._conn.execute(sql, row).rowcount for row in rows)

    def _upsert_scheduled_ingestion_jobs(
        self,
        jobs: Iterable[ScheduledIngestionJob],
    ) -> None:
        rows = []
        for job in jobs:
            rows.append(
                (
                    job.id,
                    job.name,
                    job.status.value,
                    job.state.next_run_at,
                    _dumps(job.model_dump(mode="json", exclude_none=True)),
                    job.created_at,
                    job.updated_at,
                )
            )
        self._conn.executemany(
            """
            INSERT INTO scheduled_ingestion_jobs (
                id, name, status, next_run_at, payload_json, created_at, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                name = excluded.name,
                status = excluded.status,
                next_run_at = excluded.next_run_at,
                payload_json = excluded.payload_json,
                created_at = excluded.created_at,
                updated_at = excluded.updated_at
            """,
            rows,
        )

    def _upsert_scheduled_ingestion_runs(
        self,
        runs: Iterable[ScheduledIngestionRun],
    ) -> None:
        rows = []
        for run in runs:
            rows.append(
                (
                    run.id,
                    run.job_id,
                    run.trigger.value,
                    run.status.value,
                    run.started_at,
                    run.finished_at,
                    _dumps(run.model_dump(mode="json", exclude_none=True)),
                )
            )
        self._conn.executemany(
            """
            INSERT INTO scheduled_ingestion_runs (
                id, job_id, trigger, status, started_at, finished_at, payload_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                job_id = excluded.job_id,
                trigger = excluded.trigger,
                status = excluded.status,
                started_at = excluded.started_at,
                finished_at = excluded.finished_at,
                payload_json = excluded.payload_json
            """,
            rows,
        )

    def _delete_relation_ids(self, relation_ids: Iterable[str]) -> None:
        self._conn.executemany(
            "DELETE FROM relations WHERE id = ?",
            [(relation_id,) for relation_id in relation_ids],
        )

    def _delete_group_keys(self, group_keys: Iterable[tuple[str, str, str]]) -> None:
        self._conn.executemany(
            """
            DELETE FROM exclusive_groups
            WHERE group_id = ? AND context_id = ? AND store_id = ?
            """,
            list(group_keys),
        )

    def _delete_independence_ids(self, record_ids: Iterable[str]) -> None:
        self._conn.executemany(
            "DELETE FROM independence_records WHERE id = ?",
            [(record_id,) for record_id in record_ids],
        )

    def _replace_all_records(
        self,
        relations: Iterable[RelationRecord],
        groups: Iterable[ExclusiveGroupRecord],
        independence_records: Iterable[IndependenceRecord],
        context_metadata: dict[str, Any],
        propositions: Iterable[PropositionRecord],
    ) -> None:
        relation_list = list(relations)
        group_list = list(groups)
        independence_list = list(independence_records)
        proposition_list = list(propositions)
        _require_unique((record.id for record in relation_list), "relation id")
        _require_unique((record.id for record in independence_list), "independence id")
        _require_unique((record.id for record in proposition_list), "proposition id")

        self._sync_single_key_table(
            "relations",
            "id",
            [record.id for record in relation_list],
            "desired_relation_ids",
        )
        self._sync_exclusive_groups(group_list)
        self._sync_single_key_table(
            "independence_records",
            "id",
            [record.id for record in independence_list],
            "desired_independence_ids",
        )
        self._sync_single_key_table(
            "context_metadata",
            "context_id",
            list(context_metadata),
            "desired_context_ids",
        )
        self._sync_single_key_table(
            "propositions",
            "id",
            [record.id for record in proposition_list],
            "desired_proposition_ids",
        )
        self._upsert_relation_records(relation_list)
        self._upsert_exclusive_group_records(group_list)
        self._upsert_independence_record_rows(independence_list)
        self._upsert_context_metadata(context_metadata)
        self._upsert_proposition_records(proposition_list)

    def _sync_single_key_table(
        self,
        table: str,
        key_column: str,
        desired_keys: list[str],
        temp_table: str,
    ) -> None:
        table = _checked_identifier(table, set(_SINGLE_KEY_SYNC_COLUMNS), "table")
        key_column = _checked_identifier(
            key_column,
            _SINGLE_KEY_SYNC_COLUMNS[table],
            "column",
        )
        temp_table = _checked_identifier(temp_table, _SYNC_TEMP_TABLES, "temp table")
        try:
            self._conn.execute(f"DROP TABLE IF EXISTS {temp_table}")
            self._conn.execute(f"CREATE TEMP TABLE {temp_table} (value TEXT PRIMARY KEY)")
            self._conn.executemany(
                f"INSERT INTO {temp_table} (value) VALUES (?)",
                [(key,) for key in desired_keys],
            )
            self._conn.execute(
                f"""
                DELETE FROM {table}
                WHERE NOT EXISTS (
                  SELECT 1 FROM {temp_table}
                  WHERE {temp_table}.value = {table}.{key_column}
                )
                """
            )
        finally:
            with suppress(sqlite3.Error):
                self._conn.execute(f"DROP TABLE {temp_table}")

    def _sync_exclusive_groups(self, records: list[ExclusiveGroupRecord]) -> None:
        temp_table = "desired_exclusive_group_rows"
        try:
            self._conn.execute(f"DROP TABLE IF EXISTS {temp_table}")
            self._conn.execute(
                """
                CREATE TEMP TABLE desired_exclusive_group_rows (
                  group_id TEXT NOT NULL,
                  member TEXT NOT NULL,
                  context_id TEXT NOT NULL,
                  store_id TEXT NOT NULL,
                  PRIMARY KEY (group_id, member, context_id, store_id)
                )
                """
            )
            self._conn.executemany(
                """
                INSERT INTO desired_exclusive_group_rows (
                    group_id, member, context_id, store_id
                )
                VALUES (?, ?, ?, ?)
                """,
                [
                    (record.group_id, member, record.context_id, record.store_id)
                    for record in records
                    for member in record.members
                ],
            )
            self._conn.execute(
                """
                DELETE FROM exclusive_groups
                WHERE NOT EXISTS (
                  SELECT 1 FROM desired_exclusive_group_rows desired
                  WHERE desired.group_id = exclusive_groups.group_id
                    AND desired.member = exclusive_groups.member
                    AND desired.context_id = exclusive_groups.context_id
                    AND desired.store_id = exclusive_groups.store_id
                )
                """
            )
        finally:
            with suppress(sqlite3.Error):
                self._conn.execute(f"DROP TABLE {temp_table}")

    def _upsert_relation_records(self, records: Iterable[RelationRecord]) -> None:
        self._conn.executemany(
            """
            INSERT INTO relations (
                id, source, source_id, target, target_id, relation_type, polarity,
                confidence, context_id, store_id, temporal_json, assumptions_json,
                provenance_json, metadata_json,
                created_at, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                source = excluded.source,
                source_id = excluded.source_id,
                target = excluded.target,
                target_id = excluded.target_id,
                relation_type = excluded.relation_type,
                polarity = excluded.polarity,
                confidence = excluded.confidence,
                context_id = excluded.context_id,
                store_id = excluded.store_id,
                temporal_json = excluded.temporal_json,
                assumptions_json = excluded.assumptions_json,
                provenance_json = excluded.provenance_json,
                metadata_json = excluded.metadata_json,
                created_at = excluded.created_at,
                updated_at = excluded.updated_at
            """,
            [
                (
                    record.id,
                    record.source,
                    record.source_id,
                    record.target,
                    record.target_id,
                    record.relation_type.value,
                    record.polarity.value,
                    record.confidence,
                    record.context_id,
                    record.store_id,
                    _dumps(record.temporal),
                    _dumps(record.assumptions),
                    _dumps(record.provenance),
                    _dumps(record.metadata),
                    record.created_at,
                    record.updated_at,
                )
                for record in records
            ],
        )

    def _upsert_exclusive_group_records(self, records: Iterable[ExclusiveGroupRecord]) -> None:
        rows: list[tuple[str, str, str, str, str, int, str | None, str, str]] = []
        for record in records:
            rows.extend(
                (
                    record.group_id,
                    member,
                    record.context_id,
                    record.store_id,
                    record.scope.value,
                    index,
                    _dumps(record.metadata),
                    record.created_at,
                    record.updated_at,
                )
                for index, member in enumerate(record.members)
            )
        self._conn.executemany(
            """
            INSERT INTO exclusive_groups (
                group_id, member, context_id, store_id, scope, member_index,
                metadata_json, created_at, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(group_id, member, context_id, store_id) DO UPDATE SET
                scope = excluded.scope,
                member_index = excluded.member_index,
                metadata_json = excluded.metadata_json,
                created_at = excluded.created_at,
                updated_at = excluded.updated_at
            """,
            rows,
        )

    def _upsert_independence_record_rows(self, records: Iterable[IndependenceRecord]) -> None:
        self._conn.executemany(
            """
            INSERT INTO independence_records (
                id, left_value, right_value, relation, confidence, context_id, store_id,
                metadata_json, created_at, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                left_value = excluded.left_value,
                right_value = excluded.right_value,
                relation = excluded.relation,
                confidence = excluded.confidence,
                context_id = excluded.context_id,
                store_id = excluded.store_id,
                metadata_json = excluded.metadata_json,
                created_at = excluded.created_at,
                updated_at = excluded.updated_at
            """,
            [
                (
                    record.id,
                    record.left,
                    record.right,
                    record.relation,
                    record.confidence,
                    record.context_id,
                    record.store_id,
                    _dumps(record.metadata),
                    record.created_at,
                    record.updated_at,
                )
                for record in records
            ],
        )

    def _upsert_context_metadata(self, context_metadata: dict[str, Any]) -> None:
        timestamp = datetime.now(UTC).isoformat()
        self._conn.executemany(
            """
            INSERT INTO context_metadata (context_id, metadata_json, updated_at)
            VALUES (?, ?, ?)
            ON CONFLICT(context_id) DO UPDATE SET
                metadata_json = excluded.metadata_json,
                updated_at = excluded.updated_at
            """,
            [
                (context_id, _dumps(metadata), timestamp)
                for context_id, metadata in sorted(context_metadata.items())
            ],
        )

    def _upsert_proposition_records(self, records: Iterable[PropositionRecord]) -> None:
        self._conn.executemany(
            """
            INSERT INTO propositions (id, label, aliases_json, negates, metadata_json)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                label = excluded.label,
                aliases_json = excluded.aliases_json,
                negates = excluded.negates,
                metadata_json = excluded.metadata_json
            """,
            [
                (
                    record.id,
                    record.label,
                    _dumps(record.aliases),
                    record.negates,
                    _dumps(record.metadata),
                )
                for record in records
            ],
        )

    def _insert_context_metadata(self, context_metadata: dict[str, Any]) -> None:
        timestamp = datetime.now(UTC).isoformat()
        self._conn.executemany(
            """
            INSERT INTO context_metadata (context_id, metadata_json, updated_at)
            VALUES (?, ?, ?)
            """,
            [
                (context_id, _dumps(metadata), timestamp)
                for context_id, metadata in sorted(context_metadata.items())
            ],
        )
