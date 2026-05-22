"""Relation, exclusive group, and audit storage."""

from __future__ import annotations

import json
import os
import sqlite3
from collections.abc import Iterable
from copy import deepcopy
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import Any, Protocol
from uuid import uuid4

from nesy_reasoning_mcp.config import NesyConfig, StorageBackend, load_config
from nesy_reasoning_mcp.schemas import (
    CanonicalImplicationEdge,
    ExclusiveGroupInput,
    ExclusiveGroupRecord,
    GraphStats,
    IndependenceRecord,
    RelationFilter,
    RelationInput,
    RelationRecord,
    RelationType,
)


class AuditEntry:
    """Audit log entry for mutating operations."""

    def __init__(
        self,
        *,
        event_type: str,
        tool_name: str,
        input_hash: str,
        result_status: str,
        metadata: dict[str, Any] | None = None,
        entry_id: str | None = None,
        created_at: str | None = None,
    ) -> None:
        self.id = entry_id or f"audit_{uuid4().hex}"
        self.event_type = event_type
        self.tool_name = tool_name
        self.input_hash = input_hash
        self.result_status = result_status
        self.created_at = created_at or datetime.now(UTC).isoformat()
        self.metadata = metadata or {}

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-serializable audit data."""
        return {
            "id": self.id,
            "event_type": self.event_type,
            "tool_name": self.tool_name,
            "input_hash": self.input_hash,
            "result_status": self.result_status,
            "created_at": self.created_at,
            "metadata": self.metadata,
        }


class RelationStoreProtocol(Protocol):
    """Storage interface used by MCP tool handlers."""

    config: NesyConfig

    def assert_relations(
        self,
        inputs: Iterable[RelationInput],
        *,
        mode: str = "append",
        dry_run: bool = False,
    ) -> tuple[list[RelationRecord], int]:
        """Add relation records and return added records plus update count."""

    def assert_exclusive(
        self,
        inputs: Iterable[ExclusiveGroupInput],
    ) -> tuple[list[ExclusiveGroupRecord], int]:
        """Add or replace exclusive groups and return records plus update count."""

    def list_relations(
        self,
        relation_filter: RelationFilter | None = None,
        *,
        limit: int | None = None,
    ) -> list[RelationRecord]:
        """List relation records matching an optional filter."""

    def list_exclusive_groups(self) -> list[ExclusiveGroupRecord]:
        """List all stored exclusive groups."""

    def list_independence_records(self) -> list[IndependenceRecord]:
        """List all stored independence records."""

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

    def implication_edges(
        self,
        relations: Iterable[RelationRecord] | None = None,
    ) -> list[CanonicalImplicationEdge]:
        """Derive canonical implication edges from relation records."""

    def graph_stats(self) -> GraphStats:
        """Return statistics for the current graph."""

    def record_audit(
        self,
        *,
        event_type: str,
        tool_name: str,
        arguments: dict[str, Any],
        result_status: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Record one audit event."""

    def list_audit_entries(self) -> list[dict[str, Any]]:
        """List audit entries."""

    def context_metadata(self) -> dict[str, Any]:
        """Return graph context metadata."""

    def import_records(
        self,
        relations: Iterable[RelationRecord],
        exclusive_groups: Iterable[ExclusiveGroupRecord],
        independence_records: Iterable[IndependenceRecord] = (),
        *,
        mode: str,
        store_id: str,
        context_metadata: dict[str, Any] | None = None,
        dry_run: bool = False,
    ) -> tuple[int, int, int, int]:
        """Import validated records into the store."""


class MemoryRelationStore:
    """Process-local in-memory source of truth for relation records."""

    def __init__(self, config: NesyConfig | None = None) -> None:
        self.config = config or load_config()
        self._relations: list[RelationRecord] = []
        self._exclusive_groups: list[ExclusiveGroupRecord] = []
        self._independence_records: list[IndependenceRecord] = []
        self._audit_log: list[AuditEntry] = []
        self._context_metadata: dict[str, Any] = {}

    def assert_relations(
        self,
        inputs: Iterable[RelationInput],
        *,
        mode: str = "append",
        dry_run: bool = False,
    ) -> tuple[list[RelationRecord], int]:
        """Add relation records and return added records plus update count."""
        records = [RelationRecord.from_input(item) for item in inputs]
        updated = 0

        if mode == "replace_same_pair":
            replace_keys = {
                (record.source, record.target, record.context_id, record.store_id)
                for record in records
            }
            updated = sum(
                1
                for relation in self._relations
                if (relation.source, relation.target, relation.context_id, relation.store_id)
                in replace_keys
            )
            if not dry_run:
                self._relations = [
                    relation
                    for relation in self._relations
                    if (relation.source, relation.target, relation.context_id, relation.store_id)
                    not in replace_keys
                ]

        if not dry_run:
            self._relations.extend(records)

        return records, updated

    def assert_exclusive(
        self,
        inputs: Iterable[ExclusiveGroupInput],
    ) -> tuple[list[ExclusiveGroupRecord], int]:
        """Add or replace exclusive groups and return records plus update count."""
        records = [ExclusiveGroupRecord.from_input(item) for item in inputs]
        replace_keys = {(record.group_id, record.context_id, record.store_id) for record in records}
        updated = sum(
            1
            for group in self._exclusive_groups
            if (group.group_id, group.context_id, group.store_id) in replace_keys
        )
        self._exclusive_groups = [
            group
            for group in self._exclusive_groups
            if (group.group_id, group.context_id, group.store_id) not in replace_keys
        ]
        self._exclusive_groups.extend(records)
        return records, updated

    def list_relations(
        self,
        relation_filter: RelationFilter | None = None,
        *,
        limit: int | None = None,
    ) -> list[RelationRecord]:
        """List relation records matching an optional filter."""
        matched = [
            relation
            for relation in self._relations
            if relation_filter is None or _matches_filter(relation, relation_filter)
        ]
        if limit is not None:
            return matched[:limit]
        return matched

    def list_exclusive_groups(self) -> list[ExclusiveGroupRecord]:
        """List all stored exclusive groups."""
        return list(self._exclusive_groups)

    def list_independence_records(self) -> list[IndependenceRecord]:
        """List all stored independence records."""
        return list(self._independence_records)

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
            removed = len(self._relations)
            removed_groups = len(self._exclusive_groups) if include_exclusive_groups else 0
            if not dry_run:
                self._relations.clear()
                if include_exclusive_groups:
                    self._exclusive_groups.clear()
                self._independence_records.clear()
                self._context_metadata.clear()
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

        removed = sum(1 for relation in self._relations if predicate(relation))
        removed_groups = (
            sum(1 for group in self._exclusive_groups if group_matches_clear(group))
            if include_exclusive_groups
            else 0
        )
        if not dry_run:
            self._relations = [relation for relation in self._relations if not predicate(relation)]
            if include_exclusive_groups:
                self._exclusive_groups = [
                    group for group in self._exclusive_groups if not group_matches_clear(group)
                ]
            self._independence_records = [
                item
                for item in self._independence_records
                if not _independence_matches_scope(
                    item,
                    scope,
                    store_id,
                    context_id,
                    relation_filter,
                )
            ]
            if scope == "context":
                self._context_metadata.pop(context_id, None)
        return removed, removed_groups

    def implication_edges(
        self,
        relations: Iterable[RelationRecord] | None = None,
    ) -> list[CanonicalImplicationEdge]:
        """Derive canonical implication edges from relation records."""
        selected = list(self._relations if relations is None else relations)
        edges: list[CanonicalImplicationEdge] = []
        for relation in selected:
            if relation.relation_type == RelationType.SUFFICIENT:
                edges.append(_edge(relation, relation.source, relation.target, "a"))
            elif relation.relation_type == RelationType.NECESSARY:
                edges.append(_edge(relation, relation.target, relation.source, "a"))
            elif relation.relation_type == RelationType.EQUIVALENT:
                edges.append(_edge(relation, relation.source, relation.target, "a"))
                edges.append(_edge(relation, relation.target, relation.source, "b"))
        return edges

    def graph_stats(self) -> GraphStats:
        """Return statistics for the current graph."""
        return graph_stats_for(
            self._relations,
            self.implication_edges(),
            exclusive_group_count=len(self._exclusive_groups),
        )

    def record_audit(
        self,
        *,
        event_type: str,
        tool_name: str,
        arguments: dict[str, Any],
        result_status: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Record one in-memory audit event."""
        if not self.config.logging.audit_log:
            return
        self._audit_log.append(
            AuditEntry(
                event_type=event_type,
                tool_name=tool_name,
                input_hash=_input_hash(arguments),
                result_status=result_status,
                metadata=metadata,
            )
        )

    def list_audit_entries(self) -> list[dict[str, Any]]:
        """List in-memory audit entries."""
        return [entry.to_dict() for entry in self._audit_log]

    def context_metadata(self) -> dict[str, Any]:
        """Return a copy of in-memory graph context metadata."""
        return deepcopy(self._context_metadata)

    def import_records(
        self,
        relations: Iterable[RelationRecord],
        exclusive_groups: Iterable[ExclusiveGroupRecord],
        independence_records: Iterable[IndependenceRecord] = (),
        *,
        mode: str,
        store_id: str,
        context_metadata: dict[str, Any] | None = None,
        dry_run: bool = False,
    ) -> tuple[int, int, int, int]:
        """Import validated records into memory."""
        incoming_relations = [_relation_for_store(record, store_id) for record in relations]
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
            self._relations,
            self._exclusive_groups,
            self._independence_records,
            incoming_relations,
            incoming_groups,
            incoming_independence,
            mode=mode,
            store_id=store_id,
        )
        merged_metadata = _merge_context_metadata(
            self._context_metadata,
            context_metadata or {},
        )
        if not dry_run:
            self._relations = merged_relations
            self._exclusive_groups = merged_groups
            self._independence_records = merged_independence
            self._context_metadata = merged_metadata
        return len(incoming_relations), len(incoming_groups), updated_relations, updated_groups


RelationStore = MemoryRelationStore


class SqliteRelationStore:
    """SQLite source of truth for long-lived relation records."""

    def __init__(self, config: NesyConfig) -> None:
        self.config = config
        sqlite_path = config.storage.sqlite_path or "~/.nesy-reasoning/nesy.db"
        self.path = Path(sqlite_path).expanduser()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.path, timeout=30)
        self._conn.row_factory = sqlite3.Row
        self._initialize_schema()

    def assert_relations(
        self,
        inputs: Iterable[RelationInput],
        *,
        mode: str = "append",
        dry_run: bool = False,
    ) -> tuple[list[RelationRecord], int]:
        """Add relation records and return added records plus update count."""
        records = [RelationRecord.from_input(item) for item in inputs]
        updated = 0

        if mode == "replace_same_pair":
            replace_keys = {
                (record.source, record.target, record.context_id, record.store_id)
                for record in records
            }
            updated = sum(
                1
                for relation in self.list_relations()
                if (relation.source, relation.target, relation.context_id, relation.store_id)
                in replace_keys
            )
            if not dry_run:
                for source, target, context_id, store_id in replace_keys:
                    self._conn.execute(
                        """
                        DELETE FROM relations
                        WHERE source = ? AND target = ? AND context_id = ? AND store_id = ?
                        """,
                        (source, target, context_id, store_id),
                    )

        if not dry_run:
            try:
                self._insert_relations(records)
                self._conn.commit()
            except Exception:
                self._conn.rollback()
                raise

        return records, updated

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

    def list_relations(
        self,
        relation_filter: RelationFilter | None = None,
        *,
        limit: int | None = None,
    ) -> list[RelationRecord]:
        """List relation records matching an optional filter."""
        rows = self._conn.execute(
            """
            SELECT id, source, target, relation_type, polarity, confidence, context_id, store_id,
                   temporal_json, assumptions_json, provenance_json, metadata_json,
                   created_at, updated_at
            FROM relations
            ORDER BY created_at, id
            """
        ).fetchall()
        records = [_relation_from_row(row) for row in rows]
        matched = [
            relation
            for relation in records
            if relation_filter is None or _matches_filter(relation, relation_filter)
        ]
        if limit is not None:
            return matched[:limit]
        return matched

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

    def implication_edges(
        self,
        relations: Iterable[RelationRecord] | None = None,
    ) -> list[CanonicalImplicationEdge]:
        """Derive canonical implication edges from relation records."""
        selected = list(self.list_relations() if relations is None else relations)
        edges: list[CanonicalImplicationEdge] = []
        for relation in selected:
            if relation.relation_type == RelationType.SUFFICIENT:
                edges.append(_edge(relation, relation.source, relation.target, "a"))
            elif relation.relation_type == RelationType.NECESSARY:
                edges.append(_edge(relation, relation.target, relation.source, "a"))
            elif relation.relation_type == RelationType.EQUIVALENT:
                edges.append(_edge(relation, relation.source, relation.target, "a"))
                edges.append(_edge(relation, relation.target, relation.source, "b"))
        return edges

    def graph_stats(self) -> GraphStats:
        """Return statistics for the current graph."""
        relations = self.list_relations()
        return graph_stats_for(
            relations,
            self.implication_edges(relations),
            exclusive_group_count=len(self.list_exclusive_groups()),
        )

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

    def import_records(
        self,
        relations: Iterable[RelationRecord],
        exclusive_groups: Iterable[ExclusiveGroupRecord],
        independence_records: Iterable[IndependenceRecord] = (),
        *,
        mode: str,
        store_id: str,
        context_metadata: dict[str, Any] | None = None,
        dry_run: bool = False,
    ) -> tuple[int, int, int, int]:
        """Import validated records into SQLite."""
        incoming_relations = [_relation_for_store(record, store_id) for record in relations]
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
                )
                self._conn.commit()
            except Exception:
                self._conn.rollback()
                raise
        return len(incoming_relations), len(incoming_groups), updated_relations, updated_groups

    def _initialize_schema(self) -> None:
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS relations (
              id TEXT PRIMARY KEY,
              source TEXT NOT NULL,
              target TEXT NOT NULL,
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
            """
        )
        self._conn.commit()

    def _insert_relations(self, records: Iterable[RelationRecord]) -> None:
        self._conn.executemany(
            """
            INSERT INTO relations (
                id, source, target, relation_type, polarity, confidence, context_id, store_id,
                temporal_json, assumptions_json, provenance_json, metadata_json,
                created_at, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    record.id,
                    record.source,
                    record.target,
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
        rows = []
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
    ) -> None:
        self._conn.execute("DELETE FROM relations")
        self._conn.execute("DELETE FROM exclusive_groups")
        self._conn.execute("DELETE FROM independence_records")
        self._conn.execute("DELETE FROM context_metadata")
        self._insert_relations(relations)
        self._insert_exclusive_groups(groups)
        self._insert_independence_records(independence_records)
        self._insert_context_metadata(context_metadata)

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


class JsonRelationStore(MemoryRelationStore):
    """JSON-file source of truth for relation records."""

    def __init__(self, config: NesyConfig) -> None:
        super().__init__(config)
        json_path = config.storage.json_path or "~/.nesy-reasoning/relations.json"
        self.path = Path(json_path).expanduser()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            self._load_from_disk()

    def assert_relations(
        self,
        inputs: Iterable[RelationInput],
        *,
        mode: str = "append",
        dry_run: bool = False,
    ) -> tuple[list[RelationRecord], int]:
        """Add relation records and persist the JSON relation set."""
        records, updated = super().assert_relations(inputs, mode=mode, dry_run=dry_run)
        if not dry_run:
            self._persist()
        return records, updated

    def assert_exclusive(
        self,
        inputs: Iterable[ExclusiveGroupInput],
    ) -> tuple[list[ExclusiveGroupRecord], int]:
        """Add or replace exclusive groups and persist the JSON relation set."""
        records, updated = super().assert_exclusive(inputs)
        self._persist()
        return records, updated

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
        """Remove records and persist the JSON relation set."""
        removed, removed_groups = super().clear_relations(
            scope=scope,
            store_id=store_id,
            context_id=context_id,
            relation_filter=relation_filter,
            dry_run=dry_run,
            include_exclusive_groups=include_exclusive_groups,
        )
        if not dry_run:
            self._persist()
        return removed, removed_groups

    def record_audit(
        self,
        *,
        event_type: str,
        tool_name: str,
        arguments: dict[str, Any],
        result_status: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Record one audit event and persist the JSON relation set."""
        super().record_audit(
            event_type=event_type,
            tool_name=tool_name,
            arguments=arguments,
            result_status=result_status,
            metadata=metadata,
        )
        self._persist()

    def import_records(
        self,
        relations: Iterable[RelationRecord],
        exclusive_groups: Iterable[ExclusiveGroupRecord],
        independence_records: Iterable[IndependenceRecord] = (),
        *,
        mode: str,
        store_id: str,
        context_metadata: dict[str, Any] | None = None,
        dry_run: bool = False,
    ) -> tuple[int, int, int, int]:
        """Import validated records and persist the JSON relation set."""
        result = super().import_records(
            relations,
            exclusive_groups,
            independence_records,
            mode=mode,
            store_id=store_id,
            context_metadata=context_metadata,
            dry_run=dry_run,
        )
        if not dry_run:
            self._persist()
        return result

    def _load_from_disk(self) -> None:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON relation store: {self.path}") from exc

        self._relations = [
            RelationRecord.model_validate(item) for item in data.get("relations", [])
        ]
        self._exclusive_groups = [
            ExclusiveGroupRecord.model_validate(item) for item in data.get("exclusive_groups", [])
        ]
        self._independence_records = [
            IndependenceRecord.model_validate(item) for item in data.get("independence_records", [])
        ]
        self._audit_log = [_audit_from_dict(item) for item in data.get("audit_log", [])]
        self._context_metadata = data.get("context_metadata", {})

    def _persist(self) -> None:
        data = {
            "version": "2.0",
            "relations": [record.model_dump(mode="json") for record in self._relations],
            "exclusive_groups": [group.model_dump(mode="json") for group in self._exclusive_groups],
            "independence_records": [
                record.model_dump(mode="json") for record in self._independence_records
            ],
            "audit_log": [entry.to_dict() for entry in self._audit_log],
            "context_metadata": self._context_metadata,
        }
        tmp_path = self.path.with_name(f".{self.path.name}.tmp")
        tmp_path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        os.replace(tmp_path, self.path)


def create_relation_store(config: NesyConfig | None = None) -> RelationStoreProtocol:
    """Create a relation store for the configured backend."""
    resolved = config or load_config()
    if resolved.storage.backend == StorageBackend.MEMORY:
        return MemoryRelationStore(resolved)
    if resolved.storage.backend == StorageBackend.JSON:
        return JsonRelationStore(resolved)
    if resolved.storage.backend == StorageBackend.SQLITE:
        return SqliteRelationStore(resolved)
    raise ValueError(f"storage backend is not implemented yet: {resolved.storage.backend}")


def graph_stats_for(
    relations: Iterable[RelationRecord],
    edges: Iterable[CanonicalImplicationEdge],
    *,
    exclusive_group_count: int = 0,
) -> GraphStats:
    """Build graph statistics for a relation and edge view."""
    relation_list = list(relations)
    edge_list = list(edges)
    propositions = {item.source for item in relation_list} | {item.target for item in relation_list}
    contexts = {item.context_id for item in relation_list}
    stores = {item.store_id for item in relation_list}
    return GraphStats(
        relations=len(relation_list),
        propositions=len(propositions),
        implication_edges=len(edge_list),
        exclusive_groups=exclusive_group_count,
        contexts=len(contexts),
        stores=len(stores),
    )


def _matches_filter(relation: RelationRecord, relation_filter: RelationFilter) -> bool:
    if relation_filter.source is not None and relation.source != relation_filter.source:
        return False
    if relation_filter.target is not None and relation.target != relation_filter.target:
        return False
    if (
        relation_filter.relation_type is not None
        and relation.relation_type != relation_filter.relation_type
    ):
        return False
    if relation_filter.context_id is not None and relation.context_id != relation_filter.context_id:
        return False
    if relation_filter.store_id is not None and relation.store_id != relation_filter.store_id:
        return False
    return not (
        relation_filter.domain is not None
        and relation.metadata.get("domain") != relation_filter.domain
    )


def _group_matches_scope(
    group: ExclusiveGroupRecord,
    scope: str,
    store_id: str,
    context_id: str,
    relation_filter: RelationFilter,
) -> bool:
    if scope == "store":
        return group.store_id == store_id
    if scope == "context":
        return group.store_id == store_id and group.context_id == context_id
    if scope == "filter":
        if relation_filter.store_id is not None and group.store_id != relation_filter.store_id:
            return False
        if (
            relation_filter.context_id is not None
            and group.context_id != relation_filter.context_id
        ):
            return False
        return not (
            relation_filter.domain is not None
            and group.metadata.get("domain") != relation_filter.domain
        )
    return False


def _independence_matches_scope(
    record: IndependenceRecord,
    scope: str,
    store_id: str,
    context_id: str,
    relation_filter: RelationFilter,
) -> bool:
    if scope == "store":
        return record.store_id == store_id
    if scope == "context":
        return record.store_id == store_id and record.context_id == context_id
    if scope == "filter":
        pair = {record.left, record.right}
        if relation_filter.store_id is not None and record.store_id != relation_filter.store_id:
            return False
        if (
            relation_filter.context_id is not None
            and record.context_id != relation_filter.context_id
        ):
            return False
        if relation_filter.source is not None and relation_filter.source not in pair:
            return False
        if relation_filter.target is not None and relation_filter.target not in pair:
            return False
        return not (
            relation_filter.domain is not None
            and record.metadata.get("domain") != relation_filter.domain
        )
    return False


def _edge(
    relation: RelationRecord,
    antecedent: str,
    consequent: str,
    suffix: str,
) -> CanonicalImplicationEdge:
    return CanonicalImplicationEdge(
        edge_id=f"edge_{relation.id}_{suffix}",
        relation_id=relation.id,
        antecedent=antecedent,
        consequent=consequent,
        source_relation_type=relation.relation_type,
        confidence=relation.confidence,
        context_id=relation.context_id,
        store_id=relation.store_id,
        assumptions=list(relation.assumptions),
        temporal=relation.temporal,
    )


def _input_hash(arguments: dict[str, Any]) -> str:
    payload = json.dumps(arguments, sort_keys=True, ensure_ascii=False, default=str)
    return sha256(payload.encode("utf-8")).hexdigest()


def _audit_from_dict(data: dict[str, Any]) -> AuditEntry:
    return AuditEntry(
        entry_id=data["id"],
        event_type=data["event_type"],
        tool_name=data["tool_name"],
        input_hash=data["input_hash"],
        result_status=data["result_status"],
        created_at=data["created_at"],
        metadata=data.get("metadata", {}),
    )


def _relation_for_store(record: RelationRecord, store_id: str) -> RelationRecord:
    return record.model_copy(update={"store_id": store_id})


def _group_for_store(group: ExclusiveGroupRecord, store_id: str) -> ExclusiveGroupRecord:
    return group.model_copy(update={"store_id": store_id})


def _independence_for_store(
    record: IndependenceRecord,
    store_id: str,
) -> IndependenceRecord:
    return record.model_copy(update={"store_id": store_id})


def _merge_import(
    current_relations: Iterable[RelationRecord],
    current_groups: Iterable[ExclusiveGroupRecord],
    current_independence: Iterable[IndependenceRecord],
    incoming_relations: list[RelationRecord],
    incoming_groups: list[ExclusiveGroupRecord],
    incoming_independence: list[IndependenceRecord],
    *,
    mode: str,
    store_id: str,
) -> tuple[
    list[RelationRecord],
    list[ExclusiveGroupRecord],
    list[IndependenceRecord],
    int,
    int,
]:
    if mode == "append":
        relations = [*current_relations, *incoming_relations]
        updated_relations = 0
    elif mode == "upsert":
        relations, updated_relations = _upsert_relations(current_relations, incoming_relations)
    elif mode == "replace_store":
        relations = [
            relation for relation in current_relations if relation.store_id != store_id
        ] + incoming_relations
        updated_relations = 0
    else:
        raise ValueError(f"unsupported import mode: {mode}")

    groups, updated_groups = _merge_groups(
        current_groups,
        incoming_groups,
        mode=mode,
        store_id=store_id,
    )
    independence = _merge_independence_records(
        current_independence,
        incoming_independence,
        mode=mode,
        store_id=store_id,
    )
    return relations, groups, independence, updated_relations, updated_groups


def _upsert_relations(
    current_relations: Iterable[RelationRecord],
    incoming_relations: Iterable[RelationRecord],
) -> tuple[list[RelationRecord], int]:
    relations = list(current_relations)
    positions = {relation.id: index for index, relation in enumerate(relations)}
    updated = 0
    for relation in incoming_relations:
        if relation.id in positions:
            relations[positions[relation.id]] = relation
            updated += 1
        else:
            positions[relation.id] = len(relations)
            relations.append(relation)
    return relations, updated


def _merge_groups(
    current_groups: Iterable[ExclusiveGroupRecord],
    incoming_groups: Iterable[ExclusiveGroupRecord],
    *,
    mode: str,
    store_id: str,
) -> tuple[list[ExclusiveGroupRecord], int]:
    if mode == "replace_store":
        groups = [group for group in current_groups if group.store_id != store_id]
    else:
        groups = list(current_groups)

    positions = {
        (group.group_id, group.context_id, group.store_id): index
        for index, group in enumerate(groups)
    }
    updated = 0
    for group in incoming_groups:
        key = (group.group_id, group.context_id, group.store_id)
        if key in positions:
            groups[positions[key]] = group
            updated += 1
        else:
            positions[key] = len(groups)
            groups.append(group)
    return groups, updated


def _merge_independence_records(
    current_records: Iterable[IndependenceRecord],
    incoming_records: Iterable[IndependenceRecord],
    *,
    mode: str,
    store_id: str,
) -> list[IndependenceRecord]:
    if mode == "replace_store":
        records = [record for record in current_records if record.store_id != store_id]
    else:
        records = list(current_records)
    positions = {record.id: index for index, record in enumerate(records)}
    pair_positions = {
        _independence_key(record): index for index, record in enumerate(records) if mode != "append"
    }
    for record in incoming_records:
        if mode == "upsert" and record.id in positions:
            index = positions[record.id]
            pair_positions.pop(_independence_key(records[index]), None)
            records[index] = record
            pair_positions[_independence_key(record)] = index
            continue
        key = _independence_key(record)
        if mode == "upsert" and key in pair_positions:
            index = pair_positions[key]
            positions.pop(records[index].id, None)
            records[index] = record
            positions[record.id] = index
            continue
        positions[record.id] = len(records)
        pair_positions[key] = len(records)
        records.append(record)
    return records


def _independence_key(record: IndependenceRecord) -> tuple[str, str, str, str]:
    left, right = sorted((record.left, record.right))
    return left, right, record.context_id, record.store_id


def _merge_context_metadata(
    current_metadata: dict[str, Any],
    incoming_metadata: dict[str, Any],
) -> dict[str, Any]:
    merged = deepcopy(current_metadata)
    for context_id, metadata in incoming_metadata.items():
        merged[context_id] = deepcopy(metadata)
    return merged


def _dumps(value: Any) -> str | None:
    if value is None:
        return None
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _loads(value: str | None, default: Any) -> Any:
    if value is None:
        return default
    return json.loads(value)


def _independence_from_row(row: sqlite3.Row) -> IndependenceRecord:
    return IndependenceRecord(
        id=row["id"],
        left=row["left_value"],
        right=row["right_value"],
        relation=row["relation"],
        confidence=row["confidence"],
        context_id=row["context_id"],
        store_id=row["store_id"],
        metadata=_loads(row["metadata_json"], {}),
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


def _relation_from_row(row: sqlite3.Row) -> RelationRecord:
    return RelationRecord(
        id=row["id"],
        source=row["source"],
        target=row["target"],
        relation_type=row["relation_type"],
        polarity=row["polarity"],
        confidence=row["confidence"],
        context_id=row["context_id"],
        store_id=row["store_id"],
        temporal=_loads(row["temporal_json"], None),
        assumptions=_loads(row["assumptions_json"], []),
        provenance=_loads(row["provenance_json"], None),
        metadata=_loads(row["metadata_json"], {}),
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )
