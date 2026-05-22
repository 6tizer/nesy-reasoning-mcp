"""SQLite relation store backend."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from nesy_reasoning_mcp.config import NesyConfig
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
from nesy_reasoning_mcp.storage.audit import AuditEntry, _input_hash
from nesy_reasoning_mcp.storage.common import (
    _dumps,
    _edge,
    _group_for_store,
    _group_matches_scope,
    _independence_for_store,
    _independence_from_row,
    _independence_matches_scope,
    _loads,
    _matches_filter,
    _merge_context_metadata,
    _merge_import,
    _relation_for_store,
    _relation_from_row,
    _upsert_relations,
    graph_stats_for,
)


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

        if mode == "upsert":
            merged, updated = _upsert_relations(self.list_relations(), records)
            if not dry_run:
                try:
                    self._replace_all_records(
                        merged,
                        self.list_exclusive_groups(),
                        self.list_independence_records(),
                        self.context_metadata(),
                    )
                    self._conn.commit()
                except Exception:
                    self._conn.rollback()
                    raise
        elif mode == "replace_same_pair":
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

        elif mode != "append":
            raise ValueError(f"unsupported assert mode: {mode}")

        if not dry_run and mode != "upsert":
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
