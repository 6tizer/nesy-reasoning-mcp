"""SQLite relation store backend."""

from __future__ import annotations

import sqlite3
from collections.abc import Callable, Iterable
from contextlib import suppress
from datetime import UTC, datetime
from functools import wraps
from pathlib import Path
from threading import RLock
from typing import Any, Concatenate, ParamSpec, TypeVar, cast

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

            CREATE INDEX IF NOT EXISTS idx_relations_context_store_type
              ON relations (context_id, store_id, relation_type);
            CREATE INDEX IF NOT EXISTS idx_relations_source_target
              ON relations (source, target);
            CREATE INDEX IF NOT EXISTS idx_relations_created_id
              ON relations (created_at, id);
            CREATE INDEX IF NOT EXISTS idx_relations_domain
              ON relations (json_extract(metadata_json, '$.domain'));
            """
        )
        self._ensure_relation_identity_columns()
        self._conn.commit()

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
