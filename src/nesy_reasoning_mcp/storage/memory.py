"""In-memory relation store backend."""

from __future__ import annotations

from collections.abc import Iterable
from copy import deepcopy
from typing import Any

from nesy_reasoning_mcp.config import NesyConfig, load_config
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
    RelationType,
)
from nesy_reasoning_mcp.storage.audit import AuditEntry, _input_hash
from nesy_reasoning_mcp.storage.common import (
    _edge,
    _group_for_store,
    _group_matches_scope,
    _independence_for_store,
    _independence_matches_scope,
    _matches_filter,
    _merge_context_metadata,
    _merge_import,
    _merge_propositions,
    _normalize_relation_identities,
    _relation_for_store,
    _upsert_relations,
    graph_stats_for,
)


class MemoryRelationStore:
    """Process-local in-memory source of truth for relation records."""

    def __init__(self, config: NesyConfig | None = None) -> None:
        self.config = config or load_config()
        self._relations: list[RelationRecord] = []
        self._exclusive_groups: list[ExclusiveGroupRecord] = []
        self._independence_records: list[IndependenceRecord] = []
        self._propositions: list[PropositionRecord] = []
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
        normalized_inputs = _normalize_relation_identities(inputs, self._propositions)
        records = [RelationRecord.from_input(item) for item in normalized_inputs]
        updated = 0

        if mode == "upsert":
            merged, updated = _upsert_relations(self._relations, records)
            if not dry_run:
                self._relations = merged
        elif mode == "replace_same_pair":
            replace_keys = {
                (
                    record.canonical_source,
                    record.canonical_target,
                    record.context_id,
                    record.store_id,
                )
                for record in records
            }
            updated = sum(
                1
                for relation in self._relations
                if (
                    relation.canonical_source,
                    relation.canonical_target,
                    relation.context_id,
                    relation.store_id,
                )
                in replace_keys
            )
            if not dry_run:
                self._relations = [
                    relation
                    for relation in self._relations
                    if (
                        relation.canonical_source,
                        relation.canonical_target,
                        relation.context_id,
                        relation.store_id,
                    )
                    not in replace_keys
                ]

        elif mode != "append":
            raise ValueError(f"unsupported assert mode: {mode}")

        if not dry_run and mode != "upsert":
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
        offset: int = 0,
    ) -> list[RelationRecord]:
        """List relation records matching an optional filter."""
        matched = [
            relation
            for relation in self._relations
            if relation_filter is None or _matches_filter(relation, relation_filter)
        ]
        matched = matched[offset:]
        if limit is not None:
            return matched[:limit]
        return matched

    def list_exclusive_groups(self) -> list[ExclusiveGroupRecord]:
        """List all stored exclusive groups."""
        return list(self._exclusive_groups)

    def list_independence_records(self) -> list[IndependenceRecord]:
        """List all stored independence records."""
        return list(self._independence_records)

    def list_propositions(self) -> list[PropositionRecord]:
        """List all stored proposition records."""
        return [proposition.model_copy(deep=True) for proposition in self._propositions]

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
                self._propositions.clear()
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
                edges.append(
                    _edge(relation, relation.canonical_source, relation.canonical_target, "a")
                )
            elif relation.relation_type == RelationType.NECESSARY:
                edges.append(
                    _edge(relation, relation.canonical_target, relation.canonical_source, "a")
                )
            elif relation.relation_type == RelationType.EQUIVALENT:
                edges.append(
                    _edge(relation, relation.canonical_source, relation.canonical_target, "a")
                )
                edges.append(
                    _edge(relation, relation.canonical_target, relation.canonical_source, "b")
                )
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
        propositions: Iterable[PropositionRecord] = (),
        *,
        mode: str,
        store_id: str,
        context_metadata: dict[str, Any] | None = None,
        dry_run: bool = False,
    ) -> tuple[int, int, int, int]:
        """Import validated records into memory."""
        merged_propositions, _updated_propositions = _merge_propositions(
            self._propositions,
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
            self._propositions = merged_propositions
            self._context_metadata = merged_metadata
        return len(incoming_relations), len(incoming_groups), updated_relations, updated_groups
