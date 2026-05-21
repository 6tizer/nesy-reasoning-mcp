"""In-memory relation and exclusive group store."""

from __future__ import annotations

from collections.abc import Iterable

from nesy_reasoning_mcp.schemas import (
    CanonicalImplicationEdge,
    ExclusiveGroupInput,
    ExclusiveGroupRecord,
    GraphStats,
    RelationFilter,
    RelationInput,
    RelationRecord,
    RelationType,
)


class RelationStore:
    """Process-local in-memory source of truth for relation records."""

    def __init__(self) -> None:
        self._relations: list[RelationRecord] = []
        self._exclusive_groups: list[ExclusiveGroupRecord] = []

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
