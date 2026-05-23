"""Independence helpers shared by reasoning tools."""

from __future__ import annotations

from typing import Any

from nesy_reasoning_mcp.schemas import (
    ContextFilter,
    ExclusiveGroupRecord,
    IndependenceRecord,
    RelationRecord,
)
from nesy_reasoning_mcp.tool_common import _exclusive_group_compatible_with_context_filter


def _path_independence_from_if_not(
    path: Any,
    relation_map: dict[str, RelationRecord],
    independence_records: list[IndependenceRecord],
    exclusive_groups: list[ExclusiveGroupRecord],
    if_not: str,
    context_filter: ContextFilter,
) -> str:
    if if_not in path.nodes or not path.edges:
        return "unknown"
    source = str(path.nodes[0])
    if _pair_proves_independence(
        source,
        if_not,
        independence_records,
        exclusive_groups,
        context_filter,
    ):
        return "proven"
    first_edge = path.edges[0]
    relation = relation_map.get(first_edge.relation_id)
    if relation is None:
        return "unknown"
    if relation.canonical_source == first_edge.antecedent and _metadata_independent_of(
        relation.metadata,
        if_not,
    ):
        return "proven"
    if relation.canonical_source == first_edge.antecedent and _assumptions_independent_of(
        relation.assumptions,
        relation.canonical_source,
        if_not,
    ):
        return "proven"
    return "unknown"


def _pair_proves_independence(
    left: str,
    right: str,
    independence_records: list[IndependenceRecord],
    exclusive_groups: list[ExclusiveGroupRecord],
    context_filter: ContextFilter,
) -> bool:
    return _formal_independence(left, right, independence_records, context_filter) or (
        _exclusive_pair(left, right, exclusive_groups, context_filter)
    )


def _formal_independence(
    left: str,
    right: str,
    records: list[IndependenceRecord],
    context_filter: ContextFilter,
) -> bool:
    pair = {left, right}
    return any(
        {record.left, record.right} == pair
        and _independence_compatible_with_context_filter(record, context_filter)
        for record in records
    )


def _independence_compatible_with_context_filter(
    record: IndependenceRecord,
    context_filter: ContextFilter,
) -> bool:
    if context_filter.store_id and record.store_id != context_filter.store_id:
        return False
    if context_filter.context_id and record.context_id != context_filter.context_id:
        return False
    return not (context_filter.domain and record.metadata.get("domain") != context_filter.domain)


def _exclusive_pair(
    left: str,
    right: str,
    groups: list[ExclusiveGroupRecord],
    context_filter: ContextFilter,
) -> bool:
    return any(
        left in group.members
        and right in group.members
        and _exclusive_group_compatible_with_context_filter(group, context_filter)
        for group in groups
    )


def _metadata_independent_of(metadata: dict[str, Any], if_not: str) -> bool:
    return if_not in _metadata_string_values(metadata.get("independent_of"))


def _metadata_string_values(value: Any) -> set[str]:
    if value is None:
        return set()
    if isinstance(value, str):
        return {value}
    if isinstance(value, list | tuple | set):
        values: set[str] = set()
        for item in value:
            values.update(_metadata_string_values(item))
        return values
    if isinstance(value, dict):
        values = {
            str(key) for key, item in value.items() if isinstance(item, bool) and item is True
        }
        for key in ("right", "target", "proposition", "propositions", "values"):
            values.update(_metadata_string_values(value.get(key)))
        return values
    return set()


def _assumptions_independent_of(assumptions: list[str], source: str, if_not: str) -> bool:
    markers = {
        f"independent_of:{if_not}",
        f"independent_of={if_not}",
        f"{source} independent_of {if_not}",
        f"{if_not} independent_of {source}",
    }
    return bool(markers & {assumption.strip() for assumption in assumptions})
