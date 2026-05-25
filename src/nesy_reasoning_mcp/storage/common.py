"""Shared store helper functions."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterable
from copy import deepcopy
from typing import Any, TypeVar

from pydantic import BaseModel

from nesy_reasoning_mcp.schemas import (
    CanonicalImplicationEdge,
    ExclusiveGroupRecord,
    GraphStats,
    IndependenceRecord,
    PropositionRecord,
    RelationFilter,
    RelationInput,
    RelationRecord,
    RelationType,
)

RelationT = TypeVar("RelationT", bound=RelationInput)


def graph_stats_for(
    relations: Iterable[RelationRecord],
    edges: Iterable[CanonicalImplicationEdge],
    *,
    exclusive_group_count: int = 0,
) -> GraphStats:
    """Build graph statistics for a relation and edge view."""
    relation_list = list(relations)
    edge_list = list(edges)
    propositions = {item.canonical_source for item in relation_list} | {
        item.canonical_target for item in relation_list
    }
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


def normalize_relation_edges(relation: RelationRecord) -> list[CanonicalImplicationEdge]:
    """Derive canonical implication edges for one stored relation."""
    if relation.relation_type == RelationType.SUFFICIENT:
        return [_edge(relation, relation.canonical_source, relation.canonical_target, "a")]
    if relation.relation_type == RelationType.NECESSARY:
        return [_edge(relation, relation.canonical_target, relation.canonical_source, "a")]
    if relation.relation_type == RelationType.EQUIVALENT:
        return [
            _edge(relation, relation.canonical_source, relation.canonical_target, "a"),
            _edge(relation, relation.canonical_target, relation.canonical_source, "b"),
        ]
    return []


def _relation_for_store(record: RelationRecord, store_id: str) -> RelationRecord:
    return record.model_copy(update={"store_id": store_id})


def _group_for_store(group: ExclusiveGroupRecord, store_id: str) -> ExclusiveGroupRecord:
    return group.model_copy(update={"store_id": store_id})


def _independence_for_store(
    record: IndependenceRecord,
    store_id: str,
) -> IndependenceRecord:
    return record.model_copy(update={"store_id": store_id})


def _proposition_alias_index(propositions: Iterable[PropositionRecord]) -> dict[str, str]:
    index: dict[str, str] = {}
    for proposition in propositions:
        for term in (proposition.id, proposition.label, *proposition.aliases):
            key = term.strip()
            existing = index.get(key)
            if existing is not None and existing != proposition.id:
                raise ValueError(
                    f"proposition alias conflict: {key!r} maps to both "
                    f"{existing!r} and {proposition.id!r}"
                )
            index[key] = proposition.id
    return index


def _normalize_relation_identities(
    relations: Iterable[RelationT],
    propositions: Iterable[PropositionRecord],
) -> list[RelationT]:
    """Fill missing relation proposition ids from exact id/label/alias matches."""
    index = _proposition_alias_index(propositions)
    normalized: list[RelationT] = []
    for relation in relations:
        updates: dict[str, str] = {}
        if relation.source_id is None and relation.source in index:
            updates["source_id"] = index[relation.source]
        if relation.target_id is None and relation.target in index:
            updates["target_id"] = index[relation.target]
        normalized.append(relation.model_copy(update=updates) if updates else relation)
    return normalized


def _merge_propositions(
    current_propositions: Iterable[PropositionRecord],
    incoming_propositions: Iterable[PropositionRecord],
) -> tuple[list[PropositionRecord], int]:
    propositions = list(current_propositions)
    positions = {proposition.id: index for index, proposition in enumerate(propositions)}
    updated = 0
    for proposition in incoming_propositions:
        if proposition.id in positions:
            propositions[positions[proposition.id]] = proposition
            updated += 1
        else:
            positions[proposition.id] = len(propositions)
            propositions.append(proposition)
    _proposition_alias_index(propositions)
    return propositions, updated


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
    if isinstance(value, BaseModel):
        value = value.model_dump(mode="json", exclude_none=True)
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
        source_id=row["source_id"],
        target=row["target"],
        target_id=row["target_id"],
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
