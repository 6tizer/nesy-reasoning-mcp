"""Relation, exclusive, and contradiction tool handlers."""

from __future__ import annotations

from typing import Any

from nesy_reasoning_mcp.reasoning import (
    find_exclusive_contradictions,
    relations_compatible_with_filter,
)
from nesy_reasoning_mcp.schemas import (
    AssertExclusiveInput,
    AssertRelationsInput,
    CheckContradictionsInput,
    ClearRelationsInput,
    ContextFilter,
    ContradictionMode,
    Diagnostic,
    ListRelationsInput,
    OnContradiction,
    RelationRecord,
    RelationType,
)
from nesy_reasoning_mcp.storage.common import (
    _apply_assert_relations_mode,
    _merge_propositions,
    _normalize_relation_identities,
)
from nesy_reasoning_mcp.store import RelationStoreProtocol, graph_stats_for
from nesy_reasoning_mcp.tool_common import (
    _contradiction_trace,
    _exclusive_group_dump,
    _normalization_trace,
    _record_dump,
    _temporary_fact_records,
)


async def assert_relations(
    arguments: dict[str, Any],
    store: RelationStoreProtocol,
) -> dict[str, Any]:
    """Handle `nesy.assert_relations`."""
    payload = AssertRelationsInput.model_validate(arguments)
    stored_propositions = store.list_propositions()
    if payload.on_contradiction == OnContradiction.REJECT:
        records, _updated = store.assert_relations(
            payload.relations,
            mode=payload.mode,
            dry_run=True,
        )
        current_relations = store.list_relations()
        effective_relations = _normalize_relation_identities(
            _relations_after_assert(current_relations, records, payload.mode),
            stored_propositions,
        )
        rejection_contradictions, _context_separated = find_exclusive_contradictions(
            effective_relations,
            store.list_exclusive_groups(),
            context_filter=ContextFilter(),
            max_depth=8,
            propositions=stored_propositions,
        )
        if rejection_contradictions:
            diagnostic = Diagnostic(
                level="error",
                code="CONTRADICTION_REJECTED",
                message="Relation assertion rejected because it would create a hard contradiction.",
                related_ids=[
                    fact_id
                    for contradiction in rejection_contradictions
                    for fact_id in contradiction.get("fact_ids", [])
                    if isinstance(fact_id, str)
                ],
            )
            return {
                "status": "error",
                "added": 0,
                "updated": 0,
                "rejected": len(payload.relations),
                "relation_ids": [],
                "contradictions": rejection_contradictions,
                "diagnostics": [diagnostic.model_dump(mode="json")],
                "trace": ["Rejected relation assertion before writing."],
                "graph_stats": store.graph_stats().model_dump(mode="json"),
            }

    records, updated = store.assert_relations(
        payload.relations,
        mode=payload.mode,
        dry_run=payload.dry_run,
    )
    contradictions: list[dict[str, Any]] = []
    current_relations = store.list_relations()
    effective_relations = (
        _relations_after_assert(current_relations, records, payload.mode)
        if payload.dry_run
        else current_relations
    )
    effective_relations = _normalize_relation_identities(effective_relations, stored_propositions)
    diagnostics: list[Diagnostic] = []
    merge_trace: list[str] = []
    if payload.merge_equivalent:
        merge_diagnostics, merge_trace = _equivalent_normalization(effective_relations)
        diagnostics.extend(merge_diagnostics)
    if payload.check_contradictions:
        contradictions, _context_separated = find_exclusive_contradictions(
            effective_relations,
            store.list_exclusive_groups(),
            context_filter=ContextFilter(),
            max_depth=8,
            propositions=stored_propositions,
        )

    trace = [*[_normalization_trace(record) for record in records], *merge_trace]
    added = len(records) - updated if payload.mode == "upsert" else len(records)
    return {
        "status": "warning" if contradictions else "ok",
        "added": added,
        "updated": updated if payload.mode in {"replace_same_pair", "upsert"} else 0,
        "rejected": 0,
        "relation_ids": [record.id for record in records],
        "contradictions": contradictions,
        "diagnostics": [item.model_dump(mode="json") for item in diagnostics],
        "trace": trace,
        "graph_stats": store.graph_stats().model_dump(mode="json"),
    }


def _relations_after_assert(
    current: list[RelationRecord],
    records: list[RelationRecord],
    mode: str,
) -> list[RelationRecord]:
    relations, _updated = _apply_assert_relations_mode(current, records, mode)
    return relations


def _equivalent_normalization(
    relations: list[RelationRecord],
) -> tuple[list[Diagnostic], list[str]]:
    grouped: dict[tuple[str, str, str, str], dict[RelationType, list[str]]] = {}
    for relation in relations:
        if relation.relation_type not in {RelationType.SUFFICIENT, RelationType.NECESSARY}:
            continue
        key = (
            relation.canonical_source,
            relation.canonical_target,
            relation.context_id,
            relation.store_id,
        )
        grouped.setdefault(key, {}).setdefault(relation.relation_type, []).append(relation.id)

    diagnostics: list[Diagnostic] = []
    trace: list[str] = []
    for source, target, context_id, store_id in sorted(grouped):
        relation_ids_by_type = grouped[(source, target, context_id, store_id)]
        if not {
            RelationType.SUFFICIENT,
            RelationType.NECESSARY,
        }.issubset(relation_ids_by_type):
            continue
        related_ids = sorted(
            [
                *relation_ids_by_type[RelationType.SUFFICIENT],
                *relation_ids_by_type[RelationType.NECESSARY],
            ]
        )
        diagnostics.append(
            Diagnostic(
                level="info",
                code="MERGE_EQUIVALENT_NORMALIZED",
                message=(
                    "sufficient and necessary evidence for the same pair is reported as "
                    "equivalent in the canonical graph; original evidence records are preserved."
                ),
                related_ids=related_ids,
            )
        )
        trace.append(
            "reported sufficient+necessary evidence as canonical equivalent for "
            f"({source}, {target}) in context={context_id}, store={store_id}; "
            "preserved stored records."
        )
    return diagnostics, trace


async def list_relations(arguments: dict[str, Any], store: RelationStoreProtocol) -> dict[str, Any]:
    """Handle `nesy.list_relations`."""
    payload = ListRelationsInput.model_validate(arguments)
    offset = int(payload.cursor or 0)
    listed = store.list_relations(payload.filter, limit=payload.limit + 1, offset=offset)
    relations = listed[: payload.limit]
    next_cursor = str(offset + len(relations)) if len(listed) > payload.limit else None
    stats_edges = store.implication_edges(relations)
    edges = stats_edges if payload.include_implication_edges else []
    exclusive_groups = store.list_exclusive_groups() if payload.include_exclusive_groups else []
    return {
        "status": "ok",
        "relations": [_record_dump(record) for record in relations],
        "implication_edges": [edge.model_dump(mode="json") for edge in edges],
        "exclusive_groups": [_exclusive_group_dump(group) for group in exclusive_groups],
        "total": len(relations),
        "next_cursor": next_cursor,
        "diagnostics": [],
        "trace": [f"Listed {len(relations)} relation(s)."],
        "graph_stats": graph_stats_for(
            relations,
            stats_edges,
            exclusive_group_count=len(store.list_exclusive_groups()),
        ).model_dump(mode="json"),
    }


async def clear_relations(
    arguments: dict[str, Any],
    store: RelationStoreProtocol,
) -> dict[str, Any]:
    """Handle `nesy.clear_relations`."""
    payload = ClearRelationsInput.model_validate(arguments)
    if (
        payload.scope == "all"
        and not payload.dry_run
        and not store.config.security.allow_scope_all_clear
    ):
        diagnostic = Diagnostic(
            level="error",
            code="SCOPE_ALL_CLEAR_DISABLED",
            message="This server refuses scope=all clear unless dry_run=true.",
        )
        return {
            "status": "error",
            "removed_relations": 0,
            "removed_exclusive_groups": 0,
            "dry_run": payload.dry_run,
            "diagnostics": [diagnostic.model_dump(mode="json")],
            "trace": ["Rejected scope=all clear."],
            "graph_stats": store.graph_stats().model_dump(mode="json"),
        }

    removed, removed_groups = store.clear_relations(
        scope=payload.scope,
        store_id=payload.store_id,
        context_id=payload.context_id,
        relation_filter=payload.filter,
        dry_run=payload.dry_run,
        include_exclusive_groups=payload.include_exclusive_groups,
    )
    return {
        "status": "ok",
        "removed_relations": removed,
        "removed_exclusive_groups": removed_groups,
        "dry_run": payload.dry_run,
        "diagnostics": [],
        "trace": [f"Matched {removed} relation(s) for clearing."],
        "graph_stats": store.graph_stats().model_dump(mode="json"),
    }


async def assert_exclusive(
    arguments: dict[str, Any],
    store: RelationStoreProtocol,
) -> dict[str, Any]:
    """Handle `nesy.assert_exclusive`."""
    payload = AssertExclusiveInput.model_validate(arguments)
    records, updated = store.assert_exclusive(payload.groups)
    return {
        "status": "ok",
        "added_groups": len(records) - updated,
        "updated_groups": updated,
        "group_ids": [record.group_id for record in records],
        "diagnostics": [],
        "trace": [
            f"Registered exclusive group {record.group_id} with {len(record.members)} members."
            for record in records
        ],
        "graph_stats": store.graph_stats().model_dump(mode="json"),
    }


async def check_contradictions(
    arguments: dict[str, Any],
    store: RelationStoreProtocol,
) -> dict[str, Any]:
    """Handle `nesy.check_contradictions`."""
    payload = CheckContradictionsInput.model_validate(arguments)
    try:
        effective_propositions, _updated_propositions = _merge_propositions(
            store.list_propositions(),
            payload.propositions,
        )
    except ValueError as exc:
        diagnostic = Diagnostic(
            level="error", code="PROPOSITION_REGISTRY_INVALID", message=str(exc)
        )
        return {
            "status": "error",
            "has_contradictions": False,
            "contradictions": [],
            "clean_facts_count": 0,
            "total_facts_count": len(payload.facts),
            "context_separated": [],
            "diagnostics": [diagnostic.model_dump(mode="json")],
            "trace": ["Rejected contradiction check."],
            "graph_stats": store.graph_stats().model_dump(mode="json"),
        }
    fact_records = _temporary_fact_records(payload)
    if payload.mode == ContradictionMode.GRAPH:
        relations = store.list_relations()
    elif payload.mode == ContradictionMode.FACTS:
        relations = fact_records
    else:
        relations = [*store.list_relations(), *fact_records]
    relations = _normalize_relation_identities(relations, effective_propositions)

    groups = store.list_exclusive_groups()
    contradictions, context_separated = find_exclusive_contradictions(
        relations,
        groups,
        payload.context_filter,
        max_depth=payload.max_depth,
        include_soft=payload.include_soft,
        min_confidence=payload.min_confidence,
        propositions=effective_propositions,
    )
    compatible_relations = relations_compatible_with_filter(relations, payload.context_filter)
    graph_stats = graph_stats_for(
        compatible_relations,
        store.implication_edges(compatible_relations),
        exclusive_group_count=len(groups),
    ).model_dump(mode="json")
    total_facts = len(payload.facts)
    conflicting_fact_ids = {
        fact_id
        for contradiction in contradictions
        for fact_id in contradiction.get("fact_ids", [])
        if isinstance(fact_id, str) and fact_id.startswith("input_")
    }
    clean_facts = total_facts - len(conflicting_fact_ids)
    return {
        "status": "warning" if contradictions else "ok",
        "has_contradictions": bool(contradictions),
        "contradictions": contradictions,
        "clean_facts_count": clean_facts,
        "total_facts_count": total_facts,
        "context_separated": context_separated,
        "diagnostics": [],
        "trace": _contradiction_trace(payload.mode, len(fact_records), contradictions),
        "graph_stats": graph_stats,
    }
