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
)
from nesy_reasoning_mcp.store import RelationStore, graph_stats_for
from nesy_reasoning_mcp.tool_common import (
    _contradiction_trace,
    _exclusive_group_dump,
    _record_dump,
    _temporary_fact_records,
)
from nesy_reasoning_mcp.tool_reasoning import _normalization_trace


async def assert_relations(arguments: dict[str, Any], store: RelationStore) -> dict[str, Any]:
    """Handle `nesy.assert_relations`."""
    payload = AssertRelationsInput.model_validate(arguments)
    if payload.mode == "upsert":
        diagnostic = Diagnostic(
            level="error",
            code="UPSERT_NOT_IMPLEMENTED",
            message="This server currently supports append and replace_same_pair only.",
        )
        return {
            "status": "error",
            "added": 0,
            "updated": 0,
            "rejected": len(payload.relations),
            "relation_ids": [],
            "contradictions": [],
            "diagnostics": [diagnostic.model_dump(mode="json")],
            "trace": ["Rejected mode=upsert."],
            "graph_stats": store.graph_stats().model_dump(mode="json"),
        }

    records, updated = store.assert_relations(
        payload.relations,
        mode=payload.mode,
        dry_run=payload.dry_run,
    )
    contradictions: list[dict[str, Any]] = []
    diagnostics = []
    if payload.check_contradictions:
        check_relations = store.list_relations()
        if payload.dry_run:
            check_relations = [*check_relations, *records]
        contradictions, _context_separated = find_exclusive_contradictions(
            check_relations,
            store.list_exclusive_groups(),
            context_filter=ContextFilter(),
            max_depth=8,
        )

    trace = [_normalization_trace(record) for record in records]
    return {
        "status": "warning" if contradictions else "ok",
        "added": len(records),
        "updated": updated if payload.mode == "replace_same_pair" else 0,
        "rejected": 0,
        "relation_ids": [record.id for record in records],
        "contradictions": contradictions,
        "diagnostics": [item.model_dump(mode="json") for item in diagnostics],
        "trace": trace,
        "graph_stats": store.graph_stats().model_dump(mode="json"),
    }


async def list_relations(arguments: dict[str, Any], store: RelationStore) -> dict[str, Any]:
    """Handle `nesy.list_relations`."""
    payload = ListRelationsInput.model_validate(arguments)
    relations = store.list_relations(payload.filter, limit=payload.limit)
    stats_edges = store.implication_edges(relations)
    edges = stats_edges if payload.include_implication_edges else []
    exclusive_groups = store.list_exclusive_groups() if payload.include_exclusive_groups else []
    return {
        "status": "ok",
        "relations": [_record_dump(record) for record in relations],
        "implication_edges": [edge.model_dump(mode="json") for edge in edges],
        "exclusive_groups": [_exclusive_group_dump(group) for group in exclusive_groups],
        "total": len(relations),
        "next_cursor": None,
        "diagnostics": [],
        "trace": [f"Listed {len(relations)} relation(s)."],
        "graph_stats": graph_stats_for(
            relations,
            stats_edges,
            exclusive_group_count=len(store.list_exclusive_groups()),
        ).model_dump(mode="json"),
    }


async def clear_relations(arguments: dict[str, Any], store: RelationStore) -> dict[str, Any]:
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


async def assert_exclusive(arguments: dict[str, Any], store: RelationStore) -> dict[str, Any]:
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


async def check_contradictions(arguments: dict[str, Any], store: RelationStore) -> dict[str, Any]:
    """Handle `nesy.check_contradictions`."""
    payload = CheckContradictionsInput.model_validate(arguments)
    fact_records = _temporary_fact_records(payload)
    if payload.mode == ContradictionMode.GRAPH:
        relations = store.list_relations()
    elif payload.mode == ContradictionMode.FACTS:
        relations = fact_records
    else:
        relations = [*store.list_relations(), *fact_records]

    groups = store.list_exclusive_groups()
    contradictions, context_separated = find_exclusive_contradictions(
        relations,
        groups,
        payload.context_filter,
        max_depth=payload.max_depth,
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
