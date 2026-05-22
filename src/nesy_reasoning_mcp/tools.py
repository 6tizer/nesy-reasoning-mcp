"""Tool metadata and handlers."""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from mcp.types import CallToolResult, TextContent, Tool
from pydantic import ValidationError

from nesy_reasoning_mcp.file_access import (
    read_allowed_relation_file,
    write_allowed_relation_file,
)
from nesy_reasoning_mcp.reasoning import (
    build_graph,
    classify_reachability,
    expected_relation_matches,
    find_exclusive_contradictions,
    path_to_dict,
    relations_compatible_with_filter,
)
from nesy_reasoning_mcp.schemas import (
    AssertExclusiveInput,
    AssertRelationsInput,
    CheckContradictionsInput,
    Classification,
    ClassifyInput,
    ClearRelationsInput,
    ContextFilter,
    ContradictionMode,
    Diagnostic,
    ExclusiveGroupRecord,
    ExpectedRelation,
    ExportDestination,
    ExportFormat,
    ExportRelationsInput,
    ListRelationsInput,
    LoadRelationsInput,
    LoadSourceType,
    RelationFilter,
    RelationRecord,
    RelationSetData,
    SummarizeGraphInput,
    VerifyChainInput,
)
from nesy_reasoning_mcp.store import RelationStore, graph_stats_for

ASSERT_RELATIONS = "nesy.assert_relations"
LIST_RELATIONS = "nesy.list_relations"
CLEAR_RELATIONS = "nesy.clear_relations"
CLASSIFY = "nesy.classify"
VERIFY_CHAIN = "nesy.verify_chain"
ASSERT_EXCLUSIVE = "nesy.assert_exclusive"
CHECK_CONTRADICTIONS = "nesy.check_contradictions"
LOAD_RELATIONS = "nesy.load_relations"
EXPORT_RELATIONS = "nesy.export_relations"
SUMMARIZE_GRAPH = "nesy.summarize_graph"


def get_tools() -> list[Tool]:
    """Return MCP tool definitions."""
    return [
        Tool(
            name=ASSERT_RELATIONS,
            title="Assert Logical Relations",
            description=(
                "Add one or more sufficient, necessary, or equivalent relations to the "
                "NeSy reasoning graph."
            ),
            inputSchema=AssertRelationsInput.model_json_schema(),
            outputSchema=_assert_relations_output_schema(),
        ),
        Tool(
            name=LIST_RELATIONS,
            title="List Relations",
            description="List stored relation records with optional filtering.",
            inputSchema=ListRelationsInput.model_json_schema(),
            outputSchema=_list_relations_output_schema(),
        ),
        Tool(
            name=CLEAR_RELATIONS,
            title="Clear Relations",
            description="Remove relation records by scope or filter.",
            inputSchema=ClearRelationsInput.model_json_schema(),
            outputSchema=_clear_relations_output_schema(),
        ),
        Tool(
            name=CLASSIFY,
            title="Classify Logical Relation",
            description=(
                "Classify whether source is sufficient, necessary, equivalent, "
                "unknown, or contradictory with respect to target."
            ),
            inputSchema=ClassifyInput.model_json_schema(),
            outputSchema=_classify_output_schema(),
        ),
        Tool(
            name=VERIFY_CHAIN,
            title="Verify Reasoning Chain",
            description=(
                "Verify an explicit reasoning chain or search for valid implication "
                "paths between source and target."
            ),
            inputSchema=VerifyChainInput.model_json_schema(),
            outputSchema=_verify_chain_output_schema(),
        ),
        Tool(
            name=ASSERT_EXCLUSIVE,
            title="Assert Exclusive Groups",
            description="Declare propositions that cannot all be true together under a context.",
            inputSchema=AssertExclusiveInput.model_json_schema(),
            outputSchema=_assert_exclusive_output_schema(),
        ),
        Tool(
            name=CHECK_CONTRADICTIONS,
            title="Check Logical Contradictions",
            description=(
                "Detect direct, transitive, and context-separated exclusivity-based "
                "contradictions in facts or the current graph."
            ),
            inputSchema=CheckContradictionsInput.model_json_schema(),
            outputSchema=_check_contradictions_output_schema(),
        ),
        Tool(
            name=LOAD_RELATIONS,
            title="Load Relations",
            description=(
                "Load relation records and exclusive groups from inline JSON or an "
                "allowed local file."
            ),
            inputSchema=LoadRelationsInput.model_json_schema(),
            outputSchema=_load_relations_output_schema(),
        ),
        Tool(
            name=EXPORT_RELATIONS,
            title="Export Relations",
            description=(
                "Export relation records and exclusive groups as JSON or JSONL, inline "
                "or to an allowed local file."
            ),
            inputSchema=ExportRelationsInput.model_json_schema(),
            outputSchema=_export_relations_output_schema(),
        ),
        Tool(
            name=SUMMARIZE_GRAPH,
            title="Summarize Reasoning Graph",
            description=(
                "Return a compact summary of the current reasoning graph for context "
                "injection and diagnostics."
            ),
            inputSchema=SummarizeGraphInput.model_json_schema(),
            outputSchema=_summarize_graph_output_schema(),
        ),
    ]


async def call_tool(name: str, arguments: dict[str, Any], store: RelationStore) -> CallToolResult:
    """Dispatch a tool call and return a complete MCP CallToolResult."""
    handlers: dict[str, Callable[[dict[str, Any], RelationStore], Awaitable[dict[str, Any]]]] = {
        ASSERT_RELATIONS: assert_relations,
        LIST_RELATIONS: list_relations,
        CLEAR_RELATIONS: clear_relations,
        CLASSIFY: classify,
        VERIFY_CHAIN: verify_chain,
        ASSERT_EXCLUSIVE: assert_exclusive,
        CHECK_CONTRADICTIONS: check_contradictions,
        LOAD_RELATIONS: load_relations,
        EXPORT_RELATIONS: export_relations,
        SUMMARIZE_GRAPH: summarize_graph,
    }
    handler = handlers.get(name)
    if handler is None:
        raise ValueError(f"Unknown tool: {name}")

    try:
        structured = await handler(arguments, store)
        _record_audit_if_needed(name, arguments, structured, store)
        return make_result(structured, is_error=structured.get("status") == "error")
    except ValidationError as exc:
        structured = _validation_error_content(exc)
        return make_result(structured, is_error=True)


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


async def classify(arguments: dict[str, Any], store: RelationStore) -> dict[str, Any]:
    """Handle `nesy.classify`."""
    payload = ClassifyInput.model_validate(arguments)
    index = build_graph(store.list_relations(), payload.context_filter)
    fwd_paths = index.find_paths(
        payload.source,
        payload.target,
        max_depth=payload.max_depth,
        max_paths=5,
        confidence_policy=payload.confidence_policy,
        direct_only=payload.require_direct,
    )
    rev_paths = index.find_paths(
        payload.target,
        payload.source,
        max_depth=payload.max_depth,
        max_paths=5,
        confidence_policy=payload.confidence_policy,
        direct_only=payload.require_direct,
    )
    classification = classify_reachability(fwd_paths, rev_paths)

    return {
        "status": "ok",
        "source": payload.source,
        "target": payload.target,
        "classification": classification.value,
        "source_implies_target": _implication_result(fwd_paths, payload.source, payload.target),
        "target_implies_source": _implication_result(rev_paths, payload.target, payload.source),
        "necessity_status": _necessity_status(rev_paths),
        "direct_relations": index.direct_relations_between(payload.source, payload.target),
        "paths": _classify_paths(fwd_paths, rev_paths, payload.include_paths),
        "diagnostics": [],
        "trace": _classify_trace(
            payload.source,
            payload.target,
            classification,
            fwd_paths,
            rev_paths,
        ),
        "graph_stats": index.graph_stats,
    }


async def verify_chain(arguments: dict[str, Any], store: RelationStore) -> dict[str, Any]:
    """Handle `nesy.verify_chain`."""
    payload = VerifyChainInput.model_validate(arguments)
    if payload.chain is not None and (
        payload.chain[0] != payload.source or payload.chain[-1] != payload.target
    ):
        diagnostic = Diagnostic(
            level="error",
            code="CHAIN_ENDPOINT_MISMATCH",
            message="Explicit chain must start at source and end at target.",
        )
        return {
            "status": "error",
            "reachable": False,
            "relation_type": Classification.UNKNOWN.value,
            "logic_validity": False,
            "best_path": None,
            "paths": [],
            "broken_at": None,
            "diagnostics": [diagnostic.model_dump(mode="json")],
            "trace": ["Rejected explicit chain with mismatched endpoints."],
            "graph_stats": store.graph_stats().model_dump(mode="json"),
        }

    index = build_graph(store.list_relations(), payload.context_filter)
    fwd_paths = index.find_paths(
        payload.source,
        payload.target,
        max_depth=payload.max_depth,
        strategy=payload.path_strategy,
        max_paths=payload.max_paths,
        confidence_policy=payload.confidence_policy,
    )
    rev_paths = index.find_paths(
        payload.target,
        payload.source,
        max_depth=payload.max_depth,
        strategy=payload.path_strategy,
        max_paths=payload.max_paths,
        confidence_policy=payload.confidence_policy,
    )
    classification = classify_reachability(fwd_paths, rev_paths)

    if payload.chain is not None:
        path, broken = index.verify_explicit_chain(
            payload.chain,
            confidence_policy=payload.confidence_policy,
        )
        return _explicit_chain_result(payload, index.graph_stats, classification, path, broken)

    return _searched_chain_result(payload, index.graph_stats, classification, fwd_paths, rev_paths)


async def load_relations(arguments: dict[str, Any], store: RelationStore) -> dict[str, Any]:
    """Handle `nesy.load_relations`."""
    payload = LoadRelationsInput.model_validate(arguments)
    if payload.source_type == LoadSourceType.RESOURCE_URI:
        return _load_error(
            "RESOURCE_URI_NOT_IMPLEMENTED_IN_V04",
            "resource_uri loading is not implemented in v0.4.",
            store,
        )

    try:
        data, source_trace = _load_relation_set_data(payload, store)
    except (OSError, ValueError, ValidationError) as exc:
        return _load_error("LOAD_RELATIONS_FAILED", str(exc), store)

    incoming_relations = _relations_for_store(data.relations, payload.store_id)
    incoming_groups = _groups_for_store(data.exclusive_groups, payload.store_id)
    contradictions: list[dict[str, Any]] = []
    if payload.check_contradictions:
        stored_relations = store.list_relations()
        stored_groups = store.list_exclusive_groups()
        if payload.mode.value == "replace_store":
            stored_relations = [
                relation for relation in stored_relations if relation.store_id != payload.store_id
            ]
            stored_groups = [group for group in stored_groups if group.store_id != payload.store_id]
        check_relations = [*stored_relations, *incoming_relations]
        contradictions, _context_separated = find_exclusive_contradictions(
            check_relations,
            [*stored_groups, *incoming_groups],
            context_filter=ContextFilter(store_id=payload.store_id),
            max_depth=8,
        )

    try:
        loaded_relations, loaded_groups, updated_relations, updated_groups = store.import_records(
            incoming_relations,
            incoming_groups,
            mode=payload.mode.value,
            store_id=payload.store_id,
            dry_run=payload.validate_only,
        )
    except Exception as exc:
        return _load_error("LOAD_RELATIONS_FAILED", str(exc), store)
    return {
        "status": "warning" if contradictions else "ok",
        "loaded_relations": loaded_relations,
        "loaded_exclusive_groups": loaded_groups,
        "updated_relations": updated_relations,
        "updated_exclusive_groups": updated_groups,
        "rejected": 0,
        "conflicts": contradictions,
        "validate_only": payload.validate_only,
        "diagnostics": [],
        "trace": [
            *source_trace,
            (
                "Validated relation set without changing store."
                if payload.validate_only
                else f"Loaded relation set with mode={payload.mode.value}."
            ),
        ],
        "graph_stats": store.graph_stats().model_dump(mode="json"),
    }


async def export_relations(arguments: dict[str, Any], store: RelationStore) -> dict[str, Any]:
    """Handle `nesy.export_relations`."""
    payload = ExportRelationsInput.model_validate(arguments)
    relations = store.list_relations(payload.filter, limit=None)
    exclusive_groups = (
        [
            group
            for group in store.list_exclusive_groups()
            if _exclusive_group_matches_filter(group, payload.filter)
        ]
        if payload.include_exclusive_groups
        else []
    )
    exported = _relation_set_export(
        relations,
        exclusive_groups,
        include_metadata=payload.include_metadata,
    )
    text = _serialize_relation_set(exported, payload.format)
    byte_count = len(text.encode("utf-8"))

    if payload.destination == ExportDestination.INLINE:
        if byte_count > payload.max_inline_bytes:
            return _export_error(
                "INLINE_EXPORT_TOO_LARGE",
                "Inline export exceeds max_inline_bytes.",
                store,
                payload.format,
            )
        return {
            "status": "ok",
            "format": payload.format.value,
            "relation_count": len(relations),
            "exclusive_group_count": len(exclusive_groups),
            "data": exported if payload.format == ExportFormat.JSON else text,
            "path": None,
            "bytes": byte_count,
            "diagnostics": [],
            "trace": [f"Exported {len(relations)} relation(s) inline."],
            "graph_stats": store.graph_stats().model_dump(mode="json"),
        }

    if payload.path is None:
        return _export_error(
            "EXPORT_PATH_REQUIRED",
            "path is required when destination=file.",
            store,
            payload.format,
        )
    if Path(payload.path).expanduser().suffix != f".{payload.format.value}":
        return _export_error(
            "EXPORT_EXTENSION_MISMATCH",
            "Export path suffix must match requested format.",
            store,
            payload.format,
        )

    try:
        real_path = write_allowed_relation_file(payload.path, store.config, text)
    except (OSError, ValueError) as exc:
        return _export_error("EXPORT_RELATIONS_FAILED", str(exc), store, payload.format)

    return {
        "status": "ok",
        "format": payload.format.value,
        "relation_count": len(relations),
        "exclusive_group_count": len(exclusive_groups),
        "data": None,
        "path": str(real_path),
        "bytes": byte_count,
        "diagnostics": [],
        "trace": [f"Exported {len(relations)} relation(s) to {real_path}."],
        "graph_stats": store.graph_stats().model_dump(mode="json"),
    }


async def summarize_graph(arguments: dict[str, Any], store: RelationStore) -> dict[str, Any]:
    """Handle `nesy.summarize_graph`."""
    payload = SummarizeGraphInput.model_validate(arguments)
    all_relations = store.list_relations()
    all_exclusive_groups = store.list_exclusive_groups()
    compatible_relations = relations_compatible_with_filter(all_relations, payload.context_filter)
    compatible_groups = [
        group
        for group in all_exclusive_groups
        if _exclusive_group_compatible_with_context_filter(group, payload.context_filter)
    ]
    relations = _summary_relations(compatible_relations, payload)
    selected_relations = relations[: payload.max_relations]
    relation_limit_truncated = len(relations) > len(selected_relations)
    exclusive_groups = _summary_exclusive_groups(compatible_groups, payload)
    summary, char_truncated = _format_graph_summary(
        selected_relations,
        exclusive_groups,
        payload=payload,
    )
    edges = store.implication_edges(compatible_relations)
    return {
        "status": "ok",
        "summary": summary,
        "relation_count_included": len(selected_relations),
        "truncated": relation_limit_truncated or char_truncated,
        "diagnostics": [],
        "trace": [
            f"Selected {len(selected_relations)} relation(s) matching summary filters.",
            f"Selected {len(exclusive_groups)} exclusive group(s).",
        ],
        "graph_stats": graph_stats_for(
            compatible_relations,
            edges,
            exclusive_group_count=len(compatible_groups),
        ).model_dump(mode="json"),
    }


def make_result(structured: dict[str, Any], *, is_error: bool = False) -> CallToolResult:
    """Build an MCP CallToolResult with mirrored JSON text content."""
    return CallToolResult(
        content=[TextContent(type="text", text=json.dumps(structured, ensure_ascii=False))],
        structuredContent=structured,
        isError=is_error,
    )


def _load_relation_set_data(
    payload: LoadRelationsInput,
    store: RelationStore,
) -> tuple[RelationSetData, list[str]]:
    if payload.source_type == LoadSourceType.INLINE:
        if payload.data is None:
            raise ValueError("data is required when source_type=inline")
        return payload.data, ["Read inline relation set."]

    if payload.source_type == LoadSourceType.FILE:
        if payload.path is None:
            raise ValueError("path is required when source_type=file")
        real_path, text = read_allowed_relation_file(payload.path, store.config)
        return _parse_relation_set_text(text, real_path.suffix), [
            f"Read relation set from {real_path}."
        ]

    raise ValueError(f"unsupported source_type: {payload.source_type.value}")


def _parse_relation_set_text(text: str, suffix: str) -> RelationSetData:
    if suffix == ".json":
        return RelationSetData.model_validate(json.loads(text))
    if suffix == ".jsonl":
        relations = []
        exclusive_groups = []
        for line_number, line in enumerate(text.splitlines(), start=1):
            stripped = line.strip()
            if not stripped:
                continue
            item = json.loads(stripped)
            item_type = item.get("type")
            record = item.get("record", item)
            if item_type == "exclusive_group" or "members" in record:
                exclusive_groups.append(record)
            elif item_type == "relation" or "relation_type" in record:
                relations.append(record)
            else:
                raise ValueError(f"line {line_number} is not a relation or exclusive group")
        return RelationSetData.model_validate(
            {"relations": relations, "exclusive_groups": exclusive_groups}
        )
    raise ValueError("only .json and .jsonl files are allowed")


def _relation_set_export(
    relations: list[RelationRecord],
    exclusive_groups: list[ExclusiveGroupRecord],
    *,
    include_metadata: bool,
) -> dict[str, Any]:
    relation_items = [_record_dump(record) for record in relations]
    group_items = [_exclusive_group_dump(group) for group in exclusive_groups]
    if not include_metadata:
        for item in relation_items:
            item.pop("metadata", None)
            item.pop("provenance", None)
        for item in group_items:
            item.pop("metadata", None)
    return {
        "version": "2.0",
        "relations": relation_items,
        "exclusive_groups": group_items,
    }


def _serialize_relation_set(data: dict[str, Any], export_format: ExportFormat) -> str:
    if export_format == ExportFormat.JSON:
        return json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True)
    lines = []
    lines.extend(
        json.dumps({"type": "relation", "record": relation}, ensure_ascii=False, sort_keys=True)
        for relation in data["relations"]
    )
    lines.extend(
        json.dumps(
            {"type": "exclusive_group", "record": group},
            ensure_ascii=False,
            sort_keys=True,
        )
        for group in data["exclusive_groups"]
    )
    return "\n".join(lines) + ("\n" if lines else "")


def _exclusive_group_matches_filter(
    group: ExclusiveGroupRecord,
    relation_filter: RelationFilter,
) -> bool:
    if relation_filter.context_id is not None and group.context_id != relation_filter.context_id:
        return False
    if relation_filter.store_id is not None and group.store_id != relation_filter.store_id:
        return False
    return not (
        relation_filter.domain is not None
        and group.metadata.get("domain") != relation_filter.domain
    )


def _summary_relations(
    relations: list[RelationRecord],
    payload: SummarizeGraphInput,
) -> list[RelationRecord]:
    terms = _normalized_focus_terms(payload.focus_terms)
    selected = [
        relation
        for relation in relations
        if not terms or _text_matches_terms([relation.source, relation.target], terms)
    ]
    return sorted(
        selected,
        key=lambda item: (
            item.store_id,
            item.context_id,
            item.source,
            item.target,
            item.relation_type.value,
            item.id,
        ),
    )


def _summary_exclusive_groups(
    groups: list[ExclusiveGroupRecord],
    payload: SummarizeGraphInput,
) -> list[ExclusiveGroupRecord]:
    if not payload.include_exclusives:
        return []
    terms = _normalized_focus_terms(payload.focus_terms)
    selected = [
        group
        for group in groups
        if _exclusive_group_compatible_with_context_filter(group, payload.context_filter)
        and (not terms or _text_matches_terms(group.members, terms))
    ]
    return sorted(
        selected,
        key=lambda item: (item.store_id, item.context_id, item.group_id),
    )


def _exclusive_group_compatible_with_context_filter(
    group: ExclusiveGroupRecord,
    context_filter: ContextFilter,
) -> bool:
    if context_filter.store_id and group.store_id != context_filter.store_id:
        return False
    return not (
        context_filter.context_id
        and group.scope.value == "same_context"
        and group.context_id != context_filter.context_id
    )


def _format_graph_summary(
    relations: list[RelationRecord],
    exclusive_groups: list[ExclusiveGroupRecord],
    *,
    payload: SummarizeGraphInput,
) -> tuple[str, bool]:
    context_parts = []
    if payload.context_filter.store_id:
        context_parts.append(f"store={payload.context_filter.store_id}")
    if payload.context_filter.context_id:
        context_parts.append(f"context={payload.context_filter.context_id}")
    if payload.context_filter.domain:
        context_parts.append(f"domain={payload.context_filter.domain}")
    title = "Known NeSy reasoning graph"
    if context_parts:
        title = f"{title} ({', '.join(context_parts)})"

    lines = [f"{title}:"]
    if relations:
        lines.extend(_relation_summary_line(relation) for relation in relations)
    else:
        lines.append("- No matching relations.")

    if payload.include_exclusives:
        lines.append("Exclusive groups:")
        if exclusive_groups:
            lines.extend(_exclusive_group_summary_line(group) for group in exclusive_groups)
        else:
            lines.append("- No matching exclusive groups.")

    summary = "\n".join(lines)
    if len(summary) <= payload.max_chars:
        return summary, False

    suffix = "\n...truncated"
    cutoff = max(0, payload.max_chars - len(suffix))
    return f"{summary[:cutoff].rstrip()}{suffix}", True


def _relation_summary_line(relation: RelationRecord) -> str:
    return (
        f"- {relation.source} {relation.relation_type.value} {relation.target} "
        f"(conf={relation.confidence:g}, context={relation.context_id}, "
        f"store={relation.store_id}, id={relation.id})"
    )


def _exclusive_group_summary_line(group: ExclusiveGroupRecord) -> str:
    return (
        f"- {group.group_id}: {' | '.join(group.members)} "
        f"(context={group.context_id}, store={group.store_id}, scope={group.scope.value})"
    )


def _normalized_focus_terms(focus_terms: list[str]) -> list[str]:
    return [term.casefold() for term in focus_terms]


def _text_matches_terms(values: list[str], terms: list[str]) -> bool:
    haystack = "\n".join(values).casefold()
    return any(term in haystack for term in terms)


def _relations_for_store(
    relations: list[RelationRecord],
    store_id: str,
) -> list[RelationRecord]:
    return [relation.model_copy(update={"store_id": store_id}) for relation in relations]


def _groups_for_store(
    groups: list[ExclusiveGroupRecord],
    store_id: str,
) -> list[ExclusiveGroupRecord]:
    return [group.model_copy(update={"store_id": store_id}) for group in groups]


def _load_error(code: str, message: str, store: RelationStore) -> dict[str, Any]:
    diagnostic = Diagnostic(level="error", code=code, message=message)
    return {
        "status": "error",
        "loaded_relations": 0,
        "loaded_exclusive_groups": 0,
        "updated_relations": 0,
        "updated_exclusive_groups": 0,
        "rejected": 0,
        "conflicts": [],
        "validate_only": False,
        "diagnostics": [diagnostic.model_dump(mode="json")],
        "trace": ["Rejected relation load."],
        "graph_stats": store.graph_stats().model_dump(mode="json"),
    }


def _export_error(
    code: str,
    message: str,
    store: RelationStore,
    export_format: ExportFormat,
) -> dict[str, Any]:
    diagnostic = Diagnostic(level="error", code=code, message=message)
    return {
        "status": "error",
        "format": export_format.value,
        "relation_count": 0,
        "exclusive_group_count": 0,
        "data": None,
        "path": None,
        "bytes": 0,
        "diagnostics": [diagnostic.model_dump(mode="json")],
        "trace": ["Rejected relation export."],
        "graph_stats": store.graph_stats().model_dump(mode="json"),
    }


def _record_audit_if_needed(
    name: str,
    arguments: dict[str, Any],
    structured: dict[str, Any],
    store: RelationStore,
) -> None:
    if not _should_audit(name, arguments):
        return
    store.record_audit(
        event_type="tool_call",
        tool_name=name,
        arguments=arguments,
        result_status=str(structured.get("status", "unknown")),
        metadata={"is_error": structured.get("status") == "error"},
    )


def _should_audit(name: str, arguments: dict[str, Any]) -> bool:
    if name == ASSERT_RELATIONS:
        return not bool(arguments.get("dry_run", False))
    if name == ASSERT_EXCLUSIVE:
        return True
    if name == CLEAR_RELATIONS:
        return not bool(arguments.get("dry_run", False))
    if name == LOAD_RELATIONS:
        return not bool(arguments.get("validate_only", False))
    if name == EXPORT_RELATIONS:
        return arguments.get("destination") == "file"
    return False


def _validation_error_content(exc: ValidationError) -> dict[str, Any]:
    return {
        "status": "error",
        "diagnostics": [
            {
                "level": "error",
                "code": "INPUT_VALIDATION_ERROR",
                "message": str(exc),
                "related_ids": [],
            }
        ],
        "trace": [],
        "graph_stats": {
            "relations": 0,
            "propositions": 0,
            "implication_edges": 0,
            "exclusive_groups": 0,
            "contexts": 0,
            "stores": 0,
        },
    }


def _record_dump(record: RelationRecord) -> dict[str, Any]:
    return record.model_dump(mode="json")


def _exclusive_group_dump(group: ExclusiveGroupRecord) -> dict[str, Any]:
    return group.model_dump(mode="json")


def _temporary_fact_records(payload: CheckContradictionsInput) -> list[RelationRecord]:
    records: list[RelationRecord] = []
    for index, fact in enumerate(payload.facts):
        data = fact.model_dump()
        if data["id"] is None:
            data["id"] = f"input_{index}"
        records.append(RelationRecord(**data))
    return records


def _contradiction_trace(
    mode: ContradictionMode,
    fact_count: int,
    contradictions: list[dict[str, Any]],
) -> list[str]:
    trace = [f"Checked contradictions in mode={mode.value}."]
    if fact_count:
        trace.append(f"Loaded {fact_count} input fact(s) into temporary graph.")
    if contradictions:
        trace.append(f"Found {len(contradictions)} hard contradiction(s).")
    else:
        trace.append("No hard contradictions found.")
    return trace


def _implication_result(paths: list, start: str, end: str) -> dict[str, Any]:
    if paths:
        best = paths[0]
        return {
            "proven": True,
            "logic_validity": True,
            "evidence_confidence": best.evidence_confidence,
            "best_path": best.nodes,
        }
    return {
        "proven": False,
        "logic_validity": False,
        "reason": f"No path found from {start} to {end} within max_depth under the context filter.",
    }


def _necessity_status(reverse_paths: list) -> dict[str, Any]:
    if reverse_paths:
        return {
            "status": "proven_necessary",
            "reason": "Target implies source under the current graph and context filter.",
        }
    return {
        "status": "unknown",
        "reason": (
            "No proof that target implies source; absence of proof is not proof of non-necessity."
        ),
    }


def _classify_paths(fwd_paths: list, rev_paths: list, include_paths: bool) -> list[dict[str, Any]]:
    if not include_paths:
        return []
    paths: list[dict[str, Any]] = []
    paths.extend(
        {
            "direction": "source_to_target",
            **path_to_dict(path, relation_type=Classification.SUFFICIENT.value),
        }
        for path in fwd_paths
    )
    paths.extend(
        {
            "direction": "target_to_source",
            **path_to_dict(path, relation_type=Classification.NECESSARY.value),
        }
        for path in rev_paths
    )
    return paths


def _classify_trace(
    source: str,
    target: str,
    classification: Classification,
    fwd_paths: list,
    rev_paths: list,
) -> list[str]:
    if source == target:
        return ["Source and target are identical; using zero-length identity path."]
    trace = [f"Checked implication paths between {source} and {target}."]
    source_path = (
        f"Found source-to-target path: {fwd_paths[0].nodes}"
        if fwd_paths
        else "No source-to-target path found."
    )
    target_path = (
        f"Found target-to-source path: {rev_paths[0].nodes}"
        if rev_paths
        else "No target-to-source path found."
    )
    trace.append(source_path)
    trace.append(target_path)
    trace.append(f"Mapped reachability to classification: {classification.value}.")
    return trace


def _explicit_chain_result(
    payload: VerifyChainInput,
    graph_stats: dict[str, Any],
    classification: Classification,
    path: Any,
    broken: Any,
) -> dict[str, Any]:
    diagnostics = []
    reachable = path is not None
    expected_ok = reachable and expected_relation_matches(payload.expected_relation, classification)
    if broken is not None and broken.direction_mismatch:
        diagnostics.append(
            Diagnostic(
                level="warning",
                code="DIRECTION_MISMATCH",
                message="The declared relation points in the reverse implication direction.",
            ).model_dump(mode="json")
        )
    if reachable and not expected_ok:
        diagnostics.append(_expected_mismatch(payload.expected_relation, classification))
    return {
        "status": "ok",
        "reachable": reachable,
        "relation_type": classification.value,
        "logic_validity": expected_ok,
        "best_path": path_to_dict(path) if path is not None else None,
        "paths": [],
        "broken_at": broken.to_dict() if broken is not None else None,
        "diagnostics": diagnostics,
        "trace": _explicit_chain_trace(payload.chain or [], path, broken, expected_ok),
        "graph_stats": graph_stats,
    }


def _searched_chain_result(
    payload: VerifyChainInput,
    graph_stats: dict[str, Any],
    classification: Classification,
    fwd_paths: list,
    rev_paths: list,
) -> dict[str, Any]:
    reachable = classification != Classification.UNKNOWN
    expected_ok = expected_relation_matches(payload.expected_relation, classification)
    best_path = fwd_paths[0] if fwd_paths else (rev_paths[0] if rev_paths else None)
    diagnostics = (
        [] if expected_ok else [_expected_mismatch(payload.expected_relation, classification)]
    )
    all_paths = []
    if payload.path_strategy.value == "all":
        all_paths.extend(
            {"direction": "source_to_target", **path_to_dict(path)} for path in fwd_paths
        )
        all_paths.extend(
            {"direction": "target_to_source", **path_to_dict(path)} for path in rev_paths
        )
    diagnostics_out = (
        diagnostics if reachable or payload.expected_relation != ExpectedRelation.ANY else []
    )
    return {
        "status": "ok",
        "reachable": reachable,
        "relation_type": classification.value,
        "logic_validity": reachable and expected_ok,
        "best_path": path_to_dict(best_path) if best_path is not None else None,
        "paths": all_paths,
        "broken_at": None,
        "diagnostics": diagnostics_out,
        "trace": _searched_chain_trace(payload.source, payload.target, classification, best_path),
        "graph_stats": graph_stats,
    }


def _expected_mismatch(
    expected: ExpectedRelation,
    classification: Classification,
) -> dict[str, Any]:
    return Diagnostic(
        level="warning",
        code="EXPECTED_RELATION_MISMATCH",
        message=(
            f"Expected relation {expected.value} did not match classified relation "
            f"{classification.value}."
        ),
    ).model_dump(mode="json")


def _explicit_chain_trace(
    chain: list[str],
    path: Any,
    broken: Any,
    expected_ok: bool,
) -> list[str]:
    if broken is not None:
        return [
            f"{broken.from_node} -> {broken.to_node} not found.",
            f"Chain is broken at step {broken.index}.",
        ]
    trace = [f"Explicit chain verified: {' -> '.join(chain)}."]
    trace.append(
        "Expected relation matched." if expected_ok else "Expected relation did not match."
    )
    if path is not None:
        trace.append(f"Verified {len(path.edges)} edge(s).")
    return trace


def _searched_chain_trace(
    source: str,
    target: str,
    classification: Classification,
    best_path: Any,
) -> list[str]:
    trace = [f"Searched implication graph between {source} and {target}."]
    if best_path is not None:
        trace.append(f"Found path with {len(best_path.edges)} edge(s): {best_path.nodes}.")
    else:
        trace.append("No valid implication path found in either direction.")
    trace.append(f"Mapped reachability to relation_type: {classification.value}.")
    return trace


def _normalization_trace(record: RelationRecord) -> str:
    if record.relation_type == "necessary":
        return (
            f"normalized necessary({record.source}, {record.target}) into implication edge "
            f"{record.target} -> {record.source}"
        )
    if record.relation_type == "equivalent":
        return (
            f"normalized equivalent({record.source}, {record.target}) into implication edges "
            f"{record.source} -> {record.target} and {record.target} -> {record.source}"
        )
    return (
        f"normalized sufficient({record.source}, {record.target}) into implication edge "
        f"{record.source} -> {record.target}"
    )


def _common_output_properties() -> dict[str, Any]:
    return {
        "status": {"type": "string", "enum": ["ok", "warning", "error"]},
        "diagnostics": {"type": "array"},
        "trace": {"type": "array"},
        "graph_stats": {"type": "object"},
    }


def _assert_relations_output_schema() -> dict[str, Any]:
    props = _common_output_properties()
    props.update(
        {
            "added": {"type": "integer"},
            "updated": {"type": "integer"},
            "rejected": {"type": "integer"},
            "relation_ids": {"type": "array", "items": {"type": "string"}},
            "contradictions": {"type": "array"},
        }
    )
    return {
        "type": "object",
        "properties": props,
        "required": ["status", "added", "updated", "rejected", "relation_ids"],
        "additionalProperties": False,
    }


def _list_relations_output_schema() -> dict[str, Any]:
    props = _common_output_properties()
    props.update(
        {
            "relations": {"type": "array"},
            "implication_edges": {"type": "array"},
            "exclusive_groups": {"type": "array"},
            "total": {"type": "integer"},
            "next_cursor": {"type": ["string", "null"]},
        }
    )
    return {
        "type": "object",
        "properties": props,
        "required": ["status", "relations", "total"],
        "additionalProperties": False,
    }


def _clear_relations_output_schema() -> dict[str, Any]:
    props = _common_output_properties()
    props.update(
        {
            "removed_relations": {"type": "integer"},
            "removed_exclusive_groups": {"type": "integer"},
            "dry_run": {"type": "boolean"},
        }
    )
    return {
        "type": "object",
        "properties": props,
        "required": ["status", "removed_relations", "removed_exclusive_groups", "dry_run"],
        "additionalProperties": False,
    }


def _classify_output_schema() -> dict[str, Any]:
    props = _common_output_properties()
    props.update(
        {
            "source": {"type": "string"},
            "target": {"type": "string"},
            "classification": {
                "type": "string",
                "enum": ["sufficient", "necessary", "equivalent", "unknown", "contradictory"],
            },
            "source_implies_target": {"type": "object"},
            "target_implies_source": {"type": "object"},
            "necessity_status": {"type": "object"},
            "direct_relations": {"type": "array"},
            "paths": {"type": "array"},
        }
    )
    return {
        "type": "object",
        "properties": props,
        "required": ["status", "source", "target", "classification"],
        "additionalProperties": False,
    }


def _verify_chain_output_schema() -> dict[str, Any]:
    props = _common_output_properties()
    props.update(
        {
            "reachable": {"type": "boolean"},
            "relation_type": {"type": "string"},
            "logic_validity": {"type": "boolean"},
            "best_path": {"type": ["object", "null"]},
            "paths": {"type": "array"},
            "broken_at": {"type": ["object", "null"]},
        }
    )
    return {
        "type": "object",
        "properties": props,
        "required": ["status", "reachable", "logic_validity"],
        "additionalProperties": False,
    }


def _assert_exclusive_output_schema() -> dict[str, Any]:
    props = _common_output_properties()
    props.update(
        {
            "added_groups": {"type": "integer"},
            "updated_groups": {"type": "integer"},
            "group_ids": {"type": "array", "items": {"type": "string"}},
        }
    )
    return {
        "type": "object",
        "properties": props,
        "required": ["status", "added_groups", "updated_groups", "group_ids"],
        "additionalProperties": False,
    }


def _check_contradictions_output_schema() -> dict[str, Any]:
    props = _common_output_properties()
    props.update(
        {
            "has_contradictions": {"type": "boolean"},
            "contradictions": {"type": "array"},
            "clean_facts_count": {"type": "integer"},
            "total_facts_count": {"type": "integer"},
            "context_separated": {"type": "array"},
        }
    )
    return {
        "type": "object",
        "properties": props,
        "required": ["status", "has_contradictions", "contradictions"],
        "additionalProperties": False,
    }


def _load_relations_output_schema() -> dict[str, Any]:
    props = _common_output_properties()
    props.update(
        {
            "loaded_relations": {"type": "integer"},
            "loaded_exclusive_groups": {"type": "integer"},
            "updated_relations": {"type": "integer"},
            "updated_exclusive_groups": {"type": "integer"},
            "rejected": {"type": "integer"},
            "conflicts": {"type": "array"},
            "validate_only": {"type": "boolean"},
        }
    )
    return {
        "type": "object",
        "properties": props,
        "required": ["status", "loaded_relations", "loaded_exclusive_groups", "rejected"],
        "additionalProperties": False,
    }


def _export_relations_output_schema() -> dict[str, Any]:
    props = _common_output_properties()
    props.update(
        {
            "format": {"type": "string", "enum": ["json", "jsonl"]},
            "relation_count": {"type": "integer"},
            "exclusive_group_count": {"type": "integer"},
            "data": {"type": ["object", "string", "null"]},
            "path": {"type": ["string", "null"]},
            "bytes": {"type": "integer"},
        }
    )
    return {
        "type": "object",
        "properties": props,
        "required": ["status", "format", "relation_count", "exclusive_group_count"],
        "additionalProperties": False,
    }


def _summarize_graph_output_schema() -> dict[str, Any]:
    props = _common_output_properties()
    props.update(
        {
            "summary": {"type": "string"},
            "relation_count_included": {"type": "integer"},
            "truncated": {"type": "boolean"},
        }
    )
    return {
        "type": "object",
        "properties": props,
        "required": ["status", "summary", "relation_count_included", "truncated"],
        "additionalProperties": False,
    }
