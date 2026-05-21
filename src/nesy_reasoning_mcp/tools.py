"""Tool metadata and handlers."""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from typing import Any

from mcp.types import CallToolResult, TextContent, Tool
from pydantic import ValidationError

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
    ListRelationsInput,
    RelationRecord,
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
    }
    handler = handlers.get(name)
    if handler is None:
        raise ValueError(f"Unknown tool: {name}")

    try:
        structured = await handler(arguments, store)
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
    if payload.scope == "all" and not payload.dry_run:
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


def make_result(structured: dict[str, Any], *, is_error: bool = False) -> CallToolResult:
    """Build an MCP CallToolResult with mirrored JSON text content."""
    return CallToolResult(
        content=[TextContent(type="text", text=json.dumps(structured, ensure_ascii=False))],
        structuredContent=structured,
        isError=is_error,
    )


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
