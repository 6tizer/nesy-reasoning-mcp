"""MCP tool registry and dispatch."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from mcp.types import CallToolResult, Tool
from pydantic import ValidationError

from nesy_reasoning_mcp.schemas import (
    AssertExclusiveInput,
    AssertRelationsInput,
    CheckContradictionsInput,
    ClassifyInput,
    ClearRelationsInput,
    CounterfactualInput,
    ExportRelationsInput,
    ListRelationsInput,
    LoadRelationsInput,
    SummarizeGraphInput,
    VerifyChainInput,
)
from nesy_reasoning_mcp.store import RelationStoreProtocol
from nesy_reasoning_mcp.tool_common import _record_audit_if_needed
from nesy_reasoning_mcp.tool_counterfactual import counterfactual
from nesy_reasoning_mcp.tool_io import export_relations, load_relations
from nesy_reasoning_mcp.tool_names import (
    ASSERT_EXCLUSIVE,
    ASSERT_RELATIONS,
    CHECK_CONTRADICTIONS,
    CLASSIFY,
    CLEAR_RELATIONS,
    COUNTERFACTUAL,
    EXPORT_RELATIONS,
    LIST_RELATIONS,
    LOAD_RELATIONS,
    SUMMARIZE_GRAPH,
    VERIFY_CHAIN,
)
from nesy_reasoning_mcp.tool_output_schemas import (
    _assert_exclusive_output_schema,
    _assert_relations_output_schema,
    _check_contradictions_output_schema,
    _classify_output_schema,
    _clear_relations_output_schema,
    _counterfactual_output_schema,
    _export_relations_output_schema,
    _list_relations_output_schema,
    _load_relations_output_schema,
    _summarize_graph_output_schema,
    _verify_chain_output_schema,
)
from nesy_reasoning_mcp.tool_reasoning import classify, verify_chain
from nesy_reasoning_mcp.tool_relations import (
    assert_exclusive,
    assert_relations,
    check_contradictions,
    clear_relations,
    list_relations,
)
from nesy_reasoning_mcp.tool_result import (
    _validation_error_content,
    audit_failure_diagnostic,
    make_result,
    runtime_error_content,
    unknown_tool_content,
)
from nesy_reasoning_mcp.tool_summary import summarize_graph


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
                "Load relation records, proposition records, exclusive groups, and "
                "independence records from inline JSON, an allowed local file, or a "
                "safe file resource URI."
            ),
            inputSchema=LoadRelationsInput.model_json_schema(),
            outputSchema=_load_relations_output_schema(),
        ),
        Tool(
            name=EXPORT_RELATIONS,
            title="Export Relations",
            description=(
                "Export relation records, proposition records, exclusive groups, and "
                "independence records as JSON or JSONL, inline or to an allowed local file."
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
        Tool(
            name=COUNTERFACTUAL,
            title="Counterfactual Reasoning",
            description=(
                "Analyze what is necessarily blocked, possibly blocked, still possible, "
                "or unknown if a proposition is assumed false."
            ),
            inputSchema=CounterfactualInput.model_json_schema(),
            outputSchema=_counterfactual_output_schema(),
        ),
    ]


async def call_tool(
    name: str,
    arguments: dict[str, Any],
    store: RelationStoreProtocol,
) -> CallToolResult:
    """Dispatch a tool call and return a complete MCP CallToolResult."""
    handlers: dict[
        str,
        Callable[[dict[str, Any], RelationStoreProtocol], Awaitable[dict[str, Any]]],
    ] = {
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
        COUNTERFACTUAL: counterfactual,
    }
    handler = handlers.get(name)
    if handler is None:
        return make_result(unknown_tool_content(name, store), is_error=True)

    try:
        structured = await handler(arguments, store)
    except ValidationError as exc:
        structured = _validation_error_content(exc, store)
        return make_result(structured, is_error=True)
    except Exception as exc:
        structured = runtime_error_content(name, exc, store)
        return make_result(structured, is_error=True)

    try:
        _record_audit_if_needed(name, arguments, structured, store)
    except Exception as exc:
        structured.setdefault("diagnostics", []).append(audit_failure_diagnostic(exc))
    return make_result(structured, is_error=structured.get("status") == "error")
