"""Compatibility exports for tool metadata and handlers."""

from __future__ import annotations

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
from nesy_reasoning_mcp.tool_reasoning import classify, verify_chain
from nesy_reasoning_mcp.tool_registry import call_tool, get_tools
from nesy_reasoning_mcp.tool_relations import (
    assert_exclusive,
    assert_relations,
    check_contradictions,
    clear_relations,
    list_relations,
)
from nesy_reasoning_mcp.tool_result import make_result
from nesy_reasoning_mcp.tool_summary import summarize_graph

__all__ = [
    "ASSERT_EXCLUSIVE",
    "ASSERT_RELATIONS",
    "CHECK_CONTRADICTIONS",
    "CLASSIFY",
    "CLEAR_RELATIONS",
    "COUNTERFACTUAL",
    "EXPORT_RELATIONS",
    "LIST_RELATIONS",
    "LOAD_RELATIONS",
    "SUMMARIZE_GRAPH",
    "VERIFY_CHAIN",
    "assert_exclusive",
    "assert_relations",
    "call_tool",
    "check_contradictions",
    "classify",
    "clear_relations",
    "counterfactual",
    "export_relations",
    "get_tools",
    "list_relations",
    "load_relations",
    "make_result",
    "summarize_graph",
    "verify_chain",
]
