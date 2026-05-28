"""Compatibility exports for tool metadata and handlers."""

from __future__ import annotations

from nesy_reasoning_mcp.tool_candidate_validation import validate_candidate_relations
from nesy_reasoning_mcp.tool_counterfactual import counterfactual
from nesy_reasoning_mcp.tool_ephemeral import reason_over_relations
from nesy_reasoning_mcp.tool_io import export_relations, load_relations
from nesy_reasoning_mcp.tool_names import (
    ASSERT_EXCLUSIVE,
    ASSERT_RELATIONS,
    CHECK_CONTRADICTIONS,
    CLASSIFY,
    CLEAR_RELATIONS,
    COMMIT_REVIEWED_RELATIONS,
    COUNTERFACTUAL,
    EXPORT_RELATIONS,
    LIST_RELATIONS,
    LIST_REVIEW_QUEUE,
    LOAD_RELATIONS,
    QUEUE_STATUS,
    REASON_OVER_RELATIONS,
    RESOLVE_REVIEW_QUEUE,
    SUMMARIZE_GRAPH,
    VALIDATE_CANDIDATE_RELATIONS,
    VERIFY_CHAIN,
)
from nesy_reasoning_mcp.tool_queue_status import queue_status
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
from nesy_reasoning_mcp.tool_review_queue import (
    commit_reviewed_relations,
    list_review_queue,
    resolve_review_queue,
)
from nesy_reasoning_mcp.tool_summary import summarize_graph

__all__ = [
    "ASSERT_EXCLUSIVE",
    "ASSERT_RELATIONS",
    "CHECK_CONTRADICTIONS",
    "CLASSIFY",
    "CLEAR_RELATIONS",
    "COMMIT_REVIEWED_RELATIONS",
    "COUNTERFACTUAL",
    "EXPORT_RELATIONS",
    "LIST_RELATIONS",
    "LIST_REVIEW_QUEUE",
    "LOAD_RELATIONS",
    "QUEUE_STATUS",
    "REASON_OVER_RELATIONS",
    "RESOLVE_REVIEW_QUEUE",
    "SUMMARIZE_GRAPH",
    "VALIDATE_CANDIDATE_RELATIONS",
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
    "commit_reviewed_relations",
    "list_relations",
    "list_review_queue",
    "load_relations",
    "make_result",
    "queue_status",
    "reason_over_relations",
    "resolve_review_queue",
    "summarize_graph",
    "validate_candidate_relations",
    "verify_chain",
]
