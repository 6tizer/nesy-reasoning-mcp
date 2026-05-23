"""MCP tool result helpers."""

from __future__ import annotations

import json
from typing import Any

from mcp.types import CallToolResult, TextContent
from pydantic import ValidationError

ZERO_GRAPH_STATS = {
    "relations": 0,
    "propositions": 0,
    "implication_edges": 0,
    "exclusive_groups": 0,
    "contexts": 0,
    "stores": 0,
}


def make_result(structured: dict[str, Any], *, is_error: bool = False) -> CallToolResult:
    """Build an MCP CallToolResult with mirrored JSON text content."""
    return CallToolResult(
        content=[TextContent(type="text", text=json.dumps(structured, ensure_ascii=False))],
        structuredContent=structured,
        isError=is_error,
    )


def _validation_error_content(exc: ValidationError, store: Any | None = None) -> dict[str, Any]:
    """Return structured content for input validation failures."""
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
        "graph_stats": _safe_graph_stats(store),
    }


def runtime_error_content(name: str, exc: Exception, store: Any | None = None) -> dict[str, Any]:
    """Return structured content for unexpected tool runtime failures."""
    return {
        "status": "error",
        "diagnostics": [
            {
                "level": "error",
                "code": "TOOL_RUNTIME_ERROR",
                "message": str(exc),
                "related_ids": [],
            }
        ],
        "trace": [f"{name} failed at runtime."],
        "graph_stats": _safe_graph_stats(store),
    }


def unknown_tool_content(name: str, store: Any | None = None) -> dict[str, Any]:
    """Return structured content for unknown MCP tool names."""
    return {
        "status": "error",
        "diagnostics": [
            {
                "level": "error",
                "code": "UNKNOWN_TOOL",
                "message": f"Unknown tool: {name}",
                "related_ids": [],
            }
        ],
        "trace": [f"Rejected unknown tool: {name}."],
        "graph_stats": _safe_graph_stats(store),
    }


def audit_failure_diagnostic(exc: Exception) -> dict[str, Any]:
    """Return a warning diagnostic when audit logging fails after a tool call."""
    return {
        "level": "warning",
        "code": "AUDIT_LOG_FAILED",
        "message": str(exc),
        "related_ids": [],
    }


def _safe_graph_stats(store: Any | None) -> dict[str, Any]:
    if store is None:
        return dict(ZERO_GRAPH_STATS)
    try:
        return store.graph_stats().model_dump(mode="json")
    except Exception:
        return dict(ZERO_GRAPH_STATS)
