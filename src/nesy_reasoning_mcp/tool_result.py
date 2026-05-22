"""MCP tool result helpers."""

from __future__ import annotations

import json
from typing import Any

from mcp.types import CallToolResult, TextContent
from pydantic import ValidationError


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
