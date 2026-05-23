"""Tool permission policy for Agent SDK ingestion workflows."""

from __future__ import annotations

from nesy_reasoning_mcp.tool_names import (
    ASSERT_RELATIONS,
    CHECK_CONTRADICTIONS,
    LIST_RELATIONS,
    REASON_OVER_RELATIONS,
    SUMMARIZE_GRAPH,
)

DRY_RUN_TOOL_ALLOWLIST = frozenset(
    {
        REASON_OVER_RELATIONS,
        CHECK_CONTRADICTIONS,
        SUMMARIZE_GRAPH,
        LIST_RELATIONS,
    }
)
WRITE_MODE_TOOL_ALLOWLIST = DRY_RUN_TOOL_ALLOWLIST | frozenset(
    {
        ASSERT_RELATIONS,
    }
)
