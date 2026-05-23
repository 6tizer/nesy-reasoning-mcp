"""Safe write helper for Agent SDK ingestion."""

from __future__ import annotations

from typing import Any

from nesy_reasoning_mcp.auto_ingest.policy import WRITE_MODE_TOOL_ALLOWLIST
from nesy_reasoning_mcp.schemas import Diagnostic, RelationInput
from nesy_reasoning_mcp.store import RelationStoreProtocol
from nesy_reasoning_mcp.tool_names import ASSERT_RELATIONS
from nesy_reasoning_mcp.tool_registry import call_tool


async def write_approved_relations(
    *,
    relations: list[RelationInput],
    store: RelationStoreProtocol,
) -> tuple[list[str], list[Diagnostic], dict[str, Any]]:
    """Persist gate-approved relations with contradiction rejection enabled."""
    if ASSERT_RELATIONS not in WRITE_MODE_TOOL_ALLOWLIST:
        raise RuntimeError("write-mode policy does not allow relation assertion")
    if not relations:
        return [], [], {}

    result = await call_tool(
        ASSERT_RELATIONS,
        {
            "relations": [
                relation.model_dump(mode="json", exclude_none=True) for relation in relations
            ],
            "on_contradiction": "reject",
            "check_contradictions": True,
        },
        store,
    )
    structured = dict(result.structuredContent or {})
    diagnostics = _diagnostics_from_structured(structured)
    if result.isError or structured.get("status") == "error":
        return [], diagnostics, structured
    relation_ids = [str(item) for item in structured.get("relation_ids", [])]
    return relation_ids, diagnostics, structured


def _diagnostics_from_structured(structured: dict[str, Any]) -> list[Diagnostic]:
    diagnostics: list[Diagnostic] = []
    for item in structured.get("diagnostics", []):
        if isinstance(item, dict):
            diagnostics.append(Diagnostic.model_validate(item))
    return diagnostics
