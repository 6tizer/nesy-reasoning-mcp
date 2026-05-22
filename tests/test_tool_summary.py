import json

import pytest

from nesy_reasoning_mcp.store import RelationStore
from nesy_reasoning_mcp.tools import (
    ASSERT_EXCLUSIVE,
    ASSERT_RELATIONS,
    SUMMARIZE_GRAPH,
    call_tool,
)


@pytest.mark.asyncio
async def test_summarize_graph_filters_sorts_and_mirrors_content() -> None:
    store = RelationStore()
    await call_tool(
        ASSERT_EXCLUSIVE,
        {
            "groups": [
                {
                    "group_id": "profit_state",
                    "members": ["利润增加", "利润减少"],
                    "context_id": "ctx",
                }
            ]
        },
        store,
    )
    await call_tool(
        ASSERT_RELATIONS,
        {
            "relations": [
                {
                    "id": "rel_z",
                    "source": "库存下降",
                    "target": "现金回收",
                    "relation_type": "sufficient",
                    "context_id": "ctx",
                    "metadata": {"domain": "retail"},
                },
                {
                    "id": "rel_a",
                    "source": "降价",
                    "target": "利润减少",
                    "relation_type": "sufficient",
                    "context_id": "ctx",
                    "metadata": {"domain": "retail"},
                },
                {
                    "id": "rel_other",
                    "source": "降价",
                    "target": "利润增加",
                    "relation_type": "sufficient",
                    "context_id": "other",
                    "metadata": {"domain": "retail"},
                },
            ],
            "check_contradictions": False,
        },
        store,
    )

    result = await call_tool(
        SUMMARIZE_GRAPH,
        {
            "focus_terms": ["利润"],
            "context_filter": {"context_id": "ctx", "domain": "retail"},
        },
        store,
    )

    assert result.isError is False
    assert json.loads(result.content[0].text) == result.structuredContent
    assert result.structuredContent["relation_count_included"] == 1
    assert "降价 sufficient 利润减少" in result.structuredContent["summary"]
    assert "profit_state" in result.structuredContent["summary"]
    assert "现金回收" not in result.structuredContent["summary"]
    assert "利润增加" in result.structuredContent["summary"]


@pytest.mark.asyncio
async def test_summarize_graph_truncates_by_relation_and_char_limits() -> None:
    store = RelationStore()
    await call_tool(
        ASSERT_RELATIONS,
        {
            "relations": [
                {
                    "id": "rel_1",
                    "source": "A",
                    "target": "B",
                    "relation_type": "sufficient",
                },
                {
                    "id": "rel_2",
                    "source": "C",
                    "target": "D",
                    "relation_type": "sufficient",
                },
            ],
            "check_contradictions": False,
        },
        store,
    )

    relation_limited = await call_tool(
        SUMMARIZE_GRAPH,
        {"max_relations": 1, "include_exclusives": False},
        store,
    )
    char_limited = await call_tool(
        SUMMARIZE_GRAPH,
        {"max_chars": 500, "include_exclusives": False},
        store,
    )
    invalid = await call_tool(SUMMARIZE_GRAPH, {"max_chars": 499}, store)

    assert relation_limited.structuredContent["truncated"] is True
    assert relation_limited.structuredContent["relation_count_included"] == 1
    assert char_limited.structuredContent["truncated"] is False
    assert invalid.isError is True
