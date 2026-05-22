import json

import pytest

from nesy_reasoning_mcp.config import NesyConfig, SecurityConfig
from nesy_reasoning_mcp.store import RelationStore
from nesy_reasoning_mcp.tools import (
    ASSERT_EXCLUSIVE,
    ASSERT_RELATIONS,
    CLEAR_RELATIONS,
    LIST_RELATIONS,
    call_tool,
)


@pytest.mark.asyncio
async def test_assert_and_list_relations_mcp_shape() -> None:
    store = RelationStore()
    result = await call_tool(
        ASSERT_RELATIONS,
        {
            "relations": [
                {
                    "source": "降价",
                    "target": "销量增加",
                    "relation_type": "sufficient",
                    "metadata": {"domain": "ecommerce"},
                }
            ]
        },
        store,
    )

    assert result.isError is False
    assert result.structuredContent is not None
    assert json.loads(result.content[0].text) == result.structuredContent
    assert result.structuredContent["added"] == 1
    assert result.structuredContent["diagnostics"] == []

    listed = await call_tool(
        LIST_RELATIONS,
        {
            "filter": {"source": "降价", "context_id": "default", "domain": "ecommerce"},
            "include_implication_edges": True,
        },
        store,
    )

    assert listed.isError is False
    assert listed.structuredContent["total"] == 1
    assert listed.structuredContent["relations"][0]["target"] == "销量增加"
    assert listed.structuredContent["implication_edges"][0]["antecedent"] == "降价"
    assert listed.structuredContent["next_cursor"] is None


@pytest.mark.asyncio
async def test_invalid_input_returns_error_result() -> None:
    store = RelationStore()
    result = await call_tool(
        ASSERT_RELATIONS,
        {"relations": [{"source": "A", "target": "B", "relation_type": "bad"}]},
        store,
    )

    assert result.isError is True
    assert result.structuredContent["status"] == "error"
    assert result.structuredContent["diagnostics"][0]["code"] == "INPUT_VALIDATION_ERROR"


@pytest.mark.asyncio
async def test_upsert_mode_is_rejected_in_v01() -> None:
    store = RelationStore()
    result = await call_tool(
        ASSERT_RELATIONS,
        {
            "relations": [{"source": "A", "target": "B", "relation_type": "sufficient"}],
            "mode": "upsert",
        },
        store,
    )

    assert result.isError is True
    assert result.structuredContent["rejected"] == 1
    assert result.structuredContent["diagnostics"][0]["code"] == "UPSERT_NOT_IMPLEMENTED"
    assert store.list_relations() == []


@pytest.mark.asyncio
async def test_clear_dry_run_and_actual_delete_by_context() -> None:
    store = RelationStore()
    await call_tool(
        ASSERT_RELATIONS,
        {
            "relations": [
                {"source": "A", "target": "B", "relation_type": "sufficient", "context_id": "c1"},
                {"source": "A", "target": "C", "relation_type": "sufficient", "context_id": "c2"},
            ],
            "check_contradictions": False,
        },
        store,
    )

    dry_run = await call_tool(
        CLEAR_RELATIONS,
        {"scope": "context", "context_id": "c1", "dry_run": True},
        store,
    )
    assert dry_run.structuredContent["removed_relations"] == 1
    assert len(store.list_relations()) == 2

    removed = await call_tool(CLEAR_RELATIONS, {"scope": "context", "context_id": "c1"}, store)
    assert removed.structuredContent["removed_relations"] == 1
    assert len(store.list_relations()) == 1


@pytest.mark.asyncio
async def test_clear_store_filter_and_reject_scope_all() -> None:
    store = RelationStore()
    await call_tool(
        ASSERT_RELATIONS,
        {
            "relations": [
                {
                    "source": "A",
                    "target": "B",
                    "relation_type": "sufficient",
                    "store_id": "s1",
                },
                {
                    "source": "C",
                    "target": "D",
                    "relation_type": "sufficient",
                    "store_id": "s2",
                },
            ],
            "check_contradictions": False,
        },
        store,
    )

    removed_store = await call_tool(CLEAR_RELATIONS, {"scope": "store", "store_id": "s1"}, store)
    assert removed_store.structuredContent["removed_relations"] == 1

    rejected = await call_tool(CLEAR_RELATIONS, {"scope": "all"}, store)
    assert rejected.isError is True
    assert rejected.structuredContent["diagnostics"][0]["code"] == "SCOPE_ALL_CLEAR_DISABLED"

    removed_filter = await call_tool(
        CLEAR_RELATIONS,
        {"scope": "filter", "filter": {"source": "C"}},
        store,
    )
    assert removed_filter.structuredContent["removed_relations"] == 1
    assert store.list_relations() == []


@pytest.mark.asyncio
async def test_scope_all_clear_requires_config_flag() -> None:
    store = RelationStore(NesyConfig(security=SecurityConfig(allow_scope_all_clear=True)))
    await call_tool(
        ASSERT_RELATIONS,
        {
            "relations": [{"source": "A", "target": "B", "relation_type": "sufficient"}],
            "check_contradictions": False,
        },
        store,
    )

    result = await call_tool(CLEAR_RELATIONS, {"scope": "all"}, store)

    assert result.isError is False
    assert result.structuredContent["removed_relations"] == 1
    assert store.list_relations() == []


@pytest.mark.asyncio
async def test_list_relations_stats_counts_edges_even_when_edges_omitted() -> None:
    store = RelationStore()
    await call_tool(
        ASSERT_RELATIONS,
        {
            "relations": [{"source": "A", "target": "B", "relation_type": "sufficient"}],
            "check_contradictions": False,
        },
        store,
    )

    result = await call_tool(LIST_RELATIONS, {"include_implication_edges": False}, store)

    assert result.structuredContent["implication_edges"] == []
    assert result.structuredContent["graph_stats"]["implication_edges"] == 1


@pytest.mark.asyncio
async def test_clear_filter_removes_matching_exclusive_groups() -> None:
    store = RelationStore()
    await call_tool(
        ASSERT_EXCLUSIVE,
        {
            "groups": [
                {"group_id": "state_c1", "members": ["B", "C"], "context_id": "c1"},
                {"group_id": "state_c2", "members": ["B", "C"], "context_id": "c2"},
            ]
        },
        store,
    )

    result = await call_tool(
        CLEAR_RELATIONS,
        {
            "scope": "filter",
            "filter": {"context_id": "c1"},
            "include_exclusive_groups": True,
        },
        store,
    )
    listed = await call_tool(LIST_RELATIONS, {"include_exclusive_groups": True}, store)

    assert result.structuredContent["removed_exclusive_groups"] == 1
    assert [group["group_id"] for group in listed.structuredContent["exclusive_groups"]] == [
        "state_c2"
    ]
