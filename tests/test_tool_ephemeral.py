import json

import pytest

from nesy_reasoning_mcp.store import RelationStore
from nesy_reasoning_mcp.tools import (
    ASSERT_EXCLUSIVE,
    ASSERT_RELATIONS,
    CHECK_CONTRADICTIONS,
    LIST_RELATIONS,
    REASON_OVER_RELATIONS,
    call_tool,
)


@pytest.mark.asyncio
async def test_reason_over_relations_classifies_without_persisting() -> None:
    store = RelationStore()

    result = await call_tool(
        REASON_OVER_RELATIONS,
        {
            "relations": [{"source": "A", "target": "B", "relation_type": "sufficient"}],
            "query": {"mode": "classify", "source": "A", "target": "B"},
        },
        store,
    )
    listed = await call_tool(LIST_RELATIONS, {}, store)

    assert result.isError is False
    assert result.structuredContent["mode"] == "classify"
    assert result.structuredContent["persisted"] is False
    assert result.structuredContent["result"]["classification"] == "sufficient"
    assert listed.structuredContent["relations"] == []
    assert json.loads(result.content[0].text) == result.structuredContent


@pytest.mark.asyncio
async def test_reason_over_relations_verifies_multihop_chain_without_persisting() -> None:
    store = RelationStore()

    result = await call_tool(
        REASON_OVER_RELATIONS,
        {
            "relations": [
                {"source": "A", "target": "B", "relation_type": "sufficient"},
                {"source": "B", "target": "C", "relation_type": "sufficient"},
            ],
            "query": {"mode": "verify_chain", "source": "A", "target": "C", "max_paths": 3},
        },
        store,
    )

    assert result.isError is False
    assert result.structuredContent["result"]["reachable"] is True
    assert result.structuredContent["result"]["best_path"]["nodes"] == ["A", "B", "C"]
    assert store.list_relations() == []


@pytest.mark.asyncio
async def test_reason_over_relations_preserves_candidate_store_ids() -> None:
    store = RelationStore()
    arguments = {
        "relations": [
            {
                "source": "A",
                "target": "B",
                "relation_type": "sufficient",
                "store_id": "external",
            }
        ],
        "query": {"mode": "classify", "source": "A", "target": "B"},
    }

    matched = await call_tool(
        REASON_OVER_RELATIONS,
        {**arguments, "context_filter": {"store_id": "external"}},
        store,
    )
    filtered_out = await call_tool(
        REASON_OVER_RELATIONS,
        {**arguments, "context_filter": {"store_id": "default"}},
        store,
    )

    assert matched.structuredContent["result"]["classification"] == "sufficient"
    assert filtered_out.structuredContent["result"]["classification"] == "unknown"
    assert store.list_relations() == []


@pytest.mark.asyncio
async def test_reason_over_relations_exclusive_behavior_matches_persistent_graph() -> None:
    ephemeral_store = RelationStore()
    persistent_store = RelationStore()
    relations = [
        {"source": "A", "target": "B", "relation_type": "sufficient"},
        {"source": "A", "target": "C", "relation_type": "sufficient"},
    ]
    groups = [{"group_id": "state", "members": ["B", "C"]}]

    ephemeral = await call_tool(
        REASON_OVER_RELATIONS,
        {
            "relations": relations,
            "exclusive_groups": groups,
            "query": {"mode": "check_contradictions"},
        },
        ephemeral_store,
    )
    await call_tool(ASSERT_EXCLUSIVE, {"groups": groups}, persistent_store)
    await call_tool(
        ASSERT_RELATIONS,
        {"relations": relations, "check_contradictions": False},
        persistent_store,
    )
    persistent = await call_tool(CHECK_CONTRADICTIONS, {}, persistent_store)

    assert ephemeral.isError is False
    assert ephemeral.structuredContent["result"]["has_contradictions"] is True
    ephemeral_contradiction = ephemeral.structuredContent["result"]["contradictions"][0]
    persistent_contradiction = persistent.structuredContent["contradictions"][0]
    for key in ("type", "severity", "source", "targets", "exclusive_group_id"):
        assert ephemeral_contradiction[key] == persistent_contradiction[key]
    assert ephemeral_store.list_relations() == []


@pytest.mark.asyncio
async def test_reason_over_relations_rejects_persist_true_and_unknown_fields() -> None:
    store = RelationStore()

    persist_true = await call_tool(
        REASON_OVER_RELATIONS,
        {
            "relations": [],
            "query": {"mode": "summarize_graph"},
            "persist": True,
        },
        store,
    )
    unknown_field = await call_tool(
        REASON_OVER_RELATIONS,
        {
            "relations": [],
            "query": {"mode": "classify", "source": "A", "target": "B", "unexpected": True},
        },
        store,
    )

    assert persist_true.isError is True
    assert persist_true.structuredContent["status"] == "error"
    assert unknown_field.isError is True
    assert unknown_field.structuredContent["status"] == "error"


@pytest.mark.asyncio
async def test_reason_over_relations_counterfactual_and_summary_use_ephemeral_graph() -> None:
    store = RelationStore()
    relations = [
        {
            "source": "降价",
            "target": "利润减少",
            "relation_type": "sufficient",
            "provenance": {"chunk_id": "chunk-1"},
        }
    ]

    counterfactual = await call_tool(
        REASON_OVER_RELATIONS,
        {
            "relations": [{**relations[0], "relation_type": "necessary"}],
            "query": {"mode": "counterfactual", "if_not": "降价", "targets": ["利润减少"]},
        },
        store,
    )
    summary = await call_tool(
        REASON_OVER_RELATIONS,
        {
            "relations": relations,
            "query": {"mode": "summarize_graph", "focus_terms": ["利润"]},
        },
        store,
    )

    assert counterfactual.isError is False
    assert (
        counterfactual.structuredContent["result"]["necessarily_blocked"][0]["target"] == "利润减少"
    )
    assert summary.structuredContent["result"]["relation_count_included"] == 1
    assert "降价 sufficient 利润减少" in summary.structuredContent["result"]["summary"]
    assert store.list_relations() == []


@pytest.mark.asyncio
async def test_reason_over_relations_uses_ephemeral_proposition_aliases() -> None:
    store = RelationStore()

    result = await call_tool(
        REASON_OVER_RELATIONS,
        {
            "propositions": [
                {"id": "profit_up", "label": "Profit increases", "aliases": ["利润增加"]},
                {
                    "id": "profit_not_up",
                    "label": "Profit does not increase",
                    "aliases": ["利润未增加"],
                    "negates": "profit_up",
                },
            ],
            "relations": [
                {"source": "Discount", "target": "利润增加", "relation_type": "sufficient"},
                {"source": "Discount", "target": "利润未增加", "relation_type": "sufficient"},
            ],
            "query": {"mode": "check_contradictions", "include_soft": False},
        },
        store,
    )

    assert result.isError is False
    assert result.structuredContent["result"]["has_contradictions"] is True
    assert result.structuredContent["result"]["contradictions"][0]["targets"] == [
        "profit_up",
        "profit_not_up",
    ]
