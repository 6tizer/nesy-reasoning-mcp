import pytest

from nesy_reasoning_mcp.store import RelationStore
from nesy_reasoning_mcp.tools import (
    ASSERT_RELATIONS,
    COUNTERFACTUAL,
    LOAD_RELATIONS,
    call_tool,
)


@pytest.mark.asyncio
async def test_counterfactual_necessary_relation_blocks_target() -> None:
    store = RelationStore()
    await call_tool(
        ASSERT_RELATIONS,
        {
            "relations": [{"source": "A", "target": "B", "relation_type": "necessary"}],
            "check_contradictions": False,
        },
        store,
    )

    result = await call_tool(COUNTERFACTUAL, {"if_not": "A", "targets": ["B"]}, store)

    assert result.isError is False
    assert result.structuredContent["necessarily_blocked"][0]["target"] == "B"
    assert result.structuredContent["necessarily_blocked"][0]["proof"]["path"] == ["B", "A"]


@pytest.mark.asyncio
async def test_counterfactual_sufficient_path_is_only_possibly_blocked_open_world() -> None:
    store = RelationStore()
    await call_tool(
        ASSERT_RELATIONS,
        {
            "relations": [{"source": "A", "target": "B", "relation_type": "sufficient"}],
            "check_contradictions": False,
        },
        store,
    )

    result = await call_tool(COUNTERFACTUAL, {"if_not": "A", "targets": ["B"]}, store)

    assert result.structuredContent["necessarily_blocked"] == []
    assert result.structuredContent["possibly_blocked"][0]["target"] == "B"
    assert result.structuredContent["possibly_blocked"][0]["blocked_path"] == ["A", "B"]
    assert result.structuredContent["diagnostics"][0]["code"] == "OPEN_WORLD_DEFAULT"


@pytest.mark.asyncio
async def test_counterfactual_min_confidence_filters_blocked_path() -> None:
    store = RelationStore()
    await call_tool(
        ASSERT_RELATIONS,
        {
            "relations": [
                {"source": "A", "target": "B", "relation_type": "sufficient", "confidence": 0.2}
            ],
            "check_contradictions": False,
        },
        store,
    )

    result = await call_tool(
        COUNTERFACTUAL,
        {"if_not": "A", "targets": ["B"], "min_confidence": 0.5},
        store,
    )

    assert result.structuredContent["possibly_blocked"] == []
    assert result.structuredContent["unknown"][0]["target"] == "B"


@pytest.mark.asyncio
async def test_counterfactual_independent_alternative_is_still_possible() -> None:
    store = RelationStore()
    await call_tool(
        ASSERT_RELATIONS,
        {
            "relations": [
                {"source": "A", "target": "B", "relation_type": "sufficient"},
                {
                    "source": "C",
                    "target": "B",
                    "relation_type": "sufficient",
                    "metadata": {"independent_of": ["A"]},
                },
            ],
            "check_contradictions": False,
        },
        store,
    )

    result = await call_tool(COUNTERFACTUAL, {"if_not": "A", "targets": ["B"]}, store)

    assert result.structuredContent["possibly_blocked"] == []
    assert result.structuredContent["still_possible"][0]["target"] == "B"
    assert (
        result.structuredContent["still_possible"][0]["alternative_paths"][0][
            "independence_from_if_not"
        ]
        == "proven"
    )


@pytest.mark.asyncio
async def test_counterfactual_id_backed_metadata_independence_is_still_possible() -> None:
    store = RelationStore()
    await call_tool(
        ASSERT_RELATIONS,
        {
            "relations": [
                {
                    "source": "Price decrease",
                    "source_id": "price_down",
                    "target": "Sales increase",
                    "target_id": "sales_up",
                    "relation_type": "sufficient",
                },
                {
                    "source": "Channel expansion",
                    "source_id": "channel_expansion",
                    "target": "Sales increase",
                    "target_id": "sales_up",
                    "relation_type": "sufficient",
                    "metadata": {"independent_of": ["price_down"]},
                },
            ],
            "check_contradictions": False,
        },
        store,
    )

    result = await call_tool(
        COUNTERFACTUAL,
        {"if_not": "price_down", "targets": ["sales_up"]},
        store,
    )

    alternative = result.structuredContent["still_possible"][0]["alternative_paths"][0]
    assert result.structuredContent["possibly_blocked"] == []
    assert result.structuredContent["still_possible"][0]["target"] == "sales_up"
    assert alternative["nodes"] == ["channel_expansion", "sales_up"]
    assert alternative["independence_from_if_not"] == "proven"


@pytest.mark.asyncio
async def test_counterfactual_formal_independence_record_is_still_possible() -> None:
    store = RelationStore()
    await call_tool(
        LOAD_RELATIONS,
        {
            "source_type": "inline",
            "data": {
                "relations": [
                    {"source": "A", "target": "B", "relation_type": "sufficient"},
                    {"source": "C", "target": "B", "relation_type": "sufficient"},
                ],
                "independence_records": [{"id": "ind_c_a", "left": "C", "right": "A"}],
            },
            "check_contradictions": False,
        },
        store,
    )

    result = await call_tool(COUNTERFACTUAL, {"if_not": "A", "targets": ["B"]}, store)

    assert result.structuredContent["possibly_blocked"] == []
    assert result.structuredContent["still_possible"][0]["target"] == "B"
    assert (
        result.structuredContent["still_possible"][0]["alternative_paths"][0][
            "independence_from_if_not"
        ]
        == "proven"
    )


@pytest.mark.asyncio
async def test_counterfactual_unknown_is_not_derivably_affected() -> None:
    store = RelationStore()
    await call_tool(
        ASSERT_RELATIONS,
        {
            "relations": [{"source": "A", "target": "B", "relation_type": "sufficient"}],
            "check_contradictions": False,
        },
        store,
    )

    result = await call_tool(COUNTERFACTUAL, {"if_not": "A", "targets": ["Z"]}, store)

    assert result.structuredContent["unknown"][0]["target"] == "Z"
    assert result.structuredContent["not_derivably_affected"] == ["Z"]


@pytest.mark.asyncio
async def test_counterfactual_closed_world_requires_causal_completeness() -> None:
    store = RelationStore()
    await call_tool(
        ASSERT_RELATIONS,
        {
            "relations": [{"source": "A", "target": "B", "relation_type": "sufficient"}],
            "check_contradictions": False,
        },
        store,
    )

    result = await call_tool(
        COUNTERFACTUAL,
        {"if_not": "A", "targets": ["B"], "world_mode": "closed"},
        store,
    )

    assert result.structuredContent["status"] == "warning"
    assert result.structuredContent["possibly_blocked"][0]["target"] == "B"
    assert result.structuredContent["diagnostics"][0]["code"] == (
        "CLOSED_WORLD_COMPLETENESS_NOT_DECLARED"
    )


@pytest.mark.asyncio
async def test_counterfactual_closed_world_upgrades_when_all_causes_blocked() -> None:
    store = RelationStore()
    await call_tool(
        LOAD_RELATIONS,
        {
            "source_type": "inline",
            "data": {
                "relations": [
                    {
                        "source": "A",
                        "target": "B",
                        "relation_type": "sufficient",
                        "context_id": "ctx",
                    }
                ],
                "context_metadata": {"ctx": {"causal_completeness": True}},
            },
            "check_contradictions": False,
        },
        store,
    )

    result = await call_tool(
        COUNTERFACTUAL,
        {
            "if_not": "A",
            "targets": ["B"],
            "world_mode": "closed",
            "context_filter": {"context_id": "ctx"},
        },
        store,
    )

    assert result.structuredContent["possibly_blocked"] == []
    assert result.structuredContent["necessarily_blocked"][0]["target"] == "B"
    assert result.structuredContent["necessarily_blocked"][0]["proof"]["type"] == (
        "closed_world_all_causes_blocked"
    )


@pytest.mark.asyncio
async def test_counterfactual_invalid_input_returns_error() -> None:
    result = await call_tool(COUNTERFACTUAL, {"if_not": "A", "unexpected": True}, RelationStore())

    assert result.isError is True
    assert result.structuredContent["status"] == "error"
