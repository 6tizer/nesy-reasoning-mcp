import pytest

from nesy_reasoning_mcp.store import RelationStore
from nesy_reasoning_mcp.tools import (
    ASSERT_EXCLUSIVE,
    ASSERT_RELATIONS,
    CLASSIFY,
    LOAD_RELATIONS,
    VERIFY_CHAIN,
    call_tool,
)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("relations", "source", "target", "expected"),
    [
        ([{"source": "A", "target": "B", "relation_type": "sufficient"}], "A", "B", "sufficient"),
        ([{"source": "A", "target": "B", "relation_type": "necessary"}], "A", "B", "necessary"),
        ([{"source": "A", "target": "B", "relation_type": "equivalent"}], "A", "B", "equivalent"),
        (
            [
                {"source": "A", "target": "B", "relation_type": "sufficient"},
                {"source": "B", "target": "C", "relation_type": "sufficient"},
            ],
            "A",
            "C",
            "sufficient",
        ),
        (
            [
                {"source": "A", "target": "B", "relation_type": "necessary"},
                {"source": "B", "target": "C", "relation_type": "necessary"},
            ],
            "A",
            "C",
            "necessary",
        ),
        (
            [
                {"source": "A", "target": "B", "relation_type": "equivalent"},
                {"source": "B", "target": "C", "relation_type": "sufficient"},
            ],
            "A",
            "C",
            "sufficient",
        ),
        (
            [
                {"source": "A", "target": "B", "relation_type": "sufficient"},
                {"source": "B", "target": "C", "relation_type": "necessary"},
            ],
            "A",
            "C",
            "unknown",
        ),
        (
            [
                {"source": "A", "target": "B", "relation_type": "necessary"},
                {"source": "B", "target": "C", "relation_type": "sufficient"},
            ],
            "A",
            "C",
            "unknown",
        ),
    ],
)
async def test_classify_spec_matrix(relations, source, target, expected) -> None:
    store = RelationStore()
    await call_tool(
        ASSERT_RELATIONS,
        {"relations": relations, "check_contradictions": False},
        store,
    )

    result = await call_tool(CLASSIFY, {"source": source, "target": target}, store)

    assert result.isError is False
    assert result.structuredContent["classification"] == expected


@pytest.mark.asyncio
async def test_alternative_sufficient_cause_does_not_prove_not_necessary() -> None:
    store = RelationStore()
    await call_tool(
        ASSERT_RELATIONS,
        {
            "relations": [
                {"source": "A", "target": "B", "relation_type": "sufficient"},
                {"source": "C", "target": "B", "relation_type": "sufficient"},
            ],
            "check_contradictions": False,
        },
        store,
    )

    result = await call_tool(CLASSIFY, {"source": "A", "target": "B"}, store)

    assert result.structuredContent["classification"] == "sufficient"
    assert result.structuredContent["necessity_status"]["status"] == "unknown"


@pytest.mark.asyncio
async def test_classify_returns_contradictory_for_exclusive_target_conflict() -> None:
    store = RelationStore()
    await call_tool(
        ASSERT_EXCLUSIVE,
        {"groups": [{"group_id": "state", "members": ["B", "C"]}]},
        store,
    )
    await call_tool(
        ASSERT_RELATIONS,
        {
            "relations": [
                {"source": "A", "target": "B", "relation_type": "sufficient"},
                {"source": "A", "target": "C", "relation_type": "sufficient"},
            ],
            "check_contradictions": False,
        },
        store,
    )

    result = await call_tool(CLASSIFY, {"source": "A", "target": "B"}, store)

    assert result.structuredContent["classification"] == "contradictory"
    assert result.structuredContent["source_implies_target"]["proven"] is True
    assert result.structuredContent["diagnostics"][0]["code"] == "CONTRADICTORY_CLASSIFICATION"


@pytest.mark.asyncio
async def test_classify_contradictory_respects_context_scope() -> None:
    store = RelationStore()
    await call_tool(
        ASSERT_EXCLUSIVE,
        {"groups": [{"group_id": "state", "members": ["B", "C"], "context_id": "ctx1"}]},
        store,
    )
    await call_tool(
        ASSERT_RELATIONS,
        {
            "relations": [
                {"source": "A", "target": "B", "relation_type": "sufficient", "context_id": "ctx1"},
                {"source": "A", "target": "C", "relation_type": "sufficient", "context_id": "ctx2"},
            ],
            "check_contradictions": False,
        },
        store,
    )

    result = await call_tool(CLASSIFY, {"source": "A", "target": "B"}, store)

    assert result.structuredContent["classification"] == "sufficient"
    assert result.structuredContent["diagnostics"] == []


@pytest.mark.asyncio
async def test_classify_contradictory_respects_require_direct() -> None:
    store = RelationStore()
    await call_tool(
        ASSERT_EXCLUSIVE,
        {"groups": [{"group_id": "state", "members": ["B", "C"]}]},
        store,
    )
    await call_tool(
        ASSERT_RELATIONS,
        {
            "relations": [
                {"source": "A", "target": "B", "relation_type": "sufficient"},
                {"source": "A", "target": "X", "relation_type": "sufficient"},
                {"source": "X", "target": "C", "relation_type": "sufficient"},
            ],
            "check_contradictions": False,
        },
        store,
    )

    direct = await call_tool(
        CLASSIFY,
        {"source": "A", "target": "B", "require_direct": True},
        store,
    )
    transitive = await call_tool(CLASSIFY, {"source": "A", "target": "B"}, store)

    assert direct.structuredContent["classification"] == "sufficient"
    assert transitive.structuredContent["classification"] == "contradictory"


@pytest.mark.asyncio
async def test_classify_independent_counterexample_proves_not_necessary() -> None:
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

    result = await call_tool(CLASSIFY, {"source": "A", "target": "B"}, store)

    assert result.structuredContent["classification"] == "sufficient"
    assert result.structuredContent["necessity_status"]["status"] == "proven_not_necessary"
    assert result.structuredContent["necessity_status"]["counterexample"] == "C"


@pytest.mark.asyncio
async def test_cycle_search_does_not_loop() -> None:
    store = RelationStore()
    await call_tool(
        ASSERT_RELATIONS,
        {
            "relations": [
                {"source": "A", "target": "B", "relation_type": "sufficient"},
                {"source": "B", "target": "C", "relation_type": "sufficient"},
                {"source": "C", "target": "A", "relation_type": "sufficient"},
            ],
            "check_contradictions": False,
        },
        store,
    )

    result = await call_tool(CLASSIFY, {"source": "A", "target": "C", "max_depth": 3}, store)

    assert result.structuredContent["classification"] == "equivalent"
    assert result.structuredContent["source_implies_target"]["best_path"] == ["A", "B", "C"]


@pytest.mark.asyncio
async def test_verify_chain_best_confidence_path() -> None:
    store = RelationStore()
    await call_tool(
        ASSERT_RELATIONS,
        {
            "relations": [
                {"source": "A", "target": "B", "relation_type": "sufficient", "confidence": 0.5},
                {"source": "B", "target": "C", "relation_type": "sufficient", "confidence": 0.5},
                {"source": "A", "target": "D", "relation_type": "sufficient", "confidence": 0.9},
                {"source": "D", "target": "C", "relation_type": "sufficient", "confidence": 0.9},
            ],
            "check_contradictions": False,
        },
        store,
    )

    result = await call_tool(VERIFY_CHAIN, {"source": "A", "target": "C"}, store)

    assert result.isError is False
    assert result.structuredContent["reachable"] is True
    assert result.structuredContent["relation_established"] is True
    assert result.structuredContent["source_to_target_reachable"] is True
    assert result.structuredContent["target_to_source_reachable"] is False
    assert result.structuredContent["best_path"]["nodes"] == ["A", "D", "C"]
    assert result.structuredContent["best_path"]["evidence_confidence"] == pytest.approx(0.81)


@pytest.mark.asyncio
async def test_verify_chain_distinguishes_reverse_only_reachability() -> None:
    store = RelationStore()
    await call_tool(
        ASSERT_RELATIONS,
        {
            "relations": [{"source": "C", "target": "A", "relation_type": "sufficient"}],
            "check_contradictions": False,
        },
        store,
    )

    result = await call_tool(VERIFY_CHAIN, {"source": "A", "target": "C"}, store)

    assert result.structuredContent["reachable"] is True
    assert result.structuredContent["relation_established"] is True
    assert result.structuredContent["source_to_target_reachable"] is False
    assert result.structuredContent["target_to_source_reachable"] is True
    assert result.structuredContent["relation_type"] == "necessary"


@pytest.mark.asyncio
async def test_verify_explicit_chain_broken_direction_mismatch() -> None:
    store = RelationStore()
    await call_tool(
        ASSERT_RELATIONS,
        {
            "relations": [
                {"source": "A", "target": "B", "relation_type": "sufficient"},
                {"source": "B", "target": "C", "relation_type": "necessary"},
            ],
            "check_contradictions": False,
        },
        store,
    )

    result = await call_tool(
        VERIFY_CHAIN,
        {"source": "A", "target": "C", "chain": ["A", "B", "C"]},
        store,
    )

    assert result.structuredContent["reachable"] is False
    assert result.structuredContent["relation_established"] is False
    assert result.structuredContent["source_to_target_reachable"] is False
    assert result.structuredContent["target_to_source_reachable"] is False
    assert result.structuredContent["broken_at"]["index"] == 1
    assert result.structuredContent["diagnostics"][0]["code"] == "DIRECTION_MISMATCH"


@pytest.mark.asyncio
async def test_verify_expected_relation_mismatch_warns() -> None:
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
        VERIFY_CHAIN,
        {"source": "A", "target": "B", "expected_relation": "necessary"},
        store,
    )

    assert result.structuredContent["reachable"] is True
    assert result.structuredContent["relation_established"] is True
    assert result.structuredContent["source_to_target_reachable"] is True
    assert result.structuredContent["target_to_source_reachable"] is False
    assert result.structuredContent["logic_validity"] is False
    assert result.structuredContent["diagnostics"][0]["code"] == "EXPECTED_RELATION_MISMATCH"


@pytest.mark.asyncio
async def test_classify_context_filter_assumptions_and_time() -> None:
    store = RelationStore()
    await call_tool(
        ASSERT_RELATIONS,
        {
            "relations": [
                {
                    "source": "A",
                    "target": "B",
                    "relation_type": "sufficient",
                    "context_id": "ctx",
                    "assumptions": ["same_market", "no_stockout"],
                    "temporal": {"valid_from": "2026-01-01", "valid_to": "2026-12-31"},
                },
                {
                    "source": "B",
                    "target": "C",
                    "relation_type": "sufficient",
                    "context_id": "other",
                },
            ],
            "check_contradictions": False,
        },
        store,
    )

    result = await call_tool(
        CLASSIFY,
        {
            "source": "A",
            "target": "B",
            "context_filter": {
                "context_id": "ctx",
                "assumptions": ["same_market"],
                "valid_at": "2026-06-01T00:00:00Z",
            },
        },
        store,
    )

    assert result.structuredContent["classification"] == "sufficient"
    assert result.structuredContent["graph_stats"]["relations"] == 1
