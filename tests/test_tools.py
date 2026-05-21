import json

import pytest

from nesy_reasoning_mcp.store import RelationStore
from nesy_reasoning_mcp.tools import (
    ASSERT_EXCLUSIVE,
    ASSERT_RELATIONS,
    CHECK_CONTRADICTIONS,
    CLASSIFY,
    CLEAR_RELATIONS,
    LIST_RELATIONS,
    VERIFY_CHAIN,
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
    assert result.structuredContent["best_path"]["nodes"] == ["A", "D", "C"]
    assert result.structuredContent["best_path"]["evidence_confidence"] == pytest.approx(0.81)


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


@pytest.mark.asyncio
async def test_assert_exclusive_and_check_direct_contradiction() -> None:
    store = RelationStore()
    group = await call_tool(
        ASSERT_EXCLUSIVE,
        {
            "groups": [
                {
                    "group_id": "profit_state",
                    "members": ["利润增加", "利润减少", "利润不变"],
                    "context_id": "ecommerce_q3",
                }
            ]
        },
        store,
    )
    assert group.isError is False
    assert group.structuredContent["added_groups"] == 1

    await call_tool(
        ASSERT_RELATIONS,
        {
            "relations": [
                {
                    "source": "降价",
                    "target": "利润增加",
                    "relation_type": "sufficient",
                    "context_id": "ecommerce_q3",
                },
                {
                    "source": "降价",
                    "target": "利润减少",
                    "relation_type": "sufficient",
                    "context_id": "ecommerce_q3",
                },
            ],
            "check_contradictions": False,
        },
        store,
    )

    result = await call_tool(CHECK_CONTRADICTIONS, {}, store)

    assert result.isError is False
    assert result.structuredContent["status"] == "warning"
    assert result.structuredContent["has_contradictions"] is True
    contradiction = result.structuredContent["contradictions"][0]
    assert contradiction["type"] == "exclusive_targets"
    assert contradiction["severity"] == "hard"
    assert contradiction["source"] == "降价"
    assert contradiction["targets"] == ["利润增加", "利润减少"]


@pytest.mark.asyncio
async def test_check_transitive_exclusive_contradiction() -> None:
    store = RelationStore()
    await call_tool(
        ASSERT_EXCLUSIVE,
        {"groups": [{"group_id": "state", "members": ["B", "D"]}]},
        store,
    )
    await call_tool(
        ASSERT_RELATIONS,
        {
            "relations": [
                {"source": "A", "target": "B", "relation_type": "sufficient"},
                {"source": "A", "target": "C", "relation_type": "sufficient"},
                {"source": "C", "target": "D", "relation_type": "sufficient"},
            ],
            "check_contradictions": False,
        },
        store,
    )

    result = await call_tool(CHECK_CONTRADICTIONS, {"max_depth": 3}, store)

    assert result.structuredContent["has_contradictions"] is True
    assert result.structuredContent["contradictions"][0]["type"] == "transitive_exclusive_targets"


@pytest.mark.asyncio
async def test_different_contexts_are_context_separated_not_hard() -> None:
    store = RelationStore()
    await call_tool(
        ASSERT_EXCLUSIVE,
        {
            "groups": [
                {
                    "group_id": "profit_state_q3",
                    "members": ["利润增加", "利润减少"],
                    "context_id": "q3",
                },
                {
                    "group_id": "profit_state_q4",
                    "members": ["利润增加", "利润减少"],
                    "context_id": "q4",
                },
            ]
        },
        store,
    )
    await call_tool(
        ASSERT_RELATIONS,
        {
            "relations": [
                {
                    "source": "降价",
                    "target": "利润增加",
                    "relation_type": "sufficient",
                    "context_id": "q3",
                },
                {
                    "source": "降价",
                    "target": "利润减少",
                    "relation_type": "sufficient",
                    "context_id": "q4",
                },
            ],
            "check_contradictions": False,
        },
        store,
    )

    result = await call_tool(CHECK_CONTRADICTIONS, {}, store)

    assert result.structuredContent["has_contradictions"] is False
    assert result.structuredContent["context_separated"]
    assert result.structuredContent["context_separated"][0]["type"] == "context_separated_conflict"


@pytest.mark.asyncio
async def test_no_exclusive_group_means_no_semantic_contradiction() -> None:
    store = RelationStore()
    await call_tool(
        ASSERT_RELATIONS,
        {
            "relations": [
                {"source": "降价", "target": "利润增加", "relation_type": "sufficient"},
                {"source": "降价", "target": "利润减少", "relation_type": "sufficient"},
            ],
            "check_contradictions": False,
        },
        store,
    )

    result = await call_tool(CHECK_CONTRADICTIONS, {}, store)

    assert result.structuredContent["has_contradictions"] is False
    assert result.structuredContent["contradictions"] == []


@pytest.mark.asyncio
async def test_facts_mode_does_not_persist_input_facts() -> None:
    store = RelationStore()
    await call_tool(
        ASSERT_EXCLUSIVE,
        {"groups": [{"group_id": "state", "members": ["B", "C"]}]},
        store,
    )

    result = await call_tool(
        CHECK_CONTRADICTIONS,
        {
            "mode": "facts",
            "facts": [
                {"source": "A", "target": "B", "relation_type": "sufficient"},
                {"source": "A", "target": "C", "relation_type": "sufficient"},
            ],
        },
        store,
    )

    assert result.structuredContent["has_contradictions"] is True
    assert result.structuredContent["total_facts_count"] == 2
    assert store.list_relations() == []


@pytest.mark.asyncio
async def test_non_overlapping_temporal_windows_do_not_create_hard_contradiction() -> None:
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
                {
                    "source": "A",
                    "target": "B",
                    "relation_type": "sufficient",
                    "temporal": {"valid_from": "2026-01-01", "valid_to": "2026-01-31"},
                },
                {
                    "source": "A",
                    "target": "C",
                    "relation_type": "sufficient",
                    "temporal": {"valid_from": "2026-02-01", "valid_to": "2026-02-28"},
                },
            ],
            "check_contradictions": False,
        },
        store,
    )

    result = await call_tool(CHECK_CONTRADICTIONS, {}, store)

    assert result.structuredContent["has_contradictions"] is False
    assert result.structuredContent["contradictions"] == []


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


@pytest.mark.asyncio
async def test_clean_facts_count_counts_only_non_conflicting_input_facts() -> None:
    store = RelationStore()
    await call_tool(
        ASSERT_EXCLUSIVE,
        {"groups": [{"group_id": "state", "members": ["B", "C"]}]},
        store,
    )

    result = await call_tool(
        CHECK_CONTRADICTIONS,
        {
            "mode": "facts",
            "facts": [
                {"source": "A", "target": "B", "relation_type": "sufficient"},
                {"source": "A", "target": "C", "relation_type": "sufficient"},
                {"source": "X", "target": "Y", "relation_type": "sufficient"},
            ],
        },
        store,
    )

    assert result.structuredContent["has_contradictions"] is True
    assert result.structuredContent["total_facts_count"] == 3
    assert result.structuredContent["clean_facts_count"] == 1


@pytest.mark.asyncio
async def test_assert_relations_reports_warning_when_relation_creates_contradiction() -> None:
    store = RelationStore()
    await call_tool(
        ASSERT_EXCLUSIVE,
        {"groups": [{"group_id": "state", "members": ["B", "C"]}]},
        store,
    )
    await call_tool(
        ASSERT_RELATIONS,
        {
            "relations": [{"source": "A", "target": "B", "relation_type": "sufficient"}],
            "check_contradictions": False,
        },
        store,
    )

    result = await call_tool(
        ASSERT_RELATIONS,
        {"relations": [{"source": "A", "target": "C", "relation_type": "sufficient"}]},
        store,
    )

    assert result.structuredContent["status"] == "warning"
    assert result.structuredContent["contradictions"]
