import pytest
from pydantic import ValidationError

from nesy_reasoning_mcp.schemas import CheckContradictionsInput, PropositionRecord
from nesy_reasoning_mcp.store import RelationStore
from nesy_reasoning_mcp.tools import (
    ASSERT_EXCLUSIVE,
    ASSERT_RELATIONS,
    CHECK_CONTRADICTIONS,
    LOAD_RELATIONS,
    call_tool,
)


def test_check_contradictions_accepts_canonical_negation_propositions() -> None:
    payload = CheckContradictionsInput.model_validate(
        {
            "propositions": [
                {
                    "id": " profit_not_up ",
                    "label": " Profit does not increase ",
                    "negates": " profit_up ",
                }
            ]
        }
    )

    proposition = payload.propositions[0]
    assert proposition.id == "profit_not_up"
    assert proposition.label == "Profit does not increase"
    assert proposition.negates == "profit_up"


def test_proposition_negates_rejects_empty_and_self_reference() -> None:
    with pytest.raises(ValidationError):
        PropositionRecord(id="profit_not_up", label="Profit does not increase", negates=" ")

    with pytest.raises(ValidationError):
        PropositionRecord(id="profit_up", label="Profit increases", negates="profit_up")


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
async def test_check_contradictions_min_confidence_filters_low_confidence_path() -> None:
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
                {"source": "A", "target": "B", "relation_type": "sufficient", "confidence": 0.2},
                {"source": "A", "target": "C", "relation_type": "sufficient", "confidence": 0.9},
            ],
            "check_contradictions": False,
        },
        store,
    )

    result = await call_tool(CHECK_CONTRADICTIONS, {"min_confidence": 0.5}, store)

    assert result.structuredContent["has_contradictions"] is False
    assert result.structuredContent["contradictions"] == []


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
@pytest.mark.parametrize("negated_target", ["not B", "not:B", "¬B"])
async def test_direct_opposition_detects_explicit_negation_forms(negated_target: str) -> None:
    store = RelationStore()
    await call_tool(
        ASSERT_RELATIONS,
        {
            "relations": [
                {"source": "A", "target": "B", "relation_type": "sufficient"},
                {"source": "A", "target": negated_target, "relation_type": "sufficient"},
            ],
            "check_contradictions": False,
        },
        store,
    )

    result = await call_tool(CHECK_CONTRADICTIONS, {"include_soft": False}, store)

    assert result.structuredContent["has_contradictions"] is True
    contradiction = result.structuredContent["contradictions"][0]
    assert contradiction["type"] == "direct_opposition"
    assert contradiction["severity"] == "hard"
    assert contradiction["targets"] == ["B", negated_target]


@pytest.mark.asyncio
async def test_direct_opposition_uses_canonical_negation_ids() -> None:
    store = RelationStore()
    await call_tool(
        ASSERT_RELATIONS,
        {
            "relations": [
                {
                    "source": "Discount",
                    "target": "Profit increases",
                    "target_id": "profit_up",
                    "relation_type": "sufficient",
                },
                {
                    "source": "Discount",
                    "target": "Profit does not increase",
                    "target_id": "profit_not_up",
                    "relation_type": "sufficient",
                },
            ],
            "check_contradictions": False,
        },
        store,
    )

    result = await call_tool(
        CHECK_CONTRADICTIONS,
        {
            "include_soft": False,
            "propositions": [
                {
                    "id": "profit_not_up",
                    "label": "Profit does not increase",
                    "negates": "profit_up",
                }
            ],
        },
        store,
    )

    assert result.structuredContent["has_contradictions"] is True
    contradiction = result.structuredContent["contradictions"][0]
    assert contradiction["type"] == "direct_opposition"
    assert contradiction["severity"] == "hard"
    assert contradiction["targets"] == ["profit_up", "profit_not_up"]


@pytest.mark.asyncio
async def test_facts_mode_uses_canonical_negation_ids() -> None:
    store = RelationStore()

    result = await call_tool(
        CHECK_CONTRADICTIONS,
        {
            "mode": "facts",
            "include_soft": False,
            "facts": [
                {
                    "source": "Discount",
                    "target": "Profit increases",
                    "target_id": "profit_up",
                    "relation_type": "sufficient",
                },
                {
                    "source": "Discount",
                    "target": "Profit does not increase",
                    "target_id": "profit_not_up",
                    "relation_type": "sufficient",
                },
            ],
            "propositions": [
                {
                    "id": "profit_not_up",
                    "label": "Profit does not increase",
                    "negates": "profit_up",
                }
            ],
        },
        store,
    )

    assert result.structuredContent["has_contradictions"] is True
    contradiction = result.structuredContent["contradictions"][0]
    assert contradiction["type"] == "direct_opposition"
    assert contradiction["severity"] == "hard"
    assert contradiction["targets"] == ["profit_up", "profit_not_up"]
    assert {item.id for item in store.list_relations()} == set()


@pytest.mark.asyncio
async def test_check_contradictions_uses_stored_proposition_registry_aliases() -> None:
    store = RelationStore()
    await call_tool(
        LOAD_RELATIONS,
        {
            "source_type": "inline",
            "data": {
                "propositions": [
                    {"id": "profit_up", "label": "Profit increases", "aliases": ["利润增加"]},
                    {
                        "id": "profit_not_up",
                        "label": "Profit does not increase",
                        "aliases": ["利润未增加"],
                        "negates": "profit_up",
                    },
                ]
            },
            "check_contradictions": False,
        },
        store,
    )

    result = await call_tool(
        CHECK_CONTRADICTIONS,
        {
            "mode": "facts",
            "include_soft": False,
            "facts": [
                {"source": "Discount", "target": "利润增加", "relation_type": "sufficient"},
                {"source": "Discount", "target": "利润未增加", "relation_type": "sufficient"},
            ],
        },
        store,
    )

    assert result.structuredContent["has_contradictions"] is True
    contradiction = result.structuredContent["contradictions"][0]
    assert contradiction["type"] == "direct_opposition"
    assert contradiction["targets"] == ["profit_up", "profit_not_up"]


@pytest.mark.asyncio
async def test_assert_relations_uses_stored_proposition_registry_aliases() -> None:
    store = RelationStore()
    await call_tool(
        LOAD_RELATIONS,
        {
            "source_type": "inline",
            "data": {
                "propositions": [
                    {"id": "profit_up", "label": "Profit increases", "aliases": ["利润增加"]},
                    {
                        "id": "profit_not_up",
                        "label": "Profit does not increase",
                        "aliases": ["利润未增加"],
                        "negates": "profit_up",
                    },
                ]
            },
            "check_contradictions": False,
        },
        store,
    )

    result = await call_tool(
        ASSERT_RELATIONS,
        {
            "relations": [
                {"source": "Discount", "target": "利润增加", "relation_type": "sufficient"},
                {"source": "Discount", "target": "利润未增加", "relation_type": "sufficient"},
            ],
        },
        store,
    )

    assert result.structuredContent["status"] == "warning"
    assert {relation.target_id for relation in store.list_relations()} == {
        "profit_up",
        "profit_not_up",
    }
    assert result.structuredContent["contradictions"][0]["type"] == "direct_opposition"


@pytest.mark.asyncio
async def test_plain_language_negation_needs_canonical_declaration() -> None:
    store = RelationStore()
    await call_tool(
        ASSERT_RELATIONS,
        {
            "relations": [
                {
                    "source": "Discount",
                    "target": "Profit increases",
                    "target_id": "profit_up",
                    "relation_type": "sufficient",
                },
                {
                    "source": "Discount",
                    "target": "Profit does not increase",
                    "target_id": "profit_not_up",
                    "relation_type": "sufficient",
                },
            ],
            "check_contradictions": False,
        },
        store,
    )

    result = await call_tool(CHECK_CONTRADICTIONS, {"include_soft": False}, store)

    assert result.structuredContent["has_contradictions"] is False
    assert result.structuredContent["contradictions"] == []


@pytest.mark.asyncio
async def test_canonical_negation_respects_min_confidence() -> None:
    store = RelationStore()
    await call_tool(
        ASSERT_RELATIONS,
        {
            "relations": [
                {
                    "source": "Discount",
                    "target": "Profit increases",
                    "target_id": "profit_up",
                    "relation_type": "sufficient",
                    "confidence": 0.2,
                },
                {
                    "source": "Discount",
                    "target": "Profit does not increase",
                    "target_id": "profit_not_up",
                    "relation_type": "sufficient",
                    "confidence": 0.9,
                },
            ],
            "check_contradictions": False,
        },
        store,
    )

    result = await call_tool(
        CHECK_CONTRADICTIONS,
        {
            "include_soft": False,
            "min_confidence": 0.5,
            "propositions": [
                {
                    "id": "profit_not_up",
                    "label": "Profit does not increase",
                    "negates": "profit_up",
                }
            ],
        },
        store,
    )

    assert result.structuredContent["has_contradictions"] is False
    assert result.structuredContent["contradictions"] == []


@pytest.mark.asyncio
async def test_cycle_to_exclusion_detects_path_to_own_negation() -> None:
    store = RelationStore()
    await call_tool(
        ASSERT_RELATIONS,
        {
            "relations": [
                {"source": "A", "target": "B", "relation_type": "sufficient"},
                {"source": "B", "target": "not A", "relation_type": "sufficient"},
            ],
            "check_contradictions": False,
        },
        store,
    )

    result = await call_tool(CHECK_CONTRADICTIONS, {"max_depth": 2}, store)

    assert result.structuredContent["has_contradictions"] is True
    contradiction = result.structuredContent["contradictions"][0]
    assert contradiction["type"] == "cycle_to_exclusion"
    assert contradiction["severity"] == "hard"
    assert contradiction["path"] == ["A", "B", "not A"]


@pytest.mark.asyncio
async def test_cycle_to_exclusion_uses_canonical_negation_ids() -> None:
    store = RelationStore()
    await call_tool(
        ASSERT_RELATIONS,
        {
            "relations": [
                {
                    "source": "A",
                    "source_id": "a",
                    "target": "B",
                    "target_id": "b",
                    "relation_type": "sufficient",
                },
                {
                    "source": "B",
                    "source_id": "b",
                    "target": "Not A",
                    "target_id": "not_a",
                    "relation_type": "sufficient",
                },
            ],
            "check_contradictions": False,
        },
        store,
    )

    result = await call_tool(
        CHECK_CONTRADICTIONS,
        {
            "max_depth": 2,
            "propositions": [
                {
                    "id": "not_a",
                    "label": "Not A",
                    "negates": "a",
                }
            ],
        },
        store,
    )

    assert result.structuredContent["has_contradictions"] is True
    contradiction = result.structuredContent["contradictions"][0]
    assert contradiction["type"] == "cycle_to_exclusion"
    assert contradiction["severity"] == "hard"
    assert contradiction["targets"] == ["a", "not_a"]
    assert contradiction["path"] == ["a", "b", "not_a"]


@pytest.mark.asyncio
async def test_cycle_to_exclusion_respects_max_depth() -> None:
    store = RelationStore()
    await call_tool(
        ASSERT_RELATIONS,
        {
            "relations": [
                {"source": "A", "target": "B", "relation_type": "sufficient"},
                {"source": "B", "target": "not A", "relation_type": "sufficient"},
            ],
            "check_contradictions": False,
        },
        store,
    )

    result = await call_tool(CHECK_CONTRADICTIONS, {"max_depth": 1}, store)

    assert result.structuredContent["has_contradictions"] is False
    assert result.structuredContent["contradictions"] == []


@pytest.mark.asyncio
async def test_cycle_to_exclusion_ignores_temporally_disjoint_path() -> None:
    store = RelationStore()
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
                    "source": "B",
                    "target": "not A",
                    "relation_type": "sufficient",
                    "temporal": {"valid_from": "2026-02-01", "valid_to": "2026-02-28"},
                },
            ],
            "check_contradictions": False,
        },
        store,
    )

    result = await call_tool(CHECK_CONTRADICTIONS, {"max_depth": 2}, store)

    assert result.structuredContent["has_contradictions"] is False
    assert result.structuredContent["contradictions"] == []


@pytest.mark.asyncio
async def test_confidence_tension_only_returns_when_soft_included() -> None:
    store = RelationStore()
    await call_tool(
        ASSERT_RELATIONS,
        {
            "relations": [
                {
                    "id": "rel_low",
                    "source": "A",
                    "target": "B",
                    "relation_type": "sufficient",
                    "confidence": 0.2,
                },
                {
                    "id": "rel_high",
                    "source": "A",
                    "target": "B",
                    "relation_type": "sufficient",
                    "confidence": 0.8,
                },
            ],
            "check_contradictions": False,
        },
        store,
    )

    hard_only = await call_tool(CHECK_CONTRADICTIONS, {"include_soft": False}, store)
    with_soft = await call_tool(CHECK_CONTRADICTIONS, {"include_soft": True}, store)

    assert hard_only.structuredContent["contradictions"] == []
    assert with_soft.structuredContent["has_contradictions"] is True
    contradiction = with_soft.structuredContent["contradictions"][0]
    assert contradiction["type"] == "confidence_tension"
    assert contradiction["severity"] == "soft"
    assert contradiction["fact_ids"] == ["rel_low", "rel_high"]


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


@pytest.mark.asyncio
async def test_assert_relations_rejects_contradiction_without_writing() -> None:
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
        {
            "relations": [{"source": "A", "target": "C", "relation_type": "sufficient"}],
            "on_contradiction": "reject",
        },
        store,
    )

    assert result.isError is True
    assert result.structuredContent["status"] == "error"
    assert result.structuredContent["rejected"] == 1
    assert result.structuredContent["relation_ids"] == []
    assert result.structuredContent["diagnostics"][0]["code"] == "CONTRADICTION_REJECTED"
    assert [relation.target for relation in store.list_relations()] == ["B"]


@pytest.mark.asyncio
async def test_assert_relations_reject_overrides_disabled_warning_check() -> None:
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
        {
            "relations": [{"source": "A", "target": "C", "relation_type": "sufficient"}],
            "check_contradictions": False,
            "on_contradiction": "reject",
        },
        store,
    )

    assert result.isError is True
    assert result.structuredContent["status"] == "error"
    assert result.structuredContent["rejected"] == 1
    assert result.structuredContent["relation_ids"] == []
    assert [relation.target for relation in store.list_relations()] == ["B"]


@pytest.mark.asyncio
async def test_assert_relations_reject_mode_preserves_upsert_update_count() -> None:
    store = RelationStore()
    await call_tool(
        ASSERT_RELATIONS,
        {
            "relations": [
                {
                    "id": "rel_keep",
                    "source": "A",
                    "target": "B",
                    "relation_type": "sufficient",
                }
            ],
            "check_contradictions": False,
        },
        store,
    )

    result = await call_tool(
        ASSERT_RELATIONS,
        {
            "relations": [
                {
                    "id": "rel_keep",
                    "source": "A",
                    "target": "C",
                    "relation_type": "necessary",
                }
            ],
            "mode": "upsert",
            "on_contradiction": "reject",
        },
        store,
    )

    assert result.structuredContent["status"] == "ok"
    assert result.structuredContent["added"] == 0
    assert result.structuredContent["updated"] == 1
    assert [(relation.id, relation.target) for relation in store.list_relations()] == [
        ("rel_keep", "C")
    ]
