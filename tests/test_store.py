from nesy_reasoning_mcp.schemas import RelationInput, RelationType
from nesy_reasoning_mcp.store import RelationStore


def test_defaults_and_sufficient_edge() -> None:
    store = RelationStore()
    records, updated = store.assert_relations(
        [
            RelationInput(
                source="A",
                target="B",
                relation_type=RelationType.SUFFICIENT,
            )
        ]
    )

    assert updated == 0
    assert len(records) == 1
    record = records[0]
    assert record.id.startswith("rel_")
    assert record.confidence == 1.0
    assert record.context_id == "default"
    assert record.store_id == "default"
    assert record.polarity == "positive"

    edges = store.implication_edges()
    assert len(edges) == 1
    assert edges[0].antecedent == "A"
    assert edges[0].consequent == "B"


def test_necessary_generates_reverse_edge() -> None:
    store = RelationStore()
    store.assert_relations(
        [RelationInput(source="A", target="B", relation_type=RelationType.NECESSARY)]
    )

    edge = store.implication_edges()[0]
    assert edge.antecedent == "B"
    assert edge.consequent == "A"


def test_equivalent_generates_two_edges() -> None:
    store = RelationStore()
    store.assert_relations(
        [RelationInput(source="A", target="B", relation_type=RelationType.EQUIVALENT)]
    )

    edges = store.implication_edges()
    assert [(edge.antecedent, edge.consequent) for edge in edges] == [("A", "B"), ("B", "A")]


def test_dry_run_does_not_change_store() -> None:
    store = RelationStore()
    records, updated = store.assert_relations(
        [RelationInput(source="A", target="B", relation_type=RelationType.SUFFICIENT)],
        dry_run=True,
    )

    assert len(records) == 1
    assert updated == 0
    assert store.list_relations() == []


def test_replace_same_pair_only_matches_pair_context_store() -> None:
    store = RelationStore()
    store.assert_relations(
        [
            RelationInput(
                source="A",
                target="B",
                relation_type=RelationType.SUFFICIENT,
                context_id="ctx1",
            ),
            RelationInput(
                source="A",
                target="B",
                relation_type=RelationType.SUFFICIENT,
                context_id="ctx2",
            ),
        ]
    )

    records, updated = store.assert_relations(
        [
            RelationInput(
                source="A",
                target="B",
                relation_type=RelationType.NECESSARY,
                context_id="ctx1",
            )
        ],
        mode="replace_same_pair",
    )

    assert len(records) == 1
    assert updated == 1
    listed = store.list_relations()
    assert len(listed) == 2
    assert {(item.context_id, item.relation_type) for item in listed} == {
        ("ctx1", RelationType.NECESSARY),
        ("ctx2", RelationType.SUFFICIENT),
    }
