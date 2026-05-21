from nesy_reasoning_mcp.config import NesyConfig, StorageConfig
from nesy_reasoning_mcp.schemas import (
    ExclusiveGroupInput,
    RelationInput,
    RelationRecord,
    RelationType,
)
from nesy_reasoning_mcp.store import (
    JsonRelationStore,
    RelationStore,
    SqliteRelationStore,
    create_relation_store,
)


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


def test_sqlite_store_persists_relations_and_exclusive_groups(tmp_path) -> None:
    config = NesyConfig(
        storage=StorageConfig(backend="sqlite", sqlite_path=str(tmp_path / "nesy.db"))
    )
    store = SqliteRelationStore(config)
    records, _updated = store.assert_relations(
        [
            RelationInput(
                id="rel_keep",
                source="A",
                target="B",
                relation_type=RelationType.SUFFICIENT,
            )
        ]
    )
    groups, _updated_groups = store.assert_exclusive(
        [ExclusiveGroupInput(group_id="state", members=["B", "C"])]
    )

    reloaded = SqliteRelationStore(config)

    assert reloaded.list_relations()[0].id == records[0].id
    assert reloaded.list_relations()[0].source == "A"
    assert reloaded.implication_edges()[0].consequent == "B"
    assert reloaded.list_exclusive_groups()[0].group_id == groups[0].group_id
    assert reloaded.list_exclusive_groups()[0].members == ["B", "C"]


def test_create_relation_store_uses_sqlite_backend(tmp_path) -> None:
    config = NesyConfig(
        storage=StorageConfig(backend="sqlite", sqlite_path=str(tmp_path / "nesy.db"))
    )

    store = create_relation_store(config)

    assert isinstance(store, SqliteRelationStore)


def test_json_store_persists_relations(tmp_path) -> None:
    config = NesyConfig(
        storage=StorageConfig(backend="json", json_path=str(tmp_path / "relations.json"))
    )
    store = JsonRelationStore(config)
    store.assert_relations(
        [RelationInput(source="A", target="B", relation_type=RelationType.SUFFICIENT)]
    )

    reloaded = JsonRelationStore(config)

    assert reloaded.list_relations()[0].source == "A"
    assert reloaded.implication_edges()[0].consequent == "B"


def test_create_relation_store_uses_json_backend(tmp_path) -> None:
    config = NesyConfig(
        storage=StorageConfig(backend="json", json_path=str(tmp_path / "relations.json"))
    )

    store = create_relation_store(config)

    assert isinstance(store, JsonRelationStore)


def test_sqlite_import_failure_rolls_back_existing_rows(tmp_path) -> None:
    config = NesyConfig(
        storage=StorageConfig(backend="sqlite", sqlite_path=str(tmp_path / "nesy.db"))
    )
    store = SqliteRelationStore(config)
    store.assert_relations(
        [
            RelationInput(
                id="rel_keep",
                source="A",
                target="B",
                relation_type=RelationType.SUFFICIENT,
            )
        ]
    )

    try:
        store.import_records(
            [
                RelationRecord(
                    id="rel_keep",
                    source="X",
                    target="Y",
                    relation_type=RelationType.SUFFICIENT,
                )
            ],
            [],
            mode="append",
            store_id="default",
        )
    except Exception:
        pass
    else:
        raise AssertionError("expected duplicate relation id to fail")

    assert [(item.id, item.target) for item in store.list_relations()] == [("rel_keep", "B")]


def test_json_store_rejects_invalid_json(tmp_path) -> None:
    path = tmp_path / "relations.json"
    path.write_text("{bad", encoding="utf-8")
    config = NesyConfig(storage=StorageConfig(backend="json", json_path=str(path)))

    try:
        JsonRelationStore(config)
    except ValueError as exc:
        assert "invalid JSON relation store" in str(exc)
    else:
        raise AssertionError("expected invalid JSON relation store error")
