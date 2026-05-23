from concurrent.futures import ThreadPoolExecutor

from nesy_reasoning_mcp.config import NesyConfig, StorageConfig
from nesy_reasoning_mcp.schemas import (
    ExclusiveGroupInput,
    IndependenceRecord,
    PropositionRecord,
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


def test_proposition_record_and_relation_ids_strip_values() -> None:
    proposition = PropositionRecord(
        id=" profit_up ",
        label=" Profit increases ",
        aliases=[" 利润增加 ", "profit rises"],
    )
    relation = RelationInput(
        source=" Profit increases ",
        source_id=" profit_up ",
        target=" Revenue increases ",
        target_id=" revenue_up ",
        relation_type=RelationType.SUFFICIENT,
    )

    assert proposition.id == "profit_up"
    assert proposition.label == "Profit increases"
    assert proposition.aliases == ["利润增加", "profit rises"]
    assert relation.source == "Profit increases"
    assert relation.source_id == "profit_up"
    assert relation.canonical_source == "profit_up"
    assert relation.canonical_target == "revenue_up"


def test_canonical_ids_drive_memory_edges_and_stats() -> None:
    store = RelationStore()
    store.assert_relations(
        [
            RelationInput(
                source="利润增加",
                source_id="profit_up",
                target="收入增加",
                target_id="revenue_up",
                relation_type=RelationType.SUFFICIENT,
            )
        ]
    )

    edge = store.implication_edges()[0]

    assert edge.antecedent == "profit_up"
    assert edge.consequent == "revenue_up"
    assert store.graph_stats().propositions == 2


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


def test_upsert_updates_same_id_and_appends_new_records() -> None:
    store = RelationStore()
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

    records, updated = store.assert_relations(
        [
            RelationInput(
                id="rel_keep",
                source="A",
                target="C",
                relation_type=RelationType.NECESSARY,
            ),
            RelationInput(
                id="rel_new",
                source="D",
                target="E",
                relation_type=RelationType.SUFFICIENT,
            ),
        ],
        mode="upsert",
    )

    assert [record.id for record in records] == ["rel_keep", "rel_new"]
    assert updated == 1
    listed = store.list_relations()
    assert len(listed) == 2
    assert {record.id: record.target for record in listed} == {
        "rel_keep": "C",
        "rel_new": "E",
    }


def test_upsert_dry_run_does_not_change_store() -> None:
    store = RelationStore()
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

    records, updated = store.assert_relations(
        [
            RelationInput(
                id="rel_keep",
                source="A",
                target="C",
                relation_type=RelationType.NECESSARY,
            )
        ],
        mode="upsert",
        dry_run=True,
    )

    assert [record.id for record in records] == ["rel_keep"]
    assert updated == 1
    listed = store.list_relations()
    assert len(listed) == 1
    assert listed[0].target == "B"


def test_memory_list_relations_supports_offset() -> None:
    store = RelationStore()
    store.assert_relations(
        [
            RelationInput(
                id="rel_a", source="A", target="B", relation_type=RelationType.SUFFICIENT
            ),
            RelationInput(
                id="rel_b", source="C", target="D", relation_type=RelationType.SUFFICIENT
            ),
            RelationInput(
                id="rel_c", source="E", target="F", relation_type=RelationType.SUFFICIENT
            ),
        ]
    )

    listed = store.list_relations(limit=2, offset=1)

    assert [record.id for record in listed] == ["rel_b", "rel_c"]


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


def test_replace_same_pair_uses_canonical_ids() -> None:
    store = RelationStore()
    store.assert_relations(
        [
            RelationInput(
                source="Label A",
                source_id="node_a",
                target="Label B",
                target_id="node_b",
                relation_type=RelationType.SUFFICIENT,
            )
        ]
    )

    _records, updated = store.assert_relations(
        [
            RelationInput(
                source="Renamed A",
                source_id="node_a",
                target="Renamed B",
                target_id="node_b",
                relation_type=RelationType.NECESSARY,
            )
        ],
        mode="replace_same_pair",
    )

    listed = store.list_relations()
    assert updated == 1
    assert len(listed) == 1
    assert listed[0].source == "Renamed A"
    assert listed[0].relation_type == RelationType.NECESSARY


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
                source_id="node_a",
                target="B",
                target_id="node_b",
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
    assert reloaded.list_relations()[0].source_id == "node_a"
    assert reloaded.list_relations()[0].target_id == "node_b"
    assert reloaded.implication_edges()[0].antecedent == "node_a"
    assert reloaded.implication_edges()[0].consequent == "node_b"
    assert reloaded.list_exclusive_groups()[0].group_id == groups[0].group_id
    assert reloaded.list_exclusive_groups()[0].members == ["B", "C"]


def test_sqlite_upsert_persists_updated_relation(tmp_path) -> None:
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

    records, updated = store.assert_relations(
        [
            RelationInput(
                id="rel_keep",
                source="A",
                target="C",
                relation_type=RelationType.NECESSARY,
            )
        ],
        mode="upsert",
    )
    reloaded = SqliteRelationStore(config)

    assert [record.id for record in records] == ["rel_keep"]
    assert updated == 1
    assert len(reloaded.list_relations()) == 1
    assert reloaded.list_relations()[0].target == "C"
    assert reloaded.list_relations()[0].relation_type == RelationType.NECESSARY


def test_sqlite_list_relations_supports_offset(tmp_path) -> None:
    config = NesyConfig(
        storage=StorageConfig(backend="sqlite", sqlite_path=str(tmp_path / "nesy.db"))
    )
    store = SqliteRelationStore(config)
    store.assert_relations(
        [
            RelationInput(
                id="rel_a", source="A", target="B", relation_type=RelationType.SUFFICIENT
            ),
            RelationInput(
                id="rel_b", source="C", target="D", relation_type=RelationType.SUFFICIENT
            ),
            RelationInput(
                id="rel_c", source="E", target="F", relation_type=RelationType.SUFFICIENT
            ),
        ]
    )

    listed = store.list_relations(limit=2, offset=1)

    assert [record.id for record in listed] == ["rel_b", "rel_c"]


def test_sqlite_store_allows_concurrent_assert_and_list(tmp_path) -> None:
    config = NesyConfig(
        storage=StorageConfig(backend="sqlite", sqlite_path=str(tmp_path / "nesy.db"))
    )
    store = SqliteRelationStore(config)

    def assert_and_list(index: int) -> int:
        store.assert_relations(
            [
                RelationInput(
                    id=f"rel_{index}",
                    source=f"A{index}",
                    target=f"B{index}",
                    relation_type=RelationType.SUFFICIENT,
                )
            ]
        )
        return len(store.list_relations())

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(assert_and_list, range(20)))

    assert len(results) == 20
    assert len(store.list_relations()) == 20


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
        [
            RelationInput(
                source="A",
                source_id="node_a",
                target="B",
                target_id="node_b",
                relation_type=RelationType.SUFFICIENT,
            )
        ]
    )

    reloaded = JsonRelationStore(config)

    assert reloaded.list_relations()[0].source == "A"
    assert reloaded.list_relations()[0].source_id == "node_a"
    assert reloaded.implication_edges()[0].consequent == "node_b"


def test_json_upsert_persists_updated_relation(tmp_path) -> None:
    config = NesyConfig(
        storage=StorageConfig(backend="json", json_path=str(tmp_path / "relations.json"))
    )
    store = JsonRelationStore(config)
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

    _records, updated = store.assert_relations(
        [
            RelationInput(
                id="rel_keep",
                source="A",
                target="C",
                relation_type=RelationType.NECESSARY,
            )
        ],
        mode="upsert",
    )
    reloaded = JsonRelationStore(config)

    assert updated == 1
    assert len(reloaded.list_relations()) == 1
    assert reloaded.list_relations()[0].target == "C"


def test_memory_import_records_keeps_context_metadata() -> None:
    store = RelationStore()

    store.import_records(
        [],
        [],
        mode="append",
        store_id="default",
        context_metadata={"ctx": {"causal_completeness": True}},
    )

    assert store.context_metadata() == {"ctx": {"causal_completeness": True}}


def test_memory_import_records_keeps_independence_records() -> None:
    store = RelationStore()

    store.import_records(
        [],
        [],
        [IndependenceRecord(id="ind_keep", left="C", right="A", context_id="ctx")],
        mode="append",
        store_id="default",
    )

    records = store.list_independence_records()
    assert len(records) == 1
    assert records[0].id == "ind_keep"
    assert records[0].left == "C"
    assert records[0].right == "A"


def test_sqlite_store_persists_context_metadata(tmp_path) -> None:
    config = NesyConfig(
        storage=StorageConfig(backend="sqlite", sqlite_path=str(tmp_path / "nesy.db"))
    )
    store = SqliteRelationStore(config)
    store.import_records(
        [],
        [],
        mode="append",
        store_id="default",
        context_metadata={"ctx": {"causal_completeness": True}},
    )

    reloaded = SqliteRelationStore(config)

    assert reloaded.context_metadata() == {"ctx": {"causal_completeness": True}}


def test_sqlite_store_persists_independence_records(tmp_path) -> None:
    config = NesyConfig(
        storage=StorageConfig(backend="sqlite", sqlite_path=str(tmp_path / "nesy.db"))
    )
    store = SqliteRelationStore(config)
    store.import_records(
        [],
        [],
        [IndependenceRecord(id="ind_keep", left="C", right="A", context_id="ctx")],
        mode="append",
        store_id="default",
    )

    reloaded = SqliteRelationStore(config)

    records = reloaded.list_independence_records()
    assert len(records) == 1
    assert records[0].id == "ind_keep"
    assert records[0].context_id == "ctx"


def test_json_store_persists_context_metadata(tmp_path) -> None:
    config = NesyConfig(
        storage=StorageConfig(backend="json", json_path=str(tmp_path / "relations.json"))
    )
    store = JsonRelationStore(config)
    store.import_records(
        [],
        [],
        mode="append",
        store_id="default",
        context_metadata={"ctx": {"causal_completeness": True}},
    )

    reloaded = JsonRelationStore(config)

    assert reloaded.context_metadata() == {"ctx": {"causal_completeness": True}}


def test_json_store_persists_independence_records(tmp_path) -> None:
    config = NesyConfig(
        storage=StorageConfig(backend="json", json_path=str(tmp_path / "relations.json"))
    )
    store = JsonRelationStore(config)
    store.import_records(
        [],
        [],
        [IndependenceRecord(id="ind_keep", left="C", right="A", context_id="ctx")],
        mode="append",
        store_id="default",
    )

    reloaded = JsonRelationStore(config)

    records = reloaded.list_independence_records()
    assert len(records) == 1
    assert records[0].left == "C"
    assert records[0].right == "A"


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
