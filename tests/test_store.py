from concurrent.futures import ThreadPoolExecutor

import pytest

from nesy_reasoning_mcp.auto_ingest import (
    CandidateRelation,
    ConversationTurnJob,
    ConversationTurnJobFilter,
    ConversationTurnJobStatus,
    EvidenceRecord,
    GateAction,
    GateResult,
    ReviewQueueFilter,
    ReviewQueueRecord,
    ReviewQueueStatus,
    ScheduledIngestionJob,
    ScheduledIngestionJobFilter,
    ScheduledIngestionJobStatus,
    ScheduledIngestionRun,
    ScheduledIngestionRunFilter,
    ScheduledIngestionRunStatus,
    ScheduledIngestionRunTrigger,
    ScheduledIngestionSourceConfig,
    ScheduledIngestionState,
)
from nesy_reasoning_mcp.config import NesyConfig, StorageConfig
from nesy_reasoning_mcp.schemas import (
    ExclusiveGroupInput,
    IndependenceRecord,
    PropositionRecord,
    RelationFilter,
    RelationInput,
    RelationRecord,
    RelationType,
)
from nesy_reasoning_mcp.storage.common import _apply_assert_relations_mode
from nesy_reasoning_mcp.store import (
    JsonRelationStore,
    RelationStore,
    SqliteRelationStore,
    create_relation_store,
)


def _queue_record(
    record_id: str = "queue-1",
    *,
    candidate_id: str = "candidate-1",
    run_id: str = "run-1",
    context_id: str = "default",
    store_id: str = "default",
) -> ReviewQueueRecord:
    candidate = CandidateRelation(
        id=candidate_id,
        source="A",
        target="B",
        relation_type=RelationType.SUFFICIENT,
        evidence=[EvidenceRecord(url="https://example.com/source", span="A enables B.")],
        context_id=context_id,
        store_id=store_id,
    )
    return ReviewQueueRecord(
        id=record_id,
        run_id=run_id,
        candidate=candidate,
        gate_result=GateResult(candidate_id=candidate.id, action=GateAction.QUEUE),
    )


def _scheduled_job(
    job_id: str = "sched-1",
    *,
    status: ScheduledIngestionJobStatus = ScheduledIngestionJobStatus.ACTIVE,
    next_run_at: str = "2026-01-01T00:00:00+00:00",
) -> ScheduledIngestionJob:
    return ScheduledIngestionJob(
        id=job_id,
        name=f"job {job_id}",
        status=status,
        cron="*/30 * * * *",
        source_config=ScheduledIngestionSourceConfig(urls=["https://example.com/source"]),
        state=ScheduledIngestionState(next_run_at=next_run_at),
    )


def _scheduled_run(
    run_id: str = "srun-1",
    *,
    job_id: str = "sched-1",
    status: ScheduledIngestionRunStatus = ScheduledIngestionRunStatus.SUCCEEDED,
) -> ScheduledIngestionRun:
    return ScheduledIngestionRun(
        id=run_id,
        job_id=job_id,
        trigger=ScheduledIngestionRunTrigger.MANUAL,
        status=status,
    )


def _turn_job(
    job_id: str = "turn-1",
    *,
    session_id: str = "session-1",
    transcript_path: str = "/tmp/transcript.jsonl",
    turn_index: int | None = 1,
    priority: int = 0,
    status: ConversationTurnJobStatus = ConversationTurnJobStatus.PENDING,
    agent_type: str | None = "codex",
) -> ConversationTurnJob:
    return ConversationTurnJob(
        job_id=job_id,
        session_id=session_id,
        transcript_path=transcript_path,
        turn_index=turn_index,
        priority=priority,
        status=status,
        agent_type=agent_type,
    )


def test_apply_assert_relations_mode_preserves_append_order() -> None:
    current = [
        RelationRecord(id="rel_a", source="A", target="B", relation_type=RelationType.SUFFICIENT)
    ]
    incoming = [
        RelationRecord(id="rel_b", source="C", target="D", relation_type=RelationType.NECESSARY)
    ]

    merged, updated = _apply_assert_relations_mode(current, incoming, "append")

    assert [record.id for record in merged] == ["rel_a", "rel_b"]
    assert updated == 0


def test_apply_assert_relations_mode_upserts_by_id() -> None:
    current = [
        RelationRecord(id="rel_a", source="A", target="B", relation_type=RelationType.SUFFICIENT)
    ]
    incoming = [
        RelationRecord(id="rel_a", source="A", target="C", relation_type=RelationType.NECESSARY),
        RelationRecord(id="rel_b", source="D", target="E", relation_type=RelationType.SUFFICIENT),
    ]

    merged, updated = _apply_assert_relations_mode(current, incoming, "upsert")

    assert [(record.id, record.target) for record in merged] == [("rel_a", "C"), ("rel_b", "E")]
    assert updated == 1


def test_apply_assert_relations_mode_replaces_same_canonical_pair() -> None:
    current = [
        RelationRecord(
            id="rel_a",
            source="Label A",
            source_id="node_a",
            target="Label B",
            target_id="node_b",
            relation_type=RelationType.SUFFICIENT,
            context_id="ctx",
        ),
        RelationRecord(
            id="rel_keep",
            source="Label A",
            source_id="node_a",
            target="Label B",
            target_id="node_b",
            relation_type=RelationType.SUFFICIENT,
            context_id="other",
        ),
    ]
    incoming = [
        RelationRecord(
            id="rel_new",
            source="Renamed A",
            source_id="node_a",
            target="Renamed B",
            target_id="node_b",
            relation_type=RelationType.NECESSARY,
            context_id="ctx",
        )
    ]

    merged, updated = _apply_assert_relations_mode(current, incoming, "replace_same_pair")

    assert [record.id for record in merged] == ["rel_keep", "rel_new"]
    assert updated == 1


def test_apply_assert_relations_mode_rejects_unsupported_mode() -> None:
    with pytest.raises(ValueError, match="unsupported assert mode"):
        _apply_assert_relations_mode([], [], "invalid")


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


def test_relation_record_generated_timestamps_match() -> None:
    record = RelationRecord(source="A", target="B", relation_type=RelationType.SUFFICIENT)

    assert record.created_at == record.updated_at


def test_relation_record_explicit_timestamp_is_preserved() -> None:
    record = RelationRecord(
        source="A",
        target="B",
        relation_type=RelationType.SUFFICIENT,
        created_at="2026-01-01T00:00:00+00:00",
        updated_at="2026-01-02T00:00:00+00:00",
    )

    assert record.created_at == "2026-01-01T00:00:00+00:00"
    assert record.updated_at == "2026-01-02T00:00:00+00:00"


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


def test_replace_same_pair_dry_run_does_not_change_memory_store() -> None:
    store = RelationStore()
    store.assert_relations(
        [
            RelationInput(
                source="A",
                target="B",
                relation_type=RelationType.SUFFICIENT,
            )
        ]
    )

    _records, updated = store.assert_relations(
        [
            RelationInput(
                source="A",
                target="B",
                relation_type=RelationType.NECESSARY,
            )
        ],
        mode="replace_same_pair",
        dry_run=True,
    )

    listed = store.list_relations()
    assert updated == 1
    assert len(listed) == 1
    assert listed[0].relation_type == RelationType.SUFFICIENT


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


def test_sqlite_replace_same_pair_persists_updated_relation(tmp_path) -> None:
    config = NesyConfig(
        storage=StorageConfig(backend="sqlite", sqlite_path=str(tmp_path / "nesy.db"))
    )
    store = SqliteRelationStore(config)
    store.assert_relations(
        [
            RelationInput(
                id="rel_old",
                source="A",
                target="B",
                relation_type=RelationType.SUFFICIENT,
            ),
            RelationInput(
                id="rel_keep",
                source="A",
                target="B",
                relation_type=RelationType.SUFFICIENT,
                context_id="other",
            ),
        ]
    )

    records, updated = store.assert_relations(
        [
            RelationInput(
                id="rel_new",
                source="A",
                target="B",
                relation_type=RelationType.NECESSARY,
            )
        ],
        mode="replace_same_pair",
    )
    reloaded = SqliteRelationStore(config)

    assert [record.id for record in records] == ["rel_new"]
    assert updated == 1
    assert {(record.id, record.relation_type) for record in reloaded.list_relations()} == {
        ("rel_keep", RelationType.SUFFICIENT),
        ("rel_new", RelationType.NECESSARY),
    }


def test_sqlite_replace_same_pair_dry_run_does_not_change_store(tmp_path) -> None:
    config = NesyConfig(
        storage=StorageConfig(backend="sqlite", sqlite_path=str(tmp_path / "nesy.db"))
    )
    store = SqliteRelationStore(config)
    store.assert_relations(
        [
            RelationInput(
                id="rel_old",
                source="A",
                target="B",
                relation_type=RelationType.SUFFICIENT,
            )
        ]
    )

    _records, updated = store.assert_relations(
        [
            RelationInput(
                id="rel_new",
                source="A",
                target="B",
                relation_type=RelationType.NECESSARY,
            )
        ],
        mode="replace_same_pair",
        dry_run=True,
    )
    reloaded = SqliteRelationStore(config)

    assert updated == 1
    assert [(record.id, record.relation_type) for record in reloaded.list_relations()] == [
        ("rel_old", RelationType.SUFFICIENT)
    ]


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


def test_sqlite_list_relations_filters_in_sql(tmp_path) -> None:
    config = NesyConfig(
        storage=StorageConfig(backend="sqlite", sqlite_path=str(tmp_path / "nesy.db"))
    )
    store = SqliteRelationStore(config)
    store.assert_relations(
        [
            RelationInput(
                id="rel_a",
                source="A",
                target="B",
                relation_type=RelationType.SUFFICIENT,
                context_id="ctx_a",
                metadata={"domain": "finance"},
            ),
            RelationInput(
                id="rel_b",
                source="A",
                target="C",
                relation_type=RelationType.NECESSARY,
                context_id="ctx_b",
                metadata={"domain": "ops"},
            ),
        ]
    )
    traced: list[str] = []
    store._conn.set_trace_callback(traced.append)

    listed = store.list_relations(
        RelationFilter(
            source="A",
            relation_type=RelationType.SUFFICIENT,
            context_id="ctx_a",
            domain="finance",
        ),
        limit=1,
    )

    select_sql = next(statement for statement in reversed(traced) if "FROM relations" in statement)
    assert [record.id for record in listed] == ["rel_a"]
    assert "WHERE source =" in select_sql
    assert "relation_type =" in select_sql
    assert "context_id =" in select_sql
    assert "json_extract(metadata_json" in select_sql
    assert "LIMIT 1 OFFSET 0" in select_sql


def test_sqlite_domain_filter_uses_expression_index(tmp_path) -> None:
    config = NesyConfig(
        storage=StorageConfig(backend="sqlite", sqlite_path=str(tmp_path / "nesy.db"))
    )
    store = SqliteRelationStore(config)

    plan = store._conn.execute(
        """
        EXPLAIN QUERY PLAN
        SELECT id FROM relations
        WHERE json_extract(metadata_json, '$.domain') = ?
        """,
        ("finance",),
    ).fetchall()

    assert any("idx_relations_domain" in row["detail"] for row in plan)


def test_sqlite_sync_rejects_unapproved_identifiers(tmp_path) -> None:
    config = NesyConfig(
        storage=StorageConfig(backend="sqlite", sqlite_path=str(tmp_path / "nesy.db"))
    )
    store = SqliteRelationStore(config)

    try:
        store._sync_single_key_table(
            "relations; DROP TABLE relations", "id", [], "desired_relation_ids"
        )
    except ValueError as exc:
        assert "unsupported SQL table" in str(exc)
    else:
        raise AssertionError("expected unsafe identifier to be rejected")


def test_sqlite_replace_store_differential_sync_deletes_missing_rows_and_cleans_temp(
    tmp_path,
) -> None:
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
            ),
            RelationInput(
                id="rel_remove",
                source="C",
                target="D",
                relation_type=RelationType.SUFFICIENT,
            ),
            RelationInput(
                id="rel_other_store",
                source="E",
                target="F",
                relation_type=RelationType.SUFFICIENT,
                store_id="other",
            ),
        ]
    )

    store.import_records(
        [
            RelationRecord(
                id="rel_keep",
                source="A",
                target="Updated B",
                relation_type=RelationType.NECESSARY,
            )
        ],
        [],
        mode="replace_store",
        store_id="default",
    )

    listed = store.list_relations()
    temp_tables = store._conn.execute(
        "SELECT name FROM sqlite_temp_master WHERE type = 'table'"
    ).fetchall()

    assert sorted((relation.id, relation.target, relation.store_id) for relation in listed) == [
        ("rel_keep", "Updated B", "default"),
        ("rel_other_store", "F", "other"),
    ]
    assert temp_tables == []


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


def test_memory_review_queue_lifecycle() -> None:
    store = RelationStore()
    record = _queue_record()

    queued, updated = store.enqueue_review_queue([record])

    assert updated == 0
    assert queued[0].id == "queue-1"
    assert store.list_review_queue()[0].status == ReviewQueueStatus.PENDING
    assert store.list_review_queue(ReviewQueueFilter(candidate_id="candidate-1"))[0].id == (
        "queue-1"
    )

    committed = store.mark_review_queue_committed(["queue-1"], {"queue-1": ["rel-1"]})
    listed = store.list_review_queue(ReviewQueueFilter(status=ReviewQueueStatus.COMMITTED))

    assert committed == 1
    assert listed[0].committed_relation_ids == ["rel-1"]


def test_memory_review_queue_filters_by_ids_and_keyset() -> None:
    store = RelationStore()
    store.enqueue_review_queue(
        [
            _queue_record("queue-1", candidate_id="candidate-1").model_copy(
                update={"created_at": "2026-01-01T00:00:01+00:00"}
            ),
            _queue_record("queue-2", candidate_id="candidate-2").model_copy(
                update={"created_at": "2026-01-01T00:00:02+00:00"}
            ),
        ]
    )

    listed = store.list_review_queue(
        ReviewQueueFilter(
            ids=["queue-1", "queue-missing"],
            after_created_at="2026-01-01T00:00:00+00:00",
            after_id="queue-0",
        )
    )

    assert [record.id for record in listed] == ["queue-1"]


def test_memory_review_queue_resolve() -> None:
    store = RelationStore()
    store.enqueue_review_queue([_queue_record()])

    resolved = store.resolve_review_queue(["queue-1"], reason="duplicate")
    listed = store.list_review_queue(ReviewQueueFilter(status=ReviewQueueStatus.RESOLVED))

    assert resolved == 1
    assert listed[0].resolution["reason"] == "duplicate"


def test_json_store_persists_review_queue(tmp_path) -> None:
    config = NesyConfig(
        storage=StorageConfig(backend="json", json_path=str(tmp_path / "relations.json"))
    )
    store = JsonRelationStore(config)
    store.enqueue_review_queue([_queue_record()])

    reloaded = JsonRelationStore(config)

    assert reloaded.list_review_queue()[0].candidate.id == "candidate-1"


def test_sqlite_store_persists_review_queue_and_filters(tmp_path) -> None:
    config = NesyConfig(
        storage=StorageConfig(backend="sqlite", sqlite_path=str(tmp_path / "nesy.db"))
    )
    store = SqliteRelationStore(config)
    store.enqueue_review_queue(
        [
            _queue_record("queue-1", candidate_id="candidate-1", context_id="ctx-a"),
            _queue_record("queue-2", candidate_id="candidate-2", context_id="ctx-b"),
        ]
    )

    reloaded = SqliteRelationStore(config)
    listed = reloaded.list_review_queue(ReviewQueueFilter(context_id="ctx-b"))

    assert [record.id for record in listed] == ["queue-2"]
    assert listed[0].candidate.context_id == "ctx-b"


def test_sqlite_store_filters_review_queue_by_ids_and_keyset(tmp_path) -> None:
    config = NesyConfig(
        storage=StorageConfig(backend="sqlite", sqlite_path=str(tmp_path / "nesy.db"))
    )
    store = SqliteRelationStore(config)
    store.enqueue_review_queue(
        [
            _queue_record("queue-1", candidate_id="candidate-1").model_copy(
                update={
                    "created_at": "2026-01-01T00:00:01+00:00",
                    "updated_at": "2026-01-01T00:00:01+00:00",
                }
            ),
            _queue_record("queue-2", candidate_id="candidate-2").model_copy(
                update={
                    "created_at": "2026-01-01T00:00:02+00:00",
                    "updated_at": "2026-01-01T00:00:02+00:00",
                }
            ),
            _queue_record("queue-3", candidate_id="candidate-3").model_copy(
                update={
                    "created_at": "2026-01-01T00:00:03+00:00",
                    "updated_at": "2026-01-01T00:00:03+00:00",
                }
            ),
        ]
    )

    listed = store.list_review_queue(
        ReviewQueueFilter(
            ids=["queue-1", "queue-2", "queue-missing"],
            after_created_at="2026-01-01T00:00:01+00:00",
            after_id="queue-1",
        )
    )

    assert [record.id for record in listed] == ["queue-2"]


def test_sqlite_review_queue_rejects_index_payload_mismatch(tmp_path) -> None:
    config = NesyConfig(
        storage=StorageConfig(backend="sqlite", sqlite_path=str(tmp_path / "nesy.db"))
    )
    store = SqliteRelationStore(config)
    store.enqueue_review_queue([_queue_record()])
    store._conn.execute(  # noqa: SLF001
        "UPDATE review_queue SET candidate_id = ? WHERE id = ?",
        ("candidate-other", "queue-1"),
    )
    store._conn.commit()  # noqa: SLF001

    with pytest.raises(ValueError, match="indexed columns"):
        store.list_review_queue(ReviewQueueFilter(ids=["queue-1"]))


def test_sqlite_review_queue_threaded_enqueue_and_list(tmp_path) -> None:
    config = NesyConfig(
        storage=StorageConfig(backend="sqlite", sqlite_path=str(tmp_path / "nesy.db"))
    )
    store = SqliteRelationStore(config)

    def enqueue_and_read(index: int) -> int:
        record_id = f"queue-{index}"
        store.enqueue_review_queue([_queue_record(record_id, candidate_id=f"candidate-{index}")])
        return len(store.list_review_queue(ReviewQueueFilter(ids=[record_id])))

    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(enqueue_and_read, range(12)))

    assert results == [1] * 12
    assert len(store.list_review_queue()) == 12


def test_memory_ingestion_queue_filters_and_orders_by_priority() -> None:
    store = RelationStore()
    store.enqueue_ingestion_jobs(
        [
            _turn_job("turn-low", session_id="session-a", priority=0).model_copy(
                update={"enqueued_at": "2026-01-01T00:00:01+00:00"}
            ),
            _turn_job("turn-high", session_id="session-b", priority=1).model_copy(
                update={"enqueued_at": "2026-01-01T00:00:02+00:00"}
            ),
            _turn_job("turn-next", session_id="session-b", priority=1).model_copy(
                update={"enqueued_at": "2026-01-01T00:00:03+00:00"}
            ),
        ]
    )

    listed = store.list_ingestion_jobs(ConversationTurnJobFilter(session_id="session-b"))
    by_ids = store.list_ingestion_jobs(
        ConversationTurnJobFilter(ids=["turn-high", "turn-next", "turn-missing"])
    )

    assert [record.job_id for record in listed] == ["turn-high", "turn-next"]
    assert [record.job_id for record in by_ids] == ["turn-high", "turn-next"]


def test_memory_ingestion_queue_upserts_by_job_id() -> None:
    store = RelationStore()
    store.enqueue_ingestion_jobs([_turn_job("turn-1", priority=0)])

    queued, updated = store.enqueue_ingestion_jobs([_turn_job("turn-1", priority=1)])

    assert updated == 1
    assert queued[0].priority == 1
    assert store.list_ingestion_jobs()[0].priority == 1


def test_memory_ingestion_queue_dedupes_duplicate_incoming_ids() -> None:
    store = RelationStore()

    queued, updated = store.enqueue_ingestion_jobs(
        [_turn_job("turn-1", priority=0), _turn_job("turn-1", priority=1)]
    )

    assert updated == 0
    assert len(queued) == 1
    assert store.list_ingestion_jobs()[0].priority == 1


def test_json_store_persists_ingestion_queue(tmp_path) -> None:
    config = NesyConfig(
        storage=StorageConfig(backend="json", json_path=str(tmp_path / "relations.json"))
    )
    store = JsonRelationStore(config)
    store.enqueue_ingestion_jobs([_turn_job()])

    reloaded = JsonRelationStore(config)

    assert reloaded.list_ingestion_jobs()[0].job_id == "turn-1"


def test_sqlite_store_persists_ingestion_queue_and_filters(tmp_path) -> None:
    config = NesyConfig(
        storage=StorageConfig(backend="sqlite", sqlite_path=str(tmp_path / "nesy.db"))
    )
    store = SqliteRelationStore(config)
    store.enqueue_ingestion_jobs(
        [
            _turn_job("turn-1", session_id="session-a", agent_type="codex"),
            _turn_job("turn-2", session_id="session-b", agent_type="claude", priority=1),
        ]
    )

    reloaded = SqliteRelationStore(config)
    listed = reloaded.list_ingestion_jobs(ConversationTurnJobFilter(agent_type="claude"))

    assert [record.job_id for record in listed] == ["turn-2"]
    assert listed[0].session_id == "session-b"


def test_sqlite_ingestion_queue_limit_offset_and_priority_order(tmp_path) -> None:
    config = NesyConfig(
        storage=StorageConfig(backend="sqlite", sqlite_path=str(tmp_path / "nesy.db"))
    )
    store = SqliteRelationStore(config)
    store.enqueue_ingestion_jobs(
        [
            _turn_job("turn-1", priority=0).model_copy(
                update={"enqueued_at": "2026-01-01T00:00:01+00:00"}
            ),
            _turn_job("turn-2", priority=1).model_copy(
                update={"enqueued_at": "2026-01-01T00:00:02+00:00"}
            ),
            _turn_job("turn-3", priority=1).model_copy(
                update={"enqueued_at": "2026-01-01T00:00:03+00:00"}
            ),
        ]
    )

    listed = store.list_ingestion_jobs(limit=1, offset=1)

    assert [record.job_id for record in store.list_ingestion_jobs()] == [
        "turn-2",
        "turn-3",
        "turn-1",
    ]
    assert [record.job_id for record in listed] == ["turn-3"]


def test_sqlite_ingestion_queue_dedupes_duplicate_incoming_ids(tmp_path) -> None:
    config = NesyConfig(
        storage=StorageConfig(backend="sqlite", sqlite_path=str(tmp_path / "nesy.db"))
    )
    store = SqliteRelationStore(config)

    queued, updated = store.enqueue_ingestion_jobs(
        [_turn_job("turn-1", priority=0), _turn_job("turn-1", priority=1)]
    )

    assert updated == 0
    assert len(queued) == 1
    assert store.list_ingestion_jobs()[0].priority == 1


def test_sqlite_ingestion_queue_rejects_index_payload_mismatch(tmp_path) -> None:
    config = NesyConfig(
        storage=StorageConfig(backend="sqlite", sqlite_path=str(tmp_path / "nesy.db"))
    )
    store = SqliteRelationStore(config)
    store.enqueue_ingestion_jobs([_turn_job()])
    store._conn.execute(  # noqa: SLF001
        "UPDATE ingestion_queue SET session_id = ? WHERE job_id = ?",
        ("session-other", "turn-1"),
    )
    store._conn.commit()  # noqa: SLF001

    with pytest.raises(ValueError, match="indexed columns"):
        store.list_ingestion_jobs(ConversationTurnJobFilter(ids=["turn-1"]))


def test_sqlite_ingestion_queue_threaded_enqueue_and_list(tmp_path) -> None:
    config = NesyConfig(
        storage=StorageConfig(backend="sqlite", sqlite_path=str(tmp_path / "nesy.db"))
    )
    store = SqliteRelationStore(config)

    def enqueue_and_read(index: int) -> int:
        job_id = f"turn-{index}"
        store.enqueue_ingestion_jobs([_turn_job(job_id, session_id=f"session-{index}")])
        return len(store.list_ingestion_jobs(ConversationTurnJobFilter(ids=[job_id])))

    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(enqueue_and_read, range(12)))

    assert results == [1] * 12
    assert len(store.list_ingestion_jobs()) == 12


def test_memory_scheduled_ingestion_job_and_run_lifecycle() -> None:
    store = RelationStore()
    job = _scheduled_job()

    stored, updated = store.upsert_scheduled_ingestion_job(job)
    run = store.append_scheduled_ingestion_run(_scheduled_run(job_id=job.id))

    assert updated == 0
    assert stored.id == job.id
    assert store.get_scheduled_ingestion_job(job.id) is not None
    assert store.list_scheduled_ingestion_jobs()[0].id == job.id
    assert store.list_scheduled_ingestion_runs()[0].id == run.id

    state = ScheduledIngestionState(
        next_run_at="2026-01-01T00:30:00+00:00",
        last_run_id=run.id,
    )
    updated_job = store.update_scheduled_ingestion_job_state(
        job.id,
        state=state,
        status=ScheduledIngestionJobStatus.DISABLED,
    )

    assert updated_job is not None
    assert updated_job.status == ScheduledIngestionJobStatus.DISABLED
    assert updated_job.state.last_run_id == run.id


def test_memory_scheduled_ingestion_job_state_expected_status() -> None:
    store = RelationStore()
    job = _scheduled_job()
    store.upsert_scheduled_ingestion_job(job)

    rejected = store.update_scheduled_ingestion_job_state(
        job.id,
        state=job.state,
        status=ScheduledIngestionJobStatus.RUNNING,
        expected_status=ScheduledIngestionJobStatus.DISABLED,
    )
    accepted = store.update_scheduled_ingestion_job_state(
        job.id,
        state=job.state,
        status=ScheduledIngestionJobStatus.RUNNING,
        expected_status=ScheduledIngestionJobStatus.ACTIVE,
    )

    assert rejected is None
    assert accepted is not None
    assert accepted.status == ScheduledIngestionJobStatus.RUNNING


def test_json_store_persists_scheduled_ingestion_jobs_and_runs(tmp_path) -> None:
    config = NesyConfig(
        storage=StorageConfig(backend="json", json_path=str(tmp_path / "relations.json"))
    )
    store = JsonRelationStore(config)
    store.upsert_scheduled_ingestion_job(_scheduled_job())
    store.append_scheduled_ingestion_run(_scheduled_run())

    reloaded = JsonRelationStore(config)

    assert reloaded.list_scheduled_ingestion_jobs()[0].id == "sched-1"
    assert reloaded.list_scheduled_ingestion_runs()[0].id == "srun-1"


def test_sqlite_store_persists_and_filters_scheduled_ingestion(tmp_path) -> None:
    config = NesyConfig(
        storage=StorageConfig(backend="sqlite", sqlite_path=str(tmp_path / "nesy.db"))
    )
    store = SqliteRelationStore(config)
    store.upsert_scheduled_ingestion_job(
        _scheduled_job("sched-1", next_run_at="2026-01-01T00:00:00+00:00")
    )
    store.upsert_scheduled_ingestion_job(
        _scheduled_job(
            "sched-2",
            status=ScheduledIngestionJobStatus.DISABLED,
            next_run_at="2026-01-01T00:30:00+00:00",
        )
    )
    store.append_scheduled_ingestion_run(_scheduled_run("srun-1", job_id="sched-1"))
    store.append_scheduled_ingestion_run(
        _scheduled_run(
            "srun-2",
            job_id="sched-2",
            status=ScheduledIngestionRunStatus.FAILED,
        )
    )

    reloaded = SqliteRelationStore(config)
    due = reloaded.list_scheduled_ingestion_jobs(
        ScheduledIngestionJobFilter(
            status=ScheduledIngestionJobStatus.ACTIVE,
            due_before="2026-01-01T00:15:00+00:00",
        )
    )
    failed_runs = reloaded.list_scheduled_ingestion_runs(
        ScheduledIngestionRunFilter(status=ScheduledIngestionRunStatus.FAILED)
    )

    assert [job.id for job in due] == ["sched-1"]
    assert [run.id for run in failed_runs] == ["srun-2"]


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


def test_memory_import_records_keeps_propositions_and_normalizes_relations() -> None:
    store = RelationStore()

    store.import_records(
        [
            RelationRecord(
                id="rel_profit",
                source="利润增加",
                target="Revenue increases",
                relation_type=RelationType.SUFFICIENT,
            )
        ],
        [],
        propositions=[
            PropositionRecord(
                id="profit_up",
                label="Profit increases",
                aliases=["利润增加"],
            ),
            PropositionRecord(id="revenue_up", label="Revenue increases"),
        ],
        mode="append",
        store_id="default",
    )

    relation = store.list_relations()[0]
    assert [proposition.id for proposition in store.list_propositions()] == [
        "profit_up",
        "revenue_up",
    ]
    assert relation.source_id == "profit_up"
    assert relation.target_id == "revenue_up"
    assert store.implication_edges()[0].antecedent == "profit_up"


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


def test_sqlite_store_persists_propositions_and_creates_table_for_existing_db(tmp_path) -> None:
    sqlite_path = tmp_path / "nesy.db"
    config = NesyConfig(storage=StorageConfig(backend="sqlite", sqlite_path=str(sqlite_path)))
    store = SqliteRelationStore(config)
    store.assert_relations(
        [RelationInput(id="rel_keep", source="A", target="B", relation_type="sufficient")]
    )

    reloaded = SqliteRelationStore(config)
    reloaded.import_records(
        [
            RelationRecord(
                id="rel_profit",
                source="利润增加",
                target="收入增加",
                relation_type=RelationType.SUFFICIENT,
            )
        ],
        [],
        propositions=[
            PropositionRecord(
                id="profit_up",
                label="Profit increases",
                aliases=["利润增加"],
            ),
            PropositionRecord(id="revenue_up", label="收入增加"),
        ],
        mode="append",
        store_id="default",
    )

    final_store = SqliteRelationStore(config)

    assert [proposition.id for proposition in final_store.list_propositions()] == [
        "profit_up",
        "revenue_up",
    ]
    assert final_store.list_relations()[-1].source_id == "profit_up"


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


def test_json_store_persists_propositions(tmp_path) -> None:
    config = NesyConfig(
        storage=StorageConfig(backend="json", json_path=str(tmp_path / "relations.json"))
    )
    store = JsonRelationStore(config)
    store.import_records(
        [],
        [],
        propositions=[
            PropositionRecord(
                id="profit_down",
                label="Profit decreases",
                aliases=["利润下降"],
                negates="profit_up",
                metadata={"lang": "zh"},
            )
        ],
        mode="append",
        store_id="default",
    )

    reloaded = JsonRelationStore(config)
    proposition = reloaded.list_propositions()[0]

    assert proposition.id == "profit_down"
    assert proposition.aliases == ["利润下降"]
    assert proposition.negates == "profit_up"
    assert proposition.metadata == {"lang": "zh"}


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


def test_clear_all_removes_proposition_registry() -> None:
    store = RelationStore()
    store.import_records(
        [],
        [],
        propositions=[PropositionRecord(id="profit_up", label="Profit increases")],
        mode="append",
        store_id="default",
    )

    store.clear_relations(
        scope="all",
        store_id="default",
        context_id="default",
        relation_filter=RelationFilter(),
        dry_run=False,
    )

    assert store.list_propositions() == []


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
