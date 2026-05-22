import json
from pathlib import Path

import pytest

from nesy_reasoning_mcp.config import NesyConfig, SecurityConfig
from nesy_reasoning_mcp.store import RelationStore
from nesy_reasoning_mcp.tools import (
    ASSERT_EXCLUSIVE,
    ASSERT_RELATIONS,
    CHECK_CONTRADICTIONS,
    CLASSIFY,
    CLEAR_RELATIONS,
    COUNTERFACTUAL,
    EXPORT_RELATIONS,
    LIST_RELATIONS,
    LOAD_RELATIONS,
    SUMMARIZE_GRAPH,
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
async def test_summarize_graph_filters_sorts_and_mirrors_content() -> None:
    store = RelationStore()
    await call_tool(
        ASSERT_EXCLUSIVE,
        {
            "groups": [
                {
                    "group_id": "profit_state",
                    "members": ["利润增加", "利润减少"],
                    "context_id": "ctx",
                }
            ]
        },
        store,
    )
    await call_tool(
        ASSERT_RELATIONS,
        {
            "relations": [
                {
                    "id": "rel_z",
                    "source": "库存下降",
                    "target": "现金回收",
                    "relation_type": "sufficient",
                    "context_id": "ctx",
                    "metadata": {"domain": "retail"},
                },
                {
                    "id": "rel_a",
                    "source": "降价",
                    "target": "利润减少",
                    "relation_type": "sufficient",
                    "context_id": "ctx",
                    "metadata": {"domain": "retail"},
                },
                {
                    "id": "rel_other",
                    "source": "降价",
                    "target": "利润增加",
                    "relation_type": "sufficient",
                    "context_id": "other",
                    "metadata": {"domain": "retail"},
                },
            ],
            "check_contradictions": False,
        },
        store,
    )

    result = await call_tool(
        SUMMARIZE_GRAPH,
        {
            "focus_terms": ["利润"],
            "context_filter": {"context_id": "ctx", "domain": "retail"},
        },
        store,
    )

    assert result.isError is False
    assert json.loads(result.content[0].text) == result.structuredContent
    assert result.structuredContent["relation_count_included"] == 1
    assert "降价 sufficient 利润减少" in result.structuredContent["summary"]
    assert "profit_state" in result.structuredContent["summary"]
    assert "现金回收" not in result.structuredContent["summary"]
    assert "利润增加" in result.structuredContent["summary"]


@pytest.mark.asyncio
async def test_summarize_graph_truncates_by_relation_and_char_limits() -> None:
    store = RelationStore()
    await call_tool(
        ASSERT_RELATIONS,
        {
            "relations": [
                {
                    "id": "rel_1",
                    "source": "A",
                    "target": "B",
                    "relation_type": "sufficient",
                },
                {
                    "id": "rel_2",
                    "source": "C",
                    "target": "D",
                    "relation_type": "sufficient",
                },
            ],
            "check_contradictions": False,
        },
        store,
    )

    relation_limited = await call_tool(
        SUMMARIZE_GRAPH,
        {"max_relations": 1, "include_exclusives": False},
        store,
    )
    char_limited = await call_tool(
        SUMMARIZE_GRAPH,
        {"max_chars": 500, "include_exclusives": False},
        store,
    )
    invalid = await call_tool(SUMMARIZE_GRAPH, {"max_chars": 499}, store)

    assert relation_limited.structuredContent["truncated"] is True
    assert relation_limited.structuredContent["relation_count_included"] == 1
    assert char_limited.structuredContent["truncated"] is False
    assert invalid.isError is True


@pytest.mark.asyncio
async def test_load_relations_validate_only_does_not_change_store() -> None:
    store = RelationStore()
    result = await call_tool(
        LOAD_RELATIONS,
        {
            "source_type": "inline",
            "validate_only": True,
            "data": {"relations": [{"source": "A", "target": "B", "relation_type": "sufficient"}]},
        },
        store,
    )

    assert result.isError is False
    assert result.structuredContent["loaded_relations"] == 1
    assert result.structuredContent["validate_only"] is True
    assert store.list_relations() == []


@pytest.mark.asyncio
async def test_export_inline_roundtrip_through_load() -> None:
    store = RelationStore()
    await call_tool(
        ASSERT_EXCLUSIVE,
        {"groups": [{"group_id": "state", "members": ["B", "C"]}]},
        store,
    )
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
                "independence_records": [
                    {
                        "id": "ind_c_a",
                        "left": "C",
                        "right": "A",
                        "context_id": "ctx",
                    }
                ],
                "context_metadata": {"ctx": {"causal_completeness": True}},
            },
            "check_contradictions": False,
        },
        store,
    )

    exported = await call_tool(EXPORT_RELATIONS, {"destination": "inline"}, store)

    new_store = RelationStore()
    loaded = await call_tool(
        LOAD_RELATIONS,
        {
            "source_type": "inline",
            "data": exported.structuredContent["data"],
            "check_contradictions": False,
        },
        new_store,
    )

    assert exported.isError is False
    assert loaded.isError is False
    assert loaded.structuredContent["loaded_relations"] == 1
    assert loaded.structuredContent["loaded_exclusive_groups"] == 1
    assert new_store.list_relations()[0].source == "A"
    assert new_store.list_exclusive_groups()[0].members == ["B", "C"]
    assert new_store.list_independence_records()[0].id == "ind_c_a"
    assert new_store.context_metadata() == {"ctx": {"causal_completeness": True}}


@pytest.mark.asyncio
async def test_export_jsonl_roundtrip_includes_independence_records(tmp_path: Path) -> None:
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    store = RelationStore(NesyConfig(security=SecurityConfig(allowed_roots=[str(allowed)])))
    await call_tool(
        LOAD_RELATIONS,
        {
            "source_type": "inline",
            "data": {
                "relations": [{"source": "A", "target": "B", "relation_type": "sufficient"}],
                "independence_records": [{"id": "ind_c_a", "left": "C", "right": "A"}],
            },
            "check_contradictions": False,
        },
        store,
    )

    exported = await call_tool(
        EXPORT_RELATIONS,
        {"destination": "file", "format": "jsonl", "path": str(allowed / "relations.jsonl")},
        store,
    )
    new_store = RelationStore(NesyConfig(security=SecurityConfig(allowed_roots=[str(allowed)])))
    loaded = await call_tool(
        LOAD_RELATIONS,
        {
            "source_type": "file",
            "path": str(allowed / "relations.jsonl"),
            "check_contradictions": False,
        },
        new_store,
    )

    assert exported.isError is False
    assert loaded.isError is False
    assert new_store.list_independence_records()[0].id == "ind_c_a"


@pytest.mark.asyncio
async def test_load_relations_upsert_updates_same_id() -> None:
    store = RelationStore()
    await call_tool(
        LOAD_RELATIONS,
        {
            "source_type": "inline",
            "data": {
                "relations": [
                    {
                        "id": "rel_fixed",
                        "source": "A",
                        "target": "B",
                        "relation_type": "sufficient",
                    }
                ]
            },
            "check_contradictions": False,
        },
        store,
    )

    result = await call_tool(
        LOAD_RELATIONS,
        {
            "source_type": "inline",
            "mode": "upsert",
            "data": {
                "relations": [
                    {
                        "id": "rel_fixed",
                        "source": "A",
                        "target": "C",
                        "relation_type": "sufficient",
                    }
                ]
            },
            "check_contradictions": False,
        },
        store,
    )

    assert result.structuredContent["updated_relations"] == 1
    assert len(store.list_relations()) == 1
    assert store.list_relations()[0].target == "C"


@pytest.mark.asyncio
async def test_load_relations_upsert_updates_independence_same_pair() -> None:
    store = RelationStore()
    await call_tool(
        LOAD_RELATIONS,
        {
            "source_type": "inline",
            "data": {
                "independence_records": [
                    {"id": "ind_old", "left": "C", "right": "A", "confidence": 0.5}
                ]
            },
            "check_contradictions": False,
        },
        store,
    )

    result = await call_tool(
        LOAD_RELATIONS,
        {
            "source_type": "inline",
            "mode": "upsert",
            "data": {
                "independence_records": [
                    {"id": "ind_new", "left": "A", "right": "C", "confidence": 0.9}
                ]
            },
            "check_contradictions": False,
        },
        store,
    )

    records = store.list_independence_records()
    assert result.isError is False
    assert len(records) == 1
    assert records[0].id == "ind_new"
    assert records[0].confidence == 0.9


@pytest.mark.asyncio
async def test_load_relations_replace_store_only_replaces_target_store() -> None:
    store = RelationStore()
    await call_tool(
        ASSERT_RELATIONS,
        {
            "relations": [
                {"source": "A", "target": "B", "relation_type": "sufficient", "store_id": "s1"},
                {"source": "C", "target": "D", "relation_type": "sufficient", "store_id": "s2"},
            ],
            "check_contradictions": False,
        },
        store,
    )

    await call_tool(
        LOAD_RELATIONS,
        {
            "source_type": "inline",
            "mode": "replace_store",
            "store_id": "s1",
            "data": {"relations": [{"source": "X", "target": "Y", "relation_type": "sufficient"}]},
            "check_contradictions": False,
        },
        store,
    )

    assert {(item.source, item.store_id) for item in store.list_relations()} == {
        ("X", "s1"),
        ("C", "s2"),
    }


@pytest.mark.asyncio
async def test_load_and_export_file_enforce_allowed_roots(tmp_path: Path) -> None:
    allowed = tmp_path / "allowed"
    outside = tmp_path / "outside"
    allowed.mkdir()
    outside.mkdir()
    store = RelationStore(NesyConfig(security=SecurityConfig(allowed_roots=[str(allowed)])))
    allowed_file = allowed / "relations.json"
    allowed_file.write_text(
        json.dumps({"relations": [{"source": "A", "target": "B", "relation_type": "sufficient"}]}),
        encoding="utf-8",
    )

    loaded = await call_tool(
        LOAD_RELATIONS,
        {
            "source_type": "file",
            "path": str(allowed_file),
            "check_contradictions": False,
        },
        store,
    )
    rejected_load = await call_tool(
        LOAD_RELATIONS,
        {
            "source_type": "file",
            "path": str(outside / "relations.json"),
            "check_contradictions": False,
        },
        store,
    )
    exported = await call_tool(
        EXPORT_RELATIONS,
        {"destination": "file", "path": str(allowed / "export.json")},
        store,
    )
    rejected_export = await call_tool(
        EXPORT_RELATIONS,
        {"destination": "file", "path": str(outside / "export.json")},
        store,
    )
    nested_export = await call_tool(
        EXPORT_RELATIONS,
        {"destination": "file", "path": str(allowed / "nested" / "export.json")},
        store,
    )

    assert loaded.isError is False
    assert rejected_load.isError is True
    assert rejected_load.structuredContent["diagnostics"][0]["code"] == "LOAD_RELATIONS_FAILED"
    assert exported.isError is False
    assert nested_export.isError is False
    assert rejected_export.isError is True
    assert rejected_export.structuredContent["diagnostics"][0]["code"] == "EXPORT_RELATIONS_FAILED"
    assert not (outside / "export.json").exists()


@pytest.mark.asyncio
async def test_load_relations_resource_uri_file_uses_allowed_roots(tmp_path: Path) -> None:
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    relation_file = allowed / "relations.json"
    relation_file.write_text(
        json.dumps({"relations": [{"source": "A", "target": "B", "relation_type": "sufficient"}]}),
        encoding="utf-8",
    )
    store = RelationStore(NesyConfig(security=SecurityConfig(allowed_roots=[str(allowed)])))

    result = await call_tool(
        LOAD_RELATIONS,
        {
            "source_type": "resource_uri",
            "resource_uri": relation_file.resolve().as_uri(),
            "check_contradictions": False,
        },
        store,
    )

    assert result.isError is False
    assert result.structuredContent["loaded_relations"] == 1
    assert store.list_relations()[0].source == "A"


@pytest.mark.asyncio
async def test_load_relations_resource_uri_rejects_unsupported_scheme(tmp_path: Path) -> None:
    store = RelationStore(NesyConfig(security=SecurityConfig(allowed_roots=[str(tmp_path)])))

    result = await call_tool(
        LOAD_RELATIONS,
        {
            "source_type": "resource_uri",
            "resource_uri": "https://example.test/relations.json",
            "check_contradictions": False,
        },
        store,
    )

    assert result.isError is True
    assert result.structuredContent["diagnostics"][0]["code"] == "LOAD_RELATIONS_FAILED"
    assert "not supported in v0.7" in result.structuredContent["diagnostics"][0]["message"]


@pytest.mark.asyncio
async def test_file_access_rejects_extension_and_symlink_escape(tmp_path: Path) -> None:
    allowed = tmp_path / "allowed"
    outside = tmp_path / "outside"
    allowed.mkdir()
    outside.mkdir()
    outside_file = outside / "relations.json"
    outside_file.write_text('{"relations":[]}', encoding="utf-8")
    symlink = allowed / "linked.json"
    symlink.symlink_to(outside_file)
    store = RelationStore(NesyConfig(security=SecurityConfig(allowed_roots=[str(allowed)])))

    bad_extension = await call_tool(
        EXPORT_RELATIONS,
        {"destination": "file", "path": str(allowed / "relations.txt")},
        store,
    )
    bad_symlink = await call_tool(
        LOAD_RELATIONS,
        {"source_type": "file", "path": str(symlink)},
        store,
    )

    assert bad_extension.isError is True
    assert bad_extension.structuredContent["diagnostics"][0]["code"] == "EXPORT_EXTENSION_MISMATCH"
    assert bad_symlink.isError is True
    assert "outside allowed_roots" in bad_symlink.structuredContent["diagnostics"][0]["message"]


@pytest.mark.asyncio
async def test_export_inline_too_large_returns_error() -> None:
    store = RelationStore()
    await call_tool(
        ASSERT_RELATIONS,
        {
            "relations": [
                {
                    "source": "A" * 512,
                    "target": "B" * 512,
                    "relation_type": "sufficient",
                }
            ],
            "check_contradictions": False,
        },
        store,
    )

    result = await call_tool(
        EXPORT_RELATIONS,
        {"destination": "inline", "max_inline_bytes": 1000},
        store,
    )

    assert result.isError is True
    assert result.structuredContent["diagnostics"][0]["code"] == "INLINE_EXPORT_TOO_LARGE"


@pytest.mark.asyncio
async def test_audit_records_mutating_tools(tmp_path: Path) -> None:
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    store = RelationStore(NesyConfig(security=SecurityConfig(allowed_roots=[str(allowed)])))

    await call_tool(
        ASSERT_RELATIONS,
        {
            "relations": [{"source": "A", "target": "B", "relation_type": "sufficient"}],
            "check_contradictions": False,
        },
        store,
    )
    await call_tool(
        ASSERT_EXCLUSIVE,
        {"groups": [{"group_id": "state", "members": ["B", "C"]}]},
        store,
    )
    await call_tool(
        EXPORT_RELATIONS,
        {"destination": "file", "path": str(allowed / "relations.json")},
        store,
    )

    entries = store.list_audit_entries()

    assert [entry["tool_name"] for entry in entries] == [
        ASSERT_RELATIONS,
        ASSERT_EXCLUSIVE,
        EXPORT_RELATIONS,
    ]
    assert all(entry["result_status"] == "ok" for entry in entries)


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
