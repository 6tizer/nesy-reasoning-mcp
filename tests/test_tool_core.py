import json

import pytest

from nesy_reasoning_mcp import tool_registry
from nesy_reasoning_mcp.config import LoggingConfig, NesyConfig, SecurityConfig
from nesy_reasoning_mcp.store import RelationStore
from nesy_reasoning_mcp.tools import (
    ASSERT_EXCLUSIVE,
    ASSERT_RELATIONS,
    CLASSIFY,
    CLEAR_RELATIONS,
    LIST_RELATIONS,
    call_tool,
)


class AuditFailureStore(RelationStore):
    def record_audit(self, **_kwargs) -> None:
        raise RuntimeError("audit backend unavailable")


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
    await call_tool(
        ASSERT_RELATIONS,
        {
            "relations": [{"source": "Existing", "target": "Graph", "relation_type": "sufficient"}],
            "check_contradictions": False,
        },
        store,
    )
    result = await call_tool(
        ASSERT_RELATIONS,
        {"relations": [{"source": "A", "target": "B", "relation_type": "bad"}]},
        store,
    )

    assert result.isError is True
    assert result.structuredContent["status"] == "error"
    assert result.structuredContent["diagnostics"][0]["code"] == "INPUT_VALIDATION_ERROR"
    assert result.structuredContent["graph_stats"]["relations"] == 1


@pytest.mark.asyncio
async def test_unknown_tool_returns_structured_error_result() -> None:
    store = RelationStore()
    result = await call_tool("nesy.nope", {}, store)

    assert result.isError is True
    assert result.structuredContent["status"] == "error"
    assert result.structuredContent["diagnostics"][0]["code"] == "UNKNOWN_TOOL"
    assert json.loads(result.content[0].text) == result.structuredContent


@pytest.mark.asyncio
async def test_runtime_failure_returns_structured_error_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = RelationStore()

    async def broken_handler(_arguments, _store):
        raise RuntimeError("handler exploded")

    monkeypatch.setattr(tool_registry, "assert_relations", broken_handler)

    result = await call_tool(ASSERT_RELATIONS, {"relations": []}, store)

    assert result.isError is True
    assert result.structuredContent["status"] == "error"
    assert result.structuredContent["diagnostics"][0]["code"] == "TOOL_RUNTIME_ERROR"
    assert result.structuredContent["diagnostics"][0]["message"] == "handler exploded"


@pytest.mark.asyncio
async def test_audit_failure_appends_warning_without_hiding_result() -> None:
    store = AuditFailureStore(
        NesyConfig(logging=LoggingConfig(audit_log=True)),
    )

    result = await call_tool(
        ASSERT_RELATIONS,
        {
            "relations": [{"source": "A", "target": "B", "relation_type": "sufficient"}],
            "check_contradictions": False,
        },
        store,
    )

    assert result.isError is False
    assert result.structuredContent["status"] == "ok"
    assert result.structuredContent["added"] == 1
    assert result.structuredContent["diagnostics"][0]["code"] == "AUDIT_LOG_FAILED"


@pytest.mark.asyncio
async def test_assert_relations_upsert_updates_same_id() -> None:
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
            ]
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
                },
                {"source": "D", "target": "E", "relation_type": "sufficient"},
            ],
            "mode": "upsert",
        },
        store,
    )

    assert result.isError is False
    assert result.structuredContent["added"] == 1
    assert result.structuredContent["updated"] == 1
    assert result.structuredContent["rejected"] == 0
    assert len(store.list_relations()) == 2
    assert {record.id: record.target for record in store.list_relations()}["rel_keep"] == "C"


@pytest.mark.asyncio
async def test_assert_relations_upsert_dry_run_does_not_change_store() -> None:
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
            ]
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
            "dry_run": True,
        },
        store,
    )

    assert result.isError is False
    assert result.structuredContent["added"] == 0
    assert result.structuredContent["updated"] == 1
    assert len(store.list_relations()) == 1
    assert store.list_relations()[0].target == "B"


@pytest.mark.asyncio
async def test_merge_equivalent_reports_normalization_without_synthetic_record() -> None:
    store = RelationStore()
    await call_tool(
        ASSERT_RELATIONS,
        {
            "relations": [
                {
                    "id": "rel_sufficient",
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
                    "id": "rel_necessary",
                    "source": "A",
                    "target": "B",
                    "relation_type": "necessary",
                }
            ],
            "check_contradictions": False,
            "merge_equivalent": True,
        },
        store,
    )
    classification = await call_tool(CLASSIFY, {"source": "A", "target": "B"}, store)

    assert result.structuredContent["diagnostics"][0]["code"] == ("MERGE_EQUIVALENT_NORMALIZED")
    assert result.structuredContent["diagnostics"][0]["level"] == "info"
    assert len(store.list_relations()) == 2
    assert classification.structuredContent["classification"] == "equivalent"


@pytest.mark.asyncio
async def test_merge_equivalent_false_skips_normalization_diagnostic() -> None:
    store = RelationStore()
    await call_tool(
        ASSERT_RELATIONS,
        {
            "relations": [
                {
                    "id": "rel_sufficient",
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
                    "id": "rel_necessary",
                    "source": "A",
                    "target": "B",
                    "relation_type": "necessary",
                }
            ],
            "check_contradictions": False,
            "merge_equivalent": False,
        },
        store,
    )

    assert result.structuredContent["diagnostics"] == []
    assert len(store.list_relations()) == 2


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
