import json
from pathlib import Path

import pytest

from nesy_reasoning_mcp.config import NesyConfig, SecurityConfig
from nesy_reasoning_mcp.store import RelationStore
from nesy_reasoning_mcp.tools import (
    ASSERT_EXCLUSIVE,
    ASSERT_RELATIONS,
    EXPORT_RELATIONS,
    LOAD_RELATIONS,
    call_tool,
)


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
async def test_file_access_rejects_hidden_paths_by_default(tmp_path: Path) -> None:
    allowed = tmp_path / "allowed"
    hidden_dir = allowed / ".secret"
    hidden_dir.mkdir(parents=True)
    hidden_file = allowed / ".relations.json"
    hidden_nested_file = hidden_dir / "relations.json"
    hidden_file.write_text('{"relations":[]}', encoding="utf-8")
    hidden_nested_file.write_text('{"relations":[]}', encoding="utf-8")
    store = RelationStore(NesyConfig(security=SecurityConfig(allowed_roots=[str(allowed)])))

    hidden_load = await call_tool(
        LOAD_RELATIONS,
        {"source_type": "file", "path": str(hidden_file)},
        store,
    )
    hidden_nested_load = await call_tool(
        LOAD_RELATIONS,
        {"source_type": "file", "path": str(hidden_nested_file)},
        store,
    )
    hidden_resource = await call_tool(
        LOAD_RELATIONS,
        {"source_type": "resource_uri", "resource_uri": hidden_nested_file.resolve().as_uri()},
        store,
    )
    hidden_export_path = allowed / ".export.json"
    hidden_export = await call_tool(
        EXPORT_RELATIONS,
        {"destination": "file", "path": str(hidden_export_path)},
        store,
    )

    assert hidden_load.isError is True
    assert hidden_load.structuredContent["diagnostics"][0]["code"] == "LOAD_RELATIONS_FAILED"
    assert (
        "hidden relation paths blocked unless configured"
        in hidden_load.structuredContent["diagnostics"][0]["message"]
    )
    assert hidden_nested_load.isError is True
    assert hidden_resource.isError is True
    assert hidden_export.isError is True
    assert hidden_export.structuredContent["diagnostics"][0]["code"] == "EXPORT_RELATIONS_FAILED"
    assert not hidden_export_path.exists()


@pytest.mark.asyncio
async def test_file_access_allows_hidden_paths_when_configured(tmp_path: Path) -> None:
    allowed = tmp_path / "allowed"
    hidden_dir = allowed / ".secret"
    hidden_dir.mkdir(parents=True)
    hidden_file = allowed / ".relations.json"
    hidden_nested_file = hidden_dir / "relations.json"
    hidden_file.write_text(
        json.dumps({"relations": [{"source": "A", "target": "B", "relation_type": "sufficient"}]}),
        encoding="utf-8",
    )
    hidden_nested_file.write_text(
        json.dumps({"relations": [{"source": "C", "target": "D", "relation_type": "sufficient"}]}),
        encoding="utf-8",
    )
    config = NesyConfig(
        security=SecurityConfig(
            allowed_roots=[str(allowed)],
            allow_hidden_relation_paths=True,
        )
    )
    store = RelationStore(config)

    hidden_load = await call_tool(
        LOAD_RELATIONS,
        {"source_type": "file", "path": str(hidden_file), "check_contradictions": False},
        store,
    )
    hidden_nested_load = await call_tool(
        LOAD_RELATIONS,
        {"source_type": "file", "path": str(hidden_nested_file), "check_contradictions": False},
        store,
    )
    hidden_resource = await call_tool(
        LOAD_RELATIONS,
        {
            "source_type": "resource_uri",
            "resource_uri": hidden_nested_file.resolve().as_uri(),
            "check_contradictions": False,
        },
        store,
    )
    hidden_export_path = allowed / ".export.json"
    hidden_export = await call_tool(
        EXPORT_RELATIONS,
        {"destination": "file", "path": str(hidden_export_path)},
        store,
    )

    assert hidden_load.isError is False
    assert hidden_load.structuredContent["loaded_relations"] == 1
    assert hidden_nested_load.isError is False
    assert hidden_nested_load.structuredContent["loaded_relations"] == 1
    assert hidden_resource.isError is False
    assert hidden_resource.structuredContent["loaded_relations"] == 1
    assert hidden_export.isError is False
    assert hidden_export_path.exists()


@pytest.mark.asyncio
async def test_file_access_allows_non_hidden_child_under_hidden_allowed_root(
    tmp_path: Path,
) -> None:
    hidden_root = tmp_path / ".nesy-reasoning" / "relation_sets"
    hidden_root.mkdir(parents=True)
    relation_file = hidden_root / "relations.json"
    relation_file.write_text(
        json.dumps({"relations": [{"source": "A", "target": "B", "relation_type": "sufficient"}]}),
        encoding="utf-8",
    )
    store = RelationStore(NesyConfig(security=SecurityConfig(allowed_roots=[str(hidden_root)])))

    hidden_root_load = await call_tool(
        LOAD_RELATIONS,
        {"source_type": "file", "path": str(relation_file), "check_contradictions": False},
        store,
    )

    assert hidden_root_load.isError is False
    assert hidden_root_load.structuredContent["loaded_relations"] == 1


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
