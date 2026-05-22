"""Import and export tool handlers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

from pydantic import ValidationError

from nesy_reasoning_mcp.file_access import read_allowed_relation_file, write_allowed_relation_file
from nesy_reasoning_mcp.reasoning import find_exclusive_contradictions
from nesy_reasoning_mcp.schemas import (
    ContextFilter,
    Diagnostic,
    ExclusiveGroupRecord,
    ExportDestination,
    ExportFormat,
    ExportRelationsInput,
    IndependenceRecord,
    LoadRelationsInput,
    LoadSourceType,
    RelationRecord,
    RelationSetData,
)
from nesy_reasoning_mcp.store import RelationStore
from nesy_reasoning_mcp.tool_common import (
    _exclusive_group_dump,
    _exclusive_group_matches_filter,
    _groups_for_store,
    _independence_for_store,
    _independence_record_dump,
    _independence_record_matches_filter,
    _record_dump,
    _relations_for_store,
)


async def load_relations(arguments: dict[str, Any], store: RelationStore) -> dict[str, Any]:
    """Handle `nesy.load_relations`."""
    payload = LoadRelationsInput.model_validate(arguments)

    try:
        data, source_trace = _load_relation_set_data(payload, store)
    except (OSError, ValueError, ValidationError) as exc:
        return _load_error("LOAD_RELATIONS_FAILED", str(exc), store)

    incoming_relations = _relations_for_store(data.relations, payload.store_id)
    incoming_groups = _groups_for_store(data.exclusive_groups, payload.store_id)
    incoming_independence = _independence_for_store(data.independence_records, payload.store_id)
    contradictions: list[dict[str, Any]] = []
    if payload.check_contradictions:
        stored_relations = store.list_relations()
        stored_groups = store.list_exclusive_groups()
        if payload.mode.value == "replace_store":
            stored_relations = [
                relation for relation in stored_relations if relation.store_id != payload.store_id
            ]
            stored_groups = [group for group in stored_groups if group.store_id != payload.store_id]
        check_relations = [*stored_relations, *incoming_relations]
        contradictions, _context_separated = find_exclusive_contradictions(
            check_relations,
            [*stored_groups, *incoming_groups],
            context_filter=ContextFilter(store_id=payload.store_id),
            max_depth=8,
        )

    try:
        loaded_relations, loaded_groups, updated_relations, updated_groups = store.import_records(
            incoming_relations,
            incoming_groups,
            incoming_independence,
            mode=payload.mode.value,
            store_id=payload.store_id,
            context_metadata=data.context_metadata,
            dry_run=payload.validate_only,
        )
    except Exception as exc:
        return _load_error("LOAD_RELATIONS_FAILED", str(exc), store)
    return {
        "status": "warning" if contradictions else "ok",
        "loaded_relations": loaded_relations,
        "loaded_exclusive_groups": loaded_groups,
        "updated_relations": updated_relations,
        "updated_exclusive_groups": updated_groups,
        "rejected": 0,
        "conflicts": contradictions,
        "validate_only": payload.validate_only,
        "diagnostics": [],
        "trace": [
            *source_trace,
            (
                "Validated relation set without changing store."
                if payload.validate_only
                else f"Loaded relation set with mode={payload.mode.value}."
            ),
        ],
        "graph_stats": store.graph_stats().model_dump(mode="json"),
    }


async def export_relations(arguments: dict[str, Any], store: RelationStore) -> dict[str, Any]:
    """Handle `nesy.export_relations`."""
    payload = ExportRelationsInput.model_validate(arguments)
    relations = store.list_relations(payload.filter, limit=None)
    independence_records = [
        record
        for record in store.list_independence_records()
        if _independence_record_matches_filter(record, payload.filter)
    ]
    exclusive_groups = (
        [
            group
            for group in store.list_exclusive_groups()
            if _exclusive_group_matches_filter(group, payload.filter)
        ]
        if payload.include_exclusive_groups
        else []
    )
    exported = _relation_set_export(
        relations,
        exclusive_groups,
        independence_records,
        context_metadata=(
            _context_metadata_for_export(
                store.context_metadata(),
                relations,
                exclusive_groups,
                independence_records,
            )
            if payload.include_metadata
            else {}
        ),
        include_metadata=payload.include_metadata,
    )
    text = _serialize_relation_set(exported, payload.format)
    byte_count = len(text.encode("utf-8"))

    if payload.destination == ExportDestination.INLINE:
        if byte_count > payload.max_inline_bytes:
            return _export_error(
                "INLINE_EXPORT_TOO_LARGE",
                "Inline export exceeds max_inline_bytes.",
                store,
                payload.format,
            )
        return {
            "status": "ok",
            "format": payload.format.value,
            "relation_count": len(relations),
            "exclusive_group_count": len(exclusive_groups),
            "data": exported if payload.format == ExportFormat.JSON else text,
            "path": None,
            "bytes": byte_count,
            "diagnostics": [],
            "trace": [f"Exported {len(relations)} relation(s) inline."],
            "graph_stats": store.graph_stats().model_dump(mode="json"),
        }

    if payload.path is None:
        return _export_error(
            "EXPORT_PATH_REQUIRED",
            "path is required when destination=file.",
            store,
            payload.format,
        )
    if Path(payload.path).expanduser().suffix != f".{payload.format.value}":
        return _export_error(
            "EXPORT_EXTENSION_MISMATCH",
            "Export path suffix must match requested format.",
            store,
            payload.format,
        )

    try:
        real_path = write_allowed_relation_file(payload.path, store.config, text)
    except (OSError, ValueError) as exc:
        return _export_error("EXPORT_RELATIONS_FAILED", str(exc), store, payload.format)

    return {
        "status": "ok",
        "format": payload.format.value,
        "relation_count": len(relations),
        "exclusive_group_count": len(exclusive_groups),
        "data": None,
        "path": str(real_path),
        "bytes": byte_count,
        "diagnostics": [],
        "trace": [f"Exported {len(relations)} relation(s) to {real_path}."],
        "graph_stats": store.graph_stats().model_dump(mode="json"),
    }


def _load_relation_set_data(
    payload: LoadRelationsInput,
    store: RelationStore,
) -> tuple[RelationSetData, list[str]]:
    if payload.source_type == LoadSourceType.INLINE:
        if payload.data is None:
            raise ValueError("data is required when source_type=inline")
        return payload.data, ["Read inline relation set."]

    if payload.source_type == LoadSourceType.FILE:
        if payload.path is None:
            raise ValueError("path is required when source_type=file")
        real_path, text = read_allowed_relation_file(payload.path, store.config)
        return _parse_relation_set_text(text, real_path.suffix), [
            f"Read relation set from {real_path}."
        ]

    if payload.source_type == LoadSourceType.RESOURCE_URI:
        if payload.resource_uri is None:
            raise ValueError("resource_uri is required when source_type=resource_uri")
        real_path, text = _read_resource_uri(payload.resource_uri, store)
        return _parse_relation_set_text(text, real_path.suffix), [
            f"Read relation set from resource URI {payload.resource_uri}."
        ]

    raise ValueError(f"unsupported source_type: {payload.source_type.value}")


def _read_resource_uri(resource_uri: str, store: RelationStore) -> tuple[Path, str]:
    parsed = urlparse(resource_uri)
    if parsed.scheme != "file":
        raise ValueError(f"resource_uri scheme is not supported in v0.7: {parsed.scheme}")
    if parsed.netloc not in {"", "localhost"}:
        raise ValueError("file resource_uri host must be empty or localhost")
    path = unquote(parsed.path)
    if not path:
        raise ValueError("file resource_uri is missing a path")
    return read_allowed_relation_file(path, store.config)


def _parse_relation_set_text(text: str, suffix: str) -> RelationSetData:
    if suffix == ".json":
        return RelationSetData.model_validate(json.loads(text))
    if suffix == ".jsonl":
        relations = []
        exclusive_groups = []
        independence_records = []
        context_metadata: dict[str, Any] = {}
        for line_number, line in enumerate(text.splitlines(), start=1):
            stripped = line.strip()
            if not stripped:
                continue
            item = json.loads(stripped)
            if not isinstance(item, dict):
                raise ValueError(f"line {line_number} must be a JSON object")
            item_type = item.get("type")
            record = item.get("record", item)
            if not isinstance(record, dict):
                raise ValueError(f"line {line_number} record must be a JSON object")
            if item_type == "context_metadata":
                context_id = item.get("context_id")
                if not context_id:
                    raise ValueError(f"line {line_number} context_metadata is missing context_id")
                context_metadata[str(context_id)] = record
            elif item_type == "independence_record" or record.get("relation") == "independent_of":
                independence_records.append(record)
            elif item_type == "exclusive_group" or "members" in record:
                exclusive_groups.append(record)
            elif item_type == "relation" or "relation_type" in record:
                relations.append(record)
            else:
                raise ValueError(
                    f"line {line_number} is not a relation, exclusive group, "
                    "independence record, or context metadata"
                )
        return RelationSetData.model_validate(
            {
                "relations": relations,
                "exclusive_groups": exclusive_groups,
                "independence_records": independence_records,
                "context_metadata": context_metadata,
            }
        )
    raise ValueError("only .json and .jsonl files are allowed")


def _relation_set_export(
    relations: list[RelationRecord],
    exclusive_groups: list[ExclusiveGroupRecord],
    independence_records: list[IndependenceRecord],
    *,
    context_metadata: dict[str, Any],
    include_metadata: bool,
) -> dict[str, Any]:
    relation_items = [_record_dump(record) for record in relations]
    group_items = [_exclusive_group_dump(group) for group in exclusive_groups]
    independence_items = [_independence_record_dump(record) for record in independence_records]
    if not include_metadata:
        for item in relation_items:
            item.pop("metadata", None)
            item.pop("provenance", None)
        for item in group_items:
            item.pop("metadata", None)
        for item in independence_items:
            item.pop("metadata", None)
    return {
        "version": "2.0",
        "relations": relation_items,
        "exclusive_groups": group_items,
        "independence_records": independence_items,
        "context_metadata": context_metadata,
    }


def _context_metadata_for_export(
    context_metadata: dict[str, Any],
    relations: list[RelationRecord],
    exclusive_groups: list[ExclusiveGroupRecord],
    independence_records: list[IndependenceRecord],
) -> dict[str, Any]:
    context_ids = {relation.context_id for relation in relations}
    context_ids.update(group.context_id for group in exclusive_groups)
    context_ids.update(record.context_id for record in independence_records)
    if not context_ids and not relations and not exclusive_groups and not independence_records:
        return {}
    return {
        context_id: context_metadata[context_id]
        for context_id in sorted(context_ids)
        if context_id in context_metadata
    }


def _serialize_relation_set(data: dict[str, Any], export_format: ExportFormat) -> str:
    if export_format == ExportFormat.JSON:
        return json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True)
    lines = []
    lines.extend(
        json.dumps({"type": "relation", "record": relation}, ensure_ascii=False, sort_keys=True)
        for relation in data["relations"]
    )
    lines.extend(
        json.dumps(
            {"type": "exclusive_group", "record": group},
            ensure_ascii=False,
            sort_keys=True,
        )
        for group in data["exclusive_groups"]
    )
    lines.extend(
        json.dumps(
            {"type": "independence_record", "record": record},
            ensure_ascii=False,
            sort_keys=True,
        )
        for record in data["independence_records"]
    )
    lines.extend(
        json.dumps(
            {
                "type": "context_metadata",
                "context_id": context_id,
                "record": metadata,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        for context_id, metadata in data.get("context_metadata", {}).items()
    )
    return "\n".join(lines) + ("\n" if lines else "")


def _load_error(code: str, message: str, store: RelationStore) -> dict[str, Any]:
    diagnostic = Diagnostic(level="error", code=code, message=message)
    return {
        "status": "error",
        "loaded_relations": 0,
        "loaded_exclusive_groups": 0,
        "updated_relations": 0,
        "updated_exclusive_groups": 0,
        "rejected": 0,
        "conflicts": [],
        "validate_only": False,
        "diagnostics": [diagnostic.model_dump(mode="json")],
        "trace": ["Rejected relation load."],
        "graph_stats": store.graph_stats().model_dump(mode="json"),
    }


def _export_error(
    code: str,
    message: str,
    store: RelationStore,
    export_format: ExportFormat,
) -> dict[str, Any]:
    diagnostic = Diagnostic(level="error", code=code, message=message)
    return {
        "status": "error",
        "format": export_format.value,
        "relation_count": 0,
        "exclusive_group_count": 0,
        "data": None,
        "path": None,
        "bytes": 0,
        "diagnostics": [diagnostic.model_dump(mode="json")],
        "trace": ["Rejected relation export."],
        "graph_stats": store.graph_stats().model_dump(mode="json"),
    }
