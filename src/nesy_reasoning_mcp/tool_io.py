"""Import and export tool handlers."""

from __future__ import annotations

import json
from copy import deepcopy
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
    PropositionRecord,
    RelationRecord,
    RelationSetData,
    RelationType,
)
from nesy_reasoning_mcp.storage.common import _merge_propositions, _normalize_relation_identities
from nesy_reasoning_mcp.store import RelationStoreProtocol
from nesy_reasoning_mcp.tool_common import (
    _exclusive_group_dump,
    _exclusive_group_matches_filter,
    _groups_for_store,
    _independence_for_store,
    _independence_record_dump,
    _independence_record_matches_filter,
    _proposition_dump,
    _record_dump,
    _relations_for_store,
)


async def load_relations(arguments: dict[str, Any], store: RelationStoreProtocol) -> dict[str, Any]:
    """Handle `nesy.load_relations`."""
    try:
        migrated_arguments, migration_diagnostics = _migrate_load_arguments(arguments)
        payload = LoadRelationsInput.model_validate(migrated_arguments)
        data, source_trace, parse_diagnostics = _load_relation_set_data(payload, store)
    except (OSError, ValueError, ValidationError) as exc:
        return _load_error("LOAD_RELATIONS_FAILED", str(exc), store)
    diagnostics = [*migration_diagnostics, *parse_diagnostics]

    try:
        effective_propositions, _updated_propositions = _merge_propositions(
            store.list_propositions(),
            data.propositions,
        )
        normalized_data_relations = _normalize_relation_identities(
            data.relations,
            effective_propositions,
        )
    except ValueError as exc:
        return _load_error("LOAD_RELATIONS_FAILED", str(exc), store)

    incoming_relations = _relations_for_store(normalized_data_relations, payload.store_id)
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
        check_relations = [
            *_normalize_relation_identities(stored_relations, effective_propositions),
            *incoming_relations,
        ]
        contradictions, _context_separated = find_exclusive_contradictions(
            check_relations,
            [*stored_groups, *incoming_groups],
            context_filter=ContextFilter(store_id=payload.store_id),
            max_depth=8,
            propositions=effective_propositions,
        )

    try:
        loaded_relations, loaded_groups, updated_relations, updated_groups = store.import_records(
            incoming_relations,
            incoming_groups,
            incoming_independence,
            data.propositions,
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
        "loaded_propositions": len(data.propositions),
        "updated_relations": updated_relations,
        "updated_exclusive_groups": updated_groups,
        "rejected": 0,
        "conflicts": contradictions,
        "validate_only": payload.validate_only,
        "diagnostics": [item.model_dump(mode="json") for item in diagnostics],
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


async def export_relations(
    arguments: dict[str, Any],
    store: RelationStoreProtocol,
) -> dict[str, Any]:
    """Handle `nesy.export_relations`."""
    payload = ExportRelationsInput.model_validate(arguments)
    relations = store.list_relations(payload.filter, limit=None)
    propositions = store.list_propositions()
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
        propositions,
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
            "proposition_count": len(propositions),
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
        "proposition_count": len(propositions),
        "data": None,
        "path": str(real_path),
        "bytes": byte_count,
        "diagnostics": [],
        "trace": [f"Exported {len(relations)} relation(s) to {real_path}."],
        "graph_stats": store.graph_stats().model_dump(mode="json"),
    }


def _load_relation_set_data(
    payload: LoadRelationsInput,
    store: RelationStoreProtocol,
) -> tuple[RelationSetData, list[str], list[Diagnostic]]:
    if payload.source_type == LoadSourceType.INLINE:
        if payload.data is None:
            raise ValueError("data is required when source_type=inline")
        return payload.data, ["Read inline relation set."], []

    if payload.source_type == LoadSourceType.FILE:
        if payload.path is None:
            raise ValueError("path is required when source_type=file")
        real_path, text = read_allowed_relation_file(payload.path, store.config)
        data, diagnostics = _parse_relation_set_text(text, real_path.suffix)
        return data, [f"Read relation set from {real_path}."], diagnostics

    if payload.source_type == LoadSourceType.RESOURCE_URI:
        if payload.resource_uri is None:
            raise ValueError("resource_uri is required when source_type=resource_uri")
        real_path, text = _read_resource_uri(payload.resource_uri, store)
        data, diagnostics = _parse_relation_set_text(text, real_path.suffix)
        return data, [f"Read relation set from resource URI {payload.resource_uri}."], diagnostics

    raise ValueError(f"unsupported source_type: {payload.source_type.value}")


def _read_resource_uri(resource_uri: str, store: RelationStoreProtocol) -> tuple[Path, str]:
    parsed = urlparse(resource_uri)
    if parsed.scheme != "file":
        raise ValueError(f"resource_uri supports local file:// URIs only: {parsed.scheme}")
    if parsed.netloc not in {"", "localhost"}:
        raise ValueError("file resource_uri host must be empty or localhost")
    path = unquote(parsed.path)
    if not path:
        raise ValueError("file resource_uri is missing a path")
    return read_allowed_relation_file(path, store.config)


def _parse_relation_set_text(text: str, suffix: str) -> tuple[RelationSetData, list[Diagnostic]]:
    if suffix == ".json":
        migrated, count = _migrate_relation_set_payload(json.loads(text))
        return RelationSetData.model_validate(migrated), _legacy_diagnostics(count)
    if suffix == ".jsonl":
        relations = []
        exclusive_groups = []
        independence_records = []
        propositions = []
        context_metadata: dict[str, Any] = {}
        migrated_count = 0
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
            elif item_type == "proposition":
                propositions.append(record)
            elif item_type == "independence_record" or record.get("relation") == "independent_of":
                independence_records.append(record)
            elif item_type == "exclusive_group" or "members" in record:
                exclusive_groups.append(record)
            elif item_type == "relation" or _looks_like_relation_record(record):
                migrated_record, count = _migrate_relation_record_payload(record)
                migrated_count += count
                relations.append(migrated_record)
            else:
                raise ValueError(
                    f"line {line_number} is not a relation, exclusive group, "
                    "independence record, proposition, or context metadata"
                )
        data = RelationSetData.model_validate(
            {
                "relations": relations,
                "exclusive_groups": exclusive_groups,
                "independence_records": independence_records,
                "propositions": propositions,
                "context_metadata": context_metadata,
            }
        )
        return data, _legacy_diagnostics(migrated_count)
    raise ValueError("only .json and .jsonl files are allowed")


def _migrate_load_arguments(arguments: dict[str, Any]) -> tuple[dict[str, Any], list[Diagnostic]]:
    migrated = deepcopy(arguments)
    if migrated.get("source_type") != LoadSourceType.INLINE.value or "data" not in migrated:
        return migrated, []
    migrated_data, count = _migrate_relation_set_payload(migrated["data"])
    migrated["data"] = migrated_data
    return migrated, _legacy_diagnostics(count)


def _migrate_relation_set_payload(data: Any) -> tuple[Any, int]:
    if not isinstance(data, dict):
        return data, 0
    migrated = deepcopy(data)
    relations = migrated.get("relations")
    if not isinstance(relations, list):
        return migrated, 0
    migrated_count = 0
    migrated_relations = []
    for relation in relations:
        migrated_relation, count = _migrate_relation_record_payload(relation)
        migrated_count += count
        migrated_relations.append(migrated_relation)
    migrated["relations"] = migrated_relations
    return migrated, migrated_count


def _migrate_relation_record_payload(record: Any) -> tuple[Any, int]:
    if not isinstance(record, dict):
        return record, 0
    migrated = deepcopy(record)
    count = 0
    for old_key, new_key in (("from", "source"), ("to", "target"), ("type", "relation_type")):
        if old_key not in migrated:
            continue
        if old_key == "type" and migrated[old_key] not in {item.value for item in RelationType}:
            continue
        if new_key in migrated and migrated[new_key] != migrated[old_key]:
            raise ValueError(f"legacy field conflict: {old_key} differs from {new_key}")
        if new_key not in migrated:
            migrated[new_key] = migrated[old_key]
            count += 1
        del migrated[old_key]

    if "temporal_delay" in migrated:
        temporal = migrated.get("temporal")
        if temporal is None:
            temporal = {}
        if not isinstance(temporal, dict):
            raise ValueError("legacy field conflict: temporal is not an object")
        if "delay" in temporal and temporal["delay"] != migrated["temporal_delay"]:
            raise ValueError("legacy field conflict: temporal_delay differs from temporal.delay")
        if "delay" not in temporal:
            temporal["delay"] = migrated["temporal_delay"]
            count += 1
        migrated["temporal"] = temporal
        del migrated["temporal_delay"]

    return migrated, count


def _looks_like_relation_record(record: Any) -> bool:
    if not isinstance(record, dict):
        return False
    if "relation_type" in record:
        return True
    if record.get("type") in {item.value for item in RelationType}:
        return True
    return any(key in record for key in ("from", "to", "temporal_delay"))


def _legacy_diagnostics(count: int) -> list[Diagnostic]:
    if count == 0:
        return []
    return [
        Diagnostic(
            level="info",
            code="LEGACY_FIELDS_MIGRATED",
            message=f"Migrated {count} legacy relation field(s) at load boundary.",
        )
    ]


def _relation_set_export(
    relations: list[RelationRecord],
    exclusive_groups: list[ExclusiveGroupRecord],
    independence_records: list[IndependenceRecord],
    propositions: list[PropositionRecord],
    *,
    context_metadata: dict[str, Any],
    include_metadata: bool,
) -> dict[str, Any]:
    relation_items = [_record_dump(record) for record in relations]
    group_items = [_exclusive_group_dump(group) for group in exclusive_groups]
    independence_items = [_independence_record_dump(record) for record in independence_records]
    proposition_items = [_proposition_dump(record) for record in propositions]
    if not include_metadata:
        for item in relation_items:
            item.pop("metadata", None)
            item.pop("provenance", None)
        for item in group_items:
            item.pop("metadata", None)
        for item in independence_items:
            item.pop("metadata", None)
        for item in proposition_items:
            item.pop("metadata", None)
    data = {
        "version": "2.0",
        "relations": relation_items,
        "exclusive_groups": group_items,
        "independence_records": independence_items,
        "context_metadata": context_metadata,
    }
    if proposition_items:
        data["propositions"] = proposition_items
    return data


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
    lines: list[str] = []
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
            {"type": "proposition", "record": proposition},
            ensure_ascii=False,
            sort_keys=True,
        )
        for proposition in data.get("propositions", [])
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


def _load_error(code: str, message: str, store: RelationStoreProtocol) -> dict[str, Any]:
    diagnostic = Diagnostic(level="error", code=code, message=message)
    return {
        "status": "error",
        "loaded_relations": 0,
        "loaded_exclusive_groups": 0,
        "loaded_propositions": 0,
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
    store: RelationStoreProtocol,
    export_format: ExportFormat,
) -> dict[str, Any]:
    diagnostic = Diagnostic(level="error", code=code, message=message)
    return {
        "status": "error",
        "format": export_format.value,
        "relation_count": 0,
        "exclusive_group_count": 0,
        "proposition_count": 0,
        "data": None,
        "path": None,
        "bytes": 0,
        "diagnostics": [diagnostic.model_dump(mode="json")],
        "trace": ["Rejected relation export."],
        "graph_stats": store.graph_stats().model_dump(mode="json"),
    }
