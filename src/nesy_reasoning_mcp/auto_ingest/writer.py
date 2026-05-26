"""Safe write helper for Agent SDK ingestion."""

from __future__ import annotations

from typing import Any

from nesy_reasoning_mcp.auto_ingest.policy import WRITE_MODE_TOOL_ALLOWLIST
from nesy_reasoning_mcp.schemas import Diagnostic, RelationInput, RelationRecord
from nesy_reasoning_mcp.storage.common import _normalize_relation_identities
from nesy_reasoning_mcp.store import RelationStoreProtocol
from nesy_reasoning_mcp.tool_names import ASSERT_RELATIONS
from nesy_reasoning_mcp.tool_registry import call_tool


async def write_approved_relations(
    *,
    relations: list[RelationInput],
    store: RelationStoreProtocol,
) -> tuple[list[str], list[Diagnostic], dict[str, Any]]:
    """Persist gate-approved relations with contradiction rejection enabled."""
    if ASSERT_RELATIONS not in WRITE_MODE_TOOL_ALLOWLIST:
        raise RuntimeError("write-mode policy does not allow relation assertion")
    if not relations:
        return [], [], {}

    normalized_relations = _normalize_relation_identities(relations, store.list_propositions())
    prepared_records, _updated = store.assert_relations(
        normalized_relations,
        mode="append",
        dry_run=True,
    )
    dedupe = _dedupe_relations(prepared_records, store.list_relations())
    if not dedupe.new_relations:
        structured = {
            "status": "ok",
            "added": 0,
            "updated": 0,
            "rejected": 0,
            "relation_ids": dedupe.ordered_relation_ids,
            "deduplicated_count": dedupe.deduplicated_count,
            "deduplicated_relation_ids": dedupe.deduplicated_relation_ids,
            "diagnostics": [],
            "trace": ["Skipped duplicate approved relation assertion(s)."],
            "graph_stats": store.graph_stats().model_dump(mode="json"),
        }
        return dedupe.ordered_relation_ids, [], structured

    result = await call_tool(
        ASSERT_RELATIONS,
        {
            "relations": [_relation_input_dump(relation) for relation in dedupe.new_relations],
            "on_contradiction": "reject",
            "check_contradictions": True,
        },
        store,
    )
    structured = dict(result.structuredContent or {})
    diagnostics = _diagnostics_from_structured(structured)
    if result.isError or structured.get("status") == "error":
        return [], diagnostics, structured
    raw_relation_ids = structured.get("relation_ids", [])
    new_relation_ids = (
        [str(item) for item in raw_relation_ids] if isinstance(raw_relation_ids, list) else []
    )
    relation_ids = _ordered_relation_ids(
        dedupe.ordered_relation_ids,
        new_relation_ids,
        dedupe.new_placeholder_ids,
    )
    structured["relation_ids"] = relation_ids
    structured["deduplicated_count"] = dedupe.deduplicated_count
    structured["deduplicated_relation_ids"] = dedupe.deduplicated_relation_ids
    return relation_ids, diagnostics, structured


def _diagnostics_from_structured(structured: dict[str, Any]) -> list[Diagnostic]:
    diagnostics: list[Diagnostic] = []
    for item in structured.get("diagnostics", []):
        if isinstance(item, dict):
            diagnostics.append(Diagnostic.model_validate(item))
    return diagnostics


def _relation_input_dump(relation: RelationInput) -> dict[str, Any]:
    return RelationInput.model_validate(
        relation.model_dump(
            mode="json",
            exclude_none=True,
            include=set(RelationInput.model_fields),
        )
    ).model_dump(mode="json", exclude_none=True)


class _DedupePlan:
    def __init__(
        self,
        *,
        new_relations: list[RelationInput],
        ordered_relation_ids: list[str],
        deduplicated_relation_ids: list[str],
        new_placeholder_ids: list[str],
    ) -> None:
        self.new_relations = new_relations
        self.ordered_relation_ids = ordered_relation_ids
        self.deduplicated_relation_ids = deduplicated_relation_ids
        self.new_placeholder_ids = new_placeholder_ids

    @property
    def deduplicated_count(self) -> int:
        return len(self.deduplicated_relation_ids)


def _dedupe_relations(
    prepared_records: list[RelationRecord],
    existing_records: list[RelationRecord],
) -> _DedupePlan:
    existing_by_key: dict[tuple[str, str, str, str, str], RelationRecord] = {}
    for record in existing_records:
        for key in _dedupe_keys(record):
            existing_by_key.setdefault(key, record)
    batch_by_key: dict[tuple[str, str, str, str, str], str] = {}
    new_relations: list[RelationInput] = []
    new_placeholder_ids: list[str] = []
    ordered_relation_ids: list[str] = []
    deduplicated_relation_ids: list[str] = []
    for record in prepared_records:
        keys = _dedupe_keys(record)
        existing = next(
            (existing_by_key[key] for key in keys if key in existing_by_key),
            None,
        )
        if existing is not None:
            ordered_relation_ids.append(existing.id)
            deduplicated_relation_ids.append(existing.id)
            continue
        batch_relation_id = next(
            (batch_by_key[key] for key in keys if key in batch_by_key),
            None,
        )
        if batch_relation_id is not None:
            ordered_relation_ids.append(batch_relation_id)
            deduplicated_relation_ids.append(batch_relation_id)
            continue
        for key in keys:
            batch_by_key.setdefault(key, record.id)
        ordered_relation_ids.append(record.id)
        new_relations.append(record)
        new_placeholder_ids.append(record.id)
    return _DedupePlan(
        new_relations=new_relations,
        ordered_relation_ids=ordered_relation_ids,
        deduplicated_relation_ids=deduplicated_relation_ids,
        new_placeholder_ids=new_placeholder_ids,
    )


def _dedupe_key(relation: RelationInput) -> tuple[str, str, str, str, str]:
    return (
        relation.canonical_source,
        relation.canonical_target,
        relation.relation_type.value,
        relation.context_id,
        relation.store_id,
    )


def _dedupe_keys(relation: RelationInput) -> list[tuple[str, str, str, str, str]]:
    canonical_key = _dedupe_key(relation)
    label_key = (
        relation.source,
        relation.target,
        relation.relation_type.value,
        relation.context_id,
        relation.store_id,
    )
    return list(dict.fromkeys([canonical_key, label_key]))


def _ordered_relation_ids(
    planned_relation_ids: list[str],
    new_relation_ids: list[str],
    new_placeholder_ids: list[str],
) -> list[str]:
    replacements = iter(new_relation_ids)
    replaced: dict[str, str] = {}
    placeholder_set = set(new_placeholder_ids)
    ordered: list[str] = []
    for planned_id in planned_relation_ids:
        if planned_id in placeholder_set and planned_id not in replaced:
            replaced[planned_id] = next(replacements, planned_id)
        ordered.append(replaced.get(planned_id, planned_id))
    return ordered
