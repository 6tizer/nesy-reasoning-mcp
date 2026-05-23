"""Shared helpers for tool handlers."""

from __future__ import annotations

from typing import Any

from nesy_reasoning_mcp.schemas import (
    CheckContradictionsInput,
    ContextFilter,
    ContradictionMode,
    ExclusiveGroupRecord,
    IndependenceRecord,
    PropositionRecord,
    RelationFilter,
    RelationRecord,
)
from nesy_reasoning_mcp.store import RelationStoreProtocol
from nesy_reasoning_mcp.tool_names import (
    ASSERT_EXCLUSIVE,
    ASSERT_RELATIONS,
    CLEAR_RELATIONS,
    EXPORT_RELATIONS,
    LOAD_RELATIONS,
)


def _exclusive_group_matches_filter(
    group: ExclusiveGroupRecord,
    relation_filter: RelationFilter,
) -> bool:
    if relation_filter.context_id is not None and group.context_id != relation_filter.context_id:
        return False
    if relation_filter.store_id is not None and group.store_id != relation_filter.store_id:
        return False
    return not (
        relation_filter.domain is not None
        and group.metadata.get("domain") != relation_filter.domain
    )


def _exclusive_group_compatible_with_context_filter(
    group: ExclusiveGroupRecord,
    context_filter: ContextFilter,
) -> bool:
    if context_filter.store_id and group.store_id != context_filter.store_id:
        return False
    return not (
        context_filter.context_id
        and group.scope.value == "same_context"
        and group.context_id != context_filter.context_id
    )


def _independence_record_matches_filter(
    record: IndependenceRecord,
    relation_filter: RelationFilter,
) -> bool:
    pair = {record.left, record.right}
    if relation_filter.source is not None and relation_filter.source not in pair:
        return False
    if relation_filter.target is not None and relation_filter.target not in pair:
        return False
    if relation_filter.context_id is not None and record.context_id != relation_filter.context_id:
        return False
    if relation_filter.store_id is not None and record.store_id != relation_filter.store_id:
        return False
    return not (
        relation_filter.domain is not None
        and record.metadata.get("domain") != relation_filter.domain
    )


def _relations_for_store(
    relations: list[RelationRecord],
    store_id: str,
) -> list[RelationRecord]:
    return [relation.model_copy(update={"store_id": store_id}) for relation in relations]


def _groups_for_store(
    groups: list[ExclusiveGroupRecord],
    store_id: str,
) -> list[ExclusiveGroupRecord]:
    return [group.model_copy(update={"store_id": store_id}) for group in groups]


def _independence_for_store(
    records: list[IndependenceRecord],
    store_id: str,
) -> list[IndependenceRecord]:
    return [record.model_copy(update={"store_id": store_id}) for record in records]


def _record_audit_if_needed(
    name: str,
    arguments: dict[str, Any],
    structured: dict[str, Any],
    store: RelationStoreProtocol,
) -> None:
    if not _should_audit(name, arguments):
        return
    store.record_audit(
        event_type="tool_call",
        tool_name=name,
        arguments=arguments,
        result_status=str(structured.get("status", "unknown")),
        metadata={"is_error": structured.get("status") == "error"},
    )


def _should_audit(name: str, arguments: dict[str, Any]) -> bool:
    if name == ASSERT_RELATIONS:
        return not bool(arguments.get("dry_run", False))
    if name == ASSERT_EXCLUSIVE:
        return True
    if name == CLEAR_RELATIONS:
        return not bool(arguments.get("dry_run", False))
    if name == LOAD_RELATIONS:
        return not bool(arguments.get("validate_only", False))
    if name == EXPORT_RELATIONS:
        return arguments.get("destination") == "file"
    return False


def _record_dump(record: RelationRecord) -> dict[str, Any]:
    return record.model_dump(mode="json", exclude_none=True)


def _exclusive_group_dump(group: ExclusiveGroupRecord) -> dict[str, Any]:
    return group.model_dump(mode="json")


def _independence_record_dump(record: IndependenceRecord) -> dict[str, Any]:
    return record.model_dump(mode="json")


def _proposition_dump(record: PropositionRecord) -> dict[str, Any]:
    return record.model_dump(mode="json", exclude_none=True)


def _temporary_fact_records(payload: CheckContradictionsInput) -> list[RelationRecord]:
    records: list[RelationRecord] = []
    for index, fact in enumerate(payload.facts):
        data = fact.model_dump()
        if data["id"] is None:
            data["id"] = f"input_{index}"
        records.append(RelationRecord(**data))
    return records


def _contradiction_trace(
    mode: ContradictionMode,
    fact_count: int,
    contradictions: list[dict[str, Any]],
) -> list[str]:
    trace = [f"Checked contradictions in mode={mode.value}."]
    if fact_count:
        trace.append(f"Loaded {fact_count} input fact(s) into temporary graph.")
    if contradictions:
        trace.append(f"Found {len(contradictions)} hard contradiction(s).")
    else:
        trace.append("No hard contradictions found.")
    return trace


def _normalization_trace(record: RelationRecord) -> str:
    source = record.canonical_source
    target = record.canonical_target
    if record.relation_type == "necessary":
        return (
            f"normalized necessary({record.source}, {record.target}) into implication edge "
            f"{target} -> {source}"
        )
    if record.relation_type == "equivalent":
        return (
            f"normalized equivalent({record.source}, {record.target}) into implication edges "
            f"{source} -> {target} and {target} -> {source}"
        )
    return (
        f"normalized sufficient({record.source}, {record.target}) into implication edge "
        f"{source} -> {target}"
    )
