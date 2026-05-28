"""Graph summary tool handler."""

from __future__ import annotations

from typing import Any

from nesy_reasoning_mcp.reasoning import relations_compatible_with_filter
from nesy_reasoning_mcp.schemas import (
    ExclusiveGroupRecord,
    RelationRecord,
    SummarizeGraphInput,
)
from nesy_reasoning_mcp.store import RelationStoreProtocol, graph_stats_for
from nesy_reasoning_mcp.tool_common import _exclusive_group_compatible_with_context_filter
from nesy_reasoning_mcp.tool_queue_status import queue_status_snapshot


async def summarize_graph(
    arguments: dict[str, Any],
    store: RelationStoreProtocol,
) -> dict[str, Any]:
    """Handle `nesy.summarize_graph`."""
    payload = SummarizeGraphInput.model_validate(arguments)
    all_relations = store.list_relations()
    all_exclusive_groups = store.list_exclusive_groups()
    compatible_relations = relations_compatible_with_filter(all_relations, payload.context_filter)
    compatible_groups = [
        group
        for group in all_exclusive_groups
        if _exclusive_group_compatible_with_context_filter(group, payload.context_filter)
    ]
    relations = _summary_relations(compatible_relations, payload)
    selected_relations = relations[: payload.max_relations]
    relation_limit_truncated = len(relations) > len(selected_relations)
    exclusive_groups = _summary_exclusive_groups(compatible_groups, payload)
    summary, char_truncated = _format_graph_summary(
        selected_relations,
        exclusive_groups,
        payload=payload,
    )
    queue_snapshot = queue_status_snapshot(store)
    summary, queue_truncated = _append_background_status(
        summary,
        queue_snapshot,
        max_chars=payload.max_chars,
    )
    edges = store.implication_edges(compatible_relations)
    return {
        "status": "ok",
        "summary": summary,
        "relation_count_included": len(selected_relations),
        "truncated": relation_limit_truncated or char_truncated or queue_truncated,
        "diagnostics": [],
        "trace": [
            f"Selected {len(selected_relations)} relation(s) matching summary filters.",
            f"Selected {len(exclusive_groups)} exclusive group(s).",
        ],
        "graph_stats": graph_stats_for(
            compatible_relations,
            edges,
            exclusive_group_count=len(compatible_groups),
        ).model_dump(mode="json"),
    }


def _summary_relations(
    relations: list[RelationRecord],
    payload: SummarizeGraphInput,
) -> list[RelationRecord]:
    terms = _normalized_focus_terms(payload.focus_terms)
    selected = [
        relation
        for relation in relations
        if not terms or _text_matches_terms([relation.source, relation.target], terms)
    ]
    return sorted(
        selected,
        key=lambda item: (
            item.store_id,
            item.context_id,
            item.source,
            item.target,
            item.relation_type.value,
            item.id,
        ),
    )


def _summary_exclusive_groups(
    groups: list[ExclusiveGroupRecord],
    payload: SummarizeGraphInput,
) -> list[ExclusiveGroupRecord]:
    if not payload.include_exclusives:
        return []
    terms = _normalized_focus_terms(payload.focus_terms)
    selected = [
        group
        for group in groups
        if _exclusive_group_compatible_with_context_filter(group, payload.context_filter)
        and (not terms or _text_matches_terms(group.members, terms))
    ]
    return sorted(
        selected,
        key=lambda item: (item.store_id, item.context_id, item.group_id),
    )


def _format_graph_summary(
    relations: list[RelationRecord],
    exclusive_groups: list[ExclusiveGroupRecord],
    *,
    payload: SummarizeGraphInput,
) -> tuple[str, bool]:
    context_parts = []
    if payload.context_filter.store_id:
        context_parts.append(f"store={payload.context_filter.store_id}")
    if payload.context_filter.context_id:
        context_parts.append(f"context={payload.context_filter.context_id}")
    if payload.context_filter.domain:
        context_parts.append(f"domain={payload.context_filter.domain}")
    title = "Known NeSy reasoning graph"
    if context_parts:
        title = f"{title} ({', '.join(context_parts)})"

    lines = [f"{title}:"]
    if relations:
        lines.extend(_relation_summary_line(relation) for relation in relations)
    else:
        lines.append("- No matching relations.")

    if payload.include_exclusives:
        lines.append("Exclusive groups:")
        if exclusive_groups:
            lines.extend(_exclusive_group_summary_line(group) for group in exclusive_groups)
        else:
            lines.append("- No matching exclusive groups.")

    summary = "\n".join(lines)
    if len(summary) <= payload.max_chars:
        return summary, False

    suffix = "\n...truncated"
    cutoff = max(0, payload.max_chars - len(suffix))
    return f"{summary[:cutoff].rstrip()}{suffix}", True


def _relation_summary_line(relation: RelationRecord) -> str:
    return (
        f"- {relation.source} {relation.relation_type.value} {relation.target} "
        f"(conf={relation.confidence:g}, context={relation.context_id}, "
        f"store={relation.store_id}, id={relation.id})"
    )


def _exclusive_group_summary_line(group: ExclusiveGroupRecord) -> str:
    return (
        f"- {group.group_id}: {' | '.join(group.members)} "
        f"(context={group.context_id}, store={group.store_id}, scope={group.scope.value})"
    )


def _normalized_focus_terms(focus_terms: list[str]) -> list[str]:
    return [term.casefold() for term in focus_terms]


def _text_matches_terms(values: list[str], terms: list[str]) -> bool:
    haystack = "\n".join(values).casefold()
    return any(term in haystack for term in terms)


def _append_background_status(
    summary: str,
    queue_snapshot: dict[str, Any],
    *,
    max_chars: int,
) -> tuple[str, bool]:
    if queue_snapshot["in_flight_total"] == 0:
        return summary, False
    status_block = _background_status_block(queue_snapshot)
    combined = f"{summary}\n{status_block}"
    if len(combined) <= max_chars:
        return combined, False

    suffix = "\n...truncated"
    separator = "\n"
    cutoff = max(0, max_chars - len(status_block) - len(separator) - len(suffix))
    truncated_summary = f"{summary[:cutoff].rstrip()}{suffix}"
    return f"{truncated_summary}{separator}{status_block}", True


def _background_status_block(queue_snapshot: dict[str, Any]) -> str:
    last_write = "Last write: none."
    if queue_snapshot["last_write_at"] is not None:
        last_write = (
            f"Last write: {queue_snapshot['last_write_at']} "
            f"({queue_snapshot['last_write_relation_count']} relations added)."
        )
    return (
        "Background processing: "
        f"{queue_snapshot['pending']} turns pending extraction, "
        f"{queue_snapshot['extracting']} extracting, "
        f"{queue_snapshot['reviewing']} under review. "
        f"{last_write}"
    )
