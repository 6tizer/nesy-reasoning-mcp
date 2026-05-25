"""MCP tool output schemas."""

from __future__ import annotations

from typing import Any


def _common_output_properties() -> dict[str, Any]:
    return {
        "status": {"type": "string", "enum": ["ok", "warning", "error"]},
        "diagnostics": {"type": "array"},
        "trace": {"type": "array"},
        "graph_stats": {"type": "object"},
    }


def _assert_relations_output_schema() -> dict[str, Any]:
    props = _common_output_properties()
    props.update(
        {
            "added": {"type": "integer"},
            "updated": {"type": "integer"},
            "rejected": {"type": "integer"},
            "relation_ids": {"type": "array", "items": {"type": "string"}},
            "contradictions": {"type": "array"},
        }
    )
    return {
        "type": "object",
        "properties": props,
        "required": ["status", "added", "updated", "rejected", "relation_ids"],
        "additionalProperties": False,
    }


def _list_relations_output_schema() -> dict[str, Any]:
    props = _common_output_properties()
    props.update(
        {
            "relations": {"type": "array"},
            "implication_edges": {"type": "array"},
            "exclusive_groups": {"type": "array"},
            "total": {"type": "integer"},
            "next_cursor": {"type": ["string", "null"]},
        }
    )
    return {
        "type": "object",
        "properties": props,
        "required": ["status", "relations", "total"],
        "additionalProperties": False,
    }


def _clear_relations_output_schema() -> dict[str, Any]:
    props = _common_output_properties()
    props.update(
        {
            "removed_relations": {"type": "integer"},
            "removed_exclusive_groups": {"type": "integer"},
            "dry_run": {"type": "boolean"},
        }
    )
    return {
        "type": "object",
        "properties": props,
        "required": ["status", "removed_relations", "removed_exclusive_groups", "dry_run"],
        "additionalProperties": False,
    }


def _classify_output_schema() -> dict[str, Any]:
    props = _common_output_properties()
    props.update(
        {
            "source": {"type": "string"},
            "target": {"type": "string"},
            "classification": {
                "type": "string",
                "enum": ["sufficient", "necessary", "equivalent", "unknown", "contradictory"],
            },
            "source_implies_target": {"type": "object"},
            "target_implies_source": {"type": "object"},
            "necessity_status": {"type": "object"},
            "direct_relations": {"type": "array"},
            "paths": {"type": "array"},
        }
    )
    return {
        "type": "object",
        "properties": props,
        "required": ["status", "source", "target", "classification"],
        "additionalProperties": False,
    }


def _verify_chain_output_schema() -> dict[str, Any]:
    props = _common_output_properties()
    props.update(
        {
            "reachable": {"type": "boolean"},
            "relation_established": {"type": "boolean"},
            "source_to_target_reachable": {"type": "boolean"},
            "target_to_source_reachable": {"type": "boolean"},
            "relation_type": {"type": "string"},
            "logic_validity": {"type": "boolean"},
            "best_path": {"type": ["object", "null"]},
            "paths": {"type": "array"},
            "broken_at": {"type": ["object", "null"]},
        }
    )
    return {
        "type": "object",
        "properties": props,
        "required": [
            "status",
            "reachable",
            "relation_established",
            "source_to_target_reachable",
            "target_to_source_reachable",
            "logic_validity",
        ],
        "additionalProperties": False,
    }


def _assert_exclusive_output_schema() -> dict[str, Any]:
    props = _common_output_properties()
    props.update(
        {
            "added_groups": {"type": "integer"},
            "updated_groups": {"type": "integer"},
            "group_ids": {"type": "array", "items": {"type": "string"}},
        }
    )
    return {
        "type": "object",
        "properties": props,
        "required": ["status", "added_groups", "updated_groups", "group_ids"],
        "additionalProperties": False,
    }


def _check_contradictions_output_schema() -> dict[str, Any]:
    props = _common_output_properties()
    props.update(
        {
            "has_contradictions": {"type": "boolean"},
            "contradictions": {"type": "array"},
            "clean_facts_count": {"type": "integer"},
            "total_facts_count": {"type": "integer"},
            "context_separated": {"type": "array"},
        }
    )
    return {
        "type": "object",
        "properties": props,
        "required": ["status", "has_contradictions", "contradictions"],
        "additionalProperties": False,
    }


def _load_relations_output_schema() -> dict[str, Any]:
    props = _common_output_properties()
    props.update(
        {
            "loaded_relations": {"type": "integer"},
            "loaded_exclusive_groups": {"type": "integer"},
            "loaded_propositions": {"type": "integer"},
            "updated_relations": {"type": "integer"},
            "updated_exclusive_groups": {"type": "integer"},
            "rejected": {"type": "integer"},
            "conflicts": {"type": "array"},
            "validate_only": {"type": "boolean"},
        }
    )
    return {
        "type": "object",
        "properties": props,
        "required": [
            "status",
            "loaded_relations",
            "loaded_exclusive_groups",
            "loaded_propositions",
            "rejected",
        ],
        "additionalProperties": False,
    }


def _export_relations_output_schema() -> dict[str, Any]:
    props = _common_output_properties()
    props.update(
        {
            "format": {"type": "string", "enum": ["json", "jsonl"]},
            "relation_count": {"type": "integer"},
            "exclusive_group_count": {"type": "integer"},
            "proposition_count": {"type": "integer"},
            "data": {"type": ["object", "string", "null"]},
            "path": {"type": ["string", "null"]},
            "bytes": {"type": "integer"},
        }
    )
    return {
        "type": "object",
        "properties": props,
        "required": [
            "status",
            "format",
            "relation_count",
            "exclusive_group_count",
            "proposition_count",
        ],
        "additionalProperties": False,
    }


def _summarize_graph_output_schema() -> dict[str, Any]:
    props = _common_output_properties()
    props.update(
        {
            "summary": {"type": "string"},
            "relation_count_included": {"type": "integer"},
            "truncated": {"type": "boolean"},
        }
    )
    return {
        "type": "object",
        "properties": props,
        "required": ["status", "summary", "relation_count_included", "truncated"],
        "additionalProperties": False,
    }


def _counterfactual_output_schema() -> dict[str, Any]:
    props = _common_output_properties()
    props.update(
        {
            "if_not": {"type": "string"},
            "world_mode": {"type": "string", "enum": ["open", "closed"]},
            "necessarily_blocked": {"type": "array"},
            "possibly_blocked": {"type": "array"},
            "still_possible": {"type": "array"},
            "unknown": {"type": "array"},
            "not_derivably_affected": {"type": "array"},
        }
    )
    return {
        "type": "object",
        "properties": props,
        "required": ["status", "if_not", "world_mode"],
        "additionalProperties": False,
    }


def _reason_over_relations_output_schema() -> dict[str, Any]:
    props = _common_output_properties()
    props.update(
        {
            "mode": {
                "type": "string",
                "enum": [
                    "classify",
                    "verify_chain",
                    "counterfactual",
                    "check_contradictions",
                    "summarize_graph",
                ],
            },
            "persisted": {"type": "boolean"},
            "result": {"type": "object"},
            "relation_count": {"type": "integer"},
            "exclusive_group_count": {"type": "integer"},
            "proposition_count": {"type": "integer"},
        }
    )
    return {
        "type": "object",
        "properties": props,
        "required": ["status", "mode", "persisted", "result"],
        "additionalProperties": False,
    }


def _validate_candidate_relations_output_schema() -> dict[str, Any]:
    props = _common_output_properties()
    props.update(
        {
            "persisted": {"type": "boolean"},
            "candidate_count": {"type": "integer"},
            "approved_count": {"type": "integer"},
            "queued_count": {"type": "integer"},
            "rejected_count": {"type": "integer"},
            "gate_results": {"type": "array"},
            "approved_relations": {"type": "array"},
            "review_aggregation": {"type": "object"},
            "reasoning": {"type": "object"},
        }
    )
    return {
        "type": "object",
        "properties": props,
        "required": [
            "status",
            "persisted",
            "candidate_count",
            "approved_count",
            "queued_count",
            "rejected_count",
            "gate_results",
            "approved_relations",
            "review_aggregation",
            "diagnostics",
            "reasoning",
            "graph_stats",
            "trace",
        ],
        "additionalProperties": False,
    }


def _list_review_queue_output_schema() -> dict[str, Any]:
    props = _common_output_properties()
    props.update(
        {
            "records": {"type": "array"},
            "total": {"type": "integer"},
            "next_cursor": {"type": ["string", "null"]},
        }
    )
    return {
        "type": "object",
        "properties": props,
        "required": ["status", "records", "total"],
        "additionalProperties": False,
    }


def _commit_reviewed_relations_output_schema() -> dict[str, Any]:
    props = _common_output_properties()
    props.update(
        {
            "committed_count": {"type": "integer"},
            "queue_ids": {"type": "array", "items": {"type": "string"}},
            "relation_ids": {"type": "array", "items": {"type": "string"}},
            "validation": {"type": "object"},
            "write_result": {"type": "object"},
        }
    )
    return {
        "type": "object",
        "properties": props,
        "required": ["status", "committed_count", "queue_ids", "relation_ids"],
        "additionalProperties": False,
    }


def _resolve_review_queue_output_schema() -> dict[str, Any]:
    props = _common_output_properties()
    props.update(
        {
            "resolved_count": {"type": "integer"},
            "queue_ids": {"type": "array", "items": {"type": "string"}},
        }
    )
    return {
        "type": "object",
        "properties": props,
        "required": ["status", "resolved_count", "queue_ids"],
        "additionalProperties": False,
    }
