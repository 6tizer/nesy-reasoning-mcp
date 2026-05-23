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
        "required": ["status", "loaded_relations", "loaded_exclusive_groups", "rejected"],
        "additionalProperties": False,
    }


def _export_relations_output_schema() -> dict[str, Any]:
    props = _common_output_properties()
    props.update(
        {
            "format": {"type": "string", "enum": ["json", "jsonl"]},
            "relation_count": {"type": "integer"},
            "exclusive_group_count": {"type": "integer"},
            "data": {"type": ["object", "string", "null"]},
            "path": {"type": ["string", "null"]},
            "bytes": {"type": "integer"},
        }
    )
    return {
        "type": "object",
        "properties": props,
        "required": ["status", "format", "relation_count", "exclusive_group_count"],
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
