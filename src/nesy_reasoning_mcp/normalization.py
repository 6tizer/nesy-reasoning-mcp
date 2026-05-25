"""Shared relation normalization helpers."""

from __future__ import annotations

from typing import TypedDict

from nesy_reasoning_mcp.schemas import CanonicalImplicationEdge, RelationRecord, RelationType


class NormalizedImplicationPreview(TypedDict):
    """Human-readable preview of a relation's canonical implication edge."""

    antecedent: str
    consequent: str


def normalized_implication_preview(
    source: str,
    target: str,
    relation_type: RelationType | str,
) -> list[NormalizedImplicationPreview]:
    """Return canonical implication direction(s) for a relation shape."""
    resolved_type = RelationType(relation_type)
    if resolved_type == RelationType.SUFFICIENT:
        return [{"antecedent": source, "consequent": target}]
    if resolved_type == RelationType.NECESSARY:
        return [{"antecedent": target, "consequent": source}]
    if resolved_type == RelationType.EQUIVALENT:
        return [
            {"antecedent": source, "consequent": target},
            {"antecedent": target, "consequent": source},
        ]
    return []


def _edge(
    relation: RelationRecord,
    antecedent: str,
    consequent: str,
    suffix: str,
) -> CanonicalImplicationEdge:
    return CanonicalImplicationEdge(
        edge_id=f"edge_{relation.id}_{suffix}",
        relation_id=relation.id,
        antecedent=antecedent,
        consequent=consequent,
        source_relation_type=relation.relation_type,
        confidence=relation.confidence,
        context_id=relation.context_id,
        store_id=relation.store_id,
        assumptions=list(relation.assumptions),
        temporal=relation.temporal,
    )


def normalize_relation_edges(relation: RelationRecord) -> list[CanonicalImplicationEdge]:
    """Derive canonical implication edges for one stored relation."""
    suffixes = ("a", "b")
    return [
        _edge(relation, preview["antecedent"], preview["consequent"], suffix)
        for preview, suffix in zip(
            normalized_implication_preview(
                relation.canonical_source,
                relation.canonical_target,
                relation.relation_type,
            ),
            suffixes,
            strict=False,
        )
    ]
