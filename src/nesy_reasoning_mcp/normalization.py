"""Shared relation normalization helpers."""

from __future__ import annotations

from nesy_reasoning_mcp.schemas import CanonicalImplicationEdge, RelationRecord, RelationType


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
    if relation.relation_type == RelationType.SUFFICIENT:
        return [_edge(relation, relation.canonical_source, relation.canonical_target, "a")]
    if relation.relation_type == RelationType.NECESSARY:
        return [_edge(relation, relation.canonical_target, relation.canonical_source, "a")]
    if relation.relation_type == RelationType.EQUIVALENT:
        return [
            _edge(relation, relation.canonical_source, relation.canonical_target, "a"),
            _edge(relation, relation.canonical_target, relation.canonical_source, "b"),
        ]
    return []
