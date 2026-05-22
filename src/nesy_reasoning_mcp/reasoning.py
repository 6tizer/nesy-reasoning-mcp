"""Graph-based reasoning and contradiction utilities."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from math import log
from typing import Any

import networkx as nx

from nesy_reasoning_mcp.schemas import (
    CanonicalImplicationEdge,
    Classification,
    ConfidencePolicy,
    ContextFilter,
    ExclusiveGroupRecord,
    ExclusiveScope,
    ExpectedRelation,
    PathStrategy,
    RelationRecord,
    RelationType,
)
from nesy_reasoning_mcp.store import graph_stats_for

ZERO_CONFIDENCE_WEIGHT = 1_000_000_000.0


@dataclass(frozen=True)
class ReasoningPath:
    """A concrete implication path."""

    nodes: list[str]
    edges: list[CanonicalImplicationEdge]
    evidence_confidence: float | None
    confidence_policy: ConfidencePolicy

    @property
    def temporal_window(self) -> tuple[datetime | None, datetime | None]:
        """Return the intersection of temporal windows across path edges."""
        start: datetime | None = None
        end: datetime | None = None
        for edge in self.edges:
            edge_start, edge_end = edge.temporal_window
            if edge_start is not None:
                start = edge_start if start is None else max(start, edge_start)
            if edge_end is not None:
                end = edge_end if end is None else min(end, edge_end)
        return start, end


@dataclass(frozen=True)
class BrokenChain:
    """Explicit chain verification failure."""

    index: int
    from_node: str
    to_node: str
    reason: str
    direction_mismatch: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Return the public structured representation."""
        return {
            "index": self.index,
            "from": self.from_node,
            "to": self.to_node,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class ClassificationContradiction:
    """Exclusive-group contradiction found during relation classification."""

    exclusive_group_id: str
    context_id: str
    store_id: str
    target: str
    conflicting_target: str
    target_path: ReasoningPath
    conflicting_path: ReasoningPath

    @property
    def fact_ids(self) -> list[str]:
        """Return unique relation ids supporting both contradictory paths."""
        return list(
            dict.fromkeys(
                [
                    *(edge.relation_id for edge in self.target_path.edges),
                    *(edge.relation_id for edge in self.conflicting_path.edges),
                ]
            )
        )


class GraphIndex:
    """Context-filtered NetworkX index over canonical implication edges."""

    def __init__(
        self,
        relations: list[RelationRecord],
        context_filter: ContextFilter | None = None,
    ) -> None:
        self.relations = [
            relation for relation in relations if _context_compatible(relation, context_filter)
        ]
        self.edges = [edge for relation in self.relations for edge in normalize_relation(relation)]
        self.graph: nx.MultiDiGraph = nx.MultiDiGraph()
        for edge in self.edges:
            self.graph.add_edge(
                edge.antecedent,
                edge.consequent,
                key=edge.edge_id,
                edge=edge,
                weight=edge_weight(edge),
            )

    @property
    def graph_stats(self) -> dict[str, Any]:
        """Return graph statistics for this filtered index."""
        return graph_stats_for(self.relations, self.edges).model_dump(mode="json")

    def direct_relations_between(self, source: str, target: str) -> list[dict[str, Any]]:
        """Return stored records directly connecting source and target in either direction."""
        return [
            relation.model_dump(mode="json")
            for relation in self.relations
            if {relation.source, relation.target} == {source, target}
        ]

    def find_paths(
        self,
        start: str,
        end: str,
        *,
        max_depth: int,
        strategy: PathStrategy = PathStrategy.BEST_CONFIDENCE,
        max_paths: int = 5,
        confidence_policy: ConfidencePolicy = ConfidencePolicy.PRODUCT_INDEPENDENT,
        direct_only: bool = False,
    ) -> list[ReasoningPath]:
        """Find simple implication paths between two propositions."""
        if start == end:
            return [
                ReasoningPath(
                    nodes=[start],
                    edges=[],
                    evidence_confidence=aggregate_confidence([], confidence_policy),
                    confidence_policy=confidence_policy,
                )
            ]
        if start not in self.graph or end not in self.graph:
            return []

        cutoff = 1 if direct_only else max_depth
        if direct_only:
            edge = self.best_direct_edge(start, end)
            if edge is None:
                return []
            return [
                ReasoningPath(
                    nodes=[start, end],
                    edges=[edge],
                    evidence_confidence=aggregate_confidence([edge], confidence_policy),
                    confidence_policy=confidence_policy,
                )
            ]

        if strategy in {PathStrategy.BEST_CONFIDENCE, PathStrategy.SHORTEST}:
            path = self._shortest_path(
                start,
                end,
                max_depth=cutoff,
                confidence_policy=confidence_policy,
                weighted=strategy == PathStrategy.BEST_CONFIDENCE,
            )
            return [path] if path is not None else []

        node_paths = nx.all_simple_paths(self.graph, start, end, cutoff=cutoff)
        paths = [
            ReasoningPath(
                nodes=list(nodes),
                edges=self._best_edges_for_nodes(nodes),
                evidence_confidence=aggregate_confidence(
                    self._best_edges_for_nodes(nodes), confidence_policy
                ),
                confidence_policy=confidence_policy,
            )
            for nodes in node_paths
        ]
        if strategy == PathStrategy.SHORTEST:
            paths.sort(key=lambda path: (len(path.edges), -rank_confidence(path.edges)))
        else:
            paths.sort(key=lambda path: (-rank_confidence(path.edges), len(path.edges)))
        return paths[:max_paths]

    def _shortest_path(
        self,
        start: str,
        end: str,
        *,
        max_depth: int,
        confidence_policy: ConfidencePolicy,
        weighted: bool,
    ) -> ReasoningPath | None:
        weight = "weight" if weighted else None
        try:
            nodes = nx.shortest_path(self.graph, start, end, weight=weight)
        except nx.NetworkXNoPath:
            return None
        if len(nodes) - 1 > max_depth:
            return None
        edges = self._best_edges_for_nodes(nodes)
        return ReasoningPath(
            nodes=nodes,
            edges=edges,
            evidence_confidence=aggregate_confidence(edges, confidence_policy),
            confidence_policy=confidence_policy,
        )

    def best_direct_edge(self, antecedent: str, consequent: str) -> CanonicalImplicationEdge | None:
        """Return the highest-confidence direct edge for an ordered node pair."""
        if not self.graph.has_edge(antecedent, consequent):
            return None
        edge_data = self.graph.get_edge_data(antecedent, consequent, default={})
        edges = [data["edge"] for data in edge_data.values()]
        return max(edges, key=lambda edge: edge.confidence) if edges else None

    def verify_explicit_chain(
        self,
        chain: list[str],
        *,
        confidence_policy: ConfidencePolicy,
    ) -> tuple[ReasoningPath | None, BrokenChain | None]:
        """Verify that every adjacent pair in an explicit chain has a direct implication edge."""
        edges: list[CanonicalImplicationEdge] = []
        for index, (left, right) in enumerate(zip(chain, chain[1:], strict=False)):
            edge = self.best_direct_edge(left, right)
            if edge is not None:
                edges.append(edge)
                continue
            reverse = self.best_direct_edge(right, left)
            if reverse is not None:
                return None, BrokenChain(
                    index=index,
                    from_node=left,
                    to_node=right,
                    reason=(
                        f"No implication edge {left} -> {right} exists. Existing relation "
                        f"{reverse.source_relation_type} maps to {right} -> {left}, not "
                        f"{left} -> {right}."
                    ),
                    direction_mismatch=True,
                )
            return None, BrokenChain(
                index=index,
                from_node=left,
                to_node=right,
                reason=f"No implication edge {left} -> {right} exists.",
            )
        return (
            ReasoningPath(
                nodes=chain,
                edges=edges,
                evidence_confidence=aggregate_confidence(edges, confidence_policy),
                confidence_policy=confidence_policy,
            ),
            None,
        )

    def _best_edges_for_nodes(self, nodes: list[str]) -> list[CanonicalImplicationEdge]:
        edges: list[CanonicalImplicationEdge] = []
        for left, right in zip(nodes, nodes[1:], strict=False):
            edge = self.best_direct_edge(left, right)
            if edge is None:  # pragma: no cover - guarded by NetworkX path generation
                raise ValueError(f"missing edge for generated path: {left} -> {right}")
            edges.append(edge)
        return edges


def build_graph(
    relations: list[RelationRecord],
    context_filter: ContextFilter | None = None,
) -> GraphIndex:
    """Build a context-filtered graph index."""
    return GraphIndex(relations, context_filter)


def find_exclusive_contradictions(
    relations: list[RelationRecord],
    exclusive_groups: list[ExclusiveGroupRecord],
    context_filter: ContextFilter,
    *,
    max_depth: int,
    include_soft: bool = False,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Find exclusivity-based contradictions and context-separated tensions."""
    contradictions: list[dict[str, Any]] = []
    context_separated: list[dict[str, Any]] = []
    compatible_relations = relations_compatible_with_filter(relations, context_filter)
    contradictions.extend(_explicit_negation_contradictions(compatible_relations, max_depth))
    if include_soft:
        contradictions.extend(_confidence_tensions(compatible_relations))
    for group in exclusive_groups:
        if not _exclusive_group_compatible(group, context_filter):
            continue
        group_relations = _relations_for_group(relations, group, context_filter)
        contradictions.extend(_hard_group_contradictions(group_relations, group, max_depth))
        if group.scope == ExclusiveScope.SAME_CONTEXT:
            context_separated.extend(
                _context_separated_conflicts(relations, group, context_filter, max_depth)
            )
    return contradictions, _dedupe_context_separated(context_separated)


def normalize_relation(relation: RelationRecord) -> list[CanonicalImplicationEdge]:
    """Normalize an external relation record into canonical implication edges."""
    if relation.relation_type == RelationType.SUFFICIENT:
        return [_edge(relation, relation.source, relation.target, "a")]
    if relation.relation_type == RelationType.NECESSARY:
        return [_edge(relation, relation.target, relation.source, "a")]
    return [
        _edge(relation, relation.source, relation.target, "a"),
        _edge(relation, relation.target, relation.source, "b"),
    ]


def relations_compatible_with_filter(
    relations: list[RelationRecord],
    context_filter: ContextFilter,
) -> list[RelationRecord]:
    """Return relations compatible with a context filter."""
    return [relation for relation in relations if _context_compatible(relation, context_filter)]


def classify_reachability(
    source_to_target: list[ReasoningPath],
    target_to_source: list[ReasoningPath],
) -> Classification:
    """Map bidirectional reachability to an external relation classification."""
    forward = bool(source_to_target)
    reverse = bool(target_to_source)
    if forward and reverse:
        return Classification.EQUIVALENT
    if forward:
        return Classification.SUFFICIENT
    if reverse:
        return Classification.NECESSARY
    return Classification.UNKNOWN


def find_classification_contradiction(
    relations: list[RelationRecord],
    exclusive_groups: list[ExclusiveGroupRecord],
    *,
    source: str,
    target: str,
    context_filter: ContextFilter,
    max_depth: int,
    confidence_policy: ConfidencePolicy,
    direct_only: bool,
) -> ClassificationContradiction | None:
    """Find whether a classification target conflicts with an exclusive sibling."""
    for group in exclusive_groups:
        if target not in group.members or not _exclusive_group_compatible(group, context_filter):
            continue
        group_relations = _relations_for_group(relations, group, context_filter)
        if not group_relations:
            continue
        index = GraphIndex(group_relations)
        target_paths = index.find_paths(
            source,
            target,
            max_depth=max_depth,
            strategy=PathStrategy.BEST_CONFIDENCE,
            max_paths=1,
            confidence_policy=confidence_policy,
            direct_only=direct_only,
        )
        if not target_paths:
            continue
        target_path = target_paths[0]
        for member in sorted(item for item in group.members if item != target):
            conflicting_paths = index.find_paths(
                source,
                member,
                max_depth=max_depth,
                strategy=PathStrategy.BEST_CONFIDENCE,
                max_paths=1,
                confidence_policy=confidence_policy,
                direct_only=direct_only,
            )
            if not conflicting_paths:
                continue
            conflicting_path = conflicting_paths[0]
            if not _paths_temporally_compatible([target_path, conflicting_path]):
                continue
            return ClassificationContradiction(
                exclusive_group_id=group.group_id,
                context_id=group.context_id,
                store_id=group.store_id,
                target=target,
                conflicting_target=member,
                target_path=target_path,
                conflicting_path=conflicting_path,
            )
    return None


def aggregate_confidence(
    edges: list[CanonicalImplicationEdge],
    policy: ConfidencePolicy,
) -> float | None:
    """Aggregate path confidence under the selected policy."""
    values = [edge.confidence for edge in edges]
    if policy == ConfidencePolicy.NO_AGGREGATION:
        return None
    if policy == ConfidencePolicy.MIN:
        return min(values) if values else 1.0
    result = 1.0
    for value in values:
        result *= value
    return result


def rank_confidence(edges: list[CanonicalImplicationEdge]) -> float:
    """Return product confidence for ordering best-confidence paths."""
    result = 1.0
    for edge in edges:
        result *= edge.confidence
    return result


def edge_weight(edge: CanonicalImplicationEdge) -> float:
    """Return Dijkstra-compatible confidence weight for an implication edge."""
    if edge.confidence <= 0:
        return ZERO_CONFIDENCE_WEIGHT
    return -log(edge.confidence)


def path_to_dict(path: ReasoningPath, *, relation_type: str | None = None) -> dict[str, Any]:
    """Serialize a reasoning path."""
    data: dict[str, Any] = {
        "nodes": path.nodes,
        "steps": [
            {
                "antecedent": edge.antecedent,
                "consequent": edge.consequent,
                "relation_id": edge.relation_id,
                "source_relation_type": edge.source_relation_type.value,
                "confidence": edge.confidence,
            }
            for edge in path.edges
        ],
        "evidence_confidence": path.evidence_confidence,
        "confidence_policy": path.confidence_policy.value,
    }
    if relation_type is not None:
        data["relation_type"] = relation_type
        data["logic_validity"] = True
    return data


def expected_relation_matches(
    expected: ExpectedRelation,
    classification: Classification,
) -> bool:
    """Return whether a classified relation satisfies an expected relation."""
    if expected == ExpectedRelation.ANY:
        return classification != Classification.UNKNOWN
    if expected == ExpectedRelation.SUFFICIENT:
        return classification in {Classification.SUFFICIENT, Classification.EQUIVALENT}
    if expected == ExpectedRelation.NECESSARY:
        return classification in {Classification.NECESSARY, Classification.EQUIVALENT}
    return classification == Classification.EQUIVALENT


def _context_compatible(
    relation: RelationRecord,
    context_filter: ContextFilter | None,
) -> bool:
    if context_filter is None:
        return True
    if context_filter.store_id and relation.store_id != context_filter.store_id:
        return False
    if context_filter.context_id and relation.context_id != context_filter.context_id:
        return False
    if context_filter.domain and relation.metadata.get("domain") != context_filter.domain:
        return False
    if context_filter.assumptions and not set(context_filter.assumptions).issubset(
        set(relation.assumptions)
    ):
        return False
    return not (
        context_filter.valid_at and not _valid_at_matches(relation, context_filter.valid_at)
    )


def _exclusive_group_compatible(
    group: ExclusiveGroupRecord,
    context_filter: ContextFilter,
) -> bool:
    if context_filter.store_id and group.store_id != context_filter.store_id:
        return False
    return not (
        context_filter.context_id
        and group.scope == ExclusiveScope.SAME_CONTEXT
        and group.context_id != context_filter.context_id
    )


def _relations_for_group(
    relations: list[RelationRecord],
    group: ExclusiveGroupRecord,
    context_filter: ContextFilter,
) -> list[RelationRecord]:
    compatible = relations_compatible_with_filter(relations, context_filter)
    if group.scope == ExclusiveScope.GLOBAL:
        return [relation for relation in compatible if relation.store_id == group.store_id]
    return [
        relation
        for relation in compatible
        if relation.store_id == group.store_id and relation.context_id == group.context_id
    ]


def _hard_group_contradictions(
    relations: list[RelationRecord],
    group: ExclusiveGroupRecord,
    max_depth: int,
) -> list[dict[str, Any]]:
    if not relations:
        return []
    index = GraphIndex(relations)
    contradictions: list[dict[str, Any]] = []
    for source in sorted(index.graph.nodes):
        hits = _reachable_group_members(index, source, group.members, max_depth)
        if len(hits) < 2:
            continue
        hit_paths = list(hits.values())
        if not _paths_temporally_compatible(hit_paths):
            continue
        path_edges = [edge for path in hits.values() for edge in path.edges]
        relation_ids = list(dict.fromkeys(edge.relation_id for edge in path_edges))
        edge_lengths = [len(path.edges) for path in hits.values()]
        contradiction_type = (
            "exclusive_targets"
            if edge_lengths and max(edge_lengths) <= 1
            else "transitive_exclusive_targets"
        )
        contradictions.append(
            {
                "type": contradiction_type,
                "severity": "hard",
                "source": source,
                "targets": list(hits.keys()),
                "exclusive_group_id": group.group_id,
                "context_id": group.context_id,
                "store_id": group.store_id,
                "fact_ids": relation_ids,
                "reason": (
                    "Under the same compatible scope, the same source implies multiple "
                    "mutually exclusive targets."
                ),
            }
        )
    return contradictions


def _paths_temporally_compatible(paths: list[ReasoningPath]) -> bool:
    start: datetime | None = None
    end: datetime | None = None
    for path in paths:
        path_start, path_end = path.temporal_window
        if path_start is not None:
            start = path_start if start is None else max(start, path_start)
        if path_end is not None:
            end = path_end if end is None else min(end, path_end)
    return not (start is not None and end is not None and start > end)


def _context_separated_conflicts(
    relations: list[RelationRecord],
    group: ExclusiveGroupRecord,
    context_filter: ContextFilter,
    max_depth: int,
) -> list[dict[str, Any]]:
    compatible = relations_compatible_with_filter(relations, context_filter)
    contexts = sorted(
        {relation.context_id for relation in compatible if relation.store_id == group.store_id}
    )
    hits_by_source: dict[str, dict[str, set[str]]] = {}
    for context_id in contexts:
        context_relations = [
            relation
            for relation in compatible
            if relation.store_id == group.store_id and relation.context_id == context_id
        ]
        if not context_relations:
            continue
        index = GraphIndex(context_relations)
        for source in sorted(index.graph.nodes):
            hits = _reachable_group_members(index, source, group.members, max_depth)
            if not hits:
                continue
            source_hits = hits_by_source.setdefault(source, {})
            for member in hits:
                source_hits.setdefault(member, set()).add(context_id)

    separated: list[dict[str, Any]] = []
    for source, member_contexts in hits_by_source.items():
        if len(member_contexts) < 2:
            continue
        involved_contexts = sorted(
            {
                context_id
                for contexts_for_member in member_contexts.values()
                for context_id in contexts_for_member
            }
        )
        if len(involved_contexts) < 2:
            continue
        separated.append(
            {
                "type": "context_separated_conflict",
                "severity": "not_contradiction",
                "source": source,
                "targets": sorted(member_contexts),
                "exclusive_group_id": group.group_id,
                "contexts": involved_contexts,
                "reason": (
                    "The source implies mutually exclusive targets only across separate "
                    "contexts, so this is not a hard contradiction."
                ),
            }
        )
    return separated


def _explicit_negation_contradictions(
    relations: list[RelationRecord],
    max_depth: int,
) -> list[dict[str, Any]]:
    contradictions: list[dict[str, Any]] = []
    for (store_id, context_id), scoped_relations in _relations_by_scope(relations).items():
        index = GraphIndex(scoped_relations)
        negated_nodes = sorted(
            (node, base)
            for node in (str(item) for item in index.graph.nodes)
            if (base := _explicit_negation_base(node)) is not None
        )
        if not negated_nodes:
            continue

        for source in sorted(str(item) for item in index.graph.nodes):
            contradictions.extend(
                _cycle_to_exclusion_contradictions(
                    index,
                    source,
                    negated_nodes,
                    context_id=context_id,
                    store_id=store_id,
                    max_depth=max_depth,
                )
            )
            contradictions.extend(
                _direct_opposition_contradictions(
                    index,
                    source,
                    negated_nodes,
                    context_id=context_id,
                    store_id=store_id,
                    max_depth=max_depth,
                )
            )
    return _dedupe_contradictions(contradictions)


def _cycle_to_exclusion_contradictions(
    index: GraphIndex,
    source: str,
    negated_nodes: list[tuple[str, str]],
    *,
    context_id: str,
    store_id: str,
    max_depth: int,
) -> list[dict[str, Any]]:
    contradictions: list[dict[str, Any]] = []
    for negated_node, base in negated_nodes:
        if base != source:
            continue
        paths = index.find_paths(
            source,
            negated_node,
            max_depth=max_depth,
            strategy=PathStrategy.SHORTEST,
            max_paths=1,
        )
        if not paths or len(paths[0].edges) <= 1:
            continue
        path = paths[0]
        if not _paths_temporally_compatible([path]):
            continue
        contradictions.append(
            {
                "type": "cycle_to_exclusion",
                "severity": "hard",
                "source": source,
                "targets": [source, negated_node],
                "path": path.nodes,
                "context_id": context_id,
                "store_id": store_id,
                "fact_ids": _path_fact_ids([path]),
                "reason": "The source reaches an explicit negation of itself through a cycle.",
            }
        )
    return contradictions


def _direct_opposition_contradictions(
    index: GraphIndex,
    source: str,
    negated_nodes: list[tuple[str, str]],
    *,
    context_id: str,
    store_id: str,
    max_depth: int,
) -> list[dict[str, Any]]:
    contradictions: list[dict[str, Any]] = []
    for negated_node, base in negated_nodes:
        positive_paths = index.find_paths(
            source,
            base,
            max_depth=max_depth,
            strategy=PathStrategy.SHORTEST,
            max_paths=1,
        )
        negative_paths = index.find_paths(
            source,
            negated_node,
            max_depth=max_depth,
            strategy=PathStrategy.SHORTEST,
            max_paths=1,
        )
        if not positive_paths or not negative_paths:
            continue
        if source == base and len(negative_paths[0].edges) > 1:
            continue
        if not _paths_temporally_compatible([positive_paths[0], negative_paths[0]]):
            continue
        contradictions.append(
            {
                "type": "direct_opposition",
                "severity": "hard",
                "source": source,
                "targets": [base, negated_node],
                "context_id": context_id,
                "store_id": store_id,
                "fact_ids": _path_fact_ids([positive_paths[0], negative_paths[0]]),
                "reason": "The same source implies a proposition and its explicit negation.",
            }
        )
    return contradictions


def _confidence_tensions(relations: list[RelationRecord]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, RelationType, str, str], list[RelationRecord]] = {}
    for relation in relations:
        key = (
            relation.source,
            relation.target,
            relation.relation_type,
            relation.context_id,
            relation.store_id,
        )
        grouped.setdefault(key, []).append(relation)

    tensions: list[dict[str, Any]] = []
    for (source, target, relation_type, context_id, store_id), records in sorted(grouped.items()):
        if len(records) < 2:
            continue
        confidences = [record.confidence for record in records]
        if max(confidences) - min(confidences) < 0.5:
            continue
        tensions.append(
            {
                "type": "confidence_tension",
                "severity": "soft",
                "source": source,
                "target": target,
                "relation_type": relation_type.value,
                "context_id": context_id,
                "store_id": store_id,
                "fact_ids": [record.id for record in records],
                "confidences": confidences,
                "reason": "Multiple evidence records for the same claim have divergent confidence.",
            }
        )
    return tensions


def _relations_by_scope(
    relations: list[RelationRecord],
) -> dict[tuple[str, str], list[RelationRecord]]:
    grouped: dict[tuple[str, str], list[RelationRecord]] = {}
    for relation in relations:
        grouped.setdefault((relation.store_id, relation.context_id), []).append(relation)
    return grouped


def _explicit_negation_base(value: str) -> str | None:
    stripped = value.strip()
    if stripped.startswith("¬"):
        base = stripped[1:].strip()
        return base or None
    lowered = stripped.lower()
    if lowered.startswith("not:"):
        base = stripped[4:].strip()
        return base or None
    if lowered.startswith("not "):
        base = stripped[4:].strip()
        return base or None
    return None


def _path_fact_ids(paths: list[ReasoningPath]) -> list[str]:
    return list(dict.fromkeys(edge.relation_id for path in paths for edge in path.edges))


def _dedupe_contradictions(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[Any, ...]] = set()
    deduped: list[dict[str, Any]] = []
    for item in items:
        key = (
            item["type"],
            item.get("source"),
            tuple(item.get("targets", [])),
            item.get("context_id"),
            item.get("store_id"),
            tuple(item.get("fact_ids", [])),
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


def _reachable_group_members(
    index: GraphIndex,
    source: str,
    members: list[str],
    max_depth: int,
) -> dict[str, ReasoningPath]:
    hits: dict[str, ReasoningPath] = {}
    for member in members:
        paths = index.find_paths(
            source,
            member,
            max_depth=max_depth,
            strategy=PathStrategy.SHORTEST,
            max_paths=1,
            confidence_policy=ConfidencePolicy.PRODUCT_INDEPENDENT,
        )
        if paths:
            hits[member] = paths[0]
    return hits


def _dedupe_context_separated(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[Any, ...]] = set()
    deduped: list[dict[str, Any]] = []
    for item in items:
        key = (
            item["source"],
            tuple(item["targets"]),
            item["exclusive_group_id"],
            tuple(item["contexts"]),
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


def _valid_at_matches(relation: RelationRecord, valid_at: datetime) -> bool:
    temporal = relation.temporal or {}
    valid_from = _parse_temporal_value(temporal.get("valid_from"), valid_at)
    valid_to = _parse_temporal_value(temporal.get("valid_to"), valid_at)
    if valid_from is not None and valid_at < valid_from:
        return False
    return not (valid_to is not None and valid_at > valid_to)


def _parse_temporal_value(value: Any, reference: datetime) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str):
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    else:
        return None

    if parsed.tzinfo is None and reference.tzinfo is not None:
        return parsed.replace(tzinfo=reference.tzinfo)
    if parsed.tzinfo is not None and reference.tzinfo is None:
        return parsed.replace(tzinfo=None)
    return parsed


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
