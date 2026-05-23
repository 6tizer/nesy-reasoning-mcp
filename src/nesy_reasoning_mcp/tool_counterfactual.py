"""Counterfactual tool handler."""

from __future__ import annotations

from typing import Any

from nesy_reasoning_mcp.reasoning import build_graph, path_to_dict
from nesy_reasoning_mcp.schemas import (
    DEFAULT_CONTEXT_ID,
    CounterfactualInput,
    Diagnostic,
    ExclusiveGroupRecord,
    IndependenceRecord,
    PathStrategy,
    RelationRecord,
    RelationType,
    WorldMode,
)
from nesy_reasoning_mcp.store import RelationStoreProtocol
from nesy_reasoning_mcp.tool_independence import _path_independence_from_if_not


async def counterfactual(arguments: dict[str, Any], store: RelationStoreProtocol) -> dict[str, Any]:
    """Handle `nesy.counterfactual`."""
    payload = CounterfactualInput.model_validate(arguments)
    index = build_graph(store.list_relations(), payload.context_filter)
    targets = _counterfactual_targets(payload, index.graph.nodes)
    relation_map = {relation.id: relation for relation in index.relations}
    independence_records = store.list_independence_records()
    exclusive_groups = store.list_exclusive_groups()
    diagnostics = _counterfactual_diagnostics(payload, store.context_metadata())
    necessarily_blocked: list[dict[str, Any]] = []
    possibly_blocked: list[dict[str, Any]] = []
    still_possible: list[dict[str, Any]] = []
    unknown: list[dict[str, Any]] = []
    not_derivably_affected: list[str] = []
    trace = [f"Assume not {payload.if_not}."]

    for target in targets:
        if target == payload.if_not:
            necessarily_blocked.append(_counterfactual_self_blocked(payload.if_not))
            trace.append(f"{target} is directly blocked by the intervention.")
            continue

        dependency_paths = index.find_paths(
            target,
            payload.if_not,
            max_depth=payload.max_depth,
            strategy=PathStrategy.BEST_CONFIDENCE,
            max_paths=1,
            confidence_policy=payload.confidence_policy,
            min_confidence=payload.min_confidence,
        )
        if dependency_paths:
            necessarily_blocked.append(
                _counterfactual_necessary_item(payload.if_not, target, dependency_paths[0])
            )
            trace.append(f"{target} implies {payload.if_not}; classified as necessarily_blocked.")
            continue

        blocked_paths = index.find_paths(
            payload.if_not,
            target,
            max_depth=payload.max_depth,
            strategy=PathStrategy.BEST_CONFIDENCE,
            max_paths=1,
            confidence_policy=payload.confidence_policy,
            min_confidence=payload.min_confidence,
        )
        if blocked_paths:
            alternatives = _counterfactual_alternative_paths(
                index,
                relation_map,
                independence_records,
                exclusive_groups,
                target=target,
                if_not=payload.if_not,
                payload=payload,
            )
            proven_alternatives = [
                alternative
                for alternative in alternatives
                if alternative["independence_from_if_not"] == "proven"
            ]
            if proven_alternatives:
                still_possible.append(
                    _counterfactual_still_possible_item(
                        payload.if_not,
                        target,
                        blocked_paths[0],
                        proven_alternatives,
                        payload.include_alternative_paths,
                    )
                )
                trace.append(
                    f"{target} has an alternative path proven independent of {payload.if_not}."
                )
            else:
                possibly_blocked.append(
                    _counterfactual_possible_item(
                        payload.if_not,
                        target,
                        blocked_paths[0],
                        alternatives,
                        payload.include_alternative_paths,
                    )
                )
                trace.append(f"{payload.if_not} implies {target}; classified as possibly_blocked.")
            continue

        unknown.append(_counterfactual_unknown_item(payload.if_not, target))
        not_derivably_affected.append(target)
        trace.append(f"No dependency path found between {payload.if_not} and {target}.")

    if payload.world_mode == WorldMode.CLOSED and _causal_completeness_declared(
        payload,
        store.context_metadata(),
    ):
        possibly_blocked, closed_upgrades = _closed_world_upgrades(
            payload,
            index,
            possibly_blocked,
        )
        necessarily_blocked.extend(closed_upgrades)
        if closed_upgrades:
            trace.append(
                "Closed-world causal_completeness used to upgrade possible blocks to necessary."
            )

    return {
        "status": "warning" if any(item.level == "warning" for item in diagnostics) else "ok",
        "if_not": payload.if_not,
        "world_mode": payload.world_mode.value,
        "necessarily_blocked": necessarily_blocked,
        "possibly_blocked": possibly_blocked,
        "still_possible": still_possible,
        "unknown": unknown,
        "not_derivably_affected": not_derivably_affected,
        "diagnostics": [diagnostic.model_dump(mode="json") for diagnostic in diagnostics],
        "trace": trace,
        "graph_stats": index.graph_stats,
    }


def _counterfactual_targets(payload: CounterfactualInput, nodes: Any) -> list[str]:
    if payload.targets:
        return payload.targets
    return sorted(str(node) for node in nodes if str(node) != payload.if_not)


def _counterfactual_diagnostics(
    payload: CounterfactualInput,
    context_metadata: dict[str, Any],
) -> list[Diagnostic]:
    if payload.world_mode == WorldMode.OPEN:
        return [
            Diagnostic(
                level="info",
                code="OPEN_WORLD_DEFAULT",
                message="No missing or alternative path is treated as proof of impossibility.",
            )
        ]
    if _causal_completeness_declared(payload, context_metadata):
        return [
            Diagnostic(
                level="info",
                code="CLOSED_WORLD_CAUSAL_COMPLETENESS_DECLARED",
                message="Closed-world upgrades may use context causal_completeness metadata.",
            )
        ]
    return [
        Diagnostic(
            level="warning",
            code="CLOSED_WORLD_COMPLETENESS_NOT_DECLARED",
            message="Closed-world mode requested, but context causal_completeness is not true.",
        )
    ]


def _counterfactual_self_blocked(if_not: str) -> dict[str, Any]:
    return {
        "target": if_not,
        "proof": {
            "type": "intervention",
            "path": [if_not],
            "meaning": f"{if_not} is assumed false by the counterfactual intervention.",
        },
        "logic_validity": True,
        "evidence_confidence": 1.0,
    }


def _counterfactual_necessary_item(if_not: str, target: str, path: Any) -> dict[str, Any]:
    return {
        "target": target,
        "proof": {
            "type": "necessary_condition",
            "path": path.nodes,
            "meaning": f"{target} implies {if_not}; therefore not {if_not} implies not {target}.",
        },
        "logic_validity": True,
        "evidence_confidence": path.evidence_confidence,
    }


def _counterfactual_possible_item(
    if_not: str,
    target: str,
    blocked_path: Any,
    alternative_paths: list[dict[str, Any]],
    include_alternative_paths: bool,
) -> dict[str, Any]:
    item = {
        "target": target,
        "reason": (
            f"{if_not} is a sufficient path to {target}, but absence of {if_not} does not "
            "logically imply absence of the target under open-world semantics."
        ),
        "blocked_path": blocked_path.nodes,
        "evidence_confidence": blocked_path.evidence_confidence,
    }
    if include_alternative_paths:
        item["alternative_paths"] = alternative_paths
    return item


def _counterfactual_still_possible_item(
    if_not: str,
    target: str,
    blocked_path: Any,
    alternative_paths: list[dict[str, Any]],
    include_alternative_paths: bool,
) -> dict[str, Any]:
    item = {
        "target": target,
        "reason": f"At least one alternative sufficient path is proven independent of {if_not}.",
        "blocked_path": blocked_path.nodes,
        "logic_validity": True,
        "evidence_confidence": blocked_path.evidence_confidence,
    }
    if include_alternative_paths:
        item["alternative_paths"] = alternative_paths
    return item


def _counterfactual_unknown_item(if_not: str, target: str) -> dict[str, Any]:
    return {
        "target": target,
        "reason": (
            f"No necessary dependency on {if_not} and no sufficient path from {if_not} was "
            "found. This is unknown, not proven unaffected."
        ),
    }


def _counterfactual_alternative_paths(
    index: Any,
    relation_map: dict[str, RelationRecord],
    independence_records: list[IndependenceRecord],
    exclusive_groups: list[ExclusiveGroupRecord],
    *,
    target: str,
    if_not: str,
    payload: CounterfactualInput,
) -> list[dict[str, Any]]:
    alternatives: list[dict[str, Any]] = []
    seen: set[tuple[str, ...]] = set()
    for start in sorted(str(node) for node in index.graph.nodes):
        if start in {if_not, target}:
            continue
        paths = index.find_paths(
            start,
            target,
            max_depth=payload.max_depth,
            strategy=PathStrategy.BEST_CONFIDENCE,
            max_paths=1,
            confidence_policy=payload.confidence_policy,
            min_confidence=payload.min_confidence,
        )
        for path in paths:
            key = tuple(path.nodes)
            if not path.edges or if_not in path.nodes or key in seen:
                continue
            seen.add(key)
            item = path_to_dict(path)
            item["independence_from_if_not"] = _path_independence_from_if_not(
                path,
                relation_map,
                independence_records,
                exclusive_groups,
                if_not,
                payload.context_filter,
            )
            alternatives.append(item)
    return alternatives[:5]


def _causal_completeness_declared(
    payload: CounterfactualInput,
    context_metadata: dict[str, Any],
) -> bool:
    context_id = payload.context_filter.context_id or DEFAULT_CONTEXT_ID
    metadata = context_metadata.get(context_id, {})
    return isinstance(metadata, dict) and metadata.get("causal_completeness") is True


def _closed_world_upgrades(
    payload: CounterfactualInput,
    index: Any,
    possibly_blocked: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    remaining: list[dict[str, Any]] = []
    upgraded: list[dict[str, Any]] = []
    for item in possibly_blocked:
        target = item["target"]
        causes = _direct_sufficient_causes(index, target)
        if causes and all(_counterfactual_cause_blocked(index, payload, cause) for cause in causes):
            upgraded.append(
                {
                    "target": target,
                    "proof": {
                        "type": "closed_world_all_causes_blocked",
                        "causes": causes,
                        "meaning": (
                            "Closed-world causal completeness is declared and all known direct "
                            "sufficient causes are blocked."
                        ),
                    },
                    "logic_validity": True,
                    "blocked_path": item.get("blocked_path"),
                    "evidence_confidence": item.get("evidence_confidence"),
                }
            )
        else:
            remaining.append(item)
    return remaining, upgraded


def _direct_sufficient_causes(index: Any, target: str) -> list[str]:
    edges = [
        edge
        for edge in index.edges
        if edge.consequent == target
        and edge.source_relation_type in {RelationType.SUFFICIENT, RelationType.EQUIVALENT}
    ]
    causes: dict[str, None] = {}
    for edge in sorted(edges, key=lambda item: (item.antecedent, item.relation_id)):
        causes[edge.antecedent] = None
    return list(causes)


def _counterfactual_cause_blocked(index: Any, payload: CounterfactualInput, cause: str) -> bool:
    if cause == payload.if_not:
        return True
    return bool(
        index.find_paths(
            cause,
            payload.if_not,
            max_depth=payload.max_depth,
            strategy=PathStrategy.BEST_CONFIDENCE,
            max_paths=1,
            confidence_policy=payload.confidence_policy,
            min_confidence=payload.min_confidence,
        )
    )
