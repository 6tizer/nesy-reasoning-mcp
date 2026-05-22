"""Classify and chain verification tool handlers."""

from __future__ import annotations

from typing import Any

from nesy_reasoning_mcp.reasoning import (
    build_graph,
    classify_reachability,
    expected_relation_matches,
    path_to_dict,
)
from nesy_reasoning_mcp.schemas import (
    Classification,
    ClassifyInput,
    ContextFilter,
    Diagnostic,
    ExclusiveGroupRecord,
    ExpectedRelation,
    IndependenceRecord,
    PathStrategy,
    RelationRecord,
    VerifyChainInput,
)
from nesy_reasoning_mcp.store import RelationStore
from nesy_reasoning_mcp.tool_independence import _path_independence_from_if_not


async def classify(arguments: dict[str, Any], store: RelationStore) -> dict[str, Any]:
    """Handle `nesy.classify`."""
    payload = ClassifyInput.model_validate(arguments)
    index = build_graph(store.list_relations(), payload.context_filter)
    independence_records = store.list_independence_records()
    exclusive_groups = store.list_exclusive_groups()
    fwd_paths = index.find_paths(
        payload.source,
        payload.target,
        max_depth=payload.max_depth,
        max_paths=5,
        confidence_policy=payload.confidence_policy,
        direct_only=payload.require_direct,
    )
    rev_paths = index.find_paths(
        payload.target,
        payload.source,
        max_depth=payload.max_depth,
        max_paths=5,
        confidence_policy=payload.confidence_policy,
        direct_only=payload.require_direct,
    )
    classification = classify_reachability(fwd_paths, rev_paths)

    return {
        "status": "ok",
        "source": payload.source,
        "target": payload.target,
        "classification": classification.value,
        "source_implies_target": _implication_result(fwd_paths, payload.source, payload.target),
        "target_implies_source": _implication_result(rev_paths, payload.target, payload.source),
        "necessity_status": _necessity_status(
            rev_paths,
            source=payload.source,
            target=payload.target,
            index=index,
            context_filter=payload.context_filter,
            max_depth=payload.max_depth,
            confidence_policy=payload.confidence_policy,
            direct_only=payload.require_direct,
            relation_map={relation.id: relation for relation in index.relations},
            independence_records=independence_records,
            exclusive_groups=exclusive_groups,
        ),
        "direct_relations": index.direct_relations_between(payload.source, payload.target),
        "paths": _classify_paths(fwd_paths, rev_paths, payload.include_paths),
        "diagnostics": [],
        "trace": _classify_trace(
            payload.source,
            payload.target,
            classification,
            fwd_paths,
            rev_paths,
        ),
        "graph_stats": index.graph_stats,
    }


async def verify_chain(arguments: dict[str, Any], store: RelationStore) -> dict[str, Any]:
    """Handle `nesy.verify_chain`."""
    payload = VerifyChainInput.model_validate(arguments)
    if payload.chain is not None and (
        payload.chain[0] != payload.source or payload.chain[-1] != payload.target
    ):
        diagnostic = Diagnostic(
            level="error",
            code="CHAIN_ENDPOINT_MISMATCH",
            message="Explicit chain must start at source and end at target.",
        )
        return {
            "status": "error",
            "reachable": False,
            "relation_type": Classification.UNKNOWN.value,
            "logic_validity": False,
            "best_path": None,
            "paths": [],
            "broken_at": None,
            "diagnostics": [diagnostic.model_dump(mode="json")],
            "trace": ["Rejected explicit chain with mismatched endpoints."],
            "graph_stats": store.graph_stats().model_dump(mode="json"),
        }

    index = build_graph(store.list_relations(), payload.context_filter)
    fwd_paths = index.find_paths(
        payload.source,
        payload.target,
        max_depth=payload.max_depth,
        strategy=payload.path_strategy,
        max_paths=payload.max_paths,
        confidence_policy=payload.confidence_policy,
    )
    rev_paths = index.find_paths(
        payload.target,
        payload.source,
        max_depth=payload.max_depth,
        strategy=payload.path_strategy,
        max_paths=payload.max_paths,
        confidence_policy=payload.confidence_policy,
    )
    classification = classify_reachability(fwd_paths, rev_paths)

    if payload.chain is not None:
        path, broken = index.verify_explicit_chain(
            payload.chain,
            confidence_policy=payload.confidence_policy,
        )
        return _explicit_chain_result(payload, index.graph_stats, classification, path, broken)

    return _searched_chain_result(payload, index.graph_stats, classification, fwd_paths, rev_paths)


def _implication_result(paths: list, start: str, end: str) -> dict[str, Any]:
    if paths:
        best = paths[0]
        return {
            "proven": True,
            "logic_validity": True,
            "evidence_confidence": best.evidence_confidence,
            "best_path": best.nodes,
        }
    return {
        "proven": False,
        "logic_validity": False,
        "reason": f"No path found from {start} to {end} within max_depth under the context filter.",
    }


def _necessity_status(
    reverse_paths: list,
    *,
    source: str,
    target: str,
    index: Any,
    context_filter: ContextFilter,
    max_depth: int,
    confidence_policy: Any,
    direct_only: bool,
    relation_map: dict[str, RelationRecord],
    independence_records: list[IndependenceRecord],
    exclusive_groups: list[ExclusiveGroupRecord],
) -> dict[str, Any]:
    if reverse_paths:
        return {
            "status": "proven_necessary",
            "reason": "Target implies source under the current graph and context filter.",
        }
    counterexample = _non_necessity_counterexample(
        source,
        target,
        index,
        context_filter,
        max_depth=max_depth,
        confidence_policy=confidence_policy,
        direct_only=direct_only,
        relation_map=relation_map,
        independence_records=independence_records,
        exclusive_groups=exclusive_groups,
    )
    if counterexample is not None:
        return counterexample
    return {
        "status": "unknown",
        "reason": (
            "Alternative sufficient causes do not disprove necessity unless independence "
            "or counterexample is established."
        ),
    }


def _non_necessity_counterexample(
    source: str,
    target: str,
    index: Any,
    context_filter: ContextFilter,
    *,
    max_depth: int,
    confidence_policy: Any,
    direct_only: bool,
    relation_map: dict[str, RelationRecord],
    independence_records: list[IndependenceRecord],
    exclusive_groups: list[ExclusiveGroupRecord],
) -> dict[str, Any] | None:
    if source == target:
        return None
    for candidate in sorted(str(node) for node in index.graph.nodes):
        if candidate in {source, target}:
            continue
        paths = index.find_paths(
            candidate,
            target,
            max_depth=max_depth,
            strategy=PathStrategy.BEST_CONFIDENCE,
            max_paths=1,
            confidence_policy=confidence_policy,
            direct_only=direct_only,
        )
        for path in paths:
            if not path.edges or source in path.nodes:
                continue
            if (
                _path_independence_from_if_not(
                    path,
                    relation_map,
                    independence_records,
                    exclusive_groups,
                    source,
                    context_filter,
                )
                != "proven"
            ):
                continue
            return {
                "status": "proven_not_necessary",
                "reason": "Found an independent counterexample path to target.",
                "counterexample": candidate,
                "proof": f"{candidate} -> {target} and {candidate} independent_of {source}",
                "path": path_to_dict(path),
            }
    return None


def _classify_paths(fwd_paths: list, rev_paths: list, include_paths: bool) -> list[dict[str, Any]]:
    if not include_paths:
        return []
    paths: list[dict[str, Any]] = []
    paths.extend(
        {
            "direction": "source_to_target",
            **path_to_dict(path, relation_type=Classification.SUFFICIENT.value),
        }
        for path in fwd_paths
    )
    paths.extend(
        {
            "direction": "target_to_source",
            **path_to_dict(path, relation_type=Classification.NECESSARY.value),
        }
        for path in rev_paths
    )
    return paths


def _classify_trace(
    source: str,
    target: str,
    classification: Classification,
    fwd_paths: list,
    rev_paths: list,
) -> list[str]:
    if source == target:
        return ["Source and target are identical; using zero-length identity path."]
    trace = [f"Checked implication paths between {source} and {target}."]
    source_path = (
        f"Found source-to-target path: {fwd_paths[0].nodes}"
        if fwd_paths
        else "No source-to-target path found."
    )
    target_path = (
        f"Found target-to-source path: {rev_paths[0].nodes}"
        if rev_paths
        else "No target-to-source path found."
    )
    trace.append(source_path)
    trace.append(target_path)
    trace.append(f"Mapped reachability to classification: {classification.value}.")
    return trace


def _explicit_chain_result(
    payload: VerifyChainInput,
    graph_stats: dict[str, Any],
    classification: Classification,
    path: Any,
    broken: Any,
) -> dict[str, Any]:
    diagnostics = []
    reachable = path is not None
    expected_ok = reachable and expected_relation_matches(payload.expected_relation, classification)
    if broken is not None and broken.direction_mismatch:
        diagnostics.append(
            Diagnostic(
                level="warning",
                code="DIRECTION_MISMATCH",
                message="The declared relation points in the reverse implication direction.",
            ).model_dump(mode="json")
        )
    if reachable and not expected_ok:
        diagnostics.append(_expected_mismatch(payload.expected_relation, classification))
    return {
        "status": "ok",
        "reachable": reachable,
        "relation_type": classification.value,
        "logic_validity": expected_ok,
        "best_path": path_to_dict(path) if path is not None else None,
        "paths": [],
        "broken_at": broken.to_dict() if broken is not None else None,
        "diagnostics": diagnostics,
        "trace": _explicit_chain_trace(payload.chain or [], path, broken, expected_ok),
        "graph_stats": graph_stats,
    }


def _searched_chain_result(
    payload: VerifyChainInput,
    graph_stats: dict[str, Any],
    classification: Classification,
    fwd_paths: list,
    rev_paths: list,
) -> dict[str, Any]:
    reachable = classification != Classification.UNKNOWN
    expected_ok = expected_relation_matches(payload.expected_relation, classification)
    best_path = fwd_paths[0] if fwd_paths else (rev_paths[0] if rev_paths else None)
    diagnostics = (
        [] if expected_ok else [_expected_mismatch(payload.expected_relation, classification)]
    )
    all_paths = []
    if payload.path_strategy.value == "all":
        all_paths.extend(
            {"direction": "source_to_target", **path_to_dict(path)} for path in fwd_paths
        )
        all_paths.extend(
            {"direction": "target_to_source", **path_to_dict(path)} for path in rev_paths
        )
    diagnostics_out = (
        diagnostics if reachable or payload.expected_relation != ExpectedRelation.ANY else []
    )
    return {
        "status": "ok",
        "reachable": reachable,
        "relation_type": classification.value,
        "logic_validity": reachable and expected_ok,
        "best_path": path_to_dict(best_path) if best_path is not None else None,
        "paths": all_paths,
        "broken_at": None,
        "diagnostics": diagnostics_out,
        "trace": _searched_chain_trace(payload.source, payload.target, classification, best_path),
        "graph_stats": graph_stats,
    }


def _expected_mismatch(
    expected: ExpectedRelation,
    classification: Classification,
) -> dict[str, Any]:
    return Diagnostic(
        level="warning",
        code="EXPECTED_RELATION_MISMATCH",
        message=(
            f"Expected relation {expected.value} did not match classified relation "
            f"{classification.value}."
        ),
    ).model_dump(mode="json")


def _explicit_chain_trace(
    chain: list[str],
    path: Any,
    broken: Any,
    expected_ok: bool,
) -> list[str]:
    if broken is not None:
        return [
            f"{broken.from_node} -> {broken.to_node} not found.",
            f"Chain is broken at step {broken.index}.",
        ]
    trace = [f"Explicit chain verified: {' -> '.join(chain)}."]
    trace.append(
        "Expected relation matched." if expected_ok else "Expected relation did not match."
    )
    if path is not None:
        trace.append(f"Verified {len(path.edges)} edge(s).")
    return trace


def _searched_chain_trace(
    source: str,
    target: str,
    classification: Classification,
    best_path: Any,
) -> list[str]:
    trace = [f"Searched implication graph between {source} and {target}."]
    if best_path is not None:
        trace.append(f"Found path with {len(best_path.edges)} edge(s): {best_path.nodes}.")
    else:
        trace.append("No valid implication path found in either direction.")
    trace.append(f"Mapped reachability to relation_type: {classification.value}.")
    return trace
