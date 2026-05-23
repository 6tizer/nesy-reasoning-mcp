"""Ephemeral reasoning over caller-supplied relation candidates."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from nesy_reasoning_mcp.schemas import (
    CheckContradictionsInput,
    ClassifyInput,
    CounterfactualInput,
    ExclusiveGroupRecord,
    IndependenceRecord,
    ReasonOverRelationsCheckContradictionsQuery,
    ReasonOverRelationsClassifyQuery,
    ReasonOverRelationsCounterfactualQuery,
    ReasonOverRelationsInput,
    ReasonOverRelationsSummarizeGraphQuery,
    ReasonOverRelationsVerifyChainQuery,
    RelationRecord,
    SummarizeGraphInput,
    VerifyChainInput,
)
from nesy_reasoning_mcp.storage.common import _normalize_relation_identities
from nesy_reasoning_mcp.storage.memory import MemoryRelationStore
from nesy_reasoning_mcp.store import RelationStoreProtocol
from nesy_reasoning_mcp.tool_counterfactual import counterfactual
from nesy_reasoning_mcp.tool_reasoning import classify, verify_chain
from nesy_reasoning_mcp.tool_relations import check_contradictions
from nesy_reasoning_mcp.tool_summary import summarize_graph


async def reason_over_relations(
    arguments: dict[str, Any],
    store: RelationStoreProtocol,
) -> dict[str, Any]:
    """Handle `nesy.reason_over_relations` without writing to the persistent store."""
    payload = ReasonOverRelationsInput.model_validate(arguments)
    ephemeral_store = _ephemeral_store(payload, store)
    result = await _run_query(payload, ephemeral_store)
    trace = [
        (
            "Built ephemeral graph with "
            f"{len(payload.relations)} relation(s), "
            f"{len(payload.exclusive_groups)} exclusive group(s), "
            f"{len(payload.propositions)} proposition(s)."
        )
    ]
    trace.extend(result.get("trace", []))
    return {
        "status": result.get("status", "ok"),
        "mode": payload.query.mode,
        "persisted": False,
        "result": result,
        "relation_count": len(payload.relations),
        "exclusive_group_count": len(payload.exclusive_groups),
        "proposition_count": len(payload.propositions),
        "diagnostics": result.get("diagnostics", []),
        "trace": trace,
        "graph_stats": result.get(
            "graph_stats",
            ephemeral_store.graph_stats().model_dump(mode="json"),
        ),
    }


def _ephemeral_store(
    payload: ReasonOverRelationsInput,
    store: RelationStoreProtocol,
) -> MemoryRelationStore:
    return _EphemeralRelationStore(payload, store)


class _EphemeralRelationStore(MemoryRelationStore):
    """Read-only in-memory store for one reason_over_relations call."""

    def __init__(
        self,
        payload: ReasonOverRelationsInput,
        store: RelationStoreProtocol,
    ) -> None:
        super().__init__(store.config)
        normalized_relations = _normalize_relation_identities(
            payload.relations,
            payload.propositions,
        )
        self._relations = [RelationRecord.from_input(relation) for relation in normalized_relations]
        self._exclusive_groups = [
            ExclusiveGroupRecord.from_input(group) for group in payload.exclusive_groups
        ]
        self._independence_records = [
            IndependenceRecord.from_input(record) for record in payload.independence_records
        ]
        self._propositions = [
            proposition.model_copy(deep=True) for proposition in payload.propositions
        ]
        self._context_metadata = deepcopy(payload.context_metadata)

    def assert_relations(self, *args: Any, **kwargs: Any) -> Any:
        """Reject writes to the ephemeral store."""
        raise RuntimeError("ephemeral relation store is read-only")

    def assert_exclusive(self, *args: Any, **kwargs: Any) -> Any:
        """Reject writes to the ephemeral store."""
        raise RuntimeError("ephemeral relation store is read-only")

    def clear_relations(self, *args: Any, **kwargs: Any) -> Any:
        """Reject writes to the ephemeral store."""
        raise RuntimeError("ephemeral relation store is read-only")

    def import_records(self, *args: Any, **kwargs: Any) -> Any:
        """Reject writes to the ephemeral store."""
        raise RuntimeError("ephemeral relation store is read-only")


async def _run_query(
    payload: ReasonOverRelationsInput,
    store: RelationStoreProtocol,
) -> dict[str, Any]:
    query = payload.query
    if isinstance(query, ReasonOverRelationsClassifyQuery):
        return await classify(_classify_arguments(payload, query), store)
    if isinstance(query, ReasonOverRelationsVerifyChainQuery):
        return await verify_chain(_verify_chain_arguments(payload, query), store)
    if isinstance(query, ReasonOverRelationsCounterfactualQuery):
        return await counterfactual(_counterfactual_arguments(payload, query), store)
    if isinstance(query, ReasonOverRelationsCheckContradictionsQuery):
        return await check_contradictions(_check_contradictions_arguments(payload, query), store)
    if isinstance(query, ReasonOverRelationsSummarizeGraphQuery):
        return await summarize_graph(_summarize_graph_arguments(payload, query), store)
    raise ValueError(f"unsupported reason_over_relations mode: {query.mode}")


def _classify_arguments(
    payload: ReasonOverRelationsInput,
    query: ReasonOverRelationsClassifyQuery,
) -> dict[str, Any]:
    return ClassifyInput(
        source=query.source,
        target=query.target,
        context_filter=payload.context_filter,
        max_depth=payload.max_depth,
        include_paths=query.include_paths,
        require_direct=query.require_direct,
        confidence_policy=payload.confidence_policy,
        min_confidence=payload.min_confidence,
    ).model_dump(mode="json")


def _verify_chain_arguments(
    payload: ReasonOverRelationsInput,
    query: ReasonOverRelationsVerifyChainQuery,
) -> dict[str, Any]:
    return VerifyChainInput(
        source=query.source,
        target=query.target,
        chain=query.chain,
        expected_relation=query.expected_relation,
        context_filter=payload.context_filter,
        max_depth=payload.max_depth,
        path_strategy=query.path_strategy,
        max_paths=query.max_paths,
        confidence_policy=payload.confidence_policy,
        min_confidence=payload.min_confidence,
    ).model_dump(mode="json")


def _counterfactual_arguments(
    payload: ReasonOverRelationsInput,
    query: ReasonOverRelationsCounterfactualQuery,
) -> dict[str, Any]:
    return CounterfactualInput(
        if_not=query.if_not,
        targets=query.targets,
        context_filter=payload.context_filter,
        world_mode=query.world_mode,
        max_depth=payload.max_depth,
        include_alternative_paths=query.include_alternative_paths,
        confidence_policy=payload.confidence_policy,
        min_confidence=payload.min_confidence,
    ).model_dump(mode="json")


def _check_contradictions_arguments(
    payload: ReasonOverRelationsInput,
    query: ReasonOverRelationsCheckContradictionsQuery,
) -> dict[str, Any]:
    return CheckContradictionsInput(
        facts=query.facts,
        propositions=payload.propositions,
        mode=query.contradiction_mode,
        context_filter=payload.context_filter,
        include_soft=query.include_soft,
        max_depth=payload.max_depth,
        min_confidence=payload.min_confidence,
    ).model_dump(mode="json")


def _summarize_graph_arguments(
    payload: ReasonOverRelationsInput,
    query: ReasonOverRelationsSummarizeGraphQuery,
) -> dict[str, Any]:
    return SummarizeGraphInput(
        focus_terms=query.focus_terms,
        context_filter=payload.context_filter,
        max_relations=query.max_relations,
        max_chars=query.max_chars,
        include_exclusives=query.include_exclusives,
    ).model_dump(mode="json")
