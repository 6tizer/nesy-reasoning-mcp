import json

import pytest

from nesy_reasoning_mcp.auto_ingest import (
    CandidateRelation,
    EvidenceRecord,
    GateAction,
    GateResult,
    ReviewDecision,
    ReviewDecisionValue,
    ReviewQueueRecord,
    ReviewQueueStatus,
)
from nesy_reasoning_mcp.store import RelationStore
from nesy_reasoning_mcp.tools import (
    COMMIT_REVIEWED_RELATIONS,
    LIST_REVIEW_QUEUE,
    RESOLVE_REVIEW_QUEUE,
    call_tool,
)


def _candidate(candidate_id: str = "candidate-1") -> CandidateRelation:
    return CandidateRelation(
        id=candidate_id,
        source="A",
        target="B",
        relation_type="sufficient",
        confidence=0.9,
        evidence=[EvidenceRecord(url="https://example.com/source", span="A enables B.")],
    )


def _review(candidate_id: str = "candidate-1") -> ReviewDecision:
    return ReviewDecision(
        candidate_id=candidate_id,
        decision=ReviewDecisionValue.APPROVE,
        final_relation_type="sufficient",
        final_confidence=0.9,
        reasons=["Evidence directly supports the relation."],
    )


def _queue_record(
    record_id: str = "queue-1",
    *,
    with_review: bool = True,
) -> ReviewQueueRecord:
    candidate = _candidate()
    return ReviewQueueRecord(
        id=record_id,
        run_id="run-1",
        candidate=candidate,
        review=_review(candidate.id) if with_review else None,
        gate_result=GateResult(candidate_id=candidate.id, action=GateAction.QUEUE),
    )


@pytest.mark.asyncio
async def test_list_review_queue_does_not_mutate_graph_memory() -> None:
    store = RelationStore()
    store.enqueue_review_queue([_queue_record()])

    result = await call_tool(LIST_REVIEW_QUEUE, {}, store)

    assert result.isError is False
    assert result.structuredContent["records"][0]["id"] == "queue-1"
    assert store.list_relations() == []
    assert json.loads(result.content[0].text) == result.structuredContent


@pytest.mark.asyncio
async def test_commit_reviewed_relations_requires_explicit_ids_and_marks_committed() -> None:
    store = RelationStore()
    store.enqueue_review_queue([_queue_record()])

    result = await call_tool(COMMIT_REVIEWED_RELATIONS, {"ids": ["queue-1"]}, store)

    listed = store.list_review_queue()
    assert result.isError is False
    assert result.structuredContent["status"] == "ok"
    assert result.structuredContent["committed_count"] == 1
    assert len(store.list_relations()) == 1
    assert listed[0].status == ReviewQueueStatus.COMMITTED
    assert listed[0].committed_relation_ids == result.structuredContent["relation_ids"]


@pytest.mark.asyncio
async def test_commit_reviewed_relations_revalidates_and_leaves_pending_when_blocked() -> None:
    store = RelationStore()
    store.enqueue_review_queue([_queue_record(with_review=False)])

    result = await call_tool(COMMIT_REVIEWED_RELATIONS, {"ids": ["queue-1"]}, store)

    assert result.isError is False
    assert result.structuredContent["status"] == "warning"
    assert result.structuredContent["committed_count"] == 0
    assert store.list_review_queue()[0].status == ReviewQueueStatus.PENDING
    assert store.list_relations() == []


@pytest.mark.asyncio
async def test_resolve_review_queue_is_explicit_and_audited() -> None:
    store = RelationStore()
    store.enqueue_review_queue([_queue_record()])

    result = await call_tool(
        RESOLVE_REVIEW_QUEUE,
        {"ids": ["queue-1"], "reason": "duplicate candidate"},
        store,
    )

    listed = store.list_review_queue()
    assert result.isError is False
    assert result.structuredContent["resolved_count"] == 1
    assert listed[0].status == ReviewQueueStatus.RESOLVED
    assert listed[0].resolution["reason"] == "duplicate candidate"
    assert store.list_relations() == []
    assert store.list_audit_entries()[-1]["tool_name"] == RESOLVE_REVIEW_QUEUE
