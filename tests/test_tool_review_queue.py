import json
from typing import Any

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
from nesy_reasoning_mcp.schemas import RelationInput
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
        normalized_implication_supported=True,
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
async def test_list_review_queue_uses_keyset_cursor_without_duplicates() -> None:
    store = RelationStore()
    store.enqueue_review_queue(
        [
            _queue_record("queue-1").model_copy(update={"created_at": "2026-01-01T00:00:01+00:00"}),
            _queue_record("queue-2").model_copy(update={"created_at": "2026-01-01T00:00:02+00:00"}),
            _queue_record("queue-3").model_copy(update={"created_at": "2026-01-01T00:00:03+00:00"}),
        ]
    )

    first = await call_tool(LIST_REVIEW_QUEUE, {"limit": 2}, store)
    store.enqueue_review_queue(
        [_queue_record("queue-0").model_copy(update={"created_at": "2026-01-01T00:00:00+00:00"})]
    )
    second = await call_tool(
        LIST_REVIEW_QUEUE,
        {"limit": 2, "cursor": first.structuredContent["next_cursor"]},
        store,
    )

    assert [record["id"] for record in first.structuredContent["records"]] == [
        "queue-1",
        "queue-2",
    ]
    assert [record["id"] for record in second.structuredContent["records"]] == ["queue-3"]


@pytest.mark.asyncio
async def test_list_review_queue_invalid_cursor_returns_error() -> None:
    result = await call_tool(
        LIST_REVIEW_QUEUE, {"cursor": "not-a-review-queue-cursor"}, RelationStore()
    )

    assert result.isError is True
    assert result.structuredContent["diagnostics"][0]["code"] == "REVIEW_QUEUE_CURSOR_INVALID"


@pytest.mark.asyncio
async def test_commit_reviewed_relations_filters_by_explicit_ids_and_marks_committed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = RelationStore()
    store.enqueue_review_queue([_queue_record(), _queue_record("queue-2")])
    filters: list[Any] = []
    original_list_review_queue = store.list_review_queue

    def spy_list_review_queue(*args: Any, **kwargs: Any) -> Any:
        filters.append(args[0] if args else None)
        return original_list_review_queue(*args, **kwargs)

    monkeypatch.setattr(store, "list_review_queue", spy_list_review_queue)

    result = await call_tool(COMMIT_REVIEWED_RELATIONS, {"ids": ["queue-1"]}, store)

    listed = store.list_review_queue()
    assert result.isError is False
    assert result.structuredContent["status"] == "ok"
    assert result.structuredContent["committed_count"] == 1
    assert len(store.list_relations()) == 1
    assert listed[0].status == ReviewQueueStatus.COMMITTED
    assert listed[0].committed_relation_ids == result.structuredContent["relation_ids"]
    assert filters[0].ids == ["queue-1"]


@pytest.mark.asyncio
async def test_commit_reviewed_relations_reuses_existing_duplicate_relation() -> None:
    store = RelationStore()
    existing, _updated = store.assert_relations(
        [_candidate().to_relation_input()],
        mode="append",
    )
    store.enqueue_review_queue([_queue_record()])

    result = await call_tool(COMMIT_REVIEWED_RELATIONS, {"ids": ["queue-1"]}, store)

    listed = store.list_review_queue()
    assert result.isError is False
    assert result.structuredContent["relation_ids"] == [existing[0].id]
    assert len(store.list_relations()) == 1
    assert listed[0].status == ReviewQueueStatus.COMMITTED
    assert listed[0].committed_relation_ids == [existing[0].id]


@pytest.mark.asyncio
async def test_commit_reviewed_relations_blocks_semantic_duplicate() -> None:
    store = RelationStore()
    existing, _updated = store.assert_relations(
        [
            RelationInput(
                source="integration tests pass",
                target="release auto deploys",
                relation_type="sufficient",
            )
        ],
        mode="append",
    )
    candidate = CandidateRelation(
        id="candidate-1",
        source="integration test passes",
        target="release is auto-deployed",
        relation_type="sufficient",
        confidence=0.9,
        evidence=[EvidenceRecord(url="https://example.com/source", span="A enables B.")],
    )
    store.enqueue_review_queue(
        [
            ReviewQueueRecord(
                id="queue-1",
                run_id="run-1",
                candidate=candidate,
                review=_review(candidate.id),
                gate_result=GateResult(candidate_id=candidate.id, action=GateAction.QUEUE),
            )
        ]
    )

    result = await call_tool(COMMIT_REVIEWED_RELATIONS, {"ids": ["queue-1"]}, store)

    gate_result = result.structuredContent["validation"]["gate_results"][0]
    assert result.isError is False
    assert result.structuredContent["status"] == "warning"
    assert result.structuredContent["committed_count"] == 0
    assert gate_result["metadata"]["semantic_duplicate"]["existing_relation_ids"] == [
        existing[0].id
    ]
    assert len(store.list_relations()) == 1
    assert store.list_review_queue()[0].status == ReviewQueueStatus.PENDING


@pytest.mark.asyncio
async def test_commit_reviewed_relations_revalidates_and_leaves_pending_when_blocked() -> None:
    store = RelationStore()
    store.enqueue_review_queue([_queue_record(with_review=False)])

    result = await call_tool(COMMIT_REVIEWED_RELATIONS, {"ids": ["queue-1"]}, store)

    assert result.isError is False
    assert result.structuredContent["status"] == "warning"
    assert result.structuredContent["committed_count"] == 0
    assert "graph state changed" in result.structuredContent["trace"][0]
    assert store.list_review_queue()[0].status == ReviewQueueStatus.PENDING
    assert store.list_relations() == []


@pytest.mark.asyncio
async def test_commit_keeps_legacy_review_without_direction_check_pending() -> None:
    store = RelationStore()
    legacy_record = _queue_record().model_copy(
        update={"review": _review().model_copy(update={"normalized_implication_supported": None})}
    )
    store.enqueue_review_queue([legacy_record])

    result = await call_tool(COMMIT_REVIEWED_RELATIONS, {"ids": ["queue-1"]}, store)

    assert result.isError is False
    assert result.structuredContent["status"] == "warning"
    assert result.structuredContent["committed_count"] == 0
    assert store.list_review_queue()[0].status == ReviewQueueStatus.PENDING
    assert store.list_relations() == []


@pytest.mark.asyncio
async def test_commit_reviewed_relations_rejects_missing_and_non_pending_records() -> None:
    store = RelationStore()
    store.enqueue_review_queue([_queue_record()])
    store.resolve_review_queue(["queue-1"], reason="already handled")

    result = await call_tool(
        COMMIT_REVIEWED_RELATIONS,
        {"ids": ["queue-1", "missing"]},
        store,
    )

    assert result.isError is True
    codes = [diagnostic["code"] for diagnostic in result.structuredContent["diagnostics"]]
    assert codes == ["REVIEW_QUEUE_RECORD_NOT_PENDING", "REVIEW_QUEUE_RECORD_NOT_FOUND"]
    assert store.list_relations() == []


@pytest.mark.asyncio
async def test_commit_reviewed_relations_reports_partial_write_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import nesy_reasoning_mcp.auto_ingest.writer as writer

    async def fake_write_approved_relations(
        **_kwargs: Any,
    ) -> tuple[list[str], list[Any], dict[str, Any]]:
        return [], [], {"status": "ok", "relation_ids": []}

    store = RelationStore()
    store.enqueue_review_queue([_queue_record()])
    monkeypatch.setattr(writer, "write_approved_relations", fake_write_approved_relations)

    result = await call_tool(COMMIT_REVIEWED_RELATIONS, {"ids": ["queue-1"]}, store)

    assert result.isError is True
    assert result.structuredContent["diagnostics"][0]["code"] == "REVIEW_QUEUE_WRITE_FAILED"
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
