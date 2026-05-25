import json
from typing import Any

import pytest

from nesy_reasoning_mcp.store import RelationStore
from nesy_reasoning_mcp.tools import (
    ASSERT_EXCLUSIVE,
    ASSERT_RELATIONS,
    LIST_RELATIONS,
    VALIDATE_CANDIDATE_RELATIONS,
    call_tool,
)


def _candidate(
    *,
    candidate_id: str = "candidate-1",
    source: str = "A",
    target: str = "B",
    confidence: float = 0.9,
) -> dict[str, Any]:
    return {
        "id": candidate_id,
        "source": source,
        "target": target,
        "relation_type": "sufficient",
        "confidence": confidence,
        "evidence": [{"url": "https://example.com/source", "span": "A explicitly enables B."}],
    }


def _review(
    candidate_id: str = "candidate-1",
    *,
    decision: str = "approve",
    confidence: float = 0.9,
    reviewer_model: str | None = None,
    relation_type: str = "sufficient",
) -> dict[str, Any]:
    review: dict[str, Any] = {
        "candidate_id": candidate_id,
        "decision": decision,
        "reasons": ["Evidence directly supports the relation."],
    }
    if reviewer_model is not None:
        review["reviewer_model"] = reviewer_model
    if decision in {"approve", "downgrade"}:
        review["final_relation_type"] = relation_type
        review["final_confidence"] = confidence
        review["normalized_implication_supported"] = True
    return review


@pytest.mark.asyncio
async def test_validate_candidate_relations_approves_without_persisting() -> None:
    store = RelationStore()

    result = await call_tool(
        VALIDATE_CANDIDATE_RELATIONS,
        {
            "candidates": [_candidate()],
            "reviews": [_review()],
        },
        store,
    )
    listed = await call_tool(LIST_RELATIONS, {}, store)

    assert result.isError is False
    assert result.structuredContent["status"] == "ok"
    assert result.structuredContent["persisted"] is False
    assert result.structuredContent["candidate_count"] == 1
    assert result.structuredContent["approved_count"] == 1
    assert result.structuredContent["queued_count"] == 0
    assert result.structuredContent["rejected_count"] == 0
    assert result.structuredContent["gate_results"][0]["action"] == "auto_write"
    assert result.structuredContent["approved_relations"][0]["source"] == "A"
    assert listed.structuredContent["relations"] == []
    assert json.loads(result.content[0].text) == result.structuredContent


@pytest.mark.asyncio
async def test_validate_queues_legacy_approval_without_direction_check() -> None:
    store = RelationStore()
    legacy_review = _review()
    legacy_review.pop("normalized_implication_supported")

    result = await call_tool(
        VALIDATE_CANDIDATE_RELATIONS,
        {
            "candidates": [_candidate()],
            "reviews": [legacy_review],
        },
        store,
    )

    assert result.structuredContent["status"] == "warning"
    assert result.structuredContent["approved_count"] == 0
    assert result.structuredContent["queued_count"] == 1
    assert result.structuredContent["gate_results"][0]["reasons"] == [
        "normalized implication support was not confirmed"
    ]
    assert store.list_relations() == []


@pytest.mark.asyncio
async def test_validate_candidate_relations_rejects_unknown_fields() -> None:
    store = RelationStore()

    result = await call_tool(
        VALIDATE_CANDIDATE_RELATIONS,
        {
            "candidates": [_candidate()],
            "reviews": [_review()],
            "unexpected": True,
        },
        store,
    )

    assert result.isError is True
    assert result.structuredContent["status"] == "error"
    assert result.structuredContent["diagnostics"][0]["code"] == "INPUT_VALIDATION_ERROR"


@pytest.mark.asyncio
async def test_validate_candidate_relations_rejects_invalid_candidate_shape() -> None:
    result = await call_tool(
        VALIDATE_CANDIDATE_RELATIONS,
        {
            "candidates": [
                {
                    "source": "A",
                    "target": "B",
                    "relation_type": "sufficient",
                    "confidence": 1.2,
                    "evidence": [],
                }
            ],
            "reviews": [_review()],
        },
        RelationStore(),
    )

    assert result.isError is True
    assert result.structuredContent["status"] == "error"
    assert result.structuredContent["diagnostics"][0]["code"] == "INPUT_VALIDATION_ERROR"


@pytest.mark.asyncio
async def test_validate_candidate_relations_queues_missing_review() -> None:
    result = await call_tool(
        VALIDATE_CANDIDATE_RELATIONS,
        {"candidates": [_candidate()]},
        RelationStore(),
    )

    assert result.structuredContent["status"] == "warning"
    assert result.structuredContent["approved_count"] == 0
    assert result.structuredContent["queued_count"] == 1
    assert result.structuredContent["gate_results"][0]["reasons"] == ["missing reviewer decision"]


@pytest.mark.parametrize(
    ("decision", "expected_action"),
    [
        ("reject", "reject"),
        ("downgrade", "queue"),
        ("needs_human", "queue"),
    ],
)
@pytest.mark.asyncio
async def test_validate_candidate_relations_handles_non_approved_reviews(
    decision: str,
    expected_action: str,
) -> None:
    result = await call_tool(
        VALIDATE_CANDIDATE_RELATIONS,
        {
            "candidates": [_candidate()],
            "reviews": [_review(decision=decision)],
        },
        RelationStore(),
    )

    assert result.structuredContent["approved_count"] == 0
    assert result.structuredContent["gate_results"][0]["action"] == expected_action


@pytest.mark.asyncio
async def test_validate_candidate_relations_queues_low_confidence() -> None:
    result = await call_tool(
        VALIDATE_CANDIDATE_RELATIONS,
        {
            "candidates": [_candidate(confidence=0.9)],
            "reviews": [_review(confidence=0.7)],
            "min_write_confidence": 0.85,
        },
        RelationStore(),
    )

    assert result.structuredContent["approved_count"] == 0
    assert result.structuredContent["queued_count"] == 1
    assert "below write threshold" in result.structuredContent["gate_results"][0]["reasons"][0]


@pytest.mark.asyncio
async def test_validate_candidate_relations_queues_candidate_set_contradiction() -> None:
    result = await call_tool(
        VALIDATE_CANDIDATE_RELATIONS,
        {
            "candidates": [
                _candidate(candidate_id="candidate-1", source="A", target="B"),
                _candidate(candidate_id="candidate-2", source="A", target="not B"),
            ],
            "reviews": [_review("candidate-1"), _review("candidate-2")],
        },
        RelationStore(),
    )

    assert result.structuredContent["status"] == "warning"
    assert result.structuredContent["approved_count"] == 0
    assert {item["action"] for item in result.structuredContent["gate_results"]} == {"queue"}
    assert (
        result.structuredContent["reasoning"]["candidate_set"]["result"]["has_contradictions"]
        is True
    )


@pytest.mark.asyncio
async def test_validate_candidate_relations_queues_contradiction_with_current_graph() -> None:
    store = RelationStore()
    await call_tool(
        ASSERT_EXCLUSIVE,
        {"groups": [{"group_id": "state", "members": ["B", "C"]}]},
        store,
    )
    await call_tool(
        ASSERT_RELATIONS,
        {
            "relations": [{"source": "A", "target": "B", "relation_type": "sufficient"}],
            "check_contradictions": False,
        },
        store,
    )

    result = await call_tool(
        VALIDATE_CANDIDATE_RELATIONS,
        {
            "candidates": [_candidate(target="C")],
            "reviews": [_review()],
        },
        store,
    )

    assert result.structuredContent["approved_count"] == 0
    assert result.structuredContent["queued_count"] == 1
    assert result.structuredContent["gate_results"][0]["reasons"] == [
        "hard contradiction found against current graph"
    ]
    assert result.structuredContent["reasoning"]["combined"]["has_contradictions"] is True
    assert len(store.list_relations()) == 1


@pytest.mark.asyncio
async def test_validate_candidate_relations_uses_ephemeral_proposition_overlay() -> None:
    store = RelationStore()

    result = await call_tool(
        VALIDATE_CANDIDATE_RELATIONS,
        {
            "propositions": [
                {"id": "profit_up", "label": "Profit up", "aliases": ["利润增加"]},
                {
                    "id": "profit_not_up",
                    "label": "Profit not up",
                    "aliases": ["利润未增加"],
                    "negates": "profit_up",
                },
            ],
            "candidates": [
                _candidate(candidate_id="candidate-1", source="Discount", target="利润增加"),
                _candidate(candidate_id="candidate-2", source="Discount", target="利润未增加"),
            ],
            "reviews": [_review("candidate-1"), _review("candidate-2")],
        },
        store,
    )

    assert result.structuredContent["approved_count"] == 0
    assert {item["action"] for item in result.structuredContent["gate_results"]} == {"queue"}
    assert result.structuredContent["reasoning"]["candidate_set"]["result"]["contradictions"][0][
        "targets"
    ] == ["profit_up", "profit_not_up"]
    assert store.list_propositions() == []


@pytest.mark.asyncio
async def test_validate_candidate_relations_majority_votes_and_reports_aggregation() -> None:
    result = await call_tool(
        VALIDATE_CANDIDATE_RELATIONS,
        {
            "candidates": [_candidate()],
            "reviews": [
                _review(reviewer_model="reviewer-a", confidence=0.91),
                _review(reviewer_model="reviewer-b", confidence=0.87),
                _review(reviewer_model="reviewer-c", decision="needs_human"),
            ],
            "voting_policy": "majority",
        },
        RelationStore(),
    )

    assert result.structuredContent["approved_count"] == 1
    assert result.structuredContent["gate_results"][0]["action"] == "auto_write"
    aggregation = result.structuredContent["review_aggregation"]
    assert aggregation["policy"] == "majority"
    assert aggregation["candidates"][0]["review_count"] == 3
    assert aggregation["candidates"][0]["agreement"] is False


@pytest.mark.asyncio
async def test_validate_candidate_relations_disagreement_queues_candidate() -> None:
    result = await call_tool(
        VALIDATE_CANDIDATE_RELATIONS,
        {
            "candidates": [_candidate()],
            "reviews": [
                _review(reviewer_model="reviewer-a"),
                _review(reviewer_model="reviewer-b", relation_type="necessary"),
            ],
            "voting_policy": "unanimous",
        },
        RelationStore(),
    )

    assert result.structuredContent["approved_count"] == 0
    assert result.structuredContent["queued_count"] == 1
    assert result.structuredContent["gate_results"][0]["action"] == "queue"


@pytest.mark.asyncio
async def test_validate_candidate_relations_high_priority_reject_blocks_candidate() -> None:
    result = await call_tool(
        VALIDATE_CANDIDATE_RELATIONS,
        {
            "candidates": [_candidate()],
            "reviews": [
                _review(reviewer_model="senior", decision="reject"),
                _review(reviewer_model="reviewer-a"),
                _review(reviewer_model="reviewer-b"),
            ],
            "high_priority_reviewer_models": ["senior"],
        },
        RelationStore(),
    )

    assert result.structuredContent["approved_count"] == 0
    assert result.structuredContent["rejected_count"] == 1
    assert result.structuredContent["gate_results"][0]["action"] == "reject"
