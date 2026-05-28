import json

import pytest

from nesy_reasoning_mcp.auto_ingest import ConversationTurnJob, ConversationTurnJobStatus
from nesy_reasoning_mcp.store import RelationStore
from nesy_reasoning_mcp.tools import (
    ASSERT_EXCLUSIVE,
    ASSERT_RELATIONS,
    SUMMARIZE_GRAPH,
    call_tool,
)


@pytest.mark.asyncio
async def test_summarize_graph_filters_sorts_and_mirrors_content() -> None:
    store = RelationStore()
    await call_tool(
        ASSERT_EXCLUSIVE,
        {
            "groups": [
                {
                    "group_id": "profit_state",
                    "members": ["利润增加", "利润减少"],
                    "context_id": "ctx",
                }
            ]
        },
        store,
    )
    await call_tool(
        ASSERT_RELATIONS,
        {
            "relations": [
                {
                    "id": "rel_z",
                    "source": "库存下降",
                    "target": "现金回收",
                    "relation_type": "sufficient",
                    "context_id": "ctx",
                    "metadata": {"domain": "retail"},
                },
                {
                    "id": "rel_a",
                    "source": "降价",
                    "target": "利润减少",
                    "relation_type": "sufficient",
                    "context_id": "ctx",
                    "metadata": {"domain": "retail"},
                },
                {
                    "id": "rel_other",
                    "source": "降价",
                    "target": "利润增加",
                    "relation_type": "sufficient",
                    "context_id": "other",
                    "metadata": {"domain": "retail"},
                },
            ],
            "check_contradictions": False,
        },
        store,
    )

    result = await call_tool(
        SUMMARIZE_GRAPH,
        {
            "focus_terms": ["利润"],
            "context_filter": {"context_id": "ctx", "domain": "retail"},
        },
        store,
    )

    assert result.isError is False
    assert json.loads(result.content[0].text) == result.structuredContent
    assert result.structuredContent["relation_count_included"] == 1
    assert "降价 sufficient 利润减少" in result.structuredContent["summary"]
    assert "profit_state" in result.structuredContent["summary"]
    assert "现金回收" not in result.structuredContent["summary"]
    assert "利润增加" in result.structuredContent["summary"]
    assert "Background processing" not in result.structuredContent["summary"]


@pytest.mark.asyncio
async def test_summarize_graph_truncates_by_relation_and_char_limits() -> None:
    store = RelationStore()
    await call_tool(
        ASSERT_RELATIONS,
        {
            "relations": [
                {
                    "id": "rel_1",
                    "source": "A",
                    "target": "B",
                    "relation_type": "sufficient",
                },
                {
                    "id": "rel_2",
                    "source": "C",
                    "target": "D",
                    "relation_type": "sufficient",
                },
            ],
            "check_contradictions": False,
        },
        store,
    )

    relation_limited = await call_tool(
        SUMMARIZE_GRAPH,
        {"max_relations": 1, "include_exclusives": False},
        store,
    )
    char_limited = await call_tool(
        SUMMARIZE_GRAPH,
        {"max_chars": 500, "include_exclusives": False},
        store,
    )
    invalid = await call_tool(SUMMARIZE_GRAPH, {"max_chars": 499}, store)

    assert relation_limited.structuredContent["truncated"] is True
    assert relation_limited.structuredContent["relation_count_included"] == 1
    assert char_limited.structuredContent["truncated"] is False
    assert invalid.isError is True


@pytest.mark.asyncio
async def test_summarize_graph_appends_background_status_when_queue_in_flight() -> None:
    store = RelationStore()
    await call_tool(
        ASSERT_RELATIONS,
        {
            "relations": [{"source": "A", "target": "B", "relation_type": "sufficient"}],
            "check_contradictions": False,
        },
        store,
    )
    store.enqueue_ingestion_jobs(
        [
            ConversationTurnJob(
                job_id="turn-pending",
                session_id="session-1",
                transcript_path="/tmp/transcript.jsonl",
                status=ConversationTurnJobStatus.PENDING,
            ),
            ConversationTurnJob(
                job_id="turn-extracting",
                session_id="session-1",
                transcript_path="/tmp/transcript.jsonl",
                status=ConversationTurnJobStatus.EXTRACTING,
            ),
            ConversationTurnJob(
                job_id="turn-reviewing",
                session_id="session-1",
                transcript_path="/tmp/transcript.jsonl",
                status=ConversationTurnJobStatus.REVIEWING,
            ),
        ]
    )

    result = await call_tool(SUMMARIZE_GRAPH, {"include_exclusives": False}, store)

    summary = result.structuredContent["summary"]
    assert result.isError is False
    assert "Background processing" in summary
    assert "1 turns pending extraction" in summary
    assert "1 extracting" in summary
    assert "1 under review" in summary
    assert "Last write:" in summary
    assert "relations added" in summary


@pytest.mark.asyncio
async def test_summarize_graph_keeps_background_status_when_truncated() -> None:
    store = RelationStore()
    await call_tool(
        ASSERT_RELATIONS,
        {
            "relations": [
                {
                    "source": f"VeryLongSource{i:02d}" * 6,
                    "target": f"VeryLongTarget{i:02d}" * 6,
                    "relation_type": "sufficient",
                }
                for i in range(20)
            ],
            "check_contradictions": False,
        },
        store,
    )
    store.enqueue_ingestion_jobs(
        [
            ConversationTurnJob(
                job_id="turn-pending",
                session_id="session-1",
                transcript_path="/tmp/transcript.jsonl",
            )
        ]
    )

    result = await call_tool(
        SUMMARIZE_GRAPH,
        {"max_chars": 500, "include_exclusives": False},
        store,
    )

    summary = result.structuredContent["summary"]
    assert result.structuredContent["truncated"] is True
    assert len(summary) <= 500
    assert "...truncated" in summary
    assert "Background processing" in summary
