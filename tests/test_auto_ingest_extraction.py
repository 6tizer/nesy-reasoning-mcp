import json
from pathlib import Path
from typing import Any

import anyio
import pytest
from pydantic import BaseModel

from nesy_reasoning_mcp.auto_ingest import extraction
from nesy_reasoning_mcp.auto_ingest.extraction import (
    DEFAULT_CONTEXT_TOKEN_BUDGET,
    DEFAULT_CONTEXT_TURNS,
    ExtractionError,
    ExtractionModelConfig,
    build_transcript_context_window,
    extract_candidate_relations_from_context,
    extract_candidate_relations_with_context_metadata,
)
from nesy_reasoning_mcp.auto_ingest.openai_agents import OpenAICompatibleProviderConfig
from nesy_reasoning_mcp.auto_ingest.providers import ProviderStructuredOutputMode


def test_extraction_model_config_resolves_model_precedence() -> None:
    env = {
        "NESY_EXTRACTION_MODEL": "extraction-env-model",
        "OPENAI_DEFAULT_MODEL": "default-env-model",
    }

    assert ExtractionModelConfig(model="explicit-model").resolved_model(env) == "explicit-model"
    assert ExtractionModelConfig().resolved_model(env) == "extraction-env-model"
    assert ExtractionModelConfig().resolved_model({"OPENAI_DEFAULT_MODEL": "default"}) == "default"
    assert ExtractionModelConfig().max_tokens == 4096
    assert ExtractionModelConfig().timeout_seconds == 180


def test_extraction_model_config_rejects_invalid_runtime_values() -> None:
    with pytest.raises(ValueError, match="max_tokens"):
        ExtractionModelConfig(max_tokens=0)
    with pytest.raises(ValueError, match="timeout_seconds"):
        ExtractionModelConfig(timeout_seconds=0)


def test_transcript_context_reads_supported_shapes_and_last_default_turns(
    tmp_path: Path,
) -> None:
    transcript = tmp_path / "transcript.jsonl"
    rows = [
        {"role": "system", "content": "ignored"},
        {"role": "user", "content": "old user"},
        {"message": {"role": "assistant", "content": "old assistant"}},
        {"role": "user", "content": "turn 1"},
        {"message": {"role": "assistant", "content": "turn 2"}},
        {"role": "user", "content": "turn 3"},
        {"role": "assistant", "content": "turn 4"},
        {"role": "user", "content": "turn 5"},
        {"role": "assistant", "content": "turn 6"},
    ]
    transcript.write_text(
        "\n".join([json.dumps(row) for row in rows] + ["{bad json"]),
        encoding="utf-8",
    )

    window = build_transcript_context_window(transcript, env={})

    assert len(window.turns) == DEFAULT_CONTEXT_TURNS
    assert "old user" not in window.text
    assert "turn 1" in window.text
    assert "turn 6" in window.text
    assert [diagnostic.code for diagnostic in window.diagnostics] == [
        "TRANSCRIPT_UNKNOWN_TURN",
        "TRANSCRIPT_BAD_JSONL",
    ]


def test_transcript_context_uses_env_context_turns(tmp_path: Path) -> None:
    transcript = tmp_path / "transcript.jsonl"
    rows = [
        {"role": "assistant", "content": "turn 1"},
        {"role": "user", "content": "turn 2"},
        {"role": "assistant", "content": "turn 3"},
    ]
    transcript.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")

    window = build_transcript_context_window(
        transcript,
        env={"NESY_INGEST_CONTEXT_TURNS": "2"},
    )

    assert [turn.content for turn in window.turns] == ["turn 2", "turn 3"]


def test_transcript_context_denoises_code_truncates_tools_and_dedupes_tools(
    tmp_path: Path,
) -> None:
    transcript = tmp_path / "transcript.jsonl"
    tool_output = "x" * 700
    rows = [
        {
            "role": "assistant",
            "content": "Here:\n```python\n# keep comment\ndef run():\n    return 1\n```",
        },
        {"role": "tool", "content": tool_output},
        {"role": "tool", "content": tool_output},
    ]
    transcript.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")

    window = build_transcript_context_window(transcript, context_turns=3)

    assert len(window.turns) == 2
    assert "return 1" not in window.text
    assert "# keep comment" in window.text
    assert "def run():" in window.text
    assert "[tool output truncated]" in window.text
    assert len(window.text) < len(tool_output) + 200


def test_transcript_context_treats_claude_tool_result_content_as_tool(
    tmp_path: Path,
) -> None:
    transcript = tmp_path / "transcript.jsonl"
    shared_prefix = "x" * 520
    rows = [
        {
            "message": {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "content": [{"type": "text", "text": shared_prefix + "first"}],
                    }
                ],
            }
        },
        {
            "message": {
                "role": "user",
                "content": [{"type": "tool_result", "content": shared_prefix + "first"}],
            }
        },
        {
            "message": {
                "role": "user",
                "content": [{"type": "tool_result", "content": shared_prefix + "second"}],
            }
        },
    ]
    transcript.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")

    window = build_transcript_context_window(transcript, context_turns=3, tool_output_limit=500)

    assert len(window.turns) == 2
    assert window.turns[0].role == "tool"
    assert window.turns[1].role == "tool"
    assert "[tool output truncated]" in window.text
    assert window.text.count("[tool output truncated]") == 2
    assert "second" not in window.text


def test_transcript_context_reads_bounded_tail(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transcript = tmp_path / "transcript.jsonl"
    rows = [
        {"role": "assistant", "content": "old" * 100},
        {"role": "assistant", "content": "recent"},
    ]
    transcript.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    monkeypatch.setattr(extraction, "DEFAULT_TRANSCRIPT_TAIL_BYTES", 120)

    window = build_transcript_context_window(transcript, context_turns=1)

    assert window.turns[0].content == "recent"
    assert "old" not in window.text
    assert window.diagnostics[0].code == "TRANSCRIPT_TAIL_TRUNCATED"


def test_transcript_context_reports_truncated_tail_without_jsonl_boundary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transcript = tmp_path / "transcript.jsonl"
    transcript.write_text("x" * 500, encoding="utf-8")
    monkeypatch.setattr(extraction, "DEFAULT_TRANSCRIPT_TAIL_BYTES", 100)

    window = build_transcript_context_window(transcript)

    assert window.turns == []
    assert [diagnostic.code for diagnostic in window.diagnostics] == [
        "TRANSCRIPT_TAIL_TRUNCATED",
        "TRANSCRIPT_TAIL_NO_BOUNDARY",
    ]


def test_transcript_context_missing_file_reports_diagnostic(tmp_path: Path) -> None:
    window = build_transcript_context_window(tmp_path / "missing.jsonl")

    assert window.turns == []
    assert window.diagnostics[0].code == "TRANSCRIPT_READ_FAILED"


def test_transcript_context_marks_budget_truncation(tmp_path: Path) -> None:
    transcript = tmp_path / "transcript.jsonl"
    transcript.write_text(
        json.dumps({"role": "assistant", "content": "a" * (DEFAULT_CONTEXT_TOKEN_BUDGET * 5)}),
        encoding="utf-8",
    )

    window = build_transcript_context_window(transcript)

    assert window.truncated is True
    assert window.compaction_recommended is True
    assert len(window.text) <= DEFAULT_CONTEXT_TOKEN_BUDGET * 4


async def test_extract_candidate_relations_from_context_uses_mock_chat_completion() -> None:
    captured: dict[str, Any] = {}

    async def fake_chat_completion(**kwargs: Any) -> str:
        captured.update(kwargs)
        return json.dumps(
            {
                "candidates": [
                    {
                        "id": "candidate-1",
                        "source": "A",
                        "target": "B",
                        "relation_type": "sufficient",
                        "confidence": 0.9,
                        "evidence": [
                            {
                                "url": "conversation://current-transcript",
                                "span": "A implies B because the user said so.",
                            }
                        ],
                    }
                ]
            }
        )

    batch = await extract_candidate_relations_from_context(
        "A implies B because the user said so.",
        config=ExtractionModelConfig(
            provider_config=OpenAICompatibleProviderConfig(
                base_url="https://api.example.com",
                api_key_env="EXAMPLE_API_KEY",
                structured_output_mode=ProviderStructuredOutputMode.JSON_OBJECT,
            )
        ),
        env={"EXAMPLE_API_KEY": "secret", "NESY_EXTRACTION_MODEL": "extract-model"},
        run_chat_completion=fake_chat_completion,
    )

    assert batch.candidates[0].source == "A"
    assert captured["model"] == "extract-model"
    assert captured["max_tokens"] == 4096
    assert captured["response_format"] == {"type": "json_object"}
    assert "conversation://current-transcript" in captured["messages"][1]["content"]


async def test_extract_candidate_relations_with_context_metadata_returns_flags(
    tmp_path: Path,
) -> None:
    transcript = tmp_path / "transcript.jsonl"
    transcript.write_text(
        json.dumps({"role": "assistant", "content": "A implies B."}),
        encoding="utf-8",
    )
    context = build_transcript_context_window(transcript, token_budget=1)

    async def fake_chat_completion(**kwargs: Any) -> str:
        return json.dumps({"candidates": []})

    result = await extract_candidate_relations_with_context_metadata(
        context,
        config=ExtractionModelConfig(
            provider_config=OpenAICompatibleProviderConfig(
                base_url="https://api.example.com",
                api_key_env="EXAMPLE_API_KEY",
                structured_output_mode=ProviderStructuredOutputMode.JSON_OBJECT,
            )
        ),
        env={"EXAMPLE_API_KEY": "secret", "OPENAI_DEFAULT_MODEL": "extract-model"},
        run_chat_completion=fake_chat_completion,
    )

    assert result.candidate_batch.candidates == []
    assert result.truncated is True
    assert result.compaction_recommended is True


async def test_extract_candidate_relations_rejects_malformed_output() -> None:
    async def fake_chat_completion(**kwargs: Any) -> str:
        return json.dumps({"candidates": [{"source": "missing required fields"}]})

    with pytest.raises(ExtractionError, match="does not match CandidateRelationBatch"):
        await extract_candidate_relations_from_context(
            "A implies B.",
            config=ExtractionModelConfig(
                provider_config=OpenAICompatibleProviderConfig(
                    base_url="https://api.example.com",
                    api_key_env="EXAMPLE_API_KEY",
                    structured_output_mode=ProviderStructuredOutputMode.JSON_OBJECT,
                )
            ),
            env={"EXAMPLE_API_KEY": "secret", "OPENAI_DEFAULT_MODEL": "extract-model"},
            run_chat_completion=fake_chat_completion,
        )


async def test_extract_candidate_relations_requires_api_key() -> None:
    async def fake_chat_completion(**kwargs: Any) -> str:
        raise AssertionError("missing API key must be rejected before LLM call")

    with pytest.raises(ExtractionError, match="EXAMPLE_API_KEY"):
        await extract_candidate_relations_from_context(
            "A implies B.",
            config=ExtractionModelConfig(
                provider_config=OpenAICompatibleProviderConfig(
                    base_url="https://api.example.com",
                    api_key_env="EXAMPLE_API_KEY",
                    structured_output_mode=ProviderStructuredOutputMode.JSON_OBJECT,
                )
            ),
            env={"OPENAI_DEFAULT_MODEL": "extract-model"},
            run_chat_completion=fake_chat_completion,
        )


async def test_extract_candidate_relations_requires_json_object_provider() -> None:
    async def fake_chat_completion(**kwargs: Any) -> str:
        raise AssertionError("non-JSON_OBJECT provider must be rejected before LLM call")

    with pytest.raises(ExtractionError, match="JSON_OBJECT"):
        await extract_candidate_relations_from_context(
            "A implies B.",
            config=ExtractionModelConfig(
                provider_config=OpenAICompatibleProviderConfig(
                    base_url="https://api.example.com",
                    api_key_env="EXAMPLE_API_KEY",
                )
            ),
            env={"EXAMPLE_API_KEY": "secret", "OPENAI_DEFAULT_MODEL": "extract-model"},
            run_chat_completion=fake_chat_completion,
        )


def test_json_object_example_rejects_unknown_output_type() -> None:
    class UnknownOutput(BaseModel):
        value: str

    with pytest.raises(ExtractionError, match="Unsupported JSON Object output type"):
        extraction._json_object_example(UnknownOutput)


async def test_extract_candidate_relations_applies_timeout() -> None:
    async def fake_chat_completion(**kwargs: Any) -> str:
        await anyio.sleep(1)
        return json.dumps({"candidates": []})

    with pytest.raises(ExtractionError, match="timed out"):
        await extract_candidate_relations_from_context(
            "A implies B.",
            config=ExtractionModelConfig(
                provider_config=OpenAICompatibleProviderConfig(
                    base_url="https://api.example.com",
                    api_key_env="EXAMPLE_API_KEY",
                    structured_output_mode=ProviderStructuredOutputMode.JSON_OBJECT,
                ),
                timeout_seconds=0.01,
            ),
            env={"EXAMPLE_API_KEY": "secret", "OPENAI_DEFAULT_MODEL": "extract-model"},
            run_chat_completion=fake_chat_completion,
        )
