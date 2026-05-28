"""Conversation transcript extraction helpers for Auto-Ingest v2."""

from __future__ import annotations

import json
import os
import re
from collections import deque
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeVar

import anyio
from pydantic import BaseModel, ValidationError

from nesy_reasoning_mcp.auto_ingest.providers import ProviderStructuredOutputMode
from nesy_reasoning_mcp.auto_ingest.schemas import (
    CandidateRelationBatch,
    EvidenceRecord,
    IngestionInput,
)
from nesy_reasoning_mcp.schemas import Diagnostic

if TYPE_CHECKING:
    from nesy_reasoning_mcp.auto_ingest.openai_agents import OpenAICompatibleProviderConfig

ChatCompletionRunner = Callable[..., Awaitable[Any]]
OutputBatch = TypeVar("OutputBatch", bound=BaseModel)

DEFAULT_CONTEXT_TURNS = 6
DEFAULT_CONTEXT_TOKEN_BUDGET = 4000
DEFAULT_TOOL_OUTPUT_LIMIT = 500
DEFAULT_TRANSCRIPT_TAIL_BYTES = 256_000
_CHARS_PER_TOKEN = 4

EXTRACTOR_INSTRUCTIONS = """\
You extract candidate symbolic relations for NeSy Reasoning MCP.
Only emit sufficient, necessary, or equivalent relations directly supported by evidence.
Relation direction rules are strict: sufficient(A, B)=A -> B; necessary(A, B)=B -> A.
Equivalent(A, B)=A -> B and B -> A.
Each candidate must cite at least one provided EvidenceRecord.
Do not turn "may improve", correlation, topical similarity, or vague support into a relation.
"""

RELATION_DIRECTION_RULES = (
    "Relation direction rules: sufficient(A, B)=A -> B; "
    "necessary(A, B)=B -> A; equivalent(A, B)=A -> B and B -> A."
)

_CODE_BLOCK_RE = re.compile(r"```[^\n]*\n(?P<body>.*?)```", re.DOTALL)
_SIGNATURE_RE = re.compile(
    r"^\s*(def |class |async def |function |const \w+\s*=|let \w+\s*=|var \w+\s*=)"
)
_COMMENT_RE = re.compile(r"^\s*(#|//|/\*|\*|--)")


class ExtractionError(ValueError):
    """Raised when the extraction foundation cannot produce valid structured output."""


@dataclass(frozen=True)
class ExtractionModelConfig:
    """Resolved model and runtime controls for one conversation-turn extraction call."""

    model: str | None = None
    provider_config: OpenAICompatibleProviderConfig | None = None
    max_tokens: int = 4096
    timeout_seconds: float = 180

    def __post_init__(self) -> None:
        if self.max_tokens <= 0:
            raise ValueError("max_tokens must be greater than 0")
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be greater than 0")

    def resolved_model(self, env: Mapping[str, str] | None = None) -> str | None:
        """Return the effective extractor model from explicit config or environment."""
        runtime_env = os.environ if env is None else env
        return (
            self.model
            or runtime_env.get("NESY_EXTRACTION_MODEL")
            or runtime_env.get("OPENAI_DEFAULT_MODEL")
        )


@dataclass(frozen=True)
class TranscriptTurn:
    """A normalized user, assistant, or tool turn from a transcript JSONL file."""

    role: str
    content: str


@dataclass(frozen=True)
class TranscriptContextWindow:
    """Denoised transcript context prepared for one extraction prompt."""

    text: str
    turns: list[TranscriptTurn]
    diagnostics: list[Diagnostic]
    truncated: bool = False
    compaction_recommended: bool = False


@dataclass(frozen=True)
class ConversationExtractionResult:
    """Candidate batch plus context diagnostics from one conversation extraction call."""

    candidate_batch: CandidateRelationBatch
    diagnostics: list[Diagnostic]
    truncated: bool = False
    compaction_recommended: bool = False


def build_extraction_prompt(ingestion_input: IngestionInput) -> str:
    """Build the shared extractor prompt for dry-run and conversation-turn extraction."""
    return (
        "Extract only evidence-supported logical relations.\n"
        f"{RELATION_DIRECTION_RULES}\n"
        "Return no candidate when the evidence only shows topical similarity, "
        "correlation, weak possibility, or unsupported speculation.\n\n"
        f"Input JSON:\n{_input_json(ingestion_input)}"
    )


def build_transcript_context_window(
    transcript_path: str | Path,
    *,
    context_turns: int | None = None,
    env: Mapping[str, str] | None = None,
    token_budget: int = DEFAULT_CONTEXT_TOKEN_BUDGET,
    tool_output_limit: int = DEFAULT_TOOL_OUTPUT_LIMIT,
) -> TranscriptContextWindow:
    """Read transcript JSONL and return a denoised recent-turn context window."""
    if token_budget <= 0:
        raise ValueError("token_budget must be greater than 0")
    if tool_output_limit <= 0:
        raise ValueError("tool_output_limit must be greater than 0")

    diagnostics: list[Diagnostic] = []
    turn_count = _context_turn_count(context_turns, env)
    selected = _read_transcript_turns(Path(transcript_path), diagnostics, max_turns=turn_count)
    selected = _dedupe_consecutive_tool_results(selected)
    normalized = [
        TranscriptTurn(
            role=turn.role,
            content=_denoise_content(
                turn.content,
                is_tool_output=turn.role == "tool",
                tool_output_limit=tool_output_limit,
            ),
        )
        for turn in selected
    ]
    text = _format_context(normalized)
    limited_text, truncated = _fit_token_budget(text, token_budget)
    return TranscriptContextWindow(
        text=limited_text,
        turns=normalized,
        diagnostics=diagnostics,
        truncated=truncated,
        compaction_recommended=truncated,
    )


async def extract_candidate_relations_from_context(
    context: TranscriptContextWindow | str,
    *,
    config: ExtractionModelConfig | None = None,
    env: Mapping[str, str] | None = None,
    run_chat_completion: ChatCompletionRunner | None = None,
) -> CandidateRelationBatch:
    """Run one Chat Completions extraction call over a prepared context window."""
    result = await extract_candidate_relations_with_context_metadata(
        context,
        config=config,
        env=env,
        run_chat_completion=run_chat_completion,
    )
    return result.candidate_batch


async def extract_candidate_relations_with_context_metadata(
    context: TranscriptContextWindow | str,
    *,
    config: ExtractionModelConfig | None = None,
    env: Mapping[str, str] | None = None,
    run_chat_completion: ChatCompletionRunner | None = None,
) -> ConversationExtractionResult:
    """Run extraction and return candidate output with context truncation metadata."""
    extraction_config = config or ExtractionModelConfig()
    runtime_env = os.environ if env is None else env
    model = extraction_config.resolved_model(runtime_env)
    provider_config = extraction_config.provider_config
    if provider_config is None:
        raise ExtractionError("provider_config is required for conversation extraction")
    if provider_config.structured_output_mode is not ProviderStructuredOutputMode.JSON_OBJECT:
        raise ExtractionError("conversation extraction requires JSON_OBJECT structured output mode")
    if not model:
        raise ExtractionError(
            "ExtractionModelConfig.model, NESY_EXTRACTION_MODEL, "
            "or OPENAI_DEFAULT_MODEL is required"
        )

    if isinstance(context, TranscriptContextWindow):
        context_window = context
        context_text = context.text
    else:
        context_window = None
        context_text = context
    prompt = _context_extraction_prompt(context_text)
    try:
        with anyio.fail_after(extraction_config.timeout_seconds):
            candidate_batch = await _run_json_object_completion(
                run_chat_completion,
                model=model,
                provider_config=provider_config,
                env=runtime_env,
                instructions=EXTRACTOR_INSTRUCTIONS,
                prompt=prompt,
                output_type=CandidateRelationBatch,
                label="conversation extractor",
                max_tokens=extraction_config.max_tokens,
            )
            return ConversationExtractionResult(
                candidate_batch=candidate_batch,
                diagnostics=list(context_window.diagnostics) if context_window else [],
                truncated=context_window.truncated if context_window else False,
                compaction_recommended=(
                    context_window.compaction_recommended if context_window else False
                ),
            )
    except TimeoutError as exc:
        raise ExtractionError("conversation extractor timed out") from exc


def _read_transcript_turns(
    path: Path,
    diagnostics: list[Diagnostic],
    *,
    max_turns: int,
) -> list[TranscriptTurn]:
    turns: deque[TranscriptTurn] = deque(maxlen=max_turns)
    try:
        lines, truncated, no_boundary = _read_transcript_tail_lines(
            path, DEFAULT_TRANSCRIPT_TAIL_BYTES
        )
    except OSError as exc:
        diagnostics.append(
            Diagnostic(
                level="error",
                code="TRANSCRIPT_READ_FAILED",
                message=f"Could not read transcript: {exc.__class__.__name__}",
            )
        )
        return []
    if truncated:
        diagnostics.append(
            Diagnostic(
                level="info",
                code="TRANSCRIPT_TAIL_TRUNCATED",
                message="Read bounded transcript tail for extraction context",
            )
        )
    if no_boundary:
        diagnostics.append(
            Diagnostic(
                level="warning",
                code="TRANSCRIPT_TAIL_NO_BOUNDARY",
                message="Skipped truncated transcript tail without a JSONL boundary",
            )
        )

    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            diagnostics.append(
                Diagnostic(
                    level="warning",
                    code="TRANSCRIPT_BAD_JSONL",
                    message=f"Skipped invalid JSONL line {line_number}",
                )
            )
            continue
        turn = _turn_from_payload(payload)
        if turn is None:
            diagnostics.append(
                Diagnostic(
                    level="warning",
                    code="TRANSCRIPT_UNKNOWN_TURN",
                    message=f"Skipped unsupported transcript line {line_number}",
                )
            )
            continue
        turns.append(turn)
    return list(turns)


def _read_transcript_tail_lines(path: Path, byte_limit: int) -> tuple[list[str], bool, bool]:
    with path.open("rb") as stream:
        stream.seek(0, os.SEEK_END)
        size = stream.tell()
        truncated = size > byte_limit
        no_boundary = False
        if truncated:
            stream.seek(-byte_limit, os.SEEK_END)
            data = stream.read(byte_limit)
            newline_index = data.find(b"\n")
            if newline_index >= 0:
                data = data[newline_index + 1 :]
            else:
                data = b""
                no_boundary = True
        else:
            stream.seek(0)
            data = stream.read()
    return data.decode("utf-8", errors="replace").splitlines(), truncated, no_boundary


def _turn_from_payload(payload: Any) -> TranscriptTurn | None:
    if not isinstance(payload, Mapping):
        return None
    role = payload.get("role")
    content = payload.get("content")
    message = payload.get("message")
    if (not isinstance(role, str) or content is None) and isinstance(message, Mapping):
        role = message.get("role")
        content = message.get("content")
    if _contains_tool_result(content):
        role = "tool"
    if not isinstance(role, str) or role not in {"user", "assistant", "tool"}:
        return None
    text = _content_to_text(content)
    if not text.strip():
        return None
    return TranscriptTurn(role=role, content=text.strip())


def _content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, Mapping):
        text = content.get("text")
        if text is None:
            text = content.get("content")
        return _content_to_text(text)
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            text = _content_to_text(item).strip()
            if text:
                parts.append(text)
        return "\n".join(parts)
    return ""


def _contains_tool_result(content: Any) -> bool:
    if isinstance(content, Mapping):
        if content.get("type") == "tool_result":
            return True
        return _contains_tool_result(content.get("content"))
    if isinstance(content, list):
        return any(_contains_tool_result(item) for item in content)
    return False


def _context_turn_count(context_turns: int | None, env: Mapping[str, str] | None) -> int:
    if context_turns is not None:
        if context_turns <= 0:
            raise ValueError("context_turns must be greater than 0")
        return context_turns
    runtime_env = os.environ if env is None else env
    raw = runtime_env.get("NESY_INGEST_CONTEXT_TURNS")
    if raw is None:
        return DEFAULT_CONTEXT_TURNS
    try:
        value = int(raw)
    except ValueError:
        return DEFAULT_CONTEXT_TURNS
    return value if value > 0 else DEFAULT_CONTEXT_TURNS


def _dedupe_consecutive_tool_results(turns: list[TranscriptTurn]) -> list[TranscriptTurn]:
    deduped: list[TranscriptTurn] = []
    previous_tool_content: str | None = None
    for turn in turns:
        if turn.role == "tool":
            if turn.content == previous_tool_content:
                continue
            previous_tool_content = turn.content
        else:
            previous_tool_content = None
        deduped.append(turn)
    return deduped


def _denoise_content(content: str, *, is_tool_output: bool, tool_output_limit: int) -> str:
    denoised = _CODE_BLOCK_RE.sub(_denoise_code_block, content)
    if is_tool_output and len(denoised) > tool_output_limit:
        return denoised[:tool_output_limit].rstrip() + "\n[tool output truncated]"
    return denoised


def _denoise_code_block(match: re.Match[str]) -> str:
    kept_lines = [
        line.rstrip()
        for line in match.group("body").splitlines()
        if _SIGNATURE_RE.match(line) or _COMMENT_RE.match(line)
    ]
    if kept_lines:
        return "[code block omitted; preserved lines]\n" + "\n".join(kept_lines)
    return "[code block omitted]"


def _format_context(turns: list[TranscriptTurn]) -> str:
    return "\n\n".join(f"{turn.role.upper()}:\n{turn.content}" for turn in turns)


def _fit_token_budget(text: str, token_budget: int) -> tuple[str, bool]:
    max_chars = token_budget * _CHARS_PER_TOKEN
    if len(text) <= max_chars:
        return text, False
    marker = "\n[context truncated to fit extraction budget]"
    if max_chars <= len(marker):
        return marker[-max_chars:], True
    return text[-(max_chars - len(marker)) :].lstrip() + marker, True


def _context_extraction_prompt(context: str) -> str:
    ingestion_input = IngestionInput(
        evidence=[
            EvidenceRecord(
                url="conversation://current-transcript",
                span=context,
                source_type="conversation_transcript",
            )
        ],
        task="Extract supported symbolic relations from this conversation context.",
    )
    return build_extraction_prompt(ingestion_input)


async def _run_json_object_completion(
    run_chat_completion: ChatCompletionRunner | None,
    *,
    model: str,
    provider_config: OpenAICompatibleProviderConfig,
    env: Mapping[str, str],
    instructions: str,
    prompt: str,
    output_type: type[OutputBatch],
    label: str,
    max_tokens: int,
) -> OutputBatch:
    api_key_env = provider_config.api_key_env
    api_key = env.get(api_key_env)
    if not api_key:
        raise ExtractionError(f"{api_key_env} is required for OpenAI-compatible provider")
    request_kwargs = _json_object_request_kwargs(
        model=model,
        provider_config=provider_config,
        instructions=instructions,
        prompt=prompt,
        output_type=output_type,
        max_tokens=max_tokens,
    )
    if run_chat_completion is not None:
        response = await run_chat_completion(provider_config=provider_config, **request_kwargs)
    else:
        response = await _run_openai_compatible_json_object_completion(
            api_key=api_key,
            provider_config=provider_config,
            request_kwargs=request_kwargs,
        )
    if isinstance(response, output_type):
        return response
    content = _json_object_response_content(response)
    if not content or not content.strip():
        raise ExtractionError(f"{label} returned empty JSON Object content")
    try:
        return output_type.model_validate_json(content)
    except ValidationError as exc:
        raise ExtractionError(
            f"{label} returned JSON that does not match {output_type.__name__}"
        ) from exc


def _json_object_request_kwargs(
    *,
    model: str,
    provider_config: OpenAICompatibleProviderConfig,
    instructions: str,
    prompt: str,
    output_type: type[BaseModel],
    max_tokens: int,
) -> dict[str, Any]:
    request_kwargs: dict[str, Any] = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": _json_object_system_prompt(instructions, output_type),
            },
            {"role": "user", "content": prompt},
        ],
        "response_format": {"type": "json_object"},
        "stream": False,
        "max_tokens": max_tokens,
    }
    if provider_config.reasoning_effort:
        request_kwargs["reasoning_effort"] = provider_config.reasoning_effort
    if provider_config.extra_body:
        request_kwargs["extra_body"] = dict(provider_config.extra_body)
    return request_kwargs


def _json_object_system_prompt(
    instructions: str,
    output_type: type[BaseModel],
) -> str:
    schema_json = json.dumps(output_type.model_json_schema(), ensure_ascii=False)
    example_json = json.dumps(_json_object_example(output_type), ensure_ascii=False)
    return (
        f"{instructions}\n"
        "Return only one valid JSON object. Do not include markdown fences, prose, "
        "or comments.\n"
        "The JSON object must satisfy this JSON schema:\n"
        f"{schema_json}\n"
        "Example JSON output:\n"
        f"{example_json}"
    )


def _json_object_example(output_type: type[BaseModel]) -> dict[str, object]:
    if output_type is CandidateRelationBatch:
        return {"candidates": []}
    raise ExtractionError(f"Unsupported JSON Object output type: {output_type.__name__}")


async def _run_openai_compatible_json_object_completion(
    *,
    api_key: str,
    provider_config: OpenAICompatibleProviderConfig,
    request_kwargs: Mapping[str, Any],
) -> Any:
    from openai import AsyncOpenAI

    client = AsyncOpenAI(
        api_key=api_key,
        base_url=provider_config.base_url,
        default_headers=provider_config.default_headers or None,
    )
    return await client.chat.completions.create(**dict(request_kwargs))


def _json_object_response_content(response: Any) -> str | None:
    if isinstance(response, str):
        return response
    if isinstance(response, Mapping):
        choices = response.get("choices")
        if choices is None:
            raise ExtractionError("provider JSON Object response did not include choices")
        return _json_object_content_from_choices(choices)
    return _json_object_content_from_choices(getattr(response, "choices", None))


def _json_object_content_from_choices(choices: Any) -> str | None:
    if not choices:
        return None
    choice = choices[0]
    message = (
        choice.get("message") if isinstance(choice, Mapping) else getattr(choice, "message", None)
    )
    if message is None:
        return None
    content = (
        message.get("content")
        if isinstance(message, Mapping)
        else getattr(message, "content", None)
    )
    return content if isinstance(content, str) else None


def _input_json(ingestion_input: IngestionInput) -> str:
    return json.dumps(
        ingestion_input.model_dump(mode="json", exclude_none=True), ensure_ascii=False
    )
