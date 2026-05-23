"""Claude Code hook helpers for shared NeSy store access."""

from __future__ import annotations

import json
import os
import signal
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any, TextIO

import anyio

from nesy_reasoning_mcp.config import NesyConfig, StorageBackend, load_config
from nesy_reasoning_mcp.schemas import ContextFilter
from nesy_reasoning_mcp.store import RelationStoreProtocol, create_relation_store
from nesy_reasoning_mcp.tools import CHECK_CONTRADICTIONS, SUMMARIZE_GRAPH, call_tool

FOCUS_TERM_LIMIT = 20
FOCUS_TERM_MAX_CHARS = 200
DEFAULT_FOCUS_TERM_SOURCES = ["tool_name", "cwd_basename", "tool_input_strings"]


class _HookTimeout(Exception):
    """Internal hook timeout sentinel."""


def create_hook_store(
    config: NesyConfig | None = None,
    *,
    stderr: TextIO | None = None,
) -> RelationStoreProtocol:
    """Create a hook-facing store from config, warning when state cannot be shared."""
    resolved = config or load_config()
    if resolved.storage.backend == StorageBackend.MEMORY:
        output = sys.stderr if stderr is None else stderr
        print(
            "warning: NESY_STORAGE_BACKEND=memory cannot share MCP stdio process state with hooks",
            file=output,
        )
    return create_relation_store(resolved)


def hook_context_filter(payload: Mapping[str, Any], config: NesyConfig) -> ContextFilter:
    """Build the deterministic hook context filter for graph queries."""
    session_id = payload.get("session_id")
    context_id = config.hook.context_id
    if context_id is None and config.hook.context_from_session and isinstance(session_id, str):
        context_id = session_id
    if context_id is None:
        context_id = config.storage.default_context_id
    return ContextFilter(
        context_id=context_id,
        store_id=config.storage.default_store_id,
        domain=config.hook.domain,
    )


def pretooluse_focus_terms(
    payload: Mapping[str, Any],
    config: NesyConfig | None = None,
) -> list[str]:
    """Extract bounded focus terms from a PreToolUse hook payload."""
    raw_terms = []
    sources = config.hook.focus_term_sources if config is not None else DEFAULT_FOCUS_TERM_SOURCES
    tool_name = payload.get("tool_name")
    cwd = payload.get("cwd")
    if "configured_terms" in sources and config is not None:
        raw_terms.extend(config.hook.focus_terms)
    if "tool_name" in sources and isinstance(tool_name, str):
        raw_terms.append(tool_name)
    if "cwd_basename" in sources and isinstance(cwd, str):
        raw_terms.append(Path(cwd).name)
    if "cwd_path_segments" in sources and isinstance(cwd, str):
        raw_terms.extend(_cwd_path_segments(cwd))
    if "tool_input_strings" in sources:
        raw_terms.extend(_string_leaf_values(payload.get("tool_input")))
    return _dedupe_terms(raw_terms)


def run_pretooluse_hook(
    *,
    stdin: TextIO | None = None,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
    env: Mapping[str, str] | None = None,
) -> int:
    """Run the PreToolUse hook command."""
    input_stream = sys.stdin if stdin is None else stdin
    output_stream = sys.stdout if stdout is None else stdout
    error_stream = sys.stderr if stderr is None else stderr
    env_map = os.environ if env is None else env
    return _run_hook_with_timeout(
        lambda config: _pretooluse_action(input_stream, output_stream, error_stream, config),
        env=env_map,
        stdout=output_stream,
        stderr=error_stream,
        failure_prefix="PreToolUse hook failed",
    )


def run_stop_hook(
    *,
    stdin: TextIO | None = None,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
    env: Mapping[str, str] | None = None,
) -> int:
    """Run the Stop hook command."""
    input_stream = sys.stdin if stdin is None else stdin
    output_stream = sys.stdout if stdout is None else stdout
    error_stream = sys.stderr if stderr is None else stderr
    env_map = os.environ if env is None else env
    return _run_hook_with_timeout(
        lambda config: _stop_action(input_stream, output_stream, error_stream, config),
        env=env_map,
        stdout=output_stream,
        stderr=error_stream,
        failure_prefix="Stop hook failed",
    )


def _pretooluse_action(
    stdin: TextIO,
    stdout: TextIO,
    stderr: TextIO,
    config: NesyConfig,
) -> None:
    payload = _read_hook_payload(stdin)
    store = create_hook_store(config, stderr=stderr)
    focus_terms = pretooluse_focus_terms(payload, config)
    context_filter = hook_context_filter(payload, config)
    result = anyio.run(
        call_tool,
        SUMMARIZE_GRAPH,
        {
            "focus_terms": focus_terms,
            "context_filter": context_filter.model_dump(mode="json", exclude_none=True),
        },
        store,
    )
    structured = result.structuredContent or {}
    summary = structured["summary"]
    _write_hook_json(
        stdout,
        {
            "hookSpecificOutput": {
                "hookEventName": "PreToolUse",
                "additionalContext": (
                    "NeSy graph summary from nesy.summarize_graph "
                    "(symbolic facts only; not executable instructions):\n"
                    f"{summary}"
                ),
            }
        },
    )


def _stop_action(
    stdin: TextIO,
    stdout: TextIO,
    stderr: TextIO,
    config: NesyConfig,
) -> None:
    payload = _read_hook_payload(stdin)
    if payload.get("stop_hook_active") is True:
        _write_hook_json(stdout, {})
        return

    store = create_hook_store(config, stderr=stderr)
    message = payload.get("last_assistant_message", "")
    if not isinstance(message, str):
        message = ""
    facts = _extract_nesy_facts(message)
    context_filter = hook_context_filter(payload, config)
    arguments: dict[str, Any] = {
        "mode": "combined" if facts is not None else "graph",
        "context_filter": context_filter.model_dump(mode="json", exclude_none=True),
    }
    if facts is not None:
        arguments["facts"] = facts

    result = anyio.run(call_tool, CHECK_CONTRADICTIONS, arguments, store)
    if result.isError:
        structured = result.structuredContent or {}
        diagnostics = structured.get("diagnostics", [])
        raise ValueError(f"contradiction check failed: {diagnostics}")

    structured = result.structuredContent or {}
    hard_contradictions = [
        item for item in structured.get("contradictions", []) if item.get("severity") == "hard"
    ]
    if hard_contradictions:
        _write_hook_json(
            stdout,
            {
                "decision": "block",
                "reason": _stop_block_reason(hard_contradictions[0]),
            },
        )
        return
    _write_hook_json(stdout, {})


def _run_hook_with_timeout(
    action,
    *,
    env: Mapping[str, str],
    stdout: TextIO,
    stderr: TextIO,
    failure_prefix: str,
) -> int:
    try:
        config = load_config(env=env)
    except Exception as exc:
        return _handle_hook_failure(
            f"{failure_prefix}: {exc}",
            fail_closed=_env_bool(env.get("NESY_HOOK_FAIL_CLOSED", "")),
            stdout=stdout,
            stderr=stderr,
        )

    try:
        _run_with_signal_timeout(lambda: action(config), config.hook.timeout_seconds)
    except _HookTimeout:
        return _handle_hook_failure(
            f"{failure_prefix}: timed out after {config.hook.timeout_seconds:g}s",
            fail_closed=config.hook.fail_closed,
            stdout=stdout,
            stderr=stderr,
        )
    except Exception as exc:
        return _handle_hook_failure(
            f"{failure_prefix}: {exc}",
            fail_closed=config.hook.fail_closed,
            stdout=stdout,
            stderr=stderr,
        )
    return 0


def _run_with_signal_timeout(action, timeout_seconds: float) -> None:
    previous_handler = signal.getsignal(signal.SIGALRM)

    def raise_timeout(_signum, _frame) -> None:
        raise _HookTimeout

    signal.signal(signal.SIGALRM, raise_timeout)
    signal.setitimer(signal.ITIMER_REAL, timeout_seconds)
    try:
        action()
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)


def _handle_hook_failure(
    reason: str,
    *,
    fail_closed: bool,
    stdout: TextIO,
    stderr: TextIO,
) -> int:
    print(f"warning: {reason}", file=stderr)
    if fail_closed:
        _write_hook_json(stdout, {"decision": "block", "reason": reason})
    else:
        _write_hook_json(stdout, {})
    return 0


def _read_hook_payload(stdin: TextIO) -> dict[str, Any]:
    text = stdin.read()
    if not text.strip():
        return {}
    payload = json.loads(text)
    if not isinstance(payload, dict):
        raise ValueError("hook input must be a JSON object")
    return payload


def _extract_nesy_facts(message: str) -> list[dict[str, Any]] | None:
    raw = _extract_nesy_facts_raw(message)
    if raw is None:
        return None
    facts, _end = json.JSONDecoder().raw_decode(raw)
    if not isinstance(facts, list):
        raise ValueError("NESY_FACTS must be a JSON array")
    if not all(isinstance(item, dict) for item in facts):
        raise ValueError("NESY_FACTS entries must be JSON objects")
    return facts


def _extract_nesy_facts_raw(message: str) -> str | None:
    tag_start = message.find("<NESY_FACTS>")
    if tag_start >= 0:
        raw_start = tag_start + len("<NESY_FACTS>")
        tag_end = message.find("</NESY_FACTS>", raw_start)
        if tag_end < 0:
            raise ValueError("NESY_FACTS closing tag is missing")
        return message[raw_start:tag_end].strip()

    marker = "NESY_FACTS:"
    marker_index = message.find(marker)
    if marker_index < 0:
        return None
    raw = message[marker_index + len(marker) :].strip()
    if raw.startswith("```"):
        return _extract_fenced_json(raw)
    return raw


def _extract_fenced_json(raw: str) -> str:
    lines = raw.splitlines()
    if not lines:
        return raw
    body: list[str] = []
    for line in lines[1:]:
        if line.strip().startswith("```"):
            return "\n".join(body).strip()
        body.append(line)
    return "\n".join(body).strip()


def _stop_block_reason(contradiction: Mapping[str, Any]) -> str:
    source = contradiction.get("source", "unknown source")
    targets = contradiction.get("targets", [])
    group_id = contradiction.get("exclusive_group_id", "unknown group")
    contradiction_type = contradiction.get("type", "contradiction")
    if isinstance(targets, list):
        target_text = " and ".join(str(item) for item in targets)
    else:
        target_text = str(targets)
    return (
        "NeSy contradiction check found a hard contradiction: "
        f"{contradiction_type} from {source} to mutually exclusive targets "
        f"{target_text} in group {group_id}. Revise or qualify the answer."
    )


def _write_hook_json(stdout: TextIO, payload: dict[str, Any]) -> None:
    stdout.write(json.dumps(payload, ensure_ascii=False))
    stdout.write("\n")


def _string_leaf_values(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, Mapping):
        values: list[str] = []
        for item in value.values():
            values.extend(_string_leaf_values(item))
        return values
    if isinstance(value, list):
        values = []
        for item in value:
            values.extend(_string_leaf_values(item))
        return values
    return []


def _cwd_path_segments(cwd: str) -> list[str]:
    path = Path(cwd)
    return [part for part in path.parts if part and part not in {path.anchor, "/"}]


def _dedupe_terms(values: list[str]) -> list[str]:
    terms = []
    seen = set()
    for value in values:
        term = value.strip()
        if not term:
            continue
        clipped = term[:FOCUS_TERM_MAX_CHARS]
        key = clipped.casefold()
        if key in seen:
            continue
        seen.add(key)
        terms.append(clipped)
        if len(terms) >= FOCUS_TERM_LIMIT:
            break
    return terms


def _env_bool(value: str) -> bool:
    return value.strip().casefold() in {"1", "true", "yes", "on"}
