import json
from io import StringIO
from pathlib import Path
from time import sleep

import pytest

from nesy_reasoning_mcp import hooks
from nesy_reasoning_mcp.config import NesyConfig, StorageConfig
from nesy_reasoning_mcp.hooks import (
    create_hook_store,
    hook_context_filter,
    pretooluse_focus_terms,
    run_pretooluse_hook,
    run_stop_hook,
)
from nesy_reasoning_mcp.schemas import ExclusiveGroupInput, RelationInput, RelationType


def test_hook_context_filter_defaults_and_env_choices() -> None:
    config = NesyConfig(storage=StorageConfig(default_context_id="default_ctx"))

    default_filter = hook_context_filter({"session_id": "session_ctx"}, config)
    session_filter = hook_context_filter(
        {"session_id": "session_ctx"},
        config.model_copy(
            update={"hook": config.hook.model_copy(update={"context_from_session": True})}
        ),
    )
    explicit_filter = hook_context_filter(
        {"session_id": "session_ctx"},
        config.model_copy(
            update={"hook": config.hook.model_copy(update={"context_id": "explicit"})}
        ),
    )

    assert default_filter.context_id == "default_ctx"
    assert session_filter.context_id == "session_ctx"
    assert explicit_filter.context_id == "explicit"
    assert explicit_filter.store_id == "default"


def test_hook_store_warns_for_memory_backend() -> None:
    stderr = StringIO()

    create_hook_store(NesyConfig(), stderr=stderr)

    assert "cannot share MCP stdio process state" in stderr.getvalue()


def test_hook_store_uses_sqlite_shared_state(tmp_path: Path) -> None:
    config = NesyConfig(
        storage=StorageConfig(backend="sqlite", sqlite_path=str(tmp_path / "nesy.db"))
    )
    writer = create_hook_store(config, stderr=StringIO())
    writer.assert_relations(
        [RelationInput(source="A", target="B", relation_type=RelationType.SUFFICIENT)]
    )

    reader = create_hook_store(config, stderr=StringIO())

    assert reader.list_relations()[0].source == "A"


def test_pretooluse_focus_terms_are_bounded_and_deduped() -> None:
    terms = pretooluse_focus_terms(
        {
            "tool_name": "Bash",
            "cwd": "/tmp/project",
            "tool_input": {"command": "pytest", "nested": ["pytest", "降价"]},
        }
    )

    assert terms == ["Bash", "project", "pytest", "降价"]


def test_run_pretooluse_hook_injects_summary_from_sqlite(tmp_path: Path) -> None:
    config_path = tmp_path / "nesy.json"
    sqlite_path = tmp_path / "nesy.db"
    config_path.write_text(
        json.dumps({"storage": {"backend": "sqlite", "sqlite_path": str(sqlite_path)}}),
        encoding="utf-8",
    )
    writer = create_hook_store(
        NesyConfig(storage=StorageConfig(backend="sqlite", sqlite_path=str(sqlite_path))),
        stderr=StringIO(),
    )
    writer.assert_relations(
        [RelationInput(source="降价", target="销量增加", relation_type=RelationType.SUFFICIENT)]
    )
    stdin = StringIO(
        json.dumps(
            {
                "hook_event_name": "PreToolUse",
                "tool_name": "Bash",
                "tool_input": {"command": "降价"},
                "cwd": str(tmp_path),
            }
        )
    )
    stdout = StringIO()

    code = run_pretooluse_hook(
        stdin=stdin,
        stdout=stdout,
        stderr=StringIO(),
        env={"NESY_CONFIG": str(config_path)},
    )

    payload = json.loads(stdout.getvalue())
    assert code == 0
    assert payload["hookSpecificOutput"]["hookEventName"] == "PreToolUse"
    assert "additionalContext" in payload["hookSpecificOutput"]
    assert "not executable instructions" in payload["hookSpecificOutput"]["additionalContext"]
    assert "降价 sufficient 销量增加" in payload["hookSpecificOutput"]["additionalContext"]
    assert "permissionDecision" not in payload["hookSpecificOutput"]


def test_run_pretooluse_hook_timeout_fails_open(monkeypatch: pytest.MonkeyPatch) -> None:
    def slow_action(*_args) -> None:
        sleep(0.05)

    monkeypatch.setattr(hooks, "_pretooluse_action", slow_action)
    stdout = StringIO()
    stderr = StringIO()

    code = run_pretooluse_hook(
        stdin=StringIO("{}"),
        stdout=stdout,
        stderr=stderr,
        env={"NESY_HOOK_TIMEOUT_SECONDS": "0.01"},
    )

    assert code == 0
    assert json.loads(stdout.getvalue()) == {}
    assert "timed out" in stderr.getvalue()


def test_run_pretooluse_hook_failure_can_fail_closed() -> None:
    stdout = StringIO()
    stderr = StringIO()

    code = run_pretooluse_hook(
        stdin=StringIO("{bad"),
        stdout=stdout,
        stderr=stderr,
        env={"NESY_HOOK_FAIL_CLOSED": "true"},
    )

    payload = json.loads(stdout.getvalue())
    assert code == 0
    assert payload["decision"] == "block"
    assert "PreToolUse hook failed" in payload["reason"]


def test_stop_hook_blocks_hard_contradiction_from_nesy_facts(tmp_path: Path) -> None:
    config_path, sqlite_path = _write_sqlite_config(tmp_path)
    writer = create_hook_store(
        NesyConfig(storage=StorageConfig(backend="sqlite", sqlite_path=str(sqlite_path))),
        stderr=StringIO(),
    )
    writer.assert_exclusive([ExclusiveGroupInput(group_id="state", members=["B", "C"])])
    stdin = StringIO(
        json.dumps(
            {
                "hook_event_name": "Stop",
                "stop_hook_active": False,
                "last_assistant_message": (
                    "Done.\nNESY_FACTS:\n"
                    "["
                    '{"source":"A","target":"B","relation_type":"sufficient"},'
                    '{"source":"A","target":"C","relation_type":"sufficient"}'
                    "]"
                ),
            }
        )
    )
    stdout = StringIO()

    code = run_stop_hook(
        stdin=stdin,
        stdout=stdout,
        stderr=StringIO(),
        env={"NESY_CONFIG": str(config_path)},
    )

    payload = json.loads(stdout.getvalue())
    assert code == 0
    assert payload["decision"] == "block"
    assert "hard contradiction" in payload["reason"]


def test_stop_hook_checks_current_graph_without_nesy_facts(tmp_path: Path) -> None:
    config_path, sqlite_path = _write_sqlite_config(tmp_path)
    writer = create_hook_store(
        NesyConfig(storage=StorageConfig(backend="sqlite", sqlite_path=str(sqlite_path))),
        stderr=StringIO(),
    )
    writer.assert_exclusive([ExclusiveGroupInput(group_id="state", members=["B", "C"])])
    writer.assert_relations(
        [
            RelationInput(source="A", target="B", relation_type=RelationType.SUFFICIENT),
            RelationInput(source="A", target="C", relation_type=RelationType.SUFFICIENT),
        ]
    )
    stdout = StringIO()

    run_stop_hook(
        stdin=StringIO(json.dumps({"last_assistant_message": "No facts."})),
        stdout=stdout,
        stderr=StringIO(),
        env={"NESY_CONFIG": str(config_path)},
    )

    assert json.loads(stdout.getvalue())["decision"] == "block"


def test_stop_hook_stop_hook_active_allows_without_checking(tmp_path: Path) -> None:
    config_path, _sqlite_path = _write_sqlite_config(tmp_path)
    stdout = StringIO()

    run_stop_hook(
        stdin=StringIO(json.dumps({"stop_hook_active": True})),
        stdout=stdout,
        stderr=StringIO(),
        env={"NESY_CONFIG": str(config_path), "NESY_HOOK_FAIL_CLOSED": "true"},
    )

    assert json.loads(stdout.getvalue()) == {}


def test_stop_hook_context_separated_conflict_does_not_block(tmp_path: Path) -> None:
    config_path, sqlite_path = _write_sqlite_config(tmp_path)
    writer = create_hook_store(
        NesyConfig(storage=StorageConfig(backend="sqlite", sqlite_path=str(sqlite_path))),
        stderr=StringIO(),
    )
    writer.assert_exclusive(
        [
            ExclusiveGroupInput(group_id="state_q3", members=["B", "C"], context_id="q3"),
            ExclusiveGroupInput(group_id="state_q4", members=["B", "C"], context_id="q4"),
        ]
    )
    writer.assert_relations(
        [
            RelationInput(
                source="A",
                target="B",
                relation_type=RelationType.SUFFICIENT,
                context_id="q3",
            ),
            RelationInput(
                source="A",
                target="C",
                relation_type=RelationType.SUFFICIENT,
                context_id="q4",
            ),
        ]
    )
    stdout = StringIO()

    run_stop_hook(
        stdin=StringIO(json.dumps({"last_assistant_message": "No facts."})),
        stdout=stdout,
        stderr=StringIO(),
        env={"NESY_CONFIG": str(config_path)},
    )

    assert json.loads(stdout.getvalue()) == {}


def test_stop_hook_invalid_facts_fail_open_by_default() -> None:
    stdout = StringIO()
    stderr = StringIO()

    run_stop_hook(
        stdin=StringIO(json.dumps({"last_assistant_message": "NESY_FACTS:\n{bad"})),
        stdout=stdout,
        stderr=stderr,
        env={},
    )

    assert json.loads(stdout.getvalue()) == {}
    assert "Stop hook failed" in stderr.getvalue()


def _write_sqlite_config(tmp_path: Path) -> tuple[Path, Path]:
    config_path = tmp_path / "nesy.json"
    sqlite_path = tmp_path / "nesy.db"
    config_path.write_text(
        json.dumps({"storage": {"backend": "sqlite", "sqlite_path": str(sqlite_path)}}),
        encoding="utf-8",
    )
    return config_path, sqlite_path
