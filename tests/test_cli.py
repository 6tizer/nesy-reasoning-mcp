import json
import os
import subprocess
import sys
from pathlib import Path

import anyio

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
from nesy_reasoning_mcp.config import NesyConfig, StorageConfig
from nesy_reasoning_mcp.store import create_relation_store
from nesy_reasoning_mcp.tools import ASSERT_RELATIONS, call_tool


def _cli_env(extra: dict[str, str] | None = None) -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")
    if extra:
        env.update(extra)
    return env


def _write_sqlite_config(tmp_path: Path) -> tuple[Path, Path]:
    config_path = tmp_path / "nesy.json"
    sqlite_path = tmp_path / "nesy.db"
    config_path.write_text(
        json.dumps({"storage": {"backend": "sqlite", "sqlite_path": str(sqlite_path)}}),
        encoding="utf-8",
    )
    return config_path, sqlite_path


def _review_queue_record(record_id: str = "queue-1") -> ReviewQueueRecord:
    candidate = CandidateRelation(
        id=f"candidate-{record_id}",
        source="A",
        target="B",
        relation_type="sufficient",
        confidence=0.9,
        evidence=[EvidenceRecord(url="https://example.com/source", span="A enables B.")],
    )
    return ReviewQueueRecord(
        id=record_id,
        run_id="run-1",
        candidate=candidate,
        review=ReviewDecision(
            candidate_id=candidate.id,
            decision=ReviewDecisionValue.APPROVE,
            final_relation_type="sufficient",
            final_confidence=0.9,
            reasons=["Evidence directly supports the relation."],
        ),
        gate_result=GateResult(candidate_id=candidate.id, action=GateAction.QUEUE),
    )


def test_help_writes_no_stdout_banner() -> None:
    completed = subprocess.run(
        [sys.executable, "-m", "nesy_reasoning_mcp", "--help"],
        check=True,
        capture_output=True,
        env=_cli_env(),
        text=True,
    )

    assert "usage: nesy-reasoning-mcp" in completed.stdout
    assert "http" in completed.stdout
    assert "eval" in completed.stdout
    assert "audit" in completed.stdout
    assert "ingest" in completed.stdout
    assert completed.stderr == ""


def test_eval_help_lists_llm_subcommand() -> None:
    completed = subprocess.run(
        [sys.executable, "-m", "nesy_reasoning_mcp", "eval", "--help"],
        check=True,
        capture_output=True,
        env=_cli_env(),
        text=True,
    )

    assert "run" in completed.stdout
    assert "llm" in completed.stdout
    assert completed.stderr == ""


def test_ingest_help_lists_agent_dry_run_subcommand() -> None:
    completed = subprocess.run(
        [sys.executable, "-m", "nesy_reasoning_mcp", "ingest", "--help"],
        check=True,
        capture_output=True,
        env=_cli_env(),
        text=True,
    )

    assert "agent-dry-run" in completed.stdout
    assert "queue" in completed.stdout
    assert completed.stderr == ""


def test_ingest_agent_dry_run_help_lists_safe_write_flags() -> None:
    completed = subprocess.run(
        [sys.executable, "-m", "nesy_reasoning_mcp", "ingest", "agent-dry-run", "--help"],
        check=True,
        capture_output=True,
        env=_cli_env(),
        text=True,
    )

    assert "--auto-write" in completed.stdout
    assert "--min-write-confidence" in completed.stdout
    assert "--provider" in completed.stdout
    assert "--list-providers" in completed.stdout
    assert "--base-url" in completed.stdout
    assert "--api-key-env" in completed.stdout
    assert "--provider-header" in completed.stdout
    assert "--disable-tracing" in completed.stdout
    assert "--reviewer-model" in completed.stdout
    assert "--voting-policy" in completed.stdout
    assert "--high-priority-reviewer-model" in completed.stdout
    assert completed.stderr == ""


def test_ingest_queue_list_reads_persisted_records(tmp_path: Path) -> None:
    config_path, sqlite_path = _write_sqlite_config(tmp_path)
    store = create_relation_store(
        NesyConfig(storage=StorageConfig(backend="sqlite", sqlite_path=str(sqlite_path)))
    )
    store.enqueue_review_queue([_review_queue_record()])

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "nesy_reasoning_mcp",
            "ingest",
            "queue",
            "list",
            "--format",
            "json",
        ],
        check=True,
        capture_output=True,
        env=_cli_env({"NESY_CONFIG": str(config_path)}),
        text=True,
    )

    payload = json.loads(completed.stdout)
    assert payload["records"][0]["id"] == "queue-1"
    assert payload["records"][0]["candidate"]["id"] == "candidate-queue-1"
    assert completed.stderr == ""


def test_ingest_queue_commit_commits_persisted_record(tmp_path: Path) -> None:
    config_path, sqlite_path = _write_sqlite_config(tmp_path)
    store = create_relation_store(
        NesyConfig(storage=StorageConfig(backend="sqlite", sqlite_path=str(sqlite_path)))
    )
    store.enqueue_review_queue([_review_queue_record()])

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "nesy_reasoning_mcp",
            "ingest",
            "queue",
            "commit",
            "--id",
            "queue-1",
            "--format",
            "json",
        ],
        check=True,
        capture_output=True,
        env=_cli_env({"NESY_CONFIG": str(config_path)}),
        text=True,
    )

    payload = json.loads(completed.stdout)
    reloaded = create_relation_store(
        NesyConfig(storage=StorageConfig(backend="sqlite", sqlite_path=str(sqlite_path)))
    )
    assert payload["status"] == "ok"
    assert payload["committed_count"] == 1
    assert len(reloaded.list_relations()) == 1
    assert reloaded.list_review_queue()[0].status == ReviewQueueStatus.COMMITTED
    assert completed.stderr == ""


def test_ingest_queue_resolve_resolves_persisted_record(tmp_path: Path) -> None:
    config_path, sqlite_path = _write_sqlite_config(tmp_path)
    store = create_relation_store(
        NesyConfig(storage=StorageConfig(backend="sqlite", sqlite_path=str(sqlite_path)))
    )
    store.enqueue_review_queue([_review_queue_record()])

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "nesy_reasoning_mcp",
            "ingest",
            "queue",
            "resolve",
            "--id",
            "queue-1",
            "--reason",
            "duplicate",
            "--format",
            "json",
        ],
        check=True,
        capture_output=True,
        env=_cli_env({"NESY_CONFIG": str(config_path)}),
        text=True,
    )

    payload = json.loads(completed.stdout)
    reloaded = create_relation_store(
        NesyConfig(storage=StorageConfig(backend="sqlite", sqlite_path=str(sqlite_path)))
    )
    assert payload["status"] == "ok"
    assert payload["resolved_count"] == 1
    assert reloaded.list_review_queue()[0].status == ReviewQueueStatus.RESOLVED
    assert reloaded.list_relations() == []
    assert completed.stderr == ""


def test_audit_list_json_filters_without_raw_arguments(tmp_path: Path) -> None:
    config_path = tmp_path / "nesy.json"
    sqlite_path = tmp_path / "nesy.db"
    config_path.write_text(
        json.dumps({"storage": {"backend": "sqlite", "sqlite_path": str(sqlite_path)}}),
        encoding="utf-8",
    )
    store = create_relation_store(
        NesyConfig(storage=StorageConfig(backend="sqlite", sqlite_path=str(sqlite_path)))
    )
    store.record_audit(
        event_type="tool_call",
        tool_name="nesy.assert_relations",
        arguments={"secret": "do-not-print"},
        result_status="ok",
    )
    store.record_audit(
        event_type="tool_call",
        tool_name="nesy.clear_relations",
        arguments={"scope": "context"},
        result_status="error",
    )

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "nesy_reasoning_mcp",
            "audit",
            "list",
            "--format",
            "json",
            "--tool-name",
            "nesy.assert_relations",
        ],
        check=True,
        capture_output=True,
        env=_cli_env({"NESY_CONFIG": str(config_path)}),
        text=True,
    )

    payload = json.loads(completed.stdout)
    assert payload["count"] == 1
    assert payload["entries"][0]["tool_name"] == "nesy.assert_relations"
    assert payload["entries"][0]["result_status"] == "ok"
    assert "do-not-print" not in completed.stdout
    assert completed.stderr == ""


def test_audit_list_text_limit_shows_recent_first(tmp_path: Path) -> None:
    config_path = tmp_path / "nesy.json"
    sqlite_path = tmp_path / "nesy.db"
    config_path.write_text(
        json.dumps({"storage": {"backend": "sqlite", "sqlite_path": str(sqlite_path)}}),
        encoding="utf-8",
    )
    store = create_relation_store(
        NesyConfig(storage=StorageConfig(backend="sqlite", sqlite_path=str(sqlite_path)))
    )
    store.record_audit(
        event_type="tool_call",
        tool_name="nesy.assert_relations",
        arguments={},
        result_status="ok",
    )
    store.record_audit(
        event_type="tool_call",
        tool_name="nesy.export_relations",
        arguments={},
        result_status="ok",
    )

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "nesy_reasoning_mcp",
            "audit",
            "list",
            "--limit",
            "1",
        ],
        check=True,
        capture_output=True,
        env=_cli_env({"NESY_CONFIG": str(config_path)}),
        text=True,
    )

    assert "nesy.export_relations" in completed.stdout
    assert "nesy.assert_relations" not in completed.stdout
    assert completed.stderr == ""


def test_audit_list_reads_json_backend(tmp_path: Path) -> None:
    config_path = tmp_path / "nesy.json"
    json_path = tmp_path / "relations.json"
    config_path.write_text(
        json.dumps({"storage": {"backend": "json", "json_path": str(json_path)}}),
        encoding="utf-8",
    )
    store = create_relation_store(
        NesyConfig(storage=StorageConfig(backend="json", json_path=str(json_path)))
    )
    store.record_audit(
        event_type="tool_call",
        tool_name="nesy.load_relations",
        arguments={},
        result_status="ok",
    )

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "nesy_reasoning_mcp",
            "audit",
            "list",
            "--format",
            "json",
        ],
        check=True,
        capture_output=True,
        env=_cli_env({"NESY_CONFIG": str(config_path)}),
        text=True,
    )

    payload = json.loads(completed.stdout)
    assert payload["count"] == 1
    assert payload["entries"][0]["tool_name"] == "nesy.load_relations"
    assert completed.stderr == ""


def test_audit_list_default_memory_backend_is_empty() -> None:
    completed = subprocess.run(
        [sys.executable, "-m", "nesy_reasoning_mcp", "audit", "list"],
        check=True,
        capture_output=True,
        env=_cli_env(),
        text=True,
    )

    assert completed.stdout == "No audit entries.\n"
    assert completed.stderr == ""


def test_audit_list_reads_entry_from_mutating_tool_call(tmp_path: Path) -> None:
    config_path = tmp_path / "nesy.json"
    sqlite_path = tmp_path / "nesy.db"
    config_path.write_text(
        json.dumps({"storage": {"backend": "sqlite", "sqlite_path": str(sqlite_path)}}),
        encoding="utf-8",
    )
    store = create_relation_store(
        NesyConfig(storage=StorageConfig(backend="sqlite", sqlite_path=str(sqlite_path)))
    )

    anyio.run(
        call_tool,
        ASSERT_RELATIONS,
        {
            "relations": [{"source": "A", "target": "B", "relation_type": "sufficient"}],
            "check_contradictions": False,
        },
        store,
    )

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "nesy_reasoning_mcp",
            "audit",
            "list",
            "--format",
            "json",
        ],
        check=True,
        capture_output=True,
        env=_cli_env({"NESY_CONFIG": str(config_path)}),
        text=True,
    )

    payload = json.loads(completed.stdout)
    assert payload["count"] == 1
    assert payload["entries"][0]["tool_name"] == ASSERT_RELATIONS
    assert payload["entries"][0]["result_status"] == "ok"
    assert "arguments" not in payload["entries"][0]
    assert '"source"' not in completed.stdout
    assert '"target"' not in completed.stdout
    assert "sufficient" not in completed.stdout
    assert completed.stderr == ""
