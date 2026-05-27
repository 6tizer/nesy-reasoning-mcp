"""Run a small NeSy-backed architecture guard over observed code facts."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import anyio

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from nesy_reasoning_mcp.config import NesyConfig
from nesy_reasoning_mcp.store import MemoryRelationStore
from nesy_reasoning_mcp.tool_reasoning import verify_chain
from nesy_reasoning_mcp.tool_relations import (
    assert_exclusive,
    assert_relations,
    check_contradictions,
)

DEFAULT_ANCHOR = "ObservedArchitectureFacts"
ARCHITECTURE_VIOLATION_TARGET = "ArchitectureViolation"


def build_parser() -> argparse.ArgumentParser:
    """Build the architecture guard CLI parser."""
    parser = argparse.ArgumentParser(prog="architecture_guard.py")
    subparsers = parser.add_subparsers(dest="command")
    collect_parser = subparsers.add_parser(
        "collect",
        help="Convert a GitNexus observation into architecture guard facts JSON.",
    )
    collect_parser.add_argument("--target", required=True, help="Observed proposition target.")
    collect_parser.add_argument("--anchor", default=DEFAULT_ANCHOR)
    collect_parser.add_argument("--confidence", type=float, default=1.0)
    collect_parser.add_argument(
        "--output", default="-", help="Output facts path, or '-' for stdout."
    )
    collect_parser.add_argument("--note", default=None)
    evidence_source = collect_parser.add_mutually_exclusive_group(required=True)
    evidence_source.add_argument(
        "--gitnexus-query", default=None, help="GitNexus query text to run."
    )
    evidence_source.add_argument(
        "--evidence-file", default=None, help="Saved GitNexus output file."
    )
    collect_parser.add_argument("--repo", default="nesy-reasoning-mcp")
    collect_parser.add_argument("--limit", type=int, default=5)
    collect_parser.add_argument("--evidence-command", default=None)
    parser.add_argument("--rules", help="Architecture guard rules JSON file.")
    parser.add_argument("--facts", help="Observed facts JSON file.")
    parser.add_argument("--format", choices=["text", "json"], default="text")
    return parser


async def run_guard(rules: dict[str, Any], facts: dict[str, Any]) -> dict[str, Any]:
    """Evaluate observed facts against architecture rules using NeSy reasoning."""
    store = MemoryRelationStore(NesyConfig())
    await _load_rules(store, rules)
    await _load_facts(store, facts)

    anchor = str(facts.get("anchor") or rules.get("anchor") or DEFAULT_ANCHOR)
    checks = _required_non_empty_list(rules, "checks")
    violations = []
    for check in checks:
        target = _required_str(check, "target")
        result = await _verify_sufficient_path(
            store, anchor, target, int(check.get("max_depth", 8))
        )
        if result["source_to_target_reachable"]:
            violations.append(
                {
                    "id": _required_str(check, "id"),
                    "severity": check.get("severity", "error"),
                    "target": target,
                    "description": check.get("description", ""),
                    "best_path": result["best_path"],
                    "trace": result["trace"],
                }
            )

    violation_result = await _verify_sufficient_path(
        store, anchor, ARCHITECTURE_VIOLATION_TARGET, 8
    )
    has_error_violation = any(
        str(item.get("severity", "error")).lower() == "error" for item in violations
    )
    if violation_result["source_to_target_reachable"] and not has_error_violation:
        violations.append(
            {
                "id": "architecture-violation",
                "severity": "error",
                "target": ARCHITECTURE_VIOLATION_TARGET,
                "description": "Observed facts imply ArchitectureViolation.",
                "best_path": violation_result["best_path"],
                "trace": violation_result["trace"],
            }
        )

    contradictions = await check_contradictions(
        {
            "mode": "graph",
            "include_soft": True,
            "max_depth": 8,
        },
        store,
    )
    hard_failures = [
        item for item in violations if str(item.get("severity", "error")).lower() == "error"
    ]
    status = "fail" if hard_failures or contradictions["has_contradictions"] else "pass"
    return {
        "status": status,
        "anchor": anchor,
        "checked_rules": len(checks),
        "violations": violations,
        "contradictions": contradictions["contradictions"],
        "graph_stats": contradictions["graph_stats"],
    }


async def _verify_sufficient_path(
    store: MemoryRelationStore,
    source: str,
    target: str,
    max_depth: int,
) -> dict[str, Any]:
    return await verify_chain(
        {
            "source": source,
            "target": target,
            "expected_relation": "sufficient",
            "max_depth": max_depth,
            "max_paths": 3,
        },
        store,
    )


async def _load_rules(store: MemoryRelationStore, rules: dict[str, Any]) -> None:
    relations = _required_list(rules, "relations")
    if relations:
        await assert_relations(
            {
                "relations": relations,
                "check_contradictions": False,
                "on_contradiction": "warn",
            },
            store,
        )

    exclusive_groups = rules.get("exclusive_groups", [])
    if exclusive_groups:
        await assert_exclusive({"groups": exclusive_groups}, store)


async def _load_facts(store: MemoryRelationStore, facts: dict[str, Any]) -> None:
    relations = _required_list(facts, "relations")
    if relations:
        await assert_relations(
            {
                "relations": relations,
                "check_contradictions": False,
                "on_contradiction": "warn",
            },
            store,
        )


def _required_list(data: dict[str, Any], key: str) -> list[dict[str, Any]]:
    value = data.get(key, [])
    if not isinstance(value, list):
        raise ValueError(f"{key} must be a list")
    if not all(isinstance(item, dict) for item in value):
        raise ValueError(f"{key} must contain objects")
    return value


def _required_non_empty_list(data: dict[str, Any], key: str) -> list[dict[str, Any]]:
    value = _required_list(data, key)
    if not value:
        raise ValueError(f"{key} must contain at least one object")
    return value


def _required_str(data: dict[str, Any], key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} must be a non-empty string")
    return value.strip()


def _load_json(path: str) -> dict[str, Any]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return data


def collect_facts(args: argparse.Namespace) -> dict[str, Any]:
    """Convert GitNexus evidence into a single observed architecture fact."""
    anchor = _non_empty_value(args.anchor, "anchor")
    target = _non_empty_value(args.target, "target")
    confidence = float(args.confidence)
    if not 0 <= confidence <= 1:
        raise ValueError("confidence must be between 0 and 1")

    command, evidence = _collect_evidence(args)
    if not evidence.strip():
        raise ValueError("gitnexus evidence must not be empty")
    provenance: dict[str, Any] = {
        "source": "gitnexus",
        "command": command,
        "excerpt": evidence[:4000],
    }
    if args.note:
        provenance["note"] = args.note
    return {
        "anchor": anchor,
        "relations": [
            {
                "source": anchor,
                "target": target,
                "relation_type": "sufficient",
                "confidence": confidence,
                "provenance": provenance,
            }
        ],
    }


def _collect_evidence(args: argparse.Namespace) -> tuple[str, str]:
    if args.evidence_file:
        evidence_path = Path(args.evidence_file)
        command = args.evidence_command or f"read {evidence_path}"
        return command, evidence_path.read_text(encoding="utf-8")

    limit = int(args.limit)
    if not 1 <= limit <= 50:
        raise ValueError("limit must be between 1 and 50")
    command_parts = [
        "npx",
        "gitnexus",
        "query",
        _non_empty_value(args.gitnexus_query, "gitnexus_query"),
        "-r",
        _non_empty_value(args.repo, "repo"),
        "--limit",
        str(limit),
    ]
    try:
        completed = subprocess.run(
            command_parts,
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except subprocess.TimeoutExpired as exc:
        raise ValueError("gitnexus query timed out after 30 seconds") from exc
    command = " ".join(command_parts)
    if completed.returncode != 0:
        message = completed.stderr.strip() or completed.stdout.strip() or "gitnexus query failed"
        raise ValueError(message)
    return command, completed.stdout


def _non_empty_value(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


def _write_json_output(data: dict[str, Any], path: str) -> None:
    rendered = json.dumps(data, indent=2, sort_keys=True)
    if path == "-":
        print(rendered)
        return
    Path(path).write_text(f"{rendered}\n", encoding="utf-8")


def _render_text(report: dict[str, Any]) -> str:
    lines = [
        f"status: {report['status']}",
        f"anchor: {report['anchor']}",
        f"checked_rules: {report['checked_rules']}",
        f"violations: {len(report['violations'])}",
        f"contradictions: {len(report['contradictions'])}",
    ]
    for violation in report["violations"]:
        lines.append(f"- {violation['id']}: {violation['target']}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """Run architecture guard CLI."""
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "collect":
            _write_json_output(collect_facts(args), args.output)
            return 0
        if not args.rules or not args.facts:
            raise ValueError("guard mode requires --rules and --facts")
        report = anyio.run(run_guard, _load_json(args.rules), _load_json(args.facts))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(str(exc), file=sys.stderr)
        return 2

    if args.format == "json":
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(_render_text(report))
    return 1 if report["status"] == "fail" else 0


if __name__ == "__main__":
    raise SystemExit(main())
