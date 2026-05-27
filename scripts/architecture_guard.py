"""Run a small NeSy-backed architecture guard over observed code facts."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import json
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


def build_parser() -> argparse.ArgumentParser:
    """Build the architecture guard CLI parser."""
    parser = argparse.ArgumentParser(prog="architecture_guard.py")
    parser.add_argument("--rules", required=True, help="Architecture guard rules JSON file.")
    parser.add_argument("--facts", required=True, help="Observed facts JSON file.")
    parser.add_argument("--format", choices=["text", "json"], default="text")
    return parser


async def run_guard(rules: dict[str, Any], facts: dict[str, Any]) -> dict[str, Any]:
    """Evaluate observed facts against architecture rules using NeSy reasoning."""
    store = MemoryRelationStore(NesyConfig())
    await _load_rules(store, rules)
    await _load_facts(store, facts)

    anchor = str(facts.get("anchor") or rules.get("anchor") or DEFAULT_ANCHOR)
    checks = _required_list(rules, "checks")
    violations = []
    for check in checks:
        target = _required_str(check, "target")
        result = await verify_chain(
            {
                "source": anchor,
                "target": target,
                "expected_relation": "sufficient",
                "max_depth": int(check.get("max_depth", 8)),
                "max_paths": 3,
            },
            store,
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
