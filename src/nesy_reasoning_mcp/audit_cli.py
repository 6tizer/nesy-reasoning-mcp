"""Audit log CLI helpers."""

from __future__ import annotations

import json
from argparse import Namespace
from typing import Any

from nesy_reasoning_mcp.config import load_config
from nesy_reasoning_mcp.store import create_relation_store


def run_audit_cli(args: Namespace) -> int:
    """Run audit subcommands from parsed CLI arguments."""
    if args.audit_command == "list":
        return _run_audit_list(args)
    raise ValueError("audit command requires a subcommand")


def _run_audit_list(args: Namespace) -> int:
    if args.limit < 1:
        raise ValueError("--limit must be >= 1")
    config = load_config()
    store = create_relation_store(config)
    entries = _filtered_entries(
        store.list_audit_entries(),
        tool_name=args.tool_name,
        result_status=args.result_status,
        limit=args.limit,
    )
    if args.format == "json":
        print(json.dumps({"count": len(entries), "entries": entries}, ensure_ascii=False, indent=2))
        return 0

    if not entries:
        print("No audit entries.")
        return 0
    for entry in entries:
        print(
            " ".join(
                [
                    entry["created_at"],
                    entry["tool_name"],
                    entry["result_status"],
                    entry["event_type"],
                    entry["input_hash"],
                    entry["id"],
                ]
            )
        )
    return 0


def _filtered_entries(
    entries: list[dict[str, Any]],
    *,
    tool_name: str | None,
    result_status: str | None,
    limit: int,
) -> list[dict[str, Any]]:
    filtered = [
        entry
        for entry in entries
        if (tool_name is None or entry["tool_name"] == tool_name)
        and (result_status is None or entry["result_status"] == result_status)
    ]
    return list(reversed(filtered[-limit:]))
