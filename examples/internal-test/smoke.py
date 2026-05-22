#!/usr/bin/env python3
"""Smoke test for the internal-test SQLite + hook profile."""

from __future__ import annotations

import json
import tempfile
from io import StringIO
from pathlib import Path

from nesy_reasoning_mcp.config import load_config
from nesy_reasoning_mcp.hooks import create_hook_store, run_pretooluse_hook, run_stop_hook
from nesy_reasoning_mcp.schemas import ExclusiveGroupInput, RelationInput, RelationType


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="nesy-internal-test-") as tmp:
        base = Path(tmp)
        config_path = base / "nesy-config.json"
        sqlite_path = base / "nesy.db"
        config_path.write_text(
            json.dumps(
                {
                    "storage": {
                        "backend": "sqlite",
                        "sqlite_path": str(sqlite_path),
                    },
                    "security": {
                        "allowed_roots": [str(base)],
                    },
                    "hook": {
                        "fail_closed": False,
                    },
                }
            ),
            encoding="utf-8",
        )
        env = {"NESY_CONFIG": str(config_path)}
        config = load_config(env=env, cwd=base)
        store = create_hook_store(config, stderr=StringIO())
        store.assert_relations(
            [
                RelationInput(
                    source="降价",
                    target="销量增加",
                    relation_type=RelationType.SUFFICIENT,
                )
            ]
        )

        reloaded = create_hook_store(load_config(env=env, cwd=base), stderr=StringIO())
        assert reloaded.list_relations()[0].target == "销量增加"

        pre_stdout = StringIO()
        pre_code = run_pretooluse_hook(
            stdin=StringIO(
                json.dumps(
                    {
                        "hook_event_name": "PreToolUse",
                        "tool_name": "Bash",
                        "tool_input": {"command": "降价"},
                        "cwd": str(base),
                    }
                )
            ),
            stdout=pre_stdout,
            stderr=StringIO(),
            env=env,
        )
        pre_payload = json.loads(pre_stdout.getvalue())
        assert pre_code == 0
        assert "降价 sufficient 销量增加" in pre_payload["hookSpecificOutput"]["additionalContext"]

        reloaded.assert_exclusive([ExclusiveGroupInput(group_id="state", members=["B", "C"])])
        stop_stdout = StringIO()
        stop_code = run_stop_hook(
            stdin=StringIO(
                json.dumps(
                    {
                        "hook_event_name": "Stop",
                        "stop_hook_active": False,
                        "last_assistant_message": (
                            "NESY_FACTS:\n"
                            "["
                            '{"source":"A","target":"B","relation_type":"sufficient"},'
                            '{"source":"A","target":"C","relation_type":"sufficient"}'
                            "]"
                        ),
                    }
                )
            ),
            stdout=stop_stdout,
            stderr=StringIO(),
            env=env,
        )
        stop_payload = json.loads(stop_stdout.getvalue())
        assert stop_code == 0
        assert stop_payload["decision"] == "block"

    print("internal-test smoke ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
