# Internal Test Profile

This profile wires the v1.0 MCP server, Claude Code hooks, and SQLite persistence
for local internal testing.

## Files

- `nesy-config.json`: SQLite-backed local config.
- `mcp-stdio-config.json`: MCP client stdio config.
- `claude-hooks.json`: Claude Code hook config using the local wrappers.
- `hook-pretooluse.sh` and `hook-stop.sh`: thin hook wrappers.
- `run-http.sh`: local Streamable HTTP launcher.
- `smoke.py`: end-to-end SQLite + hook smoke test.
- `agent-instructions.md`: fact-writing protocol for agents.
- `tool-policy.md`: confirmation policy for MCP tool use.

## Quick Smoke

```bash
cd /path/to/nesy-reasoning-mcp
env PYTHONPATH=src uv run python examples/internal-test/smoke.py
```

## Stdio MCP

Use `mcp-stdio-config.json` in a local MCP client. It points `NESY_CONFIG` at
`examples/internal-test/nesy-config.json`, which stores data in
`~/.nesy-reasoning/internal-test/nesy.db`.

## Hooks

Use `claude-hooks.json` as a template for Claude Code hooks. The wrappers default
to the internal-test config and fail open unless `NESY_HOOK_FAIL_CLOSED=true` is
set.

## HTTP

Run:

```bash
bash /path/to/nesy-reasoning-mcp/examples/internal-test/run-http.sh
```

The default token is `nesy-internal-test-token`. Override it with
`NESY_LOCAL_TOKEN` before starting the daemon.
