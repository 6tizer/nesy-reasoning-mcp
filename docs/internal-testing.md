# Internal Testing

This guide packages the existing v1.0 server for local internal testing with
durable SQLite state, MCP client config, Claude Code hooks, and explicit agent
rules.

## Default Profile

Use the profile in [examples/internal-test](../examples/internal-test/README.md).
It stores relations in:

```text
~/.nesy-reasoning/internal-test/nesy.db
```

Create the local directories:

```bash
mkdir -p ~/.nesy-reasoning/internal-test ~/.nesy-reasoning/relation_sets
```

Run the profile smoke test:

```bash
cd /Users/mac-mini/Documents/nesy-reasoning-mcp
env PYTHONPATH=src uv run python examples/internal-test/smoke.py
```

Expected output:

```text
internal-test smoke ok
```

## MCP Client Setup

For stdio MCP clients, use:

```text
examples/internal-test/mcp-stdio-config.json
```

The config points `NESY_CONFIG` at the internal-test SQLite profile. This makes
MCP server restarts preserve relations.

## Streamable HTTP Setup

Start the local daemon:

```bash
bash /Users/mac-mini/Documents/nesy-reasoning-mcp/examples/internal-test/run-http.sh
```

The default token is `nesy-internal-test-token`. Override it before launch when
needed:

```bash
NESY_LOCAL_TOKEN='change-me' \
  bash /Users/mac-mini/Documents/nesy-reasoning-mcp/examples/internal-test/run-http.sh
```

Health check:

```bash
curl -H 'Authorization: Bearer nesy-internal-test-token' http://127.0.0.1:8765/healthz
```

HTTP mode is a local daemon with bearer-token auth. It is not hosted multi-user
auth.

## Claude Code Hooks

Use this hook template:

```text
examples/internal-test/claude-hooks.json
```

Hooks share the same SQLite config as MCP. Memory storage is not suitable for
hook integration because each hook runs in a separate process.

Default hook behavior is fail-open. For stricter internal tests:

```bash
export NESY_HOOK_FAIL_CLOSED=true
```

## Agent Fact Protocol

Agents must write structured facts explicitly. Plain natural-language answers
are not persisted and are not fully checked by the Stop hook.

Use:

- `nesy.assert_relations` during tool use for stable facts.
- `NESY_FACTS:` in the final answer for new facts that should be checked before
  stopping.

See [agent-instructions.md](../examples/internal-test/agent-instructions.md).

## Tool Policy

Internal tests should treat destructive and file tools as confirmation-required.
See [tool-policy.md](../examples/internal-test/tool-policy.md).

## SQLite Maintenance

Backup:

```bash
cp ~/.nesy-reasoning/internal-test/nesy.db \
  ~/.nesy-reasoning/internal-test/nesy.$(date +%Y%m%d-%H%M%S).db
```

Reset:

```bash
rm -f ~/.nesy-reasoning/internal-test/nesy.db
```

Export a relation set through MCP before sharing or archiving long-lived state.

## Known Boundaries

- Internal testing defaults to SQLite; memory is only for temporary debugging.
- Stop hook checks the explicit graph and `NESY_FACTS`, not arbitrary prose.
- `clear_relations(scope=all)`, `load_relations(mode=replace_store)`, and file
  load/export should require explicit confirmation.
- Contradiction checks are deterministic over structured facts only. Explicit
  negation uses proposition labels such as `not X`, `not:X`, and `¬X`; arbitrary
  natural-language negation is not extracted automatically.
