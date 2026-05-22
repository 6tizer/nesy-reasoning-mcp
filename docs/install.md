# Install NeSy Reasoning MCP

This project runs as a local MCP server over stdio or authenticated Streamable HTTP.

## Requirements

- Python 3.11 to 3.14
- `uv`
- Local checkout at `/Users/mac-mini/Documents/nesy-reasoning-mcp`

## Verify Locally

```bash
cd /Users/mac-mini/Documents/nesy-reasoning-mcp
uv sync
uv run ruff check .
uv run pytest
printf '' | env PYTHONPATH=src uv run nesy-reasoning-mcp --transport stdio
```

Expected:

- Ruff passes.
- Pytest passes.
- The stdio smoke command prints nothing.

## MCP Client Config

Add this server to your MCP client config:

```json
{
  "mcpServers": {
    "nesy-reasoning": {
      "command": "uv",
      "args": [
        "--directory",
        "/Users/mac-mini/Documents/nesy-reasoning-mcp",
        "run",
        "nesy-reasoning-mcp",
        "--transport",
        "stdio"
      ],
      "env": {
        "PYTHONPATH": "/Users/mac-mini/Documents/nesy-reasoning-mcp/src"
      }
    }
  }
}
```

Then restart or reload the MCP client.

## Available Tools

- `nesy.assert_relations`
- `nesy.list_relations`
- `nesy.clear_relations`
- `nesy.classify`
- `nesy.verify_chain`
- `nesy.assert_exclusive`
- `nesy.check_contradictions`
- `nesy.load_relations`
- `nesy.export_relations`
- `nesy.summarize_graph`
- `nesy.counterfactual`

## Persistent Storage

Default storage is process memory. Restarting the server clears state unless you
choose a persistent backend.

SQLite:

```json
{
  "env": {
    "PYTHONPATH": "/Users/mac-mini/Documents/nesy-reasoning-mcp/src",
    "NESY_STORAGE_BACKEND": "sqlite",
    "NESY_SQLITE_PATH": "/Users/mac-mini/.nesy-reasoning/nesy.db",
    "NESY_ALLOWED_ROOTS": "/Users/mac-mini/Documents/nesy-reasoning-mcp,/Users/mac-mini/.nesy-reasoning/relation_sets"
  }
}
```

JSON file:

```json
{
  "env": {
    "PYTHONPATH": "/Users/mac-mini/Documents/nesy-reasoning-mcp/src",
    "NESY_CONFIG": "/Users/mac-mini/.nesy-reasoning/config.json"
  }
}
```

Example config file: [examples/nesy-config.json](../examples/nesy-config.json)

```json
{
  "storage": {
    "backend": "json",
    "json_path": "/Users/mac-mini/.nesy-reasoning/relations.json"
  },
  "security": {
    "allowed_roots": [
      "/Users/mac-mini/Documents/nesy-reasoning-mcp",
      "/Users/mac-mini/.nesy-reasoning/relation_sets"
    ],
    "max_file_size_bytes": 5242880,
    "allow_hidden_relation_paths": false
  }
}
```

File load/export only accepts `.json` and `.jsonl` inside `allowed_roots`.
`nesy.load_relations` also accepts `source_type=resource_uri` for safe `file://`
URIs that resolve inside `allowed_roots`; remote URI schemes are rejected.
Hidden relation paths under an allowed root are rejected by default. Set
`security.allow_hidden_relation_paths=true` or `NESY_ALLOW_HIDDEN_RELATION_PATHS=true`
only when hidden `.json` or `.jsonl` paths are intentional.

## Streamable HTTP Mode

HTTP mode starts a local daemon at `127.0.0.1:8765/mcp` by default. It requires
`NESY_LOCAL_TOKEN`; send it as a bearer token.

```bash
cd /Users/mac-mini/Documents/nesy-reasoning-mcp
NESY_LOCAL_TOKEN='change-me' uv run nesy-reasoning-mcp --transport http
```

Common HTTP env overrides:

- `NESY_HTTP_HOST`
- `NESY_HTTP_PORT`
- `NESY_HTTP_PATH`
- `NESY_HTTP_ALLOWED_ORIGINS`
- `NESY_HTTP_ALLOWED_HOSTS`
- `NESY_HTTP_MAX_BODY_BYTES`
- `NESY_HTTP_REQUEST_TIMEOUT_SECONDS`
- `NESY_HTTP_RATE_LIMIT_PER_MINUTE`

Health check:

```bash
curl -H 'Authorization: Bearer change-me' http://127.0.0.1:8765/healthz
```

## Claude Code Hook Helpers

Hook helpers run as separate processes. Use SQLite or JSON storage so hooks see
the same graph as the MCP server.

Example hook config: [examples/claude-hooks.json](../examples/claude-hooks.json)

Commands:

```bash
uv run nesy-reasoning-mcp hook pretooluse
uv run nesy-reasoning-mcp hook stop
```

Stop hook checks `last_assistant_message`. If the answer contains a `NESY_FACTS:`
JSON array, the hook checks those facts with the current graph. Without
`NESY_FACTS`, it checks the current graph only.

```text
NESY_FACTS:
[
  {"source":"降价","target":"销量增加","relation_type":"sufficient"}
]
```

Default hook behavior is fail-open with a stderr warning. Set
`NESY_HOOK_FAIL_CLOSED=true` for projects that should block on hook failures.

## Current Limits

- Default state is in memory only.
- SQLite and JSON backends are local-only.
- Contradiction detection only uses explicit exclusive groups.
- Formal independence is stored through relation-set import/export, not a
  dedicated public tool.
- Counterfactual reasoning is conservative: open-world mode does not infer
  negation from missing facts; closed-world upgrades require
  `context_metadata.<context_id>.causal_completeness=true`.
- MCP `resource_uri` loading is limited to allowed local `file://` URIs.
- HTTP daemon auth is a local bearer token, not multi-user auth.
- No regex or LLM natural-language extraction in hooks yet.

## Offline Evaluation

v0.8 includes deterministic benchmark fixtures:

```bash
env PYTHONPATH=src uv run nesy-reasoning-mcp eval run --fixture benchmarks/fixtures/core.json
```

The default evaluator does not call a real LLM and does not require API keys.

## Troubleshooting

If CLI import fails with `ModuleNotFoundError: No module named 'nesy_reasoning_mcp'`,
run:

```bash
cd /Users/mac-mini/Documents/nesy-reasoning-mcp
chflags -R nohidden .venv
uv sync --reinstall-package nesy-reasoning-mcp
```

The MCP config includes `PYTHONPATH=.../src` to avoid this editable-install issue on
macOS environments where `.pth` files have the hidden flag.
