# Install NeSy Reasoning MCP

This project runs as a local MCP server over stdio or authenticated Streamable HTTP.

For internal testing, prefer the SQLite profile in
[Internal Testing](internal-testing.md). It packages MCP config, hook config,
HTTP launch, and a smoke test that share one SQLite store.

## Requirements

- Python 3.11 to 3.14
- `uv`
- Local checkout at `/path/to/nesy-reasoning-mcp`

## Verify Locally

```bash
cd /path/to/nesy-reasoning-mcp
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
        "/path/to/nesy-reasoning-mcp",
        "run",
        "nesy-reasoning-mcp",
        "--transport",
        "stdio"
      ],
      "env": {
        "PYTHONPATH": "/path/to/nesy-reasoning-mcp/src"
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

Internal testing should use SQLite, not memory, so MCP and hooks share durable
state across processes and restarts.

SQLite:

```json
{
  "env": {
    "PYTHONPATH": "/path/to/nesy-reasoning-mcp/src",
    "NESY_STORAGE_BACKEND": "sqlite",
    "NESY_SQLITE_PATH": "~/.nesy-reasoning/nesy.db",
    "NESY_ALLOWED_ROOTS": "/path/to/nesy-reasoning-mcp,~/.nesy-reasoning/relation_sets"
  }
}
```

JSON file:

```json
{
  "env": {
    "PYTHONPATH": "/path/to/nesy-reasoning-mcp/src",
    "NESY_CONFIG": "~/.nesy-reasoning/config.json"
  }
}
```

Example config file: [examples/nesy-config.json](../examples/nesy-config.json)

```json
{
  "storage": {
    "backend": "json",
    "json_path": "~/.nesy-reasoning/relations.json"
  },
  "security": {
    "allowed_roots": [
      "/path/to/nesy-reasoning-mcp",
      "~/.nesy-reasoning/relation_sets"
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

`nesy.load_relations` can import legacy relation field names at the load boundary:
`from`, `to`, `type`, and `temporal_delay`. Stored records and exports always use
canonical fields: `source`, `target`, `relation_type`, and `temporal.delay`.
Relation records may also include optional `source_id` and `target_id` stable
proposition IDs. When present, reasoning uses those IDs as canonical graph nodes;
`source` and `target` remain display labels. `nesy.check_contradictions` can
accept temporary `propositions` entries with `negates` for canonical ID-based
opposition checks. Alias lookup and persistence/export of proposition metadata
remain future work.

## Streamable HTTP Mode

HTTP mode starts a local daemon at `127.0.0.1:8765/mcp` by default. It requires
`NESY_LOCAL_TOKEN`; send it as a bearer token.

```bash
cd /path/to/nesy-reasoning-mcp
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

## Claude Code Setup

Claude Code setup has two layers.

Step 1: add the MCP server with the config from
[examples/mcp-config.json](../examples/mcp-config.json), or adapt the JSON in
the MCP client config section above.

Step 2: optionally add hooks with
[examples/claude-hooks.json](../examples/claude-hooks.json). Hook helpers run as
separate processes. Use SQLite, JSON, or the local HTTP daemon so hooks see the
same graph as the MCP server.

Hook commands:

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
PreToolUse focus terms can be tuned with `NESY_HOOK_FOCUS_TERM_SOURCES` and
`NESY_HOOK_FOCUS_TERMS`; use `NESY_HOOK_CONTEXT_ID` or `NESY_HOOK_DOMAIN` when
the graph should be scoped manually.

Step 3: run the internal-test smoke:

```bash
env PYTHONPATH=src uv run python examples/internal-test/smoke.py
```

Expected:

```text
internal-test smoke ok
```

## Audit CLI

Inspect write-tool audit entries from the configured store:

```bash
NESY_CONFIG=/path/to/nesy-config.json uv run nesy-reasoning-mcp audit list --format json
```

The audit CLI reports tool name, input hash, status, timestamp, and metadata. It
does not print raw tool arguments.

## Current Limits

- Default state is in memory only.
- SQLite and JSON backends are local-only.
- Contradiction detection is deterministic over structured propositions and
  explicit constraints; it does not extract arbitrary natural-language
  contradictions.
- Canonical `propositions[].negates` declarations are accepted only by
  `nesy.check_contradictions`; they are not stored, loaded, or exported yet.
- Formal independence is stored through relation-set import/export, not a
  dedicated public tool.
- Counterfactual reasoning is conservative: open-world mode does not infer
  negation from missing facts; closed-world upgrades require
  `context_metadata.<context_id>.causal_completeness=true`.
- MCP `resource_uri` loading is limited to allowed local `file://` URIs.
- HTTP daemon auth is a local bearer token, not multi-user auth.
- No regex or LLM natural-language extraction in hooks yet.

## Evaluation

v1.0 includes deterministic benchmark fixtures:

```bash
env PYTHONPATH=src uv run nesy-reasoning-mcp eval run --fixture benchmarks/fixtures/core.json
```

The default evaluator does not call a real LLM and does not require API keys.
Optional live OpenAI baseline evaluation is manual-only:

```bash
uv sync --extra eval
export OPENAI_API_KEY='<set outside the repo>'
env PYTHONPATH=src uv run --extra eval nesy-reasoning-mcp eval llm \
  --fixture benchmarks/fixtures/core.json \
  --case-id classify_direct_sufficient
```

## Troubleshooting

If CLI import fails with `ModuleNotFoundError: No module named 'nesy_reasoning_mcp'`,
run:

```bash
cd /path/to/nesy-reasoning-mcp
chflags -R nohidden .venv
uv sync --reinstall-package nesy-reasoning-mcp
```

The MCP config includes `PYTHONPATH=.../src` to avoid this editable-install issue on
macOS environments where `.pth` files have the hidden flag.

Need to roll back? See [Uninstall / rollback](uninstall.md).
