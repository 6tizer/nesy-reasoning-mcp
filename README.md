# NeSy Reasoning MCP

Deterministic neuro-symbolic reasoning MCP server.

v0.9 provides a local MCP server with stdio and authenticated Streamable HTTP
transports, memory/JSON/SQLite storage, and Claude Code hook helpers:

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

## Development

```bash
uv sync
uv run ruff format --check .
uv run pytest
uv run ruff check .
```

For version planning and contribution gates, see:

- [docs/roadmap.md](docs/roadmap.md)
- [docs/development.md](docs/development.md)
- [docs/evaluation.md](docs/evaluation.md)
- [docs/security.md](docs/security.md)

## Run

```bash
uv run nesy-reasoning-mcp --transport stdio
```

Authenticated local HTTP daemon:

```bash
NESY_LOCAL_TOKEN='change-me' uv run nesy-reasoning-mcp --transport http
```

Optional persistent storage:

```bash
NESY_STORAGE_BACKEND=sqlite NESY_SQLITE_PATH=~/.nesy-reasoning/nesy.db \
  uv run nesy-reasoning-mcp --transport stdio
```

Hook helpers:

```bash
uv run nesy-reasoning-mcp hook pretooluse
uv run nesy-reasoning-mcp hook stop
```

Hook use should share SQLite or JSON storage with the MCP server. Process memory
cannot be shared between stdio MCP and hook processes. HTTP daemon mode can also
keep one long-running in-process store for multiple MCP clients.

Relation sets can include explicit `independence_records`; `nesy.classify` uses
them to prove `proven_not_necessary`, and `nesy.counterfactual` uses them to keep
independent alternatives in `still_possible`. `nesy.load_relations` also accepts
safe `file://` resource URIs inside configured `allowed_roots`.

Offline benchmark evaluation:

```bash
env PYTHONPATH=src uv run nesy-reasoning-mcp eval run --fixture benchmarks/fixtures/core.json
```

## Install As MCP Server

See [docs/install.md](docs/install.md) for Claude Desktop, Codex, Cursor, or any MCP client
that supports stdio or Streamable HTTP servers.

Example configs:

- [examples/mcp-config.json](examples/mcp-config.json)
- [examples/claude-hooks.json](examples/claude-hooks.json)
- [examples/nesy-config.json](examples/nesy-config.json)
