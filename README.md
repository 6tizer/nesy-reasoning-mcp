# NeSy Reasoning MCP

Deterministic neuro-symbolic reasoning MCP server.

v0.5 provides a local MCP stdio server with memory, JSON, and SQLite storage,
plus Claude Code hook helpers:

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

## Run

```bash
uv run nesy-reasoning-mcp --transport stdio
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
cannot be shared between stdio MCP and hook processes.

## Install As MCP Server

See [docs/install.md](docs/install.md) for Claude Desktop, Codex, Cursor, or any MCP client
that supports stdio servers.

Example configs:

- [examples/mcp-config.json](examples/mcp-config.json)
- [examples/claude-hooks.json](examples/claude-hooks.json)
- [examples/nesy-config.json](examples/nesy-config.json)
