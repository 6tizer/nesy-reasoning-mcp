# NeSy Reasoning MCP

Deterministic neuro-symbolic reasoning MCP server.

v0.3 provides an in-memory MCP stdio server with:

- `nesy.assert_relations`
- `nesy.list_relations`
- `nesy.clear_relations`
- `nesy.classify`
- `nesy.verify_chain`
- `nesy.assert_exclusive`
- `nesy.check_contradictions`

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

## Install As MCP Server

See [docs/install.md](docs/install.md) for Claude Desktop, Codex, Cursor, or any MCP client
that supports stdio servers.

Example config: [examples/mcp-config.json](examples/mcp-config.json).
