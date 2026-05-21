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
uv run pytest
uv run ruff check .
```

## Run

```bash
uv run nesy-reasoning-mcp --transport stdio
```

## Install As MCP Server

See [docs/install.md](docs/install.md) for Claude Desktop, Codex, Cursor, or any MCP client
that supports stdio servers.

Example config: [examples/mcp-config.json](examples/mcp-config.json).
