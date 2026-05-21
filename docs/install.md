# Install NeSy Reasoning MCP

This project runs as a local stdio MCP server.

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
    "max_file_size_bytes": 5242880
  }
}
```

File load/export only accepts `.json` and `.jsonl` inside `allowed_roots`.

## Current Limits

- Default state is in memory only.
- SQLite and JSON backends are local-only.
- Contradiction detection only uses explicit exclusive groups.
- No counterfactual reasoning yet.
- No HTTP daemon or hook bridge yet.

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
