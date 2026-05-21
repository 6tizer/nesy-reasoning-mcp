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

## Current Limits

- State is in memory only.
- Restarting the MCP server clears all relations.
- Contradiction detection only uses explicit exclusive groups.
- No counterfactual reasoning yet.
- No SQLite or file import/export yet.

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
