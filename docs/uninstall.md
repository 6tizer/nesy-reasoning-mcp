# Uninstall / Rollback

NeSy Reasoning MCP is local-first. Rolling back means removing the MCP client
entry, optional hooks, optional local daemon, and any local relation store you no
longer want to keep.

## Safe Order

1. Disable hooks.
2. Disable the MCP server entry.
3. Stop the HTTP daemon, if running.
4. Back up or delete local stores.
5. Remove the local checkout, if no longer needed.

This order avoids hooks calling a server that has already been removed.

## Disable The MCP Server

Remove the `nesy-reasoning` entry from your MCP client config:

```json
{
  "mcpServers": {
    "nesy-reasoning": {
      "...": "remove this block"
    }
  }
}
```

Restart or reload the MCP client. The `nesy.*` tools should disappear from the
client tool list.

## Disable Claude Code Hooks

If you installed hooks, remove the hook commands that call:

```bash
nesy-reasoning-mcp hook pretooluse
nesy-reasoning-mcp hook stop
```

or remove the hook template entries from your Claude Code hooks config:

```json
{
  "hooks": {
    "PreToolUse": [],
    "Stop": []
  }
}
```

Then restart or reload Claude Code.

## Stop The HTTP Daemon

If you started Streamable HTTP mode, stop the process:

```bash
pkill -f "nesy-reasoning-mcp --transport http"
```

Check whether anything is still running:

```bash
ps aux | grep nesy-reasoning-mcp
```

## Back Up Or Delete Local Stores

If you used SQLite and want a backup:

```bash
mkdir -p ~/.nesy-reasoning
cp ~/.nesy-reasoning/nesy.db \
  ~/.nesy-reasoning/nesy.backup.$(date +%Y%m%d-%H%M%S).db
```

Delete common SQLite stores:

```bash
rm -f ~/.nesy-reasoning/nesy.db
rm -f ~/.nesy-reasoning/internal-test/nesy.db
```

Delete a common JSON store:

```bash
rm -f ~/.nesy-reasoning/relations.json
```

If your config uses a different `NESY_SQLITE_PATH`, `json_path`, or `NESY_CONFIG`,
delete the path you configured instead.

## Remove The Local Checkout

If you installed from a local clone and no longer need it:

```bash
rm -rf /path/to/nesy-reasoning-mcp
```

## Verify Rollback

After rollback:

- Your MCP client no longer lists `nesy.*` tools.
- Claude Code no longer runs `nesy-reasoning-mcp hook ...` commands.
- `ps aux | grep nesy-reasoning-mcp` shows no running daemon, except the `grep`
  command itself.
- New assistant turns no longer receive NeSy graph summary context.

## Keep The Data But Disable The Tool

If you only want to stop using the MCP temporarily, remove the MCP and hook
configs but keep `~/.nesy-reasoning/`. You can re-enable the server later with
the same database.
