# Security

This server is designed for local reasoning assistance. It is not a hosted
multi-user service.

## Tool Risk

Low-risk read-only tools:

- `nesy.classify`
- `nesy.verify_chain`
- `nesy.check_contradictions`
- `nesy.counterfactual`

Medium-risk tools may reveal project knowledge:

- `nesy.list_relations`
- `nesy.summarize_graph`

Write or file tools should require client-side confirmation in sensitive
projects:

- `nesy.assert_relations`
- `nesy.assert_exclusive`
- `nesy.load_relations`
- `nesy.export_relations`
- `nesy.clear_relations`

The MCP server records write-tool audit events, but it does not implement a
separate interactive confirmation prompt. Confirmation belongs in the MCP client,
hook policy, or deployment wrapper.

## File Access

`load_relations` and `export_relations` only allow `.json` and `.jsonl` under
configured `allowed_roots`.

The implementation resolves real paths before access, rejects path traversal and
symlink escapes, enforces file-size limits, and limits `resource_uri` support to
local `file://` URIs inside `allowed_roots`.

Hidden relation paths under an allowed root are blocked by default. Set
`security.allow_hidden_relation_paths=true` or `NESY_ALLOW_HIDDEN_RELATION_PATHS=true`
only when hidden `.json` or `.jsonl` relation files are intentional.

## HTTP Mode

Streamable HTTP mode defaults to `127.0.0.1` and requires `NESY_LOCAL_TOKEN`.

HTTP guards enforce:

- bearer token auth
- host and origin allowlists
- request body size limit
- request timeout
- per-client rate limit

Do not expose the local daemon to an untrusted network without a stronger
fronting auth layer.

## Hooks

Claude Code hooks are fail-open by default. Set `NESY_HOOK_FAIL_CLOSED=true` only
when blocking on hook failure is acceptable for the project.

Hooks must share state through SQLite, JSON storage, or HTTP daemon mode. Process
memory is not shared across hook and MCP processes.

## Audit

Audit entries are stored for mutating tools when `logging.audit_log=true`.
Entries include tool name, input hash, result status, timestamp, and metadata.
They intentionally do not store plaintext secrets.
