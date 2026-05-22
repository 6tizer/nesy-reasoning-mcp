# Internal Test Tool Policy

This policy is for MCP clients, wrappers, and reviewers during internal testing.
It is not enforced inside the MCP server.

## Allowed By Default

These tools are read-only reasoning checks:

- `nesy.classify`
- `nesy.verify_chain`
- `nesy.check_contradictions`
- `nesy.counterfactual`

## Controlled Read

These tools can reveal project knowledge. Use them during internal tests, but do
not paste sensitive graph summaries into unrelated contexts.

- `nesy.list_relations`
- `nesy.summarize_graph`

## Confirmation Required

Require explicit user confirmation before:

- `nesy.assert_exclusive`
- bulk `nesy.assert_relations`
- `nesy.load_relations` with `source_type=file`
- `nesy.load_relations` with `source_type=resource_uri`
- `nesy.export_relations` with `destination=file`
- any `nesy.clear_relations`

## Never Without Confirmation

Do not run these silently:

- `nesy.clear_relations` with `scope=all`
- `nesy.load_relations` with `mode=replace_store`
- changing `allowed_roots`
- enabling hidden relation paths
- changing the shared SQLite DB path

## Review Expectation

When a write operation changes long-lived graph state, record why it was needed
in the task notes or PR summary. Audit log entries contain input hashes, not the
full sensitive input.
