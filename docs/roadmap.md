# Roadmap

This roadmap turns the SPEC into small, reviewable versions. It is a planning
document, not a promise that all items already exist.

## Current Baseline

v0.7 is the current implementation baseline:

- MCP stdio server.
- Authenticated local Streamable HTTP daemon.
- Memory, JSON, and SQLite stores.
- Relation assert/list/clear.
- Classification and chain verification.
- Counterfactual reasoning with open-world and guarded closed-world modes.
- Formal independence records through relation-set load/export.
- `classify` can return `necessity_status.status=proven_not_necessary` when an
  independent counterexample is established.
- `counterfactual` can use formal independence records for `still_possible`.
- Explicit exclusive groups.
- Explicit exclusivity-based contradiction checks.
- Relation load/export with allowed roots and safe local `file://` resource URIs.
- Write-operation audit log.
- Compact graph summary.
- Claude Code Stop and PreToolUse hook helpers.
- Local install docs and CI smoke coverage.

## Version Gates

Each version must pass these gates before merge:

- `uv sync --locked`
- `uv run ruff format --check .`
- `uv run ruff check .`
- `uv run pytest`
- MCP stdio smoke.
- MCP client smoke for all public tools.
- Docs updated for any public tool, config, or behavior change.

## v0.4: Persistence And Import/Export

Goal: make the relation graph survive MCP server restarts.

Tracking issue: https://github.com/6tizer/nesy-reasoning-mcp/issues/5

Scope:

- Add storage backend config: `memory`, `json`, `sqlite`.
- Add SQLite store with migrations.
- Add JSON/JSONL export and load.
- Add allowed roots for file read/write.
- Add atomic import semantics.
- Preserve v0.3 in-memory behavior as default.

Public tools:

- `nesy.load_relations`
- `nesy.export_relations`

Acceptance:

- Restart with SQLite keeps relations and exclusive groups.
- Export output can be loaded into a clean store.
- File access outside allowed roots fails.
- Partial failed load does not corrupt existing store.

Out of scope:

- HTTP daemon.
- Hook integration.
- Natural-language relation extraction.

## v0.5: Graph Summary And Hook Bridge

Goal: prepare safe integration with Claude Code hooks.

Tracking issue: https://github.com/6tizer/nesy-reasoning-mcp/issues/4

Scope:

- Add compact graph summary API.
- Add hook-facing command helpers.
- Add Stop hook sample that checks existing graph contradictions.
- Add PreToolUse hook sample that injects graph summary.
- Add recursion guard for `stop_hook_active`.

Public tools:

- `nesy.summarize_graph`

Acceptance:

- Hook examples use SQLite or file-backed store, not process memory.
- Stop hook can block on hard contradictions from current graph.
- Stop hook can check explicit `NESY_FACTS` blocks with the current graph.
- Hook output stays small and deterministic.

Out of scope:

- LLM-based extraction from assistant text.
- HTTP daemon.

## v0.6: Local HTTP Daemon And Counterfactual

Goal: allow MCP clients to share one long-running service and add conservative
counterfactual reasoning.

Tracking issue: https://github.com/6tizer/nesy-reasoning-mcp/issues/1

Scope:

- Add streamable HTTP transport.
- Add local token auth.
- Add config file support through `NESY_CONFIG`.
- Add health endpoint.
- Add `nesy.counterfactual`.
- Support open-world and closed-world modes.
- Keep stdio transport working.

Public tools:

- `nesy.counterfactual`

Acceptance:

- stdio and HTTP expose same tool behavior.
- Local token required for HTTP.
- Open-world counterfactual mode does not infer negation from missing facts.
- Closed-world counterfactual upgrades require explicit completeness metadata.
- No stdout logging in stdio mode.

Out of scope:

- Multi-user auth.
- Hosted service.
- LLM-based extraction from assistant text.

## v0.7: Independence Records And Resource URI Load

Goal: close the main SPEC gaps left after HTTP and counterfactual.

Tracking issue: https://github.com/6tizer/nesy-reasoning-mcp/issues/3

Scope:

- Add `IndependenceRecord` to relation-set schemas and stores.
- Persist independence records in memory, JSON, and SQLite backends.
- Include independence records in JSON/JSONL load/export.
- Use independence records to prove `proven_not_necessary` in `nesy.classify`.
- Use independence records as formal alternative-path proof in
  `nesy.counterfactual`.
- Support `nesy.load_relations(source_type=resource_uri)` for safe local
  `file://` URIs inside `allowed_roots`.

Public tools:

- No new tool. Existing `nesy.load_relations`, `nesy.export_relations`,
  `nesy.classify`, and `nesy.counterfactual` gain behavior.

Acceptance:

- Alternative sufficient causes still do not prove non-necessity by themselves.
- Independent counterexamples do prove `proven_not_necessary`.
- Counterfactual alternative paths can be `still_possible` through formal
  independence records.
- File resource URI loads obey allowed roots and reject remote schemes.

Out of scope:

- Dedicated independence assertion tool.
- Remote/client MCP resource fetching.
- Richer contradiction classes beyond explicit exclusives.
- Evaluation fixtures and benchmark suite.

## Later

- Natural-language relation extraction.
- Evaluation fixtures and benchmark suite.
- Optional Postgres backend.
- Team/server deployment model.
- Richer contradiction classes beyond explicit exclusives.
