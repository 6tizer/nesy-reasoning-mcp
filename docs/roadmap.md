# Roadmap

This roadmap turns the SPEC into small, reviewable versions. It is a planning
document, not a promise that all items already exist.

## Current Baseline

v0.4 is the current implementation baseline:

- MCP stdio server.
- Memory, JSON, and SQLite stores.
- Relation assert/list/clear.
- Classification and chain verification.
- Explicit exclusive groups.
- Explicit exclusivity-based contradiction checks.
- Relation load/export with allowed roots.
- Write-operation audit log.
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
- Hook output stays small and deterministic.

Out of scope:

- LLM-based extraction from assistant text.
- HTTP daemon.

## v0.6: Local HTTP Daemon

Goal: allow MCP clients and hooks to share one long-running service.

Tracking issue: https://github.com/6tizer/nesy-reasoning-mcp/issues/1

Scope:

- Add streamable HTTP transport.
- Add local token auth.
- Add config file support through `NESY_CONFIG`.
- Add health endpoint.
- Keep stdio transport working.

Acceptance:

- stdio and HTTP expose same tool behavior.
- Local token required for HTTP.
- No stdout logging in stdio mode.
- Hooks can call shared daemon without owning store lifecycle.

Out of scope:

- Multi-user auth.
- Hosted service.

## v0.7: Counterfactual Reasoning

Goal: implement SPEC counterfactual queries with explicit world assumptions.

Tracking issue: https://github.com/6tizer/nesy-reasoning-mcp/issues/3

Scope:

- Add counterfactual input schemas.
- Support open-world and closed-world modes.
- Support explicit intervention facts.
- Return trace, assumptions, and unknown reasons.

Public tools:

- `nesy.counterfactual`

Acceptance:

- Open-world mode does not infer negation from missing facts.
- Closed-world mode requires explicit completeness boundary.
- Cycles and conflicting interventions produce diagnostics.

## Later

- Natural-language relation extraction.
- Evaluation fixtures and benchmark suite.
- Optional Postgres backend.
- Team/server deployment model.
- Richer contradiction classes beyond explicit exclusives.
