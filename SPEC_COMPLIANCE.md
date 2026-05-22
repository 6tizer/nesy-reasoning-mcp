# SPEC Compliance

Status for [`docs/spec-v2.md`](docs/spec-v2.md) at v1.0.

## Core Phases

| Area | Status | Notes |
|---|---|---|
| Phase 1: MCP skeleton and data model | Complete | stdio server, tool metadata, mirrored `content` and `structuredContent`, Pydantic schemas, memory store, canonical implication edges, assert/list/clear. |
| Phase 2: classify and verify_chain | Complete | reachability, path search, cycle protection, relation classification, confidence aggregation. |
| Phase 3: exclusives and contradictions | Complete | explicit exclusives, graph/facts/combined contradiction checks, context-separated conflicts. |
| Phase 4: counterfactual | Complete | open-world defaults, guarded closed-world upgrades, alternative path handling, independence-aware `still_possible`. |
| Phase 5: persistence and import/export | Complete | memory/JSON/SQLite stores, load/export, allowed roots, audit log. |
| Phase 6: graph summary and hooks | Complete | `nesy.summarize_graph`, Stop hook, PreToolUse hook, shared SQLite/JSON/HTTP state options, timeout and fallback. |
| Phase 7: evaluation and ablation | Complete | offline fixture, deterministic MCP score, Agent tool-access matrix, static ablations, optional live OpenAI LLM/Agent runs, failure-to-regression workflow. |

## Security

| Area | Status | Notes |
|---|---|---|
| Input validation | Complete | unknown fields rejected except metadata-style payloads, proposition limits, max relation limits, `max_depth` limits, import size limit. |
| File access | Complete | allowed roots, realpath checks, path traversal rejection, symlink escape rejection, extension limit, hidden path blocking with explicit opt-in. |
| HTTP security | Complete | local bind default, bearer token, origin/host allowlists, body size limit, timeout, rate limit. |
| Tool risk documentation | Complete | documented in `docs/security.md`; confirmation remains a client/wrapper responsibility. |
| Audit log | Complete | mutating tools record audit entries when audit logging is enabled. |

## Compatibility And Boundaries

| Item | Status | Notes |
|---|---|---|
| New `nesy.*` tool names | Complete | `tools/list` exposes only current names. |
| Old alias tools | Intentionally out of scope | SPEC allows a temporary alias period but recommends exposing only new names. |
| Natural-language extraction | Intentionally out of scope | SPEC says the deterministic engine handles structured propositions, not reliable NL extraction. |
| PostToolBatch hook | Future optional | SPEC marks it optional; v1.0 ships Stop and PreToolUse helpers. |
| Remote/client MCP resource fetching | Future optional | v1.0 supports safe local `file://` resource URI loads only. |
| Hosted multi-user auth | Future optional | v1.0 HTTP is a local daemon with bearer token and local security guards. |
| Postgres/team deployment | Future optional | SQLite is the durable local backend; Postgres remains future work. |

## Release Gate

v1.0 release candidates must pass:

```bash
uv sync --locked
uv run ruff format --check .
uv run ruff check .
uv run mypy src/nesy_reasoning_mcp
uv run pytest
env PYTHONPATH=src uv run nesy-reasoning-mcp eval run --fixture benchmarks/fixtures/core.json --format json
env PYTHONPATH=src uv run nesy-reasoning-mcp eval agent --fixture benchmarks/fixtures/core.json --format json
```

CI must remain deterministic and must not require `OPENAI_API_KEY`. Live LLM
and live Agent evaluation are manual and opt-in.
