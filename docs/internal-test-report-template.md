# Internal Test Report Template

Use this template for each internal test run. Do not paste secrets, bearer
tokens, full prompts, or private relation payloads into the report.

## Summary

- Date:
- Tester:
- Repo commit:
- Package version:
- Workspace/project:
- Verdict: pass / fail / blocked

## Environment

- Python version:
- OS:
- MCP transport: stdio / HTTP
- Storage backend: sqlite / json
- SQLite path or JSON path:
- Hook mode: disabled / fail-open / fail-closed
- HTTP bind/token policy: local bearer token only

## Setup Checks

| Check | Result | Notes |
|---|---|---|
| `uv sync --locked` |  |  |
| `uv run ruff format --check .` |  |  |
| `uv run ruff check .` |  |  |
| `uv run mypy src/nesy_reasoning_mcp` |  |  |
| `uv run pytest` |  |  |
| stdio EOF smoke |  |  |
| MCP client smoke, 11 tools |  |  |
| HTTP health smoke |  |  |
| internal-test smoke |  |  |

## Evaluation

| Command | Result | Key output |
|---|---|---|
| `nesy-reasoning-mcp eval run --fixture benchmarks/fixtures/core.json --format json` |  |  |
| `nesy-reasoning-mcp eval agent --fixture benchmarks/fixtures/core.json --format json` |  |  |
| Optional `eval llm` / `eval agent --runner openai` |  | Manual only |

Record the Agent eval matrix:

| Mode | Score | Error notes |
|---|---:|---|
| `no_mcp` |  |  |
| `tool_descriptions_only` |  |  |
| `classify_only` |  |  |
| `classify_verify` |  |  |
| `full_mcp` |  |  |

## Agent Behavior Checks

| Scenario | Result | Notes |
|---|---|---|
| Agent writes stable facts with `nesy.assert_relations` |  |  |
| Final answer uses valid `NESY_FACTS:` when needed |  |  |
| Stop hook blocks hard contradiction |  |  |
| PreToolUse injects relevant graph summary |  |  |
| Destructive/file tools require confirmation |  |  |
| Audit query identifies mutating tool, status, timestamp |  |  |

## Known Boundaries Observed

- Plain natural language is not automatically persisted.
- Stop hook checks explicit graph facts and `NESY_FACTS`, not arbitrary prose.
- PostToolBatch hook is not part of v1.0 internal testing.
- HTTP mode is a local daemon, not hosted multi-user auth.

## Follow-Ups

| Priority | Finding | Owner | Link |
|---|---|---|---|
| P0/P1/P2 |  |  |  |
