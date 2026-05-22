# NeSy Reasoning MCP

Deterministic neuro-symbolic reasoning MCP server for structured causal,
dependency, contradiction, and counterfactual checks.

v1.0 provides:

- MCP stdio and authenticated local Streamable HTTP transports.
- Memory, JSON, and SQLite relation stores.
- Relation assertion, listing, clearing, import, and export.
- Classification, chain verification, contradiction checks, graph summaries, and
  counterfactual analysis.
- Claude Code Stop and PreToolUse hook helpers.
- Offline benchmark evaluation and optional live OpenAI LLM baseline evaluation.
- Security docs, audit logging, and SPEC compliance tracking.

## Tools

- `nesy.assert_relations`
- `nesy.list_relations`
- `nesy.clear_relations`
- `nesy.classify`
- `nesy.verify_chain`
- `nesy.assert_exclusive`
- `nesy.check_contradictions`
- `nesy.load_relations`
- `nesy.export_relations`
- `nesy.summarize_graph`
- `nesy.counterfactual`

## Install

```bash
uv sync
```

For optional live OpenAI baseline evaluation:

```bash
uv sync --extra eval
```

## Quick Start

Run as a stdio MCP server:

```bash
uv run nesy-reasoning-mcp --transport stdio
```

Run as an authenticated local HTTP daemon:

```bash
NESY_LOCAL_TOKEN='change-me' uv run nesy-reasoning-mcp --transport http
```

Use persistent SQLite storage:

```bash
NESY_STORAGE_BACKEND=sqlite NESY_SQLITE_PATH=~/.nesy-reasoning/nesy.db \
  uv run nesy-reasoning-mcp --transport stdio
```

Run hook helpers:

```bash
uv run nesy-reasoning-mcp hook pretooluse
uv run nesy-reasoning-mcp hook stop
```

Hook use should share SQLite, JSON storage, or HTTP daemon state with the MCP
server. Process memory cannot be shared between stdio MCP and hook processes.

## Relation Sets And Security

Relation sets can include explicit `independence_records`; `nesy.classify` uses
them to prove `proven_not_necessary`, and `nesy.counterfactual` uses them to keep
independent alternatives in `still_possible`.

`nesy.load_relations` and `nesy.export_relations` accept `.json` and `.jsonl`
inside configured `allowed_roots`. Local `file://` resource URI loads are also
restricted to `allowed_roots`. Hidden relation paths are blocked by default unless
`security.allow_hidden_relation_paths=true` or
`NESY_ALLOW_HIDDEN_RELATION_PATHS=true` is set.

HTTP mode binds locally by default and requires `NESY_LOCAL_TOKEN`. File tools and
state-mutating tools should still be treated as user-confirmation operations in
MCP clients or wrappers.

## Evaluation

Deterministic offline benchmark:

```bash
env PYTHONPATH=src uv run nesy-reasoning-mcp eval run --fixture benchmarks/fixtures/core.json
```

Optional live OpenAI LLM-only baseline:

```bash
export OPENAI_API_KEY='<set outside the repo>'
env PYTHONPATH=src uv run --extra eval nesy-reasoning-mcp eval llm \
  --fixture benchmarks/fixtures/core.json \
  --case-id classify_direct_sufficient \
  --format json
```

Live eval is manual-only. CI runs offline fixtures and never requires an API key.

## Development

```bash
uv sync --locked
uv run ruff format --check .
uv run ruff check .
uv run pytest
env PYTHONPATH=src uv run nesy-reasoning-mcp eval run --fixture benchmarks/fixtures/core.json --format json
```

## Documentation

- [SPEC compliance](SPEC_COMPLIANCE.md)
- [Roadmap](docs/roadmap.md)
- [Development](docs/development.md)
- [Evaluation](docs/evaluation.md)
- [Security](docs/security.md)
- [Install as MCP server](docs/install.md)

Example configs:

- [examples/mcp-config.json](examples/mcp-config.json)
- [examples/claude-hooks.json](examples/claude-hooks.json)
- [examples/nesy-config.json](examples/nesy-config.json)
