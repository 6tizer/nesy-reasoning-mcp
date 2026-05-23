# NeSy Reasoning MCP

[![CI](https://github.com/6tizer/nesy-reasoning-mcp/actions/workflows/ci.yml/badge.svg)](https://github.com/6tizer/nesy-reasoning-mcp/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.11--3.14-blue)
![MCP](https://img.shields.io/badge/MCP-stdio%20%7C%20Streamable%20HTTP-green)
![Version](https://img.shields.io/badge/version-1.0.0-informational)

English | [简体中文](README.zh-CN.md)

A local MCP server that gives AI agents deterministic reasoning memory:
structured relation storage, causal classification, chain verification,
contradiction checks, graph summaries, and counterfactual analysis.

It does not try to replace the LLM. The LLM can propose structured facts; this
server checks them with a small, testable symbolic engine.

## What It Gives An Agent

- **Long-lived reasoning memory**: keep structured relations in SQLite or JSON
  instead of losing them when a chat or MCP process restarts.
- **Deterministic logic checks**: classify whether `A` is sufficient,
  necessary, equivalent, contradictory, or unknown relative to `B`.
- **Verifiable chains**: prove or reject multi-hop implication paths such as
  `A -> B -> C`.
- **Contradiction guardrails**: detect explicit exclusives, direct opposition,
  cycles to negation, and soft confidence tension.
- **Counterfactual analysis**: ask what remains possible when a proposition is
  assumed false under open-world or guarded closed-world semantics.
- **Hook integration**: inject compact graph summaries before tools run and
  block final answers that include hard contradictions in explicit `NESY_FACTS`.

## Quick Start

Install from a local checkout:

```bash
uv sync
```

Run the stdio MCP server:

```bash
uv run nesy-reasoning-mcp --transport stdio
```

Use persistent SQLite storage:

```bash
mkdir -p ~/.nesy-reasoning
NESY_STORAGE_BACKEND=sqlite NESY_SQLITE_PATH=~/.nesy-reasoning/nesy.db \
  uv run nesy-reasoning-mcp --transport stdio
```

Run the authenticated local Streamable HTTP daemon:

```bash
NESY_LOCAL_TOKEN='change-me' uv run nesy-reasoning-mcp --transport http
```

Verify the deterministic benchmark:

```bash
env PYTHONPATH=src uv run nesy-reasoning-mcp eval run \
  --fixture benchmarks/fixtures/core.json \
  --format json
```

## MCP Client Config

Use this stdio config as a starting point:

```json
{
  "mcpServers": {
    "nesy-reasoning": {
      "command": "uv",
      "args": [
        "--directory",
        "/path/to/nesy-reasoning-mcp",
        "run",
        "nesy-reasoning-mcp",
        "--transport",
        "stdio"
      ],
      "env": {
        "PYTHONPATH": "/path/to/nesy-reasoning-mcp/src",
        "NESY_STORAGE_BACKEND": "sqlite",
        "NESY_SQLITE_PATH": "~/.nesy-reasoning/nesy.db"
      }
    }
  }
}
```

More examples:

- [examples/mcp-config.json](examples/mcp-config.json)
- [examples/nesy-config.json](examples/nesy-config.json)
- [examples/claude-hooks.json](examples/claude-hooks.json)
- [examples/internal-test](examples/internal-test/README.md)

## Claude Code Setup

Claude Code integration has two separate parts:

1. Add the MCP server with a stdio config such as
   [examples/mcp-config.json](examples/mcp-config.json).
2. Optionally add hooks with [examples/claude-hooks.json](examples/claude-hooks.json)
   or the internal-test wrappers in [examples/internal-test](examples/internal-test/README.md).

For hooks, use SQLite, JSON, or the local HTTP daemon so the hook process and MCP
server see the same graph. In-process memory is not shared across processes.

Run the internal-test smoke after configuring Claude Code:

```bash
env PYTHONPATH=src uv run python examples/internal-test/smoke.py
```

Expected output:

```text
internal-test smoke ok
```

## Minimal Reasoning Example

Assert two sufficient relations:

```json
{
  "relations": [
    {"source": "A", "target": "B", "relation_type": "sufficient"},
    {"source": "B", "target": "C", "relation_type": "sufficient"}
  ]
}
```

Then ask `nesy.classify` for `A` and `C`. The server derives `A -> C` and
returns `classification="sufficient"` with a traceable path.

If `B` and `C` are declared exclusive and the graph proves both from the same
source, `nesy.check_contradictions` reports a hard contradiction. The Stop hook
can block a final answer when the answer includes conflicting structured
`NESY_FACTS`.

## Tools

| Tool | Purpose | Mutates State |
|---|---|---:|
| `nesy.assert_relations` | Add or update structured relations. | Yes |
| `nesy.list_relations` | List stored relations and derived implication edges. | No |
| `nesy.clear_relations` | Clear a context, store, filter, or allowed scope. | Yes |
| `nesy.classify` | Classify source/target relation by graph reachability. | No |
| `nesy.verify_chain` | Verify explicit or searched implication paths. | No |
| `nesy.assert_exclusive` | Declare mutually exclusive propositions. | Yes |
| `nesy.check_contradictions` | Check graph, facts, or combined contradictions. | No |
| `nesy.load_relations` | Load relation sets from inline data, files, or safe local `file://` URIs. | Yes |
| `nesy.export_relations` | Export relation sets inline or to allowed files. | Optional |
| `nesy.summarize_graph` | Return a compact deterministic graph summary. | No |
| `nesy.counterfactual` | Analyze what changes if a proposition is assumed false. | No |

## Proposition Identity

Relation records always keep `source` and `target` as human-readable labels.
Callers can optionally provide `source_id` and `target_id` as stable canonical
proposition IDs. When IDs are present, graph reasoning uses the IDs as nodes;
when they are absent, the labels remain the canonical nodes for compatibility.

This release does not add an alias registry, label/ID lookup, or explicit
`negates` model. Queries against ID-backed relations should use the canonical
IDs.

## Storage And Transports

Storage backends:

- `memory`: useful for short tests; state is lost on restart.
- `json`: local file persistence for simple single-user workflows.
- `sqlite`: recommended for long-lived local memory and hook/MCP sharing.

Transports:

- `stdio`: default MCP server mode.
- `http`: authenticated local Streamable HTTP daemon.

HTTP mode binds locally by default and requires `NESY_LOCAL_TOKEN`.

## Hooks

The CLI includes Claude Code hook helpers:

```bash
uv run nesy-reasoning-mcp hook pretooluse
uv run nesy-reasoning-mcp hook stop
```

- **PreToolUse** injects a compact graph summary as additional context.
- **Stop** checks the current graph or an explicit `NESY_FACTS:` JSON array in
  the final answer.

Hooks run in separate processes, so they should use SQLite, JSON, or the same
HTTP daemon. Process memory is not shared between stdio MCP and hook processes.

## Security Model

This project is local-first:

- HTTP mode uses a local bearer token.
- File load/export is restricted to configured `allowed_roots`.
- Hidden relation paths are blocked by default unless explicitly enabled.
- Mutating tools record audit entries when audit logging is enabled.
- Destructive or file-writing tools should still require confirmation in the MCP
  client or wrapper policy.

Inspect audit history:

```bash
NESY_CONFIG=/path/to/nesy-config.json uv run nesy-reasoning-mcp audit list --format json
```

See [docs/security.md](docs/security.md) for details.

## Evaluation

Offline deterministic evaluation:

```bash
env PYTHONPATH=src uv run nesy-reasoning-mcp eval run \
  --fixture benchmarks/fixtures/core.json \
  --format json
```

Agent mode-matrix evaluation:

```bash
env PYTHONPATH=src uv run nesy-reasoning-mcp eval agent \
  --fixture benchmarks/fixtures/core.json \
  --format json
```

Optional live OpenAI evaluation is manual-only and never required by CI:

```bash
uv sync --extra eval
export OPENAI_API_KEY='<set outside the repo>'
env PYTHONPATH=src uv run --extra eval nesy-reasoning-mcp eval llm \
  --fixture benchmarks/fixtures/core.json \
  --case-id classify_direct_sufficient \
  --format json
```

## Boundaries

- No automatic natural-language relation extraction.
- No hosted multi-user auth.
- No Postgres/team graph backend yet.
- No remote MCP resource fetching; `resource_uri` is limited to safe local
  `file://` loads.
- No PostToolBatch hook in v1.0.
- This is a reasoning aid, not a replacement for domain experts in legal,
  medical, financial, or safety-critical decisions.

## Development

```bash
uv sync --locked
uv run ruff format --check .
uv run ruff check .
uv run mypy src/nesy_reasoning_mcp
uv run pytest
env PYTHONPATH=src uv run nesy-reasoning-mcp eval run --fixture benchmarks/fixtures/core.json --format json
env PYTHONPATH=src uv run nesy-reasoning-mcp eval agent --fixture benchmarks/fixtures/core.json --format json
```

## Documentation

- [Full specification](docs/spec-v2.md)
- [SPEC compliance](SPEC_COMPLIANCE.md)
- [Agent instructions](AGENTS.md)
- [Claude Code instructions](CLAUDE.md)
- [Install as MCP server](docs/install.md)
- [Internal testing profile](docs/internal-testing.md)
- [Security](docs/security.md)
- [Uninstall / rollback](docs/uninstall.md)
- [Evaluation](docs/evaluation.md)
- [Roadmap](docs/roadmap.md)
- [Development](docs/development.md)
