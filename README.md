# NeSy Reasoning MCP

[![CI](https://github.com/6tizer/nesy-reasoning-mcp/actions/workflows/ci.yml/badge.svg)](https://github.com/6tizer/nesy-reasoning-mcp/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.11--3.14-blue)
![MCP](https://img.shields.io/badge/MCP-stdio%20%7C%20Streamable%20HTTP-green)
![Version](https://img.shields.io/badge/version-1.0.0-informational)

English | [简体中文](README.zh-CN.md)

NeSy Reasoning MCP gives agents a symbolic reasoning graph for checking
implications, necessity, contradictions, and counterfactuals.

It is an agent-side logic audit layer. It helps agents make key reasoning
relations explicit and checkable instead of relying only on hidden
natural-language reasoning. It improves reasoning auditability, not search
quality, source truth, or recall.

## What It Is And Is Not

| It is | It is not |
|---|---|
| Symbolic reasoning graph | Search engine |
| Consistency checker | Generic memory store |
| External reasoning scratchpad | Vector database |
| Implication, necessity, contradiction, and counterfactual verifier | Document summarizer |
| Persistent graph of key reasoning relations | Place to store every related fact |

Use it when hidden reasoning can go wrong: long research, codebase dependency
analysis, and product or engineering decision analysis. Skip it for simple
search, short summarization, casual Q&A, or "remember everything related to X"
workflows.

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
uv run --no-editable nesy-reasoning-mcp --help
```

Run the stdio MCP server:

```bash
uv run --no-editable nesy-reasoning-mcp --transport stdio
```

Use persistent SQLite storage:

```bash
mkdir -p ~/.nesy-reasoning
NESY_STORAGE_BACKEND=sqlite NESY_SQLITE_PATH=~/.nesy-reasoning/nesy.db \
  uv run --no-editable nesy-reasoning-mcp --transport stdio
```

Run the authenticated local Streamable HTTP daemon:

```bash
NESY_LOCAL_TOKEN='change-me' uv run --no-editable nesy-reasoning-mcp --transport http
```

Verify the deterministic benchmark:

```bash
env PYTHONPATH=src uv run --no-editable nesy-reasoning-mcp eval run \
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
- [Agent usage policy](docs/agent-usage.md)
- [Agent SDK ingestion design](docs/agent-sdk-ingestion.md)

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

## Recommended Agent Usage

Do not store every related fact an agent finds. Assert a relation only when the
source material supports a logical implication, necessity, equivalence, or
explicit exclusivity under a stated context.

Good fit prompts:

```text
Research whether A really solves B. Build a small NeSy graph of the key causal
chain, include provenance, and verify the final conclusion.

Read this codebase and model feature flags, config dependencies, and mutually
exclusive paths. Then answer what necessarily breaks if config X is removed.

Compare proposals A/B/C. Encode goals, constraints, risks, and exclusions, then
verify which proposal is sufficient for the target outcome.
```

Bad fit prompts:

```text
Search AI news.
Summarize this article.
Remember all related facts.
```

See [Agent usage policy](docs/agent-usage.md) for the do/don't table, prompts,
autonomous extraction workflow, and overclaiming examples.

For automated candidate extraction and review, see
[Agent SDK ingestion design](docs/agent-sdk-ingestion.md). By default it runs
extractor/reviewer agents and emits an `IngestionReport` without writing durable
graph memory:

```bash
OPENAI_API_KEY=... uv run --no-editable nesy-reasoning-mcp ingest agent-dry-run \
  --input evidence.json --format json
```

Durable writes require the explicit safe-write boundary:

```bash
OPENAI_API_KEY=... uv run --no-editable nesy-reasoning-mcp ingest agent-dry-run \
  --input evidence.json --auto-write --min-write-confidence 0.85 --format json
```

Known OpenAI-compatible Chat Completions providers can use registry shortcuts:

```bash
DEEPSEEK_API_KEY=... uv run --no-editable nesy-reasoning-mcp ingest agent-dry-run \
  --input evidence.json \
  --provider deepseek \
  --format json
```

```bash
MOONSHOT_API_KEY=... uv run --no-editable nesy-reasoning-mcp ingest agent-dry-run \
  --input evidence.json \
  --provider kimi \
  --format json
```

```bash
OPENROUTER_API_KEY=... uv run --no-editable nesy-reasoning-mcp ingest agent-dry-run \
  --input evidence.json \
  --provider openrouter \
  --model openai/gpt-latest \
  --format json
```

Use `--list-providers` to inspect built-in shortcuts. Advanced integrations can
still pass explicit `--base-url`, `--api-key-env`, and `--provider-header`
flags.

External orchestrators can call `nesy.validate_candidate_relations` before any
write path to reuse NeSy's deterministic gate and contradiction checks without
mutating the graph.

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
| `nesy.reason_over_relations` | Run reasoning over caller-supplied temporary relations. | No |
| `nesy.validate_candidate_relations` | Validate reviewed candidates before durable writes. | No |

## Ephemeral Reasoning

Use `nesy.reason_over_relations` when another memory or retrieval system has
candidate relations that should be checked without writing them into NeSy's
long-lived graph:

```text
external memory retrieval -> candidate relations -> NeSy ephemeral reasoning -> answer/evidence
```

The tool accepts temporary `relations`, `exclusive_groups`, `propositions`,
`independence_records`, and a `query` mode such as `classify`, `verify_chain`,
`counterfactual`, `check_contradictions`, or `summarize_graph`. It returns the
selected reasoning result under `result` with `persisted=false`.

This is for downstream logic checks only. It does not crawl documents, build an
embedding index, extract natural-language facts, or auto-write candidates into
persistent memory.

Automated external evidence ingestion is tracked separately in the
[Agent SDK ingestion design](docs/agent-sdk-ingestion.md). Its default mode is
dry-run; durable writes require explicit `--auto-write`, gate approval,
contradiction rejection, and `nesy.assert_relations`.

For Agent SDK or other external orchestrators, `nesy.validate_candidate_relations`
provides the same pre-write validation as an MCP helper. It returns gate results,
approved relation inputs, diagnostics, and reasoning details with
`persisted=false`; it does not fetch evidence, call an Agent SDK, store a queue,
or write memory.

## Proposition Identity

Relation records always keep `source` and `target` as human-readable labels.
Callers can optionally provide `source_id` and `target_id` as stable canonical
proposition IDs. When IDs are present, graph reasoning uses the IDs as nodes;
when they are absent, the labels remain the canonical nodes for compatibility.

Relation sets can include `propositions` with `id`, `label`, `aliases`,
optional `negates`, and metadata. `nesy.load_relations` stores these
propositions, `nesy.export_relations` exports them, and relation labels are
normalized to IDs when they exactly match a stored ID, label, or alias. Matching
is exact after normal schema trimming; there is no fuzzy or semantic alias
matching. `negates` declares canonical ID opposition for deterministic
contradiction checks.

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
uv run --no-editable nesy-reasoning-mcp hook pretooluse
uv run --no-editable nesy-reasoning-mcp hook stop
```

- **PreToolUse** injects a compact graph summary as additional context.
- **Stop** checks the current graph or explicit `NESY_FACTS:` in the final
  answer. `NESY_FACTS` may be a legacy relation array or an object with
  `relations` and `propositions`.

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
NESY_CONFIG=/path/to/nesy-config.json uv run --no-editable nesy-reasoning-mcp audit list --format json
```

See [docs/security.md](docs/security.md) for details.

## Evaluation

Offline deterministic evaluation:

```bash
env PYTHONPATH=src uv run --no-editable nesy-reasoning-mcp eval run \
  --fixture benchmarks/fixtures/core.json \
  --format json
```

Agent mode-matrix evaluation:

```bash
env PYTHONPATH=src uv run --no-editable nesy-reasoning-mcp eval agent \
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
env PYTHONPATH=src uv run --no-editable nesy-reasoning-mcp eval run --fixture benchmarks/fixtures/core.json --format json
env PYTHONPATH=src uv run --no-editable nesy-reasoning-mcp eval agent --fixture benchmarks/fixtures/core.json --format json
```

## Documentation

- [Full specification](docs/spec-v2.md)
- [SPEC compliance](SPEC_COMPLIANCE.md)
- [Agent instructions](AGENTS.md)
- [Agent usage policy](docs/agent-usage.md)
- [Claude Code instructions](CLAUDE.md)
- [Install as MCP server](docs/install.md)
- [Internal testing profile](docs/internal-testing.md)
- [Security](docs/security.md)
- [Uninstall / rollback](docs/uninstall.md)
- [Evaluation](docs/evaluation.md)
- [Roadmap](docs/roadmap.md)
- [Development](docs/development.md)
