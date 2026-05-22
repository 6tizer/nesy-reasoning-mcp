# Evaluation

NeSy Reasoning uses deterministic offline fixtures for v0.8 evaluation. The
default runner does not call an LLM and does not require API keys.

## Run

```bash
env PYTHONPATH=src uv run nesy-reasoning-mcp eval run --fixture benchmarks/fixtures/core.json
env PYTHONPATH=src uv run nesy-reasoning-mcp eval run --fixture benchmarks/fixtures/core.json --format json
```

The runner exits with code `0` only when the full MCP score is at least
`--min-score` and every fixture case passes.

## Fixture Shape

Each case defines:

- `relation_set`: a portable relation set loaded into a fresh in-memory store.
- `tool_name` and `tool_input`: the tool call under test.
- `expected`: path matchers over the structured tool result.
- `baselines`: static offline scores for LLM-only and ablation comparisons.

Fixture JSON must match `benchmarks/fixtures/core.schema.json`.

## Metrics

- `logical_accuracy`: classification, transitive, and business cases.
- `contradiction_recall`: hard-contradiction cases expected to be found.
- `false_contradiction_rate`: context-separated or negative contradiction cases
  incorrectly reported as hard contradictions.
- `counterfactual_conservatism`: counterfactual cases where the system avoids
  over-strong conclusions.
- `trace_completeness`: tool results with non-empty trace output.
- `latency_ms_avg`: average local tool execution latency.

## Failure To Regression

When a benchmark exposes a failure:

1. Add or update a fixture case with the smallest relation set that reproduces it.
2. Add a focused pytest assertion if the failure is a code regression.
3. Keep baseline scores static and explain any changed score in the PR.
4. Run the eval runner and the full local gate before opening the PR.

## Scope

v0.8 intentionally uses offline baselines. A real LLM runner can be added later
as an optional extension with explicit API-key setup, cost limits, and skip-by-default
CI behavior.
