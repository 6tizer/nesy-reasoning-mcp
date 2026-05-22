# Evaluation

NeSy Reasoning has three evaluation modes:

- deterministic offline fixtures for CI and regression testing
- optional live OpenAI LLM-only baseline runs for manual comparison
- Agent mode-matrix evaluation for internal-test readiness

CI never calls an external API and never requires `OPENAI_API_KEY`.

## Offline Fixture

```bash
env PYTHONPATH=src uv run nesy-reasoning-mcp eval run --fixture benchmarks/fixtures/core.json
env PYTHONPATH=src uv run nesy-reasoning-mcp eval run --fixture benchmarks/fixtures/core.json --format json
```

The offline runner exits with code `0` only when the full MCP score is at least
`--min-score` and every fixture case passes.

## Agent Matrix Eval

Run the deterministic Agent matrix locally or in CI:

```bash
env PYTHONPATH=src uv run nesy-reasoning-mcp eval agent \
  --fixture benchmarks/fixtures/core.json \
  --format json
```

The fixed modes are:

- `no_mcp`: static LLM-only fixture baseline.
- `tool_descriptions_only`: static baseline with tool descriptions but no calls.
- `classify_only`: only `nesy.classify` can be called.
- `classify_verify`: only `nesy.classify` and `nesy.verify_chain` can be called.
- `full_mcp`: all public MCP tools are available.

`deterministic` is the default runner. It never touches the network. Static
baseline modes read fixture scores; executable modes call local tool handlers
through the same benchmark relation-set setup as `eval run`.

Manual live Agent comparison is opt-in:

```bash
uv sync --extra eval
export OPENAI_API_KEY='<set outside the repo>'
env PYTHONPATH=src uv run --extra eval nesy-reasoning-mcp eval agent \
  --runner openai \
  --fixture benchmarks/fixtures/core.json \
  --case-id classify_direct_sufficient \
  --mode full_mcp \
  --format json
```

The live Agent runner asks the model for JSON tool actions, enforces the selected
mode's tool allowlist, executes allowed local MCP tool handlers, and scores the
final JSON prediction. It is manual-only and is not part of default CI.

## Live OpenAI Baseline

Install the optional eval dependency and set a key only for manual live runs:

```bash
uv sync --extra eval
export OPENAI_API_KEY='<set outside the repo>'
env PYTHONPATH=src uv run --extra eval nesy-reasoning-mcp eval llm \
  --fixture benchmarks/fixtures/core.json \
  --case-id classify_direct_sufficient \
  --format json
```

`eval llm` uses the OpenAI Responses API through the Python SDK. It sends each
case's relation set, requested tool name, tool input, and expected matcher paths,
then scores the returned JSON as an LLM-only baseline. Reports include MCP score,
`live_baseline_scores.openai_llm_only`, and
`live_marginal_contribution.openai_llm_only`.

Live reports do not store the API key or full prompt. Run a small set first with
`--case-id` to control cost.

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
3. Keep static baseline scores stable unless the fixture changes.
4. Explain any live LLM score change in the PR if the live runner was used.
5. Run offline `eval run`, deterministic `eval agent`, and the full local gate
   before opening the PR.
