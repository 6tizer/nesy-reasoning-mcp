# Agent Instructions For NeSy Reasoning MCP

Use these instructions when an AI agent works in a repository or project that has
the `nesy-reasoning` MCP server installed.

## What This MCP Is For

NeSy Reasoning MCP is a deterministic symbolic reasoning layer for structured
claims. It is useful when an answer depends on:

- causal claims
- necessary or sufficient conditions
- dependency relationships
- multi-hop implication chains
- contradictions or mutually exclusive states
- counterfactual questions
- long-lived structured reasoning memory

It is not a general natural-language extraction engine. Do not write ordinary
natural-language guesses into long-term memory unless the user approves them as
stable structured facts.

## When To Use NeSy Tools

Prefer using NeSy before finalizing an answer when the user asks about structured
logic, causality, dependencies, contradictions, or counterfactuals.

Use:

- `nesy.classify` when the user asks whether `X` causes, enables, requires, is
  sufficient for, or is necessary for `Y`.
- `nesy.verify_chain` when reasoning depends on a multi-hop chain such as
  `X -> Y -> Z`.
- `nesy.counterfactual` when the user asks "what if X is absent", "what if X is
  false", "without X", "if X is removed", or similar questions.
- `nesy.check_contradictions` before finalizing high-impact structured claims
  that may conflict with known facts or explicit `NESY_FACTS`.
- `nesy.summarize_graph` when the agent needs compact context from the current
  reasoning graph.
- `nesy.list_relations` when the agent needs to inspect stored evidence.

Do not require the user to name the tool. If the request clearly matches one of
the cases above and the MCP is available, call the appropriate tool.

## Writing Long-Term Memory

Only write to NeSy memory when the user explicitly asks to remember, store,
assert, import, or use stable structured facts.

Use:

- `nesy.assert_relations` for stable causal/dependency facts.
- `nesy.assert_exclusive` for mutually exclusive states.
- `nesy.load_relations` only when the user provides or approves the relation set.

Avoid:

- writing uncertain guesses
- writing facts inferred only from casual prose
- overwriting a store without confirmation
- clearing relations without confirmation
- importing or exporting files outside the user-approved workflow

## Final Answers And NESY_FACTS

If a final answer introduces new structured facts that should be checked by a
Stop hook, append a valid `NESY_FACTS:` JSON array:

```text
NESY_FACTS:
[
  {"source":"Login","target":"SubmitOrder","relation_type":"necessary","context_id":"default"}
]
```

The Stop hook checks the explicit graph and `NESY_FACTS`. It does not prove that
arbitrary natural-language prose is contradiction-free.

## Safety Boundaries

- Treat `clear_relations`, file load/export, and replace-store operations as
  confirmation-required.
- Do not expose or log secrets.
- HTTP mode is local-only bearer-token auth, not hosted multi-user auth.
- `resource_uri` is limited to safe local `file://` loads.
- This MCP is a reasoning aid, not a replacement for legal, medical, financial,
  or safety-critical domain review.

## If The MCP Is Not Available

If NeSy tools are not available, say that the deterministic check cannot be run
in this session. Do not pretend the result was verified by NeSy.

## Repository Development Rules

When editing this repository:

- Use Python 3.11+ and `uv`.
- Keep public MCP tool schemas and output shapes stable unless the change is
  intentional and tested.
- Keep `structuredContent` and `content[0].text` mirrored for tool results.
- Keep stdout clean in stdio mode.
- Keep HTTP mode authenticated.
- Keep default CI deterministic and API-key-free.
- Run the relevant local gate before committing:

```bash
uv run ruff format --check .
uv run ruff check .
uv run mypy src/nesy_reasoning_mcp
uv run pytest
```

If you rebuild the GitNexus index locally, use `--skip-agents-md` so generated
GitNexus guidance does not overwrite this product-level instruction file.
