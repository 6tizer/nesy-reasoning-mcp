# Claude Instructions For NeSy Reasoning MCP

Use these instructions when `nesy-reasoning` MCP is installed.

## Use NeSy When

Call NeSy before finalizing answers involving structured:

- causality, dependency, necessary/sufficient conditions
- multi-hop implication chains
- contradictions, exclusives, or direct opposition
- counterfactuals: "without X", "if X is false", "what if X is removed"
- long-lived reasoning memory

Do not require the user to name a tool. If the request matches, use the tool.

## Tool Routing

- `nesy.classify`: X causes/enables/requires Y; sufficient/necessary/equivalent/contradictory/unknown.
- `nesy.verify_chain`: X -> Y -> Z or any multi-hop implication proof.
- `nesy.counterfactual`: what changes if X is absent/false/removed.
- `nesy.check_contradictions`: before finalizing high-impact structured claims or `NESY_FACTS`.
- `nesy.summarize_graph`: compact context from current graph.
- `nesy.list_relations`: inspect stored evidence.
- `nesy.assert_relations`: only when user explicitly asks to remember/store/assert stable facts.
- `nesy.assert_exclusive`: only for user-approved mutually exclusive states.
- `nesy.load_relations`: only for user-provided or user-approved relation sets.

## Memory Rules

- Do not write uncertain guesses.
- Do not convert casual prose into memory without approval.
- Do not overwrite stores, clear relations, or load/export files without confirmation.
- If MCP is unavailable, say the deterministic check cannot be run; do not pretend it was verified.

## NESY_FACTS

When a final answer introduces new structured facts that should be checked by a Stop hook, append:

```text
NESY_FACTS:
[{"source":"Login","target":"SubmitOrder","relation_type":"necessary","context_id":"default"}]
```

Stop hooks check explicit graph facts and `NESY_FACTS`, not arbitrary prose.

## Claude Code Hooks

- PreToolUse may inject compact graph summary.
- Stop may block hard contradictions in `NESY_FACTS` or current explicit graph.
- Hook failures are usually fail-open unless `NESY_HOOK_FAIL_CLOSED=true`.

## Safety

- Treat `clear_relations`, file load/export, and replace-store operations as confirmation-required.
- Do not expose or log secrets.
- HTTP mode is local bearer-token auth, not hosted multi-user auth.
- `resource_uri` is limited to safe local `file://` loads.
- This is a reasoning aid, not legal/medical/financial/safety-critical authority.

## Repo Development

- Python 3.11+, `uv`, pytest, Ruff, mypy.
- Keep MCP schemas/output shapes stable unless intentionally changed and tested.
- Keep `structuredContent` and `content[0].text` mirrored.
- Keep stdio stdout clean and HTTP authenticated.
- Local gate: `uv run ruff format --check . && uv run ruff check . && uv run mypy src/nesy_reasoning_mcp && uv run pytest`.
