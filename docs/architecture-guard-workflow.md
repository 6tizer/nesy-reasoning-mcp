# Architecture Guard Workflow

This workflow checks architecture constraints with GitNexus-derived code facts
and NeSy reasoning. It does not treat GitHub issues or PR text as durable truth.

## Model

- GitNexus supplies observed code facts: call paths, imports, impact, and
  changed execution flows.
- NeSy stores and evaluates architecture constraints as implication rules and
  exclusive outcomes.
- GitHub issues and PRs are optional human context, not default fact sources.

## Run

Create or update an observed facts file from GitNexus output, then run:

```bash
uv run python scripts/architecture_guard.py \
  --rules examples/architecture-guard-rules.json \
  --facts examples/architecture-guard-facts.json \
  --format json
```

Exit code `0` means pass. Exit code `1` means a violation or contradiction was
found. Exit code `2` means invalid input.

## Fact Shape

Observed facts should use a stable anchor, normally `ObservedArchitectureFacts`.
Each GitNexus finding becomes an implication from the anchor to a proposition:

```json
{
  "anchor": "ObservedArchitectureFacts",
  "relations": [
    {
      "source": "ObservedArchitectureFacts",
      "target": "MCPServerCallsAgentSDK",
      "relation_type": "sufficient",
      "confidence": 1.0,
      "provenance": {
        "source": "gitnexus",
        "command": "npx gitnexus context run_stdio_server"
      }
    }
  ]
}
```

The guard is open-world: absence of a GitNexus finding is not proof that a
violation is impossible. Only explicit observed facts can trigger a violation.

## First Rules

The initial rule file checks three high-value constraints:

- MCP server code must not call Agent SDK orchestration directly.
- durable auto-write must pass contradiction validation.
- auto-written relations must carry evidence/provenance.

Add new constraints as rule propositions, not by ingesting all issue or PR text.
