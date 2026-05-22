# Agent Instructions For Internal Testing

Use this protocol when the agent is connected to the internal-test NeSy MCP
profile.

## Core Rule

Only structured facts are durable and reviewable. Plain prose is not imported
into the graph and is not fully checked by the Stop hook.

## When To Write Facts

Use `nesy.assert_relations` when a relation is stable enough to reuse later:

- `sufficient`: source is enough to imply target in the stated context.
- `necessary`: target cannot hold without source in the stated context.
- `equivalent`: source and target imply each other in the stated context.

Do not write:

- guesses
- vague associations
- one-off brainstorming
- facts with unclear context
- low-confidence claims that should be checked by a human first

## Final Answer Facts

When the final answer introduces new structured facts that should be checked
before stopping, append a `NESY_FACTS:` block with a JSON array.

Good example:

NESY_FACTS:
<!-- NESY_FACTS_EXAMPLE_START -->
```json
[
  {
    "source": "降价",
    "target": "销量增加",
    "relation_type": "sufficient",
    "context_id": "default",
    "store_id": "default",
    "confidence": 0.8,
    "metadata": {
      "domain": "pricing"
    }
  }
]
```
<!-- NESY_FACTS_EXAMPLE_END -->

Bad examples:

- Writing `NESY_FACTS: 降价会提升销量` instead of JSON.
- Omitting context when the fact is only true in one project or time period.
- Writing a relation because it sounds plausible but was not established.

## Stop Hook Boundary

The Stop hook checks the current graph and the `NESY_FACTS` array. It does not
extract hidden logical claims from ordinary natural-language paragraphs.

If the hook blocks, revise the answer or narrow the context. Do not remove facts
only to bypass the check.
