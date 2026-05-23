# Agent Usage Policy

NeSy Reasoning MCP is a symbolic reasoning graph and logic audit layer for
agents. It is useful when an answer depends on implication chains, necessary
conditions, contradictions, exclusions, or counterfactual impact.

It is not a search engine, generic memory store, vector database, document
summarizer, or proof that a source is true. The MCP checks consistency of
asserted relations. It does not make weak evidence strong.

## Core Rule

Only assert a relation when the source material supports that logical relation
under the stated context.

| Relation | Use when | Do not use for |
|---|---|---|
| `sufficient` | A is enough to imply, produce, or trigger B under stated conditions. | "A may improve B", correlation, topical similarity, marketing claims. |
| `necessary` | B requires A, or without A, B cannot hold. | "A is often used in systems that do B" or "A helps B". |
| `equivalent` | A and B are definitions of the same state, or both directions are proven. | Similar terms, related concepts, or loose paraphrases. |
| `exclusive` | The propositions cannot jointly hold in the same context. | Alternatives that could still be combined, ranked, or sequenced. |

Do not encode vague association, search co-occurrence, correlation, anecdotal
evidence, or "may lead to" claims as strong logical relations.

## Autonomous Extraction Workflow

For autonomous research or codebase analysis:

1. Search or inspect sources.
2. Extract natural-language candidate claims.
3. Keep only claims strong enough to become logical relations.
4. Add `context_id`, `store_id`, `confidence`, and provenance metadata.
5. Use `confidence < 1` unless the relation is definitional or explicitly proven.
6. Call `nesy.check_contradictions` after adding relation batches.
7. Before the final answer, use `nesy.classify` or `nesy.verify_chain` for key conclusions.
8. Report `unknown`, `possibly_blocked`, or uncertainty instead of forcing a proof.

Recommended metadata:

```json
{
  "source": "A",
  "target": "B",
  "relation_type": "sufficient",
  "confidence": 0.65,
  "context_id": "paper_x_section_3",
  "metadata": {
    "provenance_url": "https://example.com/source",
    "quote_or_excerpt": "A triggers B under condition C.",
    "extraction_method": "agent_search",
    "claim_strength": "explicit_claim",
    "review_status": "candidate"
  }
}
```

## Usage Modes

### Manual Verification

Best for first-time users.

```text
Do not just give me the conclusion. Break your final answer into 3-5 key
logical relations, write them into the NeSy graph, then use verify_chain or
classify to check the conclusion.
```

### Semi-Automatic Graph Building

Recommended default.

```text
You may search and inspect sources, but do not store every related fact in
NeSy. Only write relations important to the final conclusion: clear sufficient
conditions, necessary conditions, equivalences, or mutually exclusive
constraints. Every relation must include context, confidence, and provenance
metadata. After adding relations, call check_contradictions. Before the final
answer, call classify or verify_chain for the key conclusion.
```

### Autonomous Research

Most powerful, but needs guardrails.

```text
Research this question autonomously. You may search sources and build a NeSy
reasoning graph. Treat extracted relations as candidate assumptions unless the
evidence clearly supports a logical implication, necessity, equivalence, or
exclusivity. Do not turn vague association, correlation, or weak influence into
sufficient or necessary relations. Use the graph to verify the final answer, and
report unknowns explicitly.
```

### External Memory Or GraphRAG Integration

Use this when another system retrieves candidate relations and NeSy should only
check the logic:

```text
external memory retrieval -> candidate relations -> NeSy ephemeral reasoning -> answer/evidence
```

Call `nesy.reason_over_relations` with the retrieved `relations`,
`exclusive_groups`, optional `propositions`, and a `query` mode. The tool builds
a temporary graph, returns the selected reasoning result, and leaves
`nesy.list_relations` unchanged. Set `persist=false` or omit it; `persist=true`
is rejected.

This mode is best for noisy retrieval candidates. Promote a relation to
`nesy.assert_relations` only after the user or policy decides it is stable enough
for long-lived memory.

For a planned Agent SDK workflow that extracts candidate relations, reviews
evidence, gates writes, and uses NeSy as the reasoning/storage layer, see
[Agent SDK ingestion design](agent-sdk-ingestion.md).

## Copyable Prompts

Research:

```text
Research whether A really solves B. Build a small NeSy graph of the key causal
chain. Only assert relations with clear evidence, include provenance metadata,
check contradictions, and verify the final conclusion. If the graph cannot prove
the conclusion, say unknown.
```

Codebase analysis:

```text
Read this codebase and model feature flags, config dependencies, and mutually
exclusive paths in NeSy. Then answer what necessarily breaks if config X is
removed, what might break, and what remains unknown.
```

Decision analysis:

```text
Compare proposals A/B/C. Encode goals, constraints, risks, and mutually
exclusive choices in NeSy. Verify which proposal is sufficient for the target
outcome, which requirements remain unmet, and which assumptions are unknown.
```

## Avoiding Overclaiming

Unsafe extraction:

```text
Source says: "A may improve B in some settings."
Bad relation: A sufficient B
Reason: "may improve" is weaker than sufficient implication.
```

Safer handling:

```text
Do not assert a sufficient relation. Mention that the current evidence suggests
possible influence, then ask for stronger evidence or leave the graph result as
unknown.
```

Example final answer:

```text
Under condition C, A is supported as one possible contributor to B. The graph
does not prove A is sufficient for B, and it does not prove A is necessary for
B. Current NeSy result: unknown.
```

## Future Schema Options

These are evaluated options, not implemented behavior:

- `status`: `candidate | verified | rejected`. This could let agents stage weak
  extracted claims without turning them into graph facts.
- `claim_type`: `association | correlation | influence | implication |
  necessity | equivalence | exclusion`. Only the strong logical types should
  drive deterministic proof.
- `nesy.evaluate_candidate_relation`: a future guard tool that could review an
  excerpt and proposed relation, then recommend `assert`, `do_not_assert`, or a
  safer weaker claim type.

These options should stay separate from current relation assertions until the
product needs a formal candidate staging workflow.
