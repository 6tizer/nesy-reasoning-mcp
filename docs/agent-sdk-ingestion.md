# Agent SDK Ingestion Design

This document defines automated candidate relation ingestion. The current
implementation includes a live-capable OpenAI Agents SDK dry-run prototype and
an explicit safe write mode, plus a pre-write MCP validation helper. It does not
add a crawler, persistent review queue, or queue commit tools.

## Boundary

Run ingestion as an external Agent SDK app or CLI:

```text
external evidence
-> Agent SDK extractor/reviewer workflow
-> CandidateRelation[]
-> deterministic gate
-> nesy.reason_over_relations / nesy.check_contradictions
-> safe write or review queue
```

NeSy MCP remains the reasoning and storage layer. The ingestion app may call
read-only NeSy tools in dry-run mode and may call write tools only when explicit
write mode is enabled.

Do not make the NeSy MCP server call Agent SDK internally. That would make the
MCP server own model credentials, retries, traces, and agent loops, and would
increase hook recursion risk.

## Model Backends

OpenAI Agents SDK is the first orchestration target, but this is not an
OpenAI-only model design. Supported backend directions are:

- native OpenAI Responses models
- OpenAI-compatible Chat Completions endpoints via custom `base_url` and API key
- LiteLLM or Any-LLM adapters when provider coverage is more important than a
  direct SDK integration

Every backend must be validated before it can drive auto-write:

- structured output / JSON schema reliability
- tool calling or MCP tool compatibility
- retry, timeout, and usage behavior
- tracing configuration, including disabling OpenAI tracing for non-OpenAI keys
- context length enough to carry evidence excerpts

Claude Code support is separate from this ingestion runtime. Claude Code can use
NeSy through MCP and hooks without a Claude Agent SDK adapter.

## Shared Schemas

The shared schema module is `nesy_reasoning_mcp.auto_ingest`.

- `EvidenceRecord`: required source URL and evidence span.
- `CandidateRelation`: proposed relation plus confidence, context, store, and
  evidence.
- `ReviewDecision`: reviewer decision, final relation type, confidence, reasons,
  and risk flags.
- `GateResult`: deterministic gate action: `auto_write`, `queue`, or `reject`.
- `IngestionReport`: run-level report with candidates, reviews, gate results,
  approved relation inputs, diagnostics, and metadata.

The schemas are strict Pydantic models and reject unknown fields. Candidate
relations can be converted to existing `RelationInput` records only after the
review/gate workflow decides they are safe to commit.

## Tool Policy

Dry-run mode may use:

```text
nesy.reason_over_relations
nesy.check_contradictions
nesy.summarize_graph
nesy.list_relations
```

External orchestrators can also call:

```text
nesy.validate_candidate_relations
```

This helper accepts reviewed `CandidateRelation` records, optional
`ReviewDecision` records, and optional one-call proposition overlays. It runs
the deterministic gate, checks the candidate set for contradictions, then checks
the approved candidates against the current graph in combined mode. It always
returns `persisted=false`.

Safe write mode may additionally use:

```text
nesy.assert_relations
```

Write tools stay disabled unless a caller explicitly passes `--auto-write`.
The ingestion runtime does not call `nesy.load_relations`.

`nesy.validate_candidate_relations` is not an ingestion runtime. It does not call
the Agent SDK, fetch URLs, store a review queue, or write graph memory. Its
output is a validation report with candidate counts, `gate_results`,
`approved_relations`, diagnostics, reasoning details, graph stats, and trace
metadata.

## Dry-Run CLI Prototype

The first runnable slice is:

```bash
OPENAI_API_KEY=... uv run nesy-reasoning-mcp ingest agent-dry-run \
  --input examples/research-evidence.json \
  --format json
```

You may also pass explicit URL sources:

```bash
OPENAI_API_KEY=... uv run nesy-reasoning-mcp ingest agent-dry-run \
  --url https://example.com/report \
  --task "Extract only evidence-backed sufficient or necessary relations"
```

The script wrapper calls the same implementation:

```bash
OPENAI_API_KEY=... uv run python scripts/agent_ingest_openai.py \
  --input examples/research-evidence.json
```

JSON input accepts:

```json
{
  "task": "Find evidence-backed product dependency relations.",
  "question": "Does A require B?",
  "evidence": [
    {
      "url": "https://example.com/source",
      "span": "A cannot run unless B is configured."
    }
  ],
  "urls": ["https://example.com/explicit-source"],
  "metadata": {"case_id": "demo"}
}
```

URL support is intentionally narrow: only explicit public `http` and `https`
URLs are fetched, each with timeout and max-byte limits. Local URLs such as
`file://`, `localhost`, and loopback/private IPs are rejected. The command does
not search, crawl links, build embeddings, or write durable graph memory.

The prototype runs extractor and reviewer agents, then runs the deterministic
dry-run gate through NeSy read-only tools. The output is always an
`IngestionReport`. Approved relations appear in the report only; they are not
stored. CI tests mock the Agent SDK runner and do not call external APIs.

## Safe Write Mode

Safe write mode uses the same command with an explicit flag:

```bash
OPENAI_API_KEY=... uv run nesy-reasoning-mcp ingest agent-dry-run \
  --input examples/research-evidence.json \
  --auto-write \
  --min-write-confidence 0.85 \
  --format json
```

The script wrapper accepts the same flags:

```bash
OPENAI_API_KEY=... uv run python scripts/agent_ingest_openai.py \
  --input examples/research-evidence.json \
  --auto-write
```

When `--auto-write` is present, the report mode is `write`. Gate-approved
relations still appear in `approved_relations`, and successful persisted
relation IDs appear in `written_relation_ids`. The review queue is not
persisted; queued items remain visible through `gate_results`.

Writes use only `nesy.assert_relations` with contradiction rejection enabled.
The relation provenance includes candidate ID, evidence, reviewer reasons, risk
flags, and reviewer model. If assertion fails, the report keeps diagnostics and
does not pretend a relation was written.

## Gate Rules

Auto-write requires all of the following:

- evidence URL exists
- evidence span exists
- reviewer decision is `approve`
- confidence meets the configured threshold
- NeSy finds no hard contradiction
- NeSy dry-run reasoning succeeds
- write mode is explicitly enabled

Queue for review when the reviewer downgrades the relation type, confidence is
in the gray zone, source quality is weak, a hard contradiction appears, or the
model backend has not been validated for structured outputs and tool behavior.

Reject when the claim has no evidence, only topical similarity, only
correlation, or weak wording such as "may", "can", or "helps" was upgraded into
`sufficient` or `necessary`.

## Next PRs

The next implementation slices can add persistent review queue storage,
multi-reviewer voting, Claude-specific adapters, or richer backend validation
after model, tracing, retry, and gating behavior are validated.
