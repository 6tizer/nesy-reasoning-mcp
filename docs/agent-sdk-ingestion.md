# Agent SDK Ingestion Design

This document defines automated candidate relation ingestion. The current
implementation includes a live-capable OpenAI Agents SDK dry-run prototype. It
does not add a crawler, write mode, queue persistence, or a new MCP tool.

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

Write mode may additionally use:

```text
nesy.assert_relations
nesy.load_relations
```

Write tools must stay disabled unless a caller explicitly chooses write mode.

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

## Gate Rules

Auto-write requires all of the following:

- evidence URL exists
- evidence span exists
- reviewer decision is `approve`
- confidence meets the configured threshold
- NeSy finds no hard contradiction
- write mode is explicitly enabled

Queue for review when the reviewer downgrades the relation type, confidence is
in the gray zone, source quality is weak, a hard contradiction appears, or the
model backend has not been validated for structured outputs and tool behavior.

Reject when the claim has no evidence, only topical similarity, only
correlation, or weak wording such as "may", "can", or "helps" was upgraded into
`sufficient` or `necessary`.

## Next PRs

The next implementation slices can add review queue persistence and explicit
write mode after model, tracing, retry, and gating behavior are validated.
