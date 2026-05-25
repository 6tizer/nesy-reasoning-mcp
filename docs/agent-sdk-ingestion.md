# Agent SDK Ingestion Design

This document defines automated candidate relation ingestion. The current
implementation includes a live-capable OpenAI Agents SDK dry-run prototype and
an explicit safe write mode, plus a pre-write MCP validation helper. It also
supports explicit bounded Exa search retrieval and explicit bounded crawling
for evidence. It does not add an embedding/vector retrieval layer.

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
- `ReviewVotingPolicy`: multi-reviewer aggregation policy: `risk_tiered`,
  `unanimous`, or `majority`.
- `GateResult`: deterministic gate action: `auto_write`, `queue`, or `reject`.
- `IngestionReport`: run-level report with candidates, reviews, gate results,
  approved relation inputs, diagnostics, and metadata.
- `ReviewQueueRecord`: persisted queued candidate, review, gate result,
  diagnostics, provenance, and run metadata for later explicit action.

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
`ReviewDecision` records, optional one-call proposition overlays, and optional
multi-reviewer voting policy fields. It aggregates multiple reviews by
candidate ID, runs the deterministic gate, checks the candidate set for
contradictions, then checks the approved candidates against the current graph in
combined mode. It always returns `persisted=false`.

Safe write mode may additionally use:

```text
nesy.assert_relations
nesy.list_review_queue
nesy.commit_reviewed_relations
nesy.resolve_review_queue
```

Write tools stay disabled unless a caller explicitly passes `--auto-write`.
The ingestion runtime does not call `nesy.load_relations`.

`nesy.validate_candidate_relations` is not an ingestion runtime. It does not call
the Agent SDK, fetch URLs, store a review queue, or write graph memory. Its
output is a validation report with candidate counts, `gate_results`,
`approved_relations`, `review_aggregation`, diagnostics, reasoning details,
graph stats, and trace metadata.

## Dry-Run CLI Prototype

The first runnable slice is:

```bash
OPENAI_API_KEY=... uv run --no-editable nesy-reasoning-mcp ingest agent-dry-run \
  --input examples/research-evidence.json \
  --format json
```

You may also pass explicit URL sources:

```bash
OPENAI_API_KEY=... uv run --no-editable nesy-reasoning-mcp ingest agent-dry-run \
  --url https://example.com/report \
  --task "Extract only evidence-backed sufficient or necessary relations"
```

By default, URL sources are fetched exactly once and links are not followed.
To crawl explicit seed URLs, opt in with `--crawl`:

```bash
OPENAI_API_KEY=... uv run --no-editable nesy-reasoning-mcp ingest agent-dry-run \
  --url https://example.com/report \
  --crawl \
  --crawl-max-depth 1 \
  --crawl-max-pages 10 \
  --crawl-allow-domain docs.example.com \
  --task "Extract only evidence-backed sufficient or necessary relations" \
  --format json
```

Crawler limits are bounded by depth, page count, bytes per page, total bytes,
and per-page timeout. Seed pages are included at depth 0, and
`--crawl-max-depth 0` fetches only seed pages. When `--crawl` is set, crawled
URL evidence replaces separate one-shot URL fetches for the same seeds to avoid
duplicate requests. Discovered links are deduplicated, fragments and non-root
trailing slashes are ignored for duplicate detection, and traversal is
restricted to seed hosts unless `--crawl-allow-domain` is provided. Redirects
and discovered links use the same public HTTP(S), DNS, localhost, and
private-address protections as explicit URL fetching.

Crawler diagnostics are reported under `diagnostics`, and crawl run metadata is
stored under `metadata.crawl_retrieval`. Per-page failures, duplicate links, and
domain-filtered links do not fail the whole run if other evidence exists. If
crawling yields no evidence and no other evidence is available, the command
returns a diagnostic report and does not call the Agent SDK runtime or write
graph memory.

You may explicitly retrieve bounded search evidence through Exa. Search never
runs unless `--search-query` is provided:

```bash
EXA_API_KEY=... OPENAI_API_KEY=... uv run --no-editable nesy-reasoning-mcp \
  ingest agent-dry-run \
  --search-query "A requires B evidence" \
  --search-limit 5 \
  --search-include-domain example.com \
  --search-exclude-domain docs.example.com \
  --task "Extract only evidence-backed sufficient or necessary relations" \
  --format json
```

The search provider is `exa` in this slice and is called with
`POST https://api.exa.ai/search` using the `x-api-key` header. The key is read
from `EXA_API_KEY` by default, or from the env var named by
`--search-api-key-env`. Key values are not written to reports or logs.

Search results are converted into `EvidenceRecord` items with source URL, title,
bounded excerpt span, provider metadata, and retrieval timestamp. Include and
exclude domain filters are sent to Exa and enforced locally before any result
enters ingestion; exclude rules win over include rules. Local, private,
loopback, and reserved result URLs are rejected with diagnostics.

Search failures return an `IngestionReport` with diagnostics and
`metadata.search_retrieval`; the Agent SDK runtime is not called and graph
memory is not written. If all search results are filtered and no other evidence
exists, the command returns a diagnostic report instead of running extraction.

External GraphRAG or memory systems can pass provider-neutral retrieval batches
without adding a vector store to NeSy core:

```bash
OPENAI_API_KEY=... uv run --no-editable nesy-reasoning-mcp ingest agent-dry-run \
  --retrieval-input retrieval-batch.json \
  --task "Extract only evidence-backed sufficient or necessary relations" \
  --format json
```

The retrieval batch can contain evidence records or candidate relations. Evidence
records are converted to existing `EvidenceRecord` items with
`source_type="external_retrieval"` and provenance under
`metadata.retrieval`. Missing `retriever_name` or both `original_url` and
`source_document_id` produces a diagnostic report and does not call the Agent SDK
runtime or write graph memory. Retrieval evidence is ordered after input-file
evidence and before fetched/crawled URL evidence and search evidence.
Retrieval input JSON is bounded to 5,000,000 bytes. Scores are stored as raw
retriever metadata and are not normalized or used for gating. When an evidence
record has `source_document_id` but no `original_url`, its `EvidenceRecord.url`
uses an internal `external-retrieval://<document_id>#<chunk_id>` audit URI; this
scheme is not fetchable and is only a local provenance identifier.

Retrieved candidate relations can be validated without running extraction:

```bash
uv run --no-editable nesy-reasoning-mcp ingest retrieval validate \
  --input retrieval-batch.json \
  --voting-policy risk_tiered \
  --format json
```

This command calls the existing `nesy.validate_candidate_relations` path and
always returns `persisted=false`. Retrieved candidates that lack retriever/source
provenance are forced to `queue` even if the deterministic gate would otherwise
approve them.

The script wrapper calls the same implementation:

```bash
OPENAI_API_KEY=... uv run python scripts/agent_ingest_openai.py \
  --input examples/research-evidence.json
```

Reviewer voting is enabled by repeating `--reviewer-model`:

```bash
OPENAI_API_KEY=... uv run --no-editable nesy-reasoning-mcp ingest agent-dry-run \
  --input examples/research-evidence.json \
  --model gpt-4.1-mini \
  --reviewer-model gpt-4.1 \
  --reviewer-model gpt-4.1-mini \
  --voting-policy risk_tiered \
  --high-priority-reviewer-model gpt-4.1
```

If no reviewer model is provided, the runtime uses one reviewer with the
extractor model and preserves the previous single-reviewer behavior. Individual
review decisions stay in `IngestionReport.reviews`; the selected aggregate
review is passed to the deterministic gate, and audit details are stored under
`metadata.review_aggregation`.

Known OpenAI-compatible Chat Completions providers can use registry shortcuts.
API keys are read only from environment variables, not CLI plaintext arguments:

```bash
DEEPSEEK_API_KEY=... uv run --no-editable nesy-reasoning-mcp ingest agent-dry-run \
  --input examples/research-evidence.json \
  --provider deepseek \
  --format json
```

Use Flash by selecting the model under the same provider:

```bash
DEEPSEEK_API_KEY=... uv run --no-editable nesy-reasoning-mcp ingest agent-dry-run \
  --input examples/research-evidence.json \
  --provider deepseek \
  --model deepseek-v4-flash \
  --format json
```

The `deepseek` shortcut defaults to `deepseek-v4-pro` and uses DeepSeek JSON
Output rather than OpenAI JSON Schema structured output. The runtime sends
`response_format={"type":"json_object"}`, includes JSON schema guidance in the
prompt, and enables DeepSeek thinking with `reasoning_effort="high"` and
`extra_body={"thinking":{"type":"enabled"}}`. API failures, empty JSON content,
or schema validation errors stop before deterministic gate/write, so graph
memory is not mutated.

DeepSeek thinking defaults can be overridden explicitly when needed:

```bash
DEEPSEEK_API_KEY=... uv run --no-editable nesy-reasoning-mcp ingest agent-dry-run \
  --input examples/research-evidence.json \
  --provider deepseek \
  --provider-thinking disabled \
  --provider-reasoning-effort max \
  --format json
```

```bash
MOONSHOT_API_KEY=... uv run --no-editable nesy-reasoning-mcp ingest agent-dry-run \
  --input examples/research-evidence.json \
  --provider kimi \
  --format json
```

The `kimi` shortcut defaults to `kimi-k2.6` and uses the same direct Chat
Completions JSON Object path as DeepSeek. The runtime sends
`response_format={"type":"json_object"}`, includes JSON schema guidance in the
prompt, and enables Kimi thinking with `extra_body={"thinking":{"type":"enabled"}}`.
Kimi thinking can be disabled explicitly with `--provider-thinking disabled`;
`--provider-reasoning-effort` is not supported for Kimi.

```bash
OPENROUTER_API_KEY=... uv run --no-editable nesy-reasoning-mcp ingest agent-dry-run \
  --input examples/research-evidence.json \
  --provider openrouter \
  --model openai/gpt-latest \
  --provider-header 'HTTP-Referer=https://github.com/6tizer/nesy-reasoning-mcp' \
  --provider-header 'X-OpenRouter-Title=NeSy Reasoning MCP' \
  --format json
```

Use `--list-providers` to inspect built-in shortcuts. The registry is static
code and documentation, not a writable provider database.

Advanced integrations can still pass explicit generic flags. Provider base URLs
must be HTTPS:

```bash
DEEPSEEK_API_KEY=... uv run --no-editable nesy-reasoning-mcp ingest agent-dry-run \
  --input examples/research-evidence.json \
  --model deepseek-v4-pro \
  --base-url https://api.deepseek.com \
  --api-key-env DEEPSEEK_API_KEY \
  --format json
```

When `--provider` or `--base-url` is set, tracing is disabled by default for
that run because third-party provider calls should not be sent to OpenAI
tracing. Most provider shortcuts use `OpenAIChatCompletionsModel` with an
`AsyncOpenAI` client; DeepSeek and Kimi use a direct Chat Completions JSON
Object path because their APIs are more reliable with JSON Object response
formats for this ingestion schema. Provider metadata in the report
intentionally omits API-key environment names and base URLs. LiteLLM, Ollama,
and Claude Agent SDK adapters are separate future work.

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
not search unless `--search-query` is explicit and does not crawl unless
`--crawl` is explicit. It does not build embeddings or write durable graph
memory by default.

The prototype runs extractor and reviewer agents, aggregates reviewer votes
when needed, then runs the deterministic dry-run gate through NeSy read-only
tools. The output is always an `IngestionReport`. Approved relations appear in
the report only; they are not stored. CI tests mock the Agent SDK runner and do
not call external APIs.

## Safe Write Mode

Safe write mode uses the same command with an explicit flag:

```bash
OPENAI_API_KEY=... uv run --no-editable nesy-reasoning-mcp ingest agent-dry-run \
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
relation IDs appear in `written_relation_ids`. Queued items are persisted as
review queue records for JSON and SQLite backends; their IDs are also reported
under `metadata.review_queue_record_ids`.

Writes use only `nesy.assert_relations` with contradiction rejection enabled.
The relation provenance includes candidate ID, evidence, reviewer reasons, risk
flags, and reviewer model. If assertion fails, the report keeps diagnostics and
does not pretend a relation was written.

Review queue records can be inspected and acted on explicitly:

```bash
uv run --no-editable nesy-reasoning-mcp ingest queue list --format json
uv run --no-editable nesy-reasoning-mcp ingest queue commit --id queue_...
uv run --no-editable nesy-reasoning-mcp ingest queue resolve --id queue_... \
  --reason "duplicate or out of scope"
```

Commit re-runs validation and safe-write checks before writing relations. If
any selected pending record queues or rejects during validation, no selected
record is written.

## Scheduled Ingestion

Scheduled ingestion is explicit CLI work. The MCP stdio and HTTP servers do not
start a hidden scheduler or background daemon.

Create a dry-run schedule:

```bash
uv run --no-editable nesy-reasoning-mcp ingest schedule add \
  --name "docs dependency extraction" \
  --cron "*/30 * * * *" \
  --url https://example.com/report \
  --task "Extract only evidence-backed sufficient or necessary relations" \
  --format json
```

The v1 cron parser accepts five fields: minute, hour, day, month, weekday. It
supports `*`, numbers, comma lists, ranges, and steps such as `*/30`. It does
not support macros, seconds fields, or OS service installation. The default
timezone is `UTC`; use `--timezone` with an IANA zone name when needed.

Run jobs manually, run due jobs once, or start an explicit foreground worker:

```bash
uv run --no-editable nesy-reasoning-mcp ingest schedule list --format json
uv run --no-editable nesy-reasoning-mcp ingest schedule run --id sched_... --format json
uv run --no-editable nesy-reasoning-mcp ingest schedule run-due --format json
uv run --no-editable nesy-reasoning-mcp ingest schedule worker --poll-seconds 60
uv run --no-editable nesy-reasoning-mcp ingest schedule disable --id sched_...
```

Durable schedule state requires the JSON or SQLite storage backend. Job state
tracks last run, next run, run status, retry count, diagnostics, and report
location. Reports are written by default under
`~/.nesy-reasoning/ingestion-reports/{job_id}/{scheduled_run_id}.json`; use
`--report-dir` to choose a different report root.

Scheduled write mode has two extra safeguards beyond normal `--auto-write`.
Creation and runtime both require `--allow-scheduled-writes`, and scheduled
auto-write requires at least two `--reviewer-model` values by default:

```bash
uv run --no-editable nesy-reasoning-mcp ingest schedule add \
  --name "safe write extraction" \
  --cron "0 * * * *" \
  --url https://example.com/report \
  --auto-write \
  --allow-scheduled-writes \
  --reviewer-model gpt-4.1 \
  --reviewer-model gpt-4.1-mini \
  --voting-policy risk_tiered
```

Single-reviewer scheduled writes must opt in with
`--allow-single-reviewer-write`. Dry-run schedules have no multi-reviewer
requirement. Scheduled writes still use the same deterministic gate, safe write
path, and review queue behavior as manual `agent-dry-run --auto-write`.

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

The next implementation slices can add Claude-specific adapters,
GraphRAG/vector retrieval, scheduling, JavaScript/browser rendering, or richer
backend validation after model, tracing, retry, and gating behavior are
validated.
