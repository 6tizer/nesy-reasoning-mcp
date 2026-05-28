"""CLI helpers for Agent SDK dry-run ingestion."""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
from datetime import UTC, datetime
from pathlib import Path
from re import fullmatch
from typing import Any, TextIO
from urllib.parse import urlparse

import anyio
from pydantic import ValidationError

from nesy_reasoning_mcp.auto_ingest.crawler import (
    DEFAULT_CRAWL_MAX_DEPTH,
    DEFAULT_CRAWL_MAX_PAGE_BYTES,
    DEFAULT_CRAWL_MAX_PAGES,
    DEFAULT_CRAWL_MAX_TOTAL_BYTES,
    DEFAULT_CRAWL_TIMEOUT_SECONDS,
    MAX_CRAWL_MAX_DEPTH,
    MAX_CRAWL_MAX_PAGES,
    MAX_CRAWL_MAX_TOTAL_BYTES,
    MAX_CRAWL_TIMEOUT_SECONDS,
    CrawlOptions,
    CrawlResult,
    crawl_url_evidence,
)
from nesy_reasoning_mcp.auto_ingest.external_retrieval import (
    MAX_EXTERNAL_RETRIEVAL_INPUT_BYTES,
    ExternalRetrievalBatch,
    ExternalRetrievalConversion,
    convert_external_retrieval_batch,
)
from nesy_reasoning_mcp.auto_ingest.extraction import ExtractionModelConfig
from nesy_reasoning_mcp.auto_ingest.fetcher import (
    DEFAULT_FETCH_TIMEOUT_SECONDS,
    DEFAULT_MAX_FETCH_BYTES,
    fetch_url_evidence_many,
)
from nesy_reasoning_mcp.auto_ingest.openai_agents import (
    LLMRuntimeOptions,
    OpenAIAgentsDryRunError,
    OpenAICompatibleProviderConfig,
    ProgressCallback,
    ReviewerModelConfig,
    run_openai_agents_ingestion,
)
from nesy_reasoning_mcp.auto_ingest.providers import (
    ProviderRegistryEntry,
    ProviderStructuredOutputMode,
    get_provider_entry,
    list_provider_entries,
)
from nesy_reasoning_mcp.auto_ingest.review_worker import ReviewWorkerConfig
from nesy_reasoning_mcp.auto_ingest.scheduler import (
    DEFAULT_SCHEDULE_MAX_RETRIES,
    DEFAULT_SCHEDULE_POLL_SECONDS,
    DEFAULT_SCHEDULE_RETRY_BACKOFF_SECONDS,
    DEFAULT_SCHEDULE_TIMEZONE,
    MAX_SCHEDULE_POLL_SECONDS,
    ScheduledIngestionJob,
    ScheduledIngestionJobFilter,
    ScheduledIngestionJobStatus,
    ScheduledIngestionProviderConfig,
    ScheduledIngestionRetryPolicy,
    ScheduledIngestionRun,
    ScheduledIngestionRunFilter,
    ScheduledIngestionRunStatus,
    ScheduledIngestionRuntimeConfig,
    ScheduledIngestionRunTrigger,
    ScheduledIngestionSourceConfig,
    ScheduledIngestionState,
    ScheduledIngestionWriteConfig,
    job_due,
    next_cron_run,
    next_state_for_failure,
    next_state_for_skip,
    next_state_for_success,
    scheduled_reviewer_count,
    scheduled_write_diagnostics,
    write_scheduled_report,
)
from nesy_reasoning_mcp.auto_ingest.schemas import (
    IngestionInput,
    IngestionMode,
    IngestionReport,
    ReviewVotingPolicy,
)
from nesy_reasoning_mcp.auto_ingest.search import (
    DEFAULT_SEARCH_API_KEY_ENV,
    DEFAULT_SEARCH_LIMIT,
    DEFAULT_SEARCH_PROVIDER,
    DEFAULT_SEARCH_TIMEOUT_SECONDS,
    MAX_SEARCH_LIMIT,
    MAX_SEARCH_TIMEOUT_SECONDS,
    SearchProviderName,
    SearchRetrievalOptions,
    SearchRetrievalResult,
    retrieve_search_evidence,
)
from nesy_reasoning_mcp.auto_ingest.worker import (
    IngestionWorkerConfig,
    run_ingestion_worker,
)
from nesy_reasoning_mcp.config import load_config
from nesy_reasoning_mcp.schemas import Diagnostic
from nesy_reasoning_mcp.store import create_relation_store
from nesy_reasoning_mcp.tool_names import (
    COMMIT_REVIEWED_RELATIONS,
    LIST_REVIEW_QUEUE,
    RESOLVE_REVIEW_QUEUE,
    VALIDATE_CANDIDATE_RELATIONS,
)
from nesy_reasoning_mcp.tool_registry import call_tool

_HTTP_HEADER_KEY_PATTERN = r"[!#$%&'*+\-.^_`|~0-9A-Za-z]+"


def add_ingest_subparser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Register ingestion CLI subcommands on the top-level parser."""
    ingest_parser = subparsers.add_parser(
        "ingest",
        help="Run external evidence ingestion helpers.",
    )
    ingest_subparsers = ingest_parser.add_subparsers(dest="ingest_command")
    dry_run_parser = ingest_subparsers.add_parser(
        "agent-dry-run",
        help="Run OpenAI Agents SDK dry-run candidate ingestion.",
    )
    add_agent_dry_run_arguments(dry_run_parser)
    queue_parser = ingest_subparsers.add_parser(
        "queue",
        help="Inspect and act on persisted ingestion review queue records.",
    )
    add_review_queue_arguments(queue_parser)
    retrieval_parser = ingest_subparsers.add_parser(
        "retrieval",
        help="Validate external retrieval candidate batches.",
    )
    add_retrieval_arguments(retrieval_parser)
    schedule_parser = ingest_subparsers.add_parser(
        "schedule",
        help="Manage scheduled Agent SDK ingestion jobs.",
    )
    add_schedule_arguments(schedule_parser)
    worker_parser = ingest_subparsers.add_parser(
        "worker",
        help="Run the queued conversation-turn ingestion worker.",
    )
    add_worker_arguments(worker_parser)


def add_agent_dry_run_arguments(
    parser: argparse.ArgumentParser,
    *,
    include_canonicalize_preview: bool = True,
) -> None:
    """Add shared arguments for the OpenAI Agents SDK dry-run command."""
    parser.add_argument("--input", default=None, help="JSON input file path.")
    parser.add_argument(
        "--retrieval-input",
        default=None,
        help="External retrieval JSON batch to add as evidence.",
    )
    parser.add_argument("--url", action="append", default=[], help="Explicit HTTP(S) URL source.")
    parser.add_argument("--task", default=None, help="Optional extraction task.")
    parser.add_argument("--question", default=None, help="Optional question to answer.")
    parser.add_argument("--model", default=None, help="OpenAI Agents SDK model override.")
    parser.add_argument(
        "--reviewer-model",
        action="append",
        default=[],
        dest="reviewer_models",
        help=(
            "Reviewer model name for the default/provider config. May be repeated for "
            "multi-reviewer voting."
        ),
    )
    parser.add_argument(
        "--reviewer",
        action="append",
        default=[],
        dest="reviewers",
        help="Provider-qualified reviewer in PROVIDER:MODEL form. May be repeated.",
    )
    parser.add_argument(
        "--voting-policy",
        choices=[policy.value for policy in ReviewVotingPolicy],
        default=ReviewVotingPolicy.RISK_TIERED.value,
        help="Policy for aggregating multiple reviewer decisions.",
    )
    parser.add_argument(
        "--high-priority-reviewer-model",
        action="append",
        default=[],
        dest="high_priority_reviewer_models",
        help=(
            "High-priority reviewer model name for the default/provider config; reject or "
            "needs_human/downgrade votes have priority."
        ),
    )
    parser.add_argument(
        "--high-priority-reviewer",
        action="append",
        default=[],
        dest="high_priority_reviewers",
        help="Provider-qualified high-priority reviewer in PROVIDER:MODEL form.",
    )
    parser.add_argument(
        "--provider",
        default=None,
        help="Known provider shortcut: deepseek, kimi, or openrouter.",
    )
    parser.add_argument(
        "--list-providers",
        action="store_true",
        help="List known provider shortcuts and exit.",
    )
    parser.add_argument(
        "--base-url",
        default=None,
        help="OpenAI-compatible Chat Completions base URL.",
    )
    parser.add_argument(
        "--api-key-env",
        default=None,
        help="Environment variable containing the provider API key.",
    )
    parser.add_argument(
        "--provider-header",
        action="append",
        default=[],
        help="Provider header in KEY=VALUE form. May be repeated.",
    )
    parser.add_argument(
        "--provider-thinking",
        choices=["enabled", "disabled"],
        default=None,
        help="Override JSON Object provider thinking mode, for example DeepSeek or Kimi.",
    )
    parser.add_argument(
        "--provider-reasoning-effort",
        choices=["high", "max"],
        default=None,
        help="Override JSON Object provider reasoning effort, for example DeepSeek.",
    )
    parser.add_argument(
        "--extractor-timeout-seconds",
        type=float,
        default=180,
        help="Extractor LLM timeout in seconds.",
    )
    parser.add_argument(
        "--high-priority-reviewer-timeout-seconds",
        type=float,
        default=180,
        help="High-priority reviewer LLM timeout in seconds.",
    )
    parser.add_argument(
        "--reviewer-timeout-seconds",
        type=float,
        default=120,
        help="Reviewer LLM timeout in seconds.",
    )
    parser.add_argument(
        "--extractor-max-tokens",
        type=int,
        default=4096,
        help="JSON Object extractor max output tokens.",
    )
    parser.add_argument(
        "--reviewer-max-tokens",
        type=int,
        default=2048,
        help="JSON Object reviewer max output tokens.",
    )
    parser.add_argument(
        "--progress",
        choices=["auto", "off"],
        default="auto",
        help="Emit ingestion progress to stderr or disable it.",
    )
    parser.add_argument(
        "--disable-tracing",
        action="store_true",
        help="Disable OpenAI Agents SDK tracing for this run.",
    )
    parser.add_argument(
        "--auto-write",
        action="store_true",
        help="Persist gate-approved relations with safe write checks.",
    )
    if include_canonicalize_preview:
        parser.add_argument(
            "--canonicalize-preview",
            action="store_true",
            help=(
                "Run proposition canonicalization during dry-run preview without writing "
                "graph memory."
            ),
        )
    parser.add_argument(
        "--min-write-confidence",
        type=float,
        default=0.85,
        help="Minimum reviewed confidence required for --auto-write.",
    )
    parser.add_argument("--format", choices=["json", "text"], default="json")
    parser.add_argument("--output", default=None, help="Optional report output path.")
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=DEFAULT_FETCH_TIMEOUT_SECONDS,
        help="Per-URL fetch timeout.",
    )
    parser.add_argument(
        "--max-url-bytes",
        type=int,
        default=DEFAULT_MAX_FETCH_BYTES,
        help="Maximum bytes to read from each URL.",
    )
    parser.add_argument(
        "--search-query",
        action="append",
        default=[],
        dest="search_queries",
        help="Explicit search query for retrieval evidence. May be repeated.",
    )
    parser.add_argument(
        "--search-provider",
        choices=[provider.value for provider in SearchProviderName],
        default=DEFAULT_SEARCH_PROVIDER,
        help="Search provider for explicit retrieval queries.",
    )
    parser.add_argument(
        "--search-limit",
        type=int,
        default=DEFAULT_SEARCH_LIMIT,
        help=f"Maximum search results per query, 1-{MAX_SEARCH_LIMIT}.",
    )
    parser.add_argument(
        "--search-timeout-seconds",
        type=float,
        default=DEFAULT_SEARCH_TIMEOUT_SECONDS,
        help=f"Per-search timeout, max {MAX_SEARCH_TIMEOUT_SECONDS} seconds.",
    )
    parser.add_argument(
        "--search-include-domain",
        action="append",
        default=[],
        dest="search_include_domains",
        help="Search result domain allowlist entry. May be repeated.",
    )
    parser.add_argument(
        "--search-exclude-domain",
        action="append",
        default=[],
        dest="search_exclude_domains",
        help="Search result domain blocklist entry. May be repeated.",
    )
    parser.add_argument(
        "--search-api-key-env",
        default=DEFAULT_SEARCH_API_KEY_ENV,
        help="Environment variable containing the search provider API key.",
    )
    parser.add_argument(
        "--crawl",
        action="store_true",
        help="Crawl explicit URL seeds with bounded same-domain traversal.",
    )
    parser.add_argument(
        "--crawl-max-depth",
        type=int,
        default=DEFAULT_CRAWL_MAX_DEPTH,
        help=f"Maximum crawl link depth, 0-{MAX_CRAWL_MAX_DEPTH}.",
    )
    parser.add_argument(
        "--crawl-max-pages",
        type=int,
        default=DEFAULT_CRAWL_MAX_PAGES,
        help=f"Maximum crawl pages, 1-{MAX_CRAWL_MAX_PAGES}.",
    )
    parser.add_argument(
        "--crawl-max-page-bytes",
        type=int,
        default=DEFAULT_CRAWL_MAX_PAGE_BYTES,
        help="Maximum bytes to read from each crawled page.",
    )
    parser.add_argument(
        "--crawl-max-total-bytes",
        type=int,
        default=DEFAULT_CRAWL_MAX_TOTAL_BYTES,
        help=f"Maximum bytes to read across the crawl, max {MAX_CRAWL_MAX_TOTAL_BYTES}.",
    )
    parser.add_argument(
        "--crawl-timeout-seconds",
        type=float,
        default=DEFAULT_CRAWL_TIMEOUT_SECONDS,
        help=f"Per-page crawl timeout, max {MAX_CRAWL_TIMEOUT_SECONDS} seconds.",
    )
    parser.add_argument(
        "--crawl-allow-domain",
        action="append",
        default=[],
        dest="crawl_allow_domains",
        help="Additional domain allowed for crawl links. May be repeated.",
    )


def add_review_queue_arguments(parser: argparse.ArgumentParser) -> None:
    """Add review queue CLI subcommands."""
    queue_subparsers = parser.add_subparsers(dest="queue_command")
    list_parser = queue_subparsers.add_parser("list", help="List pending review queue records.")
    list_parser.add_argument("--status", choices=["pending", "committed", "resolved"])
    list_parser.add_argument("--run-id")
    list_parser.add_argument("--candidate-id")
    list_parser.add_argument("--store-id")
    list_parser.add_argument("--context-id")
    list_parser.add_argument("--limit", type=int, default=50)
    list_parser.add_argument("--cursor")
    list_parser.add_argument("--format", choices=["json", "text"], default="json")

    commit_parser = queue_subparsers.add_parser(
        "commit",
        help="Commit explicit pending review queue records.",
    )
    commit_parser.add_argument("--id", action="append", required=True, dest="ids")
    commit_parser.add_argument("--min-write-confidence", type=float, default=0.85)
    commit_parser.add_argument("--format", choices=["json", "text"], default="json")

    resolve_parser = queue_subparsers.add_parser(
        "resolve",
        help="Resolve explicit pending review queue records without writing graph memory.",
    )
    resolve_parser.add_argument("--id", action="append", required=True, dest="ids")
    resolve_parser.add_argument("--reason", required=True)
    resolve_parser.add_argument("--format", choices=["json", "text"], default="json")


def add_worker_arguments(parser: argparse.ArgumentParser) -> None:
    """Add foreground conversation ingestion worker arguments."""
    parser.add_argument("--poll-seconds", type=float, default=10.0)
    parser.add_argument("--max-jobs", type=int, default=None)
    parser.add_argument("--claim-limit", type=int, default=5)
    parser.add_argument("--max-merge-jobs", type=int, default=3)
    parser.add_argument("--max-queue-depth", type=int, default=None)
    parser.add_argument("--model", default=None, help="Extractor model override.")
    parser.add_argument(
        "--provider",
        choices=[entry.name for entry in list_provider_entries()],
        default=None,
    )
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--api-key-env", default=None)
    parser.add_argument("--provider-header", action="append", default=[])
    parser.add_argument("--provider-thinking", choices=["enabled", "disabled"], default=None)
    parser.add_argument("--provider-reasoning-effort", choices=["high", "max"], default=None)
    parser.add_argument("--extractor-timeout-seconds", type=float, default=180)
    parser.add_argument("--extractor-max-tokens", type=int, default=4096)
    parser.add_argument(
        "--reviewer-model",
        action="append",
        default=[],
        dest="reviewer_models",
        help=(
            "Reviewer model name for the default/provider config. May be repeated for "
            "multi-reviewer voting."
        ),
    )
    parser.add_argument(
        "--reviewer",
        action="append",
        default=[],
        dest="reviewers",
        help="Provider-qualified reviewer in PROVIDER:MODEL form. May be repeated.",
    )
    parser.add_argument(
        "--high-priority-reviewer-model",
        action="append",
        default=[],
        dest="high_priority_reviewer_models",
        help=(
            "High-priority reviewer model name for the default/provider config; reject or "
            "needs_human/downgrade votes have priority."
        ),
    )
    parser.add_argument(
        "--high-priority-reviewer",
        action="append",
        default=[],
        dest="high_priority_reviewers",
        help="Provider-qualified high-priority reviewer in PROVIDER:MODEL form.",
    )
    parser.add_argument(
        "--voting-policy",
        choices=[policy.value for policy in ReviewVotingPolicy],
        default=ReviewVotingPolicy.RISK_TIERED.value,
    )
    parser.add_argument("--reviewer-timeout-seconds", type=float, default=120)
    parser.add_argument("--high-priority-reviewer-timeout-seconds", type=float, default=180)
    parser.add_argument("--reviewer-max-tokens", type=int, default=2048)
    parser.add_argument("--review-poll-seconds", type=float, default=10.0)
    parser.add_argument("--review-claim-limit", type=int, default=20)
    parser.add_argument("--min-write-confidence", type=float, default=0.85)
    parser.add_argument("--allow-review-queue-writes", action="store_true")
    parser.add_argument("--allow-single-reviewer-write", action="store_true")
    parser.add_argument("--format", choices=["json", "text"], default="json")


def add_retrieval_arguments(parser: argparse.ArgumentParser) -> None:
    """Add external retrieval CLI subcommands."""
    retrieval_subparsers = parser.add_subparsers(dest="retrieval_command")
    validate_parser = retrieval_subparsers.add_parser(
        "validate",
        help="Validate external retrieval candidate relations without persisting.",
    )
    validate_parser.add_argument("--input", required=True, help="External retrieval JSON batch.")
    validate_parser.add_argument("--min-write-confidence", type=float, default=0.85)
    validate_parser.add_argument(
        "--voting-policy",
        choices=[policy.value for policy in ReviewVotingPolicy],
        default=ReviewVotingPolicy.RISK_TIERED.value,
    )
    validate_parser.add_argument(
        "--high-priority-reviewer-model",
        action="append",
        default=[],
        dest="high_priority_reviewer_models",
    )
    validate_parser.add_argument("--format", choices=["json", "text"], default="json")


def add_schedule_arguments(parser: argparse.ArgumentParser) -> None:
    """Add scheduled ingestion CLI subcommands."""
    schedule_subparsers = parser.add_subparsers(dest="schedule_command")

    add_parser = schedule_subparsers.add_parser(
        "add",
        help="Create or update a scheduled Agent SDK ingestion job.",
    )
    add_parser.add_argument("--name", required=True)
    add_parser.add_argument("--cron", required=True)
    add_parser.add_argument("--timezone", default=DEFAULT_SCHEDULE_TIMEZONE)
    add_parser.add_argument("--max-retries", type=int, default=DEFAULT_SCHEDULE_MAX_RETRIES)
    add_parser.add_argument(
        "--retry-backoff-seconds",
        type=int,
        default=DEFAULT_SCHEDULE_RETRY_BACKOFF_SECONDS,
    )
    add_parser.add_argument("--allow-scheduled-writes", action="store_true")
    add_parser.add_argument("--allow-single-reviewer-write", action="store_true")
    add_parser.add_argument("--report-dir", default=None)
    add_agent_dry_run_arguments(add_parser, include_canonicalize_preview=False)

    list_parser = schedule_subparsers.add_parser("list", help="List scheduled ingestion jobs.")
    list_parser.add_argument(
        "--status",
        choices=[status.value for status in ScheduledIngestionJobStatus],
    )
    list_parser.add_argument("--limit", type=int, default=50)
    list_parser.add_argument("--format", choices=["json", "text"], default="json")

    runs_parser = schedule_subparsers.add_parser("runs", help="List scheduled ingestion runs.")
    runs_parser.add_argument("--job-id")
    runs_parser.add_argument(
        "--status",
        choices=[status.value for status in ScheduledIngestionRunStatus],
    )
    runs_parser.add_argument("--limit", type=int, default=50)
    runs_parser.add_argument("--format", choices=["json", "text"], default="json")

    run_parser = schedule_subparsers.add_parser("run", help="Run one scheduled ingestion job now.")
    run_parser.add_argument("--id", required=True, dest="job_id")
    run_parser.add_argument("--format", choices=["json", "text"], default="json")

    run_due_parser = schedule_subparsers.add_parser("run-due", help="Run due active jobs once.")
    run_due_parser.add_argument("--limit", type=int, default=50)
    run_due_parser.add_argument("--format", choices=["json", "text"], default="json")

    worker_parser = schedule_subparsers.add_parser(
        "worker",
        help="Run a foreground scheduled ingestion worker.",
    )
    worker_parser.add_argument("--poll-seconds", type=float, default=DEFAULT_SCHEDULE_POLL_SECONDS)
    worker_parser.add_argument("--max-runs", type=int, default=None)
    worker_parser.add_argument("--format", choices=["json", "text"], default="json")

    disable_parser = schedule_subparsers.add_parser("disable", help="Disable one schedule.")
    disable_parser.add_argument("--id", required=True, dest="job_id")
    disable_parser.add_argument("--format", choices=["json", "text"], default="json")

    enable_parser = schedule_subparsers.add_parser("enable", help="Enable one schedule.")
    enable_parser.add_argument("--id", required=True, dest="job_id")
    enable_parser.add_argument("--format", choices=["json", "text"], default="json")


def run_ingest_cli(args: argparse.Namespace) -> int:
    """Dispatch ingestion CLI subcommands."""
    if args.ingest_command == "agent-dry-run":
        return run_agent_dry_run_cli(args)
    if args.ingest_command == "queue":
        return run_review_queue_cli(args)
    if args.ingest_command == "retrieval":
        return run_retrieval_cli(args)
    if args.ingest_command == "schedule":
        return run_schedule_cli(args)
    if args.ingest_command == "worker":
        return run_worker_cli(args)
    raise ValueError("ingest command requires a subcommand")


def run_agent_dry_run_cli(
    args: argparse.Namespace,
    *,
    stdout: TextIO = sys.stdout,
    stderr: TextIO = sys.stderr,
) -> int:
    """Run the OpenAI Agents SDK dry-run CLI command."""
    try:
        if getattr(args, "list_providers", False):
            print(_render_provider_list(), file=stdout)
            return 0
        args._progress_stderr = stderr
        report = anyio.run(_run_agent_dry_run, args)
    except (OSError, ValueError, ValidationError, OpenAIAgentsDryRunError) as exc:
        print(str(exc), file=stderr)
        return 2

    rendered = _render_report(report, args.format)
    if args.output:
        Path(args.output).write_text(rendered, encoding="utf-8")
    else:
        print(rendered, file=stdout)
    return 0


def run_review_queue_cli(
    args: argparse.Namespace,
    *,
    stdout: TextIO = sys.stdout,
    stderr: TextIO = sys.stderr,
) -> int:
    """Run review queue CLI commands."""
    try:
        result = anyio.run(_run_review_queue_command, args)
    except (OSError, ValueError, ValidationError) as exc:
        print(str(exc), file=stderr)
        return 2

    structured = dict(result.structuredContent or {})
    rendered = _render_queue_result(structured, getattr(args, "format", "json"))
    print(rendered, file=stdout)
    return 2 if result.isError or structured.get("status") == "error" else 0


def run_retrieval_cli(
    args: argparse.Namespace,
    *,
    stdout: TextIO = sys.stdout,
    stderr: TextIO = sys.stderr,
) -> int:
    """Run external retrieval CLI commands."""
    try:
        result = anyio.run(_run_retrieval_command, args)
    except (OSError, ValueError, ValidationError) as exc:
        print(str(exc), file=stderr)
        return 2

    print(_render_retrieval_result(result, getattr(args, "format", "json")), file=stdout)
    return 2 if result.get("status") == "error" else 0


def run_schedule_cli(
    args: argparse.Namespace,
    *,
    stdout: TextIO = sys.stdout,
    stderr: TextIO = sys.stderr,
) -> int:
    """Run scheduled ingestion CLI commands."""
    try:
        result = anyio.run(_run_schedule_command, args)
    except KeyboardInterrupt:
        return 130
    except (OSError, ValueError, ValidationError, OpenAIAgentsDryRunError) as exc:
        print(str(exc), file=stderr)
        return 2

    print(_render_schedule_result(result, getattr(args, "format", "json")), file=stdout)
    return 2 if result.get("status") == "error" else 0


def run_worker_cli(
    args: argparse.Namespace,
    *,
    stdout: TextIO = sys.stdout,
    stderr: TextIO = sys.stderr,
) -> int:
    """Run the foreground conversation ingestion worker CLI."""
    try:
        result = anyio.run(_run_worker_command, args)
    except KeyboardInterrupt:
        return 130
    except (OSError, ValueError, ValidationError) as exc:
        print(str(exc), file=stderr)
        return 2
    for diagnostic in result.get("diagnostics", []):
        if isinstance(diagnostic, dict) and (diagnostic.get("level") or "warning") in {
            "warning",
            "error",
        }:
            level = diagnostic.get("level") or "warning"
            print(f"{level}: {diagnostic.get('code')}: {diagnostic.get('message')}", file=stderr)
    print(_render_worker_result(result, getattr(args, "format", "json")), file=stdout)
    return 2 if result.get("status") == "error" else 0


def main(argv: list[str] | None = None) -> int:
    """Run the standalone OpenAI Agents SDK ingestion wrapper."""
    parser = argparse.ArgumentParser(prog="agent_ingest_openai.py")
    add_agent_dry_run_arguments(parser)
    args = parser.parse_args(argv)
    return run_agent_dry_run_cli(args)


async def _run_agent_dry_run(
    args: argparse.Namespace,
    progress_callback: ProgressCallback | None = None,
) -> IngestionReport:
    if not 0 <= args.min_write_confidence <= 1:
        raise ValueError("--min-write-confidence must be between 0 and 1")
    provider_entry = _provider_entry_from_args(args)
    provider_config = _provider_config_from_args(args, provider_entry)
    model = _model_from_args(args, provider_entry)
    reviewer_configs = _reviewer_configs_from_args(args)
    high_priority_reviewers = _high_priority_reviewers_from_args(args)
    runtime_options = _runtime_options_from_args(args)
    progress_callback = progress_callback or _progress_callback_from_args(args)
    if provider_entry is not None and model is None and not os.environ.get("OPENAI_DEFAULT_MODEL"):
        raise ValueError(
            f"provider '{provider_entry.name}' requires --model or OPENAI_DEFAULT_MODEL"
        )
    ingestion_input = _load_ingestion_input(args)
    external_retrieval = _external_retrieval_from_args(args)
    if external_retrieval is not None and external_retrieval.has_errors:
        return _diagnostic_report_from_search(
            args,
            ingestion_input,
            None,
            external_retrieval=external_retrieval,
            diagnostics=external_retrieval.diagnostics,
        )
    search_result = _search_result_from_args(args)
    crawl_enabled = bool(getattr(args, "crawl", False))
    if crawl_enabled and not ingestion_input.urls:
        raise ValueError("--crawl requires at least one --url or input urls")
    if (
        not ingestion_input.evidence
        and not ingestion_input.urls
        and search_result is None
        and (external_retrieval is None or not external_retrieval.evidence)
    ):
        raise ValueError(
            "agent-dry-run requires --input evidence, --retrieval-input evidence, "
            "at least one --url, or --search-query"
        )
    if search_result is not None and search_result.has_errors:
        return _diagnostic_report_from_search(
            args,
            ingestion_input,
            search_result,
            external_retrieval=external_retrieval,
        )

    crawl_result = _crawl_result_from_args(args, ingestion_input) if crawl_enabled else None
    fetched = (
        crawl_result.evidence
        if crawl_result is not None
        else fetch_url_evidence_many(
            ingestion_input.urls,
            timeout_seconds=args.timeout_seconds,
            max_bytes=args.max_url_bytes,
        )
    )
    crawl_diagnostics = crawl_result.diagnostics if crawl_result is not None else []
    retrieval_evidence = external_retrieval.evidence if external_retrieval is not None else []
    retrieval_diagnostics = external_retrieval.diagnostics if external_retrieval is not None else []
    search_evidence = search_result.evidence if search_result is not None else []
    search_diagnostics = search_result.diagnostics if search_result is not None else []
    evidence = [*ingestion_input.evidence, *retrieval_evidence, *fetched, *search_evidence]
    if not evidence:
        diagnostics = [
            *retrieval_diagnostics,
            *crawl_diagnostics,
            *search_diagnostics,
            Diagnostic(
                level="error",
                code="INGESTION_EVIDENCE_MISSING",
                message="agent-dry-run requires evidence after URL fetch and search retrieval",
            ),
        ]
        return _diagnostic_report_from_search(
            args,
            ingestion_input,
            search_result,
            crawl_result=crawl_result,
            external_retrieval=external_retrieval,
            diagnostics=diagnostics,
        )
    effective_input = ingestion_input.model_copy(
        update={
            "evidence": evidence,
        }
    )
    store = create_relation_store(load_config())
    report = await run_openai_agents_ingestion(
        effective_input,
        store=store,
        model=model,
        reviewer_models=getattr(args, "reviewer_models", []),
        reviewer_configs=reviewer_configs,
        voting_policy=ReviewVotingPolicy(
            getattr(args, "voting_policy", ReviewVotingPolicy.RISK_TIERED.value)
        ),
        high_priority_reviewer_models=[
            *(getattr(args, "high_priority_reviewer_models", []) or []),
            *high_priority_reviewers,
        ],
        auto_write=args.auto_write,
        min_write_confidence=args.min_write_confidence,
        provider_config=provider_config,
        disable_tracing=bool(getattr(args, "disable_tracing", False)),
        runtime_options=runtime_options,
        progress_callback=progress_callback,
        canonicalize_preview=bool(getattr(args, "canonicalize_preview", False)),
    )
    metadata = dict(report.metadata)
    if external_retrieval is not None:
        metadata["external_retrieval"] = external_retrieval.metadata
    if crawl_result is not None:
        metadata["crawl_retrieval"] = crawl_result.metadata
    if search_result is not None:
        metadata["search_retrieval"] = search_result.metadata
    if metadata == report.metadata and not retrieval_diagnostics and not crawl_diagnostics:
        return report
    return report.model_copy(
        update={
            "diagnostics": [
                *retrieval_diagnostics,
                *crawl_diagnostics,
                *search_diagnostics,
                *report.diagnostics,
            ],
            "metadata": metadata,
        }
    )


async def _run_agent_dry_run_with_optional_progress(
    args: argparse.Namespace,
    progress_callback: ProgressCallback,
) -> IngestionReport:
    return await _run_agent_dry_run(args, progress_callback=progress_callback)


def _external_retrieval_from_args(args: argparse.Namespace) -> ExternalRetrievalConversion | None:
    path = getattr(args, "retrieval_input", None)
    if not path:
        return None
    return _load_external_retrieval(path)


def _search_result_from_args(args: argparse.Namespace) -> SearchRetrievalResult | None:
    queries = getattr(args, "search_queries", []) or []
    if not queries:
        return None
    options = SearchRetrievalOptions(
        queries=queries,
        provider=SearchProviderName(getattr(args, "search_provider", DEFAULT_SEARCH_PROVIDER)),
        limit=getattr(args, "search_limit", DEFAULT_SEARCH_LIMIT),
        timeout_seconds=getattr(
            args,
            "search_timeout_seconds",
            DEFAULT_SEARCH_TIMEOUT_SECONDS,
        ),
        include_domains=getattr(args, "search_include_domains", []) or [],
        exclude_domains=getattr(args, "search_exclude_domains", []) or [],
        api_key_env=getattr(args, "search_api_key_env", DEFAULT_SEARCH_API_KEY_ENV),
    )
    return retrieve_search_evidence(options)


def _crawl_result_from_args(
    args: argparse.Namespace,
    ingestion_input: IngestionInput,
) -> CrawlResult:
    options = CrawlOptions(
        seed_urls=ingestion_input.urls,
        max_depth=getattr(args, "crawl_max_depth", DEFAULT_CRAWL_MAX_DEPTH),
        max_pages=getattr(args, "crawl_max_pages", DEFAULT_CRAWL_MAX_PAGES),
        max_page_bytes=getattr(args, "crawl_max_page_bytes", DEFAULT_CRAWL_MAX_PAGE_BYTES),
        max_total_bytes=getattr(args, "crawl_max_total_bytes", DEFAULT_CRAWL_MAX_TOTAL_BYTES),
        timeout_seconds=getattr(args, "crawl_timeout_seconds", DEFAULT_CRAWL_TIMEOUT_SECONDS),
        allow_domains=getattr(args, "crawl_allow_domains", []) or [],
    )
    return crawl_url_evidence(options)


def _runtime_options_from_args(args: argparse.Namespace) -> LLMRuntimeOptions:
    return LLMRuntimeOptions(
        extractor_timeout_seconds=getattr(args, "extractor_timeout_seconds", 180),
        high_priority_reviewer_timeout_seconds=getattr(
            args,
            "high_priority_reviewer_timeout_seconds",
            180,
        ),
        reviewer_timeout_seconds=getattr(args, "reviewer_timeout_seconds", 120),
        extractor_max_tokens=getattr(args, "extractor_max_tokens", 4096),
        reviewer_max_tokens=getattr(args, "reviewer_max_tokens", 2048),
        progress_mode=getattr(args, "progress", "auto"),
    )


def _progress_callback_from_args(args: argparse.Namespace) -> ProgressCallback | None:
    if getattr(args, "progress", "auto") == "off":
        return None
    stream = getattr(args, "_progress_stderr", sys.stderr)

    def emit(event: dict[str, Any]) -> None:
        print(_format_progress_event(event), file=stream)

    return emit


def _format_progress_event(event: dict[str, Any]) -> str:
    stage = str(event.get("stage", "runtime"))
    label = str(event.get("reviewer_id") or event.get("label") or "")
    prefix = f"[{stage}]"
    if label:
        prefix = f"{prefix} {label}"
    event_name = str(event.get("event", event.get("status", "progress")))
    if event_name == "started":
        return f"{prefix} started"
    duration = event.get("duration_ms")
    duration_text = f" in {float(duration) / 1000:.1f}s" if isinstance(duration, int) else ""
    if event_name == "done":
        counts = []
        if "candidate_count" in event:
            counts.append(f"candidates={event['candidate_count']}")
        if "review_count" in event:
            counts.append(f"reviews={event['review_count']}")
        if "approved_count" in event:
            counts.append(f"approved={event['approved_count']}")
        if "queued_count" in event:
            counts.append(f"queued={event['queued_count']}")
        suffix = f" {' '.join(counts)}" if counts else ""
        return f"{prefix} done{duration_text}{suffix}"
    if event_name == "timeout":
        timeout = event.get("timeout_seconds")
        timeout_text = f" after {timeout:g}s" if isinstance(timeout, (int, float)) else ""
        return f"{prefix} timeout{timeout_text}"
    if event_name == "error":
        return f"{prefix} error {event.get('error_code', 'unknown')}"
    if event_name == "skipped":
        return f"{prefix} skipped {event.get('error_code', '')}".rstrip()
    return f"{prefix} {event_name}"


def _diagnostic_report_from_search(
    args: argparse.Namespace,
    ingestion_input: IngestionInput,
    search_result: SearchRetrievalResult | None,
    *,
    crawl_result: CrawlResult | None = None,
    external_retrieval: ExternalRetrievalConversion | None = None,
    diagnostics: list[Diagnostic] | None = None,
) -> IngestionReport:
    return IngestionReport(
        mode=IngestionMode.WRITE if getattr(args, "auto_write", False) else IngestionMode.DRY_RUN,
        diagnostics=diagnostics
        if diagnostics is not None
        else search_result.diagnostics
        if search_result is not None
        else [],
        metadata={
            "task": ingestion_input.task,
            "question": ingestion_input.question,
            "input_metadata": ingestion_input.metadata,
            "evidence_count": len(ingestion_input.evidence),
            "url_count": len(ingestion_input.urls),
            "auto_write_requested": bool(getattr(args, "auto_write", False)),
            "external_retrieval": (
                external_retrieval.metadata if external_retrieval is not None else None
            ),
            "crawl_retrieval": crawl_result.metadata if crawl_result is not None else None,
            "search_retrieval": search_result.metadata if search_result is not None else None,
        },
    )


async def _run_review_queue_command(args: argparse.Namespace) -> Any:
    store = create_relation_store(load_config())
    if args.queue_command == "list":
        filters = {
            key: value
            for key, value in {
                "status": args.status,
                "run_id": args.run_id,
                "candidate_id": args.candidate_id,
                "store_id": args.store_id,
                "context_id": args.context_id,
            }.items()
            if value is not None
        }
        return await call_tool(
            LIST_REVIEW_QUEUE,
            {
                "filter": filters,
                "limit": args.limit,
                "cursor": args.cursor,
            },
            store,
        )
    if args.queue_command == "commit":
        return await call_tool(
            COMMIT_REVIEWED_RELATIONS,
            {
                "ids": args.ids,
                "min_write_confidence": args.min_write_confidence,
            },
            store,
        )
    if args.queue_command == "resolve":
        return await call_tool(
            RESOLVE_REVIEW_QUEUE,
            {
                "ids": args.ids,
                "reason": args.reason,
            },
            store,
        )
    raise ValueError("ingest queue command requires a subcommand")


async def _run_retrieval_command(args: argparse.Namespace) -> dict[str, Any]:
    if args.retrieval_command == "validate":
        conversion = _load_external_retrieval(args.input)
        if conversion.has_errors:
            return _retrieval_diagnostic_result(conversion, status="error")
        if not conversion.candidates:
            return _retrieval_diagnostic_result(
                conversion,
                status="error",
                diagnostics=[
                    Diagnostic(
                        level="error",
                        code="RETRIEVAL_CANDIDATES_MISSING",
                        message="ingest retrieval validate requires retrieved candidates",
                    )
                ],
            )
        store = create_relation_store(load_config())
        result = await call_tool(
            VALIDATE_CANDIDATE_RELATIONS,
            {
                "candidates": [
                    candidate.model_dump(mode="json", exclude_none=True)
                    for candidate in conversion.candidates
                ],
                "reviews": [
                    review.model_dump(mode="json", exclude_none=True)
                    for review in conversion.reviews
                ],
                "min_write_confidence": args.min_write_confidence,
                "voting_policy": args.voting_policy,
                "high_priority_reviewer_models": args.high_priority_reviewer_models,
            },
            store,
        )
        structured = dict(result.structuredContent or {})
        structured = _append_retrieval_diagnostics(structured, conversion.diagnostics)
        structured = _apply_retrieval_provenance_gate(structured, conversion)
        structured["external_retrieval"] = conversion.metadata
        return structured
    raise ValueError("ingest retrieval command requires a subcommand")


async def _run_schedule_command(args: argparse.Namespace) -> dict[str, Any]:
    store = create_relation_store(load_config())
    command = args.schedule_command
    if command == "add":
        job = _scheduled_job_from_args(args)
        stored, updated = store.upsert_scheduled_ingestion_job(job)
        _record_schedule_audit(
            store,
            tool_name="auto_ingest.schedule.add",
            arguments={"job_id": stored.id, "updated": bool(updated)},
            result_status="ok",
            metadata=_schedule_job_audit_metadata(stored),
        )
        return {
            "status": "ok",
            "updated": bool(updated),
            "job": _scheduled_job_dump(stored),
        }
    if command == "list":
        job_filter = ScheduledIngestionJobFilter(
            status=ScheduledIngestionJobStatus(args.status) if args.status else None
        )
        jobs = store.list_scheduled_ingestion_jobs(job_filter, limit=args.limit)
        return {
            "status": "ok",
            "jobs": [_scheduled_job_dump(job) for job in jobs],
        }
    if command == "runs":
        run_filter = ScheduledIngestionRunFilter(
            job_id=args.job_id,
            status=ScheduledIngestionRunStatus(args.status) if args.status else None,
        )
        runs = store.list_scheduled_ingestion_runs(run_filter, limit=args.limit)
        return {
            "status": "ok",
            "runs": [_scheduled_run_dump(run) for run in runs],
        }
    if command == "run":
        selected_job = store.get_scheduled_ingestion_job(args.job_id)
        if selected_job is None:
            raise ValueError(f"scheduled ingestion job not found: {args.job_id}")
        run = await _run_scheduled_ingestion_job(
            selected_job,
            store,
            trigger=ScheduledIngestionRunTrigger.MANUAL,
        )
        return _scheduled_run_command_result([run])
    if command == "run-due":
        runs = await _run_due_scheduled_jobs(
            store,
            trigger=ScheduledIngestionRunTrigger.DUE,
            limit=args.limit,
        )
        return _scheduled_run_command_result(runs)
    if command == "worker":
        return await _run_schedule_worker(args, store)
    if command == "disable":
        job = _set_scheduled_job_enabled(store, args.job_id, enabled=False)
        _record_schedule_audit(
            store,
            tool_name="auto_ingest.schedule.disable",
            arguments={"job_id": job.id},
            result_status="ok",
            metadata=_schedule_job_audit_metadata(job),
        )
        return {"status": "ok", "job": _scheduled_job_dump(job)}
    if command == "enable":
        job = _set_scheduled_job_enabled(store, args.job_id, enabled=True)
        _record_schedule_audit(
            store,
            tool_name="auto_ingest.schedule.enable",
            arguments={"job_id": job.id},
            result_status="ok",
            metadata=_schedule_job_audit_metadata(job),
        )
        return {"status": "ok", "job": _scheduled_job_dump(job)}
    raise ValueError("ingest schedule command requires a subcommand")


async def _run_worker_command(args: argparse.Namespace) -> dict[str, Any]:
    provider_entry = _provider_entry_from_args(args)
    provider_config = _provider_config_from_args(args, provider_entry)
    model = _model_from_args(args, provider_entry)
    extraction_config = ExtractionModelConfig(
        model=model,
        provider_config=provider_config,
        max_tokens=args.extractor_max_tokens,
        timeout_seconds=args.extractor_timeout_seconds,
    )
    reviewer_configs = _reviewer_configs_from_args(args)
    high_priority_reviewers = _high_priority_reviewers_from_args(args)
    review_config = ReviewWorkerConfig(
        poll_seconds=args.review_poll_seconds,
        claim_limit=args.review_claim_limit,
        model=model,
        reviewer_models=args.reviewer_models,
        reviewer_configs=reviewer_configs,
        high_priority_reviewer_models=[
            *args.high_priority_reviewer_models,
            *high_priority_reviewers,
        ],
        voting_policy=ReviewVotingPolicy(args.voting_policy),
        provider_config=provider_config,
        disable_tracing=provider_config.disable_tracing if provider_config is not None else False,
        runtime_options=_runtime_options_from_args(args),
        auto_write=bool(args.allow_review_queue_writes),
        min_write_confidence=args.min_write_confidence,
        allow_single_reviewer_write=bool(args.allow_single_reviewer_write),
    )
    worker_config = IngestionWorkerConfig(
        poll_seconds=args.poll_seconds,
        queue_max_depth=args.max_queue_depth,
        claim_limit=args.claim_limit,
        max_merge_jobs=args.max_merge_jobs,
        extraction_config=extraction_config,
        review_config=review_config,
    )
    store = create_relation_store(load_config())
    result = await run_ingestion_worker(
        store,
        config=worker_config,
        max_jobs=args.max_jobs,
    )
    return result.to_dict()


async def _run_schedule_worker(args: argparse.Namespace, store: Any) -> dict[str, Any]:
    poll_seconds = float(args.poll_seconds)
    if poll_seconds <= 0 or poll_seconds > MAX_SCHEDULE_POLL_SECONDS:
        raise ValueError(f"--poll-seconds must be between 0 and {MAX_SCHEDULE_POLL_SECONDS}")
    max_runs = args.max_runs
    if max_runs is not None and max_runs < 0:
        raise ValueError("--max-runs must be non-negative")

    runs: list[ScheduledIngestionRun] = []
    iterations = 0
    stop_signals: list[str] = []
    async with anyio.create_task_group() as task_group:
        task_group.start_soon(
            _watch_schedule_worker_shutdown, task_group.cancel_scope, stop_signals
        )
        while max_runs is None or len(runs) < max_runs:
            remaining = None if max_runs is None else max_runs - len(runs)
            due_runs = await _run_due_scheduled_jobs(
                store,
                trigger=ScheduledIngestionRunTrigger.WORKER,
                limit=remaining or 50,
            )
            runs.extend(due_runs)
            iterations += 1
            if max_runs is not None and len(runs) >= max_runs:
                break
            await anyio.sleep(poll_seconds)
        task_group.cancel_scope.cancel()

    result = _scheduled_run_command_result(runs)
    result["iterations"] = iterations
    if stop_signals:
        result["status"] = "interrupted"
        result["stop_signal"] = stop_signals[0]
    return result


async def _watch_schedule_worker_shutdown(
    cancel_scope: anyio.CancelScope,
    stop_signals: list[str],
) -> None:
    """Cancel the foreground worker when the process receives a shutdown signal."""
    try:
        with anyio.open_signal_receiver(signal.SIGINT, signal.SIGTERM) as signals:
            async for signum in signals:
                stop_signals.append(signal.Signals(signum).name)
                cancel_scope.cancel()
                return
    except NotImplementedError:
        return


async def _run_due_scheduled_jobs(
    store: Any,
    *,
    trigger: ScheduledIngestionRunTrigger,
    limit: int,
) -> list[ScheduledIngestionRun]:
    if limit < 0:
        raise ValueError("--limit must be non-negative")
    now = datetime.now(UTC)
    jobs = store.list_scheduled_ingestion_jobs(
        ScheduledIngestionJobFilter(
            status=ScheduledIngestionJobStatus.ACTIVE,
            due_before=now.isoformat(),
        ),
        limit=limit,
    )
    runs: list[ScheduledIngestionRun] = []
    for job in jobs:
        if job_due(job, now=now):
            runs.append(await _run_scheduled_ingestion_job(job, store, trigger=trigger))
    return runs


async def _run_scheduled_ingestion_job(
    job: ScheduledIngestionJob,
    store: Any,
    *,
    trigger: ScheduledIngestionRunTrigger,
) -> ScheduledIngestionRun:
    previous_status = job.status
    run = ScheduledIngestionRun(
        job_id=job.id,
        trigger=trigger,
        attempt=job.state.retry_count + 1,
    )
    if previous_status == ScheduledIngestionJobStatus.RUNNING:
        diagnostics = [
            Diagnostic(
                level="warning",
                code="SCHEDULED_JOB_ALREADY_RUNNING",
                message="scheduled ingestion job is already running",
                related_ids=[job.id],
            )
        ]
        finished = run.model_copy(
            update={
                "status": ScheduledIngestionRunStatus.SKIPPED,
                "finished_at": datetime.now(UTC).isoformat(),
                "diagnostics": diagnostics,
                "metadata": _scheduled_run_metadata(job, None, diagnostics),
            }
        )
        store.append_scheduled_ingestion_run(finished)
        return finished

    safety_diagnostics = scheduled_write_diagnostics(job)
    blocking_diagnostics = [
        diagnostic for diagnostic in safety_diagnostics if diagnostic.level == "error"
    ]
    if blocking_diagnostics:
        finished = run.model_copy(
            update={
                "status": ScheduledIngestionRunStatus.SKIPPED,
                "finished_at": datetime.now(UTC).isoformat(),
                "diagnostics": safety_diagnostics,
                "metadata": _scheduled_run_metadata(job, None, safety_diagnostics),
            }
        )
        store.append_scheduled_ingestion_run(finished)
        state = next_state_for_skip(job, finished)
        store.update_scheduled_ingestion_job_state(
            job.id,
            state=state,
            status=ScheduledIngestionJobStatus.FAILED,
        )
        _record_schedule_audit(
            store,
            tool_name="auto_ingest.schedule.run",
            arguments={"job_id": job.id, "run_id": finished.id, "trigger": trigger.value},
            result_status="skipped",
            metadata=_scheduled_run_audit_metadata(job, finished),
        )
        return finished

    running_state = job.state.model_copy(update={"current_run_id": run.id})
    claimed_job = store.update_scheduled_ingestion_job_state(
        job.id,
        state=running_state,
        status=ScheduledIngestionJobStatus.RUNNING,
        expected_status=previous_status,
    )
    if claimed_job is None:
        diagnostics = [
            Diagnostic(
                level="warning",
                code="SCHEDULED_JOB_CLAIM_CONFLICT",
                message="scheduled ingestion job was already claimed by another worker",
                related_ids=[job.id],
            )
        ]
        finished = run.model_copy(
            update={
                "status": ScheduledIngestionRunStatus.SKIPPED,
                "finished_at": datetime.now(UTC).isoformat(),
                "diagnostics": diagnostics,
                "metadata": _scheduled_run_metadata(job, None, diagnostics),
            }
        )
        store.append_scheduled_ingestion_run(finished)
        _record_schedule_audit(
            store,
            tool_name="auto_ingest.schedule.run",
            arguments={"job_id": job.id, "run_id": finished.id, "trigger": trigger.value},
            result_status="skipped",
            metadata=_scheduled_run_audit_metadata(job, finished),
        )
        return finished

    store.append_scheduled_ingestion_run(run)
    progress_events: list[dict[str, Any]] = []

    def scheduled_progress(event: dict[str, Any]) -> None:
        progress_events.append(dict(event))
        metadata = {
            **_scheduled_run_metadata(job, None, safety_diagnostics),
            "current_stage": event.get("stage"),
            "current_status": event.get("status") or event.get("event"),
            "runtime_progress": progress_events[-20:],
        }
        store.append_scheduled_ingestion_run(run.model_copy(update={"metadata": metadata}))

    cancelled = False
    cancellation_error: BaseException | None = None
    cancelled_exc_class = anyio.get_cancelled_exc_class()
    try:
        report = await _run_agent_dry_run_with_optional_progress(
            _scheduled_job_args(job),
            scheduled_progress,
        )
        diagnostics = [*safety_diagnostics, *report.diagnostics]
        report_payload = report.model_dump(mode="json", exclude_none=True)
        report_path = write_scheduled_report(job, run, report_payload)
        final_status = (
            ScheduledIngestionRunStatus.FAILED
            if any(diagnostic.level == "error" for diagnostic in diagnostics)
            else ScheduledIngestionRunStatus.SUCCEEDED
        )
        finished = run.model_copy(
            update={
                "status": final_status,
                "finished_at": datetime.now(UTC).isoformat(),
                "report_run_id": report.run_id,
                "report_path": report_path,
                "diagnostics": diagnostics,
                "metadata": _scheduled_run_metadata(job, report_payload, diagnostics),
            }
        )
    except cancelled_exc_class as exc:
        cancelled = True
        cancellation_error = exc
        diagnostics = [
            *safety_diagnostics,
            Diagnostic(
                level="error",
                code="SCHEDULED_INGESTION_RUN_CANCELLED",
                message="scheduled ingestion run was cancelled during shutdown",
                related_ids=[job.id],
            ),
        ]
        finished = run.model_copy(
            update={
                "status": ScheduledIngestionRunStatus.FAILED,
                "finished_at": datetime.now(UTC).isoformat(),
                "diagnostics": diagnostics,
                "metadata": _scheduled_run_metadata(job, None, diagnostics),
            }
        )
    except Exception as exc:  # noqa: BLE001 - failures are persisted as run diagnostics.
        diagnostics = [
            *safety_diagnostics,
            Diagnostic(
                level="error",
                code="SCHEDULED_INGESTION_RUN_FAILED",
                message=f"scheduled ingestion run failed: {exc.__class__.__name__}",
                related_ids=[job.id],
            ),
        ]
        finished = run.model_copy(
            update={
                "status": ScheduledIngestionRunStatus.FAILED,
                "finished_at": datetime.now(UTC).isoformat(),
                "diagnostics": diagnostics,
                "metadata": _scheduled_run_metadata(job, None, diagnostics),
            }
        )

    store.append_scheduled_ingestion_run(finished)
    if finished.status == ScheduledIngestionRunStatus.SUCCEEDED:
        next_state = next_state_for_success(job, finished)
        next_status = (
            ScheduledIngestionJobStatus.DISABLED
            if previous_status == ScheduledIngestionJobStatus.DISABLED
            else ScheduledIngestionJobStatus.ACTIVE
        )
    else:
        next_status, next_state = next_state_for_failure(job, finished)
        if previous_status == ScheduledIngestionJobStatus.DISABLED:
            next_status = ScheduledIngestionJobStatus.DISABLED
    store.update_scheduled_ingestion_job_state(job.id, state=next_state, status=next_status)
    _record_schedule_audit(
        store,
        tool_name="auto_ingest.schedule.run",
        arguments={"job_id": job.id, "run_id": finished.id, "trigger": trigger.value},
        result_status="cancelled" if cancelled else finished.status.value,
        metadata=_scheduled_run_audit_metadata(job, finished),
    )
    if cancelled:
        if cancellation_error is not None:
            raise cancellation_error
        raise RuntimeError("scheduled ingestion run was cancelled")
    return finished


def _scheduled_job_from_args(args: argparse.Namespace) -> ScheduledIngestionJob:
    source_config = ScheduledIngestionSourceConfig(
        input_path=args.input,
        urls=getattr(args, "url", []) or [],
        retrieval_input_path=getattr(args, "retrieval_input", None),
        task=args.task,
        question=args.question,
        timeout_seconds=args.timeout_seconds,
        max_url_bytes=args.max_url_bytes,
        search_queries=getattr(args, "search_queries", []) or [],
        search_provider=args.search_provider,
        search_limit=args.search_limit,
        search_timeout_seconds=args.search_timeout_seconds,
        search_include_domains=getattr(args, "search_include_domains", []) or [],
        search_exclude_domains=getattr(args, "search_exclude_domains", []) or [],
        search_api_key_env=args.search_api_key_env,
        crawl=bool(getattr(args, "crawl", False)),
        crawl_max_depth=args.crawl_max_depth,
        crawl_max_pages=args.crawl_max_pages,
        crawl_max_page_bytes=args.crawl_max_page_bytes,
        crawl_max_total_bytes=args.crawl_max_total_bytes,
        crawl_timeout_seconds=args.crawl_timeout_seconds,
        crawl_allow_domains=getattr(args, "crawl_allow_domains", []) or [],
    )
    provider_config = ScheduledIngestionProviderConfig(
        model=args.model,
        provider=args.provider,
        base_url=args.base_url,
        api_key_env=args.api_key_env,
        provider_headers=getattr(args, "provider_header", []) or [],
        provider_thinking=getattr(args, "provider_thinking", None),
        provider_reasoning_effort=getattr(args, "provider_reasoning_effort", None),
        disable_tracing=bool(args.disable_tracing),
        reviewer_models=getattr(args, "reviewer_models", []) or [],
        reviewers=getattr(args, "reviewers", []) or [],
        voting_policy=ReviewVotingPolicy(args.voting_policy),
        high_priority_reviewer_models=getattr(args, "high_priority_reviewer_models", []) or [],
        high_priority_reviewers=getattr(args, "high_priority_reviewers", []) or [],
    )
    _validate_provider_reviewer_specs(provider_config)
    runtime_config = ScheduledIngestionRuntimeConfig(
        extractor_timeout_seconds=getattr(args, "extractor_timeout_seconds", 180),
        high_priority_reviewer_timeout_seconds=getattr(
            args,
            "high_priority_reviewer_timeout_seconds",
            180,
        ),
        reviewer_timeout_seconds=getattr(args, "reviewer_timeout_seconds", 120),
        extractor_max_tokens=getattr(args, "extractor_max_tokens", 4096),
        reviewer_max_tokens=getattr(args, "reviewer_max_tokens", 2048),
        progress=getattr(args, "progress", "auto"),
    )
    write_config = ScheduledIngestionWriteConfig(
        auto_write=bool(args.auto_write),
        allow_scheduled_writes=bool(args.allow_scheduled_writes),
        allow_single_reviewer_write=bool(args.allow_single_reviewer_write),
        min_write_confidence=args.min_write_confidence,
        report_dir=args.report_dir,
    )
    retry_policy = ScheduledIngestionRetryPolicy(
        max_retries=args.max_retries,
        retry_backoff_seconds=args.retry_backoff_seconds,
    )
    job = ScheduledIngestionJob(
        name=args.name,
        cron=args.cron,
        timezone=args.timezone,
        source_config=source_config,
        provider_config=provider_config,
        runtime_config=runtime_config,
        write_config=write_config,
        retry_policy=retry_policy,
        state=ScheduledIngestionState(
            next_run_at=next_cron_run(args.cron, args.timezone, after=datetime.now(UTC))
        ),
    )
    diagnostics = scheduled_write_diagnostics(job)
    for diagnostic in diagnostics:
        if diagnostic.level == "error":
            raise ValueError(diagnostic.message)
    return job


def _scheduled_job_args(job: ScheduledIngestionJob) -> argparse.Namespace:
    source = job.source_config
    provider = job.provider_config
    runtime = job.runtime_config
    write = job.write_config
    return argparse.Namespace(
        input=source.input_path,
        retrieval_input=source.retrieval_input_path,
        url=list(source.urls),
        task=source.task,
        question=source.question,
        model=provider.model,
        reviewer_models=list(provider.reviewer_models),
        reviewers=list(provider.reviewers),
        voting_policy=provider.voting_policy.value,
        high_priority_reviewer_models=list(provider.high_priority_reviewer_models),
        high_priority_reviewers=list(provider.high_priority_reviewers),
        provider=provider.provider,
        list_providers=False,
        base_url=provider.base_url,
        api_key_env=provider.api_key_env,
        provider_header=list(provider.provider_headers),
        provider_thinking=provider.provider_thinking,
        provider_reasoning_effort=provider.provider_reasoning_effort,
        disable_tracing=provider.disable_tracing,
        extractor_timeout_seconds=runtime.extractor_timeout_seconds,
        high_priority_reviewer_timeout_seconds=runtime.high_priority_reviewer_timeout_seconds,
        reviewer_timeout_seconds=runtime.reviewer_timeout_seconds,
        extractor_max_tokens=runtime.extractor_max_tokens,
        reviewer_max_tokens=runtime.reviewer_max_tokens,
        progress=runtime.progress,
        auto_write=write.auto_write,
        min_write_confidence=write.min_write_confidence,
        format="json",
        output=None,
        timeout_seconds=source.timeout_seconds,
        max_url_bytes=source.max_url_bytes,
        search_queries=list(source.search_queries),
        search_provider=source.search_provider,
        search_limit=source.search_limit,
        search_timeout_seconds=source.search_timeout_seconds,
        search_include_domains=list(source.search_include_domains),
        search_exclude_domains=list(source.search_exclude_domains),
        search_api_key_env=source.search_api_key_env,
        crawl=source.crawl,
        crawl_max_depth=source.crawl_max_depth,
        crawl_max_pages=source.crawl_max_pages,
        crawl_max_page_bytes=source.crawl_max_page_bytes,
        crawl_max_total_bytes=source.crawl_max_total_bytes,
        crawl_timeout_seconds=source.crawl_timeout_seconds,
        crawl_allow_domains=list(source.crawl_allow_domains),
    )


def _set_scheduled_job_enabled(
    store: Any,
    job_id: str,
    *,
    enabled: bool,
) -> ScheduledIngestionJob:
    job = store.get_scheduled_ingestion_job(job_id)
    if job is None:
        raise ValueError(f"scheduled ingestion job not found: {job_id}")
    status = ScheduledIngestionJobStatus.ACTIVE if enabled else ScheduledIngestionJobStatus.DISABLED
    state = job.state
    if enabled:
        state = state.model_copy(
            update={"next_run_at": next_cron_run(job.cron, job.timezone, after=datetime.now(UTC))}
        )
    updated = store.update_scheduled_ingestion_job_state(job.id, state=state, status=status)
    if updated is None:
        raise ValueError(f"scheduled ingestion job not found: {job_id}")
    return updated


def _scheduled_run_command_result(runs: list[ScheduledIngestionRun]) -> dict[str, Any]:
    has_problem = any(
        run.status in {ScheduledIngestionRunStatus.FAILED, ScheduledIngestionRunStatus.SKIPPED}
        for run in runs
    )
    return {
        "status": "warning" if has_problem else "ok",
        "run_count": len(runs),
        "runs": [_scheduled_run_dump(run) for run in runs],
    }


def _scheduled_run_metadata(
    job: ScheduledIngestionJob,
    report_payload: dict[str, Any] | None,
    diagnostics: list[Diagnostic],
) -> dict[str, Any]:
    metadata = {
        "job_id": job.id,
        "job_name": job.name,
        "auto_write": job.write_config.auto_write,
        "reviewer_count": scheduled_reviewer_count(job.provider_config),
        "voting_policy": job.provider_config.voting_policy.value,
        "allow_single_reviewer_write": job.write_config.allow_single_reviewer_write,
        "diagnostic_count": len(diagnostics),
    }
    if report_payload is not None:
        metadata.update(
            {
                "written_count": len(report_payload.get("written_relation_ids", [])),
                "queued_count": sum(
                    1
                    for item in report_payload.get("gate_results", [])
                    if isinstance(item, dict) and item.get("action") == "queue"
                ),
            }
        )
        report_metadata = report_payload.get("metadata")
        if isinstance(report_metadata, dict) and isinstance(
            report_metadata.get("runtime_trace"),
            list,
        ):
            metadata["runtime_trace"] = report_metadata["runtime_trace"]
    return metadata


def _scheduled_run_audit_metadata(
    job: ScheduledIngestionJob,
    run: ScheduledIngestionRun,
) -> dict[str, Any]:
    metadata = dict(run.metadata)
    metadata.update(
        {
            "job_id": job.id,
            "run_id": run.id,
            "status": run.status.value,
            "report_path": run.report_path,
        }
    )
    return metadata


def _schedule_job_audit_metadata(job: ScheduledIngestionJob) -> dict[str, Any]:
    return {
        "job_id": job.id,
        "status": job.status.value,
        "next_run_at": job.state.next_run_at,
        "reviewer_count": scheduled_reviewer_count(job.provider_config),
        "voting_policy": job.provider_config.voting_policy.value,
        "allow_single_reviewer_write": job.write_config.allow_single_reviewer_write,
    }


def _record_schedule_audit(
    store: Any,
    *,
    tool_name: str,
    arguments: dict[str, Any],
    result_status: str,
    metadata: dict[str, Any],
) -> None:
    store.record_audit(
        event_type="scheduled_ingestion",
        tool_name=tool_name,
        arguments=arguments,
        result_status=result_status,
        metadata=metadata,
    )


def _scheduled_job_dump(job: ScheduledIngestionJob) -> dict[str, Any]:
    return job.model_dump(mode="json", exclude_none=True)


def _scheduled_run_dump(run: ScheduledIngestionRun) -> dict[str, Any]:
    return run.model_dump(mode="json", exclude_none=True)


def _load_external_retrieval(path: str) -> ExternalRetrievalConversion:
    input_path = Path(path)
    size = input_path.stat().st_size
    if size > MAX_EXTERNAL_RETRIEVAL_INPUT_BYTES:
        raise ValueError(f"retrieval input JSON exceeds {MAX_EXTERNAL_RETRIEVAL_INPUT_BYTES} bytes")
    data = json.loads(input_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("retrieval input JSON must be an object")
    batch = ExternalRetrievalBatch.model_validate(data)
    return convert_external_retrieval_batch(batch)


def _retrieval_diagnostic_result(
    conversion: ExternalRetrievalConversion,
    *,
    status: str,
    diagnostics: list[Diagnostic] | None = None,
) -> dict[str, Any]:
    rendered_diagnostics = diagnostics if diagnostics is not None else conversion.diagnostics
    return {
        "status": status,
        "persisted": False,
        "candidate_count": len(conversion.candidates),
        "approved_count": 0,
        "queued_count": 0,
        "rejected_count": 0,
        "gate_results": [],
        "approved_relations": [],
        "review_aggregation": {},
        "external_retrieval": conversion.metadata,
        "diagnostics": [item.model_dump(mode="json") for item in rendered_diagnostics],
        "reasoning": {},
        "graph_stats": {},
        "trace": ["Validated external retrieval input without persisting graph state."],
    }


def _apply_retrieval_provenance_gate(
    structured: dict[str, Any],
    conversion: ExternalRetrievalConversion,
) -> dict[str, Any]:
    missing_ids = set(conversion.missing_candidate_provenance_ids)
    if not missing_ids:
        return structured

    gate_results = []
    for item in structured.get("gate_results", []):
        if not isinstance(item, dict):
            continue
        item = dict(item)
        if item.get("candidate_id") in missing_ids and item.get("action") == "auto_write":
            item = {
                **item,
                "action": "queue",
                "reasons": ["retrieval provenance missing"],
            }
        gate_results.append(item)

    approved_relations: list[dict[str, Any]] = []
    for item in structured.get("approved_relations", []):
        if not isinstance(item, dict):
            continue
        provenance = item.get("provenance")
        if isinstance(provenance, dict) and provenance.get("candidate_id") in missing_ids:
            continue
        approved_relations.append(dict(item))
    diagnostics = list(structured.get("diagnostics", []))
    diagnostics.append(
        Diagnostic(
            level="warning",
            code="RETRIEVAL_PROVENANCE_MISSING",
            message=(
                "retrieved candidate requires retriever_name and original_url or source_document_id"
            ),
            related_ids=sorted(missing_ids),
        ).model_dump(mode="json")
    )

    queued_count = sum(1 for item in gate_results if item.get("action") == "queue")
    rejected_count = sum(1 for item in gate_results if item.get("action") == "reject")
    status = "error" if structured.get("status") == "error" else "warning"
    return {
        **structured,
        "status": status,
        "approved_count": len(approved_relations),
        "queued_count": queued_count,
        "rejected_count": rejected_count,
        "gate_results": gate_results,
        "approved_relations": approved_relations,
        "diagnostics": diagnostics,
    }


def _append_retrieval_diagnostics(
    structured: dict[str, Any],
    diagnostics: list[Diagnostic],
) -> dict[str, Any]:
    if not diagnostics:
        return structured
    rendered = [item.model_dump(mode="json") for item in diagnostics]
    status = structured.get("status")
    next_status = "error" if status == "error" else "warning"
    return {
        **structured,
        "status": next_status,
        "diagnostics": [*structured.get("diagnostics", []), *rendered],
    }


def _provider_config_from_args(
    args: argparse.Namespace,
    provider_entry: ProviderRegistryEntry | None = None,
) -> OpenAICompatibleProviderConfig | None:
    provider_entry = provider_entry or _provider_entry_from_args(args)
    base_url = getattr(args, "base_url", None) or (
        provider_entry.base_url if provider_entry is not None else None
    )
    api_key_env = getattr(args, "api_key_env", None) or (
        provider_entry.api_key_env if provider_entry is not None else None
    )
    headers = getattr(args, "provider_header", []) or []
    if not base_url:
        if api_key_env:
            raise ValueError("--api-key-env requires --base-url")
        if headers:
            raise ValueError("--provider-header requires --base-url or --provider")
        return None
    if not api_key_env:
        raise ValueError("--api-key-env is required when --base-url is set")
    _validate_provider_base_url(base_url)
    structured_output_mode = (
        provider_entry.structured_output_mode
        if provider_entry is not None
        else ProviderStructuredOutputMode.AGENT_SCHEMA
    )
    provider_thinking = getattr(args, "provider_thinking", None)
    provider_reasoning_effort = getattr(args, "provider_reasoning_effort", None)
    if (
        provider_thinking is not None or provider_reasoning_effort is not None
    ) and structured_output_mode is not ProviderStructuredOutputMode.JSON_OBJECT:
        raise ValueError(
            "--provider-thinking and --provider-reasoning-effort require a JSON Object "
            "provider such as --provider deepseek"
        )
    if (
        provider_thinking is not None
        and provider_entry is not None
        and "thinking" not in provider_entry.extra_body
    ):
        raise ValueError(
            f"--provider-thinking is not supported by provider '{provider_entry.name}'"
        )
    if (
        provider_reasoning_effort is not None
        and provider_entry is not None
        and provider_entry.reasoning_effort is None
    ):
        raise ValueError(
            f"--provider-reasoning-effort is not supported by provider '{provider_entry.name}'"
        )
    extra_body = dict(provider_entry.extra_body) if provider_entry is not None else {}
    if provider_thinking is not None:
        extra_body["thinking"] = {"type": provider_thinking}
    return OpenAICompatibleProviderConfig(
        base_url=base_url,
        api_key_env=api_key_env,
        default_headers=_parse_provider_headers(headers),
        # OpenAI-compatible providers do not participate in OpenAI tracing.
        # Keep this disabled by default to avoid sending third-party runs to tracing.
        disable_tracing=provider_entry.tracing_disabled if provider_entry is not None else True,
        structured_output_mode=structured_output_mode,
        reasoning_effort=(
            provider_reasoning_effort
            or (provider_entry.reasoning_effort if provider_entry is not None else None)
        ),
        extra_body=extra_body,
    )


def _provider_config_from_entry(
    provider_entry: ProviderRegistryEntry,
) -> OpenAICompatibleProviderConfig:
    config = _provider_config_from_args(
        argparse.Namespace(
            base_url=None,
            api_key_env=None,
            provider_header=[],
            provider_thinking=None,
            provider_reasoning_effort=None,
        ),
        provider_entry,
    )
    if config is None:
        raise ValueError(f"provider '{provider_entry.name}' requires provider configuration")
    return config


def _validate_provider_reviewer_specs(provider_config: ScheduledIngestionProviderConfig) -> None:
    for spec in [*provider_config.reviewers, *provider_config.high_priority_reviewers]:
        _parse_provider_reviewer_spec(spec)


def _reviewer_configs_from_args(args: argparse.Namespace) -> list[ReviewerModelConfig]:
    configs: list[ReviewerModelConfig] = []
    seen: set[str] = set()
    for spec in [
        *(getattr(args, "reviewers", []) or []),
        *(getattr(args, "high_priority_reviewers", []) or []),
    ]:
        provider_entry, model, reviewer_id = _parse_provider_reviewer_spec(spec)
        if reviewer_id in seen:
            continue
        seen.add(reviewer_id)
        configs.append(
            ReviewerModelConfig(
                reviewer_id=reviewer_id,
                model=model,
                provider_name=provider_entry.name,
                provider_config=_provider_config_from_entry(provider_entry),
            )
        )
    return configs


def _high_priority_reviewers_from_args(args: argparse.Namespace) -> list[str]:
    ids: list[str] = []
    seen: set[str] = set()
    for spec in getattr(args, "high_priority_reviewers", []) or []:
        _provider_entry, _model, reviewer_id = _parse_provider_reviewer_spec(spec)
        if reviewer_id in seen:
            continue
        seen.add(reviewer_id)
        ids.append(reviewer_id)
    return ids


def _parse_provider_reviewer_spec(
    spec: str,
) -> tuple[ProviderRegistryEntry, str, str]:
    stripped = spec.strip()
    if ":" not in stripped:
        raise ValueError("provider-qualified reviewer must use PROVIDER:MODEL")
    provider_name, model = stripped.split(":", 1)
    provider_name = provider_name.strip()
    model = model.strip()
    if not provider_name or not model:
        raise ValueError("provider-qualified reviewer must include provider and model")
    provider_entry = get_provider_entry(provider_name)
    reviewer_id = f"{provider_entry.name}:{model}"
    return provider_entry, model, reviewer_id


def _provider_entry_from_args(args: argparse.Namespace) -> ProviderRegistryEntry | None:
    provider_name = getattr(args, "provider", None)
    if provider_name is None:
        return None
    return get_provider_entry(provider_name)


def _model_from_args(
    args: argparse.Namespace,
    provider_entry: ProviderRegistryEntry | None = None,
) -> str | None:
    model = getattr(args, "model", None)
    if model is not None:
        return model
    provider_entry = provider_entry or _provider_entry_from_args(args)
    if provider_entry is None:
        return None
    return provider_entry.default_model


def _validate_provider_base_url(base_url: str) -> None:
    parsed = urlparse(base_url)
    if parsed.scheme != "https" or not parsed.netloc:
        raise ValueError("--base-url must be an https URL")


def _parse_provider_headers(values: list[str]) -> dict[str, str]:
    headers: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise ValueError("--provider-header must use KEY=VALUE")
        key, header_value = value.split("=", 1)
        key = key.strip()
        header_value = header_value.strip()
        if not key or not header_value:
            raise ValueError("--provider-header must use non-empty KEY=VALUE")
        if fullmatch(_HTTP_HEADER_KEY_PATTERN, key) is None:
            raise ValueError("--provider-header key must be a valid HTTP header token")
        if "\r" in header_value or "\n" in header_value:
            raise ValueError("--provider-header value must not contain newlines")
        headers[key] = header_value
    return headers


def _render_provider_list() -> str:
    # This explicit CLI listing may show env var names and base URLs,
    # but it must never include actual API key values.
    rows = [
        (
            "provider\tbase_url\tapi_key_env\tdefault_model\tstructured_output_mode"
            "\tsupported_models\treasoning_effort\tdocs_url\tnotes"
        ),
        *[
            "\t".join(
                [
                    entry.name,
                    entry.base_url,
                    entry.api_key_env,
                    entry.default_model or "-",
                    entry.structured_output_mode.value,
                    ",".join(entry.supported_models) or "-",
                    entry.reasoning_effort or "-",
                    entry.docs_url,
                    entry.notes,
                ]
            )
            for entry in list_provider_entries()
        ],
    ]
    return "\n".join(rows)


def _load_ingestion_input(args: argparse.Namespace) -> IngestionInput:
    data: dict[str, Any] = {}
    if args.input:
        data = json.loads(Path(args.input).read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("input JSON must be an object")
    urls = [*data.get("urls", []), *args.url]
    if args.task is not None:
        data["task"] = args.task
    if args.question is not None:
        data["question"] = args.question
    data["urls"] = urls
    return IngestionInput.model_validate(data)


def _render_report(report: IngestionReport, output_format: str) -> str:
    if output_format == "json":
        return json.dumps(report.model_dump(mode="json", exclude_none=True), ensure_ascii=False)
    approved = len(report.approved_relations)
    queued = sum(1 for item in report.gate_results if item.action == "queue")
    rejected = sum(1 for item in report.gate_results if item.action == "reject")
    return (
        f"run_id: {report.run_id}\n"
        f"mode: {report.mode}\n"
        f"candidates: {len(report.candidates)}\n"
        f"approved_relations: {approved}\n"
        f"written_relations: {len(report.written_relation_ids)}\n"
        f"queued: {queued}\n"
        f"rejected: {rejected}"
    )


def _render_queue_result(result: dict[str, Any], output_format: str) -> str:
    if output_format == "json":
        return json.dumps(result, ensure_ascii=False)
    if "records" in result:
        rows = [
            f"{record.get('id')}\t{record.get('status')}\t"
            f"{record.get('candidate', {}).get('source')} -> "
            f"{record.get('candidate', {}).get('target')}"
            for record in result.get("records", [])
            if isinstance(record, dict)
        ]
        return "\n".join(rows) if rows else "no review queue records"
    if "committed_count" in result:
        return (
            f"status: {result.get('status')}\n"
            f"committed: {result.get('committed_count')}\n"
            f"relations: {len(result.get('relation_ids', []))}"
        )
    if "resolved_count" in result:
        return f"status: {result.get('status')}\nresolved: {result.get('resolved_count')}"
    return f"status: {result.get('status')}"


def _render_retrieval_result(result: dict[str, Any], output_format: str) -> str:
    if output_format == "json":
        return json.dumps(result, ensure_ascii=False)
    return (
        f"status: {result.get('status')}\n"
        f"candidates: {result.get('candidate_count', 0)}\n"
        f"approved: {result.get('approved_count', 0)}\n"
        f"queued: {result.get('queued_count', 0)}\n"
        f"rejected: {result.get('rejected_count', 0)}"
    )


def _render_schedule_result(result: dict[str, Any], output_format: str) -> str:
    if output_format == "json":
        return json.dumps(result, ensure_ascii=False)
    if "job" in result and isinstance(result["job"], dict):
        job = result["job"]
        state = job.get("state", {})
        return (
            f"status: {result.get('status')}\n"
            f"job: {job.get('id')}\n"
            f"job_status: {job.get('status')}\n"
            f"next_run_at: {state.get('next_run_at')}"
        )
    if "jobs" in result:
        rows = [
            f"{job.get('id')}\t{job.get('status')}\t"
            f"{job.get('state', {}).get('next_run_at')}\t{job.get('name')}"
            for job in result.get("jobs", [])
            if isinstance(job, dict)
        ]
        return "\n".join(rows) if rows else "no scheduled ingestion jobs"
    if "runs" in result:
        rows = [
            f"{run.get('id')}\t{run.get('job_id')}\t{run.get('status')}\t{run.get('started_at')}"
            for run in result.get("runs", [])
            if isinstance(run, dict)
        ]
        return "\n".join(rows) if rows else "no scheduled ingestion runs"
    return f"status: {result.get('status')}"


def _render_worker_result(result: dict[str, Any], output_format: str) -> str:
    if output_format == "json":
        return json.dumps(result, ensure_ascii=False)
    return (
        f"status: {result.get('status')}\n"
        f"claimed: {len(result.get('claimed_job_ids', []))}\n"
        f"processed: {len(result.get('processed_job_ids', []))}\n"
        f"queued: {len(result.get('queued_record_ids', []))}\n"
        f"reviewed: {len(result.get('reviewed_record_ids', []))}\n"
        f"committed: {len(result.get('committed_record_ids', []))}\n"
        f"resolved: {len(result.get('resolved_record_ids', []))}\n"
        f"dropped: {len(result.get('dropped_job_ids', []))}\n"
        f"failed: {len(result.get('failed_job_ids', []))}"
    )
