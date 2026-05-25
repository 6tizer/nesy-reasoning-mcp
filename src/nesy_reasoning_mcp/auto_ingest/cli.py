"""CLI helpers for Agent SDK dry-run ingestion."""

from __future__ import annotations

import argparse
import json
import os
import sys
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
from nesy_reasoning_mcp.auto_ingest.fetcher import (
    DEFAULT_FETCH_TIMEOUT_SECONDS,
    DEFAULT_MAX_FETCH_BYTES,
    fetch_url_evidence_many,
)
from nesy_reasoning_mcp.auto_ingest.openai_agents import (
    OpenAIAgentsDryRunError,
    OpenAICompatibleProviderConfig,
    run_openai_agents_ingestion,
)
from nesy_reasoning_mcp.auto_ingest.providers import (
    ProviderRegistryEntry,
    get_provider_entry,
    list_provider_entries,
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
from nesy_reasoning_mcp.config import load_config
from nesy_reasoning_mcp.schemas import Diagnostic
from nesy_reasoning_mcp.store import create_relation_store
from nesy_reasoning_mcp.tool_names import (
    COMMIT_REVIEWED_RELATIONS,
    LIST_REVIEW_QUEUE,
    RESOLVE_REVIEW_QUEUE,
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


def add_agent_dry_run_arguments(parser: argparse.ArgumentParser) -> None:
    """Add shared arguments for the OpenAI Agents SDK dry-run command."""
    parser.add_argument("--input", default=None, help="JSON input file path.")
    parser.add_argument("--url", action="append", default=[], help="Explicit HTTP(S) URL source.")
    parser.add_argument("--task", default=None, help="Optional extraction task.")
    parser.add_argument("--question", default=None, help="Optional question to answer.")
    parser.add_argument("--model", default=None, help="OpenAI Agents SDK model override.")
    parser.add_argument(
        "--reviewer-model",
        action="append",
        default=[],
        dest="reviewer_models",
        help="Reviewer model override. May be repeated for multi-reviewer voting.",
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
        help="Reviewer model whose reject or needs_human/downgrade vote has priority.",
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
        "--disable-tracing",
        action="store_true",
        help="Disable OpenAI Agents SDK tracing for this run.",
    )
    parser.add_argument(
        "--auto-write",
        action="store_true",
        help="Persist gate-approved relations with safe write checks.",
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


def run_ingest_cli(args: argparse.Namespace) -> int:
    """Dispatch ingestion CLI subcommands."""
    if args.ingest_command == "agent-dry-run":
        return run_agent_dry_run_cli(args)
    if args.ingest_command == "queue":
        return run_review_queue_cli(args)
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


def main(argv: list[str] | None = None) -> int:
    """Run the standalone OpenAI Agents SDK ingestion wrapper."""
    parser = argparse.ArgumentParser(prog="agent_ingest_openai.py")
    add_agent_dry_run_arguments(parser)
    args = parser.parse_args(argv)
    return run_agent_dry_run_cli(args)


async def _run_agent_dry_run(args: argparse.Namespace) -> IngestionReport:
    if not 0 <= args.min_write_confidence <= 1:
        raise ValueError("--min-write-confidence must be between 0 and 1")
    provider_entry = _provider_entry_from_args(args)
    provider_config = _provider_config_from_args(args, provider_entry)
    model = _model_from_args(args, provider_entry)
    if provider_entry is not None and model is None and not os.environ.get("OPENAI_DEFAULT_MODEL"):
        raise ValueError(
            f"provider '{provider_entry.name}' requires --model or OPENAI_DEFAULT_MODEL"
        )
    ingestion_input = _load_ingestion_input(args)
    search_result = _search_result_from_args(args)
    crawl_enabled = bool(getattr(args, "crawl", False))
    if crawl_enabled and not ingestion_input.urls:
        raise ValueError("--crawl requires at least one --url or input urls")
    if not ingestion_input.evidence and not ingestion_input.urls and search_result is None:
        raise ValueError(
            "agent-dry-run requires --input evidence, at least one --url, or --search-query"
        )
    if search_result is not None and search_result.has_errors:
        return _diagnostic_report_from_search(args, ingestion_input, search_result)

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
    search_evidence = search_result.evidence if search_result is not None else []
    search_diagnostics = search_result.diagnostics if search_result is not None else []
    evidence = [*ingestion_input.evidence, *fetched, *search_evidence]
    if not evidence:
        diagnostics = [
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
        voting_policy=ReviewVotingPolicy(
            getattr(args, "voting_policy", ReviewVotingPolicy.RISK_TIERED.value)
        ),
        high_priority_reviewer_models=getattr(args, "high_priority_reviewer_models", []),
        auto_write=args.auto_write,
        min_write_confidence=args.min_write_confidence,
        provider_config=provider_config,
        disable_tracing=bool(getattr(args, "disable_tracing", False)),
    )
    if search_result is None:
        if crawl_result is None:
            return report
        return report.model_copy(
            update={
                "diagnostics": [*crawl_diagnostics, *report.diagnostics],
                "metadata": {
                    **report.metadata,
                    "crawl_retrieval": crawl_result.metadata,
                },
            }
        )
    return report.model_copy(
        update={
            "diagnostics": [*crawl_diagnostics, *search_diagnostics, *report.diagnostics],
            "metadata": {
                **report.metadata,
                **({"crawl_retrieval": crawl_result.metadata} if crawl_result is not None else {}),
                "search_retrieval": search_result.metadata,
            },
        }
    )


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


def _diagnostic_report_from_search(
    args: argparse.Namespace,
    ingestion_input: IngestionInput,
    search_result: SearchRetrievalResult | None,
    *,
    crawl_result: CrawlResult | None = None,
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
    return OpenAICompatibleProviderConfig(
        base_url=base_url,
        api_key_env=api_key_env,
        default_headers=_parse_provider_headers(headers),
        # OpenAI-compatible providers do not participate in OpenAI tracing.
        # Keep this disabled by default to avoid sending third-party runs to tracing.
        disable_tracing=provider_entry.tracing_disabled if provider_entry is not None else True,
    )


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
        "provider\tbase_url\tapi_key_env\tdefault_model\tdocs_url\tnotes",
        *[
            "\t".join(
                [
                    entry.name,
                    entry.base_url,
                    entry.api_key_env,
                    entry.default_model or "-",
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
