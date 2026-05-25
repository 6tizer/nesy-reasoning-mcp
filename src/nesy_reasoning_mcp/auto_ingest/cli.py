"""CLI helpers for Agent SDK dry-run ingestion."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from re import fullmatch
from typing import Any, TextIO
from urllib.parse import urlparse

import anyio
from pydantic import ValidationError

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
from nesy_reasoning_mcp.auto_ingest.schemas import IngestionInput, IngestionReport
from nesy_reasoning_mcp.config import load_config
from nesy_reasoning_mcp.store import create_relation_store

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


def add_agent_dry_run_arguments(parser: argparse.ArgumentParser) -> None:
    """Add shared arguments for the OpenAI Agents SDK dry-run command."""
    parser.add_argument("--input", default=None, help="JSON input file path.")
    parser.add_argument("--url", action="append", default=[], help="Explicit HTTP(S) URL source.")
    parser.add_argument("--task", default=None, help="Optional extraction task.")
    parser.add_argument("--question", default=None, help="Optional question to answer.")
    parser.add_argument("--model", default=None, help="OpenAI Agents SDK model override.")
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


def run_ingest_cli(args: argparse.Namespace) -> int:
    """Dispatch ingestion CLI subcommands."""
    if args.ingest_command == "agent-dry-run":
        return run_agent_dry_run_cli(args)
    raise ValueError("ingest command requires a subcommand")


def run_agent_dry_run_cli(
    args: argparse.Namespace,
    *,
    stdout: TextIO = sys.stdout,
    stderr: TextIO = sys.stderr,
) -> int:
    """Run the OpenAI Agents SDK dry-run CLI command."""
    try:
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


def main(argv: list[str] | None = None) -> int:
    """Run the standalone OpenAI Agents SDK ingestion wrapper."""
    parser = argparse.ArgumentParser(prog="agent_ingest_openai.py")
    add_agent_dry_run_arguments(parser)
    args = parser.parse_args(argv)
    return run_agent_dry_run_cli(args)


async def _run_agent_dry_run(args: argparse.Namespace) -> IngestionReport:
    if not 0 <= args.min_write_confidence <= 1:
        raise ValueError("--min-write-confidence must be between 0 and 1")
    provider_config = _provider_config_from_args(args)
    ingestion_input = _load_ingestion_input(args)
    if not ingestion_input.evidence and not ingestion_input.urls:
        raise ValueError("agent-dry-run requires --input evidence or at least one --url")

    fetched = fetch_url_evidence_many(
        ingestion_input.urls,
        timeout_seconds=args.timeout_seconds,
        max_bytes=args.max_url_bytes,
    )
    effective_input = ingestion_input.model_copy(
        update={
            "evidence": [*ingestion_input.evidence, *fetched],
        }
    )
    store = create_relation_store(load_config())
    return await run_openai_agents_ingestion(
        effective_input,
        store=store,
        model=args.model,
        auto_write=args.auto_write,
        min_write_confidence=args.min_write_confidence,
        provider_config=provider_config,
        disable_tracing=bool(getattr(args, "disable_tracing", False)),
    )


def _provider_config_from_args(
    args: argparse.Namespace,
) -> OpenAICompatibleProviderConfig | None:
    base_url = getattr(args, "base_url", None)
    api_key_env = getattr(args, "api_key_env", None)
    headers = getattr(args, "provider_header", []) or []
    if not base_url:
        if api_key_env:
            raise ValueError("--api-key-env requires --base-url")
        if headers:
            raise ValueError("--provider-header requires --base-url")
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
        disable_tracing=True,
    )


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
