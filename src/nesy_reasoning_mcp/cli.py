"""Command line entrypoint for the NeSy Reasoning MCP server."""

from __future__ import annotations

import argparse
import sys

import anyio

from nesy_reasoning_mcp.audit_cli import run_audit_cli
from nesy_reasoning_mcp.evaluation import run_eval_cli, run_llm_eval_cli
from nesy_reasoning_mcp.hooks import run_pretooluse_hook, run_stop_hook
from nesy_reasoning_mcp.http_server import run_http_server
from nesy_reasoning_mcp.server import run_stdio_server


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(prog="nesy-reasoning-mcp")
    subparsers = parser.add_subparsers(dest="command")
    hook_parser = subparsers.add_parser("hook", help="Run a Claude Code hook helper.")
    hook_parser.add_argument("hook_name", choices=["pretooluse", "stop"])
    audit_parser = subparsers.add_parser("audit", help="Inspect local audit logs.")
    audit_subparsers = audit_parser.add_subparsers(dest="audit_command")
    audit_list_parser = audit_subparsers.add_parser("list", help="List recent audit entries.")
    audit_list_parser.add_argument("--format", choices=["text", "json"], default="text")
    audit_list_parser.add_argument("--limit", type=int, default=50)
    audit_list_parser.add_argument("--tool-name", default=None)
    audit_list_parser.add_argument("--result-status", default=None)
    eval_parser = subparsers.add_parser("eval", help="Run offline evaluation helpers.")
    eval_subparsers = eval_parser.add_subparsers(dest="eval_command")
    eval_run_parser = eval_subparsers.add_parser("run", help="Run an offline benchmark fixture.")
    eval_run_parser.add_argument(
        "--fixture",
        default="benchmarks/fixtures/core.json",
        help="Benchmark fixture JSON path.",
    )
    eval_run_parser.add_argument(
        "--format",
        choices=["text", "json"],
        default="text",
        help="Evaluation report output format.",
    )
    eval_run_parser.add_argument(
        "--output",
        default=None,
        help="Optional output path. Defaults to stdout.",
    )
    eval_run_parser.add_argument(
        "--min-score",
        type=float,
        default=1.0,
        help="Minimum full MCP score required for exit 0.",
    )
    eval_llm_parser = eval_subparsers.add_parser("llm", help="Run a live LLM baseline fixture.")
    eval_llm_parser.add_argument(
        "--fixture",
        default="benchmarks/fixtures/core.json",
        help="Benchmark fixture JSON path.",
    )
    eval_llm_parser.add_argument(
        "--provider",
        choices=["openai"],
        default="openai",
        help="Live LLM provider.",
    )
    eval_llm_parser.add_argument(
        "--model",
        default="gpt-5.2",
        help="Provider model name.",
    )
    eval_llm_parser.add_argument(
        "--format",
        choices=["text", "json"],
        default="text",
        help="Evaluation report output format.",
    )
    eval_llm_parser.add_argument(
        "--output",
        default=None,
        help="Optional output path. Defaults to stdout.",
    )
    eval_llm_parser.add_argument(
        "--case-id",
        action="append",
        default=[],
        help="Optional benchmark case id. May be repeated.",
    )
    parser.add_argument(
        "--transport",
        choices=["stdio", "http"],
        default="stdio",
        help="MCP transport to use.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    """Run the NeSy Reasoning MCP server CLI."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "hook":
        if args.hook_name == "pretooluse":
            sys.exit(run_pretooluse_hook())
        sys.exit(run_stop_hook())
    if args.command == "audit":
        try:
            sys.exit(run_audit_cli(args))
        except ValueError as exc:
            parser.error(str(exc))
    if args.command == "eval":
        if args.eval_command == "run":
            sys.exit(run_eval_cli(args))
        if args.eval_command == "llm":
            sys.exit(run_llm_eval_cli(args))
        parser.error("eval command requires a subcommand")

    try:
        if args.transport == "http":
            anyio.run(run_http_server)
        else:
            anyio.run(run_stdio_server)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(2)
    except KeyboardInterrupt:
        sys.exit(130)
