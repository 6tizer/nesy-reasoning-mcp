"""Command line entrypoint for the NeSy Reasoning MCP server."""

from __future__ import annotations

import argparse
import sys

import anyio

from nesy_reasoning_mcp.evaluation import run_eval_cli
from nesy_reasoning_mcp.hooks import run_pretooluse_hook, run_stop_hook
from nesy_reasoning_mcp.http_server import run_http_server
from nesy_reasoning_mcp.server import run_stdio_server


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(prog="nesy-reasoning-mcp")
    subparsers = parser.add_subparsers(dest="command")
    hook_parser = subparsers.add_parser("hook", help="Run a Claude Code hook helper.")
    hook_parser.add_argument("hook_name", choices=["pretooluse", "stop"])
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
    if args.command == "eval":
        if args.eval_command == "run":
            sys.exit(run_eval_cli(args))
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
