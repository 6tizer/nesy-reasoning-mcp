"""Command line entrypoint for the NeSy Reasoning MCP server."""

from __future__ import annotations

import argparse
import sys

import anyio

from nesy_reasoning_mcp.hooks import run_pretooluse_hook, run_stop_hook
from nesy_reasoning_mcp.http_server import run_http_server
from nesy_reasoning_mcp.server import run_stdio_server


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(prog="nesy-reasoning-mcp")
    subparsers = parser.add_subparsers(dest="command")
    hook_parser = subparsers.add_parser("hook", help="Run a Claude Code hook helper.")
    hook_parser.add_argument("hook_name", choices=["pretooluse", "stop"])
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
