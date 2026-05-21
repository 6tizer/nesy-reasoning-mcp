"""MCP server wiring for NeSy Reasoning v0.1."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass

from mcp.server import Server
from mcp.server.models import InitializationOptions
from mcp.server.stdio import stdio_server
from mcp.types import Tool

from nesy_reasoning_mcp import __version__
from nesy_reasoning_mcp.store import RelationStoreProtocol, create_relation_store
from nesy_reasoning_mcp.tools import call_tool, get_tools


@dataclass(slots=True)
class ServerState:
    """Server lifespan state."""

    store: RelationStoreProtocol


@asynccontextmanager
async def lifespan(_server: Server[ServerState]) -> AsyncIterator[ServerState]:
    """Create per-process server state."""
    yield ServerState(store=create_relation_store())


def create_server(store: RelationStoreProtocol | None = None) -> Server[ServerState]:
    """Create and configure the NeSy Reasoning MCP server."""
    server: Server[ServerState] = Server(
        "nesy-reasoning",
        version=__version__,
        lifespan=lifespan,
    )
    fixed_store = store

    @server.list_tools()
    async def list_tools() -> list[Tool]:
        return get_tools()

    @server.call_tool(validate_input=False)
    async def dispatch_tool(name: str, arguments: dict) -> object:
        if fixed_store is not None:
            active_store = fixed_store
        else:
            active_store = server.request_context.lifespan_context.store
        return await call_tool(name, arguments, active_store)

    return server


async def run_stdio_server() -> None:
    """Run the MCP server on stdio."""
    server = create_server()
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            initialization_options=server.create_initialization_options(),
        )


def initialization_options(server: Server[ServerState]) -> InitializationOptions:
    """Return server initialization options for tests and embedding."""
    return server.create_initialization_options()
