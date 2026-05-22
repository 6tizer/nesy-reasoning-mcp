"""Streamable HTTP daemon wiring for NeSy Reasoning MCP."""

from __future__ import annotations

import json
import time
from collections.abc import Awaitable, Callable
from contextlib import asynccontextmanager
from hmac import compare_digest
from typing import Any

import anyio
import uvicorn
from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
from mcp.server.transport_security import TransportSecuritySettings
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route
from starlette.types import ASGIApp, Receive, Scope, Send

from nesy_reasoning_mcp import __version__
from nesy_reasoning_mcp.config import NesyConfig, load_config
from nesy_reasoning_mcp.server import create_server
from nesy_reasoning_mcp.store import RelationStoreProtocol, create_relation_store


class BodyTooLargeError(Exception):
    """Raised when a request body exceeds the configured limit."""


class HttpGuardMiddleware:
    """Apply local HTTP daemon auth, origin, body, rate, and timeout guards."""

    def __init__(self, app: ASGIApp, config: NesyConfig) -> None:
        self.app = app
        self.config = config
        self._hits: dict[str, tuple[float, int]] = {}

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        if not _host_allowed(scope, self.config):
            await _send_json(send, 403, {"error": "host_not_allowed"})
            return
        if not _origin_allowed(scope, self.config):
            await _send_json(send, 403, {"error": "origin_not_allowed"})
            return
        if not _authorized(scope, self.config):
            await _send_json(send, 401, {"error": "missing_or_invalid_token"})
            return
        if _content_length_exceeds_limit(scope, self.config):
            await _send_json(send, 413, {"error": "request_body_too_large"})
            return
        if not self._rate_allowed(scope):
            await _send_json(send, 429, {"error": "rate_limit_exceeded"})
            return

        sent_start = False
        limited_receive = _body_limited_receive(
            receive,
            max_body_bytes=self.config.http.max_body_bytes,
        )

        async def guarded_send(message: dict[str, Any]) -> None:
            nonlocal sent_start
            if message["type"] == "http.response.start":
                sent_start = True
            await send(message)

        try:
            with anyio.move_on_after(self.config.http.request_timeout_seconds) as cancel_scope:
                await self.app(scope, limited_receive, guarded_send)
            if cancel_scope.cancelled_caught and not sent_start:
                await _send_json(send, 504, {"error": "request_timeout"})
        except BodyTooLargeError:
            if not sent_start:
                await _send_json(send, 413, {"error": "request_body_too_large"})

    def _rate_allowed(self, scope: Scope) -> bool:
        limit = self.config.http.rate_limit_per_minute
        client = scope.get("client")
        key = str(client[0]) if client else "unknown"
        now = time.monotonic()
        window_start, count = self._hits.get(key, (now, 0))
        if now - window_start >= 60:
            self._hits[key] = (now, 1)
            return True
        if count >= limit:
            return False
        self._hits[key] = (window_start, count + 1)
        return True


class StreamableHTTPASGIApp:
    """ASGI adapter for the MCP Streamable HTTP session manager."""

    def __init__(self, session_manager: StreamableHTTPSessionManager) -> None:
        self.session_manager = session_manager

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        await self.session_manager.handle_request(scope, receive, send)


async def healthz(_request: Request) -> JSONResponse:
    """Return daemon health information."""
    return JSONResponse(
        {
            "status": "ok",
            "name": "nesy-reasoning",
            "version": __version__,
        }
    )


def create_http_app(
    config: NesyConfig | None = None,
    store: RelationStoreProtocol | None = None,
) -> ASGIApp:
    """Create the authenticated Streamable HTTP ASGI application."""
    resolved = config or load_config()
    if not resolved.http.local_token:
        raise ValueError("NESY_LOCAL_TOKEN is required when --transport http is used")

    active_store = store or create_relation_store(resolved)
    server = create_server(active_store)
    session_manager = StreamableHTTPSessionManager(
        app=server,
        json_response=False,
        stateless=False,
        security_settings=TransportSecuritySettings(
            allowed_hosts=_allowed_hosts(resolved),
            allowed_origins=_allowed_origins(resolved),
        ),
    )
    mcp_app = StreamableHTTPASGIApp(session_manager)

    @asynccontextmanager
    async def lifespan(_app: Starlette) -> Any:
        async with session_manager.run():
            yield

    app = Starlette(
        routes=[
            Route("/healthz", endpoint=healthz, methods=["GET"]),
            Route(_http_path(resolved), endpoint=mcp_app, methods=["GET", "POST", "DELETE"]),
        ],
        lifespan=lifespan,
    )
    return HttpGuardMiddleware(app, resolved)


async def run_http_server(config: NesyConfig | None = None) -> None:
    """Run the MCP Streamable HTTP daemon."""
    resolved = config or load_config()
    app = create_http_app(resolved)
    uvicorn_config = uvicorn.Config(
        app,
        host=resolved.http.host,
        port=resolved.http.port,
        log_level=resolved.logging.level.lower(),
        access_log=False,
    )
    await uvicorn.Server(uvicorn_config).serve()


def _headers(scope: Scope) -> dict[str, str]:
    return {
        key.decode("latin1").lower(): value.decode("latin1")
        for key, value in scope.get("headers", [])
    }


def _authorized(scope: Scope, config: NesyConfig) -> bool:
    token = config.http.local_token
    if not token:
        return False
    return compare_digest(_headers(scope).get("authorization", ""), f"Bearer {token}")


def _host_allowed(scope: Scope, config: NesyConfig) -> bool:
    host = _headers(scope).get("host")
    if not host:
        return True
    return host in _allowed_hosts(config)


def _origin_allowed(scope: Scope, config: NesyConfig) -> bool:
    origin = _headers(scope).get("origin")
    if not origin:
        return True
    return origin in _allowed_origins(config)


def _content_length_exceeds_limit(scope: Scope, config: NesyConfig) -> bool:
    content_length = _headers(scope).get("content-length")
    if content_length is None:
        return False
    try:
        return int(content_length) > config.http.max_body_bytes
    except ValueError:
        return True


def _allowed_hosts(config: NesyConfig) -> list[str]:
    if config.http.allowed_hosts:
        return config.http.allowed_hosts
    host = config.http.host
    port = config.http.port
    hosts = {
        host,
        f"{host}:{port}",
        "localhost",
        f"localhost:{port}",
        "127.0.0.1",
        f"127.0.0.1:{port}",
        "[::1]",
        f"[::1]:{port}",
    }
    return sorted(hosts)


def _allowed_origins(config: NesyConfig) -> list[str]:
    if config.http.allowed_origins:
        return config.http.allowed_origins
    port = config.http.port
    return [
        f"http://{config.http.host}:{port}",
        f"http://127.0.0.1:{port}",
        f"http://localhost:{port}",
    ]


def _http_path(config: NesyConfig) -> str:
    return config.http.path if config.http.path.startswith("/") else f"/{config.http.path}"


def _body_limited_receive(
    receive: Receive,
    *,
    max_body_bytes: int,
) -> Callable[[], Awaitable[dict[str, Any]]]:
    received = 0

    async def limited_receive() -> dict[str, Any]:
        nonlocal received
        message = await receive()
        if message["type"] == "http.request":
            received += len(message.get("body", b""))
            if received > max_body_bytes:
                raise BodyTooLargeError
        return message

    return limited_receive


async def _send_json(send: Send, status_code: int, payload: dict[str, Any]) -> None:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    await send(
        {
            "type": "http.response.start",
            "status": status_code,
            "headers": [(b"content-type", b"application/json")],
        }
    )
    await send({"type": "http.response.body", "body": body})
