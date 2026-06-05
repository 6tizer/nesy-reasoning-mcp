"""Background ingestion worker and standalone nesy-worker daemon."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Mapping
from contextlib import asynccontextmanager, suppress
from typing import Any

import uvicorn
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from nesy_reasoning_mcp import __version__
from nesy_reasoning_mcp.auto_ingest.worker import (
    IngestionWorkerConfig,
    IngestionWorkerResult,
    process_ingestion_queue_once,
)
from nesy_reasoning_mcp.config import NesyConfig, WorkerConfig, load_config
from nesy_reasoning_mcp.store import RelationStoreProtocol, create_relation_store

logger = logging.getLogger(__name__)


async def run_ingestion_task(
    store: RelationStoreProtocol,
    *,
    config: WorkerConfig,
    env: Mapping[str, str] | None = None,
) -> IngestionWorkerResult:
    """Run one ingestion extraction cycle: claim a job, extract, and enqueue review."""
    worker_config = IngestionWorkerConfig(
        poll_seconds=config.poll_seconds,
        claim_limit=config.claim_limit,
    )
    return await process_ingestion_queue_once(
        store,
        config=worker_config,
        env=env,
    )


async def run_worker_loop(
    store: RelationStoreProtocol,
    *,
    config: WorkerConfig | None = None,
    env: Mapping[str, str] | None = None,
    shutdown_event: asyncio.Event | None = None,
) -> IngestionWorkerResult:
    """Run the background ingestion worker loop until shutdown is signaled.

    Cycles run sequentially — each ``_do_cycle()`` completes before the next
    poll interval begins, preventing unbounded task accumulation.
    """
    worker_config = config or WorkerConfig()
    total = IngestionWorkerResult(iterations=0)

    if shutdown_event is None:
        shutdown_event = asyncio.Event()

    current_cycle: asyncio.Task[IngestionWorkerResult] | None = None

    async def _do_cycle() -> IngestionWorkerResult:
        return await run_ingestion_task(store, config=worker_config, env=env)

    try:
        while not shutdown_event.is_set():
            # Run one cycle sequentially — no concurrent accumulation.
            current_cycle = asyncio.create_task(_do_cycle())
            try:
                result = await current_cycle
                total = total.merged(result)
            except Exception:
                logger.exception("worker cycle raised unhandled exception")
            finally:
                current_cycle = None

            # Wait for next poll interval or shutdown signal, whichever comes first.
            with suppress(TimeoutError):
                await asyncio.wait_for(
                    shutdown_event.wait(),
                    timeout=worker_config.poll_seconds,
                )
    except asyncio.CancelledError:
        logger.info("ingestion worker loop cancelled, draining current cycle")
    finally:
        if current_cycle is not None and not current_cycle.done():
            logger.info("draining in-flight ingestion cycle")
            try:
                result = await current_cycle
                total = total.merged(result)
            except Exception:
                logger.exception("in-flight cycle raised unhandled exception")

    return total


def _create_health_response(
    *,
    worker_running: bool = True,
    worker_error: str | None = None,
) -> JSONResponse:
    """Build health endpoint response reflecting daemon state."""
    if worker_running:
        return JSONResponse(
            {
                "status": "ok",
                "name": "nesy-worker",
                "version": __version__,
            }
        )
    body: dict[str, Any] = {
        "status": "unhealthy",
        "name": "nesy-worker",
        "version": __version__,
    }
    if worker_error:
        body["error"] = worker_error
    return JSONResponse(body, status_code=503)


async def run_nesy_worker(config: NesyConfig | None = None) -> None:
    """Run the nesy-worker standalone daemon with health HTTP endpoint.

    Connects to the same store as the MCP server and processes the ingestion
    queue independently.  Exposes a /healthz HTTP endpoint for monitoring.

    The health endpoint requires bearer-token auth when ``NESY_LOCAL_TOKEN``
    is configured; otherwise it's openly reachable on the configured host.
    """
    resolved = config or load_config()
    store = create_relation_store(resolved)

    shutdown_event = asyncio.Event()
    worker_task: asyncio.Task[Any] | None = None

    # Shared mutable health state visible to the lifespan and health endpoint.
    health: dict[str, Any] = {"running": False, "error": None}

    async def healthz(request: Request) -> JSONResponse:
        """Return daemon health information with optional auth."""
        token = resolved.http.local_token
        if token:
            from hmac import compare_digest

            auth = request.headers.get("authorization", "")
            if not compare_digest(auth, f"Bearer {token}"):
                return JSONResponse(
                    {"error": "missing_or_invalid_token"},
                    status_code=401,
                )
        return _create_health_response(
            worker_running=health["running"],
            worker_error=health["error"],
        )

    @asynccontextmanager
    async def lifespan(_app: Starlette) -> Any:  # noqa: ANN401
        nonlocal worker_task
        health["running"] = True
        health["error"] = None
        worker_task = asyncio.create_task(
            run_worker_loop(
                store,
                config=resolved.worker,
                shutdown_event=shutdown_event,
            )
        )
        try:
            yield
        finally:
            shutdown_event.set()
            if worker_task is not None:
                try:
                    await worker_task
                except Exception:
                    logger.exception("worker task crashed during shutdown")
                    health["running"] = False
                    health["error"] = "worker crashed"
                else:
                    health["running"] = False
            logger.info("nesy-worker shutdown complete")

    app = Starlette(
        routes=[Route("/healthz", endpoint=healthz, methods=["GET"])],
        lifespan=lifespan,
    )

    uvicorn_config = uvicorn.Config(
        app,
        host=resolved.http.host,
        port=resolved.worker.health_port,
        log_level=resolved.logging.level.lower(),
        access_log=False,
    )
    logger.info(
        "nesy-worker starting on %s:%d (health) — poll=%ss",
        resolved.http.host,
        resolved.worker.health_port,
        resolved.worker.poll_seconds,
    )
    await uvicorn.Server(uvicorn_config).serve()
