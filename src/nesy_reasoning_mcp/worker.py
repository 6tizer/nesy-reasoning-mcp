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
        claim_limit=1,
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
    """Run the background ingestion worker loop until shutdown is signaled."""
    worker_config = config or WorkerConfig()
    total = IngestionWorkerResult(iterations=0)
    in_flight: set[asyncio.Task[Any]] = set()

    if shutdown_event is None:
        shutdown_event = asyncio.Event()

    async def _do_cycle() -> IngestionWorkerResult:
        return await run_ingestion_task(store, config=worker_config, env=env)

    try:
        while not shutdown_event.is_set():
            task = asyncio.create_task(_do_cycle())
            in_flight.add(task)

            with suppress(TimeoutError):
                await asyncio.wait_for(shutdown_event.wait(), timeout=worker_config.poll_seconds)

            # Merge results from completed tasks
            done = {t for t in in_flight if t.done() and not t.cancelled()}
            for t in done:
                try:
                    result = t.result()
                    total = total.merged(result)
                except Exception:
                    logger.exception("worker cycle task raised unhandled exception")
            in_flight -= done
    except asyncio.CancelledError:
        logger.info("ingestion worker loop cancelled, draining in-flight tasks")
    finally:
        if in_flight:
            logger.info("draining %d in-flight ingestion tasks", len(in_flight))
            results = await asyncio.gather(*in_flight, return_exceptions=True)
            for result in results:
                if isinstance(result, IngestionWorkerResult):
                    total = total.merged(result)

    return total


async def _healthz(_request: Request) -> JSONResponse:
    """Return daemon health information."""
    return JSONResponse(
        {
            "status": "ok",
            "name": "nesy-worker",
            "version": __version__,
        }
    )


async def run_nesy_worker(config: NesyConfig | None = None) -> None:
    """Run the nesy-worker standalone daemon with health HTTP endpoint.

    Connects to the same store as the MCP server and processes the ingestion
    queue independently.  Exposes a /healthz HTTP endpoint for monitoring.
    """
    resolved = config or load_config()
    store = create_relation_store(resolved)

    shutdown_event = asyncio.Event()
    worker_task: asyncio.Task[Any] | None = None

    @asynccontextmanager
    async def lifespan(_app: Starlette) -> Any:  # noqa: ANN401
        nonlocal worker_task
        worker_task = asyncio.create_task(
            run_worker_loop(
                store,
                config=resolved.worker,
                shutdown_event=shutdown_event,
            )
        )
        yield
        shutdown_event.set()
        if worker_task is not None:
            await worker_task
        logger.info("nesy-worker shutdown complete")

    app = Starlette(
        routes=[Route("/healthz", endpoint=_healthz, methods=["GET"])],
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
