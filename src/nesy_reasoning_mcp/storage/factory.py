"""Factory for configured relation stores."""

from __future__ import annotations

import logging

from nesy_reasoning_mcp.config import NesyConfig, StorageBackend, load_config
from nesy_reasoning_mcp.storage.json_store import JsonRelationStore
from nesy_reasoning_mcp.storage.memory import MemoryRelationStore
from nesy_reasoning_mcp.storage.protocol import RelationStoreProtocol
from nesy_reasoning_mcp.storage.sqlite import SqliteRelationStore

logger = logging.getLogger(__name__)


def create_relation_store(config: NesyConfig | None = None) -> RelationStoreProtocol:
    """Create a relation store for the configured backend."""
    resolved = config or load_config()
    if resolved.storage.backend == StorageBackend.MEMORY:
        return MemoryRelationStore(resolved)
    if resolved.storage.backend == StorageBackend.JSON:
        logger.warning(
            "JSON backend has no file locking — running multiple workers "
            "(HTTP worker + standalone nesy-worker) against the same JSON file "
            "may cause double-claims and data races.  Use SQLite for multi-worker "
            "deployments."
        )
        return JsonRelationStore(resolved)
    if resolved.storage.backend == StorageBackend.SQLITE:
        return SqliteRelationStore(resolved)
    raise ValueError(f"storage backend is not implemented yet: {resolved.storage.backend}")
