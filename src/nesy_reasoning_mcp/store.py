"""Compatibility exports for relation storage."""

from __future__ import annotations

from nesy_reasoning_mcp.storage.audit import AuditEntry
from nesy_reasoning_mcp.storage.common import graph_stats_for
from nesy_reasoning_mcp.storage.factory import create_relation_store
from nesy_reasoning_mcp.storage.json_store import JsonRelationStore
from nesy_reasoning_mcp.storage.memory import MemoryRelationStore
from nesy_reasoning_mcp.storage.protocol import RelationStoreProtocol
from nesy_reasoning_mcp.storage.sqlite import SqliteRelationStore

RelationStore = MemoryRelationStore

__all__ = [
    "AuditEntry",
    "JsonRelationStore",
    "MemoryRelationStore",
    "RelationStore",
    "RelationStoreProtocol",
    "SqliteRelationStore",
    "create_relation_store",
    "graph_stats_for",
]
