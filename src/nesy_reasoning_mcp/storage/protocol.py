"""Storage protocol for relation stores."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Protocol

from nesy_reasoning_mcp.config import NesyConfig
from nesy_reasoning_mcp.schemas import (
    CanonicalImplicationEdge,
    ExclusiveGroupInput,
    ExclusiveGroupRecord,
    GraphStats,
    IndependenceRecord,
    PropositionRecord,
    RelationFilter,
    RelationInput,
    RelationRecord,
)


class RelationStoreProtocol(Protocol):
    """Storage interface used by MCP tool handlers."""

    config: NesyConfig

    def assert_relations(
        self,
        inputs: Iterable[RelationInput],
        *,
        mode: str = "append",
        dry_run: bool = False,
    ) -> tuple[list[RelationRecord], int]:
        """Add relation records and return added records plus update count."""

    def assert_exclusive(
        self,
        inputs: Iterable[ExclusiveGroupInput],
    ) -> tuple[list[ExclusiveGroupRecord], int]:
        """Add or replace exclusive groups and return records plus update count."""

    def list_relations(
        self,
        relation_filter: RelationFilter | None = None,
        *,
        limit: int | None = None,
        offset: int = 0,
    ) -> list[RelationRecord]:
        """List relation records matching an optional filter."""

    def list_exclusive_groups(self) -> list[ExclusiveGroupRecord]:
        """List all stored exclusive groups."""

    def list_independence_records(self) -> list[IndependenceRecord]:
        """List all stored independence records."""

    def list_propositions(self) -> list[PropositionRecord]:
        """List all stored proposition records."""

    def clear_relations(
        self,
        *,
        scope: str,
        store_id: str,
        context_id: str,
        relation_filter: RelationFilter,
        dry_run: bool,
        include_exclusive_groups: bool = False,
    ) -> tuple[int, int]:
        """Remove relations and optionally exclusive groups by scope."""

    def implication_edges(
        self,
        relations: Iterable[RelationRecord] | None = None,
    ) -> list[CanonicalImplicationEdge]:
        """Derive canonical implication edges from relation records."""

    def graph_stats(self) -> GraphStats:
        """Return statistics for the current graph."""

    def record_audit(
        self,
        *,
        event_type: str,
        tool_name: str,
        arguments: dict[str, Any],
        result_status: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Record one audit event."""

    def list_audit_entries(self) -> list[dict[str, Any]]:
        """List audit entries."""

    def context_metadata(self) -> dict[str, Any]:
        """Return graph context metadata."""

    def import_records(
        self,
        relations: Iterable[RelationRecord],
        exclusive_groups: Iterable[ExclusiveGroupRecord],
        independence_records: Iterable[IndependenceRecord] = (),
        propositions: Iterable[PropositionRecord] = (),
        *,
        mode: str,
        store_id: str,
        context_metadata: dict[str, Any] | None = None,
        dry_run: bool = False,
    ) -> tuple[int, int, int, int]:
        """Import validated records into the store."""
