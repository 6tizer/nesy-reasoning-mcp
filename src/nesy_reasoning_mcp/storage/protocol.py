"""Storage protocol for relation stores."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any, Protocol

from nesy_reasoning_mcp.auto_ingest.scheduler import (
    ScheduledIngestionJob,
    ScheduledIngestionJobFilter,
    ScheduledIngestionJobStatus,
    ScheduledIngestionRun,
    ScheduledIngestionRunFilter,
    ScheduledIngestionState,
)
from nesy_reasoning_mcp.auto_ingest.schemas import (
    ConversationTurnJob,
    ConversationTurnJobFilter,
    ReviewQueueFilter,
    ReviewQueueRecord,
)
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

    def enqueue_ingestion_jobs(
        self,
        records: Iterable[ConversationTurnJob],
    ) -> tuple[list[ConversationTurnJob], int]:
        """Add conversation turn ingestion jobs and return stored records plus update count."""

    def list_ingestion_jobs(
        self,
        job_filter: ConversationTurnJobFilter | None = None,
        *,
        limit: int | None = None,
        offset: int = 0,
    ) -> list[ConversationTurnJob]:
        """List conversation turn ingestion jobs matching an optional filter."""

    def enqueue_review_queue(
        self,
        records: Iterable[ReviewQueueRecord],
    ) -> tuple[list[ReviewQueueRecord], int]:
        """Add review queue records and return stored records plus update count."""

    def list_review_queue(
        self,
        queue_filter: ReviewQueueFilter | None = None,
        *,
        limit: int | None = None,
        offset: int = 0,
    ) -> list[ReviewQueueRecord]:
        """List review queue records matching an optional filter."""

    def mark_review_queue_committed(
        self,
        ids: Iterable[str],
        relation_ids_by_record: Mapping[str, list[str]],
    ) -> int:
        """Mark review queue records as committed and return updated count."""

    def resolve_review_queue(
        self,
        ids: Iterable[str],
        *,
        reason: str,
        metadata: dict[str, Any] | None = None,
    ) -> int:
        """Resolve review queue records without writing graph relations."""

    def upsert_scheduled_ingestion_job(
        self,
        job: ScheduledIngestionJob,
    ) -> tuple[ScheduledIngestionJob, int]:
        """Add or update one scheduled ingestion job."""

    def list_scheduled_ingestion_jobs(
        self,
        job_filter: ScheduledIngestionJobFilter | None = None,
        *,
        limit: int | None = None,
        offset: int = 0,
    ) -> list[ScheduledIngestionJob]:
        """List scheduled ingestion jobs matching an optional filter."""

    def get_scheduled_ingestion_job(self, job_id: str) -> ScheduledIngestionJob | None:
        """Return one scheduled ingestion job by id."""

    def update_scheduled_ingestion_job_state(
        self,
        job_id: str,
        *,
        state: ScheduledIngestionState,
        status: ScheduledIngestionJobStatus | None = None,
        expected_status: ScheduledIngestionJobStatus | None = None,
    ) -> ScheduledIngestionJob | None:
        """Update mutable scheduled ingestion job state."""

    def append_scheduled_ingestion_run(
        self,
        run: ScheduledIngestionRun,
    ) -> ScheduledIngestionRun:
        """Append or replace one scheduled ingestion run record."""

    def list_scheduled_ingestion_runs(
        self,
        run_filter: ScheduledIngestionRunFilter | None = None,
        *,
        limit: int | None = None,
        offset: int = 0,
    ) -> list[ScheduledIngestionRun]:
        """List scheduled ingestion runs matching an optional filter."""

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
