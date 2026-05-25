"""JSON relation store backend."""

from __future__ import annotations

import json
import os
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from nesy_reasoning_mcp.auto_ingest.schemas import ReviewQueueRecord
from nesy_reasoning_mcp.config import NesyConfig
from nesy_reasoning_mcp.schemas import (
    ExclusiveGroupInput,
    ExclusiveGroupRecord,
    IndependenceRecord,
    PropositionRecord,
    RelationFilter,
    RelationInput,
    RelationRecord,
)
from nesy_reasoning_mcp.storage.audit import _audit_from_dict
from nesy_reasoning_mcp.storage.memory import MemoryRelationStore


class JsonRelationStore(MemoryRelationStore):
    """JSON-file source of truth for relation records."""

    def __init__(self, config: NesyConfig) -> None:
        super().__init__(config)
        json_path = config.storage.json_path or "~/.nesy-reasoning/relations.json"
        self.path = Path(json_path).expanduser()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            self._load_from_disk()

    def assert_relations(
        self,
        inputs: Iterable[RelationInput],
        *,
        mode: str = "append",
        dry_run: bool = False,
    ) -> tuple[list[RelationRecord], int]:
        """Add relation records and persist the JSON relation set."""
        records, updated = super().assert_relations(inputs, mode=mode, dry_run=dry_run)
        if not dry_run:
            self._persist()
        return records, updated

    def assert_exclusive(
        self,
        inputs: Iterable[ExclusiveGroupInput],
    ) -> tuple[list[ExclusiveGroupRecord], int]:
        """Add or replace exclusive groups and persist the JSON relation set."""
        records, updated = super().assert_exclusive(inputs)
        self._persist()
        return records, updated

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
        """Remove records and persist the JSON relation set."""
        removed, removed_groups = super().clear_relations(
            scope=scope,
            store_id=store_id,
            context_id=context_id,
            relation_filter=relation_filter,
            dry_run=dry_run,
            include_exclusive_groups=include_exclusive_groups,
        )
        if not dry_run:
            self._persist()
        return removed, removed_groups

    def enqueue_review_queue(
        self,
        records: Iterable[ReviewQueueRecord],
    ) -> tuple[list[ReviewQueueRecord], int]:
        """Add review queue records and persist the JSON relation set."""
        queued, updated = super().enqueue_review_queue(records)
        self._persist()
        return queued, updated

    def mark_review_queue_committed(
        self,
        ids: Iterable[str],
        relation_ids_by_record: Mapping[str, list[str]],
    ) -> int:
        """Mark review queue records as committed and persist the JSON relation set."""
        updated = super().mark_review_queue_committed(ids, relation_ids_by_record)
        self._persist()
        return updated

    def resolve_review_queue(
        self,
        ids: Iterable[str],
        *,
        reason: str,
        metadata: dict[str, Any] | None = None,
    ) -> int:
        """Resolve review queue records and persist the JSON relation set."""
        updated = super().resolve_review_queue(ids, reason=reason, metadata=metadata)
        self._persist()
        return updated

    def record_audit(
        self,
        *,
        event_type: str,
        tool_name: str,
        arguments: dict[str, Any],
        result_status: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Record one audit event and persist the JSON relation set."""
        super().record_audit(
            event_type=event_type,
            tool_name=tool_name,
            arguments=arguments,
            result_status=result_status,
            metadata=metadata,
        )
        self._persist()

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
        """Import validated records and persist the JSON relation set."""
        result = super().import_records(
            relations,
            exclusive_groups,
            independence_records,
            propositions,
            mode=mode,
            store_id=store_id,
            context_metadata=context_metadata,
            dry_run=dry_run,
        )
        if not dry_run:
            self._persist()
        return result

    def _load_from_disk(self) -> None:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON relation store: {self.path}") from exc

        self._relations = [
            RelationRecord.model_validate(item) for item in data.get("relations", [])
        ]
        self._exclusive_groups = [
            ExclusiveGroupRecord.model_validate(item) for item in data.get("exclusive_groups", [])
        ]
        self._independence_records = [
            IndependenceRecord.model_validate(item) for item in data.get("independence_records", [])
        ]
        self._propositions = [
            PropositionRecord.model_validate(item) for item in data.get("propositions", [])
        ]
        self._review_queue = [
            ReviewQueueRecord.model_validate(item) for item in data.get("review_queue", [])
        ]
        self._audit_log = [_audit_from_dict(item) for item in data.get("audit_log", [])]
        self._context_metadata = data.get("context_metadata", {})

    def _persist(self) -> None:
        data = {
            "version": "2.0",
            "relations": [
                record.model_dump(mode="json", exclude_none=True) for record in self._relations
            ],
            "exclusive_groups": [group.model_dump(mode="json") for group in self._exclusive_groups],
            "independence_records": [
                record.model_dump(mode="json") for record in self._independence_records
            ],
            "propositions": [
                proposition.model_dump(mode="json", exclude_none=True)
                for proposition in self._propositions
            ],
            "review_queue": [
                record.model_dump(mode="json", exclude_none=True) for record in self._review_queue
            ],
            "audit_log": [entry.to_dict() for entry in self._audit_log],
            "context_metadata": self._context_metadata,
        }
        tmp_path = self.path.with_name(f".{self.path.name}.tmp")
        tmp_path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        os.replace(tmp_path, self.path)
