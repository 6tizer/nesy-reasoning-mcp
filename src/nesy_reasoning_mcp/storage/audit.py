"""Audit helpers for relation stores."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from hashlib import sha256
from typing import Any
from uuid import uuid4


class AuditEntry:
    """Audit log entry for mutating operations."""

    def __init__(
        self,
        *,
        event_type: str,
        tool_name: str,
        input_hash: str,
        result_status: str,
        metadata: dict[str, Any] | None = None,
        entry_id: str | None = None,
        created_at: str | None = None,
    ) -> None:
        self.id = entry_id or f"audit_{uuid4().hex}"
        self.event_type = event_type
        self.tool_name = tool_name
        self.input_hash = input_hash
        self.result_status = result_status
        self.created_at = created_at or datetime.now(UTC).isoformat()
        self.metadata = metadata or {}

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-serializable audit data."""
        return {
            "id": self.id,
            "event_type": self.event_type,
            "tool_name": self.tool_name,
            "input_hash": self.input_hash,
            "result_status": self.result_status,
            "created_at": self.created_at,
            "metadata": self.metadata,
        }


def _input_hash(arguments: dict[str, Any]) -> str:
    payload = json.dumps(arguments, sort_keys=True, ensure_ascii=False, default=str)
    return sha256(payload.encode("utf-8")).hexdigest()


def _audit_from_dict(data: dict[str, Any]) -> AuditEntry:
    return AuditEntry(
        entry_id=data["id"],
        event_type=data["event_type"],
        tool_name=data["tool_name"],
        input_hash=data["input_hash"],
        result_status=data["result_status"],
        created_at=data["created_at"],
        metadata=data.get("metadata", {}),
    )
