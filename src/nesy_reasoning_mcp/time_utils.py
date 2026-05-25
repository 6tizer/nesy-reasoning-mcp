"""Shared datetime parsing helpers."""

from __future__ import annotations

from datetime import UTC, datetime, tzinfo
from typing import Any


def parse_datetime_value(
    value: Any,
    *,
    default_tz: tzinfo | None = UTC,
    reference: datetime | None = None,
) -> datetime | None:
    """Parse ISO datetime/date-like values with stable timezone normalization."""
    if value is None:
        return None
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str):
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    else:
        return None

    if reference is not None:
        if parsed.tzinfo is None and reference.tzinfo is not None:
            return parsed.replace(tzinfo=reference.tzinfo)
        if parsed.tzinfo is not None and reference.tzinfo is None:
            return parsed.replace(tzinfo=None)
        return parsed
    if parsed.tzinfo is None and default_tz is not None:
        return parsed.replace(tzinfo=default_tz)
    return parsed


def utc_now_iso() -> str:
    """Return the current UTC timestamp as an ISO-8601 string."""
    return datetime.now(UTC).isoformat()
