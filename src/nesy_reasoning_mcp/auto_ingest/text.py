"""Text normalization helpers for auto-ingest modules."""

from __future__ import annotations

from collections.abc import Iterable


def dedupe_non_empty_text(values: Iterable[str]) -> list[str]:
    """Strip text values, drop empties, and de-duplicate in input order."""
    stripped = [value.strip() for value in values]
    return list(dict.fromkeys(value for value in stripped if value))
