"""Explicit URL fetching for Agent SDK dry-run ingestion."""

from __future__ import annotations

from datetime import UTC, datetime
from ipaddress import ip_address
from typing import Any
from urllib import request
from urllib.parse import urlparse

from nesy_reasoning_mcp.auto_ingest.schemas import EvidenceRecord

DEFAULT_FETCH_TIMEOUT_SECONDS = 10.0
DEFAULT_MAX_FETCH_BYTES = 200_000


def fetch_url_evidence(
    url: str,
    *,
    timeout_seconds: float = DEFAULT_FETCH_TIMEOUT_SECONDS,
    max_bytes: int = DEFAULT_MAX_FETCH_BYTES,
) -> EvidenceRecord:
    """Fetch one explicit HTTP(S) URL and return a bounded evidence record."""
    normalized_url = _validate_url(url)
    if max_bytes < 1:
        raise ValueError("max_bytes must be positive")
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")

    req = request.Request(
        normalized_url,
        headers={"User-Agent": "nesy-reasoning-mcp-agent-ingest/1.0"},
    )
    with request.urlopen(req, timeout=timeout_seconds) as response:  # noqa: S310
        body = response.read(max_bytes + 1)
        truncated = len(body) > max_bytes
        if truncated:
            body = body[:max_bytes]
        headers = getattr(response, "headers", {})
        content_type = _header_get(headers, "content-type")
        charset = _charset_from_content_type(content_type)
        text = body.decode(charset, errors="replace")

    return EvidenceRecord(
        url=normalized_url,
        span=text,
        source_type="url",
        retrieved_at=datetime.now(UTC).isoformat(),
        metadata={
            "content_type": content_type,
            "bytes_read": len(body),
            "truncated": truncated,
        },
    )


def fetch_url_evidence_many(
    urls: list[str],
    *,
    timeout_seconds: float = DEFAULT_FETCH_TIMEOUT_SECONDS,
    max_bytes: int = DEFAULT_MAX_FETCH_BYTES,
) -> list[EvidenceRecord]:
    """Fetch explicit HTTP(S) URLs without crawling links."""
    return [
        fetch_url_evidence(
            url,
            timeout_seconds=timeout_seconds,
            max_bytes=max_bytes,
        )
        for url in urls
    ]


def _validate_url(url: str) -> str:
    normalized = url.strip()
    if not normalized:
        raise ValueError("url must not be empty")
    parsed = urlparse(normalized)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError("only explicit http/https URLs are supported")
    if not parsed.hostname:
        raise ValueError("url must include a host")
    if _is_local_host(parsed.hostname):
        raise ValueError("local URLs are not supported")
    return normalized


def _is_local_host(hostname: str) -> bool:
    normalized = hostname.strip().lower().rstrip(".")
    if (
        normalized == "localhost"
        or normalized.endswith(".localhost")
        or normalized.endswith(".local")
    ):
        return True
    try:
        address = ip_address(normalized)
    except ValueError:
        return False
    return (
        address.is_loopback
        or address.is_private
        or address.is_link_local
        or address.is_reserved
        or address.is_multicast
    )


def _header_get(headers: Any, key: str) -> str | None:
    value = headers.get(key) if hasattr(headers, "get") else None
    return str(value) if value else None


def _charset_from_content_type(content_type: str | None) -> str:
    if not content_type:
        return "utf-8"
    for part in content_type.split(";"):
        key, _, value = part.strip().partition("=")
        if key.lower() == "charset" and value.strip():
            return value.strip()
    return "utf-8"
