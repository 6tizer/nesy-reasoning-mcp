"""Explicit URL fetching for Agent SDK dry-run ingestion."""

from __future__ import annotations

from dataclasses import dataclass
from ipaddress import ip_address
from socket import getaddrinfo
from typing import Any
from urllib import request
from urllib.parse import urlparse

from nesy_reasoning_mcp.auto_ingest.schemas import EvidenceRecord
from nesy_reasoning_mcp.time_utils import utc_now_iso

DEFAULT_FETCH_TIMEOUT_SECONDS = 10.0
DEFAULT_MAX_FETCH_BYTES = 200_000


@dataclass(frozen=True)
class FetchedUrlPage:
    """Bounded body and metadata from a public HTTP(S) fetch."""

    requested_url: str
    final_url: str
    body: bytes
    content_type: str | None
    charset: str
    truncated: bool

    @property
    def text(self) -> str:
        """Decode the fetched body using the response charset."""
        return self.body.decode(self.charset, errors="replace")


def fetch_url_evidence(
    url: str,
    *,
    timeout_seconds: float = DEFAULT_FETCH_TIMEOUT_SECONDS,
    max_bytes: int = DEFAULT_MAX_FETCH_BYTES,
) -> EvidenceRecord:
    """Fetch one explicit HTTP(S) URL and return a bounded evidence record."""
    page = fetch_public_http_url(
        url,
        timeout_seconds=timeout_seconds,
        max_bytes=max_bytes,
    )

    return EvidenceRecord(
        url=page.requested_url,
        span=page.text,
        source_type="url",
        retrieved_at=utc_now_iso(),
        metadata={
            "content_type": page.content_type,
            "bytes_read": len(page.body),
            "truncated": page.truncated,
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


def validate_public_http_url(url: str) -> str:
    """Validate an explicit public HTTP(S) URL without fetching it."""
    return _validate_url(url)


def fetch_public_http_url(
    url: str,
    *,
    timeout_seconds: float = DEFAULT_FETCH_TIMEOUT_SECONDS,
    max_bytes: int = DEFAULT_MAX_FETCH_BYTES,
) -> FetchedUrlPage:
    """Fetch a public HTTP(S) URL with redirect validation and byte limits."""
    normalized_url = validate_public_http_url(url)
    if max_bytes < 1:
        raise ValueError("max_bytes must be positive")
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")

    req = request.Request(
        normalized_url,
        headers={"User-Agent": "nesy-reasoning-mcp-agent-ingest/1.0"},
    )
    opener = request.build_opener(_SafeRedirectHandler)
    with opener.open(req, timeout=timeout_seconds) as response:
        body = response.read(max_bytes + 1)
        truncated = len(body) > max_bytes
        if truncated:
            body = body[:max_bytes]
        headers = getattr(response, "headers", {})
        content_type = _header_get(headers, "content-type")
        charset = _charset_from_content_type(content_type)
        geturl = getattr(response, "geturl", None)
        final_url = validate_public_http_url(str(geturl() if callable(geturl) else normalized_url))

    return FetchedUrlPage(
        requested_url=normalized_url,
        final_url=final_url,
        body=body,
        content_type=content_type,
        charset=charset,
        truncated=truncated,
    )


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
    _validate_resolved_host(parsed.hostname, parsed.port)
    return normalized


class _SafeRedirectHandler(request.HTTPRedirectHandler):
    """Validate redirect targets before following them."""

    def redirect_request(
        self, req: Any, fp: Any, code: int, msg: str, headers: Any, newurl: str
    ) -> Any:
        _validate_url(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


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


def _validate_resolved_host(hostname: str, port: int | None) -> None:
    try:
        results = getaddrinfo(hostname, port or 443)
    except OSError as exc:
        raise ValueError(f"could not resolve URL host: {hostname}") from exc

    for result in results:
        sockaddr = result[4]
        if not sockaddr:
            continue
        resolved_host = str(sockaddr[0]).split("%", maxsplit=1)[0]
        if _is_local_host(resolved_host):
            raise ValueError("resolved URL host is local or private")


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
