"""Bounded crawler for explicit Agent SDK ingestion source URLs."""

from __future__ import annotations

from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from html.parser import HTMLParser
from typing import Any
from urllib.parse import urljoin, urlparse, urlunparse

from nesy_reasoning_mcp.auto_ingest.fetcher import (
    DEFAULT_FETCH_TIMEOUT_SECONDS,
    DEFAULT_MAX_FETCH_BYTES,
    FetchedUrlPage,
    fetch_public_http_url,
    validate_public_http_url,
)
from nesy_reasoning_mcp.auto_ingest.schemas import EvidenceRecord
from nesy_reasoning_mcp.auto_ingest.text import dedupe_non_empty_text
from nesy_reasoning_mcp.auto_ingest.url_safety import (
    host_matches_domain,
    normalize_domain_filters,
)
from nesy_reasoning_mcp.schemas import Diagnostic
from nesy_reasoning_mcp.time_utils import utc_now_iso

DEFAULT_CRAWL_MAX_DEPTH = 1
MAX_CRAWL_MAX_DEPTH = 3
DEFAULT_CRAWL_MAX_PAGES = 10
MAX_CRAWL_MAX_PAGES = 50
DEFAULT_CRAWL_MAX_PAGE_BYTES = DEFAULT_MAX_FETCH_BYTES
DEFAULT_CRAWL_MAX_TOTAL_BYTES = 1_000_000
MAX_CRAWL_MAX_TOTAL_BYTES = 5_000_000
DEFAULT_CRAWL_TIMEOUT_SECONDS = DEFAULT_FETCH_TIMEOUT_SECONDS
MAX_CRAWL_TIMEOUT_SECONDS = 30.0


@dataclass(frozen=True)
class CrawlOptions:
    """Bounded crawl options for explicit seed URLs."""

    seed_urls: list[str]
    max_depth: int = DEFAULT_CRAWL_MAX_DEPTH
    max_pages: int = DEFAULT_CRAWL_MAX_PAGES
    max_page_bytes: int = DEFAULT_CRAWL_MAX_PAGE_BYTES
    max_total_bytes: int = DEFAULT_CRAWL_MAX_TOTAL_BYTES
    timeout_seconds: float = DEFAULT_CRAWL_TIMEOUT_SECONDS
    allow_domains: list[str] | None = None

    def __post_init__(self) -> None:
        normalized = {
            "seed_urls": dedupe_non_empty_text(self.seed_urls),
            "allow_domains": normalize_domain_filters(self.allow_domains or []),
        }
        if not normalized["seed_urls"]:
            raise ValueError("crawl seed URLs must not be empty")
        if not 0 <= self.max_depth <= MAX_CRAWL_MAX_DEPTH:
            raise ValueError(f"crawl max depth must be between 0 and {MAX_CRAWL_MAX_DEPTH}")
        if not 1 <= self.max_pages <= MAX_CRAWL_MAX_PAGES:
            raise ValueError(f"crawl max pages must be between 1 and {MAX_CRAWL_MAX_PAGES}")
        if self.max_page_bytes < 1:
            raise ValueError("crawl max page bytes must be positive")
        if not 1 <= self.max_total_bytes <= MAX_CRAWL_MAX_TOTAL_BYTES:
            raise ValueError(
                f"crawl max total bytes must be between 1 and {MAX_CRAWL_MAX_TOTAL_BYTES}"
            )
        if not 0 < self.timeout_seconds <= MAX_CRAWL_TIMEOUT_SECONDS:
            raise ValueError(
                f"crawl timeout must be between 0 and {MAX_CRAWL_TIMEOUT_SECONDS} seconds"
            )
        self.__dict__.update(normalized)


@dataclass(frozen=True)
class CrawlResult:
    """Evidence, diagnostics, and metadata from a bounded crawl."""

    evidence: list[EvidenceRecord]
    diagnostics: list[Diagnostic]
    metadata: dict[str, Any]


PageFetcher = Callable[[str, float, int], FetchedUrlPage]


@dataclass(frozen=True)
class _QueuedUrl:
    url: str
    parent_url: str | None
    depth: int


def crawl_url_evidence(
    options: CrawlOptions,
    *,
    fetch_page: PageFetcher | None = None,
) -> CrawlResult:
    """Crawl explicit seed URLs and convert accepted pages into evidence records."""
    fetcher = fetch_page or _fetch_page
    diagnostics: list[Diagnostic] = []
    evidence: list[EvidenceRecord] = []
    seen: set[str] = set()
    fetched_count = 0
    rejected_count = 0
    total_bytes = 0
    seed_hosts = _seed_hosts(options.seed_urls)
    queue: deque[_QueuedUrl] = deque()
    for seed_url in options.seed_urls:
        queued = _validated_queue_item(
            seed_url,
            parent_url=None,
            depth=0,
            seed_hosts=seed_hosts,
            allow_domains=options.allow_domains or [],
            seen=seen,
            diagnostics=diagnostics,
        )
        if queued is None:
            rejected_count += 1
            continue
        queue.append(queued)

    while queue and fetched_count < options.max_pages and total_bytes < options.max_total_bytes:
        item = queue.popleft()
        remaining_total_bytes = options.max_total_bytes - total_bytes
        max_bytes = min(options.max_page_bytes, remaining_total_bytes)
        try:
            page = fetcher(item.url, options.timeout_seconds, max_bytes)
        except (OSError, TimeoutError, ValueError) as exc:
            rejected_count += 1
            diagnostics.append(
                Diagnostic(
                    level="warning",
                    code="CRAWL_FETCH_FAILED",
                    message=f"could not fetch crawl URL: {item.url} ({exc.__class__.__name__})",
                )
            )
            continue

        fetched_count += 1
        total_bytes += len(page.body)
        final_canonical = _canonical_url(page.final_url)
        if final_canonical in seen and final_canonical != _canonical_url(item.url):
            rejected_count += 1
            diagnostics.append(
                Diagnostic(
                    level="info",
                    code="CRAWL_URL_DUPLICATE",
                    message=f"skipped duplicate crawl redirect target: {page.final_url}",
                )
            )
            continue
        if not _url_allowed(
            page.final_url,
            seed_hosts=seed_hosts,
            allow_domains=options.allow_domains or [],
        ):
            rejected_count += 1
            diagnostics.append(
                Diagnostic(
                    level="warning",
                    code="CRAWL_URL_FILTERED",
                    message=f"crawl redirect target outside allowed domains: {page.final_url}",
                )
            )
            continue
        seen.add(final_canonical)

        record, links = _evidence_and_links(page, parent_url=item.parent_url, depth=item.depth)
        if record is None:
            rejected_count += 1
            diagnostics.append(
                Diagnostic(
                    level="warning",
                    code="CRAWL_PAGE_EMPTY",
                    message=f"crawl page produced no text evidence: {page.final_url}",
                )
            )
        else:
            evidence.append(record)

        if item.depth >= options.max_depth:
            if links:
                diagnostics.append(
                    Diagnostic(
                        level="info",
                        code="CRAWL_DEPTH_LIMIT_REACHED",
                        message=f"crawl depth limit reached at: {page.final_url}",
                    )
                )
            continue
        for link in links:
            queued = _validated_queue_item(
                link,
                parent_url=page.final_url,
                depth=item.depth + 1,
                seed_hosts=seed_hosts,
                allow_domains=options.allow_domains or [],
                seen=seen,
                diagnostics=diagnostics,
            )
            if queued is None:
                rejected_count += 1
                continue
            queue.append(queued)

    if queue and fetched_count >= options.max_pages:
        diagnostics.append(
            Diagnostic(
                level="info",
                code="CRAWL_PAGE_LIMIT_REACHED",
                message=f"crawl page limit reached at {options.max_pages} pages",
            )
        )
    if queue and total_bytes >= options.max_total_bytes:
        diagnostics.append(
            Diagnostic(
                level="info",
                code="CRAWL_TOTAL_BYTES_LIMIT_REACHED",
                message=f"crawl total byte limit reached at {options.max_total_bytes} bytes",
            )
        )

    return CrawlResult(
        evidence=evidence,
        diagnostics=diagnostics,
        metadata={
            "seed_count": len(options.seed_urls),
            "accepted_count": len(evidence),
            "rejected_count": rejected_count,
            "fetched_count": fetched_count,
            "queued_remaining": len(queue),
            "total_bytes": total_bytes,
            "limits": {
                "max_depth": options.max_depth,
                "max_pages": options.max_pages,
                "max_page_bytes": options.max_page_bytes,
                "max_total_bytes": options.max_total_bytes,
                "timeout_seconds": options.timeout_seconds,
            },
            "allow_domains": list(options.allow_domains or []),
            "seed_hosts": sorted(seed_hosts),
            "diagnostic_count": len(diagnostics),
        },
    )


def _fetch_page(url: str, timeout_seconds: float, max_bytes: int) -> FetchedUrlPage:
    return fetch_public_http_url(url, timeout_seconds=timeout_seconds, max_bytes=max_bytes)


def _validated_queue_item(
    url: str,
    *,
    parent_url: str | None,
    depth: int,
    seed_hosts: set[str],
    allow_domains: list[str],
    seen: set[str],
    diagnostics: list[Diagnostic],
) -> _QueuedUrl | None:
    try:
        normalized = validate_public_http_url(url)
    except ValueError as exc:
        diagnostics.append(
            Diagnostic(
                level="warning",
                code="CRAWL_URL_REJECTED",
                message=f"rejected crawl URL: {url} ({exc})",
            )
        )
        return None
    canonical = _canonical_url(normalized)
    if canonical in seen:
        diagnostics.append(
            Diagnostic(
                level="info",
                code="CRAWL_URL_DUPLICATE",
                message=f"skipped duplicate crawl URL: {normalized}",
            )
        )
        return None
    if not _url_allowed(normalized, seed_hosts=seed_hosts, allow_domains=allow_domains):
        diagnostics.append(
            Diagnostic(
                level="warning",
                code="CRAWL_URL_FILTERED",
                message=f"crawl URL outside allowed domains: {normalized}",
            )
        )
        return None
    seen.add(canonical)
    return _QueuedUrl(url=normalized, parent_url=parent_url, depth=depth)


def _evidence_and_links(
    page: FetchedUrlPage,
    *,
    parent_url: str | None,
    depth: int,
) -> tuple[EvidenceRecord | None, list[str]]:
    content_type = page.content_type or ""
    title: str | None = None
    links: list[str] = []
    if "html" in content_type.lower():
        parsed = _parse_html_page(page.text, page.final_url)
        span = parsed.text
        title = parsed.title
        links = parsed.links
    else:
        span = _normalize_text(page.text)

    if not span:
        return None, links
    return (
        EvidenceRecord(
            url=page.final_url,
            title=title,
            span=span,
            source_type="crawl",
            retrieved_at=utc_now_iso(),
            metadata={
                "parent_url": parent_url,
                "crawl_depth": depth,
                "content_type": page.content_type,
                "bytes_read": len(page.body),
                "truncated": page.truncated,
                "final_url": page.final_url,
            },
        ),
        links,
    )


@dataclass(frozen=True)
class _ParsedHtml:
    title: str | None
    text: str
    links: list[str]


class _EvidenceHTMLParser(HTMLParser):
    def __init__(self, base_url: str) -> None:
        super().__init__(convert_charrefs=True)
        self.base_url = base_url
        self.link_base_url = base_url
        self.title_parts: list[str] = []
        self.text_parts: list[str] = []
        self.links: list[str] = []
        self._in_title = False
        self._skip_depth = 0
        self._base_seen = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        normalized = tag.lower()
        if normalized in {"script", "style", "noscript"}:
            self._skip_depth += 1
            return
        if normalized == "base" and not self._base_seen:
            href = dict(attrs).get("href")
            if href:
                self.link_base_url = urljoin(self.base_url, href)
            self._base_seen = True
            return
        if normalized == "title":
            self._in_title = True
            return
        if normalized == "a":
            href = dict(attrs).get("href")
            if href:
                self.links.append(urljoin(self.link_base_url, href))
        if normalized in {"br", "p", "div", "section", "article", "li", "tr", "h1", "h2", "h3"}:
            self.text_parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        normalized = tag.lower()
        if normalized in {"script", "style", "noscript"} and self._skip_depth > 0:
            self._skip_depth -= 1
            return
        if normalized == "title":
            self._in_title = False
            return
        if normalized in {"p", "div", "section", "article", "li", "tr", "h1", "h2", "h3"}:
            self.text_parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._skip_depth:
            return
        if self._in_title:
            self.title_parts.append(data)
            return
        self.text_parts.append(data)


def _parse_html_page(text: str, base_url: str) -> _ParsedHtml:
    parser = _EvidenceHTMLParser(base_url)
    parser.feed(text)
    return _ParsedHtml(
        title=_normalize_text(" ".join(parser.title_parts)) or None,
        text=_normalize_text(" ".join(parser.text_parts)),
        links=dedupe_non_empty_text(parser.links),
    )


def _normalize_text(value: str) -> str:
    lines = [" ".join(line.split()) for line in value.splitlines()]
    return "\n".join(line for line in lines if line).strip()


def _seed_hosts(seed_urls: list[str]) -> set[str]:
    hosts: set[str] = set()
    for url in seed_urls:
        parsed = urlparse(url)
        host = (parsed.hostname or "").lower().rstrip(".")
        if host:
            hosts.add(host)
    return hosts


def _url_allowed(url: str, *, seed_hosts: set[str], allow_domains: list[str]) -> bool:
    host = (urlparse(url).hostname or "").lower().rstrip(".")
    if host in seed_hosts:
        return True
    return any(host_matches_domain(host, domain) for domain in allow_domains)


def _canonical_url(url: str) -> str:
    parsed = urlparse(url)
    scheme = parsed.scheme.lower()
    netloc = parsed.netloc.lower()
    path = parsed.path or "/"
    if path != "/":
        path = path.rstrip("/") or "/"
    return urlunparse((scheme, netloc, path, "", parsed.query, ""))
