"""Explicit search retrieval for Agent SDK ingestion evidence."""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Any
from urllib import error, request
from urllib.parse import urlparse

from nesy_reasoning_mcp.auto_ingest.fetcher import validate_public_http_url
from nesy_reasoning_mcp.auto_ingest.schemas import EvidenceRecord
from nesy_reasoning_mcp.auto_ingest.text import dedupe_non_empty_text
from nesy_reasoning_mcp.auto_ingest.url_safety import (
    host_matches_domain,
    normalize_domain_filters,
)
from nesy_reasoning_mcp.schemas import Diagnostic
from nesy_reasoning_mcp.time_utils import utc_now_iso

EXA_SEARCH_ENDPOINT = "https://api.exa.ai/search"
DEFAULT_SEARCH_PROVIDER = "exa"
DEFAULT_SEARCH_API_KEY_ENV = "EXA_API_KEY"
DEFAULT_SEARCH_LIMIT = 5
MAX_SEARCH_LIMIT = 20
DEFAULT_SEARCH_TIMEOUT_SECONDS = 10.0
MAX_SEARCH_TIMEOUT_SECONDS = 30.0
DEFAULT_SEARCH_SPAN_CHARS = 2_000
DEFAULT_SEARCH_RESPONSE_BYTES = 500_000

_USER_AGENT = "nesy-reasoning-mcp-agent-ingest/1.0"


class SearchProviderName(StrEnum):
    """Supported search evidence providers."""

    EXA = "exa"


@dataclass(frozen=True)
class SearchRetrievalOptions:
    """Bounded search request options for evidence retrieval."""

    queries: list[str]
    provider: SearchProviderName = SearchProviderName.EXA
    limit: int = DEFAULT_SEARCH_LIMIT
    timeout_seconds: float = DEFAULT_SEARCH_TIMEOUT_SECONDS
    include_domains: list[str] | None = None
    exclude_domains: list[str] | None = None
    api_key_env: str = DEFAULT_SEARCH_API_KEY_ENV
    max_response_bytes: int = DEFAULT_SEARCH_RESPONSE_BYTES

    def __post_init__(self) -> None:
        normalized = {
            "queries": dedupe_non_empty_text(self.queries),
            "provider": SearchProviderName(self.provider),
            "include_domains": normalize_domain_filters(self.include_domains or []),
            "exclude_domains": normalize_domain_filters(self.exclude_domains or []),
            "api_key_env": self.api_key_env.strip(),
        }
        if not normalized["queries"]:
            raise ValueError("search queries must not be empty")
        if not 1 <= self.limit <= MAX_SEARCH_LIMIT:
            raise ValueError(f"search limit must be between 1 and {MAX_SEARCH_LIMIT}")
        if not 0 < self.timeout_seconds <= MAX_SEARCH_TIMEOUT_SECONDS:
            raise ValueError(
                f"search timeout must be between 0 and {MAX_SEARCH_TIMEOUT_SECONDS} seconds"
            )
        if not normalized["api_key_env"]:
            raise ValueError("search API key env var name must not be empty")
        if self.max_response_bytes < 1:
            raise ValueError("search response byte limit must be positive")
        self.__dict__.update(normalized)


@dataclass(frozen=True)
class SearchRetrievalResult:
    """Evidence, diagnostics, and metadata returned by a search retrieval run."""

    evidence: list[EvidenceRecord]
    diagnostics: list[Diagnostic]
    metadata: dict[str, Any]

    @property
    def has_errors(self) -> bool:
        """Return whether retrieval produced a blocking error diagnostic."""
        return any(diagnostic.level == "error" for diagnostic in self.diagnostics)


UrlOpen = Callable[[request.Request, float], Any]


def retrieve_search_evidence(
    options: SearchRetrievalOptions,
    *,
    env: Mapping[str, str] | None = None,
    urlopen: UrlOpen | None = None,
) -> SearchRetrievalResult:
    """Search configured providers and convert accepted results into evidence records."""
    env = env if env is not None else os.environ
    opener = urlopen or _open_url
    api_key = env.get(options.api_key_env, "").strip()
    metadata = _base_metadata(options)
    if not api_key:
        diagnostic = Diagnostic(
            level="error",
            code="SEARCH_API_KEY_MISSING",
            message=f"{options.api_key_env} is not set",
        )
        metadata["diagnostic_count"] = 1
        return SearchRetrievalResult(evidence=[], diagnostics=[diagnostic], metadata=metadata)
    if options.provider is not SearchProviderName.EXA:
        diagnostic = Diagnostic(
            level="error",
            code="SEARCH_PROVIDER_UNSUPPORTED",
            message=f"unsupported search provider: {options.provider.value}",
        )
        metadata["diagnostic_count"] = 1
        return SearchRetrievalResult(evidence=[], diagnostics=[diagnostic], metadata=metadata)

    evidence: list[EvidenceRecord] = []
    diagnostics: list[Diagnostic] = []
    query_metadata: list[dict[str, Any]] = []
    for query in options.queries:
        query_result = _retrieve_exa_query(
            query=query,
            options=options,
            api_key=api_key,
            urlopen=opener,
        )
        evidence.extend(query_result.evidence)
        diagnostics.extend(query_result.diagnostics)
        query_metadata.append(query_result.metadata)

    metadata.update(
        {
            "queries": query_metadata,
            "accepted_count": len(evidence),
            "rejected_count": sum(int(item.get("rejected_count", 0)) for item in query_metadata),
            "diagnostic_count": len(diagnostics),
        }
    )
    return SearchRetrievalResult(evidence=evidence, diagnostics=diagnostics, metadata=metadata)


@dataclass(frozen=True)
class _QueryResult:
    evidence: list[EvidenceRecord]
    diagnostics: list[Diagnostic]
    metadata: dict[str, Any]


def _retrieve_exa_query(
    *,
    query: str,
    options: SearchRetrievalOptions,
    api_key: str,
    urlopen: UrlOpen,
) -> _QueryResult:
    request_payload = _exa_request_payload(query, options)
    req = request.Request(
        EXA_SEARCH_ENDPOINT,
        data=json.dumps(request_payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "User-Agent": _USER_AGENT,
            "x-api-key": api_key,
        },
        method="POST",
    )
    base_metadata: dict[str, Any] = {
        "query": query,
        "requested_limit": options.limit,
        "accepted_count": 0,
        "rejected_count": 0,
    }
    try:
        payload = _load_json_response(req, options, urlopen)
    except error.HTTPError as exc:
        return _query_error(
            base_metadata,
            code="EXA_SEARCH_HTTP_ERROR",
            message=f"Exa search failed with HTTP {exc.code}",
        )
    except (OSError, TimeoutError) as exc:
        return _query_error(
            base_metadata,
            code="EXA_SEARCH_REQUEST_FAILED",
            message=f"Exa search request failed: {exc.__class__.__name__}",
        )
    except ValueError as exc:
        return _query_error(
            base_metadata,
            code="EXA_SEARCH_RESPONSE_INVALID",
            message=str(exc),
        )

    request_id = _optional_str(payload.get("requestId"))
    search_type = _optional_str(payload.get("searchType"))
    results = payload.get("results")
    if not isinstance(results, list):
        return _query_error(
            {
                **base_metadata,
                "request_id": request_id,
                "search_type": search_type,
            },
            code="EXA_SEARCH_RESPONSE_INVALID",
            message="Exa search response missing results",
        )

    evidence: list[EvidenceRecord] = []
    diagnostics: list[Diagnostic] = []
    rejected_count = 0
    for index, raw_result in enumerate(results[: options.limit], start=1):
        if not isinstance(raw_result, dict):
            rejected_count += 1
            continue
        record, diagnostic = _evidence_from_exa_result(
            raw_result,
            query=query,
            rank=index,
            request_id=request_id,
            search_type=search_type,
            options=options,
        )
        if record is None:
            rejected_count += 1
            if diagnostic is not None:
                diagnostics.append(diagnostic)
            continue
        evidence.append(record)

    if not evidence:
        diagnostics.append(
            Diagnostic(
                level="warning",
                code="SEARCH_NO_ACCEPTED_RESULTS",
                message=f"no Exa search results accepted for query: {query}",
            )
        )

    metadata = {
        **base_metadata,
        "request_id": request_id,
        "search_type": search_type,
        "accepted_count": len(evidence),
        "rejected_count": rejected_count,
    }
    return _QueryResult(evidence=evidence, diagnostics=diagnostics, metadata=metadata)


def _load_json_response(
    req: request.Request,
    options: SearchRetrievalOptions,
    urlopen: UrlOpen,
) -> dict[str, Any]:
    with urlopen(req, options.timeout_seconds) as response:
        status = int(getattr(response, "status", 200) or 200)
        if status >= 400:
            raise ValueError(f"Exa search failed with HTTP {status}")
        _set_response_read_timeout(response, options.timeout_seconds)
        body = response.read(options.max_response_bytes + 1)
    if len(body) > options.max_response_bytes:
        raise ValueError("Exa search response exceeded byte limit")
    parsed = json.loads(body.decode("utf-8"))
    if not isinstance(parsed, dict):
        raise ValueError("Exa search response must be a JSON object")
    return parsed


def _evidence_from_exa_result(
    raw_result: dict[str, Any],
    *,
    query: str,
    rank: int,
    request_id: str | None,
    search_type: str | None,
    options: SearchRetrievalOptions,
) -> tuple[EvidenceRecord | None, Diagnostic | None]:
    url = _optional_str(raw_result.get("url"))
    if not url:
        return None, _result_diagnostic("SEARCH_RESULT_REJECTED", "search result missing URL")
    try:
        normalized_url = validate_public_http_url(url)
    except ValueError as exc:
        return None, _result_diagnostic("SEARCH_RESULT_REJECTED", str(exc), url=url)

    allowed, reason = _domain_allowed(
        normalized_url,
        include_domains=options.include_domains or [],
        exclude_domains=options.exclude_domains or [],
    )
    if not allowed:
        return None, _result_diagnostic("SEARCH_RESULT_FILTERED", reason, url=normalized_url)

    span = _result_span(raw_result)
    if not span:
        return None, _result_diagnostic("SEARCH_RESULT_REJECTED", "search result missing span")

    metadata = {
        "provider": options.provider.value,
        "query": query,
        "rank": rank,
        "result_id": _optional_str(raw_result.get("id")),
        "score": _optional_float(raw_result.get("score")),
        "published_date": _optional_str(raw_result.get("publishedDate")),
        "search_type": search_type,
        "request_id": request_id,
    }
    return (
        EvidenceRecord(
            url=normalized_url,
            title=_optional_str(raw_result.get("title")),
            span=span,
            source_type="search",
            retrieved_at=utc_now_iso(),
            metadata={key: value for key, value in metadata.items() if value is not None},
        ),
        None,
    )


def _exa_request_payload(query: str, options: SearchRetrievalOptions) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "query": query,
        "type": "auto",
        "numResults": options.limit,
        "contents": {
            "highlights": {
                "query": query,
                "maxCharacters": DEFAULT_SEARCH_SPAN_CHARS,
            }
        },
    }
    if options.include_domains:
        payload["includeDomains"] = list(options.include_domains)
    if options.exclude_domains:
        payload["excludeDomains"] = list(options.exclude_domains)
    return payload


def _result_span(raw_result: dict[str, Any]) -> str | None:
    highlights = raw_result.get("highlights")
    if isinstance(highlights, list):
        joined = "\n".join(str(item).strip() for item in highlights if str(item).strip())
        if joined:
            return joined[:DEFAULT_SEARCH_SPAN_CHARS]
    text = raw_result.get("text")
    if isinstance(text, str) and text.strip():
        return text.strip()[:DEFAULT_SEARCH_SPAN_CHARS]
    summary = raw_result.get("summary")
    if isinstance(summary, str) and summary.strip():
        return summary.strip()[:DEFAULT_SEARCH_SPAN_CHARS]
    title = raw_result.get("title")
    if isinstance(title, str) and title.strip():
        return title.strip()[:DEFAULT_SEARCH_SPAN_CHARS]
    return None


def _domain_allowed(
    url: str,
    *,
    include_domains: list[str],
    exclude_domains: list[str],
) -> tuple[bool, str]:
    host = (urlparse(url).hostname or "").lower().rstrip(".")
    if any(host_matches_domain(host, domain) for domain in exclude_domains):
        return False, "search result blocked by excluded domain"
    if include_domains and not any(host_matches_domain(host, domain) for domain in include_domains):
        return False, "search result not in included domains"
    return True, ""


def _base_metadata(options: SearchRetrievalOptions) -> dict[str, Any]:
    return {
        "provider": options.provider.value,
        "query_count": len(options.queries),
        "requested_limit": options.limit,
        "domain_filters": {
            "include": list(options.include_domains or []),
            "exclude": list(options.exclude_domains or []),
        },
        "accepted_count": 0,
        "rejected_count": 0,
        "diagnostic_count": 0,
    }


def _query_error(metadata: dict[str, Any], *, code: str, message: str) -> _QueryResult:
    return _QueryResult(
        evidence=[],
        diagnostics=[Diagnostic(level="error", code=code, message=message)],
        metadata={**metadata, "accepted_count": 0, "rejected_count": 0},
    )


def _result_diagnostic(code: str, message: str, *, url: str | None = None) -> Diagnostic:
    return Diagnostic(
        level="warning",
        code=code,
        message=f"{message}: {url}" if url else message,
    )


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _optional_float(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int | float):
        return float(value)
    return None


def _set_response_read_timeout(response: Any, timeout_seconds: float) -> None:
    for path in (
        ("fp", "raw", "_sock"),
        ("fp", "_sock"),
        ("raw", "_sock"),
        ("_sock",),
    ):
        target = _nested_attr(response, path)
        settimeout = getattr(target, "settimeout", None)
        if callable(settimeout):
            settimeout(timeout_seconds)
            return


def _nested_attr(value: Any, path: tuple[str, ...]) -> Any:
    current = value
    for item in path:
        current = getattr(current, item, None)
        if current is None:
            return None
    return current


def _open_url(req: request.Request, timeout_seconds: float) -> Any:
    return request.urlopen(req, timeout=timeout_seconds)
