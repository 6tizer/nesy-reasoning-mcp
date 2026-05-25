from collections.abc import Callable
from urllib.parse import urlparse

import pytest

from nesy_reasoning_mcp.auto_ingest import crawler
from nesy_reasoning_mcp.auto_ingest.crawler import CrawlOptions, crawl_url_evidence
from nesy_reasoning_mcp.auto_ingest.fetcher import FetchedUrlPage


def _page(
    url: str,
    body: str,
    *,
    content_type: str = "text/html; charset=utf-8",
    final_url: str | None = None,
    truncated: bool = False,
) -> FetchedUrlPage:
    return FetchedUrlPage(
        requested_url=url,
        final_url=final_url or url,
        body=body.encode("utf-8"),
        content_type=content_type,
        charset="utf-8",
        truncated=truncated,
    )


def _fake_validate(url: str) -> str:
    stripped = url.strip()
    parsed = urlparse(stripped)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError("only explicit http/https URLs are supported")
    if parsed.hostname in {"localhost", "127.0.0.1"}:
        raise ValueError("local URLs are not supported")
    return stripped


def _mapping_fetcher(
    pages: dict[str, FetchedUrlPage],
) -> Callable[[str, float, int], FetchedUrlPage]:
    def fetch(url: str, timeout_seconds: float, max_bytes: int) -> FetchedUrlPage:
        page = pages[url]
        body = page.body[: max_bytes + 1]
        truncated = len(body) > max_bytes
        if truncated:
            body = body[:max_bytes]
        return FetchedUrlPage(
            requested_url=page.requested_url,
            final_url=page.final_url,
            body=body,
            content_type=page.content_type,
            charset=page.charset,
            truncated=truncated,
        )

    return fetch


def test_crawler_emits_seed_and_child_evidence_with_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(crawler, "validate_public_http_url", _fake_validate)
    pages = {
        "https://example.com/index": _page(
            "https://example.com/index",
            "<html><head><title>Home</title><style>hidden</style></head>"
            "<body><h1>Hello</h1><script>ignored</script><a href='/child'>child</a></body></html>",
        ),
        "https://example.com/child": _page(
            "https://example.com/child",
            "<html><body><p>Child page text.</p></body></html>",
        ),
    }

    result = crawl_url_evidence(
        CrawlOptions(seed_urls=["https://example.com/index"], max_depth=1),
        fetch_page=_mapping_fetcher(pages),
    )

    assert [record.url for record in result.evidence] == [
        "https://example.com/index",
        "https://example.com/child",
    ]
    assert result.evidence[0].title == "Home"
    assert "Hello" in result.evidence[0].span
    assert "hidden" not in result.evidence[0].span
    assert "ignored" not in result.evidence[0].span
    assert result.evidence[0].metadata["parent_url"] is None
    assert result.evidence[0].metadata["crawl_depth"] == 0
    assert result.evidence[1].metadata["parent_url"] == "https://example.com/index"
    assert result.evidence[1].metadata["crawl_depth"] == 1
    assert result.metadata["accepted_count"] == 2


def test_crawler_deduplicates_urls_without_fragments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(crawler, "validate_public_http_url", _fake_validate)
    fetched: list[str] = []
    pages = {
        "https://example.com/index": _page(
            "https://example.com/index",
            "<a href='/child#one'>one</a><a href='/child#two'>two</a><a href='/child'>plain</a>",
        ),
        "https://example.com/child#one": _page(
            "https://example.com/child#one",
            "<p>Child one.</p>",
        ),
    }

    def fetch(url: str, timeout_seconds: float, max_bytes: int) -> FetchedUrlPage:
        fetched.append(url)
        return pages[url]

    result = crawl_url_evidence(
        CrawlOptions(seed_urls=["https://example.com/index"], max_depth=1),
        fetch_page=fetch,
    )

    assert fetched == ["https://example.com/index", "https://example.com/child#one"]
    assert [diagnostic.code for diagnostic in result.diagnostics].count("CRAWL_URL_DUPLICATE") == 2


def test_crawler_deduplicates_trailing_slash_variants(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(crawler, "validate_public_http_url", _fake_validate)
    fetched: list[str] = []
    pages = {
        "https://example.com/index": _page(
            "https://example.com/index",
            "<a href='/child'>plain</a><a href='/child/'>slash</a>",
        ),
        "https://example.com/child": _page(
            "https://example.com/child",
            "<p>Child text.</p>",
        ),
    }

    def fetch(url: str, timeout_seconds: float, max_bytes: int) -> FetchedUrlPage:
        fetched.append(url)
        return pages[url]

    result = crawl_url_evidence(
        CrawlOptions(seed_urls=["https://example.com/index"], max_depth=1),
        fetch_page=fetch,
    )

    assert fetched == ["https://example.com/index", "https://example.com/child"]
    assert [record.url for record in result.evidence] == [
        "https://example.com/index",
        "https://example.com/child",
    ]
    assert any(diagnostic.code == "CRAWL_URL_DUPLICATE" for diagnostic in result.diagnostics)


def test_crawler_deduplicates_redirect_final_urls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(crawler, "validate_public_http_url", _fake_validate)
    pages = {
        "https://example.com/target": _page(
            "https://example.com/target",
            "<p>Target text.</p>",
        ),
        "https://example.com/redirect": _page(
            "https://example.com/redirect",
            "<p>Redirect text.</p>",
            final_url="https://example.com/target",
        ),
    }

    result = crawl_url_evidence(
        CrawlOptions(
            seed_urls=["https://example.com/target", "https://example.com/redirect"],
            max_depth=0,
        ),
        fetch_page=_mapping_fetcher(pages),
    )

    assert [record.url for record in result.evidence] == ["https://example.com/target"]
    assert any(diagnostic.code == "CRAWL_URL_DUPLICATE" for diagnostic in result.diagnostics)


def test_crawler_enforces_same_host_and_allow_domains(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(crawler, "validate_public_http_url", _fake_validate)
    pages = {
        "https://example.com/index": _page(
            "https://example.com/index",
            "<a href='https://sub.allowed.com/page'>allowed</a>"
            "<a href='https://other.test/page'>other</a>",
        ),
        "https://sub.allowed.com/page": _page(
            "https://sub.allowed.com/page",
            "<p>Allowed child.</p>",
        ),
    }

    result = crawl_url_evidence(
        CrawlOptions(
            seed_urls=["https://example.com/index"],
            max_depth=1,
            allow_domains=["allowed.com"],
        ),
        fetch_page=_mapping_fetcher(pages),
    )

    assert [record.url for record in result.evidence] == [
        "https://example.com/index",
        "https://sub.allowed.com/page",
    ]
    assert any(diagnostic.code == "CRAWL_URL_FILTERED" for diagnostic in result.diagnostics)


def test_crawler_resolves_links_against_first_base_href(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(crawler, "validate_public_http_url", _fake_validate)
    pages = {
        "https://example.com/index": _page(
            "https://example.com/index",
            "<html><head><base href='https://docs.example.com/root/'>"
            "<base href='https://ignored.example.com/'></head>"
            "<body><a href='guide'>guide</a><p>Seed text.</p></body></html>",
        ),
        "https://docs.example.com/root/guide": _page(
            "https://docs.example.com/root/guide",
            "<p>Guide text.</p>",
        ),
    }

    result = crawl_url_evidence(
        CrawlOptions(
            seed_urls=["https://example.com/index"],
            max_depth=1,
            allow_domains=["docs.example.com"],
        ),
        fetch_page=_mapping_fetcher(pages),
    )

    assert [record.url for record in result.evidence] == [
        "https://example.com/index",
        "https://docs.example.com/root/guide",
    ]
    assert result.evidence[1].metadata["parent_url"] == "https://example.com/index"


def test_crawler_depth_page_and_total_byte_limits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(crawler, "validate_public_http_url", _fake_validate)
    pages = {
        "https://example.com/index": _page(
            "https://example.com/index",
            "<a href='/child'>child</a><p>Seed text.</p>",
        ),
        "https://example.com/limit": _page(
            "https://example.com/limit",
            "<a href='/a'>a</a><a href='/b'>b</a>",
        ),
        "https://example.com/child": _page(
            "https://example.com/child",
            "<p>Child text.</p>",
        ),
        "https://example.com/a": _page("https://example.com/a", "<p>A text.</p>"),
        "https://example.com/b": _page("https://example.com/b", "<p>B text.</p>"),
    }

    depth_result = crawl_url_evidence(
        CrawlOptions(seed_urls=["https://example.com/index"], max_depth=0),
        fetch_page=_mapping_fetcher(pages),
    )
    page_result = crawl_url_evidence(
        CrawlOptions(seed_urls=["https://example.com/index"], max_depth=1, max_pages=1),
        fetch_page=_mapping_fetcher(pages),
    )
    byte_result = crawl_url_evidence(
        CrawlOptions(seed_urls=["https://example.com/limit"], max_depth=1, max_total_bytes=40),
        fetch_page=_mapping_fetcher(pages),
    )

    assert [record.url for record in depth_result.evidence] == ["https://example.com/index"]
    assert any(
        diagnostic.code == "CRAWL_DEPTH_LIMIT_REACHED" for diagnostic in depth_result.diagnostics
    )
    assert any(
        diagnostic.code == "CRAWL_PAGE_LIMIT_REACHED" for diagnostic in page_result.diagnostics
    )
    assert any(
        diagnostic.code == "CRAWL_TOTAL_BYTES_LIMIT_REACHED"
        for diagnostic in byte_result.diagnostics
    )


def test_crawler_rejects_unsafe_seed_without_fetch() -> None:
    def fail_fetch(url: str, timeout_seconds: float, max_bytes: int) -> FetchedUrlPage:
        raise AssertionError("unsafe seed must not be fetched")

    result = crawl_url_evidence(
        CrawlOptions(seed_urls=["http://127.0.0.1/source"]),
        fetch_page=fail_fetch,
    )

    assert result.evidence == []
    assert result.diagnostics[0].code == "CRAWL_URL_REJECTED"


def test_crawler_text_plain_does_not_discover_links(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(crawler, "validate_public_http_url", _fake_validate)
    pages = {
        "https://example.com/index": _page(
            "https://example.com/index",
            "Plain text with https://example.com/child",
            content_type="text/plain; charset=utf-8",
        )
    }

    result = crawl_url_evidence(
        CrawlOptions(seed_urls=["https://example.com/index"], max_depth=1),
        fetch_page=_mapping_fetcher(pages),
    )

    assert [record.url for record in result.evidence] == ["https://example.com/index"]
    assert result.diagnostics == []


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"seed_urls": []}, "seed"),
        ({"seed_urls": ["https://example.com"], "max_depth": 4}, "depth"),
        ({"seed_urls": ["https://example.com"], "max_pages": 0}, "pages"),
        ({"seed_urls": ["https://example.com"], "max_page_bytes": 0}, "page bytes"),
        ({"seed_urls": ["https://example.com"], "max_total_bytes": 0}, "total bytes"),
        ({"seed_urls": ["https://example.com"], "timeout_seconds": 31}, "timeout"),
        ({"seed_urls": ["https://example.com"], "allow_domains": ["localhost"]}, "local"),
    ],
)
def test_crawl_options_validate_bounds(kwargs: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        CrawlOptions(**kwargs)
