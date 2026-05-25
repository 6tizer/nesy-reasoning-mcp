import json
from typing import Any
from urllib import error, request

import pytest

from nesy_reasoning_mcp.auto_ingest import search
from nesy_reasoning_mcp.auto_ingest.search import (
    SearchProviderName,
    SearchRetrievalOptions,
    retrieve_search_evidence,
)


class _Response:
    status = 200

    def __init__(self, payload: dict[str, Any]) -> None:
        self._body = json.dumps(payload).encode("utf-8")

    def __enter__(self) -> "_Response":
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def read(self, size: int) -> bytes:
        return self._body[:size]


def test_exa_search_response_converts_to_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(search, "validate_public_http_url", lambda url: url.strip())
    captured_body: dict[str, Any] = {}

    def fake_open(req: request.Request, timeout_seconds: float) -> _Response:
        captured_body.update(json.loads(req.data or b"{}"))
        assert req.full_url == search.EXA_SEARCH_ENDPOINT
        assert timeout_seconds == 3
        assert req.get_header("X-api-key") == "secret"
        return _Response(
            {
                "requestId": "request-1",
                "searchType": "auto",
                "results": [
                    {
                        "id": "result-1",
                        "title": "Example Source",
                        "url": "https://www.example.com/source",
                        "publishedDate": "2026-01-01T00:00:00Z",
                        "score": 0.92,
                        "highlights": ["A requires B."],
                    }
                ],
            }
        )

    result = retrieve_search_evidence(
        SearchRetrievalOptions(
            queries=["A requires B"],
            limit=2,
            timeout_seconds=3,
            include_domains=["example.com"],
            exclude_domains=["blocked.example.com"],
        ),
        env={"EXA_API_KEY": "secret"},
        urlopen=fake_open,
    )

    assert result.diagnostics == []
    assert captured_body == {
        "query": "A requires B",
        "type": "auto",
        "numResults": 2,
        "contents": {"highlights": {"query": "A requires B", "maxCharacters": 2000}},
        "includeDomains": ["example.com"],
        "excludeDomains": ["blocked.example.com"],
    }
    record = result.evidence[0]
    assert record.url == "https://www.example.com/source"
    assert record.title == "Example Source"
    assert record.span == "A requires B."
    assert record.source_type == "search"
    assert record.metadata == {
        "provider": "exa",
        "query": "A requires B",
        "rank": 1,
        "result_id": "result-1",
        "score": 0.92,
        "published_date": "2026-01-01T00:00:00Z",
        "search_type": "auto",
        "request_id": "request-1",
    }
    assert result.metadata["accepted_count"] == 1
    assert result.metadata["domain_filters"] == {
        "include": ["example.com"],
        "exclude": ["blocked.example.com"],
    }


def test_search_domain_filters_are_enforced_locally(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(search, "validate_public_http_url", lambda url: url.strip())

    def fake_open(req: request.Request, timeout_seconds: float) -> _Response:
        return _Response(
            {
                "results": [
                    {
                        "title": "Allowed",
                        "url": "https://sub.example.com/source",
                        "highlights": ["Allowed evidence."],
                    },
                    {
                        "title": "Blocked",
                        "url": "https://blocked.example.com/source",
                        "highlights": ["Blocked evidence."],
                    },
                    {
                        "title": "Other",
                        "url": "https://other.test/source",
                        "highlights": ["Other evidence."],
                    },
                ]
            }
        )

    result = retrieve_search_evidence(
        SearchRetrievalOptions(
            queries=["query"],
            include_domains=["example.com"],
            exclude_domains=["blocked.example.com"],
        ),
        env={"EXA_API_KEY": "secret"},
        urlopen=fake_open,
    )

    assert [record.url for record in result.evidence] == ["https://sub.example.com/source"]
    assert [diagnostic.code for diagnostic in result.diagnostics] == [
        "SEARCH_RESULT_FILTERED",
        "SEARCH_RESULT_FILTERED",
    ]
    assert result.metadata["rejected_count"] == 2


def test_search_rejects_unsafe_result_url() -> None:
    def fake_open(req: request.Request, timeout_seconds: float) -> _Response:
        return _Response(
            {
                "results": [
                    {
                        "title": "Local",
                        "url": "http://127.0.0.1/source",
                        "highlights": ["Local evidence."],
                    }
                ]
            }
        )

    result = retrieve_search_evidence(
        SearchRetrievalOptions(queries=["query"]),
        env={"EXA_API_KEY": "secret"},
        urlopen=fake_open,
    )

    assert result.evidence == []
    assert [diagnostic.code for diagnostic in result.diagnostics] == [
        "SEARCH_RESULT_REJECTED",
        "SEARCH_NO_ACCEPTED_RESULTS",
    ]


def test_search_provider_failure_returns_diagnostic() -> None:
    def fake_open(req: request.Request, timeout_seconds: float) -> _Response:
        raise error.HTTPError(
            req.full_url,
            429,
            "rate limited",
            hdrs=None,
            fp=None,
        )

    result = retrieve_search_evidence(
        SearchRetrievalOptions(queries=["query"]),
        env={"EXA_API_KEY": "secret"},
        urlopen=fake_open,
    )

    assert result.evidence == []
    assert result.has_errors is True
    assert result.diagnostics[0].code == "EXA_SEARCH_HTTP_ERROR"
    assert "secret" not in result.diagnostics[0].message


def test_search_missing_api_key_does_not_call_provider() -> None:
    def fail_open(req: request.Request, timeout_seconds: float) -> _Response:
        raise AssertionError("provider must not be called without an API key")

    result = retrieve_search_evidence(
        SearchRetrievalOptions(queries=["query"], provider=SearchProviderName.EXA),
        env={},
        urlopen=fail_open,
    )

    assert result.evidence == []
    assert result.has_errors is True
    assert result.diagnostics[0].code == "SEARCH_API_KEY_MISSING"


def test_search_options_normalize_provider_string() -> None:
    options = SearchRetrievalOptions(queries=["query"], provider="exa")  # type: ignore[arg-type]

    assert options.provider is SearchProviderName.EXA


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"queries": [""]}, "queries"),
        ({"queries": ["query"], "limit": 0}, "limit"),
        ({"queries": ["query"], "limit": 21}, "limit"),
        ({"queries": ["query"], "timeout_seconds": 0}, "timeout"),
        ({"queries": ["query"], "timeout_seconds": 31}, "timeout"),
        ({"queries": ["query"], "include_domains": ["localhost"]}, "local"),
    ],
)
def test_search_options_validate_bounds(kwargs: dict[str, Any], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        SearchRetrievalOptions(**kwargs)
