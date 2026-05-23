import json
from concurrent.futures import ThreadPoolExecutor

import pytest
from starlette.testclient import TestClient

from nesy_reasoning_mcp.config import HttpConfig, NesyConfig, StorageConfig
from nesy_reasoning_mcp.http_server import create_http_app
from nesy_reasoning_mcp.store import RelationStore, SqliteRelationStore


def _config(**overrides) -> NesyConfig:
    defaults = {
        "local_token": "secret",
        "allowed_hosts": ["testserver"],
        "allowed_origins": ["http://allowed.test"],
    }
    defaults.update(overrides)
    return NesyConfig(http=HttpConfig(**defaults))


def test_http_app_requires_local_token() -> None:
    with pytest.raises(ValueError, match="NESY_LOCAL_TOKEN is required"):
        create_http_app(NesyConfig(), RelationStore())


def test_http_health_requires_bearer_token() -> None:
    config = _config()
    app = create_http_app(config, RelationStore(config))

    with TestClient(app) as client:
        rejected = client.get("/healthz")
        accepted = client.get("/healthz", headers={"Authorization": "Bearer secret"})

    assert rejected.status_code == 401
    assert accepted.status_code == 200
    assert accepted.json()["version"] == "1.0.0"


def test_http_rejects_bad_origin() -> None:
    config = _config()
    app = create_http_app(config, RelationStore(config))

    with TestClient(app) as client:
        rejected = client.get(
            "/healthz",
            headers={"Authorization": "Bearer secret", "Origin": "http://bad.test"},
        )
        accepted = client.get(
            "/healthz",
            headers={"Authorization": "Bearer secret", "Origin": "http://allowed.test"},
        )

    assert rejected.status_code == 403
    assert accepted.status_code == 200


def test_http_rejects_large_body() -> None:
    config = _config(max_body_bytes=5)
    app = create_http_app(config, RelationStore(config))

    with TestClient(app) as client:
        response = client.post(
            "/mcp",
            headers={
                "Authorization": "Bearer secret",
                "Accept": "application/json, text/event-stream",
                "Content-Type": "application/json",
            },
            content=b"123456",
        )

    assert response.status_code == 413
    assert response.json()["error"] == "request_body_too_large"


def test_http_rate_limit() -> None:
    config = _config(rate_limit_per_minute=1)
    app = create_http_app(config, RelationStore(config))

    with TestClient(app) as client:
        first = client.get("/healthz", headers={"Authorization": "Bearer secret"})
        second = client.get("/healthz", headers={"Authorization": "Bearer secret"})

    assert first.status_code == 200
    assert second.status_code == 429


def test_http_with_sqlite_store_allows_concurrent_tool_calls(tmp_path) -> None:
    config = NesyConfig(
        http=_config().http,
        storage=StorageConfig(backend="sqlite", sqlite_path=str(tmp_path / "nesy.db")),
    )
    store = SqliteRelationStore(config)
    app = create_http_app(config, store)
    headers = {
        "Authorization": "Bearer secret",
        "Accept": "application/json, text/event-stream",
        "Content-Type": "application/json",
    }

    with TestClient(app) as client:
        session_id = _initialize_http_session(client, headers)

        def call_assert(index: int) -> dict:
            response = client.post(
                "/mcp",
                headers={**headers, "mcp-session-id": session_id},
                json={
                    "jsonrpc": "2.0",
                    "id": index + 2,
                    "method": "tools/call",
                    "params": {
                        "name": "nesy.assert_relations",
                        "arguments": {
                            "relations": [
                                {
                                    "id": f"rel_{index}",
                                    "source": f"A{index}",
                                    "target": f"B{index}",
                                    "relation_type": "sufficient",
                                }
                            ],
                            "check_contradictions": False,
                        },
                    },
                },
            )
            return _sse_payload(response)

        with ThreadPoolExecutor(max_workers=6) as pool:
            payloads = list(pool.map(call_assert, range(12)))

    assert all(item["result"]["structuredContent"]["status"] == "ok" for item in payloads)
    assert len(store.list_relations()) == 12


def test_http_load_and_check_supports_proposition_registry() -> None:
    config = _config()
    store = RelationStore(config)
    app = create_http_app(config, store)
    headers = {
        "Authorization": "Bearer secret",
        "Accept": "application/json, text/event-stream",
        "Content-Type": "application/json",
    }

    with TestClient(app) as client:
        session_id = _initialize_http_session(client, headers)
        load_response = client.post(
            "/mcp",
            headers={**headers, "mcp-session-id": session_id},
            json={
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/call",
                "params": {
                    "name": "nesy.load_relations",
                    "arguments": {
                        "source_type": "inline",
                        "data": {
                            "propositions": [
                                {
                                    "id": "profit_up",
                                    "label": "Profit increases",
                                    "aliases": ["利润增加"],
                                },
                                {
                                    "id": "profit_not_up",
                                    "label": "Profit does not increase",
                                    "aliases": ["利润未增加"],
                                    "negates": "profit_up",
                                },
                            ]
                        },
                        "check_contradictions": False,
                    },
                },
            },
        )
        check_response = client.post(
            "/mcp",
            headers={**headers, "mcp-session-id": session_id},
            json={
                "jsonrpc": "2.0",
                "id": 3,
                "method": "tools/call",
                "params": {
                    "name": "nesy.check_contradictions",
                    "arguments": {
                        "mode": "facts",
                        "include_soft": False,
                        "facts": [
                            {
                                "source": "Discount",
                                "target": "利润增加",
                                "relation_type": "sufficient",
                            },
                            {
                                "source": "Discount",
                                "target": "利润未增加",
                                "relation_type": "sufficient",
                            },
                        ],
                    },
                },
            },
        )

    loaded = _sse_payload(load_response)
    checked = _sse_payload(check_response)

    assert loaded["result"]["structuredContent"]["loaded_propositions"] == 2
    assert checked["result"]["structuredContent"]["has_contradictions"] is True
    assert checked["result"]["structuredContent"]["contradictions"][0]["targets"] == [
        "profit_up",
        "profit_not_up",
    ]


def _initialize_http_session(client: TestClient, headers: dict[str, str]) -> str:
    response = client.post(
        "/mcp",
        headers=headers,
        json={
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2025-06-18",
                "capabilities": {},
                "clientInfo": {"name": "test", "version": "0"},
            },
        },
    )
    session_id = response.headers["mcp-session-id"]
    client.post(
        "/mcp",
        headers={**headers, "mcp-session-id": session_id},
        json={"jsonrpc": "2.0", "method": "notifications/initialized", "params": {}},
    )
    return session_id


def _sse_payload(response) -> dict:
    assert response.status_code == 200
    for line in response.text.splitlines():
        if line.startswith("data: "):
            return json.loads(line.removeprefix("data: "))
    raise AssertionError("missing SSE data line")
