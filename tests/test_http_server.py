import pytest
from starlette.testclient import TestClient

from nesy_reasoning_mcp.config import HttpConfig, NesyConfig
from nesy_reasoning_mcp.http_server import create_http_app
from nesy_reasoning_mcp.store import RelationStore


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
    assert accepted.json()["version"] == "0.7.0"


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
