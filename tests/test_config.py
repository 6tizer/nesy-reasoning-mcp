import json
from pathlib import Path

from nesy_reasoning_mcp.config import StorageBackend, load_config, normalized_allowed_roots


def test_config_defaults_to_memory_and_allowed_roots(tmp_path: Path) -> None:
    config = load_config(env={}, cwd=tmp_path)

    assert config.storage.backend == StorageBackend.MEMORY
    assert str(tmp_path) in config.security.allowed_roots
    assert normalized_allowed_roots(config)[0] == tmp_path.resolve()
    assert config.security.allow_hidden_relation_paths is False


def test_config_file_and_env_override(tmp_path: Path) -> None:
    config_path = tmp_path / "nesy.json"
    db_path = tmp_path / "env.db"
    config_path.write_text(
        json.dumps(
            {
                "storage": {"backend": "json", "json_path": str(tmp_path / "data.json")},
                "security": {
                    "allowed_roots": [str(tmp_path / "from_config")],
                    "allow_hidden_relation_paths": False,
                },
            }
        ),
        encoding="utf-8",
    )

    config = load_config(
        env={
            "NESY_CONFIG": str(config_path),
            "NESY_STORAGE_BACKEND": "sqlite",
            "NESY_SQLITE_PATH": str(db_path),
            "NESY_ALLOWED_ROOTS": f"{tmp_path / 'a'},{tmp_path / 'b'}",
            "NESY_ALLOW_HIDDEN_RELATION_PATHS": "true",
            "NESY_LOG_LEVEL": "debug",
        },
        cwd=tmp_path,
    )

    assert config.storage.backend == StorageBackend.SQLITE
    assert config.storage.sqlite_path == str(db_path)
    assert config.storage.json_path == str(tmp_path / "data.json")
    assert config.security.allowed_roots == [str(tmp_path / "a"), str(tmp_path / "b")]
    assert config.security.allow_hidden_relation_paths is True
    assert config.logging.level == "debug"


def test_hook_config_file_and_env_override(tmp_path: Path) -> None:
    config_path = tmp_path / "nesy.json"
    config_path.write_text(
        json.dumps(
            {
                "hook": {
                    "timeout_seconds": 9,
                    "fail_closed": True,
                    "context_id": "from_file",
                    "domain": "from_file_domain",
                }
            }
        ),
        encoding="utf-8",
    )

    config = load_config(
        env={
            "NESY_CONFIG": str(config_path),
            "NESY_HOOK_TIMEOUT_SECONDS": "3",
            "NESY_HOOK_FAIL_CLOSED": "false",
            "NESY_HOOK_CONTEXT_ID": "from_env",
            "NESY_HOOK_DOMAIN": "from_env_domain",
            "NESY_HOOK_CONTEXT_FROM_SESSION": "true",
        },
        cwd=tmp_path,
    )

    assert config.hook.timeout_seconds == 3
    assert config.hook.fail_closed is False
    assert config.hook.context_id == "from_env"
    assert config.hook.domain == "from_env_domain"
    assert config.hook.context_from_session is True


def test_http_config_file_and_env_override(tmp_path: Path) -> None:
    config_path = tmp_path / "nesy.json"
    config_path.write_text(
        json.dumps(
            {
                "http": {
                    "host": "127.0.0.2",
                    "port": 9000,
                    "path": "/custom",
                    "local_token": "from-file",
                }
            }
        ),
        encoding="utf-8",
    )

    config = load_config(
        env={
            "NESY_CONFIG": str(config_path),
            "NESY_HTTP_HOST": "127.0.0.1",
            "NESY_HTTP_PORT": "8766",
            "NESY_HTTP_PATH": "/mcp",
            "NESY_LOCAL_TOKEN": "from-env",
            "NESY_HTTP_ALLOWED_ORIGINS": "http://127.0.0.1:8766,http://localhost:8766",
            "NESY_HTTP_ALLOWED_HOSTS": "127.0.0.1:8766,localhost:8766",
            "NESY_HTTP_MAX_BODY_BYTES": "2048",
            "NESY_HTTP_REQUEST_TIMEOUT_SECONDS": "7",
            "NESY_HTTP_RATE_LIMIT_PER_MINUTE": "11",
        },
        cwd=tmp_path,
    )

    assert config.http.host == "127.0.0.1"
    assert config.http.port == 8766
    assert config.http.path == "/mcp"
    assert config.http.local_token == "from-env"
    assert config.http.allowed_origins == ["http://127.0.0.1:8766", "http://localhost:8766"]
    assert config.http.allowed_hosts == ["127.0.0.1:8766", "localhost:8766"]
    assert config.http.max_body_bytes == 2048
    assert config.http.request_timeout_seconds == 7
    assert config.http.rate_limit_per_minute == 11
