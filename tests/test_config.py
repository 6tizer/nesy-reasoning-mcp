import json
from pathlib import Path

from nesy_reasoning_mcp.config import StorageBackend, load_config, normalized_allowed_roots


def test_config_defaults_to_memory_and_allowed_roots(tmp_path: Path) -> None:
    config = load_config(env={}, cwd=tmp_path)

    assert config.storage.backend == StorageBackend.MEMORY
    assert str(tmp_path) in config.security.allowed_roots
    assert normalized_allowed_roots(config)[0] == tmp_path.resolve()


def test_config_file_and_env_override(tmp_path: Path) -> None:
    config_path = tmp_path / "nesy.json"
    db_path = tmp_path / "env.db"
    config_path.write_text(
        json.dumps(
            {
                "storage": {"backend": "json", "json_path": str(tmp_path / "data.json")},
                "security": {"allowed_roots": [str(tmp_path / "from_config")]},
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
            "NESY_LOG_LEVEL": "debug",
        },
        cwd=tmp_path,
    )

    assert config.storage.backend == StorageBackend.SQLITE
    assert config.storage.sqlite_path == str(db_path)
    assert config.storage.json_path == str(tmp_path / "data.json")
    assert config.security.allowed_roots == [str(tmp_path / "a"), str(tmp_path / "b")]
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
