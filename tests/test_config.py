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
