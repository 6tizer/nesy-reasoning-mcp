"""Runtime configuration for the NeSy Reasoning MCP server."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class StorageBackend(StrEnum):
    """Supported relation storage backends."""

    MEMORY = "memory"
    JSON = "json"
    SQLITE = "sqlite"


class StorageConfig(BaseModel):
    """Persistence backend settings."""

    model_config = ConfigDict(extra="forbid")

    backend: StorageBackend = StorageBackend.MEMORY
    json_path: str | None = None
    sqlite_path: str | None = None
    default_store_id: str = "default"
    default_context_id: str = "default"


class SecurityConfig(BaseModel):
    """Local file access limits."""

    model_config = ConfigDict(extra="forbid")

    allowed_roots: list[str] = Field(default_factory=list)
    max_file_size_bytes: int = Field(default=5 * 1024 * 1024, ge=1)
    allow_scope_all_clear: bool = False


class LoggingConfig(BaseModel):
    """Logging and audit settings."""

    model_config = ConfigDict(extra="forbid")

    level: str = "info"
    audit_log: bool = True


class NesyConfig(BaseModel):
    """Complete runtime configuration."""

    model_config = ConfigDict(extra="forbid")

    storage: StorageConfig = Field(default_factory=StorageConfig)
    security: SecurityConfig = Field(default_factory=SecurityConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)


def load_config(
    *,
    env: Mapping[str, str] | None = None,
    cwd: Path | None = None,
) -> NesyConfig:
    """Load runtime config from defaults, `NESY_CONFIG`, and environment overrides."""
    env_map = os.environ if env is None else env
    base_cwd = Path.cwd() if cwd is None else cwd
    data = _default_config_data(base_cwd)

    config_path = env_map.get("NESY_CONFIG")
    if config_path:
        loaded = json.loads(_expand_path(config_path).read_text(encoding="utf-8"))
        data = _deep_merge(data, loaded)

    if backend := env_map.get("NESY_STORAGE_BACKEND"):
        data.setdefault("storage", {})["backend"] = backend
    if sqlite_path := env_map.get("NESY_SQLITE_PATH"):
        data.setdefault("storage", {})["sqlite_path"] = sqlite_path
    if roots := env_map.get("NESY_ALLOWED_ROOTS"):
        data.setdefault("security", {})["allowed_roots"] = [
            item.strip() for item in roots.split(",") if item.strip()
        ]
    if log_level := env_map.get("NESY_LOG_LEVEL"):
        data.setdefault("logging", {})["level"] = log_level

    return NesyConfig.model_validate(data)


def normalized_allowed_roots(config: NesyConfig) -> list[Path]:
    """Return real allowed root paths, creating no directories."""
    return [_expand_path(root).resolve() for root in config.security.allowed_roots]


def _default_config_data(cwd: Path) -> dict[str, Any]:
    return {
        "storage": {"backend": StorageBackend.MEMORY.value},
        "security": {
            "allowed_roots": [
                str(cwd),
                str(Path.home() / ".nesy-reasoning" / "relation_sets"),
            ],
            "max_file_size_bytes": 5 * 1024 * 1024,
            "allow_scope_all_clear": False,
        },
        "logging": {"level": "info", "audit_log": True},
    }


def _deep_merge(base: dict[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _expand_path(path: str) -> Path:
    return Path(path).expanduser()
