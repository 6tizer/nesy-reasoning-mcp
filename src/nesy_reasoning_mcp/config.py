"""Runtime configuration for the NeSy Reasoning MCP server."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal

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
    allow_hidden_relation_paths: bool = False


class LoggingConfig(BaseModel):
    """Logging and audit settings."""

    model_config = ConfigDict(extra="forbid")

    level: str = "info"
    audit_log: bool = True


FocusTermSource = Literal[
    "tool_name",
    "cwd_basename",
    "tool_input_strings",
    "cwd_path_segments",
    "configured_terms",
]


def _default_focus_term_sources() -> list[FocusTermSource]:
    return ["tool_name", "cwd_basename", "tool_input_strings"]


class HookConfig(BaseModel):
    """Claude Code hook integration settings."""

    model_config = ConfigDict(extra="forbid")

    timeout_seconds: float = Field(default=5, gt=0, le=60)
    fail_closed: bool = False
    context_id: str | None = None
    domain: str | None = None
    context_from_session: bool = False
    focus_term_sources: list[FocusTermSource] = Field(default_factory=_default_focus_term_sources)
    focus_terms: list[str] = Field(default_factory=list)


class HttpConfig(BaseModel):
    """Streamable HTTP daemon settings."""

    model_config = ConfigDict(extra="forbid")

    host: str = "127.0.0.1"
    port: int = Field(default=8765, ge=1, le=65535)
    path: str = "/mcp"
    local_token: str | None = None
    allowed_origins: list[str] = Field(default_factory=list)
    allowed_hosts: list[str] = Field(default_factory=list)
    max_body_bytes: int = Field(default=1_000_000, ge=1)
    request_timeout_seconds: float = Field(default=30, gt=0, le=300)
    rate_limit_per_minute: int = Field(default=120, ge=1)


class NesyConfig(BaseModel):
    """Complete runtime configuration."""

    model_config = ConfigDict(extra="forbid")

    storage: StorageConfig = Field(default_factory=StorageConfig)
    security: SecurityConfig = Field(default_factory=SecurityConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    hook: HookConfig = Field(default_factory=HookConfig)
    http: HttpConfig = Field(default_factory=HttpConfig)


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
    if allow_hidden := env_map.get("NESY_ALLOW_HIDDEN_RELATION_PATHS"):
        data.setdefault("security", {})["allow_hidden_relation_paths"] = _env_bool(allow_hidden)
    if log_level := env_map.get("NESY_LOG_LEVEL"):
        data.setdefault("logging", {})["level"] = log_level
    if timeout := env_map.get("NESY_HOOK_TIMEOUT_SECONDS"):
        data.setdefault("hook", {})["timeout_seconds"] = float(timeout)
    if fail_closed := env_map.get("NESY_HOOK_FAIL_CLOSED"):
        data.setdefault("hook", {})["fail_closed"] = _env_bool(fail_closed)
    if hook_context_id := env_map.get("NESY_HOOK_CONTEXT_ID"):
        data.setdefault("hook", {})["context_id"] = hook_context_id
    if hook_domain := env_map.get("NESY_HOOK_DOMAIN"):
        data.setdefault("hook", {})["domain"] = hook_domain
    if context_from_session := env_map.get("NESY_HOOK_CONTEXT_FROM_SESSION"):
        data.setdefault("hook", {})["context_from_session"] = _env_bool(context_from_session)
    if focus_term_sources := env_map.get("NESY_HOOK_FOCUS_TERM_SOURCES"):
        data.setdefault("hook", {})["focus_term_sources"] = _env_csv(focus_term_sources)
    if focus_terms := env_map.get("NESY_HOOK_FOCUS_TERMS"):
        data.setdefault("hook", {})["focus_terms"] = _env_csv(focus_terms)
    if http_host := env_map.get("NESY_HTTP_HOST"):
        data.setdefault("http", {})["host"] = http_host
    if http_port := env_map.get("NESY_HTTP_PORT"):
        data.setdefault("http", {})["port"] = int(http_port)
    if http_path := env_map.get("NESY_HTTP_PATH"):
        data.setdefault("http", {})["path"] = http_path
    if local_token := env_map.get("NESY_LOCAL_TOKEN"):
        data.setdefault("http", {})["local_token"] = local_token
    if origins := env_map.get("NESY_HTTP_ALLOWED_ORIGINS"):
        data.setdefault("http", {})["allowed_origins"] = _env_csv(origins)
    if hosts := env_map.get("NESY_HTTP_ALLOWED_HOSTS"):
        data.setdefault("http", {})["allowed_hosts"] = _env_csv(hosts)
    if max_body := env_map.get("NESY_HTTP_MAX_BODY_BYTES"):
        data.setdefault("http", {})["max_body_bytes"] = int(max_body)
    if http_timeout := env_map.get("NESY_HTTP_REQUEST_TIMEOUT_SECONDS"):
        data.setdefault("http", {})["request_timeout_seconds"] = float(http_timeout)
    if rate_limit := env_map.get("NESY_HTTP_RATE_LIMIT_PER_MINUTE"):
        data.setdefault("http", {})["rate_limit_per_minute"] = int(rate_limit)

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
            "allow_hidden_relation_paths": False,
        },
        "logging": {"level": "info", "audit_log": True},
        "hook": {
            "timeout_seconds": 5,
            "fail_closed": False,
            "context_from_session": False,
            "focus_term_sources": ["tool_name", "cwd_basename", "tool_input_strings"],
            "focus_terms": [],
        },
        "http": {
            "host": "127.0.0.1",
            "port": 8765,
            "path": "/mcp",
            "max_body_bytes": 1_000_000,
            "request_timeout_seconds": 30,
            "rate_limit_per_minute": 120,
        },
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


def parse_env_bool(value: str) -> bool:
    """Parse common truthy environment flag values."""
    normalized = value.strip().casefold()
    return normalized in {"1", "true", "yes", "on"}


def _env_bool(value: str) -> bool:
    return parse_env_bool(value)


def _env_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]
