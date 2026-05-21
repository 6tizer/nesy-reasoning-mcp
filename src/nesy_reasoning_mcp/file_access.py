"""Safe local file access for relation import and export."""

from __future__ import annotations

import os
from pathlib import Path

from nesy_reasoning_mcp.config import NesyConfig, normalized_allowed_roots

ALLOWED_RELATION_EXTENSIONS = {".json", ".jsonl"}


def read_allowed_relation_file(path: str, config: NesyConfig) -> tuple[Path, str]:
    """Read an allowed relation file and return its real path plus text."""
    real_path = _resolve_allowed_path(path, config, for_write=False)
    size = real_path.stat().st_size
    if size > config.security.max_file_size_bytes:
        raise ValueError("file exceeds max_file_size_bytes")
    return real_path, real_path.read_text(encoding="utf-8")


def write_allowed_relation_file(path: str, config: NesyConfig, text: str) -> Path:
    """Write an allowed relation file atomically and return its real path."""
    encoded = text.encode("utf-8")
    if len(encoded) > config.security.max_file_size_bytes:
        raise ValueError("file exceeds max_file_size_bytes")
    real_path = _resolve_allowed_path(path, config, for_write=True)
    tmp_path = real_path.with_name(f".{real_path.name}.tmp")
    tmp_path.write_bytes(encoded)
    os.replace(tmp_path, real_path)
    return real_path


def _resolve_allowed_path(path: str, config: NesyConfig, *, for_write: bool) -> Path:
    candidate = Path(path).expanduser()
    if candidate.suffix not in ALLOWED_RELATION_EXTENSIONS:
        raise ValueError("only .json and .jsonl files are allowed")

    if for_write:
        if candidate.exists():
            real_path = candidate.resolve(strict=True)
        else:
            real_path = candidate.parent.resolve(strict=False) / candidate.name
    else:
        real_path = candidate.resolve(strict=True)

    roots = normalized_allowed_roots(config)
    if not any(real_path == root or root in real_path.parents for root in roots):
        raise ValueError("path is outside allowed_roots")
    if for_write:
        real_path.parent.mkdir(parents=True, exist_ok=True)
    return real_path
