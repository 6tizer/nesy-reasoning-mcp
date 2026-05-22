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
    tmp_path = real_path.with_name(f"{real_path.name}.tmp")
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

    root = _matching_allowed_root(real_path, normalized_allowed_roots(config))
    if root is None:
        raise ValueError("path is outside allowed_roots")
    if not config.security.allow_hidden_relation_paths and _has_hidden_relative_part(
        real_path,
        root,
    ):
        raise ValueError("hidden relation paths blocked unless configured")
    if for_write:
        real_path.parent.mkdir(parents=True, exist_ok=True)
    return real_path


def _matching_allowed_root(path: Path, roots: list[Path]) -> Path | None:
    matches = [root for root in roots if path == root or root in path.parents]
    return max(matches, key=lambda root: len(root.parts), default=None)


def _has_hidden_relative_part(path: Path, root: Path) -> bool:
    relative = path.relative_to(root)
    return any(part.startswith(".") for part in relative.parts)
