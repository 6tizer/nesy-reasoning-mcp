"""Shared runner type aliases for Auto-Ingest workers and agents."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

AgentRunner = Callable[..., Awaitable[Any]]
