"""Deterministic Stop-hook enqueue decisions for conversation turns."""

from __future__ import annotations

import os
import re
from collections.abc import Mapping
from dataclasses import dataclass, field

from nesy_reasoning_mcp.config import parse_env_bool

MIN_MESSAGE_CHARS = 200
MAX_CODE_BLOCK_RATIO = 0.8
NESY_FACTS_TAG = "NESY_FACTS:"
NESY_FACTS_XML_TAG = "<NESY_FACTS>"

_FENCED_CODE_RE = re.compile(r"```.*?```", re.DOTALL)
_STRUCTURAL_KEYWORD_RE = re.compile(
    r"\b("
    r"requires|causes|implies|necessary|sufficient|depends|enables|blocks|"
    r"prevents|because|therefore"
    r")\b|因为|因此|导致|需要|依赖|必要|充分|意味着|矛盾",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class EnqueueDecision:
    """A deterministic decision for whether a conversation turn should be queued."""

    enqueue: bool
    priority: int = 0
    skip_extraction: bool = False
    reason: str = "not_matched"
    diagnostics: list[str] = field(default_factory=list)


def should_enqueue(
    message: str,
    *,
    env: Mapping[str, str] | None = None,
) -> EnqueueDecision:
    """Return whether a Stop-hook message should become a conversation turn job."""
    normalized = message.strip()
    if NESY_FACTS_TAG in normalized or NESY_FACTS_XML_TAG in normalized:
        return EnqueueDecision(
            enqueue=True,
            priority=1,
            skip_extraction=True,
            reason="nesy_facts",
        )

    diagnostics: list[str] = []
    if _classifier_enabled(env):
        diagnostics.append(
            "NESY_ENQUEUE_CLASSIFIER is set; classifier support is not yet active "
            "and will be enabled in a later phase."
        )

    if len(normalized) < MIN_MESSAGE_CHARS:
        return EnqueueDecision(False, reason="too_short", diagnostics=diagnostics)
    if _code_block_ratio(normalized) > MAX_CODE_BLOCK_RATIO:
        return EnqueueDecision(False, reason="code_heavy", diagnostics=diagnostics)
    if _STRUCTURAL_KEYWORD_RE.search(normalized) is None:
        return EnqueueDecision(False, reason="no_structural_keyword", diagnostics=diagnostics)
    return EnqueueDecision(True, reason="structural_keyword", diagnostics=diagnostics)


def _classifier_enabled(env: Mapping[str, str] | None) -> bool:
    env_map = os.environ if env is None else env
    value = env_map.get("NESY_ENQUEUE_CLASSIFIER")
    return parse_env_bool(value) if value is not None else False


def _code_block_ratio(message: str) -> float:
    if not message:
        return 0.0
    code_chars = sum(match.end() - match.start() for match in _FENCED_CODE_RE.finditer(message))
    return code_chars / len(message)
