"""Shared NESY_FACTS parsing helpers."""

from __future__ import annotations

import json
from typing import Any


def extract_nesy_facts(message: str) -> dict[str, list[dict[str, Any]]] | None:
    """Extract structured NESY_FACTS relations and propositions from text."""
    raw = extract_nesy_facts_raw(message)
    if raw is None:
        return None
    payload, _end = json.JSONDecoder().raw_decode(raw)
    if isinstance(payload, list):
        if not all(isinstance(item, dict) for item in payload):
            raise ValueError("NESY_FACTS entries must be JSON objects")
        return {"relations": payload, "propositions": []}
    if isinstance(payload, dict):
        relations = payload.get("relations")
        propositions = payload.get("propositions", [])
        if not isinstance(relations, list):
            raise ValueError("NESY_FACTS relations must be a JSON array")
        if not isinstance(propositions, list):
            raise ValueError("NESY_FACTS propositions must be a JSON array")
        if not all(isinstance(item, dict) for item in relations):
            raise ValueError("NESY_FACTS relation entries must be JSON objects")
        if not all(isinstance(item, dict) for item in propositions):
            raise ValueError("NESY_FACTS proposition entries must be JSON objects")
        return {"relations": relations, "propositions": propositions}
    raise ValueError("NESY_FACTS must be a JSON array or object")


def extract_nesy_facts_raw(message: str) -> str | None:
    """Extract the raw JSON payload from NESY_FACTS text or XML-style tags."""
    tag_start = message.find("<NESY_FACTS>")
    if tag_start >= 0:
        raw_start = tag_start + len("<NESY_FACTS>")
        tag_end = message.find("</NESY_FACTS>", raw_start)
        if tag_end < 0:
            raise ValueError("NESY_FACTS closing tag is missing")
        return message[raw_start:tag_end].strip()

    marker = "NESY_FACTS:"
    marker_index = message.find(marker)
    if marker_index < 0:
        return None
    raw = message[marker_index + len(marker) :].strip()
    if raw.startswith("```"):
        return _extract_fenced_json(raw)
    return raw


def _extract_fenced_json(raw: str) -> str:
    lines = raw.splitlines()
    if not lines:
        return raw
    body: list[str] = []
    for line in lines[1:]:
        if line.strip().startswith("```"):
            return "\n".join(body).strip()
        body.append(line)
    return "\n".join(body).strip()
