"""Proposition canonicalization for auto-write ingestion."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from nesy_reasoning_mcp.auto_ingest.schemas import (
    CandidateRelation,
    CandidateRelationBatch,
    IngestionInput,
)
from nesy_reasoning_mcp.auto_ingest.text import dedupe_non_empty_text
from nesy_reasoning_mcp.schemas import MAX_PROPOSITION_LENGTH, Diagnostic, PropositionRecord


class PropositionCanonicalizationDecision(BaseModel):
    """Canonical proposition group returned by the LLM canonicalizer."""

    model_config = ConfigDict(extra="forbid")

    endpoint_refs: list[str] = Field(min_length=1)
    canonical_label: str = Field(min_length=1, max_length=MAX_PROPOSITION_LENGTH)
    canonical_id: str | None = Field(default=None, min_length=1, max_length=MAX_PROPOSITION_LENGTH)
    aliases: list[str] = Field(default_factory=list)

    @field_validator("endpoint_refs", "aliases")
    @classmethod
    def strip_string_list(cls, value: list[str]) -> list[str]:
        """Strip entries and reject empty values."""
        stripped = [item.strip() for item in value]
        if any(not item for item in stripped):
            raise ValueError("must not contain empty values")
        return list(dict.fromkeys(stripped))

    @field_validator("canonical_label", "canonical_id")
    @classmethod
    def strip_optional_text(cls, value: str | None) -> str | None:
        """Strip text fields and reject empty provided values."""
        if value is None:
            return None
        stripped = value.strip()
        if not stripped:
            raise ValueError("must not be empty")
        return stripped


class PropositionCanonicalizationBatch(BaseModel):
    """Structured canonicalizer output."""

    model_config = ConfigDict(extra="forbid")

    propositions: list[PropositionCanonicalizationDecision] = Field(default_factory=list)


@dataclass(frozen=True)
class PropositionCanonicalizationResult:
    """Canonicalized candidates plus proposition registry records."""

    candidates: list[CandidateRelation]
    propositions: list[PropositionRecord]
    diagnostics: list[Diagnostic]
    metadata: dict[str, Any]


@dataclass(frozen=True)
class _Endpoint:
    ref: str
    candidate_id: str
    role: str
    text: str
    proposition_id: str | None


_QUALIFIER_RE = re.compile(
    r"\b("
    r"eligible|eligibility|may|might|can|could|allowed|permitted|permission|"
    r"ready|capable|candidate|possible|potential|optional"
    r")\b",
    re.IGNORECASE,
)


def canonicalization_prompt(
    ingestion_input: IngestionInput,
    candidate_batch: CandidateRelationBatch,
    known_propositions: list[PropositionRecord],
) -> str:
    """Build the LLM prompt for proposition canonicalization."""
    payload = {
        "input": ingestion_input.model_dump(mode="json", exclude_none=True),
        "candidate_endpoints": canonicalization_endpoint_payload(candidate_batch),
        "known_propositions": [
            proposition.model_dump(mode="json", exclude_none=True)
            for proposition in known_propositions
        ],
    }
    return (
        "Canonicalize candidate relation endpoints into stable proposition nodes.\n"
        "Return proposition groups that cover every endpoint_ref exactly once.\n"
        "Reuse a known proposition id only when the endpoint is the same proposition "
        "as that known label or one of its aliases.\n"
        "Do not merge a qualified state with the actual event or action. For example, "
        "'eligible for auto-deploy' is not the same proposition as 'auto-deploy'.\n"
        "Use aliases for alternate wording of the same proposition only.\n\n"
        f"Canonicalization payload JSON:\n{json.dumps(payload, ensure_ascii=False)}"
    )


def canonicalization_endpoint_payload(
    candidate_batch: CandidateRelationBatch,
) -> list[dict[str, Any]]:
    """Return candidate endpoint refs for the canonicalizer prompt."""
    return [
        {
            "endpoint_ref": endpoint.ref,
            "candidate_id": endpoint.candidate_id,
            "role": endpoint.role,
            "text": endpoint.text,
            "proposition_id": endpoint.proposition_id,
        }
        for endpoint in _candidate_endpoints(candidate_batch.candidates)
    ]


def canonicalize_candidate_relations(
    *,
    candidates: list[CandidateRelation],
    known_propositions: list[PropositionRecord],
    canonicalization: PropositionCanonicalizationBatch | None = None,
) -> PropositionCanonicalizationResult:
    """Apply deterministic and optional LLM-assisted canonical proposition IDs."""
    endpoints = _candidate_endpoints(candidates)
    known = _merge_known_propositions(known_propositions)
    term_index = _known_term_index(known)
    assignments: dict[str, PropositionRecord] = {}
    diagnostics: list[Diagnostic] = []

    if canonicalization is None:
        for endpoint in endpoints:
            assignments[endpoint.ref] = _record_for_endpoint(endpoint, known, term_index)
        mode = "deterministic"
    else:
        assigned_refs: set[str] = set()
        endpoint_by_ref = {endpoint.ref: endpoint for endpoint in endpoints}
        for decision in canonicalization.propositions:
            unknown_refs = [
                endpoint_ref
                for endpoint_ref in decision.endpoint_refs
                if endpoint_ref not in endpoint_by_ref
            ]
            if unknown_refs:
                diagnostics.append(
                    Diagnostic(
                        level="error",
                        code="PROPOSITION_CANONICALIZATION_UNKNOWN_ENDPOINT",
                        message="canonicalizer returned unknown endpoint refs",
                        related_ids=unknown_refs,
                    )
                )
                continue
            duplicate_refs = [
                endpoint_ref
                for endpoint_ref in decision.endpoint_refs
                if endpoint_ref in assigned_refs
            ]
            if duplicate_refs:
                diagnostics.append(
                    Diagnostic(
                        level="error",
                        code="PROPOSITION_CANONICALIZATION_DUPLICATE_ENDPOINT",
                        message="canonicalizer returned duplicate endpoint refs",
                        related_ids=duplicate_refs,
                    )
                )
                continue
            grouped_endpoints = [endpoint_by_ref[ref] for ref in decision.endpoint_refs]
            record, record_diagnostics = _record_for_decision(
                decision,
                grouped_endpoints,
                known,
                term_index,
            )
            diagnostics.extend(record_diagnostics)
            if record_diagnostics:
                continue
            for endpoint in grouped_endpoints:
                assignments[endpoint.ref] = record
                assigned_refs.add(endpoint.ref)
        missing_refs = [endpoint.ref for endpoint in endpoints if endpoint.ref not in assignments]
        if missing_refs:
            diagnostics.append(
                Diagnostic(
                    level="error",
                    code="PROPOSITION_CANONICALIZATION_MISSING_ENDPOINT",
                    message="canonicalizer did not cover every candidate endpoint",
                    related_ids=missing_refs,
                )
            )
        mode = "llm_assisted"

    if any(diagnostic.level == "error" for diagnostic in diagnostics):
        return PropositionCanonicalizationResult(
            candidates=candidates,
            propositions=[],
            diagnostics=diagnostics,
            metadata={
                "mode": mode,
                "candidate_count": len(candidates),
                "endpoint_count": len(endpoints),
                "known_proposition_count": len(known),
                "diagnostic_count": len(diagnostics),
            },
        )

    proposition_by_id: dict[str, PropositionRecord] = {}
    for record in assignments.values():
        existing = proposition_by_id.get(record.id)
        proposition_by_id[record.id] = (
            _merge_proposition_record(existing, record) if existing is not None else record
        )
    canonicalized_candidates = [
        _canonicalized_candidate(candidate, assignments) for candidate in candidates
    ]
    return PropositionCanonicalizationResult(
        candidates=canonicalized_candidates,
        propositions=list(proposition_by_id.values()),
        diagnostics=diagnostics,
        metadata={
            "mode": mode,
            "candidate_count": len(candidates),
            "endpoint_count": len(endpoints),
            "known_proposition_count": len(known),
            "proposition_count": len(proposition_by_id),
            "diagnostic_count": len(diagnostics),
        },
    )


def _candidate_endpoints(candidates: list[CandidateRelation]) -> list[_Endpoint]:
    endpoints: list[_Endpoint] = []
    for candidate in candidates:
        endpoints.append(
            _Endpoint(
                ref=f"{candidate.id}:source",
                candidate_id=candidate.id,
                role="source",
                text=candidate.source,
                proposition_id=candidate.source_id,
            )
        )
        endpoints.append(
            _Endpoint(
                ref=f"{candidate.id}:target",
                candidate_id=candidate.id,
                role="target",
                text=candidate.target,
                proposition_id=candidate.target_id,
            )
        )
    return endpoints


def _record_for_endpoint(
    endpoint: _Endpoint,
    known: dict[str, PropositionRecord],
    term_index: dict[str, str],
) -> PropositionRecord:
    proposition_id = endpoint.proposition_id if endpoint.proposition_id in known else None
    proposition_id = proposition_id or term_index.get(endpoint.text)
    if proposition_id is not None:
        return _record_with_alias(known[proposition_id], endpoint.text)
    return PropositionRecord(
        id=endpoint.proposition_id or _stable_proposition_id(endpoint.text),
        label=endpoint.text,
        aliases=[],
        negates=_negates_for_text(endpoint.text, term_index),
        metadata={"canonicalization": {"source": "agent_ingest"}},
    )


def _record_for_decision(
    decision: PropositionCanonicalizationDecision,
    endpoints: list[_Endpoint],
    known: dict[str, PropositionRecord],
    term_index: dict[str, str],
) -> tuple[PropositionRecord, list[Diagnostic]]:
    diagnostics: list[Diagnostic] = []
    merge_terms = [
        decision.canonical_label,
        *decision.aliases,
        *(endpoint.text for endpoint in endpoints),
    ]
    if _has_qualifier_mismatch(merge_terms) or _has_negation_mismatch(merge_terms):
        diagnostics.append(
            Diagnostic(
                level="error",
                code="PROPOSITION_CANONICALIZATION_UNSAFE_MERGE",
                message="canonicalizer attempted to merge logically distinct propositions",
                related_ids=[endpoint.ref for endpoint in endpoints],
            )
        )
        return _empty_proposition(), diagnostics

    proposition_id = _known_id_for(decision, endpoints, known, term_index)
    existing = known.get(proposition_id) if proposition_id is not None else None
    label = existing.label if existing is not None else decision.canonical_label
    proposition_id = proposition_id or _stable_proposition_id(label)
    aliases = _aliases_for_record(
        proposition_id=proposition_id,
        label=label,
        terms=merge_terms,
        term_index=term_index,
    )
    conflicts = [
        alias for alias in aliases if alias in term_index and term_index[alias] != proposition_id
    ]
    if conflicts:
        diagnostics.append(
            Diagnostic(
                level="error",
                code="PROPOSITION_CANONICALIZATION_ALIAS_CONFLICT",
                message="canonicalizer alias conflicts with another known proposition",
                related_ids=conflicts,
            )
        )
        return _empty_proposition(), diagnostics

    base_aliases = existing.aliases if existing is not None else []
    return (
        PropositionRecord(
            id=proposition_id,
            label=label,
            aliases=dedupe_non_empty_text([*base_aliases, *aliases]),
            negates=existing.negates
            if existing is not None
            else _negates_for_text(label, term_index),
            metadata={
                **(existing.metadata if existing is not None else {}),
                "canonicalization": {"source": "agent_ingest"},
            },
        ),
        diagnostics,
    )


def _known_id_for(
    decision: PropositionCanonicalizationDecision,
    endpoints: list[_Endpoint],
    known: dict[str, PropositionRecord],
    term_index: dict[str, str],
) -> str | None:
    if decision.canonical_id in known:
        return decision.canonical_id
    candidate_terms = [
        decision.canonical_label,
        *decision.aliases,
        *(endpoint.text for endpoint in endpoints),
        *(endpoint.proposition_id for endpoint in endpoints if endpoint.proposition_id),
    ]
    matched_ids = [
        term_index[term] for term in candidate_terms if term is not None and term in term_index
    ]
    if not matched_ids:
        return None
    first = matched_ids[0]
    return first if all(item == first for item in matched_ids) else None


def _aliases_for_record(
    *,
    proposition_id: str,
    label: str,
    terms: list[str],
    term_index: dict[str, str],
) -> list[str]:
    aliases: list[str] = []
    for term in dedupe_non_empty_text(terms):
        if term in {label, proposition_id}:
            continue
        mapped_id = term_index.get(term)
        if mapped_id is not None and mapped_id != proposition_id:
            aliases.append(term)
            continue
        aliases.append(term)
    return aliases


def _canonicalized_candidate(
    candidate: CandidateRelation,
    assignments: dict[str, PropositionRecord],
) -> CandidateRelation:
    source = assignments[f"{candidate.id}:source"]
    target = assignments[f"{candidate.id}:target"]
    metadata = dict(candidate.metadata)
    metadata["proposition_canonicalization"] = {
        "original_source": candidate.source,
        "original_target": candidate.target,
        "source_id": source.id,
        "target_id": target.id,
        "source_label": source.label,
        "target_label": target.label,
    }
    return candidate.model_copy(
        update={
            "source": source.label,
            "source_id": source.id,
            "target": target.label,
            "target_id": target.id,
            "metadata": metadata,
        }
    )


def _merge_known_propositions(
    propositions: list[PropositionRecord],
) -> dict[str, PropositionRecord]:
    merged: dict[str, PropositionRecord] = {}
    for proposition in propositions:
        current = merged.get(proposition.id)
        merged[proposition.id] = (
            _merge_proposition_record(current, proposition) if current is not None else proposition
        )
    return merged


def _merge_proposition_record(
    left: PropositionRecord,
    right: PropositionRecord,
) -> PropositionRecord:
    aliases = dedupe_non_empty_text(
        [
            *left.aliases,
            right.label,
            *right.aliases,
        ]
    )
    aliases = [alias for alias in aliases if alias not in {left.id, left.label}]
    return left.model_copy(
        update={
            "aliases": aliases,
            "metadata": {**left.metadata, **right.metadata},
        }
    )


def _record_with_alias(record: PropositionRecord, alias: str) -> PropositionRecord:
    if alias in {record.id, record.label, *record.aliases}:
        return record
    return record.model_copy(update={"aliases": dedupe_non_empty_text([*record.aliases, alias])})


def _known_term_index(propositions: dict[str, PropositionRecord]) -> dict[str, str]:
    index: dict[str, str] = {}
    for proposition in propositions.values():
        for term in (proposition.id, proposition.label, *proposition.aliases):
            if term in index and index[term] != proposition.id:
                continue
            index[term] = proposition.id
    return index


def _stable_proposition_id(label: str) -> str:
    normalized = " ".join(label.lower().split())
    digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]
    return f"prop_{digest}"


def _has_qualifier_mismatch(terms: list[str]) -> bool:
    flags = {_has_qualifier(term) for term in dedupe_non_empty_text(terms)}
    return len(flags) > 1


def _has_qualifier(term: str) -> bool:
    return bool(_QUALIFIER_RE.search(term))


def _has_negation_mismatch(terms: list[str]) -> bool:
    flags = {_explicit_negation_base(term) is not None for term in dedupe_non_empty_text(terms)}
    return len(flags) > 1


def _negates_for_text(text: str, term_index: dict[str, str]) -> str | None:
    base = _explicit_negation_base(text)
    if base is None:
        return None
    return term_index.get(base) or _stable_proposition_id(base)


def _explicit_negation_base(value: str) -> str | None:
    stripped = value.strip()
    if stripped.startswith("¬"):
        base = stripped[1:].strip()
        return base or None
    lowered = stripped.lower()
    if lowered.startswith("not:"):
        base = stripped[4:].strip()
        return base or None
    if lowered.startswith("not "):
        base = stripped[4:].strip()
        return base or None
    return None


def _empty_proposition() -> PropositionRecord:
    return PropositionRecord(id="invalid", label="invalid")
