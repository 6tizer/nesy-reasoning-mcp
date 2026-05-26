"""Conservative semantic duplicate detection for reviewed ingestion writes."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from nesy_reasoning_mcp.schemas import PropositionRecord, RelationInput, RelationRecord

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "be",
        "by",
        "for",
        "from",
        "in",
        "is",
        "of",
        "on",
        "or",
        "the",
        "to",
        "with",
    }
)
_QUALIFIER_RE = re.compile(
    r"\b("
    r"eligible|eligibility|may|might|can|could|allowed|permitted|permission|"
    r"ready|capable|candidate|possible|potential|optional"
    r")\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class SemanticDuplicateConcern:
    """Likely duplicate relation that must be reviewed instead of auto-written."""

    existing_relation_ids: list[str]
    matches: list[dict[str, Any]]

    def to_metadata(self) -> dict[str, Any]:
        """Return JSON-serializable queue metadata for the duplicate concern."""
        return {
            "existing_relation_ids": self.existing_relation_ids,
            "matches": self.matches,
            "reason": "likely_semantic_duplicate",
        }


@dataclass(frozen=True)
class _Endpoint:
    label: str
    proposition_id: str | None


@dataclass(frozen=True)
class _EndpointMatch:
    reason: str
    shared_terms: list[str]
    shared_tokens: list[str]

    def to_metadata(self) -> dict[str, Any]:
        return {
            "reason": self.reason,
            "shared_terms": self.shared_terms,
            "shared_tokens": self.shared_tokens,
        }


def semantic_duplicate_concerns(
    *,
    relations: list[RelationInput],
    existing_relations: list[RelationRecord],
    propositions: list[PropositionRecord],
) -> list[SemanticDuplicateConcern | None]:
    """Find likely semantic duplicates while leaving exact dedupe to the writer."""
    proposition_index = _proposition_index(propositions)
    proposition_alias_index = _proposition_alias_index(propositions)
    concerns: list[SemanticDuplicateConcern | None] = []
    for relation in relations:
        matches: list[dict[str, Any]] = []
        for existing in existing_relations:
            if not _same_relation_scope(relation, existing):
                continue
            if _is_exact_duplicate(relation, existing, proposition_alias_index):
                continue
            source_match = _endpoint_match(
                _Endpoint(relation.source, relation.source_id),
                _Endpoint(existing.source, existing.source_id),
                proposition_index,
            )
            if source_match is None:
                continue
            target_match = _endpoint_match(
                _Endpoint(relation.target, relation.target_id),
                _Endpoint(existing.target, existing.target_id),
                proposition_index,
            )
            if target_match is None:
                continue
            matches.append(
                {
                    "existing_relation_id": existing.id,
                    "existing_relation": {
                        "source": existing.source,
                        "source_id": existing.source_id,
                        "target": existing.target,
                        "target_id": existing.target_id,
                        "relation_type": existing.relation_type.value,
                    },
                    "source": source_match.to_metadata(),
                    "target": target_match.to_metadata(),
                }
            )
        concerns.append(
            SemanticDuplicateConcern(
                existing_relation_ids=[match["existing_relation_id"] for match in matches],
                matches=matches,
            )
            if matches
            else None
        )
    return concerns


def _same_relation_scope(left: RelationInput, right: RelationInput) -> bool:
    return (
        left.relation_type == right.relation_type
        and left.context_id == right.context_id
        and left.store_id == right.store_id
    )


def _is_exact_duplicate(
    left: RelationInput,
    right: RelationInput,
    proposition_alias_index: dict[str, str],
) -> bool:
    return bool(
        set(_relation_keys(left, proposition_alias_index))
        & set(_relation_keys(right, proposition_alias_index))
    )


def _relation_keys(
    relation: RelationInput,
    proposition_alias_index: dict[str, str],
) -> list[tuple[str, str, str, str, str]]:
    source = _resolved_endpoint_key(relation.source, relation.source_id, proposition_alias_index)
    target = _resolved_endpoint_key(relation.target, relation.target_id, proposition_alias_index)
    return list(
        dict.fromkeys(
            [
                (
                    source,
                    target,
                    relation.relation_type.value,
                    relation.context_id,
                    relation.store_id,
                ),
                (
                    relation.source,
                    relation.target,
                    relation.relation_type.value,
                    relation.context_id,
                    relation.store_id,
                ),
            ]
        )
    )


def _endpoint_match(
    left: _Endpoint,
    right: _Endpoint,
    proposition_index: dict[str, PropositionRecord],
) -> _EndpointMatch | None:
    if left.proposition_id is not None and left.proposition_id == right.proposition_id:
        return _EndpointMatch(reason="same_proposition_id", shared_terms=[], shared_tokens=[])
    if _has_qualifier_mismatch(left.label, right.label):
        return None

    left_terms = _endpoint_terms(left, proposition_index)
    right_terms = _endpoint_terms(right, proposition_index)
    shared_terms = sorted(set(left_terms) & set(right_terms))
    if shared_terms:
        return _EndpointMatch(
            reason="shared_label_or_alias", shared_terms=shared_terms, shared_tokens=[]
        )

    left_tokens = _endpoint_tokens(left_terms)
    right_tokens = _endpoint_tokens(right_terms)
    shared_tokens = sorted(left_tokens & right_tokens)
    if _token_overlap_is_likely_duplicate(shared_tokens, left_tokens, right_tokens):
        return _EndpointMatch(reason="token_overlap", shared_terms=[], shared_tokens=shared_tokens)
    return None


def _endpoint_terms(
    endpoint: _Endpoint,
    proposition_index: dict[str, PropositionRecord],
) -> list[str]:
    terms = [endpoint.label]
    if endpoint.proposition_id is not None:
        terms.append(endpoint.proposition_id)
        proposition = proposition_index.get(endpoint.proposition_id)
        if proposition is not None:
            terms.extend([proposition.label, *proposition.aliases])
    return [_normalize_term(term) for term in terms if _normalize_term(term)]


def _proposition_index(propositions: list[PropositionRecord]) -> dict[str, PropositionRecord]:
    return {proposition.id: proposition for proposition in propositions}


def _proposition_alias_index(propositions: list[PropositionRecord]) -> dict[str, str]:
    index: dict[str, str] = {}
    for proposition in propositions:
        for term in (proposition.id, proposition.label, *proposition.aliases):
            index.setdefault(_normalize_term(term), proposition.id)
    return index


def _resolved_endpoint_key(
    label: str,
    proposition_id: str | None,
    proposition_alias_index: dict[str, str],
) -> str:
    if proposition_id is not None:
        return proposition_id
    return proposition_alias_index.get(_normalize_term(label), label)


def _normalize_term(term: str) -> str:
    return " ".join(term.casefold().split())


def _endpoint_tokens(terms: list[str]) -> set[str]:
    tokens: set[str] = set()
    for term in terms:
        for token in _TOKEN_RE.findall(term):
            if len(token) < 3 or token in _STOPWORDS:
                continue
            tokens.add(token)
            tokens.update(_stems(token))
    return tokens


def _stems(token: str) -> set[str]:
    stems: set[str] = set()
    for suffix in ("ing", "ed", "es", "s"):
        if token.endswith(suffix) and len(token) > len(suffix) + 3:
            stems.add(token[: -len(suffix)])
    return stems


def _token_overlap_is_likely_duplicate(
    shared_tokens: list[str],
    left_tokens: set[str],
    right_tokens: set[str],
) -> bool:
    if len(shared_tokens) >= 2:
        return True
    if not shared_tokens:
        return False
    token = shared_tokens[0]
    return len(token) >= 8 and (len(left_tokens) == 1 or len(right_tokens) == 1)


def _has_qualifier_mismatch(left: str, right: str) -> bool:
    return bool(_QUALIFIER_RE.search(left)) != bool(_QUALIFIER_RE.search(right))
