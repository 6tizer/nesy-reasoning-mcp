"""Pydantic schemas for relation storage and tool inputs."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator

MAX_PROPOSITION_LENGTH = 512
MAX_ASSERT_RELATIONS = 500
DEFAULT_CONTEXT_ID = "default"
DEFAULT_STORE_ID = "default"


class RelationType(StrEnum):
    """Supported external relation types."""

    SUFFICIENT = "sufficient"
    NECESSARY = "necessary"
    EQUIVALENT = "equivalent"


class Classification(StrEnum):
    """Logical relation classification values."""

    SUFFICIENT = "sufficient"
    NECESSARY = "necessary"
    EQUIVALENT = "equivalent"
    UNKNOWN = "unknown"
    CONTRADICTORY = "contradictory"


class ConfidencePolicy(StrEnum):
    """Supported evidence confidence aggregation policies."""

    PRODUCT_INDEPENDENT = "product_independent"
    MIN = "min"
    NO_AGGREGATION = "no_aggregation"


class ExpectedRelation(StrEnum):
    """Expected relation values for chain verification."""

    SUFFICIENT = "sufficient"
    NECESSARY = "necessary"
    EQUIVALENT = "equivalent"
    ANY = "any"


class PathStrategy(StrEnum):
    """Path search strategies."""

    BEST_CONFIDENCE = "best_confidence"
    SHORTEST = "shortest"
    ALL = "all"


class ExclusiveScope(StrEnum):
    """Exclusive group applicability scope."""

    SAME_CONTEXT = "same_context"
    GLOBAL = "global"


class ContradictionMode(StrEnum):
    """Contradiction check input mode."""

    GRAPH = "graph"
    FACTS = "facts"
    COMBINED = "combined"


class Polarity(StrEnum):
    """Supported relation polarity values."""

    POSITIVE = "positive"


class Diagnostic(BaseModel):
    """Tool diagnostic entry."""

    model_config = ConfigDict(extra="forbid")

    level: Literal["info", "warning", "error"]
    code: str
    message: str
    related_ids: list[str] = Field(default_factory=list)


class RelationInput(BaseModel):
    """Input shape for asserting relation records."""

    model_config = ConfigDict(extra="forbid")

    id: str | None = None
    source: str = Field(min_length=1, max_length=MAX_PROPOSITION_LENGTH)
    target: str = Field(min_length=1, max_length=MAX_PROPOSITION_LENGTH)
    relation_type: RelationType
    confidence: float = Field(default=1.0, ge=0, le=1)
    context_id: str = DEFAULT_CONTEXT_ID
    store_id: str = DEFAULT_STORE_ID
    temporal: dict[str, Any] | None = None
    assumptions: list[str] = Field(default_factory=list)
    provenance: dict[str, Any] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("source", "target", "context_id", "store_id")
    @classmethod
    def strip_non_empty(cls, value: str) -> str:
        """Strip surrounding whitespace and reject empty strings."""
        stripped = value.strip()
        if not stripped:
            raise ValueError("must not be empty")
        return stripped


class RelationRecord(RelationInput):
    """Stored relation record with generated identity and defaults."""

    id: str = Field(default_factory=lambda: f"rel_{uuid4().hex}")
    polarity: Polarity = Polarity.POSITIVE
    created_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    updated_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())

    @classmethod
    def from_input(cls, relation: RelationInput) -> RelationRecord:
        """Create a persisted relation record from validated input."""
        data = relation.model_dump()
        if data["id"] is None:
            data.pop("id")
        return cls(**data)


class CanonicalImplicationEdge(BaseModel):
    """Canonical implication edge used by reasoning indexes."""

    model_config = ConfigDict(extra="forbid")

    edge_id: str
    relation_id: str
    antecedent: str
    consequent: str
    source_relation_type: RelationType
    confidence: float
    context_id: str
    store_id: str
    assumptions: list[str] = Field(default_factory=list)
    temporal: dict[str, Any] | None = None

    @property
    def temporal_window(self) -> tuple[datetime | None, datetime | None]:
        """Return parsed temporal valid_from/valid_to bounds."""
        temporal = self.temporal or {}
        return (
            _parse_datetime(temporal.get("valid_from")),
            _parse_datetime(temporal.get("valid_to")),
        )


def _parse_datetime(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    return None


class GraphStats(BaseModel):
    """Current relation graph statistics."""

    relations: int
    propositions: int
    implication_edges: int
    exclusive_groups: int = 0
    contexts: int
    stores: int


class ExclusiveGroupInput(BaseModel):
    """Input shape for declaring mutually exclusive propositions."""

    model_config = ConfigDict(extra="forbid")

    group_id: str | None = None
    members: list[str] = Field(min_length=2)
    context_id: str = DEFAULT_CONTEXT_ID
    store_id: str = DEFAULT_STORE_ID
    scope: ExclusiveScope = ExclusiveScope.SAME_CONTEXT
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("group_id", "context_id", "store_id")
    @classmethod
    def strip_optional_non_empty(cls, value: str | None) -> str | None:
        """Strip strings and reject empty provided values."""
        if value is None:
            return None
        stripped = value.strip()
        if not stripped:
            raise ValueError("must not be empty")
        return stripped

    @field_validator("members")
    @classmethod
    def strip_unique_members(cls, value: list[str]) -> list[str]:
        """Strip and de-duplicate exclusive group members."""
        members = [item.strip() for item in value]
        if any(not item for item in members):
            raise ValueError("members must not be empty")
        deduped = list(dict.fromkeys(members))
        if len(deduped) < 2:
            raise ValueError("at least two unique members are required")
        return deduped


class ExclusiveGroupRecord(ExclusiveGroupInput):
    """Stored exclusive group record."""

    group_id: str = Field(default_factory=lambda: f"excl_{uuid4().hex}")
    created_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    updated_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())

    @classmethod
    def from_input(cls, group: ExclusiveGroupInput) -> ExclusiveGroupRecord:
        """Create a persisted exclusive group from validated input."""
        data = group.model_dump()
        if data["group_id"] is None:
            data.pop("group_id")
        return cls(**data)


class ContextFilter(BaseModel):
    """Context, store, domain, assumption, and temporal query filter."""

    model_config = ConfigDict(extra="forbid")

    context_id: str | None = None
    store_id: str | None = None
    domain: str | None = None
    assumptions: list[str] = Field(default_factory=list)
    valid_at: datetime | None = None


class AssertRelationsInput(BaseModel):
    """Input for `nesy.assert_relations`."""

    model_config = ConfigDict(extra="forbid")

    relations: list[RelationInput] = Field(min_length=1, max_length=MAX_ASSERT_RELATIONS)
    mode: Literal["append", "upsert", "replace_same_pair"] = "append"
    check_contradictions: bool = True
    merge_equivalent: bool = True
    dry_run: bool = False


class RelationFilter(BaseModel):
    """Filter used by list and clear tools."""

    model_config = ConfigDict(extra="forbid")

    source: str | None = None
    target: str | None = None
    relation_type: RelationType | None = None
    context_id: str | None = None
    store_id: str | None = None
    domain: str | None = None


class ListRelationsInput(BaseModel):
    """Input for `nesy.list_relations`."""

    model_config = ConfigDict(extra="forbid")

    filter: RelationFilter = Field(default_factory=RelationFilter)
    include_implication_edges: bool = False
    include_exclusive_groups: bool = False
    limit: int = Field(default=100, ge=1, le=500)
    cursor: str | None = None


class ClearRelationsInput(BaseModel):
    """Input for `nesy.clear_relations`."""

    model_config = ConfigDict(extra="forbid")

    scope: Literal["all", "store", "context", "filter"] = "context"
    store_id: str = DEFAULT_STORE_ID
    context_id: str = DEFAULT_CONTEXT_ID
    filter: RelationFilter = Field(default_factory=RelationFilter)
    include_exclusive_groups: bool = False
    dry_run: bool = False


class AssertExclusiveInput(BaseModel):
    """Input for `nesy.assert_exclusive`."""

    model_config = ConfigDict(extra="forbid")

    groups: list[ExclusiveGroupInput] = Field(min_length=1)


class CheckContradictionsInput(BaseModel):
    """Input for `nesy.check_contradictions`."""

    model_config = ConfigDict(extra="forbid")

    facts: list[RelationInput] = Field(default_factory=list)
    mode: ContradictionMode = ContradictionMode.GRAPH
    context_filter: ContextFilter = Field(default_factory=ContextFilter)
    include_soft: bool = True
    max_depth: int = Field(default=8, ge=1, le=20)


class _PropositionPairInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source: str = Field(min_length=1, max_length=MAX_PROPOSITION_LENGTH)
    target: str = Field(min_length=1, max_length=MAX_PROPOSITION_LENGTH)

    @field_validator("source", "target")
    @classmethod
    def strip_non_empty(cls, value: str) -> str:
        """Strip surrounding whitespace and reject empty proposition names."""
        stripped = value.strip()
        if not stripped:
            raise ValueError("must not be empty")
        return stripped


class ClassifyInput(_PropositionPairInput):
    """Input for `nesy.classify`."""

    context_filter: ContextFilter = Field(default_factory=ContextFilter)
    max_depth: int = Field(default=8, ge=1, le=20)
    include_paths: bool = True
    require_direct: bool = False
    confidence_policy: ConfidencePolicy = ConfidencePolicy.PRODUCT_INDEPENDENT


class VerifyChainInput(_PropositionPairInput):
    """Input for `nesy.verify_chain`."""

    chain: list[str] | None = Field(default=None, min_length=2)
    expected_relation: ExpectedRelation = ExpectedRelation.ANY
    context_filter: ContextFilter = Field(default_factory=ContextFilter)
    max_depth: int = Field(default=8, ge=1, le=20)
    path_strategy: PathStrategy = PathStrategy.BEST_CONFIDENCE
    max_paths: int = Field(default=5, ge=1, le=50)
    confidence_policy: ConfidencePolicy = ConfidencePolicy.PRODUCT_INDEPENDENT

    @field_validator("chain")
    @classmethod
    def strip_chain_nodes(cls, value: list[str] | None) -> list[str] | None:
        """Strip chain nodes and reject empty values."""
        if value is None:
            return None
        stripped = [item.strip() for item in value]
        if any(not item for item in stripped):
            raise ValueError("chain nodes must not be empty")
        return stripped
