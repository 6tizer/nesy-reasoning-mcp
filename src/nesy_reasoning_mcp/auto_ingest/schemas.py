"""Pydantic schemas for Agent SDK candidate relation ingestion."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from nesy_reasoning_mcp.schemas import (
    DEFAULT_CONTEXT_ID,
    DEFAULT_STORE_ID,
    MAX_PROPOSITION_LENGTH,
    Diagnostic,
    RelationInput,
    RelationType,
)


class ReviewDecisionValue(StrEnum):
    """Reviewer decisions for a candidate relation."""

    APPROVE = "approve"
    DOWNGRADE = "downgrade"
    REJECT = "reject"
    NEEDS_HUMAN = "needs_human"


class GateAction(StrEnum):
    """Deterministic gate actions for a reviewed candidate."""

    AUTO_WRITE = "auto_write"
    QUEUE = "queue"
    REJECT = "reject"


class IngestionMode(StrEnum):
    """Supported ingestion run modes."""

    DRY_RUN = "dry_run"
    WRITE = "write"


class EvidenceRecord(BaseModel):
    """A source excerpt supporting a candidate relation."""

    model_config = ConfigDict(extra="forbid")

    url: str = Field(min_length=1)
    span: str = Field(min_length=1)
    title: str | None = None
    source_type: str | None = None
    retrieved_at: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("url", "span", "title", "source_type", "retrieved_at")
    @classmethod
    def strip_optional_text(cls, value: str | None) -> str | None:
        """Strip text fields and reject empty provided values."""
        if value is None:
            return None
        stripped = value.strip()
        if not stripped:
            raise ValueError("must not be empty")
        return stripped


class CandidateRelation(BaseModel):
    """A relation proposed by an external evidence ingestion workflow."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(default_factory=lambda: f"cand_{uuid4().hex}", min_length=1)
    source: str = Field(min_length=1, max_length=MAX_PROPOSITION_LENGTH)
    source_id: str | None = Field(default=None, min_length=1, max_length=MAX_PROPOSITION_LENGTH)
    target: str = Field(min_length=1, max_length=MAX_PROPOSITION_LENGTH)
    target_id: str | None = Field(default=None, min_length=1, max_length=MAX_PROPOSITION_LENGTH)
    relation_type: RelationType
    confidence: float = Field(default=1.0, ge=0, le=1)
    context_id: str = DEFAULT_CONTEXT_ID
    store_id: str = DEFAULT_STORE_ID
    evidence: list[EvidenceRecord] = Field(min_length=1)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("id", "source", "target", "context_id", "store_id")
    @classmethod
    def strip_non_empty(cls, value: str) -> str:
        """Strip surrounding whitespace and reject empty strings."""
        stripped = value.strip()
        if not stripped:
            raise ValueError("must not be empty")
        return stripped

    @field_validator("source_id", "target_id")
    @classmethod
    def strip_optional_id(cls, value: str | None) -> str | None:
        """Strip optional proposition IDs and reject empty provided values."""
        if value is None:
            return None
        stripped = value.strip()
        if not stripped:
            raise ValueError("must not be empty")
        return stripped

    def to_relation_input(self) -> RelationInput:
        """Convert a reviewed candidate into an assertable relation input."""
        return RelationInput(
            source=self.source,
            source_id=self.source_id,
            target=self.target,
            target_id=self.target_id,
            relation_type=self.relation_type,
            confidence=self.confidence,
            context_id=self.context_id,
            store_id=self.store_id,
            metadata=self.metadata,
            provenance={
                "candidate_id": self.id,
                "evidence": [
                    item.model_dump(mode="json", exclude_none=True) for item in self.evidence
                ],
            },
        )


class ReviewDecision(BaseModel):
    """AI reviewer decision for a candidate relation."""

    model_config = ConfigDict(extra="forbid")

    candidate_id: str = Field(min_length=1)
    decision: ReviewDecisionValue
    final_relation_type: RelationType | None = None
    final_confidence: float | None = Field(default=None, ge=0, le=1)
    reasons: list[str] = Field(default_factory=list)
    risk_flags: list[str] = Field(default_factory=list)
    reviewer_model: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("candidate_id", "reviewer_model")
    @classmethod
    def strip_optional_text(cls, value: str | None) -> str | None:
        """Strip text fields and reject empty provided values."""
        if value is None:
            return None
        stripped = value.strip()
        if not stripped:
            raise ValueError("must not be empty")
        return stripped

    @field_validator("reasons", "risk_flags")
    @classmethod
    def strip_string_lists(cls, value: list[str]) -> list[str]:
        """Strip list entries and reject empty provided values."""
        stripped = [item.strip() for item in value]
        if any(not item for item in stripped):
            raise ValueError("must not contain empty values")
        return stripped

    @model_validator(mode="after")
    def require_final_relation_for_positive_decisions(self) -> ReviewDecision:
        """Require final relation fields when a candidate can proceed."""
        if self.decision in {ReviewDecisionValue.APPROVE, ReviewDecisionValue.DOWNGRADE} and (
            self.final_relation_type is None or self.final_confidence is None
        ):
            raise ValueError("approve and downgrade decisions require final relation info")
        return self


class GateResult(BaseModel):
    """Deterministic gate result for a reviewed candidate."""

    model_config = ConfigDict(extra="forbid")

    candidate_id: str = Field(min_length=1)
    action: GateAction
    reasons: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("candidate_id")
    @classmethod
    def strip_candidate_id(cls, value: str) -> str:
        """Strip candidate IDs and reject empty values."""
        stripped = value.strip()
        if not stripped:
            raise ValueError("must not be empty")
        return stripped

    @field_validator("reasons")
    @classmethod
    def strip_reasons(cls, value: list[str]) -> list[str]:
        """Strip reason entries and reject empty provided values."""
        stripped = [item.strip() for item in value]
        if any(not item for item in stripped):
            raise ValueError("must not contain empty values")
        return stripped


class IngestionReport(BaseModel):
    """Structured report emitted by an Agent SDK ingestion run."""

    model_config = ConfigDict(extra="forbid")

    run_id: str = Field(default_factory=lambda: f"ing_{uuid4().hex}", min_length=1)
    generated_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    mode: IngestionMode = IngestionMode.DRY_RUN
    candidates: list[CandidateRelation] = Field(default_factory=list)
    reviews: list[ReviewDecision] = Field(default_factory=list)
    gate_results: list[GateResult] = Field(default_factory=list)
    approved_relations: list[RelationInput] = Field(default_factory=list)
    diagnostics: list[Diagnostic] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("run_id", "generated_at")
    @classmethod
    def strip_non_empty(cls, value: str) -> str:
        """Strip report identifiers and reject empty values."""
        stripped = value.strip()
        if not stripped:
            raise ValueError("must not be empty")
        return stripped
