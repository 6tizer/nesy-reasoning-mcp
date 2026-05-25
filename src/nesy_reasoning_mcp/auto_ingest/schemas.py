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
    ExclusiveGroupInput,
    IndependenceInput,
    PropositionRecord,
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


class ReviewQueueStatus(StrEnum):
    """Lifecycle status for persisted review queue records."""

    PENDING = "pending"
    COMMITTED = "committed"
    RESOLVED = "resolved"


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


class IngestionInput(BaseModel):
    """External evidence input for an Agent SDK dry-run ingestion pass."""

    model_config = ConfigDict(extra="forbid")

    evidence: list[EvidenceRecord] = Field(default_factory=list)
    urls: list[str] = Field(default_factory=list)
    propositions: list[PropositionRecord] = Field(default_factory=list)
    exclusive_groups: list[ExclusiveGroupInput] = Field(default_factory=list)
    independence_records: list[IndependenceInput] = Field(default_factory=list)
    context_metadata: dict[str, Any] = Field(default_factory=dict)
    task: str | None = None
    question: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("urls")
    @classmethod
    def strip_urls(cls, value: list[str]) -> list[str]:
        """Strip URL inputs and reject empty values."""
        urls = [item.strip() for item in value]
        if any(not item for item in urls):
            raise ValueError("urls must not contain empty values")
        return urls

    @field_validator("task", "question")
    @classmethod
    def strip_optional_text(cls, value: str | None) -> str | None:
        """Strip optional prompt text and reject empty provided values."""
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


class CandidateRelationBatch(BaseModel):
    """Structured extractor output."""

    model_config = ConfigDict(extra="forbid")

    candidates: list[CandidateRelation] = Field(default_factory=list)


class ReviewDecisionBatch(BaseModel):
    """Structured reviewer output."""

    model_config = ConfigDict(extra="forbid")

    reviews: list[ReviewDecision] = Field(default_factory=list)


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
    written_relation_ids: list[str] = Field(default_factory=list)
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


class ReviewQueueRecord(BaseModel):
    """A persisted candidate relation awaiting explicit review action."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(default_factory=lambda: f"queue_{uuid4().hex}", min_length=1)
    status: ReviewQueueStatus = ReviewQueueStatus.PENDING
    run_id: str = Field(min_length=1)
    run_metadata: dict[str, Any] = Field(default_factory=dict)
    candidate: CandidateRelation
    review: ReviewDecision | None = None
    gate_result: GateResult
    diagnostics: list[Diagnostic] = Field(default_factory=list)
    propositions: list[PropositionRecord] = Field(default_factory=list)
    context_metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    updated_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    committed_relation_ids: list[str] = Field(default_factory=list)
    resolution: dict[str, Any] = Field(default_factory=dict)

    @field_validator("id", "run_id", "created_at", "updated_at")
    @classmethod
    def strip_required_text(cls, value: str) -> str:
        """Strip required text fields and reject empty values."""
        stripped = value.strip()
        if not stripped:
            raise ValueError("must not be empty")
        return stripped

    @field_validator("committed_relation_ids")
    @classmethod
    def strip_relation_ids(cls, value: list[str]) -> list[str]:
        """Strip relation IDs and reject empty provided values."""
        stripped = [item.strip() for item in value]
        if any(not item for item in stripped):
            raise ValueError("committed_relation_ids must not contain empty values")
        return stripped

    @model_validator(mode="after")
    def require_candidate_consistency(self) -> ReviewQueueRecord:
        """Ensure review and gate entries refer to the queued candidate."""
        if self.gate_result.action != GateAction.QUEUE:
            raise ValueError("gate_result action must be queue")
        if self.gate_result.candidate_id != self.candidate.id:
            raise ValueError("gate_result candidate_id must match candidate id")
        if self.review is not None and self.review.candidate_id != self.candidate.id:
            raise ValueError("review candidate_id must match candidate id")
        return self


class ReviewQueueFilter(BaseModel):
    """Filter for listing persisted review queue records."""

    model_config = ConfigDict(extra="forbid")

    status: ReviewQueueStatus | None = None
    run_id: str | None = None
    candidate_id: str | None = None
    store_id: str | None = None
    context_id: str | None = None

    @field_validator("run_id", "candidate_id", "store_id", "context_id")
    @classmethod
    def strip_optional_text(cls, value: str | None) -> str | None:
        """Strip optional filter values and reject empty provided values."""
        if value is None:
            return None
        stripped = value.strip()
        if not stripped:
            raise ValueError("must not be empty")
        return stripped


class ListReviewQueueInput(BaseModel):
    """Input for `nesy.list_review_queue`."""

    model_config = ConfigDict(extra="forbid")

    filter: ReviewQueueFilter = Field(default_factory=ReviewQueueFilter)
    limit: int = Field(default=50, ge=1, le=200)
    cursor: str | None = None

    @field_validator("cursor")
    @classmethod
    def strip_cursor(cls, value: str | None) -> str | None:
        """Strip cursor values and reject empty provided values."""
        if value is None:
            return None
        stripped = value.strip()
        if not stripped:
            raise ValueError("must not be empty")
        return stripped


class CommitReviewedRelationsInput(BaseModel):
    """Input for `nesy.commit_reviewed_relations`."""

    model_config = ConfigDict(extra="forbid")

    ids: list[str] = Field(min_length=1)
    min_write_confidence: float = Field(default=0.85, ge=0, le=1)
    max_depth: int = Field(default=8, ge=1, le=20)
    min_confidence: float = Field(default=0.0, ge=0, le=1)
    include_soft: bool = False

    @field_validator("ids")
    @classmethod
    def strip_ids(cls, value: list[str]) -> list[str]:
        """Strip IDs, reject empties, and de-duplicate in input order."""
        stripped = [item.strip() for item in value]
        if any(not item for item in stripped):
            raise ValueError("ids must not contain empty values")
        return list(dict.fromkeys(stripped))


class ResolveReviewQueueInput(BaseModel):
    """Input for `nesy.resolve_review_queue`."""

    model_config = ConfigDict(extra="forbid")

    ids: list[str] = Field(min_length=1)
    reason: str = Field(min_length=1)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("ids")
    @classmethod
    def strip_ids(cls, value: list[str]) -> list[str]:
        """Strip IDs, reject empties, and de-duplicate in input order."""
        stripped = [item.strip() for item in value]
        if any(not item for item in stripped):
            raise ValueError("ids must not contain empty values")
        return list(dict.fromkeys(stripped))

    @field_validator("reason")
    @classmethod
    def strip_reason(cls, value: str) -> str:
        """Strip resolution reason and reject empty values."""
        stripped = value.strip()
        if not stripped:
            raise ValueError("must not be empty")
        return stripped


class ValidateCandidateRelationsInput(BaseModel):
    """Input for `nesy.validate_candidate_relations` pre-write checks."""

    model_config = ConfigDict(extra="forbid")

    candidates: list[CandidateRelation] = Field(min_length=1)
    reviews: list[ReviewDecision] = Field(default_factory=list)
    propositions: list[PropositionRecord] = Field(default_factory=list)
    min_write_confidence: float = Field(default=0.85, ge=0, le=1)
    max_depth: int = Field(default=8, ge=1, le=20)
    min_confidence: float = Field(default=0.0, ge=0, le=1)
    include_soft: bool = False
