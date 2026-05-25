"""Provider-neutral external retrieval inputs for ingestion workflows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from urllib.parse import quote

from pydantic import BaseModel, ConfigDict, Field, field_validator

from nesy_reasoning_mcp.auto_ingest.schemas import (
    CandidateRelation,
    EvidenceRecord,
    ReviewDecision,
)
from nesy_reasoning_mcp.schemas import Diagnostic
from nesy_reasoning_mcp.time_utils import utc_now_iso


class ExternalRetrievedEvidence(BaseModel):
    """Evidence returned by an external GraphRAG or memory retriever."""

    model_config = ConfigDict(extra="forbid")

    span: str = Field(min_length=1)
    title: str | None = None
    original_url: str | None = None
    source_document_id: str | None = None
    chunk_id: str | None = None
    score: float | None = Field(default=None, ge=0)
    retrieved_at: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator(
        "span",
        "title",
        "original_url",
        "source_document_id",
        "chunk_id",
        "retrieved_at",
    )
    @classmethod
    def strip_optional_text(cls, value: str | None) -> str | None:
        """Strip text fields and reject empty provided values."""
        if value is None:
            return None
        stripped = value.strip()
        if not stripped:
            raise ValueError("must not be empty")
        return stripped


class ExternalRetrievedCandidate(BaseModel):
    """Candidate relation returned by an external retriever."""

    model_config = ConfigDict(extra="forbid")

    candidate: CandidateRelation
    original_url: str | None = None
    source_document_id: str | None = None
    chunk_id: str | None = None
    score: float | None = Field(default=None, ge=0)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("original_url", "source_document_id", "chunk_id")
    @classmethod
    def strip_optional_text(cls, value: str | None) -> str | None:
        """Strip text fields and reject empty provided values."""
        if value is None:
            return None
        stripped = value.strip()
        if not stripped:
            raise ValueError("must not be empty")
        return stripped


class ExternalRetrievalBatch(BaseModel):
    """Batch of evidence or candidates from an external retrieval system."""

    model_config = ConfigDict(extra="forbid")

    retriever_name: str | None = None
    run_id: str | None = None
    evidence: list[ExternalRetrievedEvidence] = Field(default_factory=list)
    candidates: list[ExternalRetrievedCandidate] = Field(default_factory=list)
    reviews: list[ReviewDecision] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("retriever_name", "run_id")
    @classmethod
    def strip_optional_text(cls, value: str | None) -> str | None:
        """Strip text fields and reject empty provided values."""
        if value is None:
            return None
        stripped = value.strip()
        if not stripped:
            raise ValueError("must not be empty")
        return stripped


@dataclass(frozen=True)
class ExternalRetrievalConversion:
    """Converted retrieval records plus diagnostics and audit metadata."""

    evidence: list[EvidenceRecord]
    candidates: list[CandidateRelation]
    reviews: list[ReviewDecision]
    diagnostics: list[Diagnostic]
    metadata: dict[str, Any]
    missing_candidate_provenance_ids: list[str]

    @property
    def has_errors(self) -> bool:
        """Return whether conversion produced a blocking diagnostic."""
        return any(diagnostic.level == "error" for diagnostic in self.diagnostics)


def convert_external_retrieval_batch(
    batch: ExternalRetrievalBatch,
) -> ExternalRetrievalConversion:
    """Convert an external retrieval batch into existing ingestion schemas."""
    diagnostics: list[Diagnostic] = []
    evidence: list[EvidenceRecord] = []
    missing_candidate_provenance_ids: list[str] = []

    for index, item in enumerate(batch.evidence, start=1):
        provenance = _retrieval_metadata(batch, item)
        if not _has_complete_provenance(provenance):
            diagnostics.append(
                Diagnostic(
                    level="error",
                    code="RETRIEVAL_PROVENANCE_MISSING",
                    message=(
                        "retrieved evidence requires retriever_name and original_url "
                        "or source_document_id"
                    ),
                    related_ids=[f"evidence:{index}"],
                )
            )
            continue
        evidence.append(_evidence_record(item, provenance))

    candidates: list[CandidateRelation] = []
    for candidate_item in batch.candidates:
        provenance = _retrieval_metadata(batch, candidate_item)
        if not _has_complete_provenance(provenance):
            missing_candidate_provenance_ids.append(candidate_item.candidate.id)
        candidates.append(_candidate_relation(candidate_item, provenance))

    diagnostic_count = len(diagnostics) + (1 if missing_candidate_provenance_ids else 0)
    metadata = {
        "batch_count": 1,
        "retriever_name": batch.retriever_name,
        "run_id": batch.run_id,
        "evidence_count": len(batch.evidence),
        "accepted_evidence_count": len(evidence),
        "candidate_count": len(batch.candidates),
        "review_count": len(batch.reviews),
        "missing_candidate_provenance_ids": missing_candidate_provenance_ids,
        "diagnostic_count": diagnostic_count,
    }
    return ExternalRetrievalConversion(
        evidence=evidence,
        candidates=candidates,
        reviews=list(batch.reviews),
        diagnostics=diagnostics,
        metadata={key: value for key, value in metadata.items() if value is not None},
        missing_candidate_provenance_ids=missing_candidate_provenance_ids,
    )


def _evidence_record(
    item: ExternalRetrievedEvidence,
    provenance: dict[str, Any],
) -> EvidenceRecord:
    return EvidenceRecord(
        url=item.original_url or _retrieval_uri(provenance),
        title=item.title,
        span=item.span,
        source_type="external_retrieval",
        retrieved_at=item.retrieved_at or utc_now_iso(),
        metadata=_metadata_with_retrieval(item.metadata, provenance),
    )


def _candidate_relation(
    item: ExternalRetrievedCandidate,
    provenance: dict[str, Any],
) -> CandidateRelation:
    return item.candidate.model_copy(
        update={
            "metadata": _metadata_with_retrieval(item.candidate.metadata, provenance),
        }
    )


def _retrieval_metadata(
    batch: ExternalRetrievalBatch,
    item: ExternalRetrievedEvidence | ExternalRetrievedCandidate,
) -> dict[str, Any]:
    provenance = {
        "retriever_name": batch.retriever_name,
        "run_id": batch.run_id,
        "source_document_id": item.source_document_id,
        "chunk_id": item.chunk_id,
        "score": item.score,
        "original_url": item.original_url,
        "metadata": item.metadata,
    }
    return {key: value for key, value in provenance.items() if value is not None}


def _metadata_with_retrieval(
    metadata: dict[str, Any], provenance: dict[str, Any]
) -> dict[str, Any]:
    merged = dict(metadata)
    merged["retrieval"] = {
        **provenance,
        "provenance_complete": _has_complete_provenance(provenance),
    }
    return merged


def _has_complete_provenance(provenance: dict[str, Any]) -> bool:
    return bool(provenance.get("retriever_name")) and bool(
        provenance.get("original_url") or provenance.get("source_document_id")
    )


def _retrieval_uri(provenance: dict[str, Any]) -> str:
    document_id = quote(str(provenance["source_document_id"]), safe="")
    chunk_id = provenance.get("chunk_id")
    suffix = f"#{quote(str(chunk_id), safe='')}" if chunk_id else ""
    return f"external-retrieval://{document_id}{suffix}"
