import pytest
from pydantic import ValidationError

from nesy_reasoning_mcp.auto_ingest.external_retrieval import (
    ExternalRetrievalBatch,
    ExternalRetrievedCandidate,
    ExternalRetrievedEvidence,
    convert_external_retrieval_batch,
)
from nesy_reasoning_mcp.auto_ingest.schemas import CandidateRelation, EvidenceRecord


def _evidence() -> EvidenceRecord:
    return EvidenceRecord(url="https://example.com/source", span="A explicitly enables B.")


def _candidate() -> CandidateRelation:
    return CandidateRelation(
        id="candidate-1",
        source="A",
        target="B",
        relation_type="sufficient",
        confidence=0.9,
        evidence=[_evidence()],
    )


def test_external_retrieval_batch_rejects_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        ExternalRetrievalBatch.model_validate(
            {
                "retriever_name": "graph-rag",
                "evidence": [],
                "unexpected": True,
            }
        )


def test_external_retrieval_evidence_round_trips_provenance() -> None:
    conversion = convert_external_retrieval_batch(
        ExternalRetrievalBatch(
            retriever_name="graph-rag",
            run_id="retrieval-run-1",
            evidence=[
                ExternalRetrievedEvidence(
                    span="Retrieved evidence.",
                    title="Source",
                    source_document_id="doc-1",
                    chunk_id="chunk-1",
                    score=0.72,
                )
            ],
        )
    )

    assert conversion.has_errors is False
    record = conversion.evidence[0]
    assert record.url == "external-retrieval://doc-1#chunk-1"
    assert record.source_type == "external_retrieval"
    assert record.metadata["retrieval"] == {
        "retriever_name": "graph-rag",
        "run_id": "retrieval-run-1",
        "source_document_id": "doc-1",
        "chunk_id": "chunk-1",
        "score": 0.72,
        "metadata": {},
        "provenance_complete": True,
    }


def test_external_retrieval_candidate_round_trips_provenance() -> None:
    conversion = convert_external_retrieval_batch(
        ExternalRetrievalBatch(
            retriever_name="graph-rag",
            candidates=[
                ExternalRetrievedCandidate(
                    candidate=_candidate(),
                    original_url="https://example.com/doc",
                    source_document_id="doc-1",
                    score=0.91,
                    metadata={"rank": 1},
                )
            ],
        )
    )

    assert conversion.missing_candidate_provenance_ids == []
    assert conversion.candidates[0].metadata["retrieval"] == {
        "retriever_name": "graph-rag",
        "source_document_id": "doc-1",
        "score": 0.91,
        "original_url": "https://example.com/doc",
        "metadata": {"rank": 1},
        "provenance_complete": True,
    }


def test_external_retrieval_missing_evidence_provenance_is_error() -> None:
    conversion = convert_external_retrieval_batch(
        ExternalRetrievalBatch(
            evidence=[
                ExternalRetrievedEvidence(
                    span="Retrieved evidence without source.",
                )
            ],
        )
    )

    assert conversion.evidence == []
    assert conversion.has_errors is True
    assert conversion.diagnostics[0].code == "RETRIEVAL_PROVENANCE_MISSING"


def test_external_retrieval_missing_candidate_provenance_is_marked_for_queue() -> None:
    conversion = convert_external_retrieval_batch(
        ExternalRetrievalBatch(
            retriever_name="graph-rag",
            candidates=[ExternalRetrievedCandidate(candidate=_candidate())],
        )
    )

    assert conversion.has_errors is False
    assert conversion.missing_candidate_provenance_ids == ["candidate-1"]
    assert conversion.metadata["diagnostic_count"] == 1
    assert conversion.candidates[0].metadata["retrieval"]["provenance_complete"] is False
