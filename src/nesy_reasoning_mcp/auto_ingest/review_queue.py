"""Helpers for persistent Agent SDK ingestion review queue records."""

from __future__ import annotations

from typing import Any

from pydantic import ValidationError

from nesy_reasoning_mcp.auto_ingest.schemas import (
    GateAction,
    IngestionReport,
    ReviewDecision,
    ReviewQueueRecord,
)
from nesy_reasoning_mcp.schemas import Diagnostic, PropositionRecord


def queued_records_from_report(
    report: IngestionReport,
    *,
    propositions: list[PropositionRecord],
    context_metadata: dict[str, Any],
) -> list[ReviewQueueRecord]:
    """Build pending review queue records for queued gate results in a report."""
    candidates_by_id = {candidate.id: candidate for candidate in report.candidates}
    reviews_by_id = _queue_reviews_by_candidate(report)
    records: list[ReviewQueueRecord] = []
    for gate_result in report.gate_results:
        if gate_result.action != GateAction.QUEUE:
            continue
        candidate = candidates_by_id.get(gate_result.candidate_id)
        if candidate is None:
            continue
        records.append(
            ReviewQueueRecord(
                run_id=report.run_id,
                run_metadata={
                    "generated_at": report.generated_at,
                    "mode": report.mode.value,
                    "metadata": report.metadata,
                    "diagnostic_count": len(report.diagnostics),
                },
                candidate=candidate,
                review=reviews_by_id.get(candidate.id),
                gate_result=gate_result,
                diagnostics=_diagnostics_for_candidate(report.diagnostics, candidate.id),
                propositions=propositions,
                context_metadata=context_metadata,
            )
        )
    return records


def _diagnostics_for_candidate(
    diagnostics: list[Diagnostic],
    candidate_id: str,
) -> list[Diagnostic]:
    return [diagnostic for diagnostic in diagnostics if candidate_id in diagnostic.related_ids]


def _queue_reviews_by_candidate(report: IngestionReport) -> dict[str, ReviewDecision]:
    aggregation = report.metadata.get("review_aggregation", {})
    aggregate_reviews = (
        aggregation.get("aggregate_reviews") if isinstance(aggregation, dict) else []
    )
    if isinstance(aggregate_reviews, list):
        reviews: list[ReviewDecision] = []
        for review in aggregate_reviews:
            if not isinstance(review, dict):
                continue
            try:
                reviews.append(ReviewDecision.model_validate(review))
            except ValidationError:
                continue
        if reviews:
            reviews_by_id: dict[str, ReviewDecision] = {}
            for review in reviews:
                reviews_by_id.setdefault(review.candidate_id, review)
            return reviews_by_id
    return {review.candidate_id: review for review in report.reviews}
