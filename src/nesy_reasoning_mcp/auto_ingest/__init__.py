"""Shared schemas for automated evidence ingestion workflows."""

from nesy_reasoning_mcp.auto_ingest.policy import (
    DRY_RUN_TOOL_ALLOWLIST,
    WRITE_MODE_TOOL_ALLOWLIST,
)
from nesy_reasoning_mcp.auto_ingest.schemas import (
    CandidateRelation,
    CandidateRelationBatch,
    EvidenceRecord,
    GateAction,
    GateResult,
    IngestionInput,
    IngestionMode,
    IngestionReport,
    ReviewDecision,
    ReviewDecisionBatch,
    ReviewDecisionValue,
    ValidateCandidateRelationsInput,
)

__all__ = [
    "CandidateRelation",
    "CandidateRelationBatch",
    "DRY_RUN_TOOL_ALLOWLIST",
    "EvidenceRecord",
    "GateAction",
    "GateResult",
    "IngestionInput",
    "IngestionMode",
    "IngestionReport",
    "ReviewDecision",
    "ReviewDecisionBatch",
    "ReviewDecisionValue",
    "ValidateCandidateRelationsInput",
    "WRITE_MODE_TOOL_ALLOWLIST",
]
