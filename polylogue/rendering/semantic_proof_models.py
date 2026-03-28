"""Small public root for semantic-proof report models."""

from __future__ import annotations

from polylogue.rendering.semantic_proof_conversation_models import (
    SemanticConversationProof,
    SemanticMetricCheck,
)
from polylogue.rendering.semantic_proof_model_support import empty_metric_counts
from polylogue.rendering.semantic_proof_provider_models import (
    ProviderSemanticProof,
    build_provider_reports,
)
from polylogue.rendering.semantic_proof_report_models import (
    SemanticProofReport,
    SemanticProofSuiteReport,
)


def _empty_metric_counts() -> dict[str, int]:
    return empty_metric_counts()


def _build_provider_reports(conversations):
    return build_provider_reports(conversations)


__all__ = [
    "ProviderSemanticProof",
    "SemanticConversationProof",
    "SemanticMetricCheck",
    "SemanticProofReport",
    "SemanticProofSuiteReport",
    "_build_provider_reports",
    "_empty_metric_counts",
]
