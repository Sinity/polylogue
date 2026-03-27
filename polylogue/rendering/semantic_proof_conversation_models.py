"""Conversation-scoped semantic proof models."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from polylogue.rendering.semantic_proof_model_support import empty_metric_counts


@dataclass(frozen=True)
class SemanticMetricCheck:
    """One measurable preservation or declared-loss claim."""

    metric: str
    status: str
    policy: str
    input_value: Any
    output_value: Any

    def to_dict(self) -> dict[str, Any]:
        return {
            "metric": self.metric,
            "status": self.status,
            "policy": self.policy,
            "input_value": self.input_value,
            "output_value": self.output_value,
        }


@dataclass(frozen=True)
class SemanticConversationProof:
    """Semantic proof result for one rendered/exported conversation surface."""

    conversation_id: str
    provider: str
    surface: str
    input_facts: dict[str, Any]
    output_facts: dict[str, Any]
    checks: list[SemanticMetricCheck] = field(default_factory=list)

    @property
    def critical_loss_checks(self) -> list[SemanticMetricCheck]:
        return [check for check in self.checks if check.status == "critical_loss"]

    @property
    def declared_loss_checks(self) -> list[SemanticMetricCheck]:
        return [check for check in self.checks if check.status == "declared_loss"]

    @property
    def preserved_checks(self) -> list[SemanticMetricCheck]:
        return [check for check in self.checks if check.status == "preserved"]

    @property
    def metric_summary(self) -> dict[str, dict[str, int]]:
        summary: dict[str, dict[str, int]] = {}
        for check in self.checks:
            counts = summary.setdefault(check.metric, empty_metric_counts())
            counts[check.status] += 1
        return dict(sorted(summary.items()))

    @property
    def is_clean(self) -> bool:
        return not self.critical_loss_checks

    def to_dict(self) -> dict[str, Any]:
        return {
            "conversation_id": self.conversation_id,
            "provider": self.provider,
            "surface": self.surface,
            "input_facts": self.input_facts,
            "output_facts": self.output_facts,
            "summary": {
                "preserved_checks": len(self.preserved_checks),
                "declared_loss_checks": len(self.declared_loss_checks),
                "critical_loss_checks": len(self.critical_loss_checks),
                "metric_summary": self.metric_summary,
                "clean": self.is_clean,
            },
            "checks": [check.to_dict() for check in self.checks],
        }


__all__ = [
    "SemanticConversationProof",
    "SemanticMetricCheck",
]
