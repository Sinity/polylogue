"""Provider-scoped semantic proof models."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from polylogue.rendering.semantic_proof_conversation_models import SemanticConversationProof
from polylogue.rendering.semantic_proof_model_support import empty_metric_counts


@dataclass(frozen=True)
class ProviderSemanticProof:
    """Per-provider semantic-preservation proof summary."""

    provider: str
    total_conversations: int = 0
    clean_conversations: int = 0
    critical_conversations: int = 0
    preserved_checks: int = 0
    declared_loss_checks: int = 0
    critical_loss_checks: int = 0
    metric_summary: dict[str, dict[str, int]] = field(default_factory=dict)

    @property
    def is_clean(self) -> bool:
        return self.critical_conversations == 0

    @property
    def clean(self) -> bool:
        return self.is_clean

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "total_conversations": self.total_conversations,
            "clean_conversations": self.clean_conversations,
            "critical_conversations": self.critical_conversations,
            "preserved_checks": self.preserved_checks,
            "declared_loss_checks": self.declared_loss_checks,
            "critical_loss_checks": self.critical_loss_checks,
            "metric_summary": dict(sorted(self.metric_summary.items())),
            "clean": self.clean,
        }


def build_provider_reports(
    conversations: list[SemanticConversationProof],
) -> dict[str, ProviderSemanticProof]:
    provider_totals: dict[str, dict[str, Any]] = {}
    for proof in conversations:
        state = provider_totals.setdefault(
            proof.provider,
            {
                "total_conversations": 0,
                "clean_conversations": 0,
                "critical_conversations": 0,
                "preserved_checks": 0,
                "declared_loss_checks": 0,
                "critical_loss_checks": 0,
                "metric_summary": {},
            },
        )
        state["total_conversations"] += 1
        if proof.is_clean:
            state["clean_conversations"] += 1
        else:
            state["critical_conversations"] += 1
        state["preserved_checks"] += len(proof.preserved_checks)
        state["declared_loss_checks"] += len(proof.declared_loss_checks)
        state["critical_loss_checks"] += len(proof.critical_loss_checks)
        for metric, counts in proof.metric_summary.items():
            metric_counts = state["metric_summary"].setdefault(metric, empty_metric_counts())
            for status, count in counts.items():
                metric_counts[status] += count

    return {
        provider: ProviderSemanticProof(
            provider=provider,
            total_conversations=state["total_conversations"],
            clean_conversations=state["clean_conversations"],
            critical_conversations=state["critical_conversations"],
            preserved_checks=state["preserved_checks"],
            declared_loss_checks=state["declared_loss_checks"],
            critical_loss_checks=state["critical_loss_checks"],
            metric_summary=dict(sorted(state["metric_summary"].items())),
        )
        for provider, state in sorted(provider_totals.items())
    }


__all__ = [
    "ProviderSemanticProof",
    "build_provider_reports",
]
