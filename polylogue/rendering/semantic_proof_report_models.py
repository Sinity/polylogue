"""Surface/suite semantic proof report models."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from polylogue.rendering.semantic_proof_conversation_models import SemanticConversationProof
from polylogue.rendering.semantic_proof_model_support import empty_metric_counts
from polylogue.rendering.semantic_proof_provider_models import ProviderSemanticProof


@dataclass(frozen=True)
class SemanticProofReport:
    """Aggregate semantic proof report for one output surface."""

    surface: str
    conversations: list[SemanticConversationProof]
    provider_reports: dict[str, ProviderSemanticProof]
    record_limit: int | None = None
    record_offset: int = 0
    provider_filters: list[str] = field(default_factory=list)

    @property
    def total_conversations(self) -> int:
        return len(self.conversations)

    @property
    def provider_count(self) -> int:
        return len(self.provider_reports)

    @property
    def providers(self) -> dict[str, ProviderSemanticProof]:
        return self.provider_reports

    @property
    def clean_conversations(self) -> int:
        return sum(1 for proof in self.conversations if proof.is_clean)

    @property
    def critical_conversations(self) -> int:
        return sum(1 for proof in self.conversations if not proof.is_clean)

    @property
    def preserved_checks(self) -> int:
        return sum(len(proof.preserved_checks) for proof in self.conversations)

    @property
    def declared_loss_checks(self) -> int:
        return sum(len(proof.declared_loss_checks) for proof in self.conversations)

    @property
    def critical_loss_checks(self) -> int:
        return sum(len(proof.critical_loss_checks) for proof in self.conversations)

    @property
    def metric_summary(self) -> dict[str, dict[str, int]]:
        summary: dict[str, dict[str, int]] = {}
        for proof in self.conversations:
            for check in proof.checks:
                metric = summary.setdefault(check.metric, empty_metric_counts())
                metric[check.status] += 1
        return dict(sorted(summary.items()))

    @property
    def is_clean(self) -> bool:
        return self.critical_conversations == 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "surface": self.surface,
            "record_limit": self.record_limit if self.record_limit is not None else "all",
            "record_offset": self.record_offset,
            "provider_filters": list(self.provider_filters),
            "summary": {
                "total_conversations": self.total_conversations,
                "provider_count": self.provider_count,
                "clean_conversations": self.clean_conversations,
                "critical_conversations": self.critical_conversations,
                "preserved_checks": self.preserved_checks,
                "declared_loss_checks": self.declared_loss_checks,
                "critical_loss_checks": self.critical_loss_checks,
                "metric_summary": self.metric_summary,
                "clean": self.is_clean,
            },
            "providers": {
                provider: stats.to_dict() for provider, stats in sorted(self.provider_reports.items())
            },
            "conversations": [proof.to_dict() for proof in self.conversations],
        }


@dataclass(frozen=True)
class SemanticProofSuiteReport:
    """Aggregate semantic proof report spanning multiple output surfaces."""

    surface_reports: dict[str, SemanticProofReport]
    record_limit: int | None = None
    record_offset: int = 0
    provider_filters: list[str] = field(default_factory=list)
    surface_filters: list[str] = field(default_factory=list)

    @property
    def surfaces(self) -> dict[str, SemanticProofReport]:
        return self.surface_reports

    @property
    def surface_count(self) -> int:
        return len(self.surface_reports)

    @property
    def clean_surfaces(self) -> int:
        return sum(1 for report in self.surface_reports.values() if report.is_clean)

    @property
    def critical_surfaces(self) -> int:
        return sum(1 for report in self.surface_reports.values() if not report.is_clean)

    @property
    def total_conversations(self) -> int:
        return sum(report.total_conversations for report in self.surface_reports.values())

    @property
    def preserved_checks(self) -> int:
        return sum(report.preserved_checks for report in self.surface_reports.values())

    @property
    def declared_loss_checks(self) -> int:
        return sum(report.declared_loss_checks for report in self.surface_reports.values())

    @property
    def critical_loss_checks(self) -> int:
        return sum(report.critical_loss_checks for report in self.surface_reports.values())

    @property
    def metric_summary(self) -> dict[str, dict[str, int]]:
        summary: dict[str, dict[str, int]] = {}
        for report in self.surface_reports.values():
            for metric, counts in report.metric_summary.items():
                metric_counts = summary.setdefault(metric, empty_metric_counts())
                for status, count in counts.items():
                    metric_counts[status] += count
        return dict(sorted(summary.items()))

    @property
    def is_clean(self) -> bool:
        return self.critical_surfaces == 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "record_limit": self.record_limit if self.record_limit is not None else "all",
            "record_offset": self.record_offset,
            "provider_filters": list(self.provider_filters),
            "surface_filters": list(self.surface_filters),
            "summary": {
                "surface_count": self.surface_count,
                "clean_surfaces": self.clean_surfaces,
                "critical_surfaces": self.critical_surfaces,
                "total_conversations": self.total_conversations,
                "preserved_checks": self.preserved_checks,
                "declared_loss_checks": self.declared_loss_checks,
                "critical_loss_checks": self.critical_loss_checks,
                "metric_summary": self.metric_summary,
                "clean": self.is_clean,
            },
            "surfaces": {
                surface: report.to_dict()
                for surface, report in sorted(self.surface_reports.items())
            },
        }


__all__ = [
    "SemanticProofReport",
    "SemanticProofSuiteReport",
]
