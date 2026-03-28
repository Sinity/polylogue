"""Typed report models for semantic-proof surfaces."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


def _empty_metric_counts() -> dict[str, int]:
    return {"preserved": 0, "declared_loss": 0, "critical_loss": 0}


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
            counts = summary.setdefault(check.metric, _empty_metric_counts())
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


def _build_provider_reports(
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
            metric_counts = state["metric_summary"].setdefault(metric, _empty_metric_counts())
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
                metric = summary.setdefault(check.metric, _empty_metric_counts())
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
                metric_counts = summary.setdefault(metric, _empty_metric_counts())
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
    "ProviderSemanticProof",
    "SemanticConversationProof",
    "SemanticMetricCheck",
    "SemanticProofReport",
    "SemanticProofSuiteReport",
    "_build_provider_reports",
    "_empty_metric_counts",
]
