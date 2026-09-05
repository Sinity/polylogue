"""Typed reports for derived insight rows."""

from __future__ import annotations

from dataclasses import dataclass

from pydantic import Field

from polylogue.analysis.archive_models import ARCHIVE_INSIGHT_CONTRACT_VERSION, ArchiveInsightModel
from polylogue.maintenance.targets import build_maintenance_target_catalog

_REPAIR_HINT = build_maintenance_target_catalog().repair_hint(("session_insights",), include_run_all=True)


class InsightReadinessQuery(ArchiveInsightModel):
    insights: tuple[str, ...] = ()
    origin: str | None = None
    since: str | None = None
    until: str | None = None


class InsightStorageArtifact(ArchiveInsightModel):
    name: str
    present: bool


class InsightVersionCoverage(ArchiveInsightModel):
    field: str
    current_version: int
    versions: dict[str, int] = Field(default_factory=dict)
    incompatible_count: int = 0


class InsightOriginCoverage(ArchiveInsightModel):
    origin: str
    row_count: int
    min_time: str | None = None
    max_time: str | None = None


class InsightReadinessEntry(ArchiveInsightModel):
    insight_name: str
    display_name: str
    contract_version: int = ARCHIVE_INSIGHT_CONTRACT_VERSION
    table_present: bool = True
    row_count: int = 0
    expected_row_count: int | None = None
    missing_count: int = 0
    stale_count: int = 0
    orphan_count: int = 0
    incompatible_count: int = 0
    degraded_count: int = 0
    fallback_reason_counts: dict[str, int] = Field(default_factory=dict)
    storage_artifacts: tuple[InsightStorageArtifact, ...] = ()
    origin_coverage: tuple[InsightOriginCoverage, ...] = ()
    version_coverage: tuple[InsightVersionCoverage, ...] = ()
    schema_contract_issues: tuple[str, ...] = ()
    min_time: str | None = None
    max_time: str | None = None
    repair_command: str = _REPAIR_HINT
    evidence: tuple[str, ...] = ()

    @property
    def diverged(self) -> bool:
        """The stored rows do not reflect the sources they were built from.

        Divergence is the content comparison every derived object makes: the
        table is absent, its rows outlive their session, or they were built
        against a schema shape the reader cannot trust. Callers that must not
        publish untrustworthy rows gate on this; callers reporting coverage
        read the counts directly.
        """
        return not self.table_present or bool(self.incompatible_count or self.stale_count or self.orphan_count)

    @property
    def incomplete(self) -> bool:
        """Sources exist that have no row yet -- ordinary convergence backlog."""
        return bool(self.missing_count) or (
            self.expected_row_count is not None and self.row_count < self.expected_row_count
        )


class InsightReadinessReport(ArchiveInsightModel):
    checked_at: str
    total_sessions: int = 0
    origin: str | None = None
    since: str | None = None
    until: str | None = None
    insights: tuple[InsightReadinessEntry, ...] = ()
    # Readiness is one signal: has convergence caught up. ``None`` means the
    # debt ledger could not be read -- unknown is never success. ``debt_stages``
    # names the stages still holding retryable debt when it has not.
    converged: bool | None = None
    debt_stages: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class InsightReadinessSpec:
    """Public name and display label for one archive insight surface."""

    insight_name: str
    display_name: str


_SPECS: tuple[InsightReadinessSpec, ...] = (
    InsightReadinessSpec("session_profiles", "Session Profiles"),
    InsightReadinessSpec("session_work_events", "Work Events"),
    InsightReadinessSpec("session_phases", "Session Phases"),
    InsightReadinessSpec("session_runs", "Session Runs"),
    InsightReadinessSpec("session_observed_events", "Observed Events"),
    InsightReadinessSpec("session_context_snapshots", "Context Snapshots"),
    InsightReadinessSpec("threads", "Threads"),
    InsightReadinessSpec("session_tag_rollups", "Session Tag Rollups"),
    InsightReadinessSpec("archive_coverage", "Archive Coverage"),
)


_SPEC_BY_NAME = {spec.insight_name: spec for spec in _SPECS}
_ALIASES = {
    **{spec.insight_name.replace("_", "-"): spec.insight_name for spec in _SPECS},
    "profiles": "session_profiles",
    "work-events": "session_work_events",
    "phases": "session_phases",
    "runs": "session_runs",
    "run-projection": "session_runs",
    "observed-events": "session_observed_events",
    "context-snapshots": "session_context_snapshots",
    "threads": "threads",
    "tags": "session_tag_rollups",
    "coverage": "archive_coverage",
}


def known_insight_readiness_names() -> tuple[str, ...]:
    return tuple(spec.insight_name for spec in _SPECS)


def insight_display_name(name: str) -> str:
    """Public display label for one insight readiness surface."""
    return _SPEC_BY_NAME[normalize_insight_readiness_name(name)].display_name


def normalize_insight_readiness_name(value: str) -> str:
    normalized = value.strip().replace("-", "_")
    if normalized in _SPEC_BY_NAME:
        return normalized
    alias = _ALIASES.get(value.strip()) or _ALIASES.get(value.strip().replace("_", "-"))
    if alias is not None:
        return alias
    raise ValueError(f"Unknown insight readiness target: {value}")


__all__ = [
    "InsightOriginCoverage",
    "InsightReadinessEntry",
    "InsightReadinessQuery",
    "InsightReadinessReport",
    "InsightStorageArtifact",
    "InsightVersionCoverage",
    "insight_display_name",
    "known_insight_readiness_names",
    "normalize_insight_readiness_name",
]
