"""Shared typed models for the check command surface."""

from __future__ import annotations

from dataclasses import dataclass

from polylogue.readiness import ReadinessReport
from polylogue.schemas.verification_models import ArtifactProofReport, SchemaVerificationReport
from polylogue.storage.artifact_views import ArtifactCohortSummary
from polylogue.storage.repair import RepairResult
from polylogue.storage.store import ArtifactObservationRecord


@dataclass(frozen=True)
class VacuumResult:
    """Machine-safe VACUUM result payload shared by workflow and renderers."""

    ok: bool
    detail: str
    preview: bool = False

    def to_dict(self) -> dict[str, str | bool]:
        payload: dict[str, str | bool] = {"ok": self.ok, "detail": self.detail}
        if self.preview:
            payload["preview"] = True
        return payload


@dataclass
class CheckCommandResult:
    """Typed output surface for the check command workflow."""

    report: ReadinessReport
    runtime_report: ReadinessReport | None = None
    schema_report: SchemaVerificationReport | None = None
    proof_report: ArtifactProofReport | None = None
    artifact_rows: list[ArtifactObservationRecord] | None = None
    cohort_rows: list[ArtifactCohortSummary] | None = None
    maintenance_results: list[RepairResult] | None = None
    maintenance_targets: tuple[str, ...] = ()
    vacuum_result: VacuumResult | None = None


__all__ = ["CheckCommandResult", "VacuumResult"]
