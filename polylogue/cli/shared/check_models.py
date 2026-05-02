"""Shared typed models for the check command surface."""

from __future__ import annotations

from dataclasses import dataclass

from polylogue.core.json import JSONDocument, json_document
from polylogue.maintenance.resources import ResourceBoundaryReport
from polylogue.readiness import ReadinessReport
from polylogue.schemas.validation.models import ArtifactProofReport, SchemaVerificationReport
from polylogue.storage.artifacts.views import ArtifactCohortSummary
from polylogue.storage.repair import RepairResult
from polylogue.storage.runtime import ArtifactObservationRecord


@dataclass(frozen=True)
class VacuumResult:
    """Machine-safe VACUUM result payload shared by workflow and renderers."""

    ok: bool
    detail: str
    preview: bool = False

    def to_dict(self) -> JSONDocument:
        payload: dict[str, str | bool] = {"ok": self.ok, "detail": self.detail}
        if self.preview:
            payload["preview"] = True
        return json_document(payload)


@dataclass
class CheckCommandResult:
    """Typed output surface for the check command workflow."""

    report: ReadinessReport
    runtime_report: ReadinessReport | None = None
    daemon_report: JSONDocument | None = None
    schema_report: SchemaVerificationReport | None = None
    proof_report: ArtifactProofReport | None = None
    artifact_rows: list[ArtifactObservationRecord] | None = None
    cohort_rows: list[ArtifactCohortSummary] | None = None
    blob_report: JSONDocument | None = None
    maintenance_results: list[RepairResult] | None = None
    maintenance_targets: tuple[str, ...] = ()
    resource_boundary: ResourceBoundaryReport | None = None
    vacuum_result: VacuumResult | None = None


__all__ = ["CheckCommandResult", "VacuumResult"]
