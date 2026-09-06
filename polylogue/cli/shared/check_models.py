"""Shared typed models for the check command surface."""

from __future__ import annotations

from dataclasses import dataclass

from polylogue.core.json import JSONDocument
from polylogue.readiness import ReadinessReport
from polylogue.schemas.validation.models import ArtifactCoverageReport, SchemaVerificationReport
from polylogue.storage.artifacts.views import ArtifactCohortSummary
from polylogue.storage.runtime import ArtifactObservationRecord


@dataclass
class CheckCommandResult:
    """Typed output surface for the check command workflow."""

    report: ReadinessReport
    runtime_report: ReadinessReport | None = None
    daemon_report: JSONDocument | None = None
    schema_report: SchemaVerificationReport | None = None
    coverage_report: ArtifactCoverageReport | None = None
    artifact_rows: list[ArtifactObservationRecord] | None = None
    cohort_rows: list[ArtifactCohortSummary] | None = None
    blob_report: JSONDocument | None = None


__all__ = ["CheckCommandResult"]
