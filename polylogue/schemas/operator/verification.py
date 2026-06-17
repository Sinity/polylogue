"""Verification/listing helpers for schema operator workflows."""

from __future__ import annotations

from pathlib import Path

from polylogue.schemas.operator.models import (
    ArtifactCohortListResult,
    ArtifactCoverageResult,
    ArtifactObservationListResult,
)
from polylogue.schemas.validation.artifacts import (
    inspect_raw_artifact_coverage,
    list_artifact_cohort_rows,
    list_artifact_observation_rows,
)
from polylogue.schemas.validation.corpus import verify_raw_corpus
from polylogue.schemas.validation.models import SchemaVerificationReport
from polylogue.schemas.validation.requests import (
    ArtifactCoverageRequest,
    ArtifactObservationQuery,
    SchemaVerificationRequest,
)


def run_schema_verification(request: SchemaVerificationRequest, *, db_path: Path) -> SchemaVerificationReport:
    return verify_raw_corpus(db_path=db_path, request=request)


def run_artifact_coverage(request: ArtifactCoverageRequest, *, db_path: Path) -> ArtifactCoverageResult:
    return ArtifactCoverageResult(report=inspect_raw_artifact_coverage(db_path=db_path, request=request))


def list_artifact_observations(
    request: ArtifactObservationQuery,
    *,
    db_path: Path,
) -> ArtifactObservationListResult:
    return ArtifactObservationListResult(
        rows=list_artifact_observation_rows(db_path=db_path, request=request),
    )


def list_artifact_cohorts(
    request: ArtifactObservationQuery,
    *,
    db_path: Path,
) -> ArtifactCohortListResult:
    return ArtifactCohortListResult(
        rows=list_artifact_cohort_rows(db_path=db_path, request=request),
    )


__all__ = [
    "list_artifact_cohorts",
    "list_artifact_observations",
    "run_artifact_coverage",
    "run_schema_verification",
]
