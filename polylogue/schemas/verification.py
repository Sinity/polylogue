"""Schema verification facade over typed artifact-proof and corpus workflows."""

from __future__ import annotations

from pathlib import Path

from polylogue.paths import db_path as default_db_path
from polylogue.protocols import ProgressCallback
from polylogue.schemas.validator import SchemaValidator
from polylogue.schemas.verification_artifacts import (
    list_artifact_cohort_rows as _list_artifact_cohort_rows_impl,
)
from polylogue.schemas.verification_artifacts import (
    list_artifact_observation_rows as _list_artifact_observation_rows_impl,
)
from polylogue.schemas.verification_artifacts import (
    prove_raw_artifact_coverage as _prove_raw_artifact_coverage_impl,
)
from polylogue.schemas.verification_corpus import verify_raw_corpus as _verify_raw_corpus_impl
from polylogue.schemas.verification_models import (
    ArtifactProofReport,
    ProviderArtifactProof,
    ProviderSchemaVerification,
    SchemaVerificationReport,
)
from polylogue.schemas.verification_requests import (
    ArtifactObservationQuery,
    ArtifactProofRequest,
    SchemaVerificationRequest,
)
from polylogue.storage.store import ArtifactCohortSummary, ArtifactObservationRecord


def list_artifact_observation_rows(
    *,
    db_path: Path | None = None,
    providers: list[str] | None = None,
    support_statuses: list[str] | None = None,
    artifact_kinds: list[str] | None = None,
    record_limit: int | None = None,
    record_offset: int = 0,
) -> list[ArtifactObservationRecord]:
    return _list_artifact_observation_rows_impl(
        db_path=db_path or default_db_path(),
        request=ArtifactObservationQuery(
            providers=providers,
            support_statuses=support_statuses,
            artifact_kinds=artifact_kinds,
            record_limit=record_limit,
            record_offset=record_offset,
        ),
    )


def list_artifact_cohort_rows(
    *,
    db_path: Path | None = None,
    providers: list[str] | None = None,
    support_statuses: list[str] | None = None,
    artifact_kinds: list[str] | None = None,
    record_limit: int | None = None,
    record_offset: int = 0,
) -> list[ArtifactCohortSummary]:
    return _list_artifact_cohort_rows_impl(
        db_path=db_path or default_db_path(),
        request=ArtifactObservationQuery(
            providers=providers,
            support_statuses=support_statuses,
            artifact_kinds=artifact_kinds,
            record_limit=record_limit,
            record_offset=record_offset,
        ),
    )


def prove_raw_artifact_coverage(
    *,
    db_path: Path | None = None,
    providers: list[str] | None = None,
    record_limit: int | None = None,
    record_offset: int = 0,
) -> ArtifactProofReport:
    return _prove_raw_artifact_coverage_impl(
        db_path=db_path or default_db_path(),
        request=ArtifactProofRequest(
            providers=providers,
            record_limit=record_limit,
            record_offset=record_offset,
        ),
    )


def verify_raw_corpus(
    *,
    db_path: Path | None = None,
    providers: list[str] | None = None,
    max_samples: int | None = None,
    record_limit: int | None = None,
    record_offset: int = 0,
    quarantine_malformed: bool = False,
    progress_callback: ProgressCallback | None = None,
) -> SchemaVerificationReport:
    return _verify_raw_corpus_impl(
        db_path=db_path or default_db_path(),
        request=SchemaVerificationRequest(
            providers=providers,
            max_samples=max_samples,
            record_limit=record_limit,
            record_offset=record_offset,
            quarantine_malformed=quarantine_malformed,
            progress_callback=progress_callback,
        ),
    )


__all__ = [
    "ArtifactProofReport",
    "ArtifactObservationQuery",
    "ArtifactProofRequest",
    "ProviderArtifactProof",
    "ProviderSchemaVerification",
    "SchemaValidator",
    "SchemaVerificationReport",
    "SchemaVerificationRequest",
    "list_artifact_cohort_rows",
    "list_artifact_observation_rows",
    "prove_raw_artifact_coverage",
    "verify_raw_corpus",
]
