"""Publication-summary helpers for static-site generation."""

from __future__ import annotations

from pathlib import Path

from polylogue.publication import (
    ArchiveMaintenanceSummary,
    ArtifactProofSummary,
    DerivedModelPublicationSummary,
    PublicationRunSummary,
)
from polylogue.storage.store import RunRecord


def build_latest_run_summary(record: RunRecord | None) -> PublicationRunSummary | None:
    """Convert a persisted run record into the embedded publication summary."""
    if record is None:
        return None
    run_id = getattr(record, "run_id", None)
    timestamp = getattr(record, "timestamp", None)
    if not isinstance(run_id, str) or not isinstance(timestamp, str):
        return None
    counts = getattr(record, "counts", None)
    indexed = getattr(record, "indexed", None)
    duration_ms = getattr(record, "duration_ms", None)
    return PublicationRunSummary(
        run_id=run_id,
        timestamp=timestamp,
        counts=counts if isinstance(counts, dict) or counts is None else None,
        indexed=indexed if isinstance(indexed, bool) or indexed is None else None,
        duration_ms=duration_ms if isinstance(duration_ms, int) or duration_ms is None else None,
    )


def load_artifact_proof_summary(*, db_path: Path) -> ArtifactProofSummary:
    """Load the durable artifact-proof summary for publication embedding."""
    from polylogue.schemas.verification_artifacts import prove_raw_artifact_coverage
    from polylogue.schemas.verification_requests import ArtifactProofRequest

    report = prove_raw_artifact_coverage(db_path=db_path, request=ArtifactProofRequest())
    return ArtifactProofSummary(
        total_records=report.total_records,
        provider_count=len(report.providers),
        contract_backed_records=report.contract_backed_records,
        unsupported_parseable_records=report.unsupported_parseable_records,
        recognized_non_parseable_records=report.recognized_non_parseable_records,
        unknown_records=report.unknown_records,
        decode_errors=report.decode_errors,
        linked_sidecars=report.linked_sidecars,
        orphan_sidecars=report.orphan_sidecars,
        subagent_streams=report.subagent_streams,
        streams_with_sidecars=report.streams_with_sidecars,
        artifact_counts=report.artifact_counts,
        package_versions=report.package_versions,
        element_kinds=report.element_kinds,
        resolution_reasons=report.resolution_reasons,
        clean=report.is_clean,
    )


def load_archive_maintenance_summary(*, db_path: Path) -> ArchiveMaintenanceSummary:
    """Load the live derived-model maintenance snapshot for publication embedding."""
    from polylogue.storage.backends.connection import open_connection
    from polylogue.storage.derived_status import collect_derived_model_statuses_sync

    with open_connection(db_path) as conn:
        statuses = collect_derived_model_statuses_sync(conn)
    return ArchiveMaintenanceSummary(
        truth_source="live",
        derived_models={
            name: DerivedModelPublicationSummary(
                ready=status.ready,
                detail=status.detail,
                source_documents=status.source_documents,
                materialized_documents=status.materialized_documents,
                source_rows=status.source_rows,
                materialized_rows=status.materialized_rows,
                pending_documents=status.pending_documents,
                pending_rows=status.pending_rows,
                stale_rows=status.stale_rows,
                orphan_rows=status.orphan_rows,
                missing_provenance_rows=status.missing_provenance_rows,
                materializer_version=status.materializer_version,
                matches_version=status.matches_version,
            )
            for name, status in sorted(statuses.items())
        },
    )
