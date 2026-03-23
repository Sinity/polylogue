"""Publication-summary helpers for static-site generation."""

from __future__ import annotations

from pathlib import Path

from polylogue.paths import archive_root as default_archive_root
from polylogue.publication import (
    ArtifactProofSummary,
    PublicationRunSummary,
    SemanticProofSuiteSummary,
    SemanticProofSummary,
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


def load_semantic_proof_summary(
    *,
    db_path: Path,
    archive_root: Path | None = None,
) -> SemanticProofSuiteSummary:
    """Load the semantic-preservation proof summary for publication embedding."""
    from polylogue.rendering.semantic_proof import prove_semantic_surface_suite

    report = prove_semantic_surface_suite(
        db_path=db_path,
        archive_root=archive_root or default_archive_root(),
    )
    return SemanticProofSuiteSummary(
        surface_count=report.surface_count,
        clean_surfaces=report.clean_surfaces,
        critical_surfaces=report.critical_surfaces,
        total_conversations=report.total_conversations,
        preserved_checks=report.preserved_checks,
        declared_loss_checks=report.declared_loss_checks,
        critical_loss_checks=report.critical_loss_checks,
        metric_summary=report.metric_summary,
        surfaces={
            surface: SemanticProofSummary(
                surface=surface,
                total_conversations=surface_report.total_conversations,
                provider_count=surface_report.provider_count,
                clean_conversations=surface_report.clean_conversations,
                critical_conversations=surface_report.critical_conversations,
                preserved_checks=surface_report.preserved_checks,
                declared_loss_checks=surface_report.declared_loss_checks,
                critical_loss_checks=surface_report.critical_loss_checks,
                metric_summary=surface_report.metric_summary,
                clean=surface_report.is_clean,
            )
            for surface, surface_report in sorted(report.surfaces.items())
        },
        clean=report.is_clean,
    )
