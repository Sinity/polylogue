"""Publication-summary helpers for static-site generation."""

from __future__ import annotations

from pathlib import Path

from polylogue.publication import ArtifactProofSummary, PublicationRunSummary
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
    from polylogue.schemas.verification import prove_raw_artifact_coverage

    report = prove_raw_artifact_coverage(db_path=db_path)
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
