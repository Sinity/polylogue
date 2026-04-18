"""Publication-manifest assembly and persistence for static-site builds."""

from __future__ import annotations

import asyncio
import inspect
import json
from collections.abc import Awaitable
from pathlib import Path
from typing import Protocol
from uuid import uuid4

from polylogue.publication import (
    ArchiveMaintenanceSummary,
    ArchivePublicationSummary,
    ArtifactProofSummary,
    DerivedModelPublicationSummary,
    OutputManifest,
    PublicationRunSummary,
    SiteOutputSummary,
    SitePublicationManifest,
)
from polylogue.site.models import ArchiveIndexStats, ConversationPageBuildStats, SiteConfig
from polylogue.storage.store import PublicationRecord, RunRecord


class _LatestRunQueries(Protocol):
    def get_latest_run(self) -> RunRecord | None | Awaitable[RunRecord | None]: ...


class _PublicationRepository(Protocol):
    async def record_publication(self, record: PublicationRecord) -> None: ...


def _backend_db_path(backend: object) -> Path | None:
    db_path = getattr(backend, "db_path", None)
    return db_path if isinstance(db_path, Path) else None


def _backend_queries(backend: object) -> _LatestRunQueries | None:
    queries = getattr(backend, "queries", None)
    return queries if queries is not None and hasattr(queries, "get_latest_run") else None


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


async def load_latest_run_summary(backend: object) -> PublicationRunSummary | None:
    """Return the latest pipeline run summary for manifest embedding."""
    queries = _backend_queries(backend)
    if queries is None:
        return None
    record = queries.get_latest_run()
    if inspect.isawaitable(record):
        record = await record
    return build_latest_run_summary(record)


async def load_artifact_proof_summary_for_backend(
    backend: object,
) -> ArtifactProofSummary | None:
    """Return durable artifact-proof summary for manifest embedding."""
    db_path = _backend_db_path(backend)
    if db_path is None:
        return None
    return await asyncio.to_thread(load_artifact_proof_summary, db_path=db_path)


async def load_archive_maintenance_summary_for_backend(
    backend: object,
) -> ArchiveMaintenanceSummary | None:
    """Return derived-model maintenance summary for manifest embedding."""
    db_path = _backend_db_path(backend)
    if db_path is None:
        return None
    return await asyncio.to_thread(load_archive_maintenance_summary, db_path=db_path)


async def build_site_publication_manifest(
    *,
    output_dir: Path,
    config: SiteConfig,
    archive_stats: ArchiveIndexStats,
    conversation_pages: ConversationPageBuildStats,
    generated_at: str,
    duration_ms: int,
    provider_index_pages: int,
    dashboard_pages: int,
    search_status: str,
    incremental: bool,
    latest_run: PublicationRunSummary | None,
    artifact_proof: ArtifactProofSummary | None,
    maintenance: ArchiveMaintenanceSummary | None,
) -> SitePublicationManifest:
    """Build the typed site publication manifest from build outputs."""
    artifact_manifest = await asyncio.to_thread(
        OutputManifest.scan,
        output_dir,
        include_hashes=True,
        exclude_paths={"site-manifest.json"},
    )
    return SitePublicationManifest(
        publication_id=f"site-{uuid4().hex[:16]}",
        generated_at=generated_at,
        output_dir=str(output_dir),
        duration_ms=duration_ms,
        config={
            "title": config.title,
            "description": config.description,
            "enable_search": config.enable_search,
            "search_provider": str(config.search_provider),
            "conversations_per_page": config.conversations_per_page,
            "include_dashboard": config.include_dashboard,
        },
        archive=ArchivePublicationSummary(
            total_conversations=archive_stats.total_conversations,
            total_messages=archive_stats.total_messages,
            provider_count=len(archive_stats.provider_counts),
            provider_counts=dict(sorted(archive_stats.provider_counts.items())),
            provider_messages=dict(sorted(archive_stats.provider_messages.items())),
        ),
        outputs=SiteOutputSummary(
            root_index_pages=1,
            provider_index_pages=provider_index_pages,
            dashboard_pages=dashboard_pages,
            total_index_pages=1 + provider_index_pages + dashboard_pages,
            total_conversation_pages=conversation_pages.total,
            rendered_conversation_pages=conversation_pages.rendered,
            reused_conversation_pages=conversation_pages.reused,
            failed_conversation_pages=conversation_pages.failed,
            search_documents=(archive_stats.total_conversations if config.enable_search else 0),
            search_enabled=config.enable_search,
            search_provider=(str(config.search_provider) if config.enable_search else None),
            search_status=search_status,
            incremental=incremental,
        ),
        latest_run=latest_run,
        artifact_proof=artifact_proof,
        maintenance=maintenance,
        artifacts=artifact_manifest,
    )


def write_site_publication_manifest(
    output_dir: Path,
    manifest: SitePublicationManifest,
) -> Path:
    """Persist the site publication manifest to disk."""
    manifest_path = output_dir / "site-manifest.json"
    manifest_path.write_text(
        json.dumps(manifest.model_dump(mode="json"), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return manifest_path


async def record_site_publication_manifest(
    repository: _PublicationRepository,
    manifest: SitePublicationManifest,
) -> None:
    """Persist the site publication manifest to the archive database."""
    await repository.record_publication(
        PublicationRecord(
            publication_id=manifest.publication_id,
            publication_kind=manifest.publication_kind,
            generated_at=manifest.generated_at,
            output_dir=manifest.output_dir,
            duration_ms=manifest.duration_ms,
            manifest=manifest.model_dump(mode="json"),
        )
    )


__all__ = [
    "build_site_publication_manifest",
    "load_archive_maintenance_summary_for_backend",
    "load_artifact_proof_summary_for_backend",
    "load_latest_run_summary",
    "record_site_publication_manifest",
    "write_site_publication_manifest",
]
