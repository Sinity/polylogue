"""Publication-manifest assembly and persistence for static-site builds."""

from __future__ import annotations

import asyncio
import inspect
import json
from pathlib import Path
from uuid import uuid4

from polylogue.publication import (
    ArchivePublicationSummary,
    OutputManifest,
    SiteOutputSummary,
    SitePublicationManifest,
)
from polylogue.site.models import ArchiveIndexStats, ConversationPageBuildStats, SiteConfig
from polylogue.site.publication_support import (
    build_latest_run_summary,
    load_archive_maintenance_summary,
    load_artifact_proof_summary,
    load_semantic_proof_summary,
)
from polylogue.storage.store import PublicationRecord


async def load_latest_run_summary(backend) -> object | None:
    """Return the latest pipeline run summary for manifest embedding."""
    record = backend.queries.get_latest_run()
    if inspect.isawaitable(record):
        record = await record
    return build_latest_run_summary(record)


async def load_artifact_proof_summary_for_backend(backend) -> object | None:
    """Return durable artifact-proof summary for manifest embedding."""
    if not isinstance(getattr(backend, "db_path", None), Path):
        return None
    return await asyncio.to_thread(load_artifact_proof_summary, db_path=backend.db_path)


async def load_semantic_proof_summary_for_backend(backend) -> object | None:
    """Return semantic-preservation proof summary for manifest embedding."""
    if not isinstance(getattr(backend, "db_path", None), Path):
        return None
    return await asyncio.to_thread(
        load_semantic_proof_summary,
        db_path=backend.db_path,
    )


async def load_archive_maintenance_summary_for_backend(backend) -> object | None:
    """Return derived-model maintenance summary for manifest embedding."""
    if not isinstance(getattr(backend, "db_path", None), Path):
        return None
    return await asyncio.to_thread(load_archive_maintenance_summary, db_path=backend.db_path)


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
    latest_run: object | None,
    artifact_proof: object | None,
    semantic_proof: object | None,
    maintenance: object | None,
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
            search_documents=(
                archive_stats.total_conversations if config.enable_search else 0
            ),
            search_enabled=config.enable_search,
            search_provider=(
                str(config.search_provider) if config.enable_search else None
            ),
            search_status=search_status,
            incremental=incremental,
        ),
        latest_run=latest_run,
        artifact_proof=artifact_proof,
        semantic_proof=semantic_proof,
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
    repository,
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
    "load_semantic_proof_summary_for_backend",
    "record_site_publication_manifest",
    "write_site_publication_manifest",
]
