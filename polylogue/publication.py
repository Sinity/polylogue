"""Typed publication manifest models and output artifact scanning."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, field_validator


class PublishedArtifact(BaseModel):
    """One materialized artifact written to an output directory."""

    relative_path: str
    size_bytes: int
    sha256: str | None = None

    @field_validator("relative_path")
    @classmethod
    def non_empty_relative_path(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("relative_path cannot be empty")
        return value


class OutputManifest(BaseModel):
    """Stable manifest for a directory of materialized output artifacts."""

    schema_version: int = 1
    entry_count: int = 0
    entries: list[PublishedArtifact] = Field(default_factory=list)

    @classmethod
    def scan(
        cls,
        output_dir: Path,
        *,
        include_hashes: bool = True,
        exclude_paths: set[str] | None = None,
    ) -> OutputManifest:
        """Scan ``output_dir`` into a stable manifest."""
        entries: list[PublishedArtifact] = []
        excluded = {path.replace("\\", "/") for path in (exclude_paths or set())}
        if output_dir.exists():
            for path in sorted(output_dir.rglob("*")):
                if not path.is_file():
                    continue
                relative_path = str(path.relative_to(output_dir)).replace("\\", "/")
                if relative_path in excluded:
                    continue
                sha256: str | None = None
                if include_hashes:
                    digest = hashlib.sha256()
                    with path.open("rb") as handle:
                        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                            digest.update(chunk)
                    sha256 = digest.hexdigest()
                entries.append(
                    PublishedArtifact(
                        relative_path=relative_path,
                        size_bytes=path.stat().st_size,
                        sha256=sha256,
                    )
                )
        return cls(entry_count=len(entries), entries=entries)


class PublicationRunSummary(BaseModel):
    """Persisted summary of the latest ingest run seen by a publication build."""

    run_id: str
    timestamp: str
    counts: dict[str, Any] | None = None
    indexed: bool | None = None
    duration_ms: int | None = None


class ArtifactProofSummary(BaseModel):
    """Compact durable artifact-proof summary embedded in publication manifests."""

    total_records: int
    provider_count: int
    contract_backed_records: int
    unsupported_parseable_records: int
    recognized_non_parseable_records: int
    unknown_records: int
    decode_errors: int
    linked_sidecars: int
    orphan_sidecars: int
    subagent_streams: int
    streams_with_sidecars: int
    artifact_counts: dict[str, int] = Field(default_factory=dict)
    package_versions: dict[str, int] = Field(default_factory=dict)
    element_kinds: dict[str, int] = Field(default_factory=dict)
    resolution_reasons: dict[str, int] = Field(default_factory=dict)
    clean: bool


class SemanticProofSummary(BaseModel):
    """Per-surface semantic-preservation proof summary embedded in publication manifests."""

    surface: str
    total_conversations: int
    provider_count: int
    clean_conversations: int
    critical_conversations: int
    preserved_checks: int
    declared_loss_checks: int
    critical_loss_checks: int
    metric_summary: dict[str, dict[str, int]] = Field(default_factory=dict)
    clean: bool


class SemanticProofSuiteSummary(BaseModel):
    """Compact multi-surface semantic-proof summary embedded in publication manifests."""

    surface_count: int
    clean_surfaces: int
    critical_surfaces: int
    total_conversations: int
    preserved_checks: int
    declared_loss_checks: int
    critical_loss_checks: int
    metric_summary: dict[str, dict[str, int]] = Field(default_factory=dict)
    surfaces: dict[str, SemanticProofSummary] = Field(default_factory=dict)
    clean: bool


class DerivedModelPublicationSummary(BaseModel):
    """Compact readiness/freshness summary for one durable derived model."""

    ready: bool
    detail: str
    source_documents: int = 0
    materialized_documents: int = 0
    source_rows: int = 0
    materialized_rows: int = 0
    pending_documents: int = 0
    pending_rows: int = 0
    stale_rows: int = 0
    orphan_rows: int = 0
    missing_provenance_rows: int = 0
    materializer_version: int | None = None
    matches_version: bool | None = None


class ArchiveMaintenanceSummary(BaseModel):
    """Archive maintenance/provenance snapshot embedded in publication manifests."""

    truth_source: str = "live"
    derived_models: dict[str, DerivedModelPublicationSummary] = Field(default_factory=dict)


class ArchivePublicationSummary(BaseModel):
    """Archive-scale summary embedded in publication manifests."""

    total_conversations: int
    total_messages: int
    provider_count: int
    provider_counts: dict[str, int] = Field(default_factory=dict)
    provider_messages: dict[str, int] = Field(default_factory=dict)


class SiteOutputSummary(BaseModel):
    """Typed summary of what a site build materialized."""

    root_index_pages: int
    provider_index_pages: int
    dashboard_pages: int
    total_index_pages: int
    total_conversation_pages: int
    rendered_conversation_pages: int
    reused_conversation_pages: int
    failed_conversation_pages: int
    search_documents: int
    search_enabled: bool
    search_provider: str | None = None
    search_status: str
    incremental: bool


class SitePublicationManifest(BaseModel):
    """Typed persisted manifest for a static site build."""

    schema_version: int = 1
    publication_id: str
    publication_kind: str = "site"
    generated_at: str
    output_dir: str
    duration_ms: int
    config: dict[str, Any]
    archive: ArchivePublicationSummary
    outputs: SiteOutputSummary
    latest_run: PublicationRunSummary | None = None
    artifact_proof: ArtifactProofSummary | None = None
    semantic_proof: SemanticProofSuiteSummary | None = None
    maintenance: ArchiveMaintenanceSummary | None = None
    artifacts: OutputManifest
