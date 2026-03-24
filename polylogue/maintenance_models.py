"""Typed runtime-governance and maintenance metadata."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any


class TruthSource(StrEnum):
    LIVE = "live"
    CACHE = "cache"


class MaintenanceCategory(StrEnum):
    DERIVED_REPAIR = "derived_repair"
    ARCHIVE_CLEANUP = "archive_cleanup"
    DATABASE_MAINTENANCE = "database_maintenance"


@dataclass(frozen=True)
class ReportProvenance:
    source: TruthSource = TruthSource.LIVE
    cache_age_seconds: int | None = None
    cache_ttl_seconds: int | None = None
    cache_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source.value,
            "cache_age_seconds": self.cache_age_seconds,
            "cache_ttl_seconds": self.cache_ttl_seconds,
            "cache_path": self.cache_path,
        }


@dataclass(frozen=True)
class DerivedModelStatus:
    name: str
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

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "ready": self.ready,
            "detail": self.detail,
            "source_documents": self.source_documents,
            "materialized_documents": self.materialized_documents,
            "source_rows": self.source_rows,
            "materialized_rows": self.materialized_rows,
            "pending_documents": self.pending_documents,
            "pending_rows": self.pending_rows,
            "stale_rows": self.stale_rows,
            "orphan_rows": self.orphan_rows,
            "missing_provenance_rows": self.missing_provenance_rows,
            "materializer_version": self.materializer_version,
            "matches_version": self.matches_version,
        }


@dataclass(frozen=True)
class ArchiveDebtStatus:
    name: str
    category: MaintenanceCategory
    destructive: bool
    issue_count: int
    detail: str
    maintenance_target: str

    @property
    def healthy(self) -> bool:
        return self.issue_count == 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "category": self.category.value,
            "destructive": self.destructive,
            "issue_count": self.issue_count,
            "detail": self.detail,
            "maintenance_target": self.maintenance_target,
            "healthy": self.healthy,
        }


__all__ = [
    "ArchiveDebtStatus",
    "DerivedModelStatus",
    "MaintenanceCategory",
    "ReportProvenance",
    "TruthSource",
]
