"""Derived model status type used by readiness and derived-status modules."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


class MaintenanceCategory(str, Enum):
    DERIVED_REPAIR = "derived_repair"
    ARCHIVE_CLEANUP = "archive_cleanup"
    DATABASE_MAINTENANCE = "database_maintenance"


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


__all__ = ["DerivedModelStatus", "MaintenanceCategory"]
