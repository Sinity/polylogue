"""Shared archive-product contract base models."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

ARCHIVE_PRODUCT_CONTRACT_VERSION = 4


class ArchiveProductModel(BaseModel):
    """Shared base for public archive data product payloads."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    def to_json(self, *, exclude_none: bool = False) -> str:
        return self.model_dump_json(indent=2, exclude_none=exclude_none)


class ArchiveProductProvenance(ArchiveProductModel):
    materializer_version: int
    materialized_at: str
    source_updated_at: str | None = None
    source_sort_key: float | None = None


class ArchiveInferenceProvenance(ArchiveProductProvenance):
    inference_version: int
    inference_family: str


class ArchiveEnrichmentProvenance(ArchiveProductProvenance):
    enrichment_version: int
    enrichment_family: str


__all__ = [
    "ARCHIVE_PRODUCT_CONTRACT_VERSION",
    "ArchiveEnrichmentProvenance",
    "ArchiveInferenceProvenance",
    "ArchiveProductModel",
    "ArchiveProductProvenance",
]
