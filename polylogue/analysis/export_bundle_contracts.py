"""Closed version 2 insight export bundle contracts."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, field_validator

from polylogue.analysis.archive_models import ARCHIVE_INSIGHT_CONTRACT_VERSION
from polylogue.core.errors import PolylogueError
from polylogue.core.sources import source_name_to_origin
from polylogue.surfaces.outcome import OutcomeEnvelope

InsightExportFormat = Literal["jsonl"]
INSIGHT_EXPORT_BUNDLE_VERSION = 2


class InsightExportBundleError(PolylogueError):
    """Raised when an insight export bundle cannot be written."""


class InsightExportModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    @field_validator("output_path", "manifest_path", "coverage_path", mode="before", check_fields=False)
    @classmethod
    def path_from_json(cls, value: object) -> object:
        return Path(value) if isinstance(value, str) else value


class InsightExportBundleRequest(InsightExportModel):
    output_path: Path
    insights: tuple[str, ...] = ()
    origin: str | None = None
    since: str | None = None
    until: str | None = None
    output_format: InsightExportFormat = "jsonl"
    overwrite: bool = False
    include_readme: bool = True

    @field_validator("origin", mode="before")
    @classmethod
    def normalize_origin(cls, value: object) -> str | None:
        return None if value is None else source_name_to_origin(value)


class InsightExportFileSummary(InsightExportModel):
    insight_name: str
    file: str
    schema_file: str
    row_count: int = 0
    withheld_reason: str | None = None
    warnings: tuple[str, ...] = ()
    errors: tuple[str, ...] = ()


class InsightExportBundleManifest(InsightExportModel):
    bundle_version: int = INSIGHT_EXPORT_BUNDLE_VERSION
    insight_contract_version: int = ARCHIVE_INSIGHT_CONTRACT_VERSION
    generated_at: str
    polylogue_version: str
    git_revision: str | None = None
    git_dirty: bool = False
    archive_root: str
    database_path: str
    output_format: InsightExportFormat = "jsonl"
    query: dict[str, str | tuple[str, ...] | None]
    insights: tuple[InsightExportFileSummary, ...] = ()
    warnings: tuple[str, ...] = ()


class InsightExportBundleResult(InsightExportModel):
    output_path: Path
    manifest_path: Path
    coverage_path: Path
    manifest: InsightExportBundleManifest
    outcome: OutcomeEnvelope
