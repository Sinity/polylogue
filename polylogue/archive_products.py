"""Versioned archive data product contracts."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from polylogue.storage.store import (
    MaintenanceRunRecord,
    SessionProfileRecord,
    SessionWorkEventRecord,
    WorkThreadRecord,
)

ARCHIVE_PRODUCT_CONTRACT_VERSION = 1


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


class SessionProfileProduct(ArchiveProductModel):
    contract_version: int = ARCHIVE_PRODUCT_CONTRACT_VERSION
    product_kind: str = "session_profile"
    conversation_id: str
    provider_name: str
    title: str | None = None
    primary_work_kind: str | None = None
    provenance: ArchiveProductProvenance
    profile: dict[str, Any]

    @classmethod
    def from_record(cls, record: SessionProfileRecord) -> SessionProfileProduct:
        return cls(
            conversation_id=record.conversation_id,
            provider_name=record.provider_name,
            title=record.title,
            primary_work_kind=record.primary_work_kind,
            provenance=ArchiveProductProvenance(
                materializer_version=record.materializer_version,
                materialized_at=record.materialized_at,
                source_updated_at=record.source_updated_at,
                source_sort_key=record.source_sort_key,
            ),
            profile=dict(record.payload),
        )


class SessionWorkEventProduct(ArchiveProductModel):
    contract_version: int = ARCHIVE_PRODUCT_CONTRACT_VERSION
    product_kind: str = "session_work_event"
    event_id: str
    conversation_id: str
    provider_name: str
    event_index: int
    kind: str
    provenance: ArchiveProductProvenance
    event: dict[str, Any]

    @classmethod
    def from_record(cls, record: SessionWorkEventRecord) -> SessionWorkEventProduct:
        return cls(
            event_id=record.event_id,
            conversation_id=record.conversation_id,
            provider_name=record.provider_name,
            event_index=record.event_index,
            kind=record.kind,
            provenance=ArchiveProductProvenance(
                materializer_version=record.materializer_version,
                materialized_at=record.materialized_at,
                source_updated_at=record.source_updated_at,
                source_sort_key=record.source_sort_key,
            ),
            event=dict(record.payload),
        )


class WorkThreadProduct(ArchiveProductModel):
    contract_version: int = ARCHIVE_PRODUCT_CONTRACT_VERSION
    product_kind: str = "work_thread"
    thread_id: str
    root_id: str
    dominant_project: str | None = None
    provenance: ArchiveProductProvenance
    thread: dict[str, Any]

    @classmethod
    def from_record(cls, record: WorkThreadRecord) -> WorkThreadProduct:
        return cls(
            thread_id=record.thread_id,
            root_id=record.root_id,
            dominant_project=record.dominant_project,
            provenance=ArchiveProductProvenance(
                materializer_version=record.materializer_version,
                materialized_at=record.materialized_at,
                source_updated_at=record.end_time or record.start_time,
                source_sort_key=None,
            ),
            thread=dict(record.payload),
        )


class SessionTagRollupProduct(ArchiveProductModel):
    contract_version: int = ARCHIVE_PRODUCT_CONTRACT_VERSION
    product_kind: str = "session_tag_rollup"
    tag: str
    conversation_count: int
    explicit_count: int
    auto_count: int
    provider_breakdown: dict[str, int]
    project_breakdown: dict[str, int]
    provenance: ArchiveProductProvenance


class DaySessionSummaryProduct(ArchiveProductModel):
    contract_version: int = ARCHIVE_PRODUCT_CONTRACT_VERSION
    product_kind: str = "day_session_summary"
    date: str
    provenance: ArchiveProductProvenance
    summary: dict[str, Any]


class WeekSessionSummaryProduct(ArchiveProductModel):
    contract_version: int = ARCHIVE_PRODUCT_CONTRACT_VERSION
    product_kind: str = "week_session_summary"
    iso_week: str
    provenance: ArchiveProductProvenance
    summary: dict[str, Any]


class MaintenanceRunProduct(ArchiveProductModel):
    contract_version: int = ARCHIVE_PRODUCT_CONTRACT_VERSION
    product_kind: str = "maintenance_run"
    maintenance_run_id: str
    executed_at: str
    mode: str
    preview: bool
    repair_selected: bool
    cleanup_selected: bool
    vacuum_requested: bool
    target_names: tuple[str, ...] = ()
    success: bool
    schema_version: int
    manifest: dict[str, Any]

    @classmethod
    def from_record(cls, record: MaintenanceRunRecord) -> MaintenanceRunProduct:
        return cls(
            maintenance_run_id=record.maintenance_run_id,
            executed_at=record.executed_at,
            mode=record.mode,
            preview=record.preview,
            repair_selected=record.repair_selected,
            cleanup_selected=record.cleanup_selected,
            vacuum_requested=record.vacuum_requested,
            target_names=record.target_names,
            success=record.success,
            schema_version=record.schema_version,
            manifest=dict(record.manifest),
        )


class SessionProfileProductQuery(ArchiveProductModel):
    provider: str | None = None
    since: str | None = None
    until: str | None = None
    limit: int | None = 50
    offset: int = 0
    query: str | None = None


class SessionWorkEventProductQuery(ArchiveProductModel):
    provider: str | None = None
    since: str | None = None
    until: str | None = None
    kind: str | None = None
    limit: int | None = 50
    offset: int = 0
    query: str | None = None


class WorkThreadProductQuery(ArchiveProductModel):
    since: str | None = None
    until: str | None = None
    limit: int | None = 50
    offset: int = 0
    query: str | None = None


class MaintenanceRunProductQuery(ArchiveProductModel):
    limit: int = Field(default=20, ge=1)


class SessionTagRollupQuery(ArchiveProductModel):
    provider: str | None = None
    since: str | None = None
    until: str | None = None
    limit: int | None = 100
    offset: int = 0
    query: str | None = None


class DaySessionSummaryProductQuery(ArchiveProductModel):
    provider: str | None = None
    since: str | None = None
    until: str | None = None
    limit: int | None = 90
    offset: int = 0


class WeekSessionSummaryProductQuery(ArchiveProductModel):
    provider: str | None = None
    since: str | None = None
    until: str | None = None
    limit: int | None = 52
    offset: int = 0


__all__ = [
    "ARCHIVE_PRODUCT_CONTRACT_VERSION",
    "ArchiveProductModel",
    "ArchiveProductProvenance",
    "DaySessionSummaryProduct",
    "DaySessionSummaryProductQuery",
    "MaintenanceRunProduct",
    "MaintenanceRunProductQuery",
    "SessionTagRollupProduct",
    "SessionTagRollupQuery",
    "SessionProfileProduct",
    "SessionProfileProductQuery",
    "SessionWorkEventProduct",
    "SessionWorkEventProductQuery",
    "WeekSessionSummaryProduct",
    "WeekSessionSummaryProductQuery",
    "WorkThreadProduct",
    "WorkThreadProductQuery",
]
