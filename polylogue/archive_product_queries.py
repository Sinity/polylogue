"""Archive-product query models."""

from __future__ import annotations

from pydantic import Field

from polylogue.archive_product_base import ArchiveProductModel


class SessionProfileProductQuery(ArchiveProductModel):
    provider: str | None = None
    since: str | None = None
    until: str | None = None
    first_message_since: str | None = None
    first_message_until: str | None = None
    session_date_since: str | None = None
    session_date_until: str | None = None
    tier: str = "merged"
    limit: int | None = 50
    offset: int = 0
    query: str | None = None


class SessionEnrichmentProductQuery(ArchiveProductModel):
    provider: str | None = None
    since: str | None = None
    until: str | None = None
    first_message_since: str | None = None
    first_message_until: str | None = None
    session_date_since: str | None = None
    session_date_until: str | None = None
    refined_work_kind: str | None = None
    limit: int | None = 50
    offset: int = 0
    query: str | None = None


class SessionWorkEventProductQuery(ArchiveProductModel):
    conversation_id: str | None = None
    provider: str | None = None
    since: str | None = None
    until: str | None = None
    kind: str | None = None
    limit: int | None = 50
    offset: int = 0
    query: str | None = None


class SessionPhaseProductQuery(ArchiveProductModel):
    conversation_id: str | None = None
    provider: str | None = None
    since: str | None = None
    until: str | None = None
    kind: str | None = None
    limit: int | None = 50
    offset: int = 0


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


class ProviderAnalyticsProductQuery(ArchiveProductModel):
    provider: str | None = None
    limit: int | None = None
    offset: int = 0


class ArchiveDebtProductQuery(ArchiveProductModel):
    category: str | None = None
    only_actionable: bool = False
    limit: int | None = None
    offset: int = 0


__all__ = [
    "ArchiveDebtProductQuery",
    "DaySessionSummaryProductQuery",
    "MaintenanceRunProductQuery",
    "ProviderAnalyticsProductQuery",
    "SessionEnrichmentProductQuery",
    "SessionPhaseProductQuery",
    "SessionProfileProductQuery",
    "SessionTagRollupQuery",
    "SessionWorkEventProductQuery",
    "WeekSessionSummaryProductQuery",
    "WorkThreadProductQuery",
]
