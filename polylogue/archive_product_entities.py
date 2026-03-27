"""Concrete archive-product entity models."""

from __future__ import annotations

from polylogue.archive_product_base import ARCHIVE_PRODUCT_CONTRACT_VERSION, ArchiveProductModel
from polylogue.archive_product_session_entities import (
    SessionEnrichmentProduct,
    SessionPhaseProduct,
    SessionProfileProduct,
    SessionWorkEventProduct,
    WorkThreadProduct,
)
from polylogue.archive_product_summary_entities import (
    DaySessionSummaryProduct,
    ProviderAnalyticsProduct,
    SessionTagRollupProduct,
    WeekSessionSummaryProduct,
)


class ArchiveDebtProduct(ArchiveProductModel):
    contract_version: int = ARCHIVE_PRODUCT_CONTRACT_VERSION
    product_kind: str = "archive_debt"
    debt_name: str
    category: str
    maintenance_target: str
    destructive: bool
    issue_count: int
    healthy: bool
    detail: str

    @classmethod
    def from_status(cls, status: object) -> ArchiveDebtProduct:
        return cls(
            debt_name=status.name,
            category=status.category if isinstance(status.category, str) else status.category.value,
            maintenance_target=status.maintenance_target,
            destructive=status.destructive,
            issue_count=status.issue_count,
            healthy=status.healthy,
            detail=status.detail,
        )


__all__ = [
    "ArchiveDebtProduct",
    "DaySessionSummaryProduct",
    "ProviderAnalyticsProduct",
    "SessionEnrichmentProduct",
    "SessionPhaseProduct",
    "SessionProfileProduct",
    "SessionTagRollupProduct",
    "SessionWorkEventProduct",
    "WeekSessionSummaryProduct",
    "WorkThreadProduct",
]
