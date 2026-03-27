"""Governance and debt archive product entities."""

from __future__ import annotations

from typing import Any

from polylogue.archive_product_base import ARCHIVE_PRODUCT_CONTRACT_VERSION, ArchiveProductModel
from polylogue.archive_product_payloads import ArchiveDebtTargetLineage
from polylogue.maintenance_models import ArchiveDebtStatus
from polylogue.storage.store import MaintenanceRunRecord


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
    governance_stage: str
    lineage: ArchiveDebtTargetLineage | None = None

    @classmethod
    def from_status(
        cls,
        status: ArchiveDebtStatus,
        *,
        governance_stage: str,
        lineage: ArchiveDebtTargetLineage | None = None,
    ) -> ArchiveDebtProduct:
        return cls(
            debt_name=status.name,
            category=status.category.value,
            maintenance_target=status.maintenance_target,
            destructive=status.destructive,
            issue_count=status.issue_count,
            healthy=status.healthy,
            detail=status.detail,
            governance_stage=governance_stage,
            lineage=lineage,
        )
