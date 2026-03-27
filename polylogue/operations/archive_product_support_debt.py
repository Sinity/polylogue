"""Debt lineage helpers and debt-product mixin."""

from __future__ import annotations

from collections.abc import Iterable

from polylogue.archive_products import (
    ArchiveDebtProduct,
    ArchiveDebtProductQuery,
    ArchiveDebtTargetLineage,
)
from polylogue.storage.archive_debt import collect_archive_debt_statuses_sync
from polylogue.storage.backends.connection import connection_context
from polylogue.storage.store import MaintenanceRunRecord


def maintenance_issue_count(record: MaintenanceRunRecord, target_name: str) -> int | None:
    preview_counts = record.manifest.get("preview_counts")
    if isinstance(preview_counts, dict) and target_name in preview_counts:
        try:
            return int(preview_counts[target_name] or 0)
        except (TypeError, ValueError):
            return None
    for item in record.manifest.get("results", []):
        if not isinstance(item, dict):
            continue
        if item.get("name") != target_name:
            continue
        try:
            return int(item.get("repaired_count") or 0)
        except (TypeError, ValueError):
            return None
    return None


def target_lineage(records: Iterable[MaintenanceRunRecord]) -> dict[str, ArchiveDebtTargetLineage]:
    records_by_target: dict[str, list[MaintenanceRunRecord]] = {}
    for record in records:
        for target_name in record.target_names:
            records_by_target.setdefault(target_name, []).append(record)

    lineage_by_target: dict[str, ArchiveDebtTargetLineage] = {}
    for target_name, target_records in records_by_target.items():
        ordered = sorted(
            target_records,
            key=lambda record: (record.executed_at, record.maintenance_run_id),
            reverse=True,
        )
        latest_run = ordered[0]
        latest_preview = next((record for record in ordered if record.preview), None)
        latest_apply = next((record for record in ordered if not record.preview), None)
        latest_successful_apply = next(
            (record for record in ordered if not record.preview and record.success),
            None,
        )
        validation_anchor = latest_successful_apply or latest_apply
        validation_candidates = (
            [
                record
                for record in ordered
                if record.preview and record.executed_at > validation_anchor.executed_at
            ]
            if validation_anchor is not None
            else []
        )
        latest_validation = validation_candidates[0] if validation_candidates else None
        latest_validation_issue_count = (
            maintenance_issue_count(latest_validation, target_name)
            if latest_validation is not None
            else None
        )
        latest_successful_validation = next(
            (
                record
                for record in validation_candidates
                if record.success and (maintenance_issue_count(record, target_name) or 0) == 0
            ),
            None,
        )
        latest_regressed = next(
            (
                record
                for record in validation_candidates
                if (maintenance_issue_count(record, target_name) or 0) > 0
            ),
            None,
        )
        lineage_by_target[target_name] = ArchiveDebtTargetLineage(
            latest_run_at=latest_run.executed_at,
            latest_mode=latest_run.mode,
            latest_preview_at=latest_preview.executed_at if latest_preview is not None else None,
            latest_preview_issue_count=(
                maintenance_issue_count(latest_preview, target_name)
                if latest_preview is not None
                else None
            ),
            latest_apply_at=latest_apply.executed_at if latest_apply is not None else None,
            latest_successful_apply_at=(
                latest_successful_apply.executed_at if latest_successful_apply is not None else None
            ),
            latest_validation_at=(
                latest_validation.executed_at if latest_validation is not None else None
            ),
            latest_validation_issue_count=latest_validation_issue_count,
            latest_successful_validation_at=(
                latest_successful_validation.executed_at
                if latest_successful_validation is not None
                else None
            ),
            latest_regressed_at=(
                latest_regressed.executed_at if latest_regressed is not None else None
            ),
        )
    return lineage_by_target


def lineage_governance_stage(*, issue_count: int, lineage: ArchiveDebtTargetLineage | None) -> str:
    if issue_count <= 0:
        if lineage and (lineage.latest_successful_validation_at or lineage.latest_successful_apply_at):
            return "validated"
        return "healthy"
    if lineage is None:
        return "unreviewed"
    if lineage.latest_regressed_at or lineage.latest_successful_validation_at:
        return "regressed"
    if lineage.latest_validation_at:
        return "previewed"
    if lineage.latest_successful_apply_at:
        return "applied"
    if lineage.latest_preview_at:
        return "previewed"
    return "unreviewed"


class ArchiveProductDebtMixin:
    async def list_archive_debt_products(
        self,
        query: ArchiveDebtProductQuery | None = None,
    ) -> list[ArchiveDebtProduct]:
        request = query or ArchiveDebtProductQuery()
        maintenance_records = await self.repository.list_maintenance_runs(limit=100)
        lineage_by_target = target_lineage(maintenance_records)
        with connection_context(self.config.db_path) as conn:
            statuses = collect_archive_debt_statuses_sync(conn)
        products = [
            ArchiveDebtProduct.from_status(
                status,
                governance_stage=lineage_governance_stage(
                    issue_count=status.issue_count,
                    lineage=lineage_by_target.get(status.maintenance_target),
                ),
                lineage=lineage_by_target.get(status.maintenance_target),
            )
            for status in statuses.values()
        ]
        products.sort(key=lambda product: (product.category, product.debt_name))
        if request.category:
            products = [product for product in products if product.category == request.category]
        if request.only_actionable:
            products = [product for product in products if not product.healthy]
        if request.offset:
            products = products[request.offset :]
        if request.limit is not None:
            products = products[: request.limit]
        return products


__all__ = [
    "ArchiveProductDebtMixin",
    "lineage_governance_stage",
    "maintenance_issue_count",
    "target_lineage",
]
