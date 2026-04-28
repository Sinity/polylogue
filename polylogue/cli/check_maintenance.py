"""Maintenance-target selection and preview-count helpers for check workflow."""

from __future__ import annotations

from typing import TYPE_CHECKING

from polylogue.cli.check_models import VacuumResult
from polylogue.cli.shared.types import AppEnv
from polylogue.maintenance.targets import MaintenanceTargetMode, build_maintenance_target_catalog
from polylogue.readiness import ReadinessReport
from polylogue.storage.repair import RepairResult, preview_counts_from_archive_debt

if TYPE_CHECKING:
    from polylogue.cli.check_workflow import CheckCommandOptions


def build_preview_counts(report: ReadinessReport) -> dict[str, int]:
    return preview_counts_from_archive_debt(report.archive_debt)


def resolve_selected_maintenance_targets(
    options: CheckCommandOptions,
) -> tuple[str, ...]:
    if options.maintenance_targets:
        return tuple(options.maintenance_targets)
    catalog = build_maintenance_target_catalog()
    targets: list[str] = []
    if options.repair:
        targets.extend(catalog.names_for_mode(MaintenanceTargetMode.REPAIR))
    if options.cleanup:
        targets.extend(catalog.names_for_mode(MaintenanceTargetMode.CLEANUP))
    return tuple(targets)


def persist_maintenance_run(
    env: AppEnv,
    *,
    report: ReadinessReport,
    options: CheckCommandOptions,
    targets: tuple[str, ...],
    maintenance_results: list[RepairResult],
    vacuum_result: VacuumResult | None,
    preview_counts: dict[str, int] | None,
) -> None:
    """No-op — maintenance run recording removed."""


__all__ = [
    "build_preview_counts",
    "persist_maintenance_run",
    "resolve_selected_maintenance_targets",
]
