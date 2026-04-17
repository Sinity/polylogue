"""Maintenance-target selection and preview-count helpers for check workflow."""

from __future__ import annotations

from typing import Any

from polylogue.maintenance_targets import MaintenanceTargetMode, build_maintenance_target_catalog
from polylogue.storage.repair import preview_counts_from_archive_debt


def build_preview_counts(report: Any) -> dict[str, int]:
    return preview_counts_from_archive_debt(getattr(report, "archive_debt", {}) or {})


def resolve_selected_maintenance_targets(
    options: Any,
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
    env: Any,
    *,
    report: Any,
    options: Any,
    targets: tuple[str, ...],
    maintenance_results: list[Any],
    vacuum_result: dict[str, Any] | None,
    preview_counts: dict[str, int] | None,
) -> None:
    """No-op — maintenance run recording removed."""


__all__ = [
    "build_preview_counts",
    "persist_maintenance_run",
    "resolve_selected_maintenance_targets",
]
