"""Maintenance-target selection and preview-count helpers for check workflow."""

from __future__ import annotations

from typing import Any

from polylogue.storage.repair import preview_counts_from_archive_debt


def build_preview_counts(report: Any) -> dict[str, int]:
    return preview_counts_from_archive_debt(getattr(report, "archive_debt", {}) or {})


def resolve_selected_maintenance_targets(
    options,
    *,
    safe_repair_targets: tuple[str, ...],
    cleanup_targets: tuple[str, ...],
) -> tuple[str, ...]:
    if options.maintenance_targets:
        return tuple(options.maintenance_targets)
    targets: list[str] = []
    if options.repair:
        targets.extend(safe_repair_targets)
    if options.cleanup:
        targets.extend(cleanup_targets)
    return tuple(targets)


def persist_maintenance_run(
    env,
    *,
    report: Any,
    options,
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
