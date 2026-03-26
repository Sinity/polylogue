"""Maintenance-target selection and persistence helpers for check workflow."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from polylogue.storage.archive_debt import preview_counts_from_archive_debt
from polylogue.storage.backends.connection import _clear_connection_cache
from polylogue.storage.store import MaintenanceRunRecord
from polylogue.sync_bridge import run_coroutine_sync


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


def build_maintenance_manifest(
    *,
    report: Any,
    targets: tuple[str, ...],
    maintenance_results: list[Any],
    vacuum_result: dict[str, Any] | None,
    preview_counts: dict[str, int] | None,
) -> dict[str, Any]:
    return {
        "report_summary": dict(report.summary),
        "report_provenance": report.provenance.to_dict(),
        "derived_models": {
            name: status.to_dict()
            for name, status in sorted((report.derived_models or {}).items())
        },
        "preview_counts": dict(preview_counts or {}),
        "targets": list(targets),
        "results": [result.to_dict() for result in maintenance_results],
        "vacuum": vacuum_result,
    }


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
    vacuum_ok = vacuum_result is None or bool(vacuum_result.get("ok", False))
    _clear_connection_cache()
    record = MaintenanceRunRecord(
        maintenance_run_id=f"maint-{uuid4().hex[:16]}",
        executed_at=datetime.now(timezone.utc).isoformat(),
        mode="preview" if options.preview else "apply",
        preview=options.preview,
        repair_selected=options.repair,
        cleanup_selected=options.cleanup,
        vacuum_requested=options.vacuum,
        target_names=targets,
        success=all(result_item.success for result_item in maintenance_results) and vacuum_ok,
        manifest=build_maintenance_manifest(
            report=report,
            targets=targets,
            maintenance_results=maintenance_results,
            vacuum_result=vacuum_result,
            preview_counts=preview_counts,
        ),
    )
    run_coroutine_sync(env.backend.record_maintenance_run(record))


__all__ = [
    "build_maintenance_manifest",
    "build_preview_counts",
    "persist_maintenance_run",
    "resolve_selected_maintenance_targets",
]
