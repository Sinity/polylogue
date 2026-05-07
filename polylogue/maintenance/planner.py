"""Backfill planner: typed operation model, preview, and execute.

Provides a unified model for maintenance backfill operations — what is
stale, how many rows are affected, estimated time, and execution results.
Uses existing repair.py infrastructure for the actual work.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum

from polylogue.config import Config
from polylogue.core.json import JSONDocument, json_document
from polylogue.logging import get_logger
from polylogue.maintenance.targets import (
    CLEANUP_TARGETS,
    SAFE_REPAIR_TARGETS,
    build_maintenance_target_catalog,
)

logger = get_logger(__name__)


class BackfillKind(str, Enum):
    """What kind of maintenance operation this is."""

    REBUILD = "rebuild"
    REINDEX = "reindex"
    RESET = "reset"
    BACKFILL = "backfill"


class BackfillStatus(str, Enum):
    """Lifecycle status of a backfill operation."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class BackfillOperation:
    """Typed model for a maintenance backfill operation.

    Describes what will be (or was) rebuilt, how many rows are affected,
    progress, and per-target results.
    """

    operation_id: str
    kind: BackfillKind
    targets: tuple[str, ...]
    status: BackfillStatus = BackfillStatus.PENDING
    progress: float = 0.0
    started_at: str | None = None
    completed_at: str | None = None
    error: str | None = None
    affected_rows: int = 0
    estimated_time_s: float = 0.0
    results: list[JSONDocument] = field(default_factory=list)

    def to_dict(self) -> JSONDocument:
        return json_document(
            {
                "operation_id": self.operation_id,
                "kind": self.kind.value,
                "targets": list(self.targets),
                "status": self.status.value,
                "progress": self.progress,
                "started_at": self.started_at,
                "completed_at": self.completed_at,
                "error": self.error,
                "affected_rows": self.affected_rows,
                "estimated_time_s": self.estimated_time_s,
                "results": self.results,
            }
        )


def preview_backfill(
    config: Config,
    targets: tuple[str, ...],
) -> BackfillOperation:
    """Preview what would be rebuilt for the given targets. Read-only.

    Uses existing DerivedModelStatus queries and archive debt collection
    to report how many rows are affected and an estimated completion time.
    No mutations are performed.
    """
    from polylogue.storage.repair import (
        collect_archive_debt_statuses_sync,
        preview_counts_from_archive_debt,
    )
    from polylogue.storage.sqlite.connection import connection_context

    operation_id = str(uuid.uuid4())
    catalog = build_maintenance_target_catalog()
    resolved = catalog.resolve(targets)
    resolved_names = tuple(spec.name for spec in resolved)

    if not resolved_names:
        return BackfillOperation(
            operation_id=operation_id,
            kind=BackfillKind.BACKFILL,
            targets=(),
            status=BackfillStatus.FAILED,
            error="No valid targets resolved from input",
        )

    with connection_context(None) as conn:
        debt_statuses = collect_archive_debt_statuses_sync(
            conn,
            include_expensive=False,
            probe_only=False,
        )

    preview = preview_counts_from_archive_debt(debt_statuses)

    # Compute affected rows and estimated time
    total_rows = 0
    for name in resolved_names:
        total_rows += preview.get(name, 0)

    # Rough estimate: ~50 rows/s for complex rebuilds (session insights,
    # action events), ~500 rows/s for simple repairs (FTS, WAL).
    estimated_time_s = total_rows / 50.0 if total_rows > 0 else 0.0

    # Build per-target preview results from debt statuses
    preview_results: list[JSONDocument] = []
    for name in resolved_names:
        status = debt_statuses.get(name)
        if status is not None:
            preview_results.append(status.to_dict())

    return BackfillOperation(
        operation_id=operation_id,
        kind=BackfillKind.BACKFILL,
        targets=resolved_names,
        status=BackfillStatus.PENDING,
        affected_rows=total_rows,
        estimated_time_s=estimated_time_s,
        results=preview_results,
    )


def execute_backfill(
    config: Config,
    targets: tuple[str, ...],
    *,
    dry_run: bool = False,
) -> BackfillOperation:
    """Execute (or dry-run) a backfill for the given targets.

    Dispatches to existing repair.py infrastructure via
    run_selected_maintenance. Progress is reported via structured logging.
    """
    from polylogue.storage.repair import (
        collect_archive_debt_statuses_sync,
        preview_counts_from_archive_debt,
        run_selected_maintenance,
    )
    from polylogue.storage.sqlite.connection import connection_context

    operation_id = str(uuid.uuid4())
    catalog = build_maintenance_target_catalog()
    resolved = catalog.resolve(targets)
    resolved_names = tuple(spec.name for spec in resolved)

    if not resolved_names:
        return BackfillOperation(
            operation_id=operation_id,
            kind=BackfillKind.BACKFILL,
            targets=(),
            status=BackfillStatus.FAILED,
            error="No valid targets resolved from input",
        )

    logger.info(
        "backfill_starting",
        operation_id=operation_id,
        targets=resolved_names,
        dry_run=dry_run,
    )

    # Collect preview counts for dispatch
    with connection_context(None) as conn:
        debt_statuses = collect_archive_debt_statuses_sync(
            conn,
            include_expensive=False,
            probe_only=False,
        )
    preview = preview_counts_from_archive_debt(debt_statuses)

    started_at = datetime.now(timezone.utc).isoformat()

    try:
        repair_targets = tuple(n for n in resolved_names if n in SAFE_REPAIR_TARGETS)
        cleanup_targets = tuple(n for n in resolved_names if n in CLEANUP_TARGETS)

        repair_results = run_selected_maintenance(
            config,
            repair=bool(repair_targets),
            cleanup=bool(cleanup_targets),
            dry_run=dry_run,
            preview_counts=preview,
            targets=resolved_names,
        )

        completed_at = datetime.now(timezone.utc).isoformat()
        all_success = all(r.success for r in repair_results)
        total_repaired = sum(r.repaired_count for r in repair_results)

        logger.info(
            "backfill_completed",
            operation_id=operation_id,
            targets=resolved_names,
            dry_run=dry_run,
            repaired_count=total_repaired,
            success=all_success,
        )

        return BackfillOperation(
            operation_id=operation_id,
            kind=BackfillKind.BACKFILL,
            targets=resolved_names,
            status=BackfillStatus.COMPLETED if all_success else BackfillStatus.FAILED,
            progress=1.0,
            started_at=started_at,
            completed_at=completed_at,
            affected_rows=total_repaired,
            estimated_time_s=0.0,
            results=[r.to_dict() for r in repair_results],
        )
    except Exception as exc:
        logger.exception(
            "backfill_failed",
            operation_id=operation_id,
            targets=resolved_names,
            error=str(exc),
        )
        return BackfillOperation(
            operation_id=operation_id,
            kind=BackfillKind.BACKFILL,
            targets=resolved_names,
            status=BackfillStatus.FAILED,
            progress=0.0,
            started_at=started_at,
            error=f"Backfill failed: {exc}",
        )


__all__ = [
    "BackfillKind",
    "BackfillOperation",
    "BackfillStatus",
    "execute_backfill",
    "preview_backfill",
]
