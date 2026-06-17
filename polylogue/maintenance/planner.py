"""Typed maintenance planner contract.

The planner owns the typed contract used by every maintenance / backfill
surface (CLI ``maintenance``, daemon HTTP, MCP, Python API). A
:class:`BackfillOperation` carries everything a caller needs to reason
about *one* maintenance attempt without having to peek into the repair
internals:

* identity — ``operation_id`` plus a typed ``kind``;
* scope — a typed :class:`MaintenanceScope` (target names + optional
  filter) instead of a bare tuple, so future surfaces can attach
  session-id filters, time windows, source roots, etc. without
  changing the signature again;
* reason — a typed :class:`~polylogue.maintenance.invalidation.InvalidationReason`
  recording *why* the planner scheduled the work;
* progress — ``status``, ``progress``, ``started_at`` / ``completed_at``,
  ``affected_rows``, ``estimated_time_s``;
* resumability — an opaque ``resume_cursor`` string that the executor
  can hand back to itself on the next attempt;
* failures — a :class:`BoundedFailureSamples` envelope with at most
  ``MAX_FAILURE_SAMPLES`` entries plus a ``truncated`` flag so the
  unbounded raw failure list never leaks through;
* metrics — a free-form numeric dict (counters and timings) reported by
  the executor.

The shape is intentionally additive over the earlier scaffold so that
existing callers (CLI, daemon, MCP, API write surfaces, contract
tests) continue to work unchanged.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import TYPE_CHECKING

from polylogue.config import Config

if TYPE_CHECKING:
    from polylogue.storage.repair import ArchiveDebtStatus
from polylogue.core.json import JSONDocument, json_document
from polylogue.logging import get_logger
from polylogue.maintenance.invalidation import InvalidationReason
from polylogue.maintenance.scope import MaintenanceScopeFilter
from polylogue.maintenance.targets import (
    CLEANUP_TARGETS,
    SAFE_REPAIR_TARGETS,
    build_maintenance_target_catalog,
)

logger = get_logger(__name__)


#: Hard cap on per-operation failure samples. Real executors must
#: truncate before populating :class:`BoundedFailureSamples`; the
#: envelope refuses to grow beyond this.
MAX_FAILURE_SAMPLES = 50


def _coerce_float(value: object) -> float | None:
    """Best-effort projection of a JSON value onto ``float`` (or ``None``).

    Used by :meth:`BackfillOperation.from_dict` to rehydrate numeric
    fields from untrusted JSON without scattering bracketed mypy
    suppression directives across the payload-coercion paths.
    """
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def _coerce_int(value: object) -> int | None:
    """Best-effort projection of a JSON value onto ``int`` (or ``None``)."""
    if isinstance(value, bool):
        # bool is a subclass of int but we don't want True / False
        # silently becoming 1 / 0 in user-visible row counts.
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return None
    return None


class BackfillKind(str, Enum):
    """What kind of maintenance operation this is.

    The values map to the typed-operation taxonomy the rest of the
    maintenance cluster expects (see #1144).
    """

    # Typed taxonomy (issue #1144).
    ARCHIVE_SUBSET = "archive-subset"
    DERIVED_REBUILD = "derived-rebuild"
    INDEX_REPAIR = "index-repair"
    SEMANTIC_REMATERIALIZE = "semantic-rematerialize"
    CONFIG_DRIVEN = "config-driven"


class BackfillStatus(str, Enum):
    """Lifecycle status of a backfill operation."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass(frozen=True)
class MaintenanceScope:
    """Typed scope (target ids + typed filter) for a backfill.

    ``targets`` is the resolved canonical target-name tuple; ``filter``
    is a typed :class:`MaintenanceScopeFilter` carrying the named scope
    dimensions agreed across CLI / daemon HTTP / MCP. An empty filter
    means "full scope for the listed targets".
    """

    targets: tuple[str, ...]
    filter: MaintenanceScopeFilter = field(default_factory=MaintenanceScopeFilter)

    def to_dict(self) -> JSONDocument:
        return json_document(
            {
                "targets": list(self.targets),
                "filter": self.filter.to_dict(),
            }
        )

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> MaintenanceScope:
        targets_raw = payload.get("targets", ()) or ()
        if not isinstance(targets_raw, (list, tuple)):
            raise TypeError(f"scope.targets must be a list/tuple, got {type(targets_raw).__name__}")
        targets = tuple(str(t) for t in targets_raw)
        filter_raw = payload.get("filter")
        if filter_raw is None:
            filter_obj = MaintenanceScopeFilter()
        elif isinstance(filter_raw, MaintenanceScopeFilter):
            filter_obj = filter_raw
        elif isinstance(filter_raw, dict):
            filter_obj = MaintenanceScopeFilter.from_dict(filter_raw)
        else:
            raise TypeError(f"scope.filter must be a dict, got {type(filter_raw).__name__}")
        return cls(targets=targets, filter=filter_obj)


@dataclass(frozen=True)
class FailureSample:
    """One bounded, structured failure sample.

    Executors must classify the failure (``kind``) and supply enough
    locator to find the offending row (``locator``) and a short
    human-readable ``message``. The full raw exception payload must not
    leak through this surface.
    """

    kind: str
    locator: str
    message: str

    def to_dict(self) -> JSONDocument:
        return json_document(
            {
                "kind": self.kind,
                "locator": self.locator,
                "message": self.message,
            }
        )


@dataclass(frozen=True)
class BoundedFailureSamples:
    """Bounded list of :class:`FailureSample` with a ``truncated`` flag.

    Use :meth:`from_samples` to construct: it clamps to
    :data:`MAX_FAILURE_SAMPLES` and sets ``truncated`` accordingly.
    """

    samples: tuple[FailureSample, ...] = ()
    truncated: bool = False

    @classmethod
    def from_samples(cls, samples: list[FailureSample] | tuple[FailureSample, ...]) -> BoundedFailureSamples:
        seq = tuple(samples)
        if len(seq) <= MAX_FAILURE_SAMPLES:
            return cls(samples=seq, truncated=False)
        return cls(samples=seq[:MAX_FAILURE_SAMPLES], truncated=True)

    def to_dict(self) -> JSONDocument:
        return json_document(
            {
                "samples": [sample.to_dict() for sample in self.samples],
                "truncated": self.truncated,
            }
        )


@dataclass
class BackfillOperation:
    """Typed model for a maintenance backfill operation.

    Carries identity, scope, reason, status, progress, resume cursor,
    bounded failure samples, and metrics for one maintenance attempt.
    The legacy ``targets`` tuple, ``affected_rows``, ``estimated_time_s``,
    ``results``, and ``error`` fields are preserved so existing
    surfaces continue to work; new code should prefer the typed
    :attr:`scope`, :attr:`reason`, :attr:`resume_cursor`,
    :attr:`failure_samples`, and :attr:`metrics` fields.
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

    # Typed planner contract (issue #1144).
    scope: MaintenanceScope | None = None
    reason: InvalidationReason | None = None
    resume_cursor: str | None = None
    failure_samples: BoundedFailureSamples = field(default_factory=BoundedFailureSamples)
    metrics: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Always keep ``scope.targets`` synchronized with ``targets`` so
        # callers can rely on either field without divergence. If the
        # constructor was called without a scope, derive one from the
        # tuple; if a scope was supplied, trust it as authoritative.
        if self.scope is None:
            object.__setattr__(self, "scope", MaintenanceScope(targets=self.targets))
        elif self.scope.targets != self.targets:
            object.__setattr__(self, "targets", self.scope.targets)

    def to_dict(self) -> JSONDocument:
        scope = self.scope if self.scope is not None else MaintenanceScope(targets=self.targets)
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
                "scope": scope.to_dict(),
                "reason": (self.reason.value if self.reason is not None else None),
                "resume_cursor": self.resume_cursor,
                "failure_samples": self.failure_samples.to_dict(),
                "metrics": dict(self.metrics),
            }
        )

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> BackfillOperation:
        """Reconstruct a :class:`BackfillOperation` from its ``to_dict`` form.

        Used by :mod:`polylogue.maintenance.registry` to rehydrate
        persisted operation snapshots from
        ``<archive_root>/.maintenance-state/*.json``. Unknown enum values
        round-trip through their literal string form so a snapshot
        written by an older or newer build is still readable.
        """

        op_id = str(payload.get("operation_id", ""))
        kind_raw = str(payload.get("kind", BackfillKind.DERIVED_REBUILD.value))
        try:
            kind = BackfillKind(kind_raw)
        except ValueError:
            kind = BackfillKind.DERIVED_REBUILD
        status_raw = str(payload.get("status", BackfillStatus.PENDING.value))
        try:
            status = BackfillStatus(status_raw)
        except ValueError:
            status = BackfillStatus.PENDING
        targets_raw = payload.get("targets") or ()
        if not isinstance(targets_raw, (list, tuple)):
            targets_raw = ()
        targets = tuple(str(t) for t in targets_raw)
        results_raw = payload.get("results") or []
        results: list[JSONDocument] = []
        if isinstance(results_raw, (list, tuple)):
            for entry in results_raw:
                if isinstance(entry, dict):
                    results.append(json_document(entry))
        scope_raw = payload.get("scope")
        scope_obj: MaintenanceScope | None
        if isinstance(scope_raw, dict):
            try:
                scope_obj = MaintenanceScope.from_dict(scope_raw)
            except (TypeError, ValueError):
                scope_obj = None
        else:
            scope_obj = None
        reason_raw = payload.get("reason")
        reason_obj: InvalidationReason | None
        if isinstance(reason_raw, str):
            try:
                reason_obj = InvalidationReason(reason_raw)
            except ValueError:
                reason_obj = None
        else:
            reason_obj = None
        resume_cursor_raw = payload.get("resume_cursor")
        resume_cursor = resume_cursor_raw if isinstance(resume_cursor_raw, str) else None
        failure_raw = payload.get("failure_samples")
        if isinstance(failure_raw, dict):
            samples_raw = failure_raw.get("samples") or ()
            truncated = bool(failure_raw.get("truncated", False))
            samples: list[FailureSample] = []
            if isinstance(samples_raw, (list, tuple)):
                for entry in samples_raw:
                    if isinstance(entry, dict):
                        samples.append(
                            FailureSample(
                                kind=str(entry.get("kind", "")),
                                locator=str(entry.get("locator", "")),
                                message=str(entry.get("message", "")),
                            )
                        )
            failure_samples = BoundedFailureSamples(samples=tuple(samples), truncated=truncated)
        else:
            failure_samples = BoundedFailureSamples()
        metrics_raw = payload.get("metrics") or {}
        metrics: dict[str, float] = {}
        if isinstance(metrics_raw, dict):
            for key, value in metrics_raw.items():
                coerced = _coerce_float(value)
                if coerced is not None:
                    metrics[str(key)] = coerced
        progress = _coerce_float(payload.get("progress", 0.0)) or 0.0
        affected_rows = _coerce_int(payload.get("affected_rows", 0)) or 0
        estimated_time_s = _coerce_float(payload.get("estimated_time_s", 0.0)) or 0.0
        started_raw = payload.get("started_at")
        completed_raw = payload.get("completed_at")
        error_raw = payload.get("error")
        return cls(
            operation_id=op_id,
            kind=kind,
            targets=targets,
            status=status,
            progress=progress,
            started_at=started_raw if isinstance(started_raw, str) else None,
            completed_at=completed_raw if isinstance(completed_raw, str) else None,
            error=error_raw if isinstance(error_raw, str) else None,
            affected_rows=affected_rows,
            estimated_time_s=estimated_time_s,
            results=results,
            scope=scope_obj,
            reason=reason_obj,
            resume_cursor=resume_cursor,
            failure_samples=failure_samples,
            metrics=metrics,
        )


def _collect_archive_debt_statuses(config: Config, *, include_expensive: bool) -> dict[str, ArchiveDebtStatus]:
    """Collect archive-debt statuses over ``index.db``.

    Returns an empty mapping when ``index.db`` does not yet exist (fresh
    archive before first ingest), mirroring the staleness-inventory contract.
    """
    from contextlib import closing

    from polylogue.paths import archive_file_set_root_for_paths
    from polylogue.storage.repair import collect_archive_debt_statuses_sync
    from polylogue.storage.sqlite.connection_profile import open_readonly_connection

    index_db = (
        archive_file_set_root_for_paths(archive_root_path=config.archive_root, db_anchor=config.db_path) / "index.db"
    )
    if not index_db.exists():
        return {}
    with closing(open_readonly_connection(index_db)) as conn:
        return collect_archive_debt_statuses_sync(
            conn,
            db_path=index_db,
            include_expensive=include_expensive,
            probe_only=False,
        )


def preview_backfill(
    config: Config,
    targets: tuple[str, ...],
    *,
    scope_filter: MaintenanceScopeFilter | None = None,
) -> BackfillOperation:
    """Preview what would be rebuilt for the given targets. Read-only.

    Uses existing DerivedModelStatus queries and archive debt collection
    to report how many rows are affected and an estimated completion time.
    No mutations are performed.
    """
    from polylogue.storage.repair import (
        preview_counts_from_archive_debt,
    )

    operation_id = str(uuid.uuid4())
    catalog = build_maintenance_target_catalog()
    resolved = catalog.resolve(targets)
    resolved_names = tuple(spec.name for spec in resolved)
    effective_filter = scope_filter or MaintenanceScopeFilter()

    if not resolved_names:
        return BackfillOperation(
            operation_id=operation_id,
            kind=BackfillKind.DERIVED_REBUILD,
            targets=(),
            status=BackfillStatus.FAILED,
            error="No valid targets resolved from input",
            scope=MaintenanceScope(targets=(), filter=effective_filter),
        )

    if resolved_names == ("orphaned_blobs",):
        from polylogue.storage.repair import repair_orphaned_blobs

        preview_result = repair_orphaned_blobs(config, dry_run=True)
        return BackfillOperation(
            operation_id=operation_id,
            kind=BackfillKind.ARCHIVE_SUBSET,
            targets=resolved_names,
            status=BackfillStatus.PENDING,
            affected_rows=preview_result.repaired_count,
            estimated_time_s=0.0,
            results=[preview_result.to_dict()],
            scope=MaintenanceScope(targets=resolved_names, filter=effective_filter),
            reason=InvalidationReason.UNKNOWN if preview_result.repaired_count else None,
            metrics={"repaired_count": float(preview_result.repaired_count)},
        )

    # Thread the caller's archive db_path through the planner instead of
    # relying on ambient defaults. The original ``connection_context(None)``
    # call ignored ``config.db_path`` entirely, which made the planner
    # behave inconsistently in tests and multi-archive runtimes.
    include_expensive = any(spec.archive_readiness_requires_deep for spec in resolved)
    debt_statuses = _collect_archive_debt_statuses(config, include_expensive=include_expensive)

    preview = preview_counts_from_archive_debt(debt_statuses)

    # Compute affected rows and estimated time
    total_rows = 0
    for name in resolved_names:
        total_rows += preview.get(name, 0)

    # Rough estimate: ~50 rows/s for complex rebuilds (session insights,
    # actions), ~500 rows/s for simple repairs (FTS, WAL).
    estimated_time_s = total_rows / 50.0 if total_rows > 0 else 0.0

    # Build per-target preview results from debt statuses
    preview_results: list[JSONDocument] = []
    reason: InvalidationReason | None = None
    for name in resolved_names:
        status = debt_statuses.get(name)
        if status is not None:
            preview_results.append(status.to_dict())
            if reason is None:
                reason = _derive_invalidation_reason(status)

    # When the caller narrows by session_ids, the affected-rows
    # estimate must shrink to match: a one-session scope cannot
    # legitimately advertise the full archive's debt as its plan.
    if effective_filter.session_ids is not None:
        scope_size = len(effective_filter.session_ids)
        total_rows = min(total_rows, scope_size) if total_rows > 0 else 0
        estimated_time_s = total_rows / 50.0 if total_rows > 0 else 0.0

    return BackfillOperation(
        operation_id=operation_id,
        kind=BackfillKind.DERIVED_REBUILD,
        targets=resolved_names,
        status=BackfillStatus.PENDING,
        affected_rows=total_rows,
        estimated_time_s=estimated_time_s,
        results=preview_results,
        scope=MaintenanceScope(targets=resolved_names, filter=effective_filter),
        reason=reason,
    )


def execute_backfill(
    config: Config,
    targets: tuple[str, ...],
    *,
    dry_run: bool = False,
    scope_filter: MaintenanceScopeFilter | None = None,
) -> BackfillOperation:
    """Execute (or dry-run) a backfill for the given targets.

    Dispatches to existing repair.py infrastructure via
    run_selected_maintenance. Progress is reported via structured logging.
    """
    from polylogue.storage.repair import (
        preview_counts_from_archive_debt,
        run_selected_maintenance,
    )

    operation_id = str(uuid.uuid4())
    catalog = build_maintenance_target_catalog()
    resolved = catalog.resolve(targets)
    resolved_names = tuple(spec.name for spec in resolved)
    effective_filter = scope_filter or MaintenanceScopeFilter()

    if not resolved_names:
        return BackfillOperation(
            operation_id=operation_id,
            kind=BackfillKind.DERIVED_REBUILD,
            targets=(),
            status=BackfillStatus.FAILED,
            error="No valid targets resolved from input",
            scope=MaintenanceScope(targets=(), filter=effective_filter),
        )

    logger.info(
        "backfill_starting",
        operation_id=operation_id,
        targets=resolved_names,
        dry_run=dry_run,
    )

    # Thread the caller's archive db_path through the planner instead of
    # relying on ambient defaults. See ``preview_backfill`` above.
    debt_statuses = _collect_archive_debt_statuses(config, include_expensive=False)
    preview = preview_counts_from_archive_debt(debt_statuses)

    reason: InvalidationReason | None = None
    for name in resolved_names:
        status = debt_statuses.get(name)
        if status is not None:
            reason = _derive_invalidation_reason(status)
            if reason is not None:
                break

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
            kind=BackfillKind.DERIVED_REBUILD,
            targets=resolved_names,
            status=BackfillStatus.COMPLETED if all_success else BackfillStatus.FAILED,
            progress=1.0,
            started_at=started_at,
            completed_at=completed_at,
            affected_rows=total_repaired,
            estimated_time_s=0.0,
            results=[r.to_dict() for r in repair_results],
            scope=MaintenanceScope(targets=resolved_names, filter=effective_filter),
            reason=reason,
            metrics={"repaired_count": float(total_repaired)},
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
            kind=BackfillKind.DERIVED_REBUILD,
            targets=resolved_names,
            status=BackfillStatus.FAILED,
            progress=0.0,
            started_at=started_at,
            error=f"Backfill failed: {exc}",
            scope=MaintenanceScope(targets=resolved_names, filter=effective_filter),
            reason=reason,
            failure_samples=BoundedFailureSamples.from_samples(
                [
                    FailureSample(
                        kind=type(exc).__name__,
                        locator="planner.execute_backfill",
                        message=str(exc),
                    )
                ]
            ),
        )


def _derive_invalidation_reason(status: object) -> InvalidationReason | None:
    """Translate a :class:`DerivedModelStatus` into an :class:`InvalidationReason`.

    Kept as a module-level helper (rather than a method on
    :class:`DerivedModelStatus`) so the storage-level status type does
    not depend on the planner enum; the planner is the surface that
    classifies staleness.
    """
    from polylogue.maintenance.models import DerivedModelStatus

    if not isinstance(status, DerivedModelStatus):
        return None
    if status.invalidated_reason is not None:
        return status.invalidated_reason
    if status.ready:
        return None
    if status.materialized_documents == 0 and status.source_documents > 0:
        return InvalidationReason.MISSING
    if status.matches_version is False:
        return InvalidationReason.STALE_MATERIALIZER_VERSION
    if status.stale_rows > 0:
        return InvalidationReason.SOURCE_CHANGED
    if status.missing_provenance_rows > 0:
        return InvalidationReason.PARSER_OR_SCHEMA_CHANGED
    return InvalidationReason.UNKNOWN


__all__ = [
    "MAX_FAILURE_SAMPLES",
    "BackfillKind",
    "BackfillOperation",
    "BackfillStatus",
    "BoundedFailureSamples",
    "FailureSample",
    "InvalidationReason",
    "MaintenanceScope",
    "MaintenanceScopeFilter",
    "execute_backfill",
    "preview_backfill",
]
