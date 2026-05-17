"""Typed maintenance planner contract.

The planner owns the typed contract used by every maintenance / backfill
surface (CLI ``maintenance``, daemon HTTP, MCP, Python API). A
:class:`BackfillOperation` carries everything a caller needs to reason
about *one* maintenance attempt without having to peek into the repair
internals:

* identity — ``operation_id`` plus a typed ``kind``;
* scope — a typed :class:`MaintenanceScope` (target names + optional
  filter) instead of a bare tuple, so future surfaces can attach
  conversation-id filters, time windows, source roots, etc. without
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

from polylogue.config import Config
from polylogue.core.json import JSONDocument, json_document
from polylogue.logging import get_logger
from polylogue.maintenance.invalidation import InvalidationReason
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


class BackfillKind(str, Enum):
    """What kind of maintenance operation this is.

    The values map to the typed-operation taxonomy the rest of the
    maintenance cluster expects (see #1144). The legacy aliases
    (``REBUILD``, ``REINDEX``, ``RESET``, ``BACKFILL``) are retained as
    compatibility shims for callers that haven't migrated yet.
    """

    # Typed taxonomy (issue #1144).
    SOURCE_REPLAY = "source-replay"
    ARCHIVE_SUBSET = "archive-subset"
    DERIVED_REBUILD = "derived-rebuild"
    INDEX_REPAIR = "index-repair"
    SEMANTIC_REMATERIALIZE = "semantic-rematerialize"
    CONFIG_DRIVEN = "config-driven"

    # Legacy aliases — kept so existing surfaces keep round-tripping.
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


@dataclass(frozen=True)
class MaintenanceScope:
    """Typed scope (target ids + optional filter) for a backfill.

    ``targets`` is the resolved canonical target-name tuple; ``filter``
    is a free-form JSON-shaped dict so callers can later attach
    conversation id sets, source-path globs, time windows, etc. without
    a planner-API churn.
    """

    targets: tuple[str, ...]
    filter: JSONDocument = field(default_factory=lambda: json_document({}))

    def to_dict(self) -> JSONDocument:
        return json_document(
            {
                "targets": list(self.targets),
                "filter": self.filter,
            }
        )


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
            kind=BackfillKind.DERIVED_REBUILD,
            targets=(),
            status=BackfillStatus.FAILED,
            error="No valid targets resolved from input",
            scope=MaintenanceScope(targets=()),
        )

    # Thread the caller's archive db_path through the planner instead of
    # relying on ambient defaults. The original ``connection_context(None)``
    # call ignored ``config.db_path`` entirely, which made the planner
    # behave inconsistently in tests and multi-archive runtimes.
    with connection_context(config.db_path) as conn:
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
    reason: InvalidationReason | None = None
    for name in resolved_names:
        status = debt_statuses.get(name)
        if status is not None:
            preview_results.append(status.to_dict())
            if reason is None:
                reason = _derive_invalidation_reason(status)

    return BackfillOperation(
        operation_id=operation_id,
        kind=BackfillKind.DERIVED_REBUILD,
        targets=resolved_names,
        status=BackfillStatus.PENDING,
        affected_rows=total_rows,
        estimated_time_s=estimated_time_s,
        results=preview_results,
        scope=MaintenanceScope(targets=resolved_names),
        reason=reason,
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
            kind=BackfillKind.DERIVED_REBUILD,
            targets=(),
            status=BackfillStatus.FAILED,
            error="No valid targets resolved from input",
            scope=MaintenanceScope(targets=()),
        )

    logger.info(
        "backfill_starting",
        operation_id=operation_id,
        targets=resolved_names,
        dry_run=dry_run,
    )

    # Thread the caller's archive db_path through the planner instead of
    # relying on ambient defaults. See ``preview_backfill`` above.
    with connection_context(config.db_path) as conn:
        debt_statuses = collect_archive_debt_statuses_sync(
            conn,
            include_expensive=False,
            probe_only=False,
        )
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
            scope=MaintenanceScope(targets=resolved_names),
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
            scope=MaintenanceScope(targets=resolved_names),
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
    "execute_backfill",
    "preview_backfill",
]
