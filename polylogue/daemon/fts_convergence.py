"""Single daemon owner for recurring FTS convergence.

Startup recovery, archive-wide exact auditing, and persisted surface-debt
repair used to be scheduled by separate daemon loops.  This owner is the
single recurring route; it delegates to the existing transaction-bound FTS
repair and exact-generation snapshot routines.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from pathlib import Path

from polylogue.logging import get_logger

logger = get_logger(__name__)

FTS_CONVERGENCE_EXACT_INTERVAL = timedelta(hours=24)


def _is_transient_sqlite_lock(exc: BaseException) -> bool:
    import sqlite3

    return isinstance(exc, sqlite3.OperationalError) and any(
        token in str(exc).lower() for token in ("database is locked", "database table is locked", "database is busy")
    )


class FtsRunReason(StrEnum):
    STARTUP = "startup"
    PERIODIC = "periodic"
    DEBT_RETRY = "debt_retry"


class FtsOwnerState(StrEnum):
    ABSENT = "absent"
    READY_EXACT = "ready_exact"
    PENDING = "pending"
    DEFERRED = "deferred"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class FtsOwnerResult:
    state: FtsOwnerState
    exact: bool
    repaired_surfaces: int = 0
    detail: str | None = None

    @property
    def ready(self) -> bool:
        return self.state is FtsOwnerState.READY_EXACT

    @property
    def deferred(self) -> bool:
        return self.state is FtsOwnerState.DEFERRED


class FtsConvergenceOwner:
    """Own every recurring daemon route that may publish FTS readiness."""

    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path

    def run_once_sync(
        self,
        *,
        reason: FtsRunReason,
        surfaces: tuple[str, ...] = (),
    ) -> FtsOwnerResult:
        if not self._db_path.exists():
            return FtsOwnerResult(FtsOwnerState.ABSENT, exact=False)
        try:
            if reason is FtsRunReason.STARTUP:
                from polylogue.daemon.fts_startup import ensure_fts_startup_readiness_sync

                ensure_fts_startup_readiness_sync()
                return self._exact_result()
            if reason is FtsRunReason.PERIODIC:
                if not self._exact_due():
                    return FtsOwnerResult(FtsOwnerState.PENDING, exact=False, detail="exact FTS audit not due")
                from polylogue.daemon.fts_identity_convergence import run_fts_identity_drift_recompute_once_sync
                from polylogue.daemon.fts_orphan_audit import run_fts_orphan_audit_once_sync

                audit = run_fts_identity_drift_recompute_once_sync(self._db_path)
                orphan = run_fts_orphan_audit_once_sync(self._db_path)
                return FtsOwnerResult(
                    FtsOwnerState.READY_EXACT if audit.ready else FtsOwnerState.PENDING,
                    exact=audit.ran,
                    repaired_surfaces=orphan.orphaned_sessions_found,
                )
            return self._repair_debt(surfaces)
        except Exception as exc:
            if _is_transient_sqlite_lock(exc):
                return FtsOwnerResult(FtsOwnerState.DEFERRED, exact=False, detail=str(exc))
            logger.warning("fts convergence owner failed reason=%s", reason, exc_info=True)
            return FtsOwnerResult(FtsOwnerState.FAILED, exact=False, detail=f"{type(exc).__name__}: {exc}")

    def _repair_debt(self, surfaces: tuple[str, ...]) -> FtsOwnerResult:
        from polylogue.daemon.convergence_stages import repair_fts_surface_result

        attempted = tuple(dict.fromkeys(surfaces))
        if not attempted:
            return FtsOwnerResult(FtsOwnerState.PENDING, exact=False)
        results = tuple(repair_fts_surface_result(self._db_path, surface) for surface in attempted)
        if any(result.deferred for result in results):
            return FtsOwnerResult(FtsOwnerState.DEFERRED, exact=False, detail="SQLite writer busy")
        if all(result.success for result in results):
            return self._exact_result(repaired_surfaces=len(results))
        return FtsOwnerResult(FtsOwnerState.PENDING, exact=True, repaired_surfaces=len(results))

    def _exact_due(self) -> bool:
        import sqlite3

        try:
            with sqlite3.connect(f"file:{self._db_path}?mode=ro", uri=True) as conn:
                row = conn.execute(
                    "SELECT exact_checked_at FROM fts_freshness_state WHERE surface = 'messages_fts'"
                ).fetchone()
        except sqlite3.Error:
            return True
        if row is None or row[0] is None:
            return True
        try:
            return (
                datetime.now(UTC) - datetime.fromisoformat(str(row[0])).astimezone(UTC)
                >= FTS_CONVERGENCE_EXACT_INTERVAL
            )
        except ValueError:
            return True

    def _exact_result(self, *, repaired_surfaces: int = 0) -> FtsOwnerResult:
        from polylogue.daemon.fts_identity_convergence import run_fts_identity_drift_recompute_once_sync

        result = run_fts_identity_drift_recompute_once_sync(self._db_path)
        return FtsOwnerResult(
            FtsOwnerState.READY_EXACT if result.ready else FtsOwnerState.PENDING,
            exact=result.ran,
            repaired_surfaces=repaired_surfaces,
        )


__all__ = ["FTS_CONVERGENCE_EXACT_INTERVAL", "FtsConvergenceOwner", "FtsOwnerResult", "FtsOwnerState", "FtsRunReason"]
