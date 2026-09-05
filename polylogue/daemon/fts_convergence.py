"""Single daemon owner for recurring message-FTS convergence."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from polylogue.logging import get_logger
from polylogue.storage.sqlite.connection_profile import open_daemon_connection

logger = get_logger(__name__)


def _is_transient_sqlite_lock(exc: BaseException) -> bool:
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
        partition_keys: tuple[str, ...] | None = None,
    ) -> FtsOwnerResult:
        if not self._db_path.exists():
            return FtsOwnerResult(FtsOwnerState.ABSENT, exact=False)
        try:
            from polylogue.storage.fts.derivation import FtsDerivationAdapter, FtsOutcome

            # ``surfaces`` is the legacy debt subject field.  The domain has
            # one surface and its partitions are session keys, so a
            # messages_fts debt item means a whole-domain pass rather than a
            # partition literally named ``messages_fts``.
            unsupported = tuple(surface for surface in surfaces if surface != "messages_fts")
            if unsupported:
                return FtsOwnerResult(
                    FtsOwnerState.FAILED,
                    exact=False,
                    detail=f"unsupported FTS surface(s): {', '.join(unsupported)}",
                )
            with open_daemon_connection(self._db_path, timeout=30.0) as conn:
                result = FtsDerivationAdapter().converge(conn, keys=partition_keys)
                # The readiness projection is an exact archive-wide audit;
                # a partition-scoped pass cannot claim it and must not pay
                # for it (``make_fts_readiness_stage`` publishes it once per
                # whole-archive convergence pass).
                if result.outcome is FtsOutcome.DONE and partition_keys is None:
                    self._publish_readiness_projection(conn)
            state = {
                FtsOutcome.DONE: FtsOwnerState.READY_EXACT,
                FtsOutcome.PENDING: FtsOwnerState.PENDING,
                FtsOutcome.FAILED: FtsOwnerState.FAILED,
            }[result.outcome]
            return FtsOwnerResult(
                state,
                exact=result.outcome is FtsOutcome.DONE,
                repaired_surfaces=result.written_partitions,
                detail=result.detail,
            )
        except Exception as exc:
            if _is_transient_sqlite_lock(exc):
                return FtsOwnerResult(FtsOwnerState.DEFERRED, exact=False, detail=str(exc))
            logger.warning("fts convergence owner failed reason=%s", reason, exc_info=True)
            return FtsOwnerResult(FtsOwnerState.FAILED, exact=False, detail=f"{type(exc).__name__}: {exc}")

    @staticmethod
    def _publish_readiness_projection(conn: sqlite3.Connection) -> None:
        """Expose a completed domain pass without making freshness authoritative."""
        from polylogue.storage.fts.freshness import (
            EXACT,
            FRESHNESS_TABLE,
            READY,
            ensure_fts_freshness_table_sync,
            record_fts_invariant_snapshot_sync,
        )
        from polylogue.storage.fts.fts_lifecycle import fts_invariant_snapshot_sync

        snapshot = fts_invariant_snapshot_sync(conn)
        surface = snapshot.messages
        if not surface.ready:
            return
        ensure_fts_freshness_table_sync(conn)
        generation = int(conn.execute("PRAGMA user_version").fetchone()[0] or 0)
        row = conn.execute(
            f"SELECT state, verification_kind, exact_generation FROM {FRESHNESS_TABLE} WHERE surface = ?",
            ("messages_fts",),
        ).fetchone()
        if row is not None and row[0] == READY and row[1] == EXACT and row[2] == generation:
            return
        record_fts_invariant_snapshot_sync(conn, snapshot)


__all__ = ["FtsConvergenceOwner", "FtsOwnerResult", "FtsOwnerState", "FtsRunReason"]
