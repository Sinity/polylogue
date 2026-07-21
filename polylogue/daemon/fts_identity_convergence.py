"""Periodic exact FTS identity-ledger drift recompute (polylogue-miwv).

``messages_fts_identity`` self-heals inline on every ordinary write path (see
``storage/fts/sql.py`` and the polylogue-1xc.12/polylogue-miwv notes in
``docs/internals.md``), and full reconciliation (``fts_invariant_snapshot_sync``)
already runs during rebuild, batched repair, and daemon startup. None of those
call sites is a *scheduled*, quiet-cadence pass whose sole job is recomputing
the exact snapshot and re-recording it -- they all run opportunistically,
attached to some other event (a rebuild, a repair pass, a boot).

That matters because ``daemon/fts_startup.py``'s bounded STALE-write path
(``_ensure_archive_messages_fts_startup_readiness_sync``) can call
``record_fts_surface_state_sync`` with only the bounded ``source_rows``/
``indexed_rows``/``missing_rows``/``excess_rows`` counts it has on hand,
never ``identity_mismatch_rows`` -- the parameter defaults to ``0`` and the
``ON CONFLICT`` upsert overwrites the freshness row unconditionally, so a
previously recorded nonzero ``identity_mismatch_rows`` silently resets to 0
without ever being recomputed. This module is the scheduled recompute
authority that corrects that: a periodic, quiet-cadence pass that always
runs the real exact reconciliation (``fts_invariant_snapshot_sync``) and
records its real ``identity_mismatch_rows``, rather than trusting whatever a
narrower bounded caller last wrote.

Deliberately NOT a ``DaemonConverger``/``ConvergenceStage`` (see
``docs/retro/2026-05-24-1498-cascade.md``: ``convergence_stages.py`` is
~1,100 lines accreted through the #1498 cascade and its own verdict is
"refactor before adding a fourth stage"). Instead this follows the
``periodic_judgment_automation_sweep`` (polylogue-6qjc) shape: a plain
``asyncio`` loop scheduled directly in ``daemon/cli.py`` alongside the other
maintenance loops, self-contained and independently testable.
"""

from __future__ import annotations

import asyncio
import contextlib
import sqlite3
from dataclasses import dataclass
from pathlib import Path

from polylogue.logging import get_logger
from polylogue.sources.live.sqlite_locking import is_transient_sqlite_lock

logger = get_logger(__name__)

#: Quiet-cadence interval for the scheduled exact recompute, matching
#: ``_periodic_db_optimize``'s reasoning: this is a full-archive exact
#: reconciliation scan (several aggregate joins over ``blocks``/
#: ``messages_fts_docsize``/``messages_fts_identity``), the same cost class
#: as PRAGMA optimize's planner-stat refresh, not a hot-path check. No
#: separate config knob -- reuses the same fixed-interval, no-toggle pattern
#: ``_periodic_db_optimize`` already established for this cost class.
FTS_IDENTITY_DRIFT_RECOMPUTE_INTERVAL_SECONDS = 86_400  # 24 hours


@dataclass(frozen=True, slots=True)
class FtsIdentityDriftRecomputeResult:
    """Bounded, secret-safe summary of one recompute pass for daemon logging."""

    ran: bool = False
    ready: bool = False
    identity_mismatch_rows: int = 0
    missing_rows: int = 0
    excess_rows: int = 0
    duplicate_rows: int = 0
    drift_samples_written: int = 0


def run_fts_identity_drift_recompute_once_sync(db_path: Path) -> FtsIdentityDriftRecomputeResult:
    """Run one exact FTS invariant recompute against ``db_path`` (index.db).

    Always calls the real exact reconciliation
    (:func:`polylogue.storage.fts.fts_lifecycle.fts_invariant_snapshot_sync`)
    -- never a bounded/heuristic substitute -- then records the result via
    :func:`polylogue.storage.fts.freshness.record_fts_invariant_snapshot_sync`
    and appends an ops.db drift-magnitude sample via
    :func:`polylogue.storage.fts.drift_sampling.sample_fts_drift_to_ops_sync`.
    This is the same three-call composition already used by
    ``fts_lifecycle.py``'s rebuild/repair paths and
    ``fts_startup.py``'s successful-startup path -- this module's only job is
    scheduling that composition on a standalone quiet cadence, independent of
    any other event.

    A missing ``db_path`` (archive not yet initialized) or any
    ``sqlite3.Error`` is a bounded no-op, matching every other periodic
    daemon maintenance loop's tolerance for an archive that isn't ready yet.
    """
    if not db_path.exists():
        return FtsIdentityDriftRecomputeResult()

    from polylogue.storage.fts.drift_sampling import sample_fts_drift_to_ops_sync
    from polylogue.storage.fts.freshness import record_fts_invariant_snapshot_sync
    from polylogue.storage.fts.fts_lifecycle import fts_invariant_snapshot_sync
    from polylogue.storage.sqlite.connection_profile import open_connection

    conn = open_connection(db_path, timeout=30.0)
    try:
        snapshot = fts_invariant_snapshot_sync(conn)
        record_fts_invariant_snapshot_sync(conn, snapshot)
        samples_written = sample_fts_drift_to_ops_sync(conn)
        conn.commit()
        return FtsIdentityDriftRecomputeResult(
            ran=True,
            ready=snapshot.ready,
            identity_mismatch_rows=snapshot.messages.identity_mismatch_rows,
            missing_rows=snapshot.messages.missing_rows,
            excess_rows=snapshot.messages.excess_rows,
            duplicate_rows=snapshot.messages.duplicate_rows,
            drift_samples_written=samples_written,
        )
    finally:
        with contextlib.suppress(sqlite3.Error):
            conn.close()


async def periodic_fts_identity_drift_recompute(
    *,
    catch_up_complete: asyncio.Event | None = None,
) -> None:
    """Periodically run one exact FTS identity-ledger drift recompute.

    Gated on ``catch_up_complete`` (when given) so the first pass never races
    initial source catch-up's own FTS churn -- same gating shape as
    ``periodic_embedding_backlog_check``/``periodic_judgment_automation_sweep``.
    Runs on ``daemon_write_coordinator`` like every other periodic write, so
    it serializes with live ingest instead of racing it for the SQLite
    writer lock.
    """
    from polylogue.daemon.write_coordinator import daemon_write_coordinator
    from polylogue.paths import active_index_db_path

    if catch_up_complete is not None:
        await catch_up_complete.wait()
    while True:
        await asyncio.sleep(FTS_IDENTITY_DRIFT_RECOMPUTE_INTERVAL_SECONDS)
        db_path = active_index_db_path()
        try:
            result = await daemon_write_coordinator().run_sync(
                "maintenance.fts_identity_drift_recompute",
                run_fts_identity_drift_recompute_once_sync,
                db_path,
            )
            if result.ran:
                logger.info(
                    "fts_identity_drift_recompute: ready=%s identity_mismatch_rows=%d "
                    "missing_rows=%d excess_rows=%d duplicate_rows=%d drift_samples_written=%d",
                    result.ready,
                    result.identity_mismatch_rows,
                    result.missing_rows,
                    result.excess_rows,
                    result.duplicate_rows,
                    result.drift_samples_written,
                )
        except sqlite3.OperationalError as exc:
            if is_transient_sqlite_lock(exc):
                logger.info("fts_identity_drift_recompute: archive busy; retrying on next tick: %s", exc)
                continue
            logger.warning("fts_identity_drift_recompute: recompute failed", exc_info=True)
        except Exception:
            logger.warning("fts_identity_drift_recompute: recompute failed", exc_info=True)


__all__ = [
    "FTS_IDENTITY_DRIFT_RECOMPUTE_INTERVAL_SECONDS",
    "FtsIdentityDriftRecomputeResult",
    "periodic_fts_identity_drift_recompute",
    "run_fts_identity_drift_recompute_once_sync",
]
