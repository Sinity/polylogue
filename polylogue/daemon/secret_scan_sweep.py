"""Periodic bounded archive-wide secret-candidate sweep (polylogue-layg.1).

``polylogue ops scan-secrets --session <id>`` (polylogue-27m fix round) and
``--all`` (this bead's CLI half, ``polylogue/security/secret_scan.py``) are
operator-triggered. Nothing kept coverage current on its own: a session
ingested after the operator's last ``--all`` run would sit unscanned until
someone remembered to run it again.

This module is that missing feeder: a bounded, quiet-cadence sweep that
drains a small page of :func:`~polylogue.security.secret_scan.scan_archive_for_secret_candidates`
each tick, so newly ingested sessions get incremental candidate coverage
without ever doing a full-archive rescan on a single tick.

Deliberately NOT a ``DaemonConverger``/``ConvergenceStage`` (see
``docs/retro/2026-05-24-1498-cascade.md``: ``convergence_stages.py`` is
already large and its own verdict is "refactor before adding a fourth
stage"). Instead this follows the ``fts_orphan_audit``/
``periodic_fts_identity_drift_recompute`` shape: a plain ``asyncio`` loop
scheduled directly in ``daemon/cli.py`` alongside the other maintenance
loops, self-contained and independently testable. Runs on
``daemon_write_coordinator`` like every other periodic write, so it
serializes with live ingest instead of racing it for the SQLite writer lock.
"""

from __future__ import annotations

import asyncio
import sqlite3
from dataclasses import dataclass
from pathlib import Path

from polylogue.logging import get_logger
from polylogue.sources.live.sqlite_locking import is_transient_sqlite_lock

logger = get_logger(__name__)

#: Sessions scanned per sweep tick. Small enough that even a large pending
#: backlog is drained incrementally across many ticks rather than blocking
#: the writer coordinator for a long window.
SECRET_SCAN_SWEEP_SESSION_LIMIT = 50

#: Quiet-cadence interval -- this is ambient background coverage, not a
#: hot-path check, so it runs far less often than live convergence.
SECRET_SCAN_SWEEP_INTERVAL_SECONDS = 900  # 15 minutes


@dataclass(frozen=True, slots=True)
class SecretScanSweepResult:
    """Bounded, secret-safe summary of one sweep tick for daemon logging."""

    ran: bool = False
    sessions_scanned: int = 0
    candidates_found: int = 0
    errors: int = 0
    remaining_pending: int = 0


def run_secret_scan_sweep_once_sync(
    archive_root: Path,
    *,
    max_sessions: int = SECRET_SCAN_SWEEP_SESSION_LIMIT,
) -> SecretScanSweepResult:
    """Scan one bounded page of not-yet-covered sessions for secret candidates.

    Thin wrapper around :func:`scan_archive_for_secret_candidates` that
    returns the daemon-loop-shaped result type. An archive that doesn't
    exist yet (no ``index.db``) is a bounded no-op, matching every other
    periodic daemon maintenance probe's tolerance for an archive that is not
    ready.
    """
    if not (archive_root / "index.db").exists():
        return SecretScanSweepResult()

    from polylogue.security.secret_scan import scan_archive_for_secret_candidates

    result = scan_archive_for_secret_candidates(archive_root, max_sessions=max_sessions)
    return SecretScanSweepResult(
        ran=True,
        sessions_scanned=result.sessions_scanned,
        candidates_found=result.candidates_found,
        errors=result.errors,
        remaining_pending=result.remaining_pending,
    )


async def periodic_secret_scan_sweep(
    *,
    catch_up_complete: asyncio.Event | None = None,
) -> None:
    """Periodically drain one bounded page of the archive-wide secret sweep.

    Gated on ``catch_up_complete`` (when given) so the first pass never
    races initial source catch-up -- same gating shape as every other
    ``catch_up_complete``-gated periodic loop in ``daemon/cli.py``.
    """
    from polylogue.daemon.write_coordinator import daemon_write_coordinator
    from polylogue.paths import archive_root

    if catch_up_complete is not None:
        await catch_up_complete.wait()
    while True:
        await asyncio.sleep(SECRET_SCAN_SWEEP_INTERVAL_SECONDS)
        root = archive_root()
        try:
            result = await daemon_write_coordinator().run_sync(
                "maintenance.secret_scan_sweep",
                run_secret_scan_sweep_once_sync,
                root,
            )
            if result.ran and (result.sessions_scanned or result.candidates_found):
                logger.info(
                    "secret_scan_sweep: scanned=%d candidates_found=%d errors=%d remaining_pending=%d",
                    result.sessions_scanned,
                    result.candidates_found,
                    result.errors,
                    result.remaining_pending,
                )
        except sqlite3.OperationalError as exc:
            if is_transient_sqlite_lock(exc):
                logger.info("secret_scan_sweep: archive busy; retrying on next tick: %s", exc)
                continue
            logger.warning("secret_scan_sweep: sweep failed", exc_info=True)
        except Exception:
            logger.warning("secret_scan_sweep: sweep failed", exc_info=True)


__all__ = [
    "SECRET_SCAN_SWEEP_INTERVAL_SECONDS",
    "SECRET_SCAN_SWEEP_SESSION_LIMIT",
    "SecretScanSweepResult",
    "periodic_secret_scan_sweep",
    "run_secret_scan_sweep_once_sync",
]
