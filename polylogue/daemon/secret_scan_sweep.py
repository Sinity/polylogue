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
stage"). Instead this is a plain ``asyncio`` maintenance loop
scheduled directly in ``daemon/cli.py`` alongside the other maintenance
loops, self-contained and independently testable. Runs on
``daemon_write_coordinator`` like every other periodic write, so it
serializes with live ingest instead of racing it for the SQLite writer lock.
"""

from __future__ import annotations

import asyncio
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path

from polylogue.daemon.periodic import daemon_periodic_runner, watcher_registered_gate
from polylogue.logging import WARNING, emit, span
from polylogue.sources.live.sqlite_locking import is_transient_sqlite_lock

#: Sessions scanned per sweep tick. Small enough that even a large pending
#: backlog is drained incrementally across many ticks rather than blocking
#: the writer coordinator for a long window.
SECRET_SCAN_SWEEP_SESSION_LIMIT = 50

#: Quiet-cadence interval -- this is ambient background coverage, not a
#: hot-path check, so it runs far less often than live convergence.
SECRET_SCAN_SWEEP_INTERVAL_SECONDS = 900  # 15 minutes
SECRET_SCAN_SWEEP_STAGE = "maintenance.secret_scan_sweep"


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
    watcher_registered: asyncio.Event | None = None,
) -> None:
    """Periodically drain one bounded page of the archive-wide secret sweep.

    Gated on ``watcher_registered`` (when given) so the first pass never
    races initial source catch-up -- same gating shape as every other
    ``watcher_registered``-gated periodic loop in ``daemon/cli.py``.
    """
    from polylogue.daemon.write_coordinator import daemon_write_coordinator
    from polylogue.paths import archive_root

    async def once() -> None:
        root = archive_root()
        with span("daemon.secret_scan.sweep", stage=SECRET_SCAN_SWEEP_STAGE) as sweep:
            try:
                result = await daemon_write_coordinator().run_sync(
                    SECRET_SCAN_SWEEP_STAGE,
                    run_secret_scan_sweep_once_sync,
                    root,
                )
            except sqlite3.OperationalError as exc:
                await _record_secret_scan_sweep_event_coordinated(root, status="failed", error=exc)
                if is_transient_sqlite_lock(exc):
                    sweep.skipped(reason="archive_busy", error_detail=str(exc))
                else:
                    sweep.degraded("sweep_failed", error_type=type(exc).__name__, error_detail=str(exc))
            except Exception as exc:
                await _record_secret_scan_sweep_event_coordinated(root, status="failed", error=exc)
                sweep.degraded("sweep_failed", error_type=type(exc).__name__, error_detail=str(exc))
            else:
                await _record_secret_scan_sweep_event_coordinated(
                    root,
                    status="failed" if result.errors else "completed",
                    result=result,
                )
                fields = {
                    "scanned": result.sessions_scanned,
                    "candidates": result.candidates_found,
                    "errors": result.errors,
                    "pending": result.remaining_pending,
                }
                if not result.ran:
                    sweep.skipped(reason="sweep_did_not_run", **fields)
                elif result.errors:
                    # Per-session scan errors leave sessions unscanned. Prose
                    # printed them at INFO next to the success counts.
                    sweep.degraded("scan_errors", **fields)
                elif result.sessions_scanned or result.candidates_found:
                    sweep.ok(**fields)
                else:
                    sweep.empty(**fields)

    await daemon_periodic_runner().run(
        "secret_scan_sweep",
        once,
        interval_s=SECRET_SCAN_SWEEP_INTERVAL_SECONDS,
        gate=watcher_registered_gate(watcher_registered),
        run_first=False,
        on_error="record",
        error_event=None,
    )


async def _record_secret_scan_sweep_event_coordinated(
    archive_root_path: Path,
    *,
    status: str,
    result: SecretScanSweepResult | None = None,
    error: BaseException | None = None,
) -> None:
    """Record the sweep outcome through the writer the module claims to use.

    The scan itself runs under ``daemon_write_coordinator``, but the telemetry
    write that follows it opened and committed ``ops.db`` directly from the
    loop, outside the gate -- contradicting this module's own docstring and
    contending with the coordinated writer for no reason.
    """
    from polylogue.daemon.write_coordinator import daemon_write_coordinator

    await daemon_write_coordinator().run_sync(
        f"{SECRET_SCAN_SWEEP_STAGE}.event",
        _record_secret_scan_sweep_event,
        archive_root_path,
        status=status,
        result=result,
        error=error,
    )


def _record_secret_scan_sweep_event(
    archive_root_path: Path,
    *,
    status: str,
    result: SecretScanSweepResult | None = None,
    error: BaseException | None = None,
) -> None:
    """Persist the latest sweep outcome for daemon health readers."""
    from polylogue.operations.daemon_stage_recording import record_daemon_stage_event_for_archive

    payload: dict[str, object] = {"retryable": status == "failed"}
    if result is not None:
        payload.update(
            sessions_scanned=result.sessions_scanned,
            candidates_found=result.candidates_found,
            errors=result.errors,
            remaining_pending=result.remaining_pending,
        )
    if error is not None:
        payload.update(error_type=type(error).__name__)
    try:
        record_daemon_stage_event_for_archive(
            archive_root_path,
            stage=SECRET_SCAN_SWEEP_STAGE,
            status=status,
            observed_at_ms=int(time.time() * 1000),
            payload=payload,
        )
    except Exception as exc:
        emit(
            "daemon.secret_scan.event_record_failed",
            level=WARNING,
            outcome="degraded",
            stage=SECRET_SCAN_SWEEP_STAGE,
            reason="stage_event_not_recorded",
            status=status,
            error_type=type(exc).__name__,
            error_detail=str(exc),
        )


__all__ = [
    "SECRET_SCAN_SWEEP_INTERVAL_SECONDS",
    "SECRET_SCAN_SWEEP_SESSION_LIMIT",
    "SECRET_SCAN_SWEEP_STAGE",
    "SecretScanSweepResult",
    "periodic_secret_scan_sweep",
    "run_secret_scan_sweep_once_sync",
]
