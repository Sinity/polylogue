"""Durable daemon death forensics and heartbeat state.

The pidfile remains a mutual-exclusion primitive.  It is deliberately not a
liveness assertion: only a fresh row in the disposable ops tier can establish
that a daemon process was recently making progress.
"""

from __future__ import annotations

import asyncio
import atexit
import contextlib
import faulthandler
import os
import signal
import sqlite3
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import FrameType
from typing import Any, cast

from polylogue.logging import ERROR, WARNING, emit
from polylogue.operations.daemon_termination import (
    HostRunIdentity,
    TerminationEvidence,
    TerminationReceipt,
    capture_host_run_identity,
    reconcile_ended_runs,
    record_termination_receipts,
    termination_status,
)
from polylogue.paths import archive_root
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_database
from polylogue.storage.sqlite.archive_tiers.ops_write import (
    latest_daemon_lifecycle,
    record_daemon_lifecycle_heartbeat,
    record_daemon_lifecycle_signal,
    record_daemon_lifecycle_start,
    record_daemon_lifecycle_stop,
)
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from polylogue.storage.sqlite.connection_profile import open_daemon_connection, open_readonly_connection

DAEMON_HEARTBEAT_INTERVAL_SECONDS = 15 * 60
# "vanished" (no stop/atexit marker, heartbeat clearly dead) at 2x the
# interval -- unchanged, pre-existing floor.
DAEMON_HEARTBEAT_STALE_AFTER_SECONDS = DAEMON_HEARTBEAT_INTERVAL_SECONDS * 2
# "stale" (missed at least one scheduled beat, worth a warning) at 1.5x the
# interval (polylogue-7eo7 #2): a heartbeat sitting between 1x and 2x the
# interval used to report as flat "running" with no signal whatsoever --
# the observed case was 1369.8s against a 900s interval, comfortably past
# one missed beat but short of the 1800s "vanished" floor.
DAEMON_HEARTBEAT_STALE_WARN_SECONDS = int(DAEMON_HEARTBEAT_INTERVAL_SECONDS * 1.5)
_SIGNAL_WRITE_TIMEOUT_SECONDS = 0.5

_last_heartbeat_monotonic: float | None = None
_active_lifecycle: DaemonLifecycle | None = None
_atexit_registered = False

SignalHandler = int | Callable[[int, FrameType | None], Any] | None


def _now_ms() -> int:
    return int(time.time() * 1000)


def _ops_db_path() -> Path:
    return archive_root() / "ops.db"


@dataclass(slots=True)
class DaemonLifecycle:
    """One daemon process's durable forensic record."""

    run_id: str
    ops_db_path: Path
    stopped: bool = False
    received_signal_name: str | None = None

    @classmethod
    def start(
        cls,
        *,
        run_id: str,
        archive_root_path: Path,
        details: dict[str, object] | None = None,
        host: HostRunIdentity | None = None,
    ) -> DaemonLifecycle:
        """Create and activate a lifecycle row for the current process.

        ``polylogued`` names the archive its writer lease is bound to, so the
        row lands in that archive's ``ops.db`` rather than one re-resolved here.
        ``run_id`` is the daemon run's own id (``daemon.cli.daemon_run_id``),
        the one its events already carry, so the row joins to its logs.

        ``host`` is the run's host identity (pid, boot id, service-manager
        invocation, cgroup instance and its ``memory.events`` baseline), stored
        under ``details['host']`` so the next start can join host evidence about
        this run's end to it (polylogue-peo). It is observed here when omitted.
        """
        global _active_lifecycle
        if not run_id:
            raise ValueError("a daemon lifecycle requires the run's id")
        ops_db_path = archive_root_path / "ops.db"
        lifecycle = cls(run_id=run_id, ops_db_path=ops_db_path)
        resolved_host = capture_host_run_identity() if host is None else host
        _write_lifecycle(
            lifecycle.ops_db_path,
            record_daemon_lifecycle_start,
            run_id=lifecycle.run_id,
            started_at_ms=_now_ms(),
            details={"pid": resolved_host.pid, **(details or {}), "host": resolved_host.to_details()},
        )
        _active_lifecycle = lifecycle
        note_process_heartbeat()
        _register_atexit_sentinel()
        return lifecycle

    def reconcile_ended_runs(self, evidence: TerminationEvidence) -> tuple[TerminationReceipt, ...]:
        """Classify every earlier run of this archive that has no receipt (read-only).

        Runs off the writer: host evidence is read at its own pace and only
        :meth:`record_termination_receipts` needs the writer.
        """
        conn = open_readonly_connection(self.ops_db_path)
        try:
            return reconcile_ended_runs(conn, current_run_id=self.run_id, evidence=evidence, now_ms=_now_ms())
        finally:
            conn.close()

    def record_termination_receipts(self, receipts: Sequence[TerminationReceipt]) -> int:
        """Persist reconciled receipts once each; return how many were new."""
        written: list[int] = []

        def write(conn: sqlite3.Connection) -> None:
            written.append(record_termination_receipts(conn, receipts))

        _write_lifecycle(self.ops_db_path, write)
        return written[0]

    def heartbeat(self) -> None:
        """Persist one periodic heartbeat and refresh the in-process probe."""
        observed_at_ms = _now_ms()
        _write_lifecycle(
            self.ops_db_path,
            record_daemon_lifecycle_heartbeat,
            run_id=self.run_id,
            heartbeat_at_ms=observed_at_ms,
        )
        note_process_heartbeat()

    def record_signal_best_effort(self, signum: int) -> None:
        """Persist a terminating signal from a synchronous signal handler.

        On the event-loop thread a synchronous write lease must not block the
        loop, so the handler only records the name there and :meth:`stop`
        writes it with the stop marker.
        """
        signal_name = signal.Signals(signum).name
        self.received_signal_name = signal_name
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            self._persist_signal(signal_name)
            return
        threading.Thread(
            target=self._persist_signal, args=(signal_name,), name="daemon-lifecycle-signal", daemon=True
        ).start()

    def _persist_signal(self, signal_name: str) -> None:
        try:
            _write_existing_lifecycle(
                self.ops_db_path,
                record_daemon_lifecycle_signal,
                run_id=self.run_id,
                signal_name=signal_name,
                observed_at_ms=_now_ms(),
            )
        except Exception as exc:
            emit(
                "daemon.lifecycle.signal_not_persisted",
                level=ERROR,
                outcome="error",
                reason="lifecycle_signal_write_failed",
                signal_name=signal_name,
                run_id=self.run_id,
                error_type=type(exc).__name__,
                error_detail=str(exc),
            )

    def stop(self, *, exit_kind: str, bounded: bool = False) -> None:
        """Mark the lifecycle row cleanly stopped exactly once."""
        if self.stopped:
            return
        if self.received_signal_name is not None:
            exit_kind = "signal"
        writer = _write_existing_lifecycle if bounded else _write_lifecycle
        signal_name = self.received_signal_name
        stopped_at_ms = _now_ms()

        def write_stop(conn: sqlite3.Connection) -> None:
            # A signal received while the event loop ran could not be written
            # from its handler; the stop marker carries it so the row never
            # reads ``exit_kind='signal'`` beside a NULL ``signal``.
            if signal_name is not None:
                record_daemon_lifecycle_signal(
                    conn, run_id=self.run_id, signal_name=signal_name, observed_at_ms=stopped_at_ms
                )
            record_daemon_lifecycle_stop(conn, run_id=self.run_id, stopped_at_ms=stopped_at_ms, exit_kind=exit_kind)

        try:
            writer(self.ops_db_path, write_stop)
        except Exception:
            # A signal row may already be durable. Do not let a contended
            # best-effort stop marker re-enter through atexit and delay exit.
            if bounded:
                self.stopped = True
            raise
        self.stopped = True


def _write_lifecycle(
    ops_db_path: Path,
    writer: Callable[..., None],
    /,
    **kwargs: object,
) -> None:
    """Run one short ops-tier lifecycle write with fresh-process recovery."""
    initialize_archive_database(ops_db_path, ArchiveTier.OPS)
    conn = open_daemon_connection(ops_db_path, archive_root=ops_db_path.parent)
    try:
        writer(conn, **kwargs)
    finally:
        conn.close()


def _write_existing_lifecycle(
    ops_db_path: Path,
    writer: Callable[..., None],
    /,
    **kwargs: object,
) -> None:
    """Best-effort bounded write for a signal handler after lifecycle startup.

    The normal start path has already initialized the disposable OPS tier.
    A terminating signal must not spend the ordinary daemon writer timeout
    waiting for an external SQLite lock, so this deliberately skips bootstrap
    DDL and uses a short connection timeout.

    The write is taken under an explicit ``write_lease``. The daemon arms
    ``arm_write_lease_enforcement(process_wide=True)`` for its whole lifetime,
    and every other lifecycle write reaches the ops tier through the write
    coordinator, which holds that lease. This path does not: it runs
    synchronously on the main thread from a signal handler, whose context holds
    no lease, so ``require_write_lease`` refused the connection with
    ``UnleasedWriteError``. ``record_signal_best_effort`` swallowed that, the
    signal name was never persisted, and the row ended up incoherent -- a NULL
    ``signal`` beside ``exit_kind='signal'``, which is what
    ``test_sigterm_read_only_daemon_records_forensics`` caught.

    The lease is an in-process, re-entrant ContextVar authority, not a
    contended resource, so taking it here cannot block the handler. The
    single-writer boundary is not weakened: the bounded connection timeout above
    remains what limits real SQLite contention.
    """
    from polylogue.core.write_lease import write_lease

    with write_lease("daemon.lifecycle.signal", archive_root=ops_db_path.parent):
        conn = open_daemon_connection(
            ops_db_path,
            timeout=_SIGNAL_WRITE_TIMEOUT_SECONDS,
            busy_timeout_ms=int(_SIGNAL_WRITE_TIMEOUT_SECONDS * 1000),
            archive_root=ops_db_path.parent,
        )
        try:
            writer(conn, **kwargs)
        finally:
            conn.close()


def note_process_heartbeat(*, now_monotonic: float | None = None) -> None:
    """Advance the in-process heartbeat used by the no-I/O liveness probe."""
    global _last_heartbeat_monotonic
    _last_heartbeat_monotonic = time.monotonic() if now_monotonic is None else now_monotonic


def process_heartbeat_age_seconds(*, now_monotonic: float | None = None) -> float | None:
    """Return the current process heartbeat age without touching storage."""
    if _last_heartbeat_monotonic is None:
        return None
    now = time.monotonic() if now_monotonic is None else now_monotonic
    return max(0.0, now - _last_heartbeat_monotonic)


def lifecycle_status(*, now_ms: int | None = None) -> dict[str, object]:
    """Project the latest durable lifecycle row into an honest status claim."""
    ops_db_path = _ops_db_path()
    if not ops_db_path.is_file():
        return {"state": "absent", "heartbeat_age_s": None, "running": False}
    try:
        conn = open_readonly_connection(ops_db_path)
    except Exception as exc:
        emit(
            "daemon.lifecycle.status_unavailable",
            level=WARNING,
            outcome="degraded",
            reason="ops_db_unopenable",
            path=ops_db_path,
            error_type=type(exc).__name__,
            error_detail=str(exc),
        )
        return {"state": "unknown", "heartbeat_age_s": None, "running": False}
    termination: dict[str, object]
    try:
        row = latest_daemon_lifecycle(conn)
        if row is None:
            return {"state": "absent", "heartbeat_age_s": None, "running": False}
        termination = termination_status(conn, current_run_id=row.run_id)
    except Exception as exc:
        emit(
            "daemon.lifecycle.status_unavailable",
            level=WARNING,
            outcome="degraded",
            reason="lifecycle_lookup_failed",
            path=ops_db_path,
            error_type=type(exc).__name__,
            error_detail=str(exc),
        )
        return {"state": "unknown", "heartbeat_age_s": None, "running": False}
    finally:
        conn.close()

    current_ms = _now_ms() if now_ms is None else now_ms
    age_s = max(0.0, (current_ms - row.last_heartbeat_at_ms) / 1000)
    if row.stopped_at_ms is not None:
        state = "stopped"
    elif age_s > DAEMON_HEARTBEAT_STALE_AFTER_SECONDS:
        # No stop/atexit marker plus a stale heartbeat is the durable trace
        # of a hard kill or vanished process.
        state = "vanished"
    elif age_s > DAEMON_HEARTBEAT_STALE_WARN_SECONDS:
        # Missed at least one scheduled beat but short of the "vanished"
        # floor -- still running, worth a staleness signal (polylogue-7eo7 #2).
        state = "stale"
    else:
        state = "fresh"
    return {
        "state": state,
        "running": state in ("fresh", "stale"),
        "heartbeat_age_s": round(age_s, 3),
        "run_id": row.run_id,
        "started_at_ms": row.started_at_ms,
        "stopped_at_ms": row.stopped_at_ms,
        "signal": row.signal,
        "exit_kind": row.exit_kind,
        **termination,
    }


def install_signal_handlers(lifecycle: DaemonLifecycle) -> dict[int, SignalHandler]:
    """Install forensic SIGTERM/SIGINT handlers and return previous handlers."""
    previous: dict[int, SignalHandler] = {}

    def handle_signal(signum: int, _frame: FrameType | None) -> None:
        signal_name = signal.Signals(signum).name
        emit(
            "daemon.lifecycle.signal_received",
            level=ERROR,
            outcome="ok",
            reason="dumping_thread_stacks",
            signal_name=signal_name,
        )
        try:
            faulthandler.dump_traceback(file=2, all_threads=True)
        except Exception as exc:
            with contextlib.suppress(OSError):
                os.write(2, f"daemon: faulthandler dump failed: {exc}\n".encode())
        lifecycle.record_signal_best_effort(signum)
        if signum == signal.SIGINT:
            raise KeyboardInterrupt
        raise SystemExit(128 + signum)

    try:
        for signum in (signal.SIGTERM, signal.SIGINT):
            previous[signum] = signal.signal(signum, handle_signal)
    except BaseException:
        restore_signal_handlers(previous)
        raise
    return previous


def restore_signal_handlers(previous: dict[int, SignalHandler]) -> None:
    """Restore signal handlers installed for the daemon run."""
    for signum, handler in previous.items():
        with contextlib.suppress(ValueError):
            signal.signal(signum, cast(signal.Handlers | Callable[[int, FrameType | None], Any] | None, handler))


def _register_atexit_sentinel() -> None:
    global _atexit_registered
    if _atexit_registered:
        return
    atexit.register(_atexit_sentinel)
    _atexit_registered = True


def _atexit_sentinel() -> None:
    """Record a non-clean Python exit; SIGKILL remains visibly stale instead."""
    lifecycle = _active_lifecycle
    if lifecycle is None or lifecycle.stopped:
        return
    with contextlib.suppress(Exception):
        lifecycle.stop(exit_kind="atexit")


__all__ = [
    "DAEMON_HEARTBEAT_INTERVAL_SECONDS",
    "DAEMON_HEARTBEAT_STALE_AFTER_SECONDS",
    "DAEMON_HEARTBEAT_STALE_WARN_SECONDS",
    "DaemonLifecycle",
    "install_signal_handlers",
    "lifecycle_status",
    "note_process_heartbeat",
    "process_heartbeat_age_seconds",
    "restore_signal_handlers",
]
