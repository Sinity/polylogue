"""Durable daemon death forensics and heartbeat state.

The pidfile remains a mutual-exclusion primitive.  It is deliberately not a
liveness assertion: only a fresh row in the disposable ops tier can establish
that a daemon process was recently making progress.
"""

from __future__ import annotations

import atexit
import contextlib
import faulthandler
import os
import signal
import sys
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from types import FrameType
from typing import Any

from polylogue.logging import get_logger
from polylogue.paths import active_index_db_path
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

logger = get_logger(__name__)

DAEMON_HEARTBEAT_INTERVAL_SECONDS = 15 * 60
DAEMON_HEARTBEAT_STALE_AFTER_SECONDS = DAEMON_HEARTBEAT_INTERVAL_SECONDS * 2

_last_heartbeat_monotonic: float | None = None
_active_lifecycle: DaemonLifecycle | None = None
_atexit_registered = False

SignalHandler = signal.Handlers | Callable[[int, FrameType | None], Any] | None


def _now_ms() -> int:
    return int(time.time() * 1000)


def _ops_db_path() -> Path:
    return active_index_db_path().with_name("ops.db")


@dataclass(slots=True)
class DaemonLifecycle:
    """One daemon process's durable forensic record."""

    run_id: str
    ops_db_path: Path
    stopped: bool = False

    @classmethod
    def start(cls, *, details: dict[str, object] | None = None) -> DaemonLifecycle:
        """Create and activate a lifecycle row for the current process."""
        global _active_lifecycle
        lifecycle = cls(run_id=str(uuid.uuid4()), ops_db_path=_ops_db_path())
        _write_lifecycle(
            lifecycle.ops_db_path,
            record_daemon_lifecycle_start,
            run_id=lifecycle.run_id,
            started_at_ms=_now_ms(),
            details={"pid": os.getpid(), **(details or {})},
        )
        _active_lifecycle = lifecycle
        note_process_heartbeat()
        _register_atexit_sentinel()
        return lifecycle

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
        """Persist a terminating signal from a synchronous signal handler."""
        signal_name = signal.Signals(signum).name
        try:
            _write_lifecycle(
                self.ops_db_path,
                record_daemon_lifecycle_signal,
                run_id=self.run_id,
                signal_name=signal_name,
                observed_at_ms=_now_ms(),
            )
        except Exception:
            logger.exception("daemon: could not persist terminating signal %s", signal_name)

    def stop(self, *, exit_kind: str) -> None:
        """Mark the lifecycle row cleanly stopped exactly once."""
        if self.stopped:
            return
        _write_lifecycle(
            self.ops_db_path,
            record_daemon_lifecycle_stop,
            run_id=self.run_id,
            stopped_at_ms=_now_ms(),
            exit_kind=exit_kind,
        )
        self.stopped = True


def _write_lifecycle(
    ops_db_path: Path,
    writer: Callable[..., None],
    /,
    **kwargs: object,
) -> None:
    """Run one short ops-tier lifecycle write with fresh-process recovery."""
    initialize_archive_database(ops_db_path, ArchiveTier.OPS)
    conn = open_daemon_connection(ops_db_path)
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
    except Exception:
        return {"state": "absent", "heartbeat_age_s": None, "running": False}
    try:
        row = latest_daemon_lifecycle(conn)
    except Exception:
        logger.warning("daemon status lifecycle lookup failed", exc_info=True)
        return {"state": "unknown", "heartbeat_age_s": None, "running": False}
    finally:
        conn.close()
    if row is None:
        return {"state": "absent", "heartbeat_age_s": None, "running": False}

    current_ms = _now_ms() if now_ms is None else now_ms
    age_s = max(0.0, (current_ms - row.last_heartbeat_at_ms) / 1000)
    if row.stopped_at_ms is not None:
        state = "stopped"
    elif age_s <= DAEMON_HEARTBEAT_STALE_AFTER_SECONDS:
        state = "fresh"
    else:
        # No stop/atexit marker plus a stale heartbeat is the durable trace
        # of a hard kill or vanished process.
        state = "vanished"
    return {
        "state": state,
        "running": state == "fresh",
        "heartbeat_age_s": round(age_s, 3),
        "run_id": row.run_id,
        "started_at_ms": row.started_at_ms,
        "stopped_at_ms": row.stopped_at_ms,
        "signal": row.signal,
        "exit_kind": row.exit_kind,
    }


def install_signal_handlers(lifecycle: DaemonLifecycle) -> dict[int, SignalHandler]:
    """Install forensic SIGTERM/SIGINT handlers and return previous handlers."""
    previous: dict[int, SignalHandler] = {}

    def handle_signal(signum: int, _frame: FrameType | None) -> None:
        signal_name = signal.Signals(signum).name
        logger.error("daemon: received %s; dumping all thread stacks", signal_name)
        with contextlib.suppress(Exception):
            faulthandler.dump_traceback(file=sys.stderr, all_threads=True)
        lifecycle.record_signal_best_effort(signum)
        if signum == signal.SIGINT:
            raise KeyboardInterrupt
        raise SystemExit(128 + signum)

    for signum in (signal.SIGTERM, signal.SIGINT):
        previous[signum] = signal.signal(signum, handle_signal)
    return previous


def restore_signal_handlers(previous: dict[int, SignalHandler]) -> None:
    """Restore signal handlers installed for the daemon run."""
    for signum, handler in previous.items():
        with contextlib.suppress(ValueError):
            signal.signal(signum, handler)


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
    "DaemonLifecycle",
    "install_signal_handlers",
    "lifecycle_status",
    "note_process_heartbeat",
    "process_heartbeat_age_seconds",
    "restore_signal_handlers",
]
