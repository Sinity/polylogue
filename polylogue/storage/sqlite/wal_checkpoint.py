"""Bounded WAL checkpoint helpers for daemon ingest and maintenance."""

from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path

from polylogue.storage.sqlite.connection_profile import open_connection

DEFAULT_WAL_WARN_BYTES = 256 * 1024 * 1024
DEFAULT_WAL_TRUNCATE_BYTES = 512 * 1024 * 1024


@dataclass(frozen=True, slots=True)
class WalCheckpointObservation:
    """Outcome of one bounded WAL checkpoint decision."""

    reason: str
    mode: str
    wal_bytes_before: int
    wal_bytes_after: int
    busy_pages: int = 0
    log_pages: int = 0
    checkpointed_pages: int = 0
    elapsed_s: float = 0.0
    error: str | None = None

    @property
    def ran(self) -> bool:
        return self.mode != "none"


def _wal_size(db: Path) -> int:
    wal = db.with_suffix(".db-wal")
    if not wal.exists():
        return 0
    try:
        return wal.stat().st_size
    except OSError:
        return 0


def maybe_checkpoint_wal(
    db: Path,
    *,
    reason: str,
    warn_bytes: int = DEFAULT_WAL_WARN_BYTES,
    truncate_bytes: int = DEFAULT_WAL_TRUNCATE_BYTES,
    timeout_s: float = 1.0,
) -> WalCheckpointObservation:
    """Checkpoint WAL when it crosses a bounded threshold.

    The helper never loops. It first attempts a PASSIVE checkpoint when the
    WAL is beyond ``warn_bytes``. If the WAL is still above
    ``truncate_bytes`` and SQLite reports no busy pages, it follows with a
    TRUNCATE checkpoint. Busy readers are reported, not fought.
    """
    before = _wal_size(db)
    if before < warn_bytes:
        return WalCheckpointObservation(reason=reason, mode="none", wal_bytes_before=before, wal_bytes_after=before)

    started = time.perf_counter()
    mode = "passive"
    busy = log = checkpointed = 0
    error: str | None = None
    try:
        conn = open_connection(db, timeout=timeout_s)
        try:
            row = conn.execute("PRAGMA wal_checkpoint(PASSIVE)").fetchone()
            if row is not None:
                busy, log, checkpointed = int(row[0] or 0), int(row[1] or 0), int(row[2] or 0)
            after_passive = _wal_size(db)
            if busy == 0 and after_passive >= truncate_bytes:
                mode = "truncate"
                row = conn.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
                if row is not None:
                    busy, log, checkpointed = int(row[0] or 0), int(row[1] or 0), int(row[2] or 0)
        finally:
            conn.close()
    except sqlite3.Error as exc:
        error = str(exc)
    after = _wal_size(db)
    return WalCheckpointObservation(
        reason=reason,
        mode=mode,
        wal_bytes_before=before,
        wal_bytes_after=after,
        busy_pages=busy,
        log_pages=log,
        checkpointed_pages=checkpointed,
        elapsed_s=round(time.perf_counter() - started, 6),
        error=error,
    )


__all__ = [
    "DEFAULT_WAL_TRUNCATE_BYTES",
    "DEFAULT_WAL_WARN_BYTES",
    "WalCheckpointObservation",
    "maybe_checkpoint_wal",
]
