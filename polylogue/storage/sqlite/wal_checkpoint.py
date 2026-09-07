"""Bounded WAL checkpoints executing the policy declared in ``connection_profile``.

This module is mechanism only: which modes an escalation may attempt, the WAL
size thresholds and the hold budget are declared once in
``connection_profile.py`` and read from there.
"""

from __future__ import annotations

import os
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path

from polylogue.storage.sqlite.connection_profile import (
    CHECKPOINT_ESCALATION_MODES,
    CHECKPOINT_HOLD_BUDGET_S,
    WAL_ESCALATION_BYTES,
    WAL_WARN_BYTES,
    CheckpointEscalation,
    open_daemon_connection,
)

ARCHIVE_TIER_WAL_FILES = ("source.db", "index.db", "embeddings.db", "user.db", "ops.db")

#: Checkpoint modes in escalation order, so an observation can report how far
#: an escalation actually reached.
CHECKPOINT_MODES = ("PASSIVE", "RESTART", "TRUNCATE")


@dataclass(frozen=True, slots=True)
class WalCheckpointObservation:
    """Outcome of one bounded WAL checkpoint decision."""

    reason: str
    mode: str
    wal_bytes_before: int
    wal_bytes_after: int
    escalation: CheckpointEscalation = "recurring"
    busy_pages: int = 0
    log_pages: int = 0
    checkpointed_pages: int = 0
    elapsed_s: float = 0.0
    error: str | None = None
    blocking_processes: tuple[str, ...] = ()

    @property
    def ran(self) -> bool:
        return self.mode != "none"

    @property
    def blocked(self) -> bool:
        """Whether a reader or writer kept the checkpoint from draining the WAL."""
        return self.busy_pages > 0 or self.checkpointed_pages < self.log_pages

    @property
    def over_hold_budget(self) -> bool:
        return self.elapsed_s > CHECKPOINT_HOLD_BUDGET_S


def _wal_size(db: Path) -> int:
    wal = db.with_suffix(".db-wal")
    if not wal.exists():
        return 0
    try:
        return wal.stat().st_size
    except OSError:
        return 0


def checkpoint_connection(conn: sqlite3.Connection, mode: str) -> tuple[int, int, int]:
    """Run one declared checkpoint mode and return busy/log/checkpointed pages."""
    if mode not in CHECKPOINT_MODES:
        raise ValueError(f"unsupported checkpoint mode: {mode}")
    row = conn.execute(f"PRAGMA wal_checkpoint({mode})").fetchone()
    if row is None:
        raise sqlite3.OperationalError(f"checkpoint returned no result for {mode}")
    return tuple(int(value or 0) for value in row)  # type: ignore[return-value]


def checkpoint_wal(
    db: Path,
    *,
    reason: str,
    escalation: CheckpointEscalation = "recurring",
    warn_bytes: int = WAL_WARN_BYTES,
    escalation_bytes: int = WAL_ESCALATION_BYTES,
    timeout_s: float = 1.0,
    collect_blockers: bool = False,
) -> WalCheckpointObservation:
    """Checkpoint one WAL as far as ``escalation`` permits, and no further.

    The helper never loops and never retries. It attempts each mode the
    escalation declares in order, stopping as soon as the WAL is back under
    ``escalation_bytes`` or SQLite reports busy pages: a busy result means a
    reader still holds frames, and the WAL is retained with that evidence
    rather than fought.

    ``collect_blockers`` walks ``/proc`` to name the processes holding the
    files. It is off by default because that walk costs a scan of every process
    on the host, which no interactive route can afford.
    """
    before = _wal_size(db)
    if before < warn_bytes:
        return WalCheckpointObservation(
            reason=reason,
            mode="none",
            escalation=escalation,
            wal_bytes_before=before,
            wal_bytes_after=before,
        )

    started = time.perf_counter()
    mode = "none"
    busy = log = checkpointed = 0
    error: str | None = None
    try:
        conn = open_daemon_connection(db, timeout=timeout_s)
        try:
            for candidate in CHECKPOINT_ESCALATION_MODES[escalation]:
                mode = candidate.lower()
                busy, log, checkpointed = checkpoint_connection(conn, candidate)
                if busy > 0 or _wal_size(db) < escalation_bytes:
                    break
        finally:
            conn.close()
    except sqlite3.Error as exc:
        error = str(exc)
    after = _wal_size(db)
    blocking_processes = (
        _sqlite_file_holders(db)
        if collect_blockers
        and (busy > 0 or after >= escalation_bytes or (error is not None and "locked" in error.lower()))
        else ()
    )
    return WalCheckpointObservation(
        reason=reason,
        mode=mode,
        escalation=escalation,
        wal_bytes_before=before,
        wal_bytes_after=after,
        busy_pages=busy,
        log_pages=log,
        checkpointed_pages=checkpointed,
        elapsed_s=round(time.perf_counter() - started, 6),
        error=error,
        blocking_processes=blocking_processes,
    )


def checkpoint_archive_wals(
    archive_root: Path,
    *,
    reason: str,
    escalation: CheckpointEscalation = "recurring",
    warn_bytes: int = WAL_WARN_BYTES,
    escalation_bytes: int = WAL_ESCALATION_BYTES,
    timeout_s: float = 1.0,
    collect_blockers: bool = False,
) -> tuple[WalCheckpointObservation, ...]:
    """Checkpoint WAL files for every existing split archive tier.

    The recurring owner is responsible for keeping all archive-tier WALs
    bounded, not only the index tier.  Missing tiers are skipped because
    archive bootstrap and tier readiness checks own creation/version semantics.
    """

    observations: list[WalCheckpointObservation] = []
    for filename in ARCHIVE_TIER_WAL_FILES:
        db = archive_root / filename
        if not db.exists():
            continue
        observations.append(
            checkpoint_wal(
                db,
                reason=reason,
                escalation=escalation,
                warn_bytes=warn_bytes,
                escalation_bytes=escalation_bytes,
                timeout_s=timeout_s,
                collect_blockers=collect_blockers,
            )
        )
    return tuple(observations)


def _sqlite_file_holders(db: Path) -> tuple[str, ...]:
    """Return processes currently holding the SQLite db/wal/shm files."""
    targets = {db.resolve(), db.with_suffix(".db-wal").resolve(), db.with_suffix(".db-shm").resolve()}
    holders: list[str] = []
    proc_root = Path("/proc")
    for entry in proc_root.iterdir():
        if not entry.name.isdigit():
            continue
        pid = entry.name
        fd_dir = entry / "fd"
        try:
            fd_paths = tuple(fd_dir.iterdir())
        except OSError:
            continue
        matched = False
        for fd in fd_paths:
            try:
                if Path(os.readlink(fd)).resolve() in targets:
                    matched = True
                    break
            except OSError:
                continue
        if not matched:
            continue
        holders.append(f"{pid}:{_process_command(entry)}")
    return tuple(sorted(holders))


def _process_command(proc_entry: Path) -> str:
    try:
        comm = (proc_entry / "comm").read_text(encoding="utf-8", errors="replace").strip()
    except OSError:
        comm = "unknown"
    try:
        cmdline = (proc_entry / "cmdline").read_bytes().replace(b"\0", b" ").decode("utf-8", errors="replace").strip()
    except OSError:
        cmdline = ""
    if not cmdline:
        return comm
    return cmdline[:240]


__all__ = [
    "ARCHIVE_TIER_WAL_FILES",
    "CHECKPOINT_MODES",
    "WalCheckpointObservation",
    "checkpoint_archive_wals",
    "checkpoint_connection",
    "checkpoint_wal",
]
