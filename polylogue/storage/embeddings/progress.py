"""Durable progress ledger for embedding catch-up runs."""

from __future__ import annotations

import sqlite3
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypedDict

CatchupRunStatus = Literal["running", "completed", "stopped", "failed", "interrupted"]


class EmbeddingCatchupRunPayload(TypedDict):
    run_id: str
    started_at: str
    updated_at: str
    completed_at: str | None
    status: str
    stop_reason: str | None
    rebuild: bool
    max_sessions: int | None
    max_messages: int | None
    stop_after_seconds: int | None
    max_errors: int | None
    planned_sessions: int
    planned_messages: int
    processed_sessions: int
    embedded_sessions: int
    skipped_sessions: int
    error_count: int
    embedded_messages: int
    estimated_cost_usd: float
    last_session_id: str | None


@dataclass(frozen=True, slots=True)
class CatchupRunStart:
    """Inputs persisted when a bounded embedding catch-up pass starts."""

    rebuild: bool
    max_sessions: int | None
    max_messages: int | None
    stop_after_seconds: int | None
    max_errors: int | None
    planned_sessions: int
    planned_messages: int


@dataclass(frozen=True, slots=True)
class CatchupRunDelta:
    """Progress increment for one attempted session."""

    session_id: str
    embedded: bool = False
    skipped: bool = False
    errored: bool = False
    embedded_messages: int = 0
    estimated_cost_usd: float = 0.0


def ensure_embedding_catchup_runs_table(conn: sqlite3.Connection) -> None:
    """Create the catch-up run ledger if it is missing."""

    conn.execute("""
        CREATE TABLE IF NOT EXISTS embedding_catchup_runs (
            run_id TEXT PRIMARY KEY,
            started_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            completed_at TEXT,
            status TEXT NOT NULL CHECK (status IN ('running', 'completed', 'stopped', 'failed', 'interrupted')),
            stop_reason TEXT,
            rebuild INTEGER NOT NULL DEFAULT 0,
            max_sessions INTEGER,
            max_messages INTEGER,
            stop_after_seconds INTEGER,
            max_errors INTEGER,
            planned_sessions INTEGER NOT NULL DEFAULT 0,
            planned_messages INTEGER NOT NULL DEFAULT 0,
            processed_sessions INTEGER NOT NULL DEFAULT 0,
            embedded_sessions INTEGER NOT NULL DEFAULT 0,
            skipped_sessions INTEGER NOT NULL DEFAULT 0,
            error_count INTEGER NOT NULL DEFAULT 0,
            embedded_messages INTEGER NOT NULL DEFAULT 0,
            estimated_cost_usd REAL NOT NULL DEFAULT 0.0,
            last_session_id TEXT
        )
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_embedding_catchup_runs_started
        ON embedding_catchup_runs(started_at DESC)
    """)


@contextmanager
def _connect(db_path: Path) -> Iterator[sqlite3.Connection]:
    # ``with conn`` commits on success / rolls back on error; the ``finally``
    # closes it (a bare ``with sqlite3.connect(...)`` only commits, never
    # closes — a per-call connection leak).
    conn = sqlite3.connect(db_path, timeout=30.0)
    try:
        with conn:
            yield conn
    finally:
        conn.close()


def start_embedding_catchup_run(db_path: Path, start: CatchupRunStart) -> str:
    """Persist and return a new catch-up run identifier."""

    run_id = uuid.uuid4().hex
    with _connect(db_path) as conn:
        ensure_embedding_catchup_runs_table(conn)
        conn.execute(
            """
            INSERT INTO embedding_catchup_runs (
                run_id, started_at, updated_at, status, rebuild,
                max_sessions, max_messages, stop_after_seconds, max_errors,
                planned_sessions, planned_messages
            ) VALUES (?, datetime('now'), datetime('now'), 'running', ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                int(start.rebuild),
                start.max_sessions,
                start.max_messages,
                start.stop_after_seconds,
                start.max_errors,
                start.planned_sessions,
                start.planned_messages,
            ),
        )
        conn.commit()
    return run_id


def record_embedding_catchup_progress(db_path: Path, run_id: str, delta: CatchupRunDelta) -> None:
    """Add one attempted session to a persisted catch-up run."""

    with _connect(db_path) as conn:
        ensure_embedding_catchup_runs_table(conn)
        cursor = conn.execute(
            """
            UPDATE embedding_catchup_runs
            SET updated_at = datetime('now'),
                processed_sessions = processed_sessions + 1,
                embedded_sessions = embedded_sessions + ?,
                skipped_sessions = skipped_sessions + ?,
                error_count = error_count + ?,
                embedded_messages = embedded_messages + ?,
                estimated_cost_usd = estimated_cost_usd + ?,
                last_session_id = ?
            WHERE run_id = ?
            """,
            (
                int(delta.embedded),
                int(delta.skipped),
                int(delta.errored),
                delta.embedded_messages,
                delta.estimated_cost_usd,
                delta.session_id,
                run_id,
            ),
        )
        if cursor.rowcount != 1:
            raise LookupError(f"embedding catch-up run not found for progress update: {run_id}")
        conn.commit()


def finish_embedding_catchup_run(
    db_path: Path,
    run_id: str,
    *,
    status: CatchupRunStatus,
    stop_reason: str | None = None,
) -> None:
    """Mark a catch-up run terminal."""

    with _connect(db_path) as conn:
        ensure_embedding_catchup_runs_table(conn)
        cursor = conn.execute(
            """
            UPDATE embedding_catchup_runs
            SET updated_at = datetime('now'),
                completed_at = datetime('now'),
                status = ?,
                stop_reason = ?
            WHERE run_id = ?
            """,
            (status, stop_reason, run_id),
        )
        if cursor.rowcount != 1:
            raise LookupError(f"embedding catch-up run not found for finalization: {run_id}")
        conn.commit()


def latest_embedding_catchup_run(conn: sqlite3.Connection) -> EmbeddingCatchupRunPayload | None:
    """Return the most recent catch-up run payload, if any."""

    row = conn.execute(
        """
        SELECT run_id, started_at, updated_at, completed_at, status, stop_reason,
               rebuild, max_sessions, max_messages, stop_after_seconds, max_errors,
               planned_sessions, planned_messages, processed_sessions,
               embedded_sessions, skipped_sessions, error_count,
               embedded_messages, estimated_cost_usd, last_session_id
        FROM embedding_catchup_runs
        ORDER BY started_at DESC, rowid DESC
        LIMIT 1
        """
    ).fetchone()
    if row is None:
        return None
    return {
        "run_id": str(row[0]),
        "started_at": str(row[1]),
        "updated_at": str(row[2]),
        "completed_at": None if row[3] is None else str(row[3]),
        "status": str(row[4]),
        "stop_reason": None if row[5] is None else str(row[5]),
        "rebuild": bool(row[6]),
        "max_sessions": row[7],
        "max_messages": row[8],
        "stop_after_seconds": row[9],
        "max_errors": row[10],
        "planned_sessions": int(row[11] or 0),
        "planned_messages": int(row[12] or 0),
        "processed_sessions": int(row[13] or 0),
        "embedded_sessions": int(row[14] or 0),
        "skipped_sessions": int(row[15] or 0),
        "error_count": int(row[16] or 0),
        "embedded_messages": int(row[17] or 0),
        "estimated_cost_usd": float(row[18] or 0.0),
        "last_session_id": None if row[19] is None else str(row[19]),
    }


__all__ = [
    "CatchupRunDelta",
    "CatchupRunStart",
    "CatchupRunStatus",
    "EmbeddingCatchupRunPayload",
    "ensure_embedding_catchup_runs_table",
    "finish_embedding_catchup_run",
    "latest_embedding_catchup_run",
    "record_embedding_catchup_progress",
    "start_embedding_catchup_run",
]
