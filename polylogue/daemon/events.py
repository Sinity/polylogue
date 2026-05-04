"""Daemon event ledger — lightweight SQLite-backed event store for daemon telemetry.

Events are written to a separate SQLite database (``daemon_events.db``) in the
same directory as the main archive database. This keeps daemon operational
events independent of the archive schema and avoids schema version coupling.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from polylogue.paths import db_path
from polylogue.storage.sqlite.connection_profile import open_connection

if TYPE_CHECKING:
    from collections.abc import Sequence

_DAEMON_EVENTS_DDL = """
CREATE TABLE IF NOT EXISTS daemon_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,
    kind TEXT NOT NULL,
    operation_id TEXT,
    payload_json TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_daemon_events_kind ON daemon_events(kind);
CREATE INDEX IF NOT EXISTS idx_daemon_events_ts ON daemon_events(ts);
"""


def _events_db_path() -> Path:
    """Return the path to the daemon events SQLite database."""
    dbf = db_path()
    return dbf.parent / "daemon_events.db"


def _ensure_events_db() -> sqlite3.Connection:
    """Open (creating if necessary) the daemon events database."""
    path = _events_db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = open_connection(path)
    conn.executescript(_DAEMON_EVENTS_DDL)
    return conn


def emit_daemon_event(
    kind: str,
    *,
    operation_id: str | None = None,
    payload: dict[str, object] | None = None,
) -> None:
    """Emit a daemon event to the event ledger."""
    conn = _ensure_events_db()
    try:
        conn.execute(
            "INSERT INTO daemon_events (ts, kind, operation_id, payload_json) VALUES (?, ?, ?, ?)",
            (
                datetime.now(UTC).isoformat(),
                kind,
                operation_id,
                json.dumps(payload or {}),
            ),
        )
        conn.commit()
    finally:
        conn.close()


def query_daemon_events(
    *,
    kind: str | None = None,
    limit: int = 100,
    offset: int = 0,
) -> Sequence[dict[str, object]]:
    """Query recent daemon events."""
    conn = _ensure_events_db()
    try:
        if kind:
            rows = conn.execute(
                "SELECT id, ts, kind, operation_id, payload_json FROM daemon_events WHERE kind = ? ORDER BY id DESC LIMIT ? OFFSET ?",
                (kind, limit, offset),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT id, ts, kind, operation_id, payload_json FROM daemon_events ORDER BY id DESC LIMIT ? OFFSET ?",
                (limit, offset),
            ).fetchall()
        result = []
        for row in rows:
            result.append(
                {
                    "id": row[0],
                    "ts": row[1],
                    "kind": row[2],
                    "operation_id": row[3],
                    "payload": json.loads(row[4]),
                }
            )
        return result
    finally:
        conn.close()


def get_last_ingestion_batch() -> dict[str, object] | None:
    """Return the most recent ingestion_batch event, if any."""
    events = query_daemon_events(kind="ingestion_batch", limit=1)
    if events:
        return events[0]
    return None


def get_recent_operations(limit: int = 10) -> Sequence[dict[str, object]]:
    """Return recent daemon operations."""
    return query_daemon_events(kind="operation", limit=limit)


def get_daemon_event_counts() -> dict[str, int]:
    """Return event counts by kind."""
    conn = _ensure_events_db()
    try:
        rows = conn.execute("SELECT kind, COUNT(*) FROM daemon_events GROUP BY kind ORDER BY COUNT(*) DESC").fetchall()
        return {row[0]: row[1] for row in rows}
    finally:
        conn.close()


__all__ = [
    "emit_daemon_event",
    "get_daemon_event_counts",
    "get_last_ingestion_batch",
    "get_recent_operations",
    "query_daemon_events",
]
