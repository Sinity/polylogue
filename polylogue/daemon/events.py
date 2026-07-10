"""Daemon event ledger backed by archive ops state."""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from polylogue.paths import active_index_db_path
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_database
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from polylogue.storage.sqlite.connection_profile import open_daemon_connection, open_readonly_connection

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

_DAEMON_EVENTS_DDL = """
CREATE TABLE IF NOT EXISTS daemon_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts_ms INTEGER NOT NULL,
    kind TEXT NOT NULL,
    operation_id TEXT,
    payload_json TEXT NOT NULL
 ) STRICT;
CREATE INDEX IF NOT EXISTS idx_daemon_events_kind ON daemon_events(kind);
CREATE INDEX IF NOT EXISTS idx_daemon_events_ts ON daemon_events(ts_ms);
"""


def _events_db_path() -> Path:
    """Return the path to the daemon events SQLite database."""
    dbf = active_index_db_path()
    return dbf.with_name("ops.db")


def _ensure_events_db() -> sqlite3.Connection:
    """Open and initialize the daemon events database for an emitter."""
    path = _events_db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    initialize_archive_database(path, ArchiveTier.OPS)
    conn = open_daemon_connection(path)
    conn.executescript(_DAEMON_EVENTS_DDL)
    return conn


def _open_events_reader() -> sqlite3.Connection | None:
    """Open the existing event ledger read-only, or return ``None``.

    Status, polling, and SSE reads must not turn observation into an ops-tier
    write. A missing file or a pre-event-schema ops database therefore has the
    same documented empty-ledger result without directory creation, tier
    initialization, DDL, or write-profile pragmas.
    """
    path = _events_db_path()
    if not path.is_file():
        return None
    try:
        conn = open_readonly_connection(path)
    except sqlite3.OperationalError:
        if not path.is_file():
            return None
        raise
    try:
        exists = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'daemon_events' LIMIT 1"
        ).fetchone()
    except BaseException:
        conn.close()
        raise
    if exists is None:
        conn.close()
        return None
    return conn


def _now_ms() -> int:
    return int(datetime.now(UTC).timestamp() * 1000)


def _iso_from_ms(value: object) -> str:
    if isinstance(value, int):
        resolved = value
    elif isinstance(value, str | bytes | bytearray):
        resolved = int(value)
    else:
        resolved = int(str(value))
    return datetime.fromtimestamp(resolved / 1000, tz=UTC).isoformat()


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
            "INSERT INTO daemon_events (ts_ms, kind, operation_id, payload_json) VALUES (?, ?, ?, ?)",
            (
                _now_ms(),
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
    conn = _open_events_reader()
    if conn is None:
        return []
    try:
        if kind:
            rows = conn.execute(
                "SELECT id, ts_ms, kind, operation_id, payload_json FROM daemon_events WHERE kind = ? ORDER BY id DESC LIMIT ? OFFSET ?",
                (kind, limit, offset),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT id, ts_ms, kind, operation_id, payload_json FROM daemon_events ORDER BY id DESC LIMIT ? OFFSET ?",
                (limit, offset),
            ).fetchall()
        result = []
        for row in rows:
            result.append(
                {
                    "id": row[0],
                    "ts": _iso_from_ms(row[1]),
                    "kind": row[2],
                    "operation_id": row[3],
                    "payload": json.loads(row[4]),
                }
            )
        return result
    finally:
        conn.close()


def query_events_since(
    last_id: int,
    *,
    kinds: Sequence[str] | None = None,
    limit: int = 200,
) -> list[dict[str, object]]:
    """Return daemon events with ``id > last_id``, oldest-first.

    Used by the live SSE stream and ETag polling fallback in the web reader.
    ``kinds`` restricts to a whitelist (empty/None means all kinds).
    """
    conn = _open_events_reader()
    if conn is None:
        return []
    try:
        kinds_tuple = tuple(kinds or ())
        if kinds_tuple:
            placeholders = ",".join("?" for _ in kinds_tuple)
            sql = (
                f"SELECT id, ts_ms, kind, operation_id, payload_json "
                f"FROM daemon_events WHERE id > ? AND kind IN ({placeholders}) "
                f"ORDER BY id ASC LIMIT ?"
            )
            params: tuple[object, ...] = (last_id, *kinds_tuple, limit)
        else:
            sql = (
                "SELECT id, ts_ms, kind, operation_id, payload_json "
                "FROM daemon_events WHERE id > ? ORDER BY id ASC LIMIT ?"
            )
            params = (last_id, limit)
        rows = conn.execute(sql, params).fetchall()
        return [
            {
                "id": row[0],
                "ts": _iso_from_ms(row[1]),
                "kind": row[2],
                "operation_id": row[3],
                "payload": json.loads(row[4]),
            }
            for row in rows
        ]
    finally:
        conn.close()


def get_latest_event_id() -> int:
    """Return the id of the most recent daemon event, or 0 if none exist."""
    conn = _open_events_reader()
    if conn is None:
        return 0
    try:
        row = conn.execute("SELECT COALESCE(MAX(id), 0) FROM daemon_events").fetchone()
        return int(row[0]) if row is not None else 0
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


def emit_catch_up_cycle(
    *,
    operation_id: str,
    phase: str,
    backlog_start: int,
    backlog_end: int,
    discovered: int,
    attempted: int,
    skipped: int,
    ingested: int,
    quarantine_count: int,
    errors_by_kind: Mapping[str, int],
    cursor_before: Mapping[str, object] | None,
    cursor_after: Mapping[str, object] | None,
    duration_ms: float,
    stage_timings_s: Mapping[str, float] | None,
    repair: Mapping[str, object] | None,
) -> None:
    """Emit one catch-up convergence cycle envelope.

    Carries the runtime observability matrix declared in #999 (cursor lag,
    attempts taxonomy, errors, queue/backlog, repair state, per-stage timings)
    so downstream tooling can read durable evidence without scraping logs.

    ``phase`` is ``"start"`` or ``"end"``; the same ``operation_id`` ties them
    together. End events carry the realized counts and timings.
    """
    payload: dict[str, object] = {
        "phase": phase,
        "backlog_start": backlog_start,
        "backlog_end": backlog_end,
        "discovered": discovered,
        "attempted": attempted,
        "skipped": skipped,
        "ingested": ingested,
        "quarantine_count": quarantine_count,
        "errors_by_kind": dict(errors_by_kind),
        "cursor_before": dict(cursor_before) if cursor_before is not None else None,
        "cursor_after": dict(cursor_after) if cursor_after is not None else None,
        "duration_ms": round(float(duration_ms), 3),
        "stage_timings_s": (
            {key: round(float(value), 6) for key, value in stage_timings_s.items()} if stage_timings_s else {}
        ),
        "repair": dict(repair) if repair is not None else None,
    }
    emit_daemon_event("catch_up_cycle", operation_id=operation_id, payload=payload)


# --------------------------------------------------------------------------
# Granular event kinds (#1204)
# --------------------------------------------------------------------------
#
# These constants name the per-topic SSE events the reader subscribes to.
# Older opaque kinds (``ingestion_batch``/``ingest``/``reset``/``operation``)
# remain on the wire for backwards compatibility with existing consumers
# (status views, polling fallback). The granular kinds below split the
# realtime channel so the reader can subscribe selectively by view and
# animate just-appended rows without rerendering the full list.

EVENT_SESSION_APPENDED = "session.appended"
EVENT_SESSION_UPDATED = "session.updated"
EVENT_MESSAGE_APPENDED = "message.appended"
EVENT_INSIGHT_UPDATED = "insight.updated"
EVENT_PROGRESS_UPDATE = "progress.update"
EVENT_PROGRESS_COMPLETE = "progress.complete"

GRANULAR_EVENT_KINDS: frozenset[str] = frozenset(
    {
        EVENT_SESSION_APPENDED,
        EVENT_SESSION_UPDATED,
        EVENT_MESSAGE_APPENDED,
        EVENT_INSIGHT_UPDATED,
        EVENT_PROGRESS_UPDATE,
        EVENT_PROGRESS_COMPLETE,
    }
)


def emit_session_appended(
    *,
    source_name: str | None,
    succeeded_file_count: int,
    failed_file_count: int = 0,
    source_paths: Sequence[str] | None = None,
) -> None:
    """Emit a ``session.appended`` event for newly-arrived sessions.

    Fired once per live-ingest batch summarising the touched source group.
    Carries enough for the reader to animate new rows without rerendering
    the whole list; the reader still calls ``/api/sessions`` to
    materialise the rows.
    """
    payload: dict[str, object] = {
        "source_name": source_name,
        "succeeded_file_count": int(succeeded_file_count),
        "failed_file_count": int(failed_file_count),
    }
    if source_paths is not None:
        payload["source_paths"] = list(source_paths)
    emit_daemon_event(EVENT_SESSION_APPENDED, payload=payload)


def emit_message_appended(
    *,
    session_id: str | None,
    source_name: str | None = None,
    appended_count: int = 0,
    source_path: str | None = None,
) -> None:
    """Emit a ``message.appended`` event for live-tail consumers.

    The reader subscribes to this topic only for the currently-open
    session; subscription is encoded via ``?kinds=message.appended``
    plus filtering by ``session_id`` on the client.
    """
    payload: dict[str, object] = {
        "session_id": session_id,
        "source_name": source_name,
        "appended_count": int(appended_count),
    }
    if source_path is not None:
        payload["source_path"] = source_path
    emit_daemon_event(EVENT_MESSAGE_APPENDED, payload=payload)


def emit_insight_updated(
    *,
    insight_kind: str,
    session_id: str | None = None,
) -> None:
    """Emit an ``insight.updated`` event when a derived insight rebuilds."""
    payload: dict[str, object] = {
        "insight_kind": insight_kind,
        "session_id": session_id,
    }
    emit_daemon_event(EVENT_INSIGHT_UPDATED, payload=payload)


def emit_progress_update(
    *,
    operation_id: str,
    operation_kind: str,
    completed: int,
    total: int | None = None,
    detail: str | None = None,
    eta_seconds: float | None = None,
) -> None:
    """Emit a ``progress.update`` event for long-running maintenance ops (#996).

    Consumers: web reader status chip; ``status --convergence --watch`` (#1218).
    Coalescing-friendly — the snapshot path collapses bursts into one
    summary frame, so emitters do not need to throttle themselves.
    """
    payload: dict[str, object] = {
        "operation_kind": operation_kind,
        "completed": int(completed),
    }
    if total is not None:
        payload["total"] = int(total)
        payload["fraction"] = round(completed / total, 6) if total > 0 else None
    if detail is not None:
        payload["detail"] = detail
    if eta_seconds is not None:
        payload["eta_seconds"] = round(float(eta_seconds), 3)
    emit_daemon_event(EVENT_PROGRESS_UPDATE, operation_id=operation_id, payload=payload)


def emit_progress_complete(
    *,
    operation_id: str,
    operation_kind: str,
    status: str = "completed",
    detail: str | None = None,
) -> None:
    """Emit a terminal ``progress.complete`` event for a long-running op."""
    payload: dict[str, object] = {
        "operation_kind": operation_kind,
        "status": status,
    }
    if detail is not None:
        payload["detail"] = detail
    emit_daemon_event(EVENT_PROGRESS_COMPLETE, operation_id=operation_id, payload=payload)


def get_daemon_event_counts() -> dict[str, int]:
    """Return event counts by kind."""
    conn = _open_events_reader()
    if conn is None:
        return {}
    try:
        rows = conn.execute("SELECT kind, COUNT(*) FROM daemon_events GROUP BY kind ORDER BY COUNT(*) DESC").fetchall()
        return {row[0]: row[1] for row in rows}
    finally:
        conn.close()


__all__ = [
    "EVENT_SESSION_APPENDED",
    "EVENT_SESSION_UPDATED",
    "EVENT_INSIGHT_UPDATED",
    "EVENT_MESSAGE_APPENDED",
    "EVENT_PROGRESS_COMPLETE",
    "EVENT_PROGRESS_UPDATE",
    "GRANULAR_EVENT_KINDS",
    "emit_catch_up_cycle",
    "emit_session_appended",
    "emit_daemon_event",
    "emit_insight_updated",
    "emit_message_appended",
    "emit_progress_complete",
    "emit_progress_update",
    "get_daemon_event_counts",
    "get_last_ingestion_batch",
    "get_latest_event_id",
    "get_recent_operations",
    "query_daemon_events",
    "query_events_since",
]
