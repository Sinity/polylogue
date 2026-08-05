"""Daemon event ledger backed by archive ops state."""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING

from polylogue.paths import archive_root
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
    return archive_root() / "ops.db"


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


def current_epoch_ms() -> int:
    return int(datetime.now(UTC).timestamp() * 1000)


class CatchUpCycleTerminalOutcome(StrEnum):
    """Typed terminal outcomes for a catch-up lifecycle."""

    SUCCESS = "success"
    FAILURE = "failure"
    CANCELLED = "cancelled"
    STOPPED = "stopped"


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
                current_epoch_ms(),
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


def query_recent_catch_up_lifecycles(*, limit: int = 20) -> Sequence[dict[str, object]]:
    """Return complete event histories for the most recent catch-up operations.

    The bound applies to lifecycle identities, rather than arbitrary ledger
    rows. Each selected operation returns every persisted start, end, and
    terminal boundary so SLO projection can verify its own pairing even when
    unrelated daemon traffic is busy. The newest bulk-import marker is included
    as the other event source that changes the catch-up verdict.
    """
    if limit < 1:
        raise ValueError("catch-up lifecycle limit must be positive")
    conn = _open_events_reader()
    if conn is None:
        return []
    try:
        rows = conn.execute(
            """
            WITH recent_operations AS (
                SELECT operation_id, MAX(id) AS latest_id
                FROM daemon_events
                WHERE kind = 'catch_up_cycle' AND operation_id IS NOT NULL
                GROUP BY operation_id
                ORDER BY latest_id DESC
                LIMIT ?
            ), latest_bulk_import_marker AS (
                SELECT id, ts_ms, kind, operation_id, payload_json
                FROM daemon_events
                WHERE kind IN ('bulk_import_started', 'bulk_import_opened', 'bulk_import_completed', 'bulk_import_closed')
                ORDER BY id DESC
                LIMIT 1
            )
            SELECT event.id, event.ts_ms, event.kind, event.operation_id, event.payload_json
            FROM daemon_events AS event
            JOIN recent_operations AS recent USING (operation_id)
            WHERE event.kind = 'catch_up_cycle'
            UNION ALL
            SELECT id, ts_ms, kind, operation_id, payload_json
            FROM latest_bulk_import_marker
            ORDER BY id DESC
            """,
            (limit,),
        ).fetchall()
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
    terminal_outcome: CatchUpCycleTerminalOutcome | str | None = None,
) -> None:
    """Emit one catch-up convergence cycle envelope.

    Carries the runtime observability matrix declared in #999 (cursor lag,
    attempts taxonomy, errors, queue/backlog, repair state, per-stage timings)
    so downstream tooling can read durable evidence without scraping logs.

    ``phase`` is ``"start"``, ``"end"``, or ``"terminal"``. Terminal events
    require a typed outcome. The same ``operation_id`` ties every boundary
    together, while an end event remains the realized backlog measurement.
    """
    if phase not in {"start", "end", "terminal"}:
        raise ValueError(f"unsupported catch-up cycle phase: {phase!r}")
    if phase == "terminal":
        if terminal_outcome is None:
            raise ValueError("terminal catch-up cycle events require an outcome")
        resolved_terminal_outcome = CatchUpCycleTerminalOutcome(terminal_outcome)
    elif terminal_outcome is not None:
        raise ValueError("only terminal catch-up cycle events may carry an outcome")
    else:
        resolved_terminal_outcome = None
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
        "terminal_outcome": (resolved_terminal_outcome.value if resolved_terminal_outcome is not None else None),
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
#
# ``insight.updated`` / ``progress.update`` / ``progress.complete`` were
# retired here (polylogue-20d.13): grepping the whole codebase found no
# production caller for ``emit_insight_updated``/``emit_progress_update``/
# ``emit_progress_complete`` -- the only callers were their own unit tests,
# and the docstring's claimed consumer (``status --convergence --watch``)
# does not exist in the CLI. Per this bead's AC ("every advertised topic has
# a declared spec and production emitter, or is removed"), an advertised
# topic with no real producer is a completeness defect, not a feature to
# preserve. Wiring real embedding-catchup/insight-rebuild progress into SSE
# remains a legitimate future bead; it should introduce these kinds fresh
# from a real call site rather than resurrect the unwired scaffolding.

EVENT_SESSION_APPENDED = "session.appended"
EVENT_SESSION_UPDATED = "session.updated"
EVENT_MESSAGE_APPENDED = "message.appended"

GRANULAR_EVENT_KINDS: frozenset[str] = frozenset(
    {
        EVENT_SESSION_APPENDED,
        EVENT_SESSION_UPDATED,
        EVENT_MESSAGE_APPENDED,
    }
)


def emit_session_appended(
    *,
    source_name: str | None,
    succeeded_file_count: int,
    failed_file_count: int = 0,
    source_paths: Sequence[str] | None = None,
    session_id: str | None = None,
) -> None:
    """Emit a ``session.appended`` event for a newly-materialized session.

    ``session_id`` is the real archive identity of the session this event
    describes (polylogue-20d.13) -- when known, callers should always pass
    it so consumers can scope refresh/animation to the exact session rather
    than treating every event as "refresh whatever is open". ``None`` is
    reserved for legacy/aggregate callers that genuinely cannot attribute a
    single session (e.g. pre-#1204 opaque batch summaries).
    """
    payload: dict[str, object] = {
        "source_name": source_name,
        "succeeded_file_count": int(succeeded_file_count),
        "failed_file_count": int(failed_file_count),
        "session_id": session_id,
    }
    if source_paths is not None:
        payload["source_paths"] = list(source_paths)
    emit_daemon_event(EVENT_SESSION_APPENDED, operation_id=session_id, payload=payload)


def emit_session_updated(
    *,
    session_id: str,
    source_name: str | None = None,
    appended_count: int = 0,
) -> None:
    """Emit a ``session.updated`` event when an existing session grows.

    Distinct from :func:`emit_session_appended`: the live-ingest append
    route only ever grows a file whose session already exists (a
    cursor-tracked prior observation), so every real producer of this event
    is describing a mutation of a session the reader may already have open
    -- exactly the identity the description this bead started from called
    unscoped (polylogue-20d.13).
    """
    payload: dict[str, object] = {
        "session_id": session_id,
        "source_name": source_name,
        "appended_count": int(appended_count),
    }
    emit_daemon_event(EVENT_SESSION_UPDATED, operation_id=session_id, payload=payload)


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
    "CatchUpCycleTerminalOutcome",
    "EVENT_SESSION_APPENDED",
    "EVENT_SESSION_UPDATED",
    "EVENT_MESSAGE_APPENDED",
    "GRANULAR_EVENT_KINDS",
    "emit_catch_up_cycle",
    "emit_session_appended",
    "emit_session_updated",
    "emit_daemon_event",
    "emit_message_appended",
    "get_daemon_event_counts",
    "get_last_ingestion_batch",
    "get_latest_event_id",
    "get_recent_operations",
    "current_epoch_ms",
    "query_daemon_events",
    "query_events_since",
    "query_recent_catch_up_lifecycles",
]
