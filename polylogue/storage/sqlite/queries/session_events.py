"""Current archive session-event queries."""

from __future__ import annotations

import json
import sqlite3
from collections import defaultdict
from collections.abc import Sequence

import aiosqlite

from polylogue.storage.runtime import SessionEventRecord
from polylogue.types import MessageId, SessionEventId, SessionId


def _payload(value: object) -> dict[str, object]:
    if not isinstance(value, str) or not value:
        return {}
    parsed = json.loads(value)
    return dict(parsed) if isinstance(parsed, dict) else {}


def _row_to_session_event(row: sqlite3.Row) -> SessionEventRecord:
    source_message_id = row["source_message_id"]
    return SessionEventRecord(
        event_id=SessionEventId(row["event_id"]),
        session_id=SessionId(row["session_id"]),
        origin=str(row["origin"]),
        event_index=int(row["position"] or 0),
        event_type=str(row["event_type"]),
        timestamp=None,
        sort_key=(float(row["occurred_at_ms"]) / 1000.0 if row["occurred_at_ms"] is not None else None),
        payload=_payload(row["payload_json"]),
        source_message_id=MessageId(source_message_id) if source_message_id is not None else None,
        raw_id=None,
        materializer_version=1,
    )


async def get_session_events(
    conn: aiosqlite.Connection,
    session_id: str,
) -> list[SessionEventRecord]:
    rows = await (
        await conn.execute(
            """
            SELECT se.*, s.origin
            FROM session_events se
            JOIN sessions s ON s.session_id = se.session_id
            WHERE se.session_id = ?
            ORDER BY se.position
            """,
            (session_id,),
        )
    ).fetchall()
    return [_row_to_session_event(row) for row in rows]


async def get_session_events_batch(
    conn: aiosqlite.Connection,
    session_ids: Sequence[str],
) -> dict[str, list[SessionEventRecord]]:
    if not session_ids:
        return {}
    placeholders = ", ".join("?" for _ in session_ids)
    rows = await (
        await conn.execute(
            f"""
            SELECT se.*, s.origin
            FROM session_events se
            JOIN sessions s ON s.session_id = se.session_id
            WHERE se.session_id IN ({placeholders})
            ORDER BY se.session_id, se.position
            """,
            tuple(session_ids),
        )
    ).fetchall()
    result: dict[str, list[SessionEventRecord]] = defaultdict(list)
    for session_id in session_ids:
        result.setdefault(session_id, [])
    for row in rows:
        record = _row_to_session_event(row)
        result[str(record.session_id)].append(record)
    return dict(result)


def sync_session_events_batch(
    conn: sqlite3.Connection,
    session_ids: Sequence[str],
) -> dict[str, list[SessionEventRecord]]:
    if not session_ids:
        return {}
    placeholders = ", ".join("?" for _ in session_ids)
    rows = conn.execute(
        f"""
        SELECT se.*, s.origin
        FROM session_events se
        JOIN sessions s ON s.session_id = se.session_id
        WHERE se.session_id IN ({placeholders})
        ORDER BY se.session_id, se.position
        """,
        tuple(session_ids),
    ).fetchall()
    result: dict[str, list[SessionEventRecord]] = defaultdict(list)
    for session_id in session_ids:
        result.setdefault(session_id, [])
    for row in rows:
        record = _row_to_session_event(row)
        result[str(record.session_id)].append(record)
    return dict(result)


async def get_session_event_compaction_counts(
    conn: aiosqlite.Connection,
    session_ids: Sequence[str],
) -> dict[str, int]:
    if not session_ids:
        return {}
    placeholders = ", ".join("?" for _ in session_ids)
    rows = await (
        await conn.execute(
            f"""
            SELECT session_id, COUNT(*) AS compaction_count
            FROM session_events
            WHERE session_id IN ({placeholders})
              AND event_type = 'compaction'
            GROUP BY session_id
            """,
            tuple(session_ids),
        )
    ).fetchall()
    result = dict.fromkeys(session_ids, 0)
    for row in rows:
        result[str(row["session_id"])] = int(row["compaction_count"] or 0)
    return result


def sync_session_event_compaction_counts(
    conn: sqlite3.Connection,
    session_ids: Sequence[str],
) -> dict[str, int]:
    if not session_ids:
        return {}
    placeholders = ", ".join("?" for _ in session_ids)
    rows = conn.execute(
        f"""
        SELECT session_id, COUNT(*) AS compaction_count
        FROM session_events
        WHERE session_id IN ({placeholders})
          AND event_type = 'compaction'
        GROUP BY session_id
        """,
        tuple(session_ids),
    ).fetchall()
    result = dict.fromkeys(session_ids, 0)
    for row in rows:
        result[str(row["session_id"])] = int(row["compaction_count"] or 0)
    return result


__all__ = [
    "get_session_event_compaction_counts",
    "get_session_events",
    "get_session_events_batch",
    "sync_session_event_compaction_counts",
    "sync_session_events_batch",
]
