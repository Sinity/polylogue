"""Current archive session-event queries."""

from __future__ import annotations

import json
import sqlite3
from collections import defaultdict
from collections.abc import Iterable, Sequence
from datetime import datetime, timezone

import aiosqlite

from polylogue.core.types import MessageId, SessionEventId, SessionId
from polylogue.storage.runtime import SessionEventRecord


def _payload(value: object) -> dict[str, object]:
    if not isinstance(value, str) or not value:
        return {}
    parsed = json.loads(value)
    return dict(parsed) if isinstance(parsed, dict) else {}


def _timestamp(value: object) -> str | None:
    if not isinstance(value, int):
        return None
    return datetime.fromtimestamp(value / 1000.0, tz=timezone.utc).isoformat()


def _row_to_session_event(row: sqlite3.Row) -> SessionEventRecord:
    source_message_id = row["source_message_id"]
    return SessionEventRecord(
        event_id=SessionEventId(row["event_id"]),
        session_id=SessionId(row["session_id"]),
        origin=str(row["origin"]),
        event_index=int(row["position"] or 0),
        event_type=str(row["event_type"]),
        timestamp=_timestamp(row["occurred_at_ms"]),
        sort_key=(float(row["occurred_at_ms"]) / 1000.0 if row["occurred_at_ms"] is not None else None),
        payload=_payload(row["payload_json"]),
        source_message_id=MessageId(source_message_id) if source_message_id is not None else None,
        source_message_provider_id=row["source_message_provider_id"],
        raw_id=None,
        materializer_version=1,
        boundary_start_position=(
            int(row["boundary_start_position"]) if row["boundary_start_position"] is not None else None
        ),
        boundary_end_position=(int(row["boundary_end_position"]) if row["boundary_end_position"] is not None else None),
        boundary_message_id=(MessageId(row["boundary_message_id"]) if row["boundary_message_id"] is not None else None),
    )


_SESSION_EVENTS_SQL = """
    SELECT se.*, s.origin
    FROM session_events se
    JOIN sessions s ON s.session_id = se.session_id
    WHERE se.session_id = ?
    ORDER BY se.position
"""


def read_session_events(conn: sqlite3.Connection, session_id: str) -> list[SessionEventRecord]:
    """Read events from a caller-held index snapshot, including attached indexes."""
    from contextlib import closing

    with closing(conn.cursor()) as cursor:
        cursor.row_factory = sqlite3.Row
        records = [_row_to_session_event(row) for row in cursor.execute(_SESSION_EVENTS_SQL, (session_id,))]
    hydrate_session_event_array_items(conn, records)
    return records


async def get_session_events(
    conn: aiosqlite.Connection,
    session_id: str,
) -> list[SessionEventRecord]:
    async with conn.execute(_SESSION_EVENTS_SQL, (session_id,)) as cursor:
        records = [_row_to_session_event(row) for row in await cursor.fetchall()]
    await _hydrate_array_items_async(conn, records)
    return records


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
    for records in result.values():
        await _hydrate_array_items_async(conn, records)
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
    for records in result.values():
        hydrate_session_event_array_items(conn, records)
    return dict(result)


def hydrate_session_event_array_items(conn: sqlite3.Connection, records: list[SessionEventRecord]) -> None:
    """Restore item rows at the read boundary, retaining ordinary JSON payloads."""
    if not records:
        return
    sessions = {str(record.session_id) for record in records}
    for session_id in sessions:
        matches = [record for record in records if str(record.session_id) == session_id]
        record_by_position = {record.event_index: record for record in matches}
        positions = tuple(record_by_position)
        for offset in range(0, len(positions), 400):
            chunk = positions[offset : offset + 400]
            placeholders = ", ".join("?" for _ in chunk)
            rows = conn.execute(
                "SELECT event_position, payload_key, value_json FROM session_event_array_items "
                f"WHERE session_id = ? AND event_position IN ({placeholders}) "
                "ORDER BY event_position, payload_key, item_ordinal",
                (session_id, *chunk),
            )
            _attach_array_rows(rows, record_by_position)


async def _hydrate_array_items_async(conn: aiosqlite.Connection, records: list[SessionEventRecord]) -> None:
    if not records:
        return
    sessions = {str(record.session_id) for record in records}
    for session_id in sessions:
        matches = [record for record in records if str(record.session_id) == session_id]
        record_by_position = {record.event_index: record for record in matches}
        positions = tuple(record_by_position)
        for offset in range(0, len(positions), 400):
            chunk = positions[offset : offset + 400]
            placeholders = ", ".join("?" for _ in chunk)
            async with conn.execute(
                "SELECT event_position, payload_key, value_json FROM session_event_array_items "
                f"WHERE session_id = ? AND event_position IN ({placeholders}) "
                "ORDER BY event_position, payload_key, item_ordinal",
                (session_id, *chunk),
            ) as cursor:
                rows = await cursor.fetchall()
            _attach_array_rows(rows, record_by_position)


def _attach_array_rows(rows: Iterable[Sequence[object]], record_by_position: dict[int, SessionEventRecord]) -> None:
    current: tuple[int, str] | None = None
    values: list[object] = []
    for position, key, encoded in rows:
        if not isinstance(position, int) or not isinstance(key, str) or not isinstance(encoded, str):
            raise TypeError("session event array item has invalid stored columns")
        coordinate = (position, key)
        if current is not None and coordinate != current:
            target = record_by_position.get(current[0])
            if target is not None:
                target.payload[current[1]] = values
            values = []
        current = coordinate
        values.append(json.loads(encoded))
    if current is not None:
        target = record_by_position.get(current[0])
        if target is not None:
            target.payload[current[1]] = values


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
    "hydrate_session_event_array_items",
    "sync_session_event_compaction_counts",
    "sync_session_events_batch",
]
