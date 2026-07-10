"""Shared archive write precedence helpers.

Writer module: index.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Sequence
from typing import Literal

from polylogue.core.timestamps import parse_timestamp
from polylogue.sources.parsers.base_models import ParsedSessionEvent

BrowserCapturePrecedence = Literal["default", "replace", "skip"]


def browser_capture_precedence(
    *,
    existing_is_dom_fallback: bool,
    incoming_is_dom_fallback: bool,
    existing_has_native_payload: bool,
    incoming_has_native_payload: bool,
    stored_message_count: int,
    incoming_message_count: int,
) -> BrowserCapturePrecedence:
    """Resolve browser-source ownership before provider timestamp freshness."""
    lower_precedence_fallback = incoming_is_dom_fallback and not existing_is_dom_fallback
    lower_precedence_export = (
        existing_has_native_payload
        and not incoming_has_native_payload
        and incoming_message_count <= stored_message_count
    )
    strictly_less_complete = incoming_message_count < stored_message_count and not (
        existing_is_dom_fallback and not incoming_is_dom_fallback
    )
    if lower_precedence_fallback or lower_precedence_export or strictly_less_complete:
        return "skip"

    incoming_owns_browser_merge = (
        (existing_is_dom_fallback and not incoming_is_dom_fallback)
        or (
            incoming_has_native_payload
            and not existing_has_native_payload
            and incoming_message_count >= stored_message_count
        )
        or (
            existing_has_native_payload
            and not incoming_has_native_payload
            and incoming_message_count > stored_message_count
        )
    )
    return "replace" if incoming_owns_browser_merge else "default"


def stored_message_count(conn: sqlite3.Connection, session_id: str) -> int:
    row = conn.execute(
        "SELECT COUNT(*) FROM messages WHERE session_id = ?",
        (session_id,),
    ).fetchone()
    return int(row[0] or 0) if row is not None else 0


def session_has_parser_ingest_flag(conn: sqlite3.Connection, session_id: str, flag: str) -> bool:
    row = conn.execute(
        """
        SELECT 1
        FROM session_tags
        WHERE session_id = ?
          AND tag = ?
          AND tag_source = 'auto'
          AND method = 'parser'
        LIMIT 1
        """,
        (session_id, flag),
    ).fetchone()
    return row is not None


def record_capture_gap_event(
    conn: sqlite3.Connection,
    *,
    session_id: str,
    existing_raw_id: str,
    incoming_raw_id: str,
    stored_message_count: int,
    incoming_message_count: int,
) -> None:
    row = conn.execute(
        """
        SELECT MAX(position) + 1
        FROM (
            SELECT position FROM session_events WHERE session_id = ?
            UNION ALL
            SELECT position FROM session_agent_policies WHERE session_id = ?
            UNION ALL
            SELECT position FROM session_provider_usage_events WHERE session_id = ?
        )
        """,
        (session_id, session_id, session_id),
    ).fetchone()
    position = int(row[0] or 0) if row is not None else 0
    summary = (
        "Skipped lower-precedence DOM browser-capture fallback "
        f"{incoming_raw_id!r}; existing raw {existing_raw_id!r} has "
        f"{stored_message_count} message(s), incoming fallback has {incoming_message_count}."
    )
    conn.execute(
        """
        INSERT OR REPLACE INTO session_events (
            session_id, source_message_id, source_message_provider_id,
            position, event_type, summary, occurred_at_ms
        ) VALUES (?, NULL, NULL, ?, 'capture_gap', ?, NULL)
        """,
        (session_id, position, summary),
    )


def record_source_outage_events(
    conn: sqlite3.Connection,
    *,
    session_id: str,
    events: Sequence[ParsedSessionEvent],
) -> int:
    """Persist source-outage telemetry a capture declares about itself even
    when its message content loses the browser-capture precedence merge and
    is otherwise discarded.

    A capture that loses the content merge can still be telling the truth
    about when it was not observing the page: that claim does not depend on
    whether its transcript content wins. Mirrors ``record_capture_gap_event``:
    a lightweight write survives the skip path instead of the whole incoming
    session's evidence being silently dropped alongside its discarded
    messages.
    """
    outage_events = [event for event in events if event.event_type == "source_outage"]
    if not outage_events:
        return 0
    row = conn.execute(
        """
        SELECT MAX(position) + 1
        FROM (
            SELECT position FROM session_events WHERE session_id = ?
            UNION ALL
            SELECT position FROM session_agent_policies WHERE session_id = ?
            UNION ALL
            SELECT position FROM session_provider_usage_events WHERE session_id = ?
        )
        """,
        (session_id, session_id, session_id),
    ).fetchone()
    position = int(row[0] or 0) if row is not None else 0
    for event in outage_events:
        summary = str(event.payload.get("summary") or "")
        occurred_at_ms: int | None = None
        if event.timestamp:
            parsed_timestamp = parse_timestamp(event.timestamp)
            if parsed_timestamp is not None:
                occurred_at_ms = int(parsed_timestamp.timestamp() * 1000)
        conn.execute(
            """
            INSERT OR REPLACE INTO session_events (
                session_id, source_message_id, source_message_provider_id,
                position, event_type, summary, payload_json, occurred_at_ms
            ) VALUES (?, NULL, NULL, ?, ?, ?, ?, ?)
            """,
            (
                session_id,
                position,
                event.event_type,
                summary,
                json.dumps(event.payload, sort_keys=True, ensure_ascii=False),
                occurred_at_ms,
            ),
        )
        position += 1
    return len(outage_events)


__all__ = [
    "BrowserCapturePrecedence",
    "browser_capture_precedence",
    "record_capture_gap_event",
    "record_source_outage_events",
    "session_has_parser_ingest_flag",
    "stored_message_count",
]
