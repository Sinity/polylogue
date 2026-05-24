"""Provider-event storage writes for the split typed schema."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterable

from polylogue.core.common import SQL_PROVIDER_EVENT_INSERT
from polylogue.storage.sqlite.provider_event_model import project_provider_event_payload

ProviderEventWriteTuple = tuple[object, ...]

_COMPACTION_SQL = """
INSERT OR REPLACE INTO provider_event_compactions (
    event_id, summary, trigger, pre_tokens, preserved_segment_id,
    is_modern, replacement_history_count
) VALUES (?, ?, ?, ?, ?, ?, ?)
"""

_TURN_CONTEXT_SQL = """
INSERT OR REPLACE INTO provider_event_turn_contexts (
    event_id, cwd, model, effort, approval_policy, sandbox_policy, summary
) VALUES (?, ?, ?, ?, ?, ?, ?)
"""

_TOOL_CALL_SQL = """
INSERT OR REPLACE INTO provider_event_tool_calls (
    event_id, call_id, tool_name, status, input_chars, output_chars,
    has_input_body, has_output_body
) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
"""

_REASONING_SQL = """
INSERT OR REPLACE INTO provider_event_reasoning (
    event_id, summary, encrypted_content_hash, encrypted_content_bytes
) VALUES (?, ?, ?, ?)
"""

_GHOST_SNAPSHOT_SQL = """
INSERT OR REPLACE INTO provider_event_ghost_snapshots (
    event_id, ghost_commit
) VALUES (?, ?)
"""


def _header_tuple(event: ProviderEventWriteTuple) -> tuple[object, ...]:
    return (
        event[0],
        event[1],
        event[2],
        event[3],
        event[4],
        event[5],
        event[6],
        event[7],
        event[9],
        event[10],
        event[11],
    )


def insert_provider_events_sync(
    conn: sqlite3.Connection,
    events: Iterable[ProviderEventWriteTuple],
    *,
    ignore_existing: bool = False,
) -> None:
    sql = SQL_PROVIDER_EVENT_INSERT
    if ignore_existing:
        sql = sql.replace("INSERT INTO", "INSERT OR IGNORE INTO", 1)
    for event in events:
        conn.execute(sql, _header_tuple(event))
        event_id = event[0]
        event_type = str(event[4])
        payload = event[8]
        projection = project_provider_event_payload(event_type, payload)
        if projection.compaction is not None:
            conn.execute(_COMPACTION_SQL, (event_id, *projection.compaction))
        if projection.turn_context is not None:
            conn.execute(_TURN_CONTEXT_SQL, (event_id, *projection.turn_context))
        if projection.tool_call is not None:
            conn.execute(_TOOL_CALL_SQL, (event_id, *projection.tool_call))
        if projection.reasoning is not None:
            conn.execute(_REASONING_SQL, (event_id, *projection.reasoning))
        if projection.ghost_snapshot is not None:
            conn.execute(_GHOST_SNAPSHOT_SQL, (event_id, *projection.ghost_snapshot))


__all__ = ["insert_provider_events_sync"]
