"""Persistence helpers for action-event read-model rebuilds."""

from __future__ import annotations

import sqlite3

import aiosqlite

from polylogue.storage.action_event_rebuild_sql import ACTION_EVENT_INSERT_SQL
from polylogue.storage.store import ActionEventRecord, _json_array_or_none


def _record_values(record: ActionEventRecord) -> tuple[object, ...]:
    return (
        record.event_id,
        record.conversation_id,
        record.message_id,
        record.materializer_version,
        record.source_block_id,
        record.timestamp,
        record.sort_key,
        record.sequence_index,
        record.provider_name,
        record.action_kind,
        record.tool_name,
        record.normalized_tool_name,
        record.tool_id,
        _json_array_or_none(record.affected_paths),
        record.cwd_path,
        _json_array_or_none(record.branch_names),
        record.command,
        record.query_text,
        record.url,
        record.output_text,
        record.search_text,
    )


def replace_action_events_sync(
    conn: sqlite3.Connection,
    conversation_id: str,
    records: list[ActionEventRecord],
) -> None:
    conn.execute("DELETE FROM action_events WHERE conversation_id = ?", (conversation_id,))
    if not records:
        return
    conn.executemany(ACTION_EVENT_INSERT_SQL, [_record_values(record) for record in records])


async def replace_action_events_async(
    conn: aiosqlite.Connection,
    conversation_id: str,
    records: list[ActionEventRecord],
) -> None:
    await conn.execute("DELETE FROM action_events WHERE conversation_id = ?", (conversation_id,))
    if not records:
        return
    await conn.executemany(ACTION_EVENT_INSERT_SQL, [_record_values(record) for record in records])
