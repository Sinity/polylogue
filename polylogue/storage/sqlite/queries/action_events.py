"""Durable canonical action-event queries."""

from __future__ import annotations

import aiosqlite

from polylogue.core.common import SQL_ACTION_EVENT_INSERT as _ACTION_EVENT_INSERT_SQL
from polylogue.storage.runtime import ActionEventRecord, _json_array_or_none

__all__ = [
    "replace_action_events",
]


async def replace_action_events(
    conn: aiosqlite.Connection,
    session_id: str,
    records: list[ActionEventRecord],
    transaction_depth: int,
) -> None:
    await conn.execute("DELETE FROM action_events WHERE session_id = ?", (session_id,))
    if records:
        await conn.executemany(
            _ACTION_EVENT_INSERT_SQL,
            [
                (
                    record.event_id,
                    record.session_id,
                    record.message_id,
                    record.materializer_version,
                    record.source_block_id,
                    record.timestamp,
                    record.sort_key,
                    record.sequence_index,
                    record.source_name,
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
                for record in records
            ],
        )
    if transaction_depth == 0:
        await conn.commit()
