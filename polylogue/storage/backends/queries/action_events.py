"""Durable canonical action-event queries."""

from __future__ import annotations

import aiosqlite

from polylogue.storage.backends.queries.mappers import _row_to_action_event
from polylogue.storage.store import ActionEventRecord, _json_array_or_none

__all__ = [
    "get_action_events",
    "get_action_events_batch",
    "replace_action_events",
]


async def get_action_events(
    conn: aiosqlite.Connection,
    conversation_id: str,
) -> list[ActionEventRecord]:
    cursor = await conn.execute(
        """
        SELECT *
        FROM action_events
        WHERE conversation_id = ?
        ORDER BY sort_key, message_id, sequence_index
        """,
        (conversation_id,),
    )
    rows = await cursor.fetchall()
    return [_row_to_action_event(row) for row in rows]


async def get_action_events_batch(
    conn: aiosqlite.Connection,
    conversation_ids: list[str],
) -> dict[str, list[ActionEventRecord]]:
    if not conversation_ids:
        return {}
    result: dict[str, list[ActionEventRecord]] = {conversation_id: [] for conversation_id in conversation_ids}
    batch_size = 900
    for index in range(0, len(conversation_ids), batch_size):
        batch = conversation_ids[index : index + batch_size]
        placeholders = ",".join("?" for _ in batch)
        cursor = await conn.execute(
            f"""
            SELECT *
            FROM action_events
            WHERE conversation_id IN ({placeholders})
            ORDER BY conversation_id, sort_key, message_id, sequence_index
            """,
            batch,
        )
        rows = await cursor.fetchall()
        for row in rows:
            conversation_id = row["conversation_id"]
            if conversation_id in result:
                result[conversation_id].append(_row_to_action_event(row))
    return result


async def replace_action_events(
    conn: aiosqlite.Connection,
    conversation_id: str,
    records: list[ActionEventRecord],
    transaction_depth: int,
) -> None:
    await conn.execute("DELETE FROM action_events WHERE conversation_id = ?", (conversation_id,))
    if records:
        await conn.executemany(
            """
            INSERT INTO action_events (
                event_id,
                conversation_id,
                message_id,
                materializer_version,
                source_block_id,
                timestamp,
                sort_key,
                sequence_index,
                provider_name,
                action_kind,
                tool_name,
                normalized_tool_name,
                tool_id,
                affected_paths_json,
                cwd_path,
                branch_names_json,
                command,
                query_text,
                url,
                output_text,
                search_text
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
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
                for record in records
            ],
        )
    if transaction_depth == 0:
        await conn.commit()
