"""Lifecycle helpers for the durable canonical action-event read model."""

from __future__ import annotations

import sqlite3
from collections import defaultdict
from collections.abc import Iterable, Sequence

import aiosqlite

from polylogue.storage.action_event_rows import attach_blocks_to_messages, build_action_event_records
from polylogue.storage.backends.queries.mappers import _row_to_content_block, _row_to_conversation, _row_to_message
from polylogue.storage.store import (
    ACTION_EVENT_MATERIALIZER_VERSION,
    ActionEventRecord,
    ContentBlockRecord,
    ConversationRecord,
    MessageRecord,
    _json_array_or_none,
)

_ACTION_EVENTS_EXISTS_SQL = "SELECT name FROM sqlite_master WHERE type='table' AND name='action_events'"
_ACTION_EVENT_DOC_COUNT_SQL = "SELECT COUNT(*) FROM action_events"
_ACTION_EVENT_MISMATCH_COUNT_SQL = """
    SELECT COUNT(*)
    FROM action_events
    WHERE materializer_version != ?
"""
_ACTION_EVENT_SOURCE_CONVERSATION_COUNT_SQL = """
    SELECT COUNT(DISTINCT conversation_id)
    FROM content_blocks
    WHERE type = 'tool_use'
"""
_ACTION_EVENT_MATERIALIZED_CONVERSATION_COUNT_SQL = """
    SELECT COUNT(DISTINCT conversation_id)
    FROM action_events
"""
_ACTION_EVENT_VALID_SOURCE_CONVERSATION_COUNT_SQL = """
    SELECT COUNT(DISTINCT cb.conversation_id)
    FROM content_blocks cb
    JOIN conversations c ON c.conversation_id = cb.conversation_id
    WHERE cb.type = 'tool_use'
"""
_ACTION_EVENT_ORPHAN_SOURCE_CONVERSATION_COUNT_SQL = """
    SELECT COUNT(DISTINCT cb.conversation_id)
    FROM content_blocks cb
    LEFT JOIN conversations c ON c.conversation_id = cb.conversation_id
    WHERE cb.type = 'tool_use' AND c.conversation_id IS NULL
"""
_ACTION_EVENT_ORPHAN_TOOL_BLOCK_COUNT_SQL = """
    SELECT COUNT(*)
    FROM content_blocks cb
    LEFT JOIN conversations c ON c.conversation_id = cb.conversation_id
    WHERE cb.type = 'tool_use' AND c.conversation_id IS NULL
"""
_ACTION_EVENT_REPAIR_CANDIDATE_IDS_SQL = """
    SELECT DISTINCT cb.conversation_id
    FROM content_blocks cb
    JOIN conversations c ON c.conversation_id = cb.conversation_id
    WHERE cb.type = 'tool_use'
      AND (
          NOT EXISTS (
              SELECT 1
              FROM action_events ae
              WHERE ae.conversation_id = cb.conversation_id
          )
          OR EXISTS (
              SELECT 1
              FROM action_events ae
              WHERE ae.conversation_id = cb.conversation_id
                AND ae.materializer_version != ?
          )
      )
    ORDER BY cb.conversation_id
"""
_ACTION_EVENT_VALID_SOURCE_IDS_SQL = """
    SELECT DISTINCT cb.conversation_id
    FROM content_blocks cb
    JOIN conversations c ON c.conversation_id = cb.conversation_id
    WHERE cb.type = 'tool_use'
    ORDER BY cb.conversation_id
"""
_ACTION_EVENT_CONVERSATION_IDS_SQL = """
    SELECT DISTINCT conversation_id
    FROM content_blocks
    WHERE type = 'tool_use'
    ORDER BY conversation_id
"""


def _chunked(items: Sequence[str], *, size: int) -> Iterable[Sequence[str]]:
    for index in range(0, len(items), size):
        yield items[index : index + size]


def _replace_action_events_sync(
    conn: sqlite3.Connection,
    conversation_id: str,
    records: list[ActionEventRecord],
) -> None:
    conn.execute("DELETE FROM action_events WHERE conversation_id = ?", (conversation_id,))
    if records:
        conn.executemany(
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


async def _replace_action_events_async(
    conn: aiosqlite.Connection,
    conversation_id: str,
    records: list[ActionEventRecord],
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


def action_event_read_model_status_sync(conn: sqlite3.Connection) -> dict[str, int | bool]:
    from polylogue.storage.fts_lifecycle import fts_index_status_sync

    exists = bool(conn.execute(_ACTION_EVENTS_EXISTS_SQL).fetchone())
    count = 0
    stale_count = 0
    source_conversation_count = int(
        conn.execute(_ACTION_EVENT_SOURCE_CONVERSATION_COUNT_SQL).fetchone()[0] or 0
    )
    valid_source_conversation_count = int(
        conn.execute(_ACTION_EVENT_VALID_SOURCE_CONVERSATION_COUNT_SQL).fetchone()[0] or 0
    )
    orphan_source_conversation_count = int(
        conn.execute(_ACTION_EVENT_ORPHAN_SOURCE_CONVERSATION_COUNT_SQL).fetchone()[0] or 0
    )
    orphan_tool_block_count = int(
        conn.execute(_ACTION_EVENT_ORPHAN_TOOL_BLOCK_COUNT_SQL).fetchone()[0] or 0
    )
    materialized_conversation_count = 0
    if exists:
        count = int(conn.execute(_ACTION_EVENT_DOC_COUNT_SQL).fetchone()[0] or 0)
        stale_count = int(
            conn.execute(
                _ACTION_EVENT_MISMATCH_COUNT_SQL,
                (ACTION_EVENT_MATERIALIZER_VERSION,),
            ).fetchone()[0]
            or 0
        )
        materialized_conversation_count = int(
            conn.execute(_ACTION_EVENT_MATERIALIZED_CONVERSATION_COUNT_SQL).fetchone()[0] or 0
        )
    fts_status = fts_index_status_sync(conn)
    action_fts_count = int(fts_status.get("action_count", 0))
    action_fts_ready = count == action_fts_count
    rows_ready = (
        valid_source_conversation_count == 0
        or (
            valid_source_conversation_count == materialized_conversation_count
            and stale_count == 0
        )
    )
    ready = rows_ready and action_fts_ready
    return {
        "exists": exists,
        "count": count,
        "stale_count": stale_count,
        "matches_version": stale_count == 0,
        "source_conversation_count": source_conversation_count,
        "valid_source_conversation_count": valid_source_conversation_count,
        "orphan_source_conversation_count": orphan_source_conversation_count,
        "orphan_tool_block_count": orphan_tool_block_count,
        "materialized_conversation_count": materialized_conversation_count,
        "rows_ready": rows_ready,
        "action_fts_exists": bool(fts_status.get("exists", False)),
        "action_fts_count": action_fts_count,
        "action_fts_ready": action_fts_ready,
        "ready": ready,
    }


async def action_event_read_model_status_async(conn: aiosqlite.Connection) -> dict[str, int | bool]:
    from polylogue.storage.fts_lifecycle import fts_index_status_async

    exists = bool(await (await conn.execute(_ACTION_EVENTS_EXISTS_SQL)).fetchone())
    count = 0
    stale_count = 0
    source_row = await (await conn.execute(_ACTION_EVENT_SOURCE_CONVERSATION_COUNT_SQL)).fetchone()
    source_conversation_count = int(source_row[0] or 0) if source_row else 0
    valid_source_row = await (await conn.execute(_ACTION_EVENT_VALID_SOURCE_CONVERSATION_COUNT_SQL)).fetchone()
    valid_source_conversation_count = int(valid_source_row[0] or 0) if valid_source_row else 0
    orphan_source_row = await (
        await conn.execute(_ACTION_EVENT_ORPHAN_SOURCE_CONVERSATION_COUNT_SQL)
    ).fetchone()
    orphan_source_conversation_count = int(orphan_source_row[0] or 0) if orphan_source_row else 0
    orphan_block_row = await (
        await conn.execute(_ACTION_EVENT_ORPHAN_TOOL_BLOCK_COUNT_SQL)
    ).fetchone()
    orphan_tool_block_count = int(orphan_block_row[0] or 0) if orphan_block_row else 0
    materialized_conversation_count = 0
    if exists:
        row = await (await conn.execute(_ACTION_EVENT_DOC_COUNT_SQL)).fetchone()
        count = int(row[0] or 0) if row else 0
        mismatch_row = await (
            await conn.execute(
                _ACTION_EVENT_MISMATCH_COUNT_SQL,
                (ACTION_EVENT_MATERIALIZER_VERSION,),
            )
        ).fetchone()
        stale_count = int(mismatch_row[0] or 0) if mismatch_row else 0
        materialized_row = await (
            await conn.execute(_ACTION_EVENT_MATERIALIZED_CONVERSATION_COUNT_SQL)
        ).fetchone()
        materialized_conversation_count = int(materialized_row[0] or 0) if materialized_row else 0
    fts_status = await fts_index_status_async(conn)
    action_fts_count = int(fts_status.get("action_count", 0))
    action_fts_ready = count == action_fts_count
    rows_ready = (
        valid_source_conversation_count == 0
        or (
            valid_source_conversation_count == materialized_conversation_count
            and stale_count == 0
        )
    )
    ready = rows_ready and action_fts_ready
    return {
        "exists": exists,
        "count": count,
        "stale_count": stale_count,
        "matches_version": stale_count == 0,
        "source_conversation_count": source_conversation_count,
        "valid_source_conversation_count": valid_source_conversation_count,
        "orphan_source_conversation_count": orphan_source_conversation_count,
        "orphan_tool_block_count": orphan_tool_block_count,
        "materialized_conversation_count": materialized_conversation_count,
        "rows_ready": rows_ready,
        "action_fts_exists": bool(fts_status.get("exists", False)),
        "action_fts_count": action_fts_count,
        "action_fts_ready": action_fts_ready,
        "ready": ready,
    }


def _load_sync_batch(
    conn: sqlite3.Connection,
    conversation_ids: Sequence[str],
) -> tuple[list[ConversationRecord], list[MessageRecord], list[ContentBlockRecord]]:
    placeholders = ", ".join("?" for _ in conversation_ids)
    conversations = [
        _row_to_conversation(row)
        for row in conn.execute(
            f"SELECT * FROM conversations WHERE conversation_id IN ({placeholders})",
            tuple(conversation_ids),
        ).fetchall()
    ]
    messages = [
        _row_to_message(row)
        for row in conn.execute(
            f"""
            SELECT *
            FROM messages
            WHERE conversation_id IN ({placeholders})
            ORDER BY conversation_id, sort_key, message_id
            """,
            tuple(conversation_ids),
        ).fetchall()
    ]
    blocks = [
        _row_to_content_block(row)
        for row in conn.execute(
            f"""
            SELECT *
            FROM content_blocks
            WHERE conversation_id IN ({placeholders})
            ORDER BY conversation_id, message_id, block_index
            """,
            tuple(conversation_ids),
        ).fetchall()
    ]
    return conversations, messages, blocks


async def _load_async_batch(
    conn: aiosqlite.Connection,
    conversation_ids: Sequence[str],
) -> tuple[list[ConversationRecord], list[MessageRecord], list[ContentBlockRecord]]:
    placeholders = ", ".join("?" for _ in conversation_ids)
    conversations = [
        _row_to_conversation(row)
        for row in await (
            await conn.execute(
                f"SELECT * FROM conversations WHERE conversation_id IN ({placeholders})",
                tuple(conversation_ids),
            )
        ).fetchall()
    ]
    messages = [
        _row_to_message(row)
        for row in await (
            await conn.execute(
                f"""
                SELECT *
                FROM messages
                WHERE conversation_id IN ({placeholders})
                ORDER BY conversation_id, sort_key, message_id
                """,
                tuple(conversation_ids),
            )
        ).fetchall()
    ]
    blocks = [
        _row_to_content_block(row)
        for row in await (
            await conn.execute(
                f"""
                SELECT *
                FROM content_blocks
                WHERE conversation_id IN ({placeholders})
                ORDER BY conversation_id, message_id, block_index
                """,
                tuple(conversation_ids),
            )
        ).fetchall()
    ]
    return conversations, messages, blocks


def _materialize_batch(
    conversations: list[ConversationRecord],
    messages: list[MessageRecord],
    blocks: list[ContentBlockRecord],
) -> dict[str, list[ActionEventRecord]]:
    messages_by_conversation: dict[str, list[MessageRecord]] = defaultdict(list)
    for message in messages:
        messages_by_conversation[str(message.conversation_id)].append(message)
    blocks_by_conversation: dict[str, list[ContentBlockRecord]] = defaultdict(list)
    for block in blocks:
        blocks_by_conversation[str(block.conversation_id)].append(block)

    materialized: dict[str, list[ActionEventRecord]] = {}
    for conversation in conversations:
        conversation_id = str(conversation.conversation_id)
        attached_messages = attach_blocks_to_messages(
            messages_by_conversation.get(conversation_id, []),
            blocks_by_conversation.get(conversation_id, []),
        )
        materialized[conversation_id] = build_action_event_records(conversation, attached_messages)
    return materialized


def rebuild_action_event_read_model_sync(
    conn: sqlite3.Connection,
    *,
    conversation_ids: Sequence[str] | None = None,
    page_size: int = 200,
) -> int:
    if conversation_ids is None:
        conn.execute("DELETE FROM action_events")
        conversation_ids = [
            row["conversation_id"]
            for row in conn.execute(_ACTION_EVENT_CONVERSATION_IDS_SQL).fetchall()
        ]
    if not conversation_ids:
        conn.execute("DELETE FROM action_events")
        return 0

    replaced = 0
    for chunk in _chunked(list(conversation_ids), size=page_size):
        conversations, messages, blocks = _load_sync_batch(conn, chunk)
        materialized = _materialize_batch(conversations, messages, blocks)
        for conversation_id in chunk:
            _replace_action_events_sync(conn, conversation_id, materialized.get(conversation_id, []))
            replaced += len(materialized.get(conversation_id, ()))
    return replaced


def action_event_repair_candidates_sync(conn: sqlite3.Connection) -> list[str]:
    rows = conn.execute(
        _ACTION_EVENT_REPAIR_CANDIDATE_IDS_SQL,
        (ACTION_EVENT_MATERIALIZER_VERSION,),
    ).fetchall()
    return [str(row["conversation_id"]) for row in rows]


async def rebuild_action_event_read_model_async(
    conn: aiosqlite.Connection,
    *,
    conversation_ids: Sequence[str] | None = None,
    page_size: int = 200,
) -> int:
    if conversation_ids is None:
        await conn.execute("DELETE FROM action_events")
        rows = await (await conn.execute(_ACTION_EVENT_CONVERSATION_IDS_SQL)).fetchall()
        conversation_ids = [row["conversation_id"] for row in rows]
    if not conversation_ids:
        await conn.execute("DELETE FROM action_events")
        return 0

    replaced = 0
    for chunk in _chunked(list(conversation_ids), size=page_size):
        conversations, messages, blocks = await _load_async_batch(conn, chunk)
        materialized = _materialize_batch(conversations, messages, blocks)
        for conversation_id in chunk:
            await _replace_action_events_async(conn, conversation_id, materialized.get(conversation_id, []))
            replaced += len(materialized.get(conversation_id, ()))
    return replaced


async def action_event_repair_candidates_async(conn: aiosqlite.Connection) -> list[str]:
    rows = await (
        await conn.execute(
            _ACTION_EVENT_REPAIR_CANDIDATE_IDS_SQL,
            (ACTION_EVENT_MATERIALIZER_VERSION,),
        )
    ).fetchall()
    return [str(row["conversation_id"]) for row in rows]


def valid_action_event_source_ids_sync(conn: sqlite3.Connection) -> list[str]:
    rows = conn.execute(_ACTION_EVENT_VALID_SOURCE_IDS_SQL).fetchall()
    return [str(row["conversation_id"]) for row in rows]


async def valid_action_event_source_ids_async(conn: aiosqlite.Connection) -> list[str]:
    rows = await (await conn.execute(_ACTION_EVENT_VALID_SOURCE_IDS_SQL)).fetchall()
    return [str(row["conversation_id"]) for row in rows]


__all__ = [
    "action_event_read_model_status_async",
    "action_event_read_model_status_sync",
    "action_event_repair_candidates_async",
    "action_event_repair_candidates_sync",
    "rebuild_action_event_read_model_async",
    "rebuild_action_event_read_model_sync",
    "valid_action_event_source_ids_async",
    "valid_action_event_source_ids_sync",
]
