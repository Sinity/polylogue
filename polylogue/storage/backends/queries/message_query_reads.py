"""Read queries for messages."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Literal

import aiosqlite

from polylogue.lib.message.roles import MessageRoleFilter, message_role_sql_values
from polylogue.lib.roles import Role
from polylogue.storage.backends.queries.mappers import _row_to_message
from polylogue.storage.runtime import MessageRecord

MessageTypeName = Literal["summary", "tool_use", "tool_result", "thinking"]

_MESSAGE_TYPE_SQL_COLUMNS: dict[str, str] = {
    "tool_use": "has_tool_use",
    "thinking": "has_thinking",
}


async def get_messages(
    conn: aiosqlite.Connection,
    conversation_id: str,
) -> list[MessageRecord]:
    cursor = await conn.execute(
        "SELECT * FROM messages WHERE conversation_id = ? ORDER BY (sort_key IS NULL), sort_key, message_id",
        (conversation_id,),
    )
    rows = await cursor.fetchall()
    return [_row_to_message(row) for row in rows]


async def get_messages_batch(
    conn: aiosqlite.Connection,
    conversation_ids: list[str],
) -> tuple[dict[str, list[MessageRecord]], list[MessageRecord]]:
    if not conversation_ids:
        return {}, []

    result: dict[str, list[MessageRecord]] = {cid: [] for cid in conversation_ids}
    all_messages: list[MessageRecord] = []
    placeholders = ",".join("?" for _ in conversation_ids)
    cursor = await conn.execute(
        f"SELECT * FROM messages WHERE conversation_id IN ({placeholders}) ORDER BY (sort_key IS NULL), sort_key, message_id",
        conversation_ids,
    )
    rows = await cursor.fetchall()

    for row in rows:
        cid = row["conversation_id"]
        msg = _row_to_message(row)
        if cid in result:
            result[cid].append(msg)
        all_messages.append(msg)

    return result, all_messages


async def get_messages_paginated(
    conn: aiosqlite.Connection,
    conversation_id: str,
    *,
    message_role: MessageRoleFilter = (),
    message_type: MessageTypeName | None = None,
    limit: int = 50,
    offset: int = 0,
) -> tuple[list[MessageRecord], int]:
    """Return paginated messages for a conversation with optional filters.

    Returns (messages, total_count) where total_count is the unfiltered
    count of messages matching the SQL-level filters (before limit/offset).
    """
    query = "SELECT * FROM messages WHERE conversation_id = ?"
    count_query = "SELECT COUNT(*) FROM messages WHERE conversation_id = ?"
    params: list[str | int] = [conversation_id]

    role_values = message_role_sql_values(message_role)
    if role_values:
        placeholders = ",".join("?" for _ in role_values)
        query += f" AND role IN ({placeholders})"
        count_query += f" AND role IN ({placeholders})"
        params.extend(role_values)

    # SQL-pushable message_type filters
    if message_type and message_type in _MESSAGE_TYPE_SQL_COLUMNS:
        col = _MESSAGE_TYPE_SQL_COLUMNS[message_type]
        query += f" AND {col} = 1"
        count_query += f" AND {col} = 1"

    # Get total count before pagination
    count_cursor = await conn.execute(count_query, tuple(params))
    total = (await count_cursor.fetchone())[0]

    query += " ORDER BY (sort_key IS NULL), sort_key, message_id"
    query += " LIMIT ? OFFSET ?"
    params.extend([limit, offset])

    cursor = await conn.execute(query, tuple(params))
    rows = await cursor.fetchall()
    messages = [_row_to_message(row) for row in rows]

    # Post-filter for types that need content_blocks inspection
    if message_type and message_type not in _MESSAGE_TYPE_SQL_COLUMNS and messages:
        messages = await _post_filter_by_message_type(conn, messages, message_type)

    return messages, total


async def _post_filter_by_message_type(
    conn: aiosqlite.Connection,
    messages: list[MessageRecord],
    message_type: str,
) -> list[MessageRecord]:
    """Filter messages by content-block-derived type (tool_result, summary)."""
    message_ids = [m.message_id for m in messages]
    placeholders = ",".join("?" for _ in message_ids)
    cursor = await conn.execute(
        f"SELECT message_id, type FROM content_blocks WHERE message_id IN ({placeholders})",
        message_ids,
    )
    rows = await cursor.fetchall()

    blocks_by_message: dict[str, list[str]] = {}
    for row in rows:
        blocks_by_message.setdefault(row["message_id"], []).append(row["type"])

    if message_type == "tool_result":
        return [m for m in messages if "tool_result" in blocks_by_message.get(m.message_id, [])]
    if message_type == "summary":
        # Summary messages: role is system, all content blocks are text
        # (Claude Code type=summary records become system-role messages)
        return [
            m
            for m in messages
            if m.role == "system" and all(t in ("text",) for t in blocks_by_message.get(m.message_id, ["text"]))
        ]

    return messages


async def iter_messages(
    conn: aiosqlite.Connection,
    conversation_id: str,
    *,
    chunk_size: int = 100,
    dialogue_only: bool = False,
    message_roles: MessageRoleFilter = (),
    limit: int | None = None,
) -> AsyncIterator[MessageRecord]:
    offset = 0
    yielded = 0
    effective_roles = message_roles or ((Role.USER, Role.ASSISTANT) if dialogue_only else ())
    role_values = message_role_sql_values(effective_roles)

    while True:
        query = "SELECT * FROM messages WHERE conversation_id = ?"
        params: list[str | int] = [conversation_id]

        if role_values:
            placeholders = ",".join("?" for _ in role_values)
            query += f" AND role IN ({placeholders})"
            params.extend(role_values)

        query += " ORDER BY (sort_key IS NULL), sort_key, message_id"

        fetch_limit = chunk_size
        if limit is not None:
            remaining = limit - yielded
            if remaining <= 0:
                break
            fetch_limit = min(chunk_size, remaining)

        query += " LIMIT ? OFFSET ?"
        params.extend([fetch_limit, offset])

        cursor = await conn.execute(query, tuple(params))
        rows = list(await cursor.fetchall())

        if not rows:
            break

        for row in rows:
            yield _row_to_message(row)
            yielded += 1
            if limit is not None and yielded >= limit:
                return

        offset += len(rows)
        if len(rows) < fetch_limit:
            break


__all__ = ["get_messages", "get_messages_batch", "get_messages_paginated", "iter_messages"]
