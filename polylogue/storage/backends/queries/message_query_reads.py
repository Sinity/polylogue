"""Read queries for messages."""

from __future__ import annotations

from collections.abc import AsyncIterator

import aiosqlite

from polylogue.lib.message_roles import MessageRoleFilter, message_role_sql_values
from polylogue.lib.roles import Role
from polylogue.storage.backends.queries.mappers import _row_to_message
from polylogue.storage.runtime import MessageRecord


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


__all__ = ["get_messages", "get_messages_batch", "iter_messages"]
