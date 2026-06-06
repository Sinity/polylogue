"""Read queries for messages."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Literal

import aiosqlite

from polylogue.archive.message.roles import MessageRoleFilter, Role, message_role_sql_values
from polylogue.archive.message.types import validate_message_type_filter
from polylogue.storage.runtime import MessageRecord
from polylogue.storage.sqlite.queries.mappers import _row_to_message

MessageTypeName = Literal["message", "summary", "tool_use", "tool_result", "thinking", "context", "protocol"]


async def get_messages(
    conn: aiosqlite.Connection,
    session_id: str,
) -> list[MessageRecord]:
    cursor = await conn.execute(
        "SELECT * FROM messages WHERE session_id = ? ORDER BY (sort_key IS NULL), sort_key, message_id",
        (session_id,),
    )
    rows = await cursor.fetchall()
    return [_row_to_message(row) for row in rows]


async def get_messages_batch(
    conn: aiosqlite.Connection,
    session_ids: list[str],
    *,
    sort_key_since: float | None = None,
    sort_key_until: float | None = None,
    message_role: MessageRoleFilter = (),
) -> tuple[dict[str, list[MessageRecord]], list[MessageRecord]]:
    if not session_ids:
        return {}, []

    result: dict[str, list[MessageRecord]] = {cid: [] for cid in session_ids}
    all_messages: list[MessageRecord] = []
    placeholders = ",".join("?" for _ in session_ids)
    query = f"SELECT * FROM messages WHERE session_id IN ({placeholders})"
    params: list[str | float] = list(session_ids)

    role_values = message_role_sql_values(message_role)
    if role_values:
        role_placeholders = ",".join("?" for _ in role_values)
        query += f" AND role IN ({role_placeholders})"
        params.extend(role_values)

    if sort_key_since is not None:
        query += " AND sort_key >= ?"
        params.append(sort_key_since)

    if sort_key_until is not None:
        query += " AND sort_key <= ?"
        params.append(sort_key_until)

    query += " ORDER BY (sort_key IS NULL), sort_key, message_id"
    cursor = await conn.execute(
        query,
        tuple(params),
    )
    rows = await cursor.fetchall()

    for row in rows:
        cid = row["session_id"]
        msg = _row_to_message(row)
        if cid in result:
            result[cid].append(msg)
        all_messages.append(msg)

    return result, all_messages


async def get_messages_paginated(
    conn: aiosqlite.Connection,
    session_id: str,
    *,
    message_role: MessageRoleFilter = (),
    message_type: MessageTypeName | None = None,
    limit: int = 50,
    offset: int = 0,
) -> tuple[list[MessageRecord], int]:
    """Return paginated messages for a session with optional filters.

    Returns (messages, total_count) where total_count is the unfiltered
    count of messages matching the SQL-level filters (before limit/offset).
    """
    query = "SELECT * FROM messages WHERE session_id = ?"
    count_query = "SELECT COUNT(*) FROM messages WHERE session_id = ?"
    params: list[str | int] = [session_id]

    role_values = message_role_sql_values(message_role)
    if role_values:
        placeholders = ",".join("?" for _ in role_values)
        query += f" AND role IN ({placeholders})"
        count_query += f" AND role IN ({placeholders})"
        params.extend(role_values)

    if message_type:
        normalized_type = validate_message_type_filter(message_type).value
        query += " AND message_type = ?"
        count_query += " AND message_type = ?"
        params.append(normalized_type)

    # Get total count before pagination
    count_cursor = await conn.execute(count_query, tuple(params))
    count_row = await count_cursor.fetchone()
    total = count_row[0] if count_row else 0

    query += " ORDER BY (sort_key IS NULL), sort_key, message_id"
    query += " LIMIT ? OFFSET ?"
    params.extend([limit, offset])

    cursor = await conn.execute(query, tuple(params))
    rows = await cursor.fetchall()
    messages = [_row_to_message(row) for row in rows]

    return messages, total


async def iter_messages(
    conn: aiosqlite.Connection,
    session_id: str,
    *,
    chunk_size: int = 100,
    dialogue_only: bool = False,
    message_roles: MessageRoleFilter = (),
    limit: int | None = None,
) -> AsyncIterator[MessageRecord]:
    """Stream a session's messages in deterministic order, chunked.

    Pagination is keyset, not ``LIMIT/OFFSET``: each chunk is seeded by the
    previous chunk's last ``(sort_key, message_id)`` so a single session's
    stream stays linear instead of re-scanning and discarding all prior rows
    (O(M^2)) on every chunk. The ordering
    ``(sort_key IS NULL), sort_key, message_id`` is unchanged and served by
    ``idx_messages_session_sortkey``. ``message_id`` is the table primary
    key (globally unique), so the keyset cursor is a total order with no skipped
    or duplicated rows across chunk boundaries. NULL-``sort_key`` rows form the
    ordering tail and are advanced by a distinct cursor branch.
    """
    yielded = 0
    effective_roles = message_roles or ((Role.USER, Role.ASSISTANT) if dialogue_only else ())
    role_values = message_role_sql_values(effective_roles)

    # Keyset cursor of the previous chunk's final row. ``have_cursor``
    # distinguishes the first chunk from a genuine NULL-sort_key boundary
    # (``last_sort`` is None in both cases).
    last_sort: float | None = None
    last_id: str = ""
    have_cursor = False

    while True:
        query = "SELECT * FROM messages WHERE session_id = ?"
        params: list[str | float] = [session_id]

        if role_values:
            placeholders = ",".join("?" for _ in role_values)
            query += f" AND role IN ({placeholders})"
            params.extend(role_values)

        if have_cursor:
            if last_sort is not None:
                # Cursor in the non-NULL sort_key group: advance within it and
                # always include the NULL group that follows in the ordering.
                query += (
                    " AND ("
                    "(sort_key IS NOT NULL"
                    " AND (sort_key > ? OR (sort_key = ? AND message_id > ?)))"
                    " OR sort_key IS NULL"
                    ")"
                )
                params.extend([last_sort, last_sort, last_id])
            else:
                # Cursor in the NULL sort_key group (ordering tail): only
                # NULL-sort_key rows with a greater message_id remain.
                query += " AND sort_key IS NULL AND message_id > ?"
                params.append(last_id)

        query += " ORDER BY (sort_key IS NULL), sort_key, message_id"

        fetch_limit = chunk_size
        if limit is not None:
            remaining = limit - yielded
            if remaining <= 0:
                break
            fetch_limit = min(chunk_size, remaining)

        query += " LIMIT ?"
        params.append(fetch_limit)

        cursor = await conn.execute(query, tuple(params))
        rows = list(await cursor.fetchall())

        if not rows:
            break

        last_row = rows[-1]
        last_sort = last_row["sort_key"]
        last_id = str(last_row["message_id"])
        have_cursor = True

        for row in rows:
            yield _row_to_message(row)
            yielded += 1
            if limit is not None and yielded >= limit:
                return

        if len(rows) < fetch_limit:
            break


__all__ = ["get_messages", "get_messages_batch", "get_messages_paginated", "iter_messages"]
