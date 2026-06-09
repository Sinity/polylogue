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

_MESSAGE_RECORD_SELECT = """
    m.message_id,
    m.session_id,
    m.native_id AS provider_message_id,
    m.role,
    COALESCE((
        SELECT group_concat(b.text, char(10))
        FROM blocks b
        WHERE b.message_id = m.message_id
          AND b.text IS NOT NULL
    ), '') AS text,
    m.occurred_at_ms / 1000.0 AS sort_key,
    lower(hex(m.content_hash)) AS content_hash,
    1 AS version,
    m.parent_message_id,
    m.variant_index AS branch_index,
    s.origin AS source_name,
    m.word_count,
    m.has_tool_use,
    m.has_thinking,
    m.has_paste,
    m.message_type,
    m.model_name,
    m.input_tokens,
    m.output_tokens,
    m.cache_read_tokens,
    m.cache_write_tokens
"""


async def _resolve_session_id(conn: aiosqlite.Connection, session_id: str) -> str:
    cursor = await conn.execute(
        "SELECT session_id FROM sessions WHERE session_id = ? OR native_id = ? LIMIT 1", (session_id, session_id)
    )
    row = await cursor.fetchone()
    return str(row["session_id"]) if row is not None else session_id


async def get_messages(
    conn: aiosqlite.Connection,
    session_id: str,
) -> list[MessageRecord]:
    session_id = await _resolve_session_id(conn, session_id)
    cursor = await conn.execute(
        f"""
        SELECT {_MESSAGE_RECORD_SELECT}
        FROM messages m
        JOIN sessions s ON s.session_id = m.session_id
        WHERE m.session_id = ?
        ORDER BY (m.occurred_at_ms IS NULL), m.occurred_at_ms, m.message_id
        """,
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
    resolved_pairs = [(session_id, await _resolve_session_id(conn, session_id)) for session_id in session_ids]
    resolved_ids = [resolved for _requested, resolved in resolved_pairs]

    result: dict[str, list[MessageRecord]] = {cid: [] for cid in session_ids}
    result.update({cid: [] for cid in resolved_ids})
    all_messages: list[MessageRecord] = []
    placeholders = ",".join("?" for _ in resolved_ids)
    query = f"""
        SELECT {_MESSAGE_RECORD_SELECT}
        FROM messages m
        JOIN sessions s ON s.session_id = m.session_id
        WHERE m.session_id IN ({placeholders})
    """
    params: list[str | float] = list(resolved_ids)

    role_values = message_role_sql_values(message_role)
    if role_values:
        role_placeholders = ",".join("?" for _ in role_values)
        query += f" AND m.role IN ({role_placeholders})"
        params.extend(role_values)

    if sort_key_since is not None:
        query += " AND m.occurred_at_ms >= ?"
        params.append(sort_key_since * 1000.0)

    if sort_key_until is not None:
        query += " AND m.occurred_at_ms <= ?"
        params.append(sort_key_until * 1000.0)

    query += " ORDER BY (m.occurred_at_ms IS NULL), m.occurred_at_ms, m.message_id"
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
        for requested, resolved in resolved_pairs:
            if requested != cid and resolved == cid and requested in result:
                result[requested].append(msg)
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
    session_id = await _resolve_session_id(conn, session_id)
    query = f"""
        SELECT {_MESSAGE_RECORD_SELECT}
        FROM messages m
        JOIN sessions s ON s.session_id = m.session_id
        WHERE m.session_id = ?
    """
    count_query = "SELECT COUNT(*) FROM messages WHERE session_id = ?"
    params: list[str | int] = [session_id]

    role_values = message_role_sql_values(message_role)
    if role_values:
        placeholders = ",".join("?" for _ in role_values)
        query += f" AND m.role IN ({placeholders})"
        count_query += f" AND role IN ({placeholders})"
        params.extend(role_values)

    if message_type:
        normalized_type = validate_message_type_filter(message_type).value
        query += " AND m.message_type = ?"
        count_query += " AND message_type = ?"
        params.append(normalized_type)

    # Get total count before pagination
    count_cursor = await conn.execute(count_query, tuple(params))
    count_row = await count_cursor.fetchone()
    total = count_row[0] if count_row else 0

    query += " ORDER BY (m.occurred_at_ms IS NULL), m.occurred_at_ms, m.message_id"
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
    session_id = await _resolve_session_id(conn, session_id)
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
        query = f"""
            SELECT {_MESSAGE_RECORD_SELECT}
            FROM messages m
            JOIN sessions s ON s.session_id = m.session_id
            WHERE m.session_id = ?
        """
        params: list[str | float] = [session_id]

        if role_values:
            placeholders = ",".join("?" for _ in role_values)
            query += f" AND m.role IN ({placeholders})"
            params.extend(role_values)

        if have_cursor:
            if last_sort is not None:
                # Cursor in the non-NULL sort_key group: advance within it and
                # always include the NULL group that follows in the ordering.
                query += (
                    " AND ("
                    "(m.occurred_at_ms IS NOT NULL"
                    " AND (m.occurred_at_ms > ? OR (m.occurred_at_ms = ? AND m.message_id > ?)))"
                    " OR m.occurred_at_ms IS NULL"
                    ")"
                )
                last_sort_ms = last_sort * 1000.0
                params.extend([last_sort_ms, last_sort_ms, last_id])
            else:
                # Cursor in the NULL sort_key group (ordering tail): only
                # NULL-sort_key rows with a greater message_id remain.
                query += " AND m.occurred_at_ms IS NULL AND m.message_id > ?"
                params.append(last_id)

        query += " ORDER BY (m.occurred_at_ms IS NULL), m.occurred_at_ms, m.message_id"

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
