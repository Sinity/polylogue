"""Message CRUD queries and topological sort."""

from __future__ import annotations

from collections.abc import AsyncIterator

import aiosqlite

from polylogue.storage.backends.queries.mappers import _row_to_message
from polylogue.storage.store import MessageRecord

__all__ = [
    "topo_sort_messages",
    "get_messages",
    "get_messages_batch",
    "save_messages",
    "iter_messages",
    "get_conversation_stats",
    "get_message_counts_batch",
]


def topo_sort_messages(records: list[MessageRecord]) -> list[MessageRecord]:
    """Sort messages so parents come before children (for FK constraint).

    Cross-conversation parent references (parent outside this batch) are left
    as-is — those FKs are never set by prepare.py anyway (only within-conversation
    parent_message_id is resolved).
    """
    ids_in_batch = {r.message_id for r in records}
    no_parent: list[MessageRecord] = []
    has_parent: list[MessageRecord] = []
    for r in records:
        if r.parent_message_id and r.parent_message_id in ids_in_batch:
            has_parent.append(r)
        else:
            no_parent.append(r)
    if not has_parent:
        return records
    ordered: list[MessageRecord] = list(no_parent)
    inserted_ids = {r.message_id for r in ordered}
    remaining = list(has_parent)
    max_passes = len(remaining) + 1
    for _ in range(max_passes):
        if not remaining:
            break
        next_remaining: list[MessageRecord] = []
        for r in remaining:
            if r.parent_message_id in inserted_ids:
                ordered.append(r)
                inserted_ids.add(r.message_id)
            else:
                next_remaining.append(r)
        remaining = next_remaining
    ordered.extend(remaining)
    return ordered


async def get_messages(
    conn: aiosqlite.Connection,
    conversation_id: str,
) -> list[MessageRecord]:
    """Get all messages for a conversation (without content_blocks)."""
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
    """Get messages for multiple conversations.

    Returns (result dict by conv_id, flat all_messages list).
    """
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


async def save_messages(
    conn: aiosqlite.Connection,
    records: list[MessageRecord],
    transaction_depth: int,
) -> None:
    """Persist multiple message records using bulk insert."""
    if not records:
        return
    records = topo_sort_messages(records)
    query = """
        INSERT INTO messages (
            message_id,
            conversation_id,
            provider_message_id,
            role,
            text,
            sort_key,
            content_hash,
            version,
            parent_message_id,
            branch_index,
            provider_name,
            word_count,
            has_tool_use,
            has_thinking
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(message_id) DO UPDATE SET
            role = excluded.role,
            text = excluded.text,
            sort_key = excluded.sort_key,
            content_hash = excluded.content_hash,
            parent_message_id = excluded.parent_message_id,
            branch_index = excluded.branch_index,
            provider_name = excluded.provider_name,
            word_count = excluded.word_count,
            has_tool_use = excluded.has_tool_use,
            has_thinking = excluded.has_thinking
        WHERE
            content_hash != excluded.content_hash
            OR IFNULL(role, '') != IFNULL(excluded.role, '')
            OR IFNULL(text, '') != IFNULL(excluded.text, '')
            OR IFNULL(parent_message_id, '') != IFNULL(excluded.parent_message_id, '')
            OR branch_index != excluded.branch_index
    """
    data = [
        (
            r.message_id,
            r.conversation_id,
            r.provider_message_id,
            r.role,
            r.text,
            r.sort_key,
            r.content_hash,
            r.version,
            r.parent_message_id,
            r.branch_index,
            r.provider_name,
            r.word_count,
            r.has_tool_use,
            r.has_thinking,
        )
        for r in records
    ]
    await conn.executemany(query, data)
    if transaction_depth == 0:
        await conn.commit()


async def iter_messages(
    conn: aiosqlite.Connection,
    conversation_id: str,
    *,
    chunk_size: int = 100,
    dialogue_only: bool = False,
    limit: int | None = None,
) -> AsyncIterator[MessageRecord]:
    """Stream messages in chunks."""
    offset = 0
    yielded = 0

    while True:
        query = "SELECT * FROM messages WHERE conversation_id = ?"
        params: list[str | int] = [conversation_id]

        if dialogue_only:
            query += " AND role IN ('user', 'assistant', 'human')"

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
        rows = await cursor.fetchall()

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


async def get_conversation_stats(
    conn: aiosqlite.Connection, conversation_id: str
) -> dict[str, int]:
    """Get message counts without loading messages."""
    cursor = await conn.execute(
        "SELECT COUNT(*) as cnt FROM messages WHERE conversation_id = ?",
        (conversation_id,),
    )
    total = (await cursor.fetchone())["cnt"]

    cursor = await conn.execute(
        "SELECT COUNT(*) as cnt FROM messages WHERE conversation_id = ? AND role IN ('user', 'assistant', 'human')",
        (conversation_id,),
    )
    dialogue = (await cursor.fetchone())["cnt"]

    return {
        "total_messages": total,
        "dialogue_messages": dialogue,
        "tool_messages": total - dialogue,
    }


async def get_message_counts_batch(
    conn: aiosqlite.Connection, conversation_ids: list[str]
) -> dict[str, int]:
    """Get message counts for multiple conversations in a single query."""
    if not conversation_ids:
        return {}
    placeholders = ",".join("?" for _ in conversation_ids)
    cursor = await conn.execute(
        f"""
        SELECT conversation_id, COUNT(*) as cnt
        FROM messages
        WHERE conversation_id IN ({placeholders})
        GROUP BY conversation_id
        """,
        conversation_ids,
    )
    rows = await cursor.fetchall()
    return {row["conversation_id"]: row["cnt"] for row in rows}
