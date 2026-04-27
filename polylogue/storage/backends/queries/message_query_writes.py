"""Write queries for messages."""

from __future__ import annotations

import aiosqlite

from polylogue.storage.runtime import MessageRecord


def topo_sort_messages(records: list[MessageRecord]) -> list[MessageRecord]:
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


async def save_messages(
    conn: aiosqlite.Connection,
    records: list[MessageRecord],
    transaction_depth: int,
) -> None:
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


__all__ = ["save_messages", "topo_sort_messages"]
