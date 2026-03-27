"""Content-block query helpers."""

from __future__ import annotations

import aiosqlite

from polylogue.storage.backends.queries.mappers import _row_to_content_block
from polylogue.storage.store import ContentBlockRecord


async def get_content_blocks(
    conn: aiosqlite.Connection,
    message_ids: list[str],
) -> dict[str, list[ContentBlockRecord]]:
    """Get content blocks for a list of message IDs."""
    if not message_ids:
        return {}
    result: dict[str, list[ContentBlockRecord]] = {mid: [] for mid in message_ids}
    batch_size = 900
    for index in range(0, len(message_ids), batch_size):
        batch = message_ids[index : index + batch_size]
        placeholders = ",".join("?" for _ in batch)
        cursor = await conn.execute(
            f"SELECT * FROM content_blocks WHERE message_id IN ({placeholders}) ORDER BY message_id, block_index",
            batch,
        )
        rows = await cursor.fetchall()
        for row in rows:
            mid = row["message_id"]
            if mid in result:
                result[mid].append(_row_to_content_block(row))
    return result


async def save_content_blocks(
    conn: aiosqlite.Connection,
    records: list[ContentBlockRecord],
    transaction_depth: int,
) -> None:
    """Persist content block records using bulk insert."""
    if not records:
        return
    query = """
        INSERT INTO content_blocks (
            block_id,
            message_id,
            conversation_id,
            block_index,
            type,
            text,
            tool_name,
            tool_id,
            tool_input,
            media_type,
            metadata,
            semantic_type
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(message_id, block_index) DO UPDATE SET
            type = excluded.type,
            text = excluded.text,
            tool_name = excluded.tool_name,
            tool_id = excluded.tool_id,
            tool_input = excluded.tool_input,
            media_type = excluded.media_type,
            metadata = excluded.metadata,
            semantic_type = excluded.semantic_type
    """
    await conn.executemany(
        query,
        [
            (
                record.block_id,
                record.message_id,
                record.conversation_id,
                record.block_index,
                record.type,
                record.text,
                record.tool_name,
                record.tool_id,
                record.tool_input,
                record.media_type,
                record.metadata,
                record.semantic_type,
            )
            for record in records
        ],
    )
    if transaction_depth == 0:
        await conn.commit()


__all__ = [
    "get_content_blocks",
    "save_content_blocks",
]
