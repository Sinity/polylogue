"""Block query helpers."""

from __future__ import annotations

import aiosqlite

from polylogue.storage.runtime import BlockRecord
from polylogue.storage.sqlite.queries.mappers import _row_to_content_block


async def get_blocks(
    conn: aiosqlite.Connection,
    message_ids: list[str],
) -> dict[str, list[BlockRecord]]:
    """Get blocks for a list of message IDs."""
    if not message_ids:
        return {}
    result: dict[str, list[BlockRecord]] = {mid: [] for mid in message_ids}
    batch_size = 900
    table_query = """
        SELECT
            block_id,
            message_id,
            session_id,
            position AS block_index,
            block_type AS type,
            text,
            tool_name,
            tool_id,
            tool_input,
            NULL AS metadata,
            semantic_type
        FROM blocks
        WHERE message_id IN ({placeholders})
        ORDER BY message_id, position
    """
    for index in range(0, len(message_ids), batch_size):
        batch = message_ids[index : index + batch_size]
        placeholders = ",".join("?" for _ in batch)
        cursor = await conn.execute(
            table_query.format(placeholders=placeholders),
            batch,
        )
        rows = await cursor.fetchall()
        for row in rows:
            mid = row["message_id"]
            if mid in result:
                result[mid].append(_row_to_content_block(row))
    return result


__all__ = [
    "get_blocks",
]
