"""Stats queries for messages."""

from __future__ import annotations

import aiosqlite


async def get_conversation_stats(conn: aiosqlite.Connection, conversation_id: str) -> dict[str, int]:
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


async def get_message_counts_batch(conn: aiosqlite.Connection, conversation_ids: list[str]) -> dict[str, int]:
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


__all__ = ["get_conversation_stats", "get_message_counts_batch"]
