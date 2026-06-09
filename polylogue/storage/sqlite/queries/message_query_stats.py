"""Stats queries for messages."""

from __future__ import annotations

import aiosqlite

from polylogue.archive.message.roles import Role, message_role_count_key, message_role_sql_values


def _role_count(role_counts: dict[str, int], role: Role) -> int:
    return sum(role_counts.get(value, 0) for value in message_role_sql_values((role,)))


async def get_session_stats(conn: aiosqlite.Connection, session_id: str) -> dict[str, int]:
    cursor = await conn.execute(
        "SELECT COUNT(*) as cnt FROM messages WHERE session_id = ?",
        (session_id,),
    )
    row = await cursor.fetchone()
    total = int(row["cnt"]) if row else 0

    cursor = await conn.execute(
        "SELECT COUNT(*) as cnt FROM messages WHERE session_id = ? AND role IN ('user', 'assistant', 'human')",
        (session_id,),
    )
    row = await cursor.fetchone()
    dialogue = int(row["cnt"]) if row else 0

    cursor = await conn.execute(
        """
        SELECT role, COUNT(*) as cnt
        FROM messages
        WHERE session_id = ?
        GROUP BY role
        """,
        (session_id,),
    )
    role_counts = {str(row["role"]): int(row["cnt"]) for row in await cursor.fetchall()}
    canonical_role_counts = {message_role_count_key(role): _role_count(role_counts, role) for role in Role}

    return {
        "total_messages": total,
        "dialogue_messages": dialogue,
        "tool_messages": total - dialogue,
        **canonical_role_counts,
    }


async def get_message_counts_batch(conn: aiosqlite.Connection, session_ids: list[str]) -> dict[str, int]:
    if not session_ids:
        return {}
    placeholders = ",".join("?" for _ in session_ids)
    cursor = await conn.execute(
        f"""
        SELECT session_id, COUNT(*) as cnt
        FROM messages
        WHERE session_id IN ({placeholders})
        GROUP BY session_id
        """,
        session_ids,
    )
    rows = await cursor.fetchall()
    return {row["session_id"]: row["cnt"] for row in rows}


__all__ = ["get_session_stats", "get_message_counts_batch"]
