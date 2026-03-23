"""Statistics and analytics queries."""

from __future__ import annotations

import aiosqlite

from polylogue.storage.store import MessageRecord

__all__ = [
    "aggregate_message_stats",
    "upsert_conversation_stats",
    "get_stats_by",
    "get_provider_conversation_counts",
    "get_provider_metrics_rows",
]


async def aggregate_message_stats(
    conn: aiosqlite.Connection,
    conversation_ids: list[str] | None = None,
) -> dict[str, int]:
    """Compute aggregate message statistics via SQL."""
    if conversation_ids is not None:
        await conn.execute("CREATE TEMP TABLE IF NOT EXISTS _stat_ids (cid TEXT PRIMARY KEY)")
        await conn.execute("DELETE FROM _stat_ids")
        await conn.executemany(
            "INSERT OR IGNORE INTO _stat_ids (cid) VALUES (?)",
            [(cid,) for cid in conversation_ids],
        )

        msg_row = await (await conn.execute("""
            SELECT SUM(cnt) FROM (
                SELECT COUNT(*) as cnt FROM messages
                WHERE conversation_id IN (SELECT cid FROM _stat_ids)
                GROUP BY conversation_id
            )
        """)).fetchone()
        msg_total = msg_row[0] or 0

        date_row = await (await conn.execute("""
            SELECT MIN(sort_key) as min_sk, MAX(sort_key) as max_sk
            FROM conversations WHERE conversation_id IN (SELECT cid FROM _stat_ids)
        """)).fetchone()

        prov_rows = await (await conn.execute("""
            SELECT provider_name, COUNT(*) as cnt
            FROM conversations WHERE conversation_id IN (SELECT cid FROM _stat_ids)
            GROUP BY provider_name ORDER BY cnt DESC
        """)).fetchall()
        providers = {r["provider_name"]: r["cnt"] for r in prov_rows}

        att_row = await (await conn.execute("""
            SELECT COUNT(*) as cnt FROM attachment_refs
            WHERE conversation_id IN (SELECT cid FROM _stat_ids)
        """)).fetchone()

        await conn.execute("DROP TABLE IF EXISTS _stat_ids")

        return {
            "total": msg_total,
            "user": 0,
            "assistant": 0,
            "system": 0,
            "words_approx": (
                (
                    await (
                        await conn.execute(
                            """
                            SELECT SUM(word_count) as words_approx
                            FROM conversation_stats
                            WHERE conversation_id IN (SELECT cid FROM _stat_ids)
                            """
                        )
                    ).fetchone()
                )["words_approx"]
                or 0
            ),
            "attachments": att_row["cnt"] or 0,
            "min_sort_key": date_row["min_sk"],
            "max_sort_key": date_row["max_sk"],
            "providers": providers,
        }

    # Unfiltered path
    msg_total = (await (await conn.execute("SELECT COUNT(*) FROM messages")).fetchone())[0] or 0

    date_row = await (await conn.execute(
        "SELECT MIN(sort_key) as min_sk, MAX(sort_key) as max_sk FROM conversations"
    )).fetchone()

    prov_rows = await (await conn.execute(
        "SELECT provider_name, COUNT(*) as cnt FROM conversations GROUP BY provider_name ORDER BY cnt DESC"
    )).fetchall()
    providers = {r["provider_name"]: r["cnt"] for r in prov_rows}

    att_cnt = (await (await conn.execute("SELECT COUNT(*) FROM attachment_refs")).fetchone())[0] or 0

    return {
        "total": msg_total,
        "user": 0,
        "assistant": 0,
        "system": 0,
        "words_approx": (
            (
                await (
                    await conn.execute("SELECT SUM(word_count) as words_approx FROM conversation_stats")
                ).fetchone()
            )["words_approx"]
            or 0
        ),
        "attachments": att_cnt,
        "min_sort_key": date_row["min_sk"],
        "max_sort_key": date_row["max_sk"],
        "providers": providers,
    }


async def upsert_conversation_stats(
    conn: aiosqlite.Connection,
    conversation_id: str,
    provider_name: str,
    messages: list[MessageRecord],
    transaction_depth: int,
) -> None:
    """Upsert precomputed per-conversation aggregate stats."""
    message_count = len(messages)
    word_count = sum(m.word_count for m in messages)
    tool_use_count = sum(1 for m in messages if m.has_tool_use)
    thinking_count = sum(1 for m in messages if m.has_thinking)
    await conn.execute(
        """
        INSERT INTO conversation_stats
            (conversation_id, provider_name, message_count, word_count, tool_use_count, thinking_count)
        VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(conversation_id) DO UPDATE SET
            provider_name  = excluded.provider_name,
            message_count  = excluded.message_count,
            word_count     = excluded.word_count,
            tool_use_count = excluded.tool_use_count,
            thinking_count = excluded.thinking_count
        """,
        (conversation_id, provider_name, message_count, word_count, tool_use_count, thinking_count),
    )
    if transaction_depth == 0:
        await conn.commit()


async def get_stats_by(
    conn: aiosqlite.Connection, group_by: str = "provider"
) -> dict[str, int]:
    """Get conversation counts grouped by provider, month, or year."""
    if group_by == "month":
        cursor = await conn.execute(
            """
            SELECT strftime('%Y-%m', updated_at) as period, COUNT(*) as count
            FROM conversations
            WHERE updated_at IS NOT NULL
            GROUP BY period ORDER BY period DESC
            """
        )
    elif group_by == "year":
        cursor = await conn.execute(
            """
            SELECT strftime('%Y', updated_at) as period, COUNT(*) as count
            FROM conversations
            WHERE updated_at IS NOT NULL
            GROUP BY period ORDER BY period DESC
            """
        )
    else:
        cursor = await conn.execute(
            """
            SELECT provider_name as period, COUNT(*) as count
            FROM conversations
            GROUP BY provider_name ORDER BY count DESC
            """
        )
    rows = await cursor.fetchall()
    return {row["period"]: row["count"] for row in rows}


async def get_provider_conversation_counts(
    conn: aiosqlite.Connection,
) -> list[dict[str, object]]:
    """Return conversation counts per provider — fast, conversations-table-only query."""
    cursor = await conn.execute(
        """
        SELECT provider_name, COUNT(*) AS conversation_count
        FROM conversations
        GROUP BY provider_name
        ORDER BY conversation_count DESC
        """
    )
    rows = await cursor.fetchall()
    return [dict(row) for row in rows]


async def get_provider_metrics_rows(
    conn: aiosqlite.Connection,
) -> list[dict[str, object]]:
    """Return raw provider aggregation rows for analytics reporting."""
    cursor = await conn.execute(
        """
        SELECT
            COALESCE(NULLIF(m.provider_name, ''), c.provider_name, 'unknown')              AS provider_name,
            COUNT(DISTINCT m.conversation_id)                                              AS conversation_count,
            COUNT(*)                                                                       AS message_count,
            SUM(CASE WHEN role = 'user'      THEN 1 ELSE 0 END)                           AS user_message_count,
            SUM(CASE WHEN role = 'assistant' THEN 1 ELSE 0 END)                           AS assistant_message_count,
            SUM(CASE WHEN role = 'user'      THEN word_count ELSE 0 END)                  AS user_word_sum,
            SUM(CASE WHEN role = 'assistant' THEN word_count ELSE 0 END)                  AS assistant_word_sum,
            SUM(has_tool_use)                                                              AS tool_use_count,
            SUM(has_thinking)                                                              AS thinking_count,
            COUNT(DISTINCT CASE WHEN has_tool_use = 1 THEN m.conversation_id END)         AS conversations_with_tools,
            COUNT(DISTINCT CASE WHEN has_thinking = 1 THEN m.conversation_id END)         AS conversations_with_thinking
        FROM messages m
        LEFT JOIN conversations c ON c.conversation_id = m.conversation_id
        GROUP BY COALESCE(NULLIF(m.provider_name, ''), c.provider_name, 'unknown')
        ORDER BY conversation_count DESC
        """
    )
    rows = await cursor.fetchall()
    return [dict(row) for row in rows]
