"""Statistics and analytics queries."""

from __future__ import annotations

import aiosqlite
from typing_extensions import TypedDict

from polylogue.storage.store import MessageRecord


class AggregateMessageStats(TypedDict):
    total: int
    user: int
    assistant: int
    system: int
    words_approx: int
    attachment_refs: int
    distinct_attachments: int
    min_sort_key: float | None
    max_sort_key: float | None
    providers: dict[str, int]


class _MessageAggregate(TypedDict):
    total: int
    user: int
    assistant: int
    system: int
    words_approx: int


class ProviderConversationCountRow(TypedDict):
    provider_name: str
    conversation_count: int


class ProviderMetricsRow(TypedDict):
    provider_name: str
    conversation_count: int
    message_count: int
    user_message_count: int
    assistant_message_count: int
    user_word_sum: int
    assistant_word_sum: int
    tool_use_count: int
    thinking_count: int
    conversations_with_tools: int
    conversations_with_thinking: int


__all__ = [
    "AggregateMessageStats",
    "ProviderConversationCountRow",
    "ProviderMetricsRow",
    "aggregate_message_stats",
    "upsert_conversation_stats",
    "get_stats_by",
    "get_provider_conversation_counts",
    "get_provider_metrics_rows",
]


async def aggregate_message_stats(
    conn: aiosqlite.Connection,
    conversation_ids: list[str] | None = None,
) -> AggregateMessageStats:
    """Compute aggregate message statistics via SQL."""

    def _row_int(row: aiosqlite.Row | None, key: int | str) -> int:
        if row is None:
            return 0
        try:
            return int(row[key])
        except (TypeError, ValueError):
            return 0

    def _row_float(row: aiosqlite.Row | None, key: int | str) -> float | None:
        if row is None:
            return None
        value = row[key]
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    async def _message_aggregate(where_clause: str = "", params: tuple[object, ...] = ()) -> _MessageAggregate:
        row = await (
            await conn.execute(
                f"""
                SELECT
                    COUNT(*) AS total,
                    SUM(CASE WHEN role = 'user' THEN 1 ELSE 0 END) AS user_count,
                    SUM(CASE WHEN role = 'assistant' THEN 1 ELSE 0 END) AS assistant_count,
                    SUM(CASE WHEN role = 'system' THEN 1 ELSE 0 END) AS system_count,
                    SUM(word_count) AS words_approx
                FROM messages
                {where_clause}
                """,
                params,
            )
        ).fetchone()
        return {
            "total": _row_int(row, "total"),
            "user": _row_int(row, "user_count"),
            "assistant": _row_int(row, "assistant_count"),
            "system": _row_int(row, "system_count"),
            "words_approx": _row_int(row, "words_approx"),
        }

    if conversation_ids is not None:
        await conn.execute("CREATE TEMP TABLE IF NOT EXISTS _stat_ids (cid TEXT PRIMARY KEY)")
        await conn.execute("DELETE FROM _stat_ids")
        await conn.executemany(
            "INSERT OR IGNORE INTO _stat_ids (cid) VALUES (?)",
            [(cid,) for cid in conversation_ids],
        )

        message_stats = await _message_aggregate("WHERE conversation_id IN (SELECT cid FROM _stat_ids)")

        date_row = await (
            await conn.execute("""
            SELECT MIN(sort_key) as min_sk, MAX(sort_key) as max_sk
            FROM conversations WHERE conversation_id IN (SELECT cid FROM _stat_ids)
        """)
        ).fetchone()

        prov_rows = await (
            await conn.execute("""
            SELECT provider_name, COUNT(*) as cnt
            FROM conversations WHERE conversation_id IN (SELECT cid FROM _stat_ids)
            GROUP BY provider_name ORDER BY cnt DESC
        """)
        ).fetchall()
        providers = {str(r["provider_name"]): _row_int(r, "cnt") for r in prov_rows}

        att_row = await (
            await conn.execute("""
            SELECT
                COUNT(*) AS attachment_ref_count,
                COUNT(DISTINCT attachment_id) AS distinct_attachment_count
            FROM attachment_refs
            WHERE conversation_id IN (SELECT cid FROM _stat_ids)
        """)
        ).fetchone()

        result: AggregateMessageStats = {
            **message_stats,
            "attachment_refs": _row_int(att_row, "attachment_ref_count"),
            "distinct_attachments": _row_int(att_row, "distinct_attachment_count"),
            "min_sort_key": _row_float(date_row, "min_sk"),
            "max_sort_key": _row_float(date_row, "max_sk"),
            "providers": providers,
        }
        await conn.execute("DROP TABLE IF EXISTS _stat_ids")
        return result

    # Unfiltered path
    message_stats = await _message_aggregate()

    date_row = await (
        await conn.execute("SELECT MIN(sort_key) as min_sk, MAX(sort_key) as max_sk FROM conversations")
    ).fetchone()

    prov_rows = await (
        await conn.execute(
            "SELECT provider_name, COUNT(*) as cnt FROM conversations GROUP BY provider_name ORDER BY cnt DESC"
        )
    ).fetchall()
    providers = {str(r["provider_name"]): _row_int(r, "cnt") for r in prov_rows}

    att_row = await (
        await conn.execute("""
        SELECT
            COUNT(*) AS attachment_ref_count,
            COUNT(DISTINCT attachment_id) AS distinct_attachment_count
        FROM attachment_refs
        """)
    ).fetchone()

    return {
        **message_stats,
        "attachment_refs": _row_int(att_row, "attachment_ref_count"),
        "distinct_attachments": _row_int(att_row, "distinct_attachment_count"),
        "min_sort_key": _row_float(date_row, "min_sk"),
        "max_sort_key": _row_float(date_row, "max_sk"),
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


async def get_stats_by(conn: aiosqlite.Connection, group_by: str = "provider") -> dict[str, int]:
    """Get conversation counts grouped by provider, month, or year.

    Raises ValueError on unknown ``group_by`` rather than silently returning
    provider counts. Each branch is a literal SQL constant — the validated
    input never reaches string interpolation — but the explicit reject closes
    the door on future branches that might.
    """
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
    elif group_by == "provider":
        cursor = await conn.execute(
            """
            SELECT provider_name as period, COUNT(*) as count
            FROM conversations
            GROUP BY provider_name ORDER BY count DESC
            """
        )
    else:
        raise ValueError(f"Unknown group_by {group_by!r}; expected one of: provider, month, year")
    rows = await cursor.fetchall()
    return {row["period"]: row["count"] for row in rows}


async def get_provider_conversation_counts(
    conn: aiosqlite.Connection,
) -> list[ProviderConversationCountRow]:
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
    return [
        {
            "provider_name": str(row["provider_name"] or "unknown"),
            "conversation_count": int(row["conversation_count"] or 0),
        }
        for row in rows
    ]


async def get_provider_metrics_rows(
    conn: aiosqlite.Connection,
) -> list[ProviderMetricsRow]:
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
    return [
        {
            "provider_name": str(row["provider_name"] or "unknown"),
            "conversation_count": int(row["conversation_count"] or 0),
            "message_count": int(row["message_count"] or 0),
            "user_message_count": int(row["user_message_count"] or 0),
            "assistant_message_count": int(row["assistant_message_count"] or 0),
            "user_word_sum": int(row["user_word_sum"] or 0),
            "assistant_word_sum": int(row["assistant_word_sum"] or 0),
            "tool_use_count": int(row["tool_use_count"] or 0),
            "thinking_count": int(row["thinking_count"] or 0),
            "conversations_with_tools": int(row["conversations_with_tools"] or 0),
            "conversations_with_thinking": int(row["conversations_with_thinking"] or 0),
        }
        for row in rows
    ]
