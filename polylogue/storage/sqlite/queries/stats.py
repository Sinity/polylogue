"""Statistics and analytics queries."""

from __future__ import annotations

import aiosqlite
from typing_extensions import TypedDict

from polylogue.core.common import SQL_STATS_UPSERT as _STATS_UPSERT_SQL
from polylogue.storage.runtime import MessageRecord


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


class ProviderSessionCountRow(TypedDict):
    source_name: str
    session_count: int


class ProviderMetricsRow(TypedDict):
    source_name: str
    session_count: int
    message_count: int
    user_message_count: int
    assistant_message_count: int
    user_word_sum: int
    assistant_word_sum: int
    tool_use_count: int
    thinking_count: int
    sessions_with_tools: int
    sessions_with_thinking: int


__all__ = [
    "AggregateMessageStats",
    "ProviderSessionCountRow",
    "ProviderMetricsRow",
    "aggregate_message_stats",
    "upsert_session_stats",
    "get_stats_by",
    "get_provider_session_counts",
    "get_provider_metrics_rows",
]


async def aggregate_message_stats(
    conn: aiosqlite.Connection,
    session_ids: list[str] | None = None,
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

    if session_ids is not None:
        # #1659: read profile is query_only=ON which rejects TEMP TABLE writes,
        # so use inline IN-clause binding instead of a temp-table join.
        if not session_ids:
            return {
                "total": 0,
                "user": 0,
                "assistant": 0,
                "system": 0,
                "words_approx": 0,
                "attachment_refs": 0,
                "distinct_attachments": 0,
                "min_sort_key": None,
                "max_sort_key": None,
                "providers": {},
            }
        placeholders = ",".join("?" * len(session_ids))
        cid_params = tuple(session_ids)

        message_stats = await _message_aggregate(f"WHERE session_id IN ({placeholders})", cid_params)

        date_row = await (
            await conn.execute(
                f"SELECT MIN(sort_key) as min_sk, MAX(sort_key) as max_sk FROM sessions "
                f"WHERE session_id IN ({placeholders})",
                cid_params,
            )
        ).fetchone()

        prov_rows = await (
            await conn.execute(
                f"SELECT source_name, COUNT(*) as cnt FROM sessions "
                f"WHERE session_id IN ({placeholders}) "
                "GROUP BY source_name ORDER BY cnt DESC",
                cid_params,
            )
        ).fetchall()
        providers = {str(r["source_name"]): _row_int(r, "cnt") for r in prov_rows}

        att_row = await (
            await conn.execute(
                f"SELECT COUNT(*) AS attachment_ref_count, COUNT(DISTINCT attachment_id) "
                f"AS distinct_attachment_count FROM attachment_refs "
                f"WHERE session_id IN ({placeholders})",
                cid_params,
            )
        ).fetchone()

        return {
            **message_stats,
            "attachment_refs": _row_int(att_row, "attachment_ref_count"),
            "distinct_attachments": _row_int(att_row, "distinct_attachment_count"),
            "min_sort_key": _row_float(date_row, "min_sk"),
            "max_sort_key": _row_float(date_row, "max_sk"),
            "providers": providers,
        }

    # Unfiltered path — #1678: total/words moved from a full messages scan
    # to session_stats (pre-aggregated, ~1 row per session).
    # The role-level splits still scan messages, but the query is index-only
    # (role is in idx_messages_provider_stats) — no table access for word_count.
    # Provider counts also move to session_stats; this is consistent with
    # get_provider_session_counts which queries sessions directly
    # because zero-message sessions have no stats row.
    stats_row = await (
        await conn.execute(
            """
            SELECT
                COALESCE(SUM(message_count), 0)  AS total_msgs,
                COALESCE(SUM(word_count), 0)     AS total_words
            FROM session_stats
            """
        )
    ).fetchone()
    total_msgs = _row_int(stats_row, "total_msgs")
    words_approx = _row_int(stats_row, "total_words")

    role_row = await (
        await conn.execute(
            """
            SELECT
                SUM(CASE WHEN role = 'user'      THEN 1 ELSE 0 END) AS user_count,
                SUM(CASE WHEN role = 'assistant' THEN 1 ELSE 0 END) AS assistant_count,
                SUM(CASE WHEN role = 'system'    THEN 1 ELSE 0 END) AS system_count
            FROM messages
            """
        )
    ).fetchone()
    unfiltered_message_stats: _MessageAggregate = {
        "total": total_msgs,
        "user": _row_int(role_row, "user_count"),
        "assistant": _row_int(role_row, "assistant_count"),
        "system": _row_int(role_row, "system_count"),
        "words_approx": words_approx,
    }

    date_row = await (
        await conn.execute("SELECT MIN(sort_key) as min_sk, MAX(sort_key) as max_sk FROM sessions")
    ).fetchone()

    prov_rows = await (
        await conn.execute(
            "SELECT source_name, COUNT(*) as cnt FROM session_stats GROUP BY source_name ORDER BY cnt DESC"
        )
    ).fetchall()
    providers = {str(r["source_name"]): _row_int(r, "cnt") for r in prov_rows}

    att_row = await (
        await conn.execute("""
        SELECT
            COUNT(*) AS attachment_ref_count,
            COUNT(DISTINCT attachment_id) AS distinct_attachment_count
        FROM attachment_refs
        """)
    ).fetchone()

    return {
        **unfiltered_message_stats,
        "attachment_refs": _row_int(att_row, "attachment_ref_count"),
        "distinct_attachments": _row_int(att_row, "distinct_attachment_count"),
        "min_sort_key": _row_float(date_row, "min_sk"),
        "max_sort_key": _row_float(date_row, "max_sk"),
        "providers": providers,
    }


async def upsert_session_stats(
    conn: aiosqlite.Connection,
    session_id: str,
    source_name: str,
    messages: list[MessageRecord],
    transaction_depth: int,
) -> None:
    """Upsert precomputed per-session aggregate stats including per-role counts."""
    message_count = len(messages)
    word_count = sum(m.word_count for m in messages)
    tool_use_count = sum(1 for m in messages if m.has_tool_use)
    thinking_count = sum(1 for m in messages if m.has_thinking)
    paste_count = sum(1 for m in messages if m.has_paste)
    from polylogue.archive.message.roles import Role

    user_msg_count = sum(1 for m in messages if m.role == Role.USER)
    assistant_msg_count = sum(1 for m in messages if m.role == Role.ASSISTANT)
    system_msg_count = sum(1 for m in messages if m.role == Role.SYSTEM)
    tool_msg_count = sum(1 for m in messages if m.role == Role.TOOL)
    user_word_count = sum(m.word_count for m in messages if m.role == Role.USER)
    assistant_word_count = sum(m.word_count for m in messages if m.role == Role.ASSISTANT)
    await conn.execute(
        _STATS_UPSERT_SQL,
        (
            session_id,
            source_name,
            message_count,
            word_count,
            tool_use_count,
            thinking_count,
            paste_count,
            user_msg_count,
            assistant_msg_count,
            system_msg_count,
            tool_msg_count,
            user_word_count,
            assistant_word_count,
        ),
    )
    if transaction_depth == 0:
        await conn.commit()


async def get_stats_by(conn: aiosqlite.Connection, group_by: str = "provider") -> dict[str, int]:
    """Get session counts grouped by provider, day, month, or year.

    Raises ValueError on unknown ``group_by`` rather than silently returning
    provider counts. Each branch is a literal SQL constant — the validated
    input never reaches string interpolation — but the explicit reject closes
    the door on future branches that might.

    These are the sessions-table calendar/provider dimensions. The CLI's
    additional ``action``/``tool``/``repo``/``work-kind`` dimensions are not
    plain session counts — they are computed via insight-summary
    aggregation paths in ``cli/query.py:_handle_stats_by`` and are not exposed
    through this substrate count (#1749).
    """
    if group_by == "day":
        cursor = await conn.execute(
            """
            SELECT strftime('%Y-%m-%d', updated_at) as period, COUNT(*) as count
            FROM sessions
            WHERE updated_at IS NOT NULL
            GROUP BY period ORDER BY period DESC
            """
        )
    elif group_by == "month":
        cursor = await conn.execute(
            """
            SELECT strftime('%Y-%m', updated_at) as period, COUNT(*) as count
            FROM sessions
            WHERE updated_at IS NOT NULL
            GROUP BY period ORDER BY period DESC
            """
        )
    elif group_by == "year":
        cursor = await conn.execute(
            """
            SELECT strftime('%Y', updated_at) as period, COUNT(*) as count
            FROM sessions
            WHERE updated_at IS NOT NULL
            GROUP BY period ORDER BY period DESC
            """
        )
    elif group_by == "provider":
        cursor = await conn.execute(
            """
            SELECT source_name as period, COUNT(*) as count
            FROM sessions
            GROUP BY source_name ORDER BY count DESC
            """
        )
    else:
        raise ValueError(f"Unknown group_by {group_by!r}; expected one of: provider, day, month, year")
    rows = await cursor.fetchall()
    return {row["period"]: row["count"] for row in rows}


async def get_provider_session_counts(
    conn: aiosqlite.Connection,
) -> list[ProviderSessionCountRow]:
    """Return session counts per provider — fast, sessions-table-only query."""
    cursor = await conn.execute(
        """
        SELECT source_name, COUNT(*) AS session_count
        FROM sessions
        GROUP BY source_name
        ORDER BY session_count DESC
        """
    )
    rows = await cursor.fetchall()
    return [
        {
            "source_name": str(row["source_name"] or "unknown"),
            "session_count": int(row["session_count"] or 0),
        }
        for row in rows
    ]


async def get_provider_metrics_rows(
    conn: aiosqlite.Connection,
) -> list[ProviderMetricsRow]:
    """Return raw provider aggregation rows for analytics reporting.

    All aggregates come from ``session_stats`` — one row per
    session, far smaller than ``messages``. As of schema v21,
    per-role message counts and word sums are also pre-aggregated,
    eliminating the former messages-table scan (#1314 companion).
    """
    stats_cursor = await conn.execute(
        """
        SELECT
            COALESCE(NULLIF(cs.source_name, ''), 'unknown')        AS source_name,
            COUNT(*)                                                  AS session_count,
            COALESCE(SUM(cs.message_count), 0)                        AS message_count,
            COALESCE(SUM(cs.tool_use_count), 0)                       AS tool_use_count,
            COALESCE(SUM(cs.thinking_count), 0)                       AS thinking_count,
            SUM(CASE WHEN cs.tool_use_count > 0 THEN 1 ELSE 0 END)    AS sessions_with_tools,
            SUM(CASE WHEN cs.thinking_count > 0 THEN 1 ELSE 0 END)    AS sessions_with_thinking,
            COALESCE(SUM(cs.user_msg_count), 0)                       AS user_message_count,
            COALESCE(SUM(cs.assistant_msg_count), 0)                  AS assistant_message_count,
            COALESCE(SUM(cs.user_word_count), 0)                      AS user_word_sum,
            COALESCE(SUM(cs.assistant_word_count), 0)                 AS assistant_word_sum
        FROM session_stats cs
        GROUP BY COALESCE(NULLIF(cs.source_name, ''), 'unknown')
        """
    )
    stats_rows = await stats_cursor.fetchall()

    merged: list[ProviderMetricsRow] = []
    for row in stats_rows:
        provider = str(row["source_name"] or "unknown")
        role_split = {
            "user_message_count": int(row["user_message_count"] or 0),
            "assistant_message_count": int(row["assistant_message_count"] or 0),
            "user_word_sum": int(row["user_word_sum"] or 0),
            "assistant_word_sum": int(row["assistant_word_sum"] or 0),
        }
        merged.append(
            {
                "source_name": provider,
                "session_count": int(row["session_count"] or 0),
                "message_count": int(row["message_count"] or 0),
                "user_message_count": role_split["user_message_count"],
                "assistant_message_count": role_split["assistant_message_count"],
                "user_word_sum": role_split["user_word_sum"],
                "assistant_word_sum": role_split["assistant_word_sum"],
                "tool_use_count": int(row["tool_use_count"] or 0),
                "thinking_count": int(row["thinking_count"] or 0),
                "sessions_with_tools": int(row["sessions_with_tools"] or 0),
                "sessions_with_thinking": int(row["sessions_with_thinking"] or 0),
            }
        )
    merged.sort(key=lambda item: item["session_count"], reverse=True)
    return merged
