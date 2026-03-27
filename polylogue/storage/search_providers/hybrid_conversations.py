"""Conversation-resolution helpers for hybrid search."""

from __future__ import annotations

from sqlite3 import Connection

from polylogue.storage.backends.connection import _build_provider_scope_filter


def _resolve_ranked_conversation_ids(
    conn: Connection,
    *,
    message_results: list[tuple[str, float]],
    limit: int,
    scope_names: list[str] | None,
) -> list[str]:
    """Resolve ranked message hits into unique conversation IDs in SQL."""
    if not message_results or limit <= 0:
        return []

    values_sql = ", ".join("(?, ?)" for _ in message_results)
    params: list[object] = []
    for rank, (message_id, _score) in enumerate(message_results, start=1):
        params.extend((message_id, rank))

    scope_clause = ""
    if scope_names:
        scope_sql, scope_params = _build_provider_scope_filter(
            scope_names,
            provider_column="conversations.provider_name",
        )
        scope_clause = f"WHERE {scope_sql}"
        params.extend(scope_params)

    params.append(limit)
    rows = conn.execute(
        f"""
        WITH ranked_messages(message_id, message_rank) AS (
            VALUES {values_sql}
        ),
        candidate_hits AS (
            SELECT
                messages.conversation_id,
                ranked_messages.message_rank
            FROM ranked_messages
            JOIN messages ON messages.message_id = ranked_messages.message_id
            JOIN conversations ON conversations.conversation_id = messages.conversation_id
            {scope_clause}
        ),
        ranked_conversations AS (
            SELECT
                conversation_id,
                message_rank,
                ROW_NUMBER() OVER (
                    PARTITION BY conversation_id
                    ORDER BY message_rank ASC, conversation_id ASC
                ) AS conversation_rank
            FROM candidate_hits
        )
        SELECT conversation_id
        FROM ranked_conversations
        WHERE conversation_rank = 1
        ORDER BY message_rank ASC, conversation_id ASC
        LIMIT ?
        """,
        tuple(params),
    ).fetchall()
    return [row["conversation_id"] for row in rows]


__all__ = ["_resolve_ranked_conversation_ids"]
