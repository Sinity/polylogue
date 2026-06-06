"""Session-resolution helpers for hybrid search."""

from __future__ import annotations

from sqlite3 import Connection

from polylogue.storage.search.models import SessionSearchResult
from polylogue.storage.sqlite.connection import _build_provider_scope_filter


def _resolve_ranked_session_hits(
    conn: Connection,
    *,
    message_results: list[tuple[str, float]],
    limit: int,
    scope_names: list[str] | None,
) -> SessionSearchResult:
    """Resolve ranked message hits into unique session IDs in SQL."""
    if not message_results or limit <= 0:
        return SessionSearchResult(hits=[])

    values_sql = ", ".join("(?, ?)" for _ in message_results)
    params: list[object] = []
    for rank, (message_id, _score) in enumerate(message_results, start=1):
        params.extend((message_id, rank))

    scope_clause = ""
    if scope_names:
        scope_sql, scope_params = _build_provider_scope_filter(
            scope_names,
            provider_column="sessions.source_name",
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
                messages.session_id,
                ranked_messages.message_rank
            FROM ranked_messages
            JOIN messages ON messages.message_id = ranked_messages.message_id
            JOIN sessions ON sessions.session_id = messages.session_id
            {scope_clause}
        ),
        ranked_sessions AS (
            SELECT
                session_id,
                message_rank,
                ROW_NUMBER() OVER (
                    PARTITION BY session_id
                    ORDER BY message_rank ASC, session_id ASC
                ) AS session_rank
            FROM candidate_hits
        )
        SELECT session_id
        FROM ranked_sessions
        WHERE session_rank = 1
        ORDER BY message_rank ASC, session_id ASC
        LIMIT ?
        """,
        tuple(params),
    ).fetchall()
    return SessionSearchResult.from_ids([row["session_id"] for row in rows])


def _resolve_ranked_session_ids(
    conn: Connection,
    *,
    message_results: list[tuple[str, float]],
    limit: int,
    scope_names: list[str] | None,
) -> list[str]:
    return _resolve_ranked_session_hits(
        conn,
        message_results=message_results,
        limit=limit,
        scope_names=scope_names,
    ).session_ids()


__all__ = ["_resolve_ranked_session_hits", "_resolve_ranked_session_ids"]
