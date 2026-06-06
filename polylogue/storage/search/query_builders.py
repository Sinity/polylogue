"""Canonical ranked SQL builders for session and action search."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import TypeAlias
from urllib.parse import quote

from polylogue.storage.search.query_support import normalize_fts5_query
from polylogue.storage.sqlite.connection import _build_provider_scope_filter

SQLiteQueryParam: TypeAlias = str | int | float


@dataclass(frozen=True, slots=True)
class RankedSearchQuery:
    sql: str
    params: tuple[SQLiteQueryParam, ...]


@dataclass(frozen=True, slots=True)
class RankedSearchShape:
    fts_table: str
    candidate_columns: tuple[str, ...]
    final_select_sql: str


def session_web_url(session_id: str) -> str:
    """Return the daemon web reader URL path for a session."""
    return f"/?session={quote(session_id, safe='')}"


def build_ranked_session_search_query(
    *,
    query: str,
    limit: int,
    scope_names: Sequence[str] | None = None,
    since: str | None = None,
    include_snippet: bool = False,
) -> RankedSearchQuery | None:
    """Build the canonical ranked session search query."""
    fts_query = normalize_fts5_query(query)
    if fts_query is None:
        return None

    candidate_columns = [
        "messages.message_id",
        "messages.session_id",
        "sessions.source_name",
        "sessions.source_name",
        "sessions.title",
        "messages.sort_key",
        "bm25(messages_fts) AS relevance",
    ]
    if include_snippet:
        candidate_columns.append("snippet(messages_fts, 2, '[', ']', '…', 12) AS snippet")
    return _build_ranked_search_query(
        query=fts_query,
        limit=limit,
        scope_names=scope_names,
        since=since,
        since_column="messages.sort_key",
        shape=RankedSearchShape(
            fts_table="messages_fts",
            candidate_columns=tuple(candidate_columns),
            final_select_sql="""
        SELECT *
        FROM ranked_hits
        WHERE session_rank = 1
        ORDER BY relevance ASC, sort_key DESC, message_id ASC
        LIMIT ?
    """,
        ),
    )


def build_ranked_action_search_query(
    *,
    query: str,
    limit: int,
    scope_names: Sequence[str] | None = None,
    since: str | None = None,
    include_snippet: bool = False,
) -> RankedSearchQuery | None:
    """Build the canonical ranked action-search query."""
    fts_query = normalize_fts5_query(query)
    if fts_query is None:
        return None

    candidate_columns = [
        "action_events_fts.event_id",
        "action_events_fts.message_id",
        "action_events_fts.session_id",
        "action_events_fts.action_kind",
        "action_events_fts.normalized_tool_name AS tool_name",
        "sessions.source_name",
        "sessions.source_name",
        "sessions.title",
        "messages.sort_key",
        "bm25(action_events_fts) AS relevance",
    ]
    if include_snippet:
        candidate_columns.append("snippet(action_events_fts, 5, '[', ']', '…', 12) AS snippet")
    return _build_ranked_search_query(
        query=fts_query,
        limit=limit,
        scope_names=scope_names,
        since=since,
        since_column="messages.sort_key",
        shape=RankedSearchShape(
            fts_table="action_events_fts",
            candidate_columns=tuple(candidate_columns),
            final_select_sql="""
        SELECT
            session_id,
            message_id,
            source_name,
            source_name,
            title,
            sort_key,
            relevance
        FROM ranked_hits
        WHERE session_rank = 1
        ORDER BY relevance ASC, sort_key DESC, message_id ASC
        LIMIT ?
    """,
        ),
    )


def _parse_since_timestamp(since: str) -> float:
    try:
        return datetime.fromisoformat(since).timestamp()
    except ValueError as exc:
        raise ValueError(f"Invalid --since date '{since}': {exc}. Use ISO format (e.g., 2023-01-01)") from exc


def _build_ranked_search_query(
    *,
    query: str,
    limit: int,
    scope_names: Sequence[str] | None,
    since: str | None,
    since_column: str,
    shape: RankedSearchShape,
) -> RankedSearchQuery:
    if shape.fts_table == "messages_fts":
        from_join_sql = """
            FROM messages_fts
            JOIN messages ON messages.rowid = messages_fts.rowid
            JOIN sessions ON sessions.session_id = messages.session_id
        """
    else:
        from_join_sql = f"""
            FROM {shape.fts_table}
            JOIN sessions ON sessions.session_id = {shape.fts_table}.session_id
            JOIN messages ON messages.message_id = {shape.fts_table}.message_id
        """
    sql = f"""
        WITH candidate_hits AS (
            SELECT
                {", ".join(shape.candidate_columns)}
            {from_join_sql}
            WHERE {shape.fts_table} MATCH ?
    """
    params: list[SQLiteQueryParam] = [query]

    if scope_names:
        scope_sql, scope_params = _build_provider_scope_filter(
            scope_names,
            provider_column="sessions.source_name",
        )
        sql += f" AND {scope_sql}"
        params.extend(scope_params)

    if since:
        sql += f" AND {since_column} >= ?"
        params.append(_parse_since_timestamp(since))

    sql += """
        ),
        ranked_hits AS (
            SELECT
                *,
                ROW_NUMBER() OVER (
                    PARTITION BY session_id
                    ORDER BY relevance ASC, sort_key DESC, message_id ASC
                ) AS session_rank
            FROM candidate_hits
        )
    """
    sql += shape.final_select_sql
    params.append(limit)
    return RankedSearchQuery(sql=sql, params=tuple(params))


__all__ = [
    "RankedSearchQuery",
    "build_ranked_action_search_query",
    "build_ranked_session_search_query",
    "session_web_url",
]
