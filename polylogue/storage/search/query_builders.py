"""Canonical ranked SQL builders for current archive search."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import TypeAlias
from urllib.parse import quote

from polylogue.storage.search.query_support import normalize_fts5_query

SQLiteQueryParam: TypeAlias = str | int | float


@dataclass(frozen=True, slots=True)
class RankedSearchQuery:
    sql: str
    params: tuple[SQLiteQueryParam, ...]


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
    """Build the canonical ranked session search query over blocks."""
    fts_query = normalize_fts5_query(query)
    if fts_query is None:
        return None

    snippet_sql = (
        ", b.search_text AS fallback_text, snippet(messages_fts, 4, '[', ']', '...', 24) AS snippet"
        if include_snippet
        else ""
    )
    sql = f"""
        WITH candidate_hits AS (
            SELECT
                b.message_id AS message_id,
                b.session_id AS session_id,
                s.origin AS source_name,
                s.title AS title,
                COALESCE(m.occurred_at_ms, s.sort_key_ms, s.updated_at_ms, s.created_at_ms) / 1000.0 AS sort_key,
                bm25(messages_fts) AS relevance
                {snippet_sql}
            FROM messages_fts
            JOIN blocks AS b ON b.rowid = messages_fts.rowid
            JOIN messages AS m ON m.message_id = b.message_id
            JOIN sessions AS s ON s.session_id = b.session_id
            WHERE messages_fts MATCH ?
    """
    params: list[SQLiteQueryParam] = [fts_query]
    sql, params = _apply_scope_and_since(
        sql,
        params,
        scope_names=scope_names,
        since=since,
        since_column="COALESCE(m.occurred_at_ms, s.sort_key_ms, s.updated_at_ms, s.created_at_ms)",
    )
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
        SELECT *
        FROM ranked_hits
        WHERE session_rank = 1
        ORDER BY relevance ASC, sort_key DESC, message_id ASC
        LIMIT ?
    """
    params.append(limit)
    return RankedSearchQuery(sql=sql, params=tuple(params))


def build_ranked_action_search_query(
    *,
    query: str,
    limit: int,
    scope_names: Sequence[str] | None = None,
    since: str | None = None,
    include_snippet: bool = False,
) -> RankedSearchQuery | None:
    """Build ranked action search over tool blocks in ``messages_fts``."""
    fts_query = normalize_fts5_query(query)
    if fts_query is None:
        return None

    snippet_sql = ", snippet(messages_fts, 4, '[', ']', '...', 24) AS snippet" if include_snippet else ""
    sql = f"""
        WITH candidate_hits AS (
            SELECT
                b.message_id AS message_id,
                b.session_id AS session_id,
                COALESCE(NULLIF(b.semantic_type, ''), b.block_type) AS action_kind,
                COALESCE(NULLIF(LOWER(b.tool_name), ''), 'unknown') AS tool_name,
                s.origin AS source_name,
                s.title AS title,
                COALESCE(m.occurred_at_ms, s.sort_key_ms, s.updated_at_ms, s.created_at_ms) / 1000.0 AS sort_key,
                bm25(messages_fts) AS relevance
                {snippet_sql}
            FROM messages_fts
            JOIN blocks AS b ON b.rowid = messages_fts.rowid
            JOIN messages AS m ON m.message_id = b.message_id
            JOIN sessions AS s ON s.session_id = b.session_id
            WHERE messages_fts MATCH ?
              AND b.block_type IN ('tool_use', 'tool_result')
    """
    params: list[SQLiteQueryParam] = [fts_query]
    sql, params = _apply_scope_and_since(
        sql,
        params,
        scope_names=scope_names,
        since=since,
        since_column="COALESCE(m.occurred_at_ms, s.sort_key_ms, s.updated_at_ms, s.created_at_ms)",
    )
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
        SELECT
            session_id,
            message_id,
            source_name,
            title,
            sort_key,
            relevance
        FROM ranked_hits
        WHERE session_rank = 1
        ORDER BY relevance ASC, sort_key DESC, message_id ASC
        LIMIT ?
    """
    params.append(limit)
    return RankedSearchQuery(sql=sql, params=tuple(params))


def _apply_scope_and_since(
    sql: str,
    params: list[SQLiteQueryParam],
    *,
    scope_names: Sequence[str] | None,
    since: str | None,
    since_column: str,
) -> tuple[str, list[SQLiteQueryParam]]:
    if scope_names:
        placeholders = ",".join("?" for _ in scope_names)
        sql += f" AND s.origin IN ({placeholders})"
        params.extend(str(name) for name in scope_names)
    if since:
        # A row with no reliable timestamp anywhere in its fallback chain
        # (since_column IS NULL) is not evidence it falls outside a --since
        # window -- include it rather than let SQL's NULL propagation
        # silently exclude it (polylogue-s5mm, sort_key_ms COALESCE audit).
        sql += f" AND ({since_column} IS NULL OR {since_column} >= ?)"
        params.append(_parse_since_timestamp(since) * 1000.0)
    return sql, params


def _parse_since_timestamp(since: str) -> float:
    try:
        return datetime.fromisoformat(since).timestamp()
    except ValueError as exc:
        raise ValueError(f"Invalid --since date '{since}': {exc}. Use ISO format (e.g., 2023-01-01)") from exc


__all__ = [
    "RankedSearchQuery",
    "build_ranked_action_search_query",
    "build_ranked_session_search_query",
    "session_web_url",
]
