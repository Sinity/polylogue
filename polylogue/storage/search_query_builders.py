"""Canonical ranked SQL builders for conversation and action search."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from pathlib import Path

from polylogue.paths import conversation_render_root

from .backends.connection import _build_provider_scope_filter
from .search_query_support import normalize_fts5_query


def resolve_conversation_path(
    archive_root: Path,
    render_root_path: Path | None,
    provider_name: str,
    conversation_id: str,
) -> Path:
    """Resolve the path to a conversation's rendered markdown file."""
    output_root = render_root_path or (archive_root / "render")
    safe_root = conversation_render_root(output_root, provider_name, conversation_id)
    return safe_root / "conversation.md"


def build_ranked_conversation_search_query(
    *,
    query: str,
    limit: int,
    scope_names: Sequence[str] | None = None,
    since: str | None = None,
    include_snippet: bool = False,
) -> tuple[str, tuple[object, ...]] | None:
    """Build the canonical ranked conversation search query."""
    fts_query = normalize_fts5_query(query)
    if fts_query is None:
        return None

    candidate_columns = [
        "messages_fts.message_id",
        "messages_fts.conversation_id",
        "conversations.provider_name",
        "conversations.source_name",
        "conversations.title",
        "messages.sort_key",
        "bm25(messages_fts) AS relevance",
    ]
    if include_snippet:
        candidate_columns.append("snippet(messages_fts, 2, '[', ']', '…', 12) AS snippet")

    sql = f"""
        WITH candidate_hits AS (
            SELECT
                {", ".join(candidate_columns)}
            FROM messages_fts
            JOIN conversations ON conversations.conversation_id = messages_fts.conversation_id
            JOIN messages ON messages.message_id = messages_fts.message_id
            WHERE messages_fts MATCH ?
    """
    params: list[object] = [fts_query]

    if scope_names:
        scope_sql, scope_params = _build_provider_scope_filter(
            scope_names,
            provider_column="conversations.provider_name",
        )
        sql += f" AND {scope_sql}"
        params.extend(scope_params)

    if since:
        try:
            since_dt = datetime.fromisoformat(since)
        except ValueError as exc:
            raise ValueError(f"Invalid --since date '{since}': {exc}. Use ISO format (e.g., 2023-01-01)") from exc
        sql += " AND messages.sort_key >= ?"
        params.append(since_dt.timestamp())

    sql += """
        ),
        ranked_hits AS (
            SELECT
                *,
                ROW_NUMBER() OVER (
                    PARTITION BY conversation_id
                    ORDER BY relevance ASC, sort_key DESC, message_id ASC
                ) AS conversation_rank
            FROM candidate_hits
        )
        SELECT *
        FROM ranked_hits
        WHERE conversation_rank = 1
        ORDER BY relevance ASC, sort_key DESC, message_id ASC
        LIMIT ?
    """
    params.append(limit)
    return sql, tuple(params)


def build_ranked_action_search_query(
    *,
    query: str,
    limit: int,
    scope_names: Sequence[str] | None = None,
    since: str | None = None,
    include_snippet: bool = False,
) -> tuple[str, tuple[object, ...]] | None:
    """Build the canonical ranked action-search query."""
    fts_query = normalize_fts5_query(query)
    if fts_query is None:
        return None

    candidate_columns = [
        "action_events_fts.event_id",
        "action_events_fts.message_id",
        "action_events_fts.conversation_id",
        "action_events_fts.action_kind",
        "action_events_fts.tool_name",
        "conversations.provider_name",
        "conversations.source_name",
        "conversations.title",
        "messages.sort_key",
        "bm25(action_events_fts) AS relevance",
    ]
    if include_snippet:
        candidate_columns.append("snippet(action_events_fts, 5, '[', ']', '…', 12) AS snippet")

    sql = f"""
        WITH candidate_hits AS (
            SELECT
                {", ".join(candidate_columns)}
            FROM action_events_fts
            JOIN conversations ON conversations.conversation_id = action_events_fts.conversation_id
            JOIN messages ON messages.message_id = action_events_fts.message_id
            WHERE action_events_fts MATCH ?
    """
    params: list[object] = [fts_query]

    if scope_names:
        scope_sql, scope_params = _build_provider_scope_filter(
            scope_names,
            provider_column="conversations.provider_name",
        )
        sql += f" AND {scope_sql}"
        params.extend(scope_params)

    if since:
        try:
            since_dt = datetime.fromisoformat(since)
        except ValueError as exc:
            raise ValueError(f"Invalid --since date '{since}': {exc}. Use ISO format (e.g., 2023-01-01)") from exc
        sql += " AND messages.sort_key >= ?"
        params.append(since_dt.timestamp())

    sql += """
        ),
        ranked_hits AS (
            SELECT
                conversation_id,
                message_id,
                provider_name,
                source_name,
                title,
                sort_key,
                relevance,
                ROW_NUMBER() OVER (
                    PARTITION BY conversation_id
                    ORDER BY relevance ASC, sort_key DESC, message_id ASC
                ) AS rank
            FROM candidate_hits
        )
        SELECT
            conversation_id,
            message_id,
            provider_name,
            source_name,
            title,
            sort_key,
            relevance
        FROM ranked_hits
        WHERE rank = 1
        ORDER BY relevance ASC, sort_key DESC, message_id ASC
        LIMIT ?
    """
    params.append(limit)
    return sql, tuple(params)


__all__ = [
    "build_ranked_action_search_query",
    "build_ranked_conversation_search_query",
    "resolve_conversation_path",
]
