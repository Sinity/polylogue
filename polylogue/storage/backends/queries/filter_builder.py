"""SQL filter builder for conversation queries."""

from __future__ import annotations

from polylogue.lib.viewports import ToolCategory
from polylogue.storage.backends.connection import _build_provider_scope_filter

_SEMANTIC_ACTION_TYPES = tuple(category.value for category in ToolCategory)


def _iso_to_epoch(iso_str: str) -> float:
    """Convert an ISO date string to epoch seconds for SQL comparison."""
    from datetime import datetime

    try:
        return datetime.fromisoformat(iso_str).timestamp()
    except (ValueError, TypeError):
        try:
            return float(iso_str)
        except (ValueError, TypeError):
            return 0.0


def _build_conversation_filters(
    *,
    source: str | None = None,
    provider: str | None = None,
    providers: list[str] | None = None,
    parent_id: str | None = None,
    since: str | None = None,
    until: str | None = None,
    title_contains: str | None = None,
    path_terms: list[str] | tuple[str, ...] | None = None,
    action_terms: list[str] | tuple[str, ...] | None = None,
    excluded_action_terms: list[str] | tuple[str, ...] | None = None,
    tool_terms: list[str] | tuple[str, ...] | None = None,
    excluded_tool_terms: list[str] | tuple[str, ...] | None = None,
    has_tool_use: bool = False,
    has_thinking: bool = False,
    min_messages: int | None = None,
    max_messages: int | None = None,
    min_words: int | None = None,
) -> tuple[str, list[str | int | float]]:
    """Build WHERE clause and params for conversation queries.

    Stats-based filters (has_tool_use, has_thinking, min_messages, max_messages,
    min_words) emit a LEFT JOIN on conversation_stats and filter on cs columns.
    Path and semantic filters emit EXISTS subqueries against content_blocks.
    Callers must prefix conversation columns with 'c.' when using stats filters.
    """
    where_clauses: list[str] = []
    params: list[str | int | float] = []
    needs_stats_join = has_tool_use or has_thinking or min_messages is not None or max_messages is not None or min_words is not None

    if source is not None:
        where_clauses.append("c.source_name = ?" if needs_stats_join else "source_name = ?")
        params.append(source)
    if provider is not None:
        where_clauses.append("c.provider_name = ?" if needs_stats_join else "provider_name = ?")
        params.append(provider)
    if providers:
        source_scope_sql, source_scope_params = _build_provider_scope_filter(
            providers,
            provider_column="c.provider_name" if needs_stats_join else "provider_name",
        )
        where_clauses.append(source_scope_sql)
        params.extend(source_scope_params)
    if parent_id is not None:
        where_clauses.append("c.parent_conversation_id = ?" if needs_stats_join else "parent_conversation_id = ?")
        params.append(parent_id)
    if since is not None:
        where_clauses.append("c.sort_key >= ?" if needs_stats_join else "sort_key >= ?")
        params.append(_iso_to_epoch(since))
    if until is not None:
        where_clauses.append("c.sort_key <= ?" if needs_stats_join else "sort_key <= ?")
        params.append(_iso_to_epoch(until))
    if title_contains is not None:
        escaped = title_contains.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
        where_clauses.append("c.title LIKE ? ESCAPE '\\'" if needs_stats_join else "title LIKE ? ESCAPE '\\'")
        params.append(f"%{escaped}%")

    # Stats-based filters (require conversation_stats JOIN)
    if has_tool_use:
        where_clauses.append("cs.tool_use_count > 0")
    if has_thinking:
        where_clauses.append("cs.thinking_count > 0")
    if min_messages is not None:
        where_clauses.append("cs.message_count >= ?")
        params.append(min_messages)
    if max_messages is not None:
        where_clauses.append("cs.message_count <= ?")
        params.append(max_messages)
    if min_words is not None:
        where_clauses.append("cs.word_count >= ?")
        params.append(min_words)

    # Semantic filters via EXISTS subquery on durable action_events.
    # When using stats join, outer table is aliased as 'c'; otherwise use fully qualified
    # table name to prevent ambiguity.
    conv_id_col = "c.conversation_id" if needs_stats_join else "conversations.conversation_id"
    if path_terms:
        for term in path_terms:
            normalized = str(term).replace("\\", "/").lower()
            where_clauses.append(
                f"EXISTS (SELECT 1 FROM action_events ae "
                f"JOIN json_each(COALESCE(ae.affected_paths_json, '[]')) path "
                f"WHERE ae.conversation_id = {conv_id_col} AND LOWER(path.value) LIKE ?)"
            )
            params.append(f"%{normalized}%")
    if action_terms:
        for term in action_terms:
            if str(term) == "none":
                placeholders = ",".join("?" for _ in _SEMANTIC_ACTION_TYPES)
                where_clauses.append(
                    f"NOT EXISTS (SELECT 1 FROM action_events ae WHERE ae.conversation_id = {conv_id_col}"
                    f" AND ae.action_kind IN ({placeholders}))"
                )
                params.extend(_SEMANTIC_ACTION_TYPES)
            else:
                where_clauses.append(
                    f"EXISTS (SELECT 1 FROM action_events ae WHERE ae.conversation_id = {conv_id_col}"
                    " AND ae.action_kind = ?)"
                )
                params.append(str(term))
    if excluded_action_terms:
        for term in excluded_action_terms:
            if str(term) == "none":
                placeholders = ",".join("?" for _ in _SEMANTIC_ACTION_TYPES)
                where_clauses.append(
                    f"EXISTS (SELECT 1 FROM action_events ae WHERE ae.conversation_id = {conv_id_col}"
                    f" AND ae.action_kind IN ({placeholders}))"
                )
                params.extend(_SEMANTIC_ACTION_TYPES)
            else:
                where_clauses.append(
                    f"NOT EXISTS (SELECT 1 FROM action_events ae WHERE ae.conversation_id = {conv_id_col}"
                    " AND ae.action_kind = ?)"
                )
                params.append(str(term))
    if tool_terms:
        for term in tool_terms:
            if str(term) == "none":
                where_clauses.append(
                    f"NOT EXISTS (SELECT 1 FROM action_events ae WHERE ae.conversation_id = {conv_id_col})"
                )
            else:
                where_clauses.append(
                    f"EXISTS (SELECT 1 FROM action_events ae WHERE ae.conversation_id = {conv_id_col}"
                    " AND ae.normalized_tool_name = ?)"
                )
                params.append(str(term).lower())
    if excluded_tool_terms:
        for term in excluded_tool_terms:
            if str(term) == "none":
                where_clauses.append(
                    f"EXISTS (SELECT 1 FROM action_events ae WHERE ae.conversation_id = {conv_id_col})"
                )
            else:
                where_clauses.append(
                    f"NOT EXISTS (SELECT 1 FROM action_events ae WHERE ae.conversation_id = {conv_id_col}"
                    " AND ae.normalized_tool_name = ?)"
                )
                params.append(str(term).lower())
    where_sql = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""
    return where_sql, params


def _needs_stats_join(
    *,
    has_tool_use: bool = False,
    has_thinking: bool = False,
    min_messages: int | None = None,
    max_messages: int | None = None,
    min_words: int | None = None,
) -> bool:
    """Return True when the query requires a JOIN on conversation_stats."""
    return has_tool_use or has_thinking or min_messages is not None or max_messages is not None or min_words is not None
