"""SQL filter builder for session queries."""

from __future__ import annotations

from polylogue.archive.message.types import validate_message_type_filter
from polylogue.archive.query.fields import storage_filters_require_stats_join
from polylogue.archive.query.path_prefix import escaped_sql_path_prefix_patterns
from polylogue.archive.viewport.viewports import ToolCategory
from polylogue.core.enums import Provider
from polylogue.core.sources import origin_from_provider
from polylogue.storage.sqlite.queries.project_refs import expand_project_refs

_SEMANTIC_ACTION_TYPES = tuple(category.value for category in ToolCategory)


def _iso_to_epoch(iso_str: str) -> float:
    """Convert an ISO date string to epoch seconds for SQL comparison."""
    from datetime import datetime, timezone

    try:
        dt = datetime.fromisoformat(iso_str)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.timestamp()
    except (ValueError, TypeError):
        pass
    try:
        return float(iso_str)
    except (ValueError, TypeError):
        raise ValueError(f"Invalid date filter value {iso_str!r}. Use ISO 8601 format (e.g., 2026-01-01).") from None


def _origin_value(value: str) -> str:
    return origin_from_provider(Provider.from_string(value)).value


def _build_session_filters(
    *,
    source: str | None = None,
    provider: str | None = None,
    providers: list[str] | None = None,
    parent_id: str | None = None,
    since: str | None = None,
    until: str | None = None,
    title_contains: str | None = None,
    referenced_path: list[str] | tuple[str, ...] | None = None,
    cwd_prefix: str | None = None,
    action_terms: list[str] | tuple[str, ...] | None = None,
    excluded_action_terms: list[str] | tuple[str, ...] | None = None,
    tool_terms: list[str] | tuple[str, ...] | None = None,
    excluded_tool_terms: list[str] | tuple[str, ...] | None = None,
    repo_names: list[str] | tuple[str, ...] | None = None,
    project_refs: list[str] | tuple[str, ...] | None = None,
    has_tool_use: bool = False,
    has_thinking: bool = False,
    has_paste: bool = False,
    typed_only: bool = False,
    min_messages: int | None = None,
    max_messages: int | None = None,
    min_words: int | None = None,
    max_words: int | None = None,
    message_type: str | None = None,
) -> tuple[str, list[str | int | float]]:
    """Build WHERE clause and params for session queries.

    Aggregate filters (has_tool_use, has_thinking, min_messages, max_messages,
    min_words) read the current aggregate columns on sessions.
    """
    where_clauses: list[str] = []
    params: list[str | int | float] = []
    needs_stats_alias = storage_filters_require_stats_join(locals())

    if source is not None:
        where_clauses.append("c.origin = ?" if needs_stats_alias else "origin = ?")
        params.append(_origin_value(source))
    if provider is not None:
        where_clauses.append("c.origin = ?" if needs_stats_alias else "origin = ?")
        params.append(_origin_value(provider))
    if providers:
        source_scope_params = [_origin_value(provider_name) for provider_name in providers]
        placeholders = ",".join("?" for _ in source_scope_params)
        origin_column = "c.origin" if needs_stats_alias else "origin"
        source_scope_sql = f"{origin_column} IN ({placeholders})"
        where_clauses.append(source_scope_sql)
        params.extend(source_scope_params)
    if parent_id is not None:
        where_clauses.append("c.parent_session_id = ?" if needs_stats_alias else "parent_session_id = ?")
        params.append(parent_id)
    if since is not None:
        where_clauses.append("c.sort_key_ms >= ?" if needs_stats_alias else "sort_key_ms >= ?")
        params.append(_iso_to_epoch(since) * 1000.0)
    if until is not None:
        where_clauses.append("c.sort_key_ms <= ?" if needs_stats_alias else "sort_key_ms <= ?")
        params.append(_iso_to_epoch(until) * 1000.0)
    if title_contains is not None:
        escaped = title_contains.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
        where_clauses.append("c.title LIKE ? ESCAPE '\\'" if needs_stats_alias else "title LIKE ? ESCAPE '\\'")
        params.append(f"%{escaped}%")

    aggregate_column = "c" if needs_stats_alias else "sessions"
    if has_tool_use:
        where_clauses.append(f"{aggregate_column}.tool_use_count > 0")
    if has_thinking:
        where_clauses.append(f"{aggregate_column}.thinking_count > 0")
    if has_paste:
        where_clauses.append(f"{aggregate_column}.paste_count > 0")
    if typed_only:
        where_clauses.append(f"{aggregate_column}.paste_count = 0")
    if min_messages is not None:
        where_clauses.append(f"{aggregate_column}.message_count >= ?")
        params.append(min_messages)
    if max_messages is not None:
        where_clauses.append(f"{aggregate_column}.message_count <= ?")
        params.append(max_messages)
    if min_words is not None:
        where_clauses.append(f"{aggregate_column}.word_count >= ?")
        params.append(min_words)
    if max_words is not None:
        where_clauses.append(f"{aggregate_column}.word_count <= ?")
        params.append(max_words)

    # Semantic filters via EXISTS subquery on the canonical actions view.
    # When using stats join, outer table is aliased as 'c'; otherwise use fully qualified
    # table name to prevent ambiguity.
    conv_id_col = "c.session_id" if needs_stats_alias else "sessions.session_id"
    if referenced_path:
        for term in referenced_path:
            normalized = str(term).replace("\\", "/").lower()
            escaped = normalized.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
            where_clauses.append(
                f"EXISTS (SELECT 1 FROM actions a "
                f"WHERE a.session_id = {conv_id_col} "
                f"AND REPLACE(LOWER(COALESCE(a.tool_path, '')), char(92), '/') LIKE ? ESCAPE '\\')"
            )
            params.append(f"%{escaped}%")
    if cwd_prefix:
        session_id_col = "c.session_id" if needs_stats_alias else "sessions.session_id"
        exact_prefix, child_prefix = escaped_sql_path_prefix_patterns(cwd_prefix)
        where_clauses.append(
            "EXISTS (SELECT 1 FROM session_working_dirs cwd "
            f"WHERE cwd.session_id = {session_id_col} "
            "AND (REPLACE(cwd.path, char(92), '/') = ? "
            "OR REPLACE(cwd.path, char(92), '/') LIKE ? ESCAPE '\\'))"
        )
        params.extend([exact_prefix, child_prefix])
    if action_terms:
        for term in action_terms:
            if str(term) == "none":
                placeholders = ",".join("?" for _ in _SEMANTIC_ACTION_TYPES)
                where_clauses.append(
                    f"NOT EXISTS (SELECT 1 FROM actions a WHERE a.session_id = {conv_id_col}"
                    f" AND a.semantic_type IN ({placeholders}))"
                )
                params.extend(_SEMANTIC_ACTION_TYPES)
            else:
                where_clauses.append(
                    f"EXISTS (SELECT 1 FROM actions a WHERE a.session_id = {conv_id_col} AND a.semantic_type = ?)"
                )
                params.append(str(term))
    if excluded_action_terms:
        for term in excluded_action_terms:
            if str(term) == "none":
                placeholders = ",".join("?" for _ in _SEMANTIC_ACTION_TYPES)
                where_clauses.append(
                    f"EXISTS (SELECT 1 FROM actions a WHERE a.session_id = {conv_id_col}"
                    f" AND a.semantic_type IN ({placeholders}))"
                )
                params.extend(_SEMANTIC_ACTION_TYPES)
            else:
                where_clauses.append(
                    f"NOT EXISTS (SELECT 1 FROM actions a WHERE a.session_id = {conv_id_col} AND a.semantic_type = ?)"
                )
                params.append(str(term))
    if tool_terms:
        for term in tool_terms:
            if str(term) == "none":
                where_clauses.append(f"NOT EXISTS (SELECT 1 FROM actions a WHERE a.session_id = {conv_id_col})")
            else:
                where_clauses.append(
                    f"EXISTS (SELECT 1 FROM actions a WHERE a.session_id = {conv_id_col} AND lower(a.tool_name) = ?)"
                )
                params.append(str(term).lower())
    if excluded_tool_terms:
        for term in excluded_tool_terms:
            if str(term) == "none":
                where_clauses.append(f"EXISTS (SELECT 1 FROM actions a WHERE a.session_id = {conv_id_col})")
            else:
                where_clauses.append(
                    f"NOT EXISTS (SELECT 1 FROM actions a WHERE a.session_id = {conv_id_col}"
                    " AND lower(a.tool_name) = ?)"
                )
                params.append(str(term).lower())
    if repo_names:
        placeholders = ",".join("?" for _ in repo_names)
        where_clauses.append(
            f"EXISTS (SELECT 1 FROM session_profiles sp "
            f"JOIN json_each(COALESCE(sp.repo_names_json, '[]')) "
            f"WHERE sp.session_id = {conv_id_col} AND value IN ({placeholders}))"
        )
        params.extend(repo_names)
    if project_refs:
        project_refs = expand_project_refs(project_refs)
        project_column = "c.provider_project_ref" if needs_stats_alias else "provider_project_ref"
        placeholders = ",".join("?" for _ in project_refs)
        where_clauses.append(f"{project_column} IN ({placeholders})")
        params.extend(project_refs)
    if message_type:
        where_clauses.append(
            f"EXISTS (SELECT 1 FROM messages mt WHERE mt.session_id = {conv_id_col} AND mt.message_type = ?)"
        )
        params.append(validate_message_type_filter(message_type).value)
    where_sql = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""
    return where_sql, params


def _needs_stats_join(
    *,
    has_tool_use: bool = False,
    has_thinking: bool = False,
    has_paste: bool = False,
    typed_only: bool = False,
    min_messages: int | None = None,
    max_messages: int | None = None,
    min_words: int | None = None,
    max_words: int | None = None,
) -> bool:
    """Return True when the query should alias sessions as ``c``."""
    return storage_filters_require_stats_join(locals())
