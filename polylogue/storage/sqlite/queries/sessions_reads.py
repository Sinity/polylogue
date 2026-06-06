"""Session read/query families."""

from __future__ import annotations

import aiosqlite

from polylogue.storage.runtime import SessionRecord
from polylogue.storage.sqlite.queries.filter_builder import (
    _build_session_filters,
    _needs_stats_join,
)
from polylogue.storage.sqlite.queries.mappers import _row_to_session


async def get_session(conn: aiosqlite.Connection, session_id: str) -> SessionRecord | None:
    cursor = await conn.execute(
        "SELECT * FROM sessions WHERE session_id = ?",
        (session_id,),
    )
    row = await cursor.fetchone()
    return _row_to_session(row) if row is not None else None


async def get_sessions_batch(conn: aiosqlite.Connection, ids: list[str]) -> list[SessionRecord]:
    if not ids:
        return []
    placeholders = ",".join("?" for _ in ids)
    cursor = await conn.execute(
        f"SELECT * FROM sessions WHERE session_id IN ({placeholders})",
        ids,
    )
    rows = await cursor.fetchall()
    by_id = {row["session_id"]: _row_to_session(row) for row in rows}
    return [by_id[cid] for cid in ids if cid in by_id]


async def list_sessions(
    conn: aiosqlite.Connection,
    *,
    source: str | None = None,
    provider: str | None = None,
    providers: list[str] | None = None,
    parent_id: str | None = None,
    since: str | None = None,
    until: str | None = None,
    title_contains: str | None = None,
    referenced_path: list[str] | None = None,
    cwd_prefix: str | None = None,
    action_terms: list[str] | None = None,
    excluded_action_terms: list[str] | None = None,
    tool_terms: list[str] | None = None,
    excluded_tool_terms: list[str] | None = None,
    repo_names: list[str] | None = None,
    limit: int | None = None,
    offset: int = 0,
    has_tool_use: bool = False,
    has_thinking: bool = False,
    has_paste: bool = False,
    typed_only: bool = False,
    min_messages: int | None = None,
    max_messages: int | None = None,
    min_words: int | None = None,
    message_type: str | None = None,
    include_provider_meta: bool = False,
) -> list[SessionRecord]:
    del include_provider_meta  # list_sessions always loads via SELECT *
    use_stats_join = _needs_stats_join(
        has_tool_use=has_tool_use,
        has_thinking=has_thinking,
        has_paste=has_paste,
        typed_only=typed_only,
        min_messages=min_messages,
        max_messages=max_messages,
        min_words=min_words,
    )
    where_sql, params = _build_session_filters(
        source=source,
        provider=provider,
        providers=providers,
        parent_id=parent_id,
        since=since,
        until=until,
        title_contains=title_contains,
        referenced_path=referenced_path,
        cwd_prefix=cwd_prefix,
        action_terms=action_terms,
        excluded_action_terms=excluded_action_terms,
        tool_terms=tool_terms,
        excluded_tool_terms=excluded_tool_terms,
        repo_names=repo_names,
        has_tool_use=has_tool_use,
        has_thinking=has_thinking,
        has_paste=has_paste,
        typed_only=typed_only,
        min_messages=min_messages,
        max_messages=max_messages,
        min_words=min_words,
        message_type=message_type,
    )

    if use_stats_join:
        from_clause = "FROM sessions c LEFT JOIN session_stats cs ON cs.session_id = c.session_id"
        select_clause = "SELECT c.*"
        order_clause = "ORDER BY (c.sort_key IS NULL) ASC, c.sort_key DESC, c.session_id DESC"
    else:
        from_clause = "FROM sessions"
        select_clause = "SELECT *"
        order_clause = "ORDER BY (sort_key IS NULL) ASC, sort_key DESC, session_id DESC"

    query = f"""
        {select_clause} {from_clause}
        {where_sql}
        {order_clause}
    """
    if limit is not None:
        query += " LIMIT ?"
        params.append(limit)
    elif offset > 0:
        query += " LIMIT -1"

    if offset > 0:
        query += " OFFSET ?"
        params.append(offset)

    cursor = await conn.execute(query, tuple(params))
    rows = await cursor.fetchall()
    return [_row_to_session(row) for row in rows]


async def list_session_summaries(
    conn: aiosqlite.Connection,
    *,
    source: str | None = None,
    provider: str | None = None,
    providers: list[str] | None = None,
    parent_id: str | None = None,
    since: str | None = None,
    until: str | None = None,
    title_contains: str | None = None,
    referenced_path: list[str] | None = None,
    cwd_prefix: str | None = None,
    action_terms: list[str] | None = None,
    excluded_action_terms: list[str] | None = None,
    tool_terms: list[str] | None = None,
    excluded_tool_terms: list[str] | None = None,
    repo_names: list[str] | None = None,
    limit: int | None = None,
    offset: int = 0,
    has_tool_use: bool = False,
    has_thinking: bool = False,
    has_paste: bool = False,
    typed_only: bool = False,
    min_messages: int | None = None,
    max_messages: int | None = None,
    min_words: int | None = None,
    message_type: str | None = None,
    include_provider_meta: bool = False,
) -> list[SessionRecord]:
    use_stats_join = _needs_stats_join(
        has_tool_use=has_tool_use,
        has_thinking=has_thinking,
        has_paste=has_paste,
        typed_only=typed_only,
        min_messages=min_messages,
        max_messages=max_messages,
        min_words=min_words,
    )
    pm_qualified = "c.provider_meta" if include_provider_meta else "NULL AS provider_meta"
    pm_bare = "provider_meta" if include_provider_meta else "NULL AS provider_meta"
    where_sql, params = _build_session_filters(
        source=source,
        provider=provider,
        providers=providers,
        parent_id=parent_id,
        since=since,
        until=until,
        title_contains=title_contains,
        referenced_path=referenced_path,
        cwd_prefix=cwd_prefix,
        action_terms=action_terms,
        excluded_action_terms=excluded_action_terms,
        tool_terms=tool_terms,
        excluded_tool_terms=excluded_tool_terms,
        repo_names=repo_names,
        has_tool_use=has_tool_use,
        has_thinking=has_thinking,
        has_paste=has_paste,
        typed_only=typed_only,
        min_messages=min_messages,
        max_messages=max_messages,
        min_words=min_words,
        message_type=message_type,
    )

    if use_stats_join:
        from_clause = "FROM sessions c LEFT JOIN session_stats cs ON cs.session_id = c.session_id"
        select_clause = f"""
        SELECT
            c.session_id,
            c.source_name,
            c.provider_session_id,
            c.title,
            c.created_at,
            c.updated_at,
            c.sort_key,
            c.content_hash,
            {pm_qualified},
            c.metadata,
            c.version,
            c.parent_session_id,
            c.branch_type,
            c.raw_id
        """
        order_clause = "ORDER BY (c.sort_key IS NULL) ASC, c.sort_key DESC, c.session_id DESC"
    else:
        from_clause = "FROM sessions"
        select_clause = f"""
        SELECT
            session_id,
            source_name,
            provider_session_id,
            title,
            created_at,
            updated_at,
            sort_key,
            content_hash,
            {pm_bare},
            metadata,
            version,
            parent_session_id,
            branch_type,
            raw_id
        """
        order_clause = "ORDER BY (sort_key IS NULL) ASC, sort_key DESC, session_id DESC"

    query = f"""
        {select_clause} {from_clause}
        {where_sql}
        {order_clause}
    """
    if limit is not None:
        query += " LIMIT ?"
        params.append(limit)
    elif offset > 0:
        query += " LIMIT -1"

    if offset > 0:
        query += " OFFSET ?"
        params.append(offset)

    cursor = await conn.execute(query, tuple(params))
    rows = await cursor.fetchall()
    return [_row_to_session(row) for row in rows]


async def count_sessions(
    conn: aiosqlite.Connection,
    *,
    source: str | None = None,
    provider: str | None = None,
    providers: list[str] | None = None,
    since: str | None = None,
    until: str | None = None,
    title_contains: str | None = None,
    referenced_path: list[str] | None = None,
    cwd_prefix: str | None = None,
    action_terms: list[str] | None = None,
    excluded_action_terms: list[str] | None = None,
    tool_terms: list[str] | None = None,
    excluded_tool_terms: list[str] | None = None,
    repo_names: list[str] | None = None,
    has_tool_use: bool = False,
    has_thinking: bool = False,
    has_paste: bool = False,
    typed_only: bool = False,
    min_messages: int | None = None,
    max_messages: int | None = None,
    min_words: int | None = None,
    message_type: str | None = None,
) -> int:
    use_stats_join = _needs_stats_join(
        has_tool_use=has_tool_use,
        has_thinking=has_thinking,
        has_paste=has_paste,
        typed_only=typed_only,
        min_messages=min_messages,
        max_messages=max_messages,
        min_words=min_words,
    )
    where_sql, params = _build_session_filters(
        source=source,
        provider=provider,
        providers=providers,
        since=since,
        until=until,
        title_contains=title_contains,
        referenced_path=referenced_path,
        cwd_prefix=cwd_prefix,
        action_terms=action_terms,
        excluded_action_terms=excluded_action_terms,
        tool_terms=tool_terms,
        excluded_tool_terms=excluded_tool_terms,
        repo_names=repo_names,
        has_tool_use=has_tool_use,
        has_thinking=has_thinking,
        has_paste=has_paste,
        typed_only=typed_only,
        min_messages=min_messages,
        max_messages=max_messages,
        min_words=min_words,
        message_type=message_type,
    )
    if use_stats_join:
        sql = f"SELECT COUNT(*) as cnt FROM sessions c LEFT JOIN session_stats cs ON cs.session_id = c.session_id {where_sql}"
    else:
        sql = f"SELECT COUNT(*) as cnt FROM sessions {where_sql}"
    cursor = await conn.execute(sql, tuple(params))
    row = await cursor.fetchone()
    return int(row["cnt"]) if row else 0


async def list_sessions_by_parent(conn: aiosqlite.Connection, parent_id: str) -> list[SessionRecord]:
    cursor = await conn.execute(
        """
        SELECT * FROM sessions
        WHERE parent_session_id = ?
        ORDER BY created_at ASC
        """,
        (parent_id,),
    )
    rows = await cursor.fetchall()
    return [_row_to_session(row) for row in rows]


__all__ = [
    "count_sessions",
    "get_session",
    "get_sessions_batch",
    "list_session_summaries",
    "list_sessions",
    "list_sessions_by_parent",
]
