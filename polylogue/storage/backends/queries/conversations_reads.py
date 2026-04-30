"""Conversation read/query families."""

from __future__ import annotations

import aiosqlite

from polylogue.storage.backends.queries.filter_builder import (
    _build_conversation_filters,
    _needs_stats_join,
)
from polylogue.storage.backends.queries.mappers import _row_to_conversation
from polylogue.storage.runtime import ConversationRecord


async def get_conversation(conn: aiosqlite.Connection, conversation_id: str) -> ConversationRecord | None:
    cursor = await conn.execute(
        "SELECT * FROM conversations WHERE conversation_id = ?",
        (conversation_id,),
    )
    row = await cursor.fetchone()
    return _row_to_conversation(row) if row is not None else None


async def get_conversations_batch(conn: aiosqlite.Connection, ids: list[str]) -> list[ConversationRecord]:
    if not ids:
        return []
    placeholders = ",".join("?" for _ in ids)
    cursor = await conn.execute(
        f"SELECT * FROM conversations WHERE conversation_id IN ({placeholders})",
        ids,
    )
    rows = await cursor.fetchall()
    by_id = {row["conversation_id"]: _row_to_conversation(row) for row in rows}
    return [by_id[cid] for cid in ids if cid in by_id]


async def list_conversations(
    conn: aiosqlite.Connection,
    *,
    source: str | None = None,
    provider: str | None = None,
    providers: list[str] | None = None,
    parent_id: str | None = None,
    since: str | None = None,
    until: str | None = None,
    title_contains: str | None = None,
    path_terms: list[str] | None = None,
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
    min_messages: int | None = None,
    max_messages: int | None = None,
    min_words: int | None = None,
) -> list[ConversationRecord]:
    use_stats_join = _needs_stats_join(
        has_tool_use=has_tool_use,
        has_thinking=has_thinking,
        min_messages=min_messages,
        max_messages=max_messages,
        min_words=min_words,
    )
    where_sql, params = _build_conversation_filters(
        source=source,
        provider=provider,
        providers=providers,
        parent_id=parent_id,
        since=since,
        until=until,
        title_contains=title_contains,
        path_terms=path_terms,
        cwd_prefix=cwd_prefix,
        action_terms=action_terms,
        excluded_action_terms=excluded_action_terms,
        tool_terms=tool_terms,
        excluded_tool_terms=excluded_tool_terms,
        repo_names=repo_names,
        has_tool_use=has_tool_use,
        has_thinking=has_thinking,
        min_messages=min_messages,
        max_messages=max_messages,
        min_words=min_words,
    )

    if use_stats_join:
        from_clause = "FROM conversations c LEFT JOIN conversation_stats cs ON cs.conversation_id = c.conversation_id"
        select_clause = "SELECT c.*"
        order_clause = "ORDER BY (c.sort_key IS NULL) ASC, c.sort_key DESC, c.conversation_id DESC"
    else:
        from_clause = "FROM conversations"
        select_clause = "SELECT *"
        order_clause = "ORDER BY (sort_key IS NULL) ASC, sort_key DESC, conversation_id DESC"

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
    return [_row_to_conversation(row) for row in rows]


async def list_conversation_summaries(
    conn: aiosqlite.Connection,
    *,
    source: str | None = None,
    provider: str | None = None,
    providers: list[str] | None = None,
    parent_id: str | None = None,
    since: str | None = None,
    until: str | None = None,
    title_contains: str | None = None,
    path_terms: list[str] | None = None,
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
    min_messages: int | None = None,
    max_messages: int | None = None,
    min_words: int | None = None,
) -> list[ConversationRecord]:
    use_stats_join = _needs_stats_join(
        has_tool_use=has_tool_use,
        has_thinking=has_thinking,
        min_messages=min_messages,
        max_messages=max_messages,
        min_words=min_words,
    )
    where_sql, params = _build_conversation_filters(
        source=source,
        provider=provider,
        providers=providers,
        parent_id=parent_id,
        since=since,
        until=until,
        title_contains=title_contains,
        path_terms=path_terms,
        cwd_prefix=cwd_prefix,
        action_terms=action_terms,
        excluded_action_terms=excluded_action_terms,
        tool_terms=tool_terms,
        excluded_tool_terms=excluded_tool_terms,
        repo_names=repo_names,
        has_tool_use=has_tool_use,
        has_thinking=has_thinking,
        min_messages=min_messages,
        max_messages=max_messages,
        min_words=min_words,
    )

    if use_stats_join:
        from_clause = "FROM conversations c LEFT JOIN conversation_stats cs ON cs.conversation_id = c.conversation_id"
        select_clause = """
        SELECT
            c.conversation_id,
            c.provider_name,
            c.provider_conversation_id,
            c.title,
            c.created_at,
            c.updated_at,
            c.sort_key,
            c.content_hash,
            CASE
                WHEN json_extract(c.provider_meta, '$.tail_overlay') IS NOT NULL
                THEN json_object('tail_overlay', json_extract(c.provider_meta, '$.tail_overlay'))
                ELSE NULL
            END AS provider_meta,
            c.metadata,
            c.version,
            c.parent_conversation_id,
            c.branch_type,
            c.raw_id
        """
        order_clause = "ORDER BY (c.sort_key IS NULL) ASC, c.sort_key DESC, c.conversation_id DESC"
    else:
        from_clause = "FROM conversations"
        select_clause = """
        SELECT
            conversation_id,
            provider_name,
            provider_conversation_id,
            title,
            created_at,
            updated_at,
            sort_key,
            content_hash,
            CASE
                WHEN json_extract(provider_meta, '$.tail_overlay') IS NOT NULL
                THEN json_object('tail_overlay', json_extract(provider_meta, '$.tail_overlay'))
                ELSE NULL
            END AS provider_meta,
            metadata,
            version,
            parent_conversation_id,
            branch_type,
            raw_id
        """
        order_clause = "ORDER BY (sort_key IS NULL) ASC, sort_key DESC, conversation_id DESC"

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
    return [_row_to_conversation(row) for row in rows]


async def count_conversations(
    conn: aiosqlite.Connection,
    *,
    source: str | None = None,
    provider: str | None = None,
    providers: list[str] | None = None,
    since: str | None = None,
    until: str | None = None,
    title_contains: str | None = None,
    path_terms: list[str] | None = None,
    cwd_prefix: str | None = None,
    action_terms: list[str] | None = None,
    excluded_action_terms: list[str] | None = None,
    tool_terms: list[str] | None = None,
    excluded_tool_terms: list[str] | None = None,
    repo_names: list[str] | None = None,
    has_tool_use: bool = False,
    has_thinking: bool = False,
    min_messages: int | None = None,
    max_messages: int | None = None,
    min_words: int | None = None,
) -> int:
    use_stats_join = _needs_stats_join(
        has_tool_use=has_tool_use,
        has_thinking=has_thinking,
        min_messages=min_messages,
        max_messages=max_messages,
        min_words=min_words,
    )
    where_sql, params = _build_conversation_filters(
        source=source,
        provider=provider,
        providers=providers,
        since=since,
        until=until,
        title_contains=title_contains,
        path_terms=path_terms,
        cwd_prefix=cwd_prefix,
        action_terms=action_terms,
        excluded_action_terms=excluded_action_terms,
        tool_terms=tool_terms,
        excluded_tool_terms=excluded_tool_terms,
        has_tool_use=has_tool_use,
        has_thinking=has_thinking,
        min_messages=min_messages,
        max_messages=max_messages,
        min_words=min_words,
    )
    if use_stats_join:
        sql = f"SELECT COUNT(*) as cnt FROM conversations c LEFT JOIN conversation_stats cs ON cs.conversation_id = c.conversation_id {where_sql}"
    else:
        sql = f"SELECT COUNT(*) as cnt FROM conversations {where_sql}"
    cursor = await conn.execute(sql, tuple(params))
    row = await cursor.fetchone()
    return int(row["cnt"]) if row else 0


async def list_conversations_by_parent(conn: aiosqlite.Connection, parent_id: str) -> list[ConversationRecord]:
    cursor = await conn.execute(
        """
        SELECT * FROM conversations
        WHERE parent_conversation_id = ?
        ORDER BY created_at ASC
        """,
        (parent_id,),
    )
    rows = await cursor.fetchall()
    return [_row_to_conversation(row) for row in rows]


__all__ = [
    "count_conversations",
    "get_conversation",
    "get_conversations_batch",
    "list_conversation_summaries",
    "list_conversations",
    "list_conversations_by_parent",
]
