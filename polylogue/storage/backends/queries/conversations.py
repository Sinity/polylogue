"""Conversation CRUD and tree-traversal queries."""

from __future__ import annotations

from collections.abc import AsyncIterator

import aiosqlite

from polylogue.lib.json import dumps as json_dumps
from polylogue.storage.backends.connection import _build_source_scope_filter
from polylogue.storage.backends.queries.filter_builder import (
    _build_conversation_filters,
    _needs_stats_join,
)
from polylogue.storage.backends.queries.mappers import _parse_json, _row_to_conversation
from polylogue.storage.store import (
    ConversationRecord,
    _json_or_none,
)

__all__ = [
    "get_conversation",
    "get_conversations_batch",
    "list_conversations",
    "count_conversations",
    "conversation_exists_by_hash",
    "save_conversation_record",
    "list_conversations_by_parent",
    "resolve_id",
    "get_last_sync_timestamp",
    "conversation_id_query",
    "count_conversation_ids",
    "iter_conversation_ids",
    "get_metadata",
    "update_metadata_raw",
    "set_metadata",
    "delete_conversation_sql",
    "list_tags",
    "search_conversations",
]


async def get_conversation(
    conn: aiosqlite.Connection, conversation_id: str
) -> ConversationRecord | None:
    """Retrieve a conversation by ID."""
    cursor = await conn.execute(
        "SELECT * FROM conversations WHERE conversation_id = ?",
        (conversation_id,),
    )
    row = await cursor.fetchone()
    return _row_to_conversation(row) if row is not None else None


async def get_conversations_batch(
    conn: aiosqlite.Connection, ids: list[str]
) -> list[ConversationRecord]:
    """Retrieve multiple conversations in a single query, preserving order."""
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
    limit: int | None = None,
    offset: int = 0,
    has_tool_use: bool = False,
    has_thinking: bool = False,
    min_messages: int | None = None,
    max_messages: int | None = None,
    min_words: int | None = None,
    has_file_ops: bool = False,
    has_git_ops: bool = False,
    has_subagent: bool = False,
) -> list[ConversationRecord]:
    """List conversations with optional filtering and pagination."""
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
        has_tool_use=has_tool_use,
        has_thinking=has_thinking,
        min_messages=min_messages,
        max_messages=max_messages,
        min_words=min_words,
        has_file_ops=has_file_ops,
        has_git_ops=has_git_ops,
        has_subagent=has_subagent,
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
    has_tool_use: bool = False,
    has_thinking: bool = False,
    min_messages: int | None = None,
    max_messages: int | None = None,
    min_words: int | None = None,
    has_file_ops: bool = False,
    has_git_ops: bool = False,
    has_subagent: bool = False,
) -> int:
    """Count conversations matching filters."""
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
        has_tool_use=has_tool_use,
        has_thinking=has_thinking,
        min_messages=min_messages,
        max_messages=max_messages,
        min_words=min_words,
        has_file_ops=has_file_ops,
        has_git_ops=has_git_ops,
        has_subagent=has_subagent,
    )
    if use_stats_join:
        sql = f"SELECT COUNT(*) as cnt FROM conversations c LEFT JOIN conversation_stats cs ON cs.conversation_id = c.conversation_id {where_sql}"
    else:
        sql = f"SELECT COUNT(*) as cnt FROM conversations {where_sql}"
    cursor = await conn.execute(sql, tuple(params))
    row = await cursor.fetchone()
    return int(row["cnt"])


async def conversation_exists_by_hash(conn: aiosqlite.Connection, content_hash: str) -> bool:
    """Check if a conversation with given content hash exists."""
    cursor = await conn.execute(
        "SELECT 1 FROM conversations WHERE content_hash = ? LIMIT 1",
        (content_hash,),
    )
    row = await cursor.fetchone()
    return row is not None


async def save_conversation_record(
    conn: aiosqlite.Connection,
    record: ConversationRecord,
    transaction_depth: int,
) -> None:
    """Persist a conversation record with upsert semantics."""
    await conn.execute(
        """
        INSERT INTO conversations (
            conversation_id,
            provider_name,
            provider_conversation_id,
            title,
            created_at,
            updated_at,
            sort_key,
            content_hash,
            provider_meta,
            metadata,
            version,
            parent_conversation_id,
            branch_type,
            raw_id
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(conversation_id) DO UPDATE SET
            title = excluded.title,
            created_at = excluded.created_at,
            updated_at = excluded.updated_at,
            sort_key = excluded.sort_key,
            content_hash = excluded.content_hash,
            provider_meta = excluded.provider_meta,
            parent_conversation_id = excluded.parent_conversation_id,
            branch_type = excluded.branch_type,
            raw_id = COALESCE(excluded.raw_id, conversations.raw_id)
        WHERE
            content_hash != excluded.content_hash
            OR IFNULL(title, '') != IFNULL(excluded.title, '')
            OR IFNULL(created_at, '') != IFNULL(excluded.created_at, '')
            OR IFNULL(updated_at, '') != IFNULL(excluded.updated_at, '')
            OR IFNULL(provider_meta, '') != IFNULL(excluded.provider_meta, '')
            OR IFNULL(parent_conversation_id, '') != IFNULL(excluded.parent_conversation_id, '')
            OR IFNULL(branch_type, '') != IFNULL(excluded.branch_type, '')
            OR IFNULL(raw_id, '') != IFNULL(excluded.raw_id, '')
        """,
        (
            record.conversation_id,
            record.provider_name,
            record.provider_conversation_id,
            record.title,
            record.created_at,
            record.updated_at,
            record.sort_key,
            record.content_hash,
            _json_or_none(record.provider_meta),
            _json_or_none(record.metadata) or "{}",
            record.version,
            record.parent_conversation_id,
            record.branch_type,
            record.raw_id,
        ),
    )
    if transaction_depth == 0:
        await conn.commit()


async def list_conversations_by_parent(
    conn: aiosqlite.Connection, parent_id: str
) -> list[ConversationRecord]:
    """List all conversations with the given parent ID."""
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


async def resolve_id(conn: aiosqlite.Connection, id_prefix: str) -> str | None:
    """Resolve a partial conversation ID to a full ID."""
    cursor = await conn.execute(
        "SELECT conversation_id FROM conversations WHERE conversation_id = ?",
        (id_prefix,),
    )
    row = await cursor.fetchone()
    if row:
        return str(row["conversation_id"])

    cursor = await conn.execute(
        "SELECT conversation_id FROM conversations WHERE conversation_id LIKE ? LIMIT 2",
        (f"{id_prefix}%",),
    )
    rows = await cursor.fetchall()
    if len(rows) == 1:
        return str(rows[0]["conversation_id"])
    return None


async def get_last_sync_timestamp(conn: aiosqlite.Connection) -> str | None:
    """Return timestamp of most recent ingestion run, or None."""
    cursor = await conn.execute("SELECT MAX(timestamp) as last FROM runs")
    row = await cursor.fetchone()
    return row["last"] if row and row["last"] else None


def conversation_id_query(
    *,
    source_names: list[str] | None = None,
) -> tuple[str, tuple[str, ...]]:
    """Build the canonical scoped conversation-ID query."""
    predicate, params = _build_source_scope_filter(
        source_names,
        source_column="source_name",
    )
    sql = "SELECT conversation_id FROM conversations"
    if predicate:
        sql += f" WHERE {predicate}"
    sql += " ORDER BY sort_key DESC, conversation_id ASC"
    return sql, tuple(params)


async def count_conversation_ids(
    conn: aiosqlite.Connection,
    *,
    source_names: list[str] | None = None,
) -> int:
    """Count conversation IDs, optionally scoped to source names."""
    predicate, params = _build_source_scope_filter(
        source_names,
        source_column="source_name",
    )
    sql = "SELECT COUNT(*) AS count FROM conversations"
    if predicate:
        sql += f" WHERE {predicate}"
    cursor = await conn.execute(sql, tuple(params))
    row = await cursor.fetchone()
    return int(row["count"]) if row is not None else 0


async def iter_conversation_ids(
    conn: aiosqlite.Connection,
    *,
    source_names: list[str] | None = None,
    page_size: int = 1000,
) -> AsyncIterator[str]:
    """Iterate conversation IDs in bounded fetch batches."""
    sql, params = conversation_id_query(source_names=source_names)
    cursor = await conn.execute(sql, params)
    while True:
        rows = await cursor.fetchmany(page_size)
        if not rows:
            break
        for row in rows:
            yield str(row["conversation_id"])


async def get_metadata(conn: aiosqlite.Connection, conversation_id: str) -> dict[str, object]:
    """Get metadata dict for a conversation."""
    cursor = await conn.execute(
        "SELECT metadata FROM conversations WHERE conversation_id = ?",
        (conversation_id,),
    )
    row = await cursor.fetchone()
    if row is None:
        return {}
    return _parse_json(row["metadata"], field="metadata", record_id=conversation_id) or {}


async def update_metadata_raw(
    conn: aiosqlite.Connection,
    conversation_id: str,
    metadata: dict[str, object],
) -> None:
    """Write metadata dict to a conversation row (used by read-modify-write)."""
    await conn.execute(
        "UPDATE conversations SET metadata = ? WHERE conversation_id = ?",
        (json_dumps(metadata), conversation_id),
    )


async def set_metadata(
    conn: aiosqlite.Connection,
    conversation_id: str,
    metadata: dict[str, object],
    transaction_depth: int,
) -> None:
    """Replace entire metadata dict."""
    await conn.execute(
        "UPDATE conversations SET metadata = ? WHERE conversation_id = ?",
        (json_dumps(metadata), conversation_id),
    )
    if transaction_depth == 0:
        await conn.commit()


async def delete_conversation_sql(
    conn: aiosqlite.Connection,
    conversation_id: str,
    transaction_depth: int,
) -> bool:
    """Delete a conversation and related records. Returns True if deleted."""
    cursor = await conn.execute(
        "SELECT parent_conversation_id FROM conversations WHERE conversation_id = ?",
        (conversation_id,),
    )
    row = await cursor.fetchone()
    if row is None:
        return False

    parent_conversation_id = row[0]

    await conn.execute(
        """
        UPDATE conversations
        SET parent_conversation_id = ?
        WHERE parent_conversation_id = ?
        """,
        (parent_conversation_id, conversation_id),
    )

    cursor = await conn.execute(
        """SELECT DISTINCT ar.attachment_id FROM attachment_refs ar
           JOIN messages m ON ar.message_id = m.message_id
           WHERE m.conversation_id = ?""",
        (conversation_id,),
    )
    affected_attachments = [r[0] for r in await cursor.fetchall()]

    await conn.execute(
        "DELETE FROM conversations WHERE conversation_id = ?",
        (conversation_id,),
    )

    if affected_attachments:
        placeholders = ",".join("?" * len(affected_attachments))
        await conn.execute(
            f"""UPDATE attachments SET ref_count = (
                    SELECT COUNT(*) FROM attachment_refs
                    WHERE attachment_refs.attachment_id = attachments.attachment_id
                ) WHERE attachment_id IN ({placeholders})""",
            affected_attachments,
        )
        await conn.execute(
            f"DELETE FROM attachments WHERE attachment_id IN ({placeholders}) AND ref_count <= 0",
            affected_attachments,
        )

    if transaction_depth == 0:
        await conn.commit()
    return True


async def list_tags(
    conn: aiosqlite.Connection, *, provider: str | None = None
) -> dict[str, int]:
    """List all tags with counts."""
    where = "WHERE metadata IS NOT NULL AND json_extract(metadata, '$.tags') IS NOT NULL"
    params: tuple[str, ...] = ()
    if provider:
        where += " AND provider_name = ?"
        params = (provider,)
    cursor = await conn.execute(
        f"""
        SELECT tag.value AS tag_name, COUNT(*) AS cnt
        FROM conversations,
             json_each(json_extract(metadata, '$.tags')) AS tag
        {where}
        GROUP BY tag.value
        ORDER BY cnt DESC
        """,
        params,
    )
    rows = await cursor.fetchall()
    return {row["tag_name"]: row["cnt"] for row in rows}


async def search_conversations(
    conn: aiosqlite.Connection,
    query: str,
    limit: int = 100,
    providers: list[str] | None = None,
) -> list[str]:
    """Search conversations using FTS5."""
    from polylogue.storage.search import build_ranked_conversation_search_query

    cursor = await conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='messages_fts'"
    )
    exists = await cursor.fetchone()
    if not exists:
        from polylogue.errors import DatabaseError

        raise DatabaseError("Search index not built. Run indexing first or use a different backend.")

    query_spec = build_ranked_conversation_search_query(
        query=query,
        limit=limit,
        scope_names=providers,
    )
    if query_spec is None:
        return []

    sql, params = query_spec
    cursor = await conn.execute(sql, params)
    rows = await cursor.fetchall()
    return [str(row["conversation_id"]) for row in rows]
