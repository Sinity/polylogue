"""Conversation identity, scope, and metadata helpers."""

from __future__ import annotations

from collections.abc import AsyncIterator

import aiosqlite

from polylogue.core.json import JSONDocument, json_document
from polylogue.core.json import dumps as json_dumps
from polylogue.storage.sqlite.connection import _build_source_scope_filter
from polylogue.storage.sqlite.queries.mappers import _parse_json


async def resolve_id(conn: aiosqlite.Connection, id_prefix: str, *, strict: bool = False) -> str | None:
    cursor = await conn.execute(
        "SELECT conversation_id FROM conversations WHERE conversation_id = ?",
        (id_prefix,),
    )
    row = await cursor.fetchone()
    if row:
        return str(row["conversation_id"])

    if strict:
        return None

    cursor = await conn.execute(
        "SELECT conversation_id FROM conversations WHERE conversation_id LIKE ? LIMIT 2",
        (f"{id_prefix}%",),
    )
    rows = list(await cursor.fetchall())
    if len(rows) == 1:
        return str(rows[0]["conversation_id"])
    return None


async def get_last_sync_timestamp(conn: aiosqlite.Connection) -> str | None:
    cursor = await conn.execute("SELECT MAX(updated_at) as last FROM conversations")
    row = await cursor.fetchone()
    return row["last"] if row and row["last"] else None


def conversation_id_query(
    *,
    source_names: list[str] | None = None,
) -> tuple[str, tuple[str, ...]]:
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
    sql, params = conversation_id_query(source_names=source_names)
    cursor = await conn.execute(sql, params)
    while True:
        rows = await cursor.fetchmany(page_size)
        if not rows:
            break
        for row in rows:
            yield str(row["conversation_id"])


async def get_metadata(conn: aiosqlite.Connection, conversation_id: str) -> JSONDocument:
    cursor = await conn.execute(
        "SELECT metadata FROM conversations WHERE conversation_id = ?",
        (conversation_id,),
    )
    row = await cursor.fetchone()
    if row is None:
        return {}
    return json_document(_parse_json(row["metadata"], field="metadata", record_id=conversation_id))


async def update_metadata_raw(
    conn: aiosqlite.Connection,
    conversation_id: str,
    metadata: JSONDocument,
) -> None:
    await conn.execute(
        "UPDATE conversations SET metadata = ? WHERE conversation_id = ?",
        (json_dumps(metadata), conversation_id),
    )


async def set_metadata(
    conn: aiosqlite.Connection,
    conversation_id: str,
    metadata: JSONDocument,
    transaction_depth: int,
) -> None:
    await conn.execute(
        "UPDATE conversations SET metadata = ? WHERE conversation_id = ?",
        (json_dumps(metadata), conversation_id),
    )
    if transaction_depth == 0:
        await conn.commit()


async def list_tags(conn: aiosqlite.Connection, *, provider: str | None = None) -> dict[str, int]:
    """List all tags with conversation counts, optionally filtered by provider.

    Reads from the normalized ``tags`` + ``conversation_tags`` tables.
    Falls back to JSON metadata extraction when the normalized tables are empty
    (e.g. before migration has run).
    """
    params: tuple[str, ...] = ()
    join = "JOIN conversations c ON ct.conversation_id = c.conversation_id"
    where = ""
    if provider:
        where = " AND c.provider_name = ?"
        params = (provider,)
    cursor = await conn.execute(
        f"""
        SELECT t.name AS tag_name, COUNT(DISTINCT ct.conversation_id) AS cnt
        FROM conversation_tags ct
        JOIN tags t ON t.id = ct.tag_id
        {join}
        WHERE 1=1{where}
        GROUP BY t.name
        ORDER BY cnt DESC
        """,
        params,
    )
    rows = await cursor.fetchall()
    if rows:
        return {row["tag_name"]: row["cnt"] for row in rows}

    # Fallback: read from JSON metadata for archives that haven't been migrated yet
    json_where = "WHERE metadata IS NOT NULL AND json_extract(metadata, '$.tags') IS NOT NULL"
    json_params: tuple[str, ...] = ()
    if provider:
        json_where += " AND provider_name = ?"
        json_params = (provider,)
    cursor = await conn.execute(
        f"""
        SELECT tag.value AS tag_name, COUNT(*) AS cnt
        FROM conversations,
             json_each(json_extract(metadata, '$.tags')) AS tag
        {json_where}
        GROUP BY tag.value
        ORDER BY cnt DESC
        """,
        json_params,
    )
    rows = await cursor.fetchall()
    return {row["tag_name"]: row["cnt"] for row in rows}


# ---------------------------------------------------------------------------
# User marks
# ---------------------------------------------------------------------------


async def add_mark(
    conn: aiosqlite.Connection,
    *,
    target_type: str,
    target_id: str,
    conversation_id: str,
    mark_type: str,
    created_at: str,
    message_id: str | None = None,
) -> bool:
    """Add a mark to a target. Returns True if newly inserted."""
    cursor = await conn.execute(
        """
        INSERT OR IGNORE INTO user_marks (
            target_type,
            target_id,
            conversation_id,
            message_id,
            mark_type,
            created_at
        )
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (target_type, target_id, conversation_id, message_id, mark_type, created_at),
    )
    await conn.commit()
    return cursor.rowcount > 0


async def remove_mark(conn: aiosqlite.Connection, target_type: str, target_id: str, mark_type: str) -> bool:
    """Remove a mark from a target. Returns True if something was deleted."""
    cursor = await conn.execute(
        "DELETE FROM user_marks WHERE target_type = ? AND target_id = ? AND mark_type = ?",
        (target_type, target_id, mark_type),
    )
    await conn.commit()
    return cursor.rowcount > 0


async def list_marks(
    conn: aiosqlite.Connection,
    *,
    mark_type: str | None = None,
    conversation_id: str | None = None,
    target_type: str | None = None,
    target_id: str | None = None,
    message_id: str | None = None,
) -> list[dict[str, str]]:
    """List marks, optionally filtered by type, target, conversation, or message."""
    where: list[str] = []
    params: list[str] = []
    if mark_type:
        where.append("mark_type = ?")
        params.append(mark_type)
    if conversation_id:
        where.append("conversation_id = ?")
        params.append(conversation_id)
    if target_type:
        where.append("target_type = ?")
        params.append(target_type)
    if target_id:
        where.append("target_id = ?")
        params.append(target_id)
    if message_id:
        where.append("message_id = ?")
        params.append(message_id)
    sql = "SELECT target_type, target_id, conversation_id, message_id, mark_type, created_at FROM user_marks"
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += " ORDER BY created_at DESC"
    cursor = await conn.execute(sql, tuple(params))
    rows = await cursor.fetchall()
    return [
        {
            "target_type": r["target_type"],
            "target_id": r["target_id"],
            "conversation_id": r["conversation_id"],
            "message_id": r["message_id"] or "",
            "mark_type": r["mark_type"],
            "created_at": r["created_at"],
        }
        for r in rows
    ]


# ---------------------------------------------------------------------------
# User annotations
# ---------------------------------------------------------------------------


async def save_annotation(
    conn: aiosqlite.Connection,
    *,
    annotation_id: str,
    target_type: str,
    target_id: str,
    conversation_id: str,
    note_text: str,
    now: str,
    message_id: str | None = None,
) -> bool:
    """Insert or update an annotation. Returns True if inserted."""
    cursor = await conn.execute("SELECT 1 FROM user_annotations WHERE annotation_id = ?", (annotation_id,))
    exists = await cursor.fetchone() is not None
    await conn.execute(
        """
        INSERT INTO user_annotations (
            annotation_id,
            target_type,
            target_id,
            conversation_id,
            message_id,
            note_text,
            created_at,
            updated_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(annotation_id) DO UPDATE SET
            target_type = excluded.target_type,
            target_id = excluded.target_id,
            conversation_id = excluded.conversation_id,
            message_id = excluded.message_id,
            note_text = excluded.note_text,
            updated_at = excluded.updated_at
        """,
        (annotation_id, target_type, target_id, conversation_id, message_id, note_text, now, now),
    )
    await conn.commit()
    return not exists


async def get_annotation(conn: aiosqlite.Connection, annotation_id: str) -> dict[str, str] | None:
    """Get an annotation by ID."""
    cursor = await conn.execute(
        """
        SELECT annotation_id, target_type, target_id, conversation_id, message_id, note_text, created_at, updated_at
        FROM user_annotations
        WHERE annotation_id = ?
        """,
        (annotation_id,),
    )
    row = await cursor.fetchone()
    return _annotation_row(row) if row is not None else None


async def list_annotations(
    conn: aiosqlite.Connection,
    *,
    conversation_id: str | None = None,
    target_type: str | None = None,
    target_id: str | None = None,
    message_id: str | None = None,
) -> list[dict[str, str]]:
    """List annotations, optionally filtered by target, conversation, or message."""
    where: list[str] = []
    params: list[str] = []
    if conversation_id:
        where.append("conversation_id = ?")
        params.append(conversation_id)
    if target_type:
        where.append("target_type = ?")
        params.append(target_type)
    if target_id:
        where.append("target_id = ?")
        params.append(target_id)
    if message_id:
        where.append("message_id = ?")
        params.append(message_id)
    sql = (
        "SELECT annotation_id, target_type, target_id, conversation_id, message_id, note_text, created_at, updated_at "
        "FROM user_annotations"
    )
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += " ORDER BY updated_at DESC"
    cursor = await conn.execute(sql, tuple(params))
    rows = await cursor.fetchall()
    return [_annotation_row(row) for row in rows]


async def delete_annotation(conn: aiosqlite.Connection, annotation_id: str) -> bool:
    """Delete an annotation. Returns True if something was deleted."""
    cursor = await conn.execute("DELETE FROM user_annotations WHERE annotation_id = ?", (annotation_id,))
    await conn.commit()
    return cursor.rowcount > 0


def _annotation_row(row: aiosqlite.Row) -> dict[str, str]:
    return {
        "annotation_id": row["annotation_id"],
        "target_type": row["target_type"],
        "target_id": row["target_id"],
        "conversation_id": row["conversation_id"],
        "message_id": row["message_id"] or "",
        "note_text": row["note_text"],
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }


# ---------------------------------------------------------------------------
# Saved views
# ---------------------------------------------------------------------------


async def save_view(conn: aiosqlite.Connection, view_id: str, name: str, query_json: str, created_at: str) -> bool:
    """Insert or replace a saved view. Returns True if inserted (not updated)."""
    cursor = await conn.execute(
        "SELECT 1 FROM saved_views WHERE view_id = ?",
        (view_id,),
    )
    exists = await cursor.fetchone() is not None
    await conn.execute(
        "INSERT OR REPLACE INTO saved_views (view_id, name, query_json, created_at) VALUES (?, ?, ?, ?)",
        (view_id, name, query_json, created_at),
    )
    await conn.commit()
    return not exists


async def get_view(conn: aiosqlite.Connection, view_id: str) -> dict[str, str] | None:
    """Get a saved view by ID."""
    cursor = await conn.execute(
        "SELECT view_id, name, query_json, created_at FROM saved_views WHERE view_id = ?",
        (view_id,),
    )
    row = await cursor.fetchone()
    if row is None:
        return None
    return {
        "view_id": row["view_id"],
        "name": row["name"],
        "query_json": row["query_json"],
        "created_at": row["created_at"],
    }


async def get_view_by_name(conn: aiosqlite.Connection, name: str) -> dict[str, str] | None:
    """Get a saved view by name."""
    cursor = await conn.execute(
        "SELECT view_id, name, query_json, created_at FROM saved_views WHERE name = ?",
        (name,),
    )
    row = await cursor.fetchone()
    if row is None:
        return None
    return {
        "view_id": row["view_id"],
        "name": row["name"],
        "query_json": row["query_json"],
        "created_at": row["created_at"],
    }


async def list_views(conn: aiosqlite.Connection) -> list[dict[str, str]]:
    """List all saved views."""
    cursor = await conn.execute(
        "SELECT view_id, name, query_json, created_at FROM saved_views ORDER BY created_at DESC"
    )
    rows = await cursor.fetchall()
    return [
        {
            "view_id": r["view_id"],
            "name": r["name"],
            "query_json": r["query_json"],
            "created_at": r["created_at"],
        }
        for r in rows
    ]


async def delete_view(conn: aiosqlite.Connection, view_id: str) -> bool:
    """Delete a saved view. Returns True if something was deleted."""
    cursor = await conn.execute("DELETE FROM saved_views WHERE view_id = ?", (view_id,))
    await conn.commit()
    return cursor.rowcount > 0


# ---------------------------------------------------------------------------
# Recall packs
# ---------------------------------------------------------------------------


async def save_recall_pack(
    conn: aiosqlite.Connection,
    pack_id: str,
    label: str,
    conversation_ids_json: str,
    payload_json: str,
    created_at: str,
) -> bool:
    """Insert or replace a recall pack. Returns True if inserted (not updated)."""
    cursor = await conn.execute(
        "SELECT 1 FROM recall_packs WHERE pack_id = ?",
        (pack_id,),
    )
    exists = await cursor.fetchone() is not None
    await conn.execute(
        "INSERT OR REPLACE INTO recall_packs (pack_id, label, conversation_ids_json, payload_json, created_at) "
        "VALUES (?, ?, ?, ?, ?)",
        (pack_id, label, conversation_ids_json, payload_json, created_at),
    )
    await conn.commit()
    return not exists


async def get_recall_pack(conn: aiosqlite.Connection, pack_id: str) -> dict[str, str] | None:
    """Get a recall pack by ID."""
    cursor = await conn.execute(
        "SELECT pack_id, label, conversation_ids_json, payload_json, created_at FROM recall_packs WHERE pack_id = ?",
        (pack_id,),
    )
    row = await cursor.fetchone()
    if row is None:
        return None
    return {
        "pack_id": row["pack_id"],
        "label": row["label"],
        "conversation_ids_json": row["conversation_ids_json"],
        "payload_json": row["payload_json"],
        "created_at": row["created_at"],
    }


async def list_recall_packs(conn: aiosqlite.Connection) -> list[dict[str, str]]:
    """List all recall packs."""
    cursor = await conn.execute(
        "SELECT pack_id, label, conversation_ids_json, payload_json, created_at "
        "FROM recall_packs ORDER BY created_at DESC"
    )
    rows = await cursor.fetchall()
    return [
        {
            "pack_id": r["pack_id"],
            "label": r["label"],
            "conversation_ids_json": r["conversation_ids_json"],
            "payload_json": r["payload_json"],
            "created_at": r["created_at"],
        }
        for r in rows
    ]


async def delete_recall_pack(conn: aiosqlite.Connection, pack_id: str) -> bool:
    """Delete a recall pack. Returns True if something was deleted."""
    cursor = await conn.execute("DELETE FROM recall_packs WHERE pack_id = ?", (pack_id,))
    await conn.commit()
    return cursor.rowcount > 0


__all__ = [
    "add_mark",
    "conversation_id_query",
    "count_conversation_ids",
    "delete_annotation",
    "delete_recall_pack",
    "delete_view",
    "get_annotation",
    "get_last_sync_timestamp",
    "get_metadata",
    "get_recall_pack",
    "get_view",
    "get_view_by_name",
    "iter_conversation_ids",
    "list_marks",
    "list_annotations",
    "list_recall_packs",
    "list_tags",
    "list_views",
    "remove_mark",
    "save_annotation",
    "resolve_id",
    "save_recall_pack",
    "save_view",
    "set_metadata",
    "update_metadata_raw",
]
