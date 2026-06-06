"""Session identity, scope, and metadata helpers."""

from __future__ import annotations

from collections.abc import AsyncIterator

import aiosqlite

from polylogue.core.json import JSONDocument, json_document
from polylogue.core.json import dumps as json_dumps
from polylogue.storage.sqlite.connection import _build_source_scope_filter
from polylogue.storage.sqlite.queries.mappers import _parse_json


async def resolve_id(conn: aiosqlite.Connection, id_prefix: str, *, strict: bool = False) -> str | None:
    cursor = await conn.execute(
        "SELECT session_id FROM sessions WHERE session_id = ?",
        (id_prefix,),
    )
    row = await cursor.fetchone()
    if row:
        return str(row["session_id"])

    if strict:
        return None

    cursor = await conn.execute(
        "SELECT session_id FROM sessions WHERE session_id LIKE ? LIMIT 2",
        (f"{id_prefix}%",),
    )
    rows = list(await cursor.fetchall())
    if len(rows) == 1:
        return str(rows[0]["session_id"])
    return None


async def get_last_sync_timestamp(conn: aiosqlite.Connection) -> str | None:
    cursor = await conn.execute("SELECT MAX(updated_at) as last FROM sessions")
    row = await cursor.fetchone()
    return row["last"] if row and row["last"] else None


def session_id_query(
    *,
    source_names: list[str] | None = None,
) -> tuple[str, tuple[str, ...]]:
    predicate, params = _build_source_scope_filter(
        source_names,
        source_column="source_name",
    )
    sql = "SELECT session_id FROM sessions"
    if predicate:
        sql += f" WHERE {predicate}"
    sql += " ORDER BY sort_key DESC, session_id ASC"
    return sql, tuple(params)


async def count_session_ids(
    conn: aiosqlite.Connection,
    *,
    source_names: list[str] | None = None,
) -> int:
    predicate, params = _build_source_scope_filter(
        source_names,
        source_column="source_name",
    )
    sql = "SELECT COUNT(*) AS count FROM sessions"
    if predicate:
        sql += f" WHERE {predicate}"
    cursor = await conn.execute(sql, tuple(params))
    row = await cursor.fetchone()
    return int(row["count"]) if row is not None else 0


async def iter_session_ids(
    conn: aiosqlite.Connection,
    *,
    source_names: list[str] | None = None,
    page_size: int = 1000,
) -> AsyncIterator[str]:
    sql, params = session_id_query(source_names=source_names)
    cursor = await conn.execute(sql, params)
    while True:
        rows = list(await cursor.fetchmany(page_size))
        if not rows:
            break
        for row in rows:
            yield str(row["session_id"])


async def get_metadata(conn: aiosqlite.Connection, session_id: str) -> JSONDocument:
    cursor = await conn.execute(
        "SELECT metadata FROM sessions WHERE session_id = ?",
        (session_id,),
    )
    row = await cursor.fetchone()
    if row is None:
        return {}
    return json_document(_parse_json(row["metadata"], field="metadata", record_id=session_id))


async def update_metadata_raw(
    conn: aiosqlite.Connection,
    session_id: str,
    metadata: JSONDocument,
) -> None:
    await conn.execute(
        "UPDATE sessions SET metadata = ? WHERE session_id = ?",
        (json_dumps(metadata), session_id),
    )


async def set_metadata(
    conn: aiosqlite.Connection,
    session_id: str,
    metadata: JSONDocument,
    transaction_depth: int,
) -> None:
    await conn.execute(
        "UPDATE sessions SET metadata = ? WHERE session_id = ?",
        (json_dumps(metadata), session_id),
    )
    if transaction_depth == 0:
        await conn.commit()


async def list_tags(conn: aiosqlite.Connection, *, provider: str | None = None) -> dict[str, int]:
    """List all tags with session counts, optionally filtered by provider.

    Reads from the normalized ``tags`` + ``session_tags`` tables.
    Falls back to JSON metadata extraction when the normalized tables are empty
    (e.g. before rebuild has run).
    """
    params: tuple[str, ...] = ()
    join = "JOIN sessions c ON ct.session_id = c.session_id"
    where = ""
    if provider:
        where = " AND c.source_name = ?"
        params = (provider,)
    cursor = await conn.execute(
        f"""
        SELECT t.name AS tag_name, COUNT(DISTINCT ct.session_id) AS cnt
        FROM session_tags ct
        JOIN tags t ON t.id = ct.tag_id
        {join}
        WHERE 1=1{where}
        GROUP BY t.name
        ORDER BY cnt DESC
        """,
        params,
    )
    rows = list(await cursor.fetchall())
    # #1240: M2M (tags + session_tags) is the canonical read surface;
    # the fallback JSON metadata fallback was removed with SCHEMA_VERSION 3.
    return {row["tag_name"]: row["cnt"] for row in rows}


# ---------------------------------------------------------------------------
# User marks
# ---------------------------------------------------------------------------


def user_state_identity_key(
    *,
    target_type: str,
    session_id: str,
    message_id: str | None,
) -> str:
    """Return the stable identity key used to repoint marks/annotations.

    Matches ``TargetRefPayload`` (``polylogue/surfaces/payloads.py``) so that
    surface-level and storage-level identity tokens are wire-compatible. The
    underlying provider IDs are deterministic across reimport, so identity_key
    survives reset+reingest of the same logical session (#1114).
    """
    if target_type == "message":
        if not message_id:
            raise ValueError("message_id is required for target_type='message'")
        return f"message:{session_id}:{message_id}"
    return f"session:{session_id}"


async def add_mark(
    conn: aiosqlite.Connection,
    *,
    target_type: str,
    target_id: str,
    session_id: str,
    mark_type: str,
    created_at: str,
    message_id: str | None = None,
) -> bool:
    """Add a mark to a target. Returns True if newly inserted."""
    identity_key = user_state_identity_key(
        target_type=target_type,
        session_id=session_id,
        message_id=message_id,
    )
    cursor = await conn.execute(
        """
        INSERT OR IGNORE INTO user_marks (
            target_type,
            target_id,
            identity_key,
            session_id,
            message_id,
            mark_type,
            created_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (target_type, target_id, identity_key, session_id, message_id, mark_type, created_at),
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
    session_id: str | None = None,
    target_type: str | None = None,
    target_id: str | None = None,
    message_id: str | None = None,
) -> list[dict[str, str]]:
    """List marks, optionally filtered by type, target, session, or message."""
    where: list[str] = []
    params: list[str] = []
    if mark_type:
        where.append("mark_type = ?")
        params.append(mark_type)
    if session_id:
        where.append("session_id = ?")
        params.append(session_id)
    if target_type:
        where.append("target_type = ?")
        params.append(target_type)
    if target_id:
        where.append("target_id = ?")
        params.append(target_id)
    if message_id:
        where.append("message_id = ?")
        params.append(message_id)
    sql = "SELECT target_type, target_id, session_id, message_id, mark_type, created_at FROM user_marks"
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += " ORDER BY created_at DESC"
    cursor = await conn.execute(sql, tuple(params))
    rows = await cursor.fetchall()
    return [
        {
            "target_type": r["target_type"],
            "target_id": r["target_id"],
            "session_id": r["session_id"],
            "message_id": r["message_id"] or "",
            "mark_type": r["mark_type"],
            "created_at": r["created_at"],
        }
        for r in rows
    ]


async def repoint_user_state_by_identity(
    conn: aiosqlite.Connection,
    session_id: str,
) -> tuple[int, int]:
    """Rebind orphan marks/annotations to a (re-)imported session.

    Implements the read side of #1114: identity_key survives session row
    deletion (FK cascade is ``SET NULL``), and once a logically identical
    session is re-imported its ``session_id``/``message_id`` are
    deterministic, so we can restore the resolved pointers by matching the
    stored identity_key against the current archive state.

    Returns ``(marks_repointed, annotations_repointed)``.
    """
    conv_key = f"session:{session_id}"
    msg_key_prefix = f"message:{session_id}:"

    # Session-target rebinds (target_type='session').
    cursor = await conn.execute(
        """
        UPDATE user_marks
        SET session_id = ?
        WHERE identity_key = ?
          AND target_type = 'session'
          AND (session_id IS NULL OR session_id != ?)
        """,
        (session_id, conv_key, session_id),
    )
    marks_conv = cursor.rowcount or 0
    cursor = await conn.execute(
        """
        UPDATE user_annotations
        SET session_id = ?
        WHERE identity_key = ?
          AND target_type = 'session'
          AND (session_id IS NULL OR session_id != ?)
        """,
        (session_id, conv_key, session_id),
    )
    ann_conv = cursor.rowcount or 0

    # Message-target rebinds: only repoint rows whose message_id is still a
    # member of the freshly written session. Messages that disappeared
    # from the reimported session stay orphaned (NULL session_id /
    # message_id) so callers can distinguish "rebound" from "orphaned by
    # message drift" without scanning the full annotation table.
    cursor = await conn.execute(
        """
        UPDATE user_marks
        SET session_id = ?, message_id = target_id
        WHERE identity_key LIKE ?
          AND target_type = 'message'
          AND EXISTS (
              SELECT 1 FROM messages m
              WHERE m.message_id = user_marks.target_id
                AND m.session_id = ?
          )
          AND (session_id IS NULL OR message_id IS NULL)
        """,
        (session_id, f"{msg_key_prefix}%", session_id),
    )
    marks_msg = cursor.rowcount or 0
    cursor = await conn.execute(
        """
        UPDATE user_annotations
        SET session_id = ?, message_id = target_id
        WHERE identity_key LIKE ?
          AND target_type = 'message'
          AND EXISTS (
              SELECT 1 FROM messages m
              WHERE m.message_id = user_annotations.target_id
                AND m.session_id = ?
          )
          AND (session_id IS NULL OR message_id IS NULL)
        """,
        (session_id, f"{msg_key_prefix}%", session_id),
    )
    ann_msg = cursor.rowcount or 0

    return marks_conv + marks_msg, ann_conv + ann_msg


# ---------------------------------------------------------------------------
# User annotations
# ---------------------------------------------------------------------------


async def save_annotation(
    conn: aiosqlite.Connection,
    *,
    annotation_id: str,
    target_type: str,
    target_id: str,
    session_id: str,
    note_text: str,
    now: str,
    message_id: str | None = None,
) -> bool:
    """Insert or update an annotation. Returns True if inserted."""
    identity_key = user_state_identity_key(
        target_type=target_type,
        session_id=session_id,
        message_id=message_id,
    )
    cursor = await conn.execute("SELECT 1 FROM user_annotations WHERE annotation_id = ?", (annotation_id,))
    exists = await cursor.fetchone() is not None
    await conn.execute(
        """
        INSERT INTO user_annotations (
            annotation_id,
            target_type,
            target_id,
            identity_key,
            session_id,
            message_id,
            note_text,
            created_at,
            updated_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(annotation_id) DO UPDATE SET
            target_type = excluded.target_type,
            target_id = excluded.target_id,
            identity_key = excluded.identity_key,
            session_id = excluded.session_id,
            message_id = excluded.message_id,
            note_text = excluded.note_text,
            updated_at = excluded.updated_at
        """,
        (annotation_id, target_type, target_id, identity_key, session_id, message_id, note_text, now, now),
    )
    await conn.commit()
    return not exists


async def get_annotation(conn: aiosqlite.Connection, annotation_id: str) -> dict[str, str] | None:
    """Get an annotation by ID."""
    cursor = await conn.execute(
        """
        SELECT annotation_id, target_type, target_id, session_id, message_id, note_text, created_at, updated_at
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
    session_id: str | None = None,
    target_type: str | None = None,
    target_id: str | None = None,
    message_id: str | None = None,
) -> list[dict[str, str]]:
    """List annotations, optionally filtered by target, session, or message."""
    where: list[str] = []
    params: list[str] = []
    if session_id:
        where.append("session_id = ?")
        params.append(session_id)
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
        "SELECT annotation_id, target_type, target_id, session_id, message_id, note_text, created_at, updated_at "
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
        "session_id": row["session_id"],
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
    session_ids_json: str,
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
        "INSERT OR REPLACE INTO recall_packs (pack_id, label, session_ids_json, payload_json, created_at) "
        "VALUES (?, ?, ?, ?, ?)",
        (pack_id, label, session_ids_json, payload_json, created_at),
    )
    await conn.commit()
    return not exists


async def get_recall_pack(conn: aiosqlite.Connection, pack_id: str) -> dict[str, str] | None:
    """Get a recall pack by ID."""
    cursor = await conn.execute(
        "SELECT pack_id, label, session_ids_json, payload_json, created_at FROM recall_packs WHERE pack_id = ?",
        (pack_id,),
    )
    row = await cursor.fetchone()
    if row is None:
        return None
    return {
        "pack_id": row["pack_id"],
        "label": row["label"],
        "session_ids_json": row["session_ids_json"],
        "payload_json": row["payload_json"],
        "created_at": row["created_at"],
    }


async def list_recall_packs(conn: aiosqlite.Connection) -> list[dict[str, str]]:
    """List all recall packs."""
    cursor = await conn.execute(
        "SELECT pack_id, label, session_ids_json, payload_json, created_at FROM recall_packs ORDER BY created_at DESC"
    )
    rows = await cursor.fetchall()
    return [
        {
            "pack_id": r["pack_id"],
            "label": r["label"],
            "session_ids_json": r["session_ids_json"],
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


# ---------------------------------------------------------------------------
# Reader workspaces
# ---------------------------------------------------------------------------


async def save_workspace(
    conn: aiosqlite.Connection,
    *,
    workspace_id: str,
    name: str,
    mode: str,
    open_targets_json: str,
    layout_json: str,
    active_target_json: str,
    now: str,
) -> bool:
    """Insert or update a reader workspace. Returns True if inserted."""
    cursor = await conn.execute("SELECT 1 FROM reader_workspaces WHERE workspace_id = ?", (workspace_id,))
    exists = await cursor.fetchone() is not None
    await conn.execute(
        """
        INSERT INTO reader_workspaces (
            workspace_id,
            name,
            mode,
            open_targets_json,
            layout_json,
            active_target_json,
            created_at,
            updated_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(workspace_id) DO UPDATE SET
            name = excluded.name,
            mode = excluded.mode,
            open_targets_json = excluded.open_targets_json,
            layout_json = excluded.layout_json,
            active_target_json = excluded.active_target_json,
            updated_at = excluded.updated_at
        """,
        (workspace_id, name, mode, open_targets_json, layout_json, active_target_json, now, now),
    )
    await conn.commit()
    return not exists


async def get_workspace(conn: aiosqlite.Connection, workspace_id: str) -> dict[str, str] | None:
    cursor = await conn.execute(
        """
        SELECT workspace_id, name, mode, open_targets_json, layout_json, active_target_json, created_at, updated_at
        FROM reader_workspaces
        WHERE workspace_id = ?
        """,
        (workspace_id,),
    )
    row = await cursor.fetchone()
    return _workspace_row(row) if row is not None else None


async def list_workspaces(conn: aiosqlite.Connection) -> list[dict[str, str]]:
    cursor = await conn.execute(
        """
        SELECT workspace_id, name, mode, open_targets_json, layout_json, active_target_json, created_at, updated_at
        FROM reader_workspaces
        ORDER BY updated_at DESC, workspace_id ASC
        """
    )
    rows = await cursor.fetchall()
    return [_workspace_row(row) for row in rows]


async def delete_workspace(conn: aiosqlite.Connection, workspace_id: str) -> bool:
    cursor = await conn.execute("DELETE FROM reader_workspaces WHERE workspace_id = ?", (workspace_id,))
    await conn.commit()
    return cursor.rowcount > 0


def _workspace_row(row: aiosqlite.Row) -> dict[str, str]:
    return {
        "workspace_id": row["workspace_id"],
        "name": row["name"],
        "mode": row["mode"],
        "open_targets_json": row["open_targets_json"],
        "layout_json": row["layout_json"],
        "active_target_json": row["active_target_json"],
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }


__all__ = [
    "add_mark",
    "session_id_query",
    "count_session_ids",
    "delete_annotation",
    "delete_recall_pack",
    "delete_view",
    "delete_workspace",
    "get_annotation",
    "get_last_sync_timestamp",
    "get_metadata",
    "get_recall_pack",
    "get_view",
    "get_view_by_name",
    "get_workspace",
    "iter_session_ids",
    "list_marks",
    "list_annotations",
    "list_recall_packs",
    "list_tags",
    "list_views",
    "list_workspaces",
    "remove_mark",
    "repoint_user_state_by_identity",
    "save_annotation",
    "resolve_id",
    "save_recall_pack",
    "save_view",
    "save_workspace",
    "set_metadata",
    "update_metadata_raw",
    "user_state_identity_key",
]
