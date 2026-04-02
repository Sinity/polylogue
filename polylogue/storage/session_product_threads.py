"""Thread row builders, queries, and lifecycle support for session products."""

from __future__ import annotations

import sqlite3
from collections.abc import Sequence

import aiosqlite

from polylogue.lib.threads import WorkThread, build_session_threads
from polylogue.storage.backends.queries.mappers import _row_to_session_profile_record
from polylogue.storage.session_product_profiles import hydrate_session_profile, now_iso
from polylogue.storage.store import SESSION_PRODUCT_MATERIALIZER_VERSION, WorkThreadRecord

_ROOT_THREAD_IDS_SQL = """
    SELECT c.conversation_id
    FROM conversations c
    LEFT JOIN conversations parent ON c.parent_conversation_id = parent.conversation_id
    WHERE parent.conversation_id IS NULL
    ORDER BY c.conversation_id
"""
_THREAD_ROOT_ID_SQL = """
    WITH RECURSIVE ancestors(conversation_id, parent_conversation_id) AS (
        SELECT conversation_id, parent_conversation_id
        FROM conversations
        WHERE conversation_id = ?
        UNION ALL
        SELECT c.conversation_id, c.parent_conversation_id
        FROM conversations c
        JOIN ancestors a ON a.parent_conversation_id = c.conversation_id
    )
    SELECT conversation_id
    FROM ancestors
    WHERE parent_conversation_id IS NULL
    LIMIT 1
"""
_THREAD_ROOT_IDS_SQL_TEMPLATE = """
    WITH RECURSIVE ancestors(target_id, conversation_id, parent_conversation_id) AS (
        SELECT conversation_id, conversation_id, parent_conversation_id
        FROM conversations
        WHERE conversation_id IN ({placeholders})
        UNION ALL
        SELECT a.target_id, c.conversation_id, c.parent_conversation_id
        FROM conversations c
        JOIN ancestors a ON a.parent_conversation_id = c.conversation_id
    )
    SELECT target_id, conversation_id
    FROM ancestors
    WHERE parent_conversation_id IS NULL
"""
_THREAD_CONVERSATION_IDS_SQL = """
    WITH RECURSIVE descendants(conversation_id) AS (
        SELECT conversation_id
        FROM conversations
        WHERE conversation_id = ?
        UNION ALL
        SELECT c.conversation_id
        FROM conversations c
        JOIN descendants d ON c.parent_conversation_id = d.conversation_id
    )
    SELECT conversation_id
    FROM descendants
    ORDER BY conversation_id
"""


# ---------------------------------------------------------------------------
# Row builders and hydration
# ---------------------------------------------------------------------------


def thread_search_text(thread: WorkThread) -> str:
    parts = [
        thread.thread_id,
        thread.root_id,
        thread.dominant_project or "",
        *thread.session_ids,
        *thread.work_event_breakdown.keys(),
    ]
    search_text = " \n".join(part.strip() for part in parts if part and str(part).strip())
    return search_text or thread.thread_id


def build_work_thread_record(
    thread: WorkThread,
    *,
    materialized_at: str | None = None,
) -> WorkThreadRecord:
    built_at = materialized_at or now_iso()
    return WorkThreadRecord(
        thread_id=thread.thread_id,
        root_id=thread.root_id,
        materializer_version=SESSION_PRODUCT_MATERIALIZER_VERSION,
        materialized_at=built_at,
        start_time=thread.start_time.isoformat() if thread.start_time else None,
        end_time=thread.end_time.isoformat() if thread.end_time else None,
        dominant_project=thread.dominant_project,
        session_ids=thread.session_ids,
        session_count=len(thread.session_ids),
        depth=thread.depth,
        branch_count=thread.branch_count,
        total_messages=thread.total_messages,
        total_cost_usd=thread.total_cost_usd,
        wall_duration_ms=thread.wall_duration_ms,
        work_event_breakdown=thread.work_event_breakdown,
        payload=thread.to_dict(),
        search_text=thread_search_text(thread),
    )


def hydrate_work_thread(record: WorkThreadRecord) -> WorkThread:
    return WorkThread.from_dict(record.payload)


# ---------------------------------------------------------------------------
# Thread queries
# ---------------------------------------------------------------------------


def thread_root_id_sync(conn: sqlite3.Connection, conversation_id: str) -> str | None:
    row = conn.execute(_THREAD_ROOT_ID_SQL, (conversation_id,)).fetchone()
    return str(row["conversation_id"]) if row else None


async def thread_root_id_async(conn: aiosqlite.Connection, conversation_id: str) -> str | None:
    row = await (await conn.execute(_THREAD_ROOT_ID_SQL, (conversation_id,))).fetchone()
    return str(row["conversation_id"]) if row else None


async def thread_root_ids_async(
    conn: aiosqlite.Connection,
    conversation_ids: Sequence[str],
) -> dict[str, str]:
    if not conversation_ids:
        return {}
    placeholders = ", ".join("?" for _ in conversation_ids)
    rows = await (
        await conn.execute(
            _THREAD_ROOT_IDS_SQL_TEMPLATE.format(placeholders=placeholders),
            tuple(conversation_ids),
        )
    ).fetchall()
    return {
        str(row["target_id"]): str(row["conversation_id"])
        for row in rows
    }


def thread_conversation_ids_sync(conn: sqlite3.Connection, root_id: str) -> list[str]:
    rows = conn.execute(_THREAD_CONVERSATION_IDS_SQL, (root_id,)).fetchall()
    return [str(row["conversation_id"]) for row in rows]


async def thread_conversation_ids_async(conn: aiosqlite.Connection, root_id: str) -> list[str]:
    rows = await (await conn.execute(_THREAD_CONVERSATION_IDS_SQL, (root_id,))).fetchall()
    return [str(row["conversation_id"]) for row in rows]


def load_thread_profile_records_sync(conn: sqlite3.Connection, root_id: str):
    conversation_ids = thread_conversation_ids_sync(conn, root_id)
    if not conversation_ids:
        return []
    placeholders = ", ".join("?" for _ in conversation_ids)
    rows = conn.execute(
        f"SELECT * FROM session_profiles WHERE conversation_id IN ({placeholders})",
        tuple(conversation_ids),
    ).fetchall()
    return [_row_to_session_profile_record(row) for row in rows]


async def load_thread_profile_records_async(conn: aiosqlite.Connection, root_id: str):
    conversation_ids = await thread_conversation_ids_async(conn, root_id)
    if not conversation_ids:
        return []
    placeholders = ", ".join("?" for _ in conversation_ids)
    rows = await (
        await conn.execute(
            f"SELECT * FROM session_profiles WHERE conversation_id IN ({placeholders})",
            tuple(conversation_ids),
        )
    ).fetchall()
    return [_row_to_session_profile_record(row) for row in rows]


def build_all_thread_records_sync(conn: sqlite3.Connection) -> list[object]:
    root_ids = [str(row["conversation_id"]) for row in conn.execute(_ROOT_THREAD_IDS_SQL).fetchall()]
    records: list[object] = []
    for root_id in root_ids:
        profile_records = load_thread_profile_records_sync(conn, root_id)
        if not profile_records:
            continue
        profiles = [hydrate_session_profile(record) for record in profile_records]
        threads = build_session_threads(profiles)
        for thread in threads:
            if thread.thread_id == root_id:
                records.append(build_work_thread_record(thread))
                break
    return records


async def build_all_thread_records_async(conn: aiosqlite.Connection) -> list[object]:
    rows = await (await conn.execute(_ROOT_THREAD_IDS_SQL)).fetchall()
    root_ids = [str(row["conversation_id"]) for row in rows]
    records: list[object] = []
    for root_id in root_ids:
        profile_records = await load_thread_profile_records_async(conn, root_id)
        if not profile_records:
            continue
        profiles = [hydrate_session_profile(record) for record in profile_records]
        threads = build_session_threads(profiles)
        for thread in threads:
            if thread.thread_id == root_id:
                records.append(build_work_thread_record(thread))
                break
    return records


__all__ = [
    "build_all_thread_records_async",
    "build_all_thread_records_sync",
    "build_work_thread_record",
    "hydrate_work_thread",
    "load_thread_profile_records_async",
    "load_thread_profile_records_sync",
    "thread_conversation_ids_async",
    "thread_conversation_ids_sync",
    "thread_root_id_async",
    "thread_root_ids_async",
    "thread_root_id_sync",
    "thread_search_text",
]
