"""Thread row builders, queries, and lifecycle support for session insights."""

from __future__ import annotations

import sqlite3
from collections.abc import AsyncIterator, Iterable, Iterator, Sequence

import aiosqlite

from polylogue.archive.session.documents import ThreadDocument
from polylogue.archive.session.threads import Thread, ThreadPayload, build_session_threads
from polylogue.core.sources import source_name_to_origin
from polylogue.core.types import SessionId
from polylogue.insights.archive_models import ThreadPayload as ArchivedThreadPayload
from polylogue.insights.temporal_source import classify_thread_hwm_source
from polylogue.storage.insights.session.profiles import hydrate_session_profile, now_iso
from polylogue.storage.runtime import (
    SESSION_INSIGHT_MATERIALIZER_VERSION,
    SessionProfileRecord,
    ThreadRecord,
)
from polylogue.storage.sqlite.queries.mappers import _row_to_session_profile_record

_ROOT_THREAD_IDS_SQL = """
    SELECT c.session_id
    FROM sessions c
    LEFT JOIN sessions parent ON c.parent_session_id = parent.session_id
    WHERE parent.session_id IS NULL
    ORDER BY c.session_id
"""
_THREAD_ROOT_ID_SQL = """
    WITH RECURSIVE ancestors(session_id, parent_session_id) AS (
        SELECT session_id, parent_session_id
        FROM sessions
        WHERE session_id = ?
        UNION ALL
        SELECT c.session_id, c.parent_session_id
        FROM sessions c
        JOIN ancestors a ON a.parent_session_id = c.session_id
    )
    SELECT session_id
    FROM ancestors
    WHERE parent_session_id IS NULL
    LIMIT 1
"""
_THREAD_ROOT_IDS_SQL_TEMPLATE = """
    WITH RECURSIVE ancestors(target_id, session_id, parent_session_id) AS (
        SELECT session_id, session_id, parent_session_id
        FROM sessions
        WHERE session_id IN ({placeholders})
        UNION ALL
        SELECT a.target_id, c.session_id, c.parent_session_id
        FROM sessions c
        JOIN ancestors a ON a.parent_session_id = c.session_id
    )
    SELECT target_id, session_id
    FROM ancestors
    WHERE parent_session_id IS NULL
"""
_THREAD_SESSION_IDS_SQL = """
    WITH RECURSIVE descendants(session_id) AS (
        SELECT session_id
        FROM sessions
        WHERE session_id = ?
        UNION ALL
        SELECT c.session_id
        FROM sessions c
        JOIN descendants d ON c.parent_session_id = d.session_id
    )
    SELECT session_id
    FROM descendants
    ORDER BY session_id
"""
_THREAD_PROFILE_RECORDS_BY_ROOT_SQL_TEMPLATE = """
    WITH RECURSIVE descendants(root_id, session_id) AS (
        SELECT session_id, session_id
        FROM sessions
        WHERE session_id IN ({placeholders})
        UNION ALL
        SELECT d.root_id, c.session_id
        FROM sessions c
        JOIN descendants d ON c.parent_session_id = d.session_id
    )
    SELECT d.root_id, sp.*
    FROM descendants d
    JOIN session_profiles sp ON sp.session_id = d.session_id
    ORDER BY d.root_id, COALESCE(sp.source_sort_key, 0) DESC, sp.session_id
"""
_ROOT_BATCH_SIZE = 200


# ---------------------------------------------------------------------------
# Row builders and hydration
# ---------------------------------------------------------------------------


def thread_search_text(thread: Thread) -> str:
    parts = [
        thread.thread_id,
        thread.root_id,
        thread.dominant_repo or "",
        *thread.session_ids,
        *thread.work_event_breakdown.keys(),
        thread.support_level,
        *thread.support_signals,
        *(signal for member in thread.member_evidence for signal in member.support_signals),
    ]
    search_text = " \n".join(part.strip() for part in parts if part and str(part).strip())
    return search_text or thread.thread_id


def build_thread_record(
    thread: Thread,
    *,
    materialized_at: str | None = None,
) -> ThreadRecord:
    built_at = materialized_at or now_iso()
    payload = _thread_payload(thread)
    source_updated_at = thread.end_time.isoformat() if thread.end_time else None
    return ThreadRecord(
        thread_id=thread.thread_id,
        root_id=SessionId(thread.root_id),
        materializer_version=SESSION_INSIGHT_MATERIALIZER_VERSION,
        materialized_at=built_at,
        source_updated_at=source_updated_at,
        input_high_water_mark=source_updated_at,
        input_high_water_mark_source=classify_thread_hwm_source(thread.end_time),
        input_row_count=len(thread.session_ids),
        start_time=thread.start_time.isoformat() if thread.start_time else None,
        end_time=thread.end_time.isoformat() if thread.end_time else None,
        dominant_repo=thread.dominant_repo,
        session_ids=thread.session_ids,
        session_count=len(thread.session_ids),
        depth=thread.depth,
        branch_count=thread.branch_count,
        total_messages=thread.total_messages,
        total_cost_usd=thread.total_cost_usd,
        wall_duration_ms=thread.wall_duration_ms,
        work_event_breakdown=thread.work_event_breakdown,
        payload=ArchivedThreadPayload.model_validate(payload),
        search_text=thread_search_text(thread),
    )


def hydrate_thread(record: ThreadRecord) -> Thread:
    return Thread.from_payload(_thread_payload_document(record))


def _thread_payload(thread: Thread) -> ThreadPayload:
    return thread.to_dict()


def _thread_payload_document(record: ThreadRecord) -> ThreadDocument:
    payload = record.payload
    return {
        "thread_id": record.thread_id,
        "root_id": str(record.root_id),
        "session_ids": list(payload.session_ids),
        "session_count": payload.session_count,
        "depth": payload.depth,
        "branch_count": payload.branch_count,
        "start_time": payload.start_time,
        "end_time": payload.end_time,
        "wall_duration_ms": payload.wall_duration_ms,
        "total_messages": payload.total_messages,
        "total_cost_usd": payload.total_cost_usd,
        "dominant_repo": payload.dominant_repo,
        "origin_breakdown": {
            source_name_to_origin(source_name): count for source_name, count in payload.provider_breakdown.items()
        },
        "work_event_breakdown": dict(payload.work_event_breakdown),
        "confidence": payload.confidence,
        "support_level": payload.support_level,
        "support_signals": list(payload.support_signals),
        "member_evidence": [
            {
                "session_id": member.session_id,
                "parent_id": member.parent_id,
                "role": member.role,
                "depth": member.depth,
                "confidence": member.confidence,
                "support_signals": list(member.support_signals),
                "evidence": list(member.evidence),
            }
            for member in payload.member_evidence
        ],
    }


# ---------------------------------------------------------------------------
# Thread queries
# ---------------------------------------------------------------------------


def thread_root_id_sync(conn: sqlite3.Connection, session_id: str) -> str | None:
    row = conn.execute(_THREAD_ROOT_ID_SQL, (session_id,)).fetchone()
    return str(row["session_id"]) if row else None


async def thread_root_id_async(conn: aiosqlite.Connection, session_id: str) -> str | None:
    row = await (await conn.execute(_THREAD_ROOT_ID_SQL, (session_id,))).fetchone()
    return str(row["session_id"]) if row else None


async def thread_root_ids_async(
    conn: aiosqlite.Connection,
    session_ids: Sequence[str],
) -> dict[str, str]:
    if not session_ids:
        return {}
    placeholders = ", ".join("?" for _ in session_ids)
    rows = await (
        await conn.execute(
            _THREAD_ROOT_IDS_SQL_TEMPLATE.format(placeholders=placeholders),
            tuple(session_ids),
        )
    ).fetchall()
    return {str(row["target_id"]): str(row["session_id"]) for row in rows}


def thread_root_ids_sync(
    conn: sqlite3.Connection,
    session_ids: Sequence[str],
) -> dict[str, str]:
    if not session_ids:
        return {}
    placeholders = ", ".join("?" for _ in session_ids)
    rows = conn.execute(
        _THREAD_ROOT_IDS_SQL_TEMPLATE.format(placeholders=placeholders),
        tuple(session_ids),
    ).fetchall()
    return {str(row["target_id"]): str(row["session_id"]) for row in rows}


def thread_session_ids_sync(conn: sqlite3.Connection, root_id: str) -> list[str]:
    rows = conn.execute(_THREAD_SESSION_IDS_SQL, (root_id,)).fetchall()
    return [str(row["session_id"]) for row in rows]


async def thread_session_ids_async(conn: aiosqlite.Connection, root_id: str) -> list[str]:
    rows = await (await conn.execute(_THREAD_SESSION_IDS_SQL, (root_id,))).fetchall()
    return [str(row["session_id"]) for row in rows]


def iter_root_id_pages_sync(
    conn: sqlite3.Connection,
    *,
    size: int = _ROOT_BATCH_SIZE,
) -> Iterator[list[str]]:
    cursor = conn.execute(_ROOT_THREAD_IDS_SQL)
    while True:
        rows = cursor.fetchmany(size)
        if not rows:
            break
        yield [str(row["session_id"]) for row in rows]


async def iter_root_id_pages_async(
    conn: aiosqlite.Connection,
    *,
    size: int = _ROOT_BATCH_SIZE,
) -> AsyncIterator[list[str]]:
    cursor = await conn.execute(_ROOT_THREAD_IDS_SQL)
    while True:
        rows = list(await cursor.fetchmany(size))
        if not rows:
            break
        yield [str(row["session_id"]) for row in rows]


def _chunk_root_ids(root_ids: Sequence[str], *, size: int = _ROOT_BATCH_SIZE) -> list[tuple[str, ...]]:
    return [
        tuple(root_ids[index : index + size])
        for index in range(0, len(root_ids), size)
        if root_ids[index : index + size]
    ]


def _empty_profile_record_groups(root_ids: Sequence[str]) -> dict[str, list[SessionProfileRecord]]:
    return {str(root_id): [] for root_id in root_ids}


def _group_profile_records_by_root(
    rows: Iterable[sqlite3.Row],
    *,
    root_ids: Sequence[str],
) -> dict[str, list[SessionProfileRecord]]:
    grouped: dict[str, list[SessionProfileRecord]] = _empty_profile_record_groups(root_ids)
    for row in rows:
        grouped[str(row["root_id"])].append(_row_to_session_profile_record(row))
    return grouped


def _repair_profile_parent_ids(
    records: Sequence[SessionProfileRecord],
    parent_ids_by_session: dict[str, str | None],
) -> list[SessionProfileRecord]:
    repaired: list[SessionProfileRecord] = []
    for record in records:
        parent_id = parent_ids_by_session.get(str(record.session_id))
        if parent_id and not record.evidence_payload.parent_id:
            repaired.append(
                record.model_copy(
                    update={
                        "evidence_payload": record.evidence_payload.model_copy(
                            update={"parent_id": parent_id, "is_continuation": True}
                        )
                    }
                )
            )
        else:
            repaired.append(record)
    return repaired


def _parent_ids_for_sessions_sync(
    conn: sqlite3.Connection,
    session_ids: Sequence[str],
) -> dict[str, str | None]:
    if not session_ids:
        return {}
    placeholders = ", ".join("?" for _ in session_ids)
    rows = conn.execute(
        f"SELECT session_id, parent_session_id FROM sessions WHERE session_id IN ({placeholders})",
        tuple(session_ids),
    ).fetchall()
    return {str(row["session_id"]): str(row["parent_session_id"]) if row["parent_session_id"] else None for row in rows}


async def _parent_ids_for_sessions_async(
    conn: aiosqlite.Connection,
    session_ids: Sequence[str],
) -> dict[str, str | None]:
    if not session_ids:
        return {}
    placeholders = ", ".join("?" for _ in session_ids)
    rows = await (
        await conn.execute(
            f"SELECT session_id, parent_session_id FROM sessions WHERE session_id IN ({placeholders})",
            tuple(session_ids),
        )
    ).fetchall()
    return {str(row["session_id"]): str(row["parent_session_id"]) if row["parent_session_id"] else None for row in rows}


def _thread_records_from_profile_records(
    profile_records: Sequence[SessionProfileRecord],
) -> dict[str, ThreadRecord]:
    if not profile_records:
        return {}
    profiles = [hydrate_session_profile(record) for record in profile_records]
    return {thread.thread_id: build_thread_record(thread) for thread in build_session_threads(profiles)}


def _thread_record_for_root(
    root_id: str,
    profile_records: Sequence[SessionProfileRecord],
) -> ThreadRecord | None:
    return _thread_records_from_profile_records(profile_records).get(str(root_id))


def load_thread_profile_records_sync(
    conn: sqlite3.Connection,
    root_id: str,
) -> list[SessionProfileRecord]:
    return load_thread_profile_records_by_root_sync(conn, [root_id]).get(root_id, [])


async def load_thread_profile_records_async(
    conn: aiosqlite.Connection,
    root_id: str,
) -> list[SessionProfileRecord]:
    return (await load_thread_profile_records_by_root_async(conn, [root_id])).get(root_id, [])


def load_thread_profile_records_by_root_sync(
    conn: sqlite3.Connection,
    root_ids: Sequence[str],
) -> dict[str, list[SessionProfileRecord]]:
    normalized_root_ids = tuple(dict.fromkeys(str(root_id) for root_id in root_ids if str(root_id)))
    if not normalized_root_ids:
        return {}
    grouped: dict[str, list[SessionProfileRecord]] = _empty_profile_record_groups(normalized_root_ids)
    for root_chunk in _chunk_root_ids(normalized_root_ids):
        placeholders = ", ".join("?" for _ in root_chunk)
        rows = conn.execute(
            _THREAD_PROFILE_RECORDS_BY_ROOT_SQL_TEMPLATE.format(placeholders=placeholders),
            root_chunk,
        ).fetchall()
        for root_id, records in _group_profile_records_by_root(rows, root_ids=root_chunk).items():
            parent_ids = _parent_ids_for_sessions_sync(conn, [str(record.session_id) for record in records])
            grouped[root_id].extend(_repair_profile_parent_ids(records, parent_ids))
    return grouped


async def load_thread_profile_records_by_root_async(
    conn: aiosqlite.Connection,
    root_ids: Sequence[str],
) -> dict[str, list[SessionProfileRecord]]:
    normalized_root_ids = tuple(dict.fromkeys(str(root_id) for root_id in root_ids if str(root_id)))
    if not normalized_root_ids:
        return {}
    grouped: dict[str, list[SessionProfileRecord]] = _empty_profile_record_groups(normalized_root_ids)
    for root_chunk in _chunk_root_ids(normalized_root_ids):
        placeholders = ", ".join("?" for _ in root_chunk)
        rows = await (
            await conn.execute(
                _THREAD_PROFILE_RECORDS_BY_ROOT_SQL_TEMPLATE.format(placeholders=placeholders),
                root_chunk,
            )
        ).fetchall()
        for root_id, records in _group_profile_records_by_root(rows, root_ids=root_chunk).items():
            parent_ids = await _parent_ids_for_sessions_async(conn, [str(record.session_id) for record in records])
            grouped[root_id].extend(_repair_profile_parent_ids(records, parent_ids))
    return grouped


def build_thread_records_for_roots_sync(
    conn: sqlite3.Connection,
    root_ids: Sequence[str],
) -> dict[str, ThreadRecord]:
    profile_records_by_root = load_thread_profile_records_by_root_sync(conn, root_ids)
    return {
        str(root_id): record
        for root_id in root_ids
        if (
            record := _thread_record_for_root(
                str(root_id),
                profile_records_by_root.get(str(root_id), []),
            )
        )
        is not None
    }


async def build_thread_records_for_roots_async(
    conn: aiosqlite.Connection,
    root_ids: Sequence[str],
) -> dict[str, ThreadRecord]:
    profile_records_by_root = await load_thread_profile_records_by_root_async(conn, root_ids)
    return {
        str(root_id): record
        for root_id in root_ids
        if (
            record := _thread_record_for_root(
                str(root_id),
                profile_records_by_root.get(str(root_id), []),
            )
        )
        is not None
    }


def build_all_thread_records_sync(conn: sqlite3.Connection) -> list[ThreadRecord]:
    records: list[ThreadRecord] = []
    for root_chunk in iter_root_id_pages_sync(conn):
        records_by_root = build_thread_records_for_roots_sync(conn, root_chunk)
        records.extend(records_by_root[root_id] for root_id in root_chunk if root_id in records_by_root)
    return records


async def build_all_thread_records_async(conn: aiosqlite.Connection) -> list[ThreadRecord]:
    records: list[ThreadRecord] = []
    async for root_chunk in iter_root_id_pages_async(conn):
        records_by_root = await build_thread_records_for_roots_async(conn, root_chunk)
        records.extend(records_by_root[root_id] for root_id in root_chunk if root_id in records_by_root)
    return records


__all__ = [
    "build_all_thread_records_async",
    "build_all_thread_records_sync",
    "build_thread_records_for_roots_async",
    "build_thread_records_for_roots_sync",
    "build_thread_record",
    "hydrate_thread",
    "iter_root_id_pages_async",
    "iter_root_id_pages_sync",
    "load_thread_profile_records_async",
    "load_thread_profile_records_by_root_async",
    "load_thread_profile_records_by_root_sync",
    "load_thread_profile_records_sync",
    "thread_session_ids_async",
    "thread_session_ids_sync",
    "thread_root_id_async",
    "thread_root_ids_async",
    "thread_root_ids_sync",
    "thread_root_id_sync",
    "thread_search_text",
]
