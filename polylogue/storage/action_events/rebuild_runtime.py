"""Runtime entry points for action-event read-model repair and rebuild."""

from __future__ import annotations

import sqlite3
from collections.abc import Callable, Sequence

import aiosqlite

from polylogue.storage.action_events.rebuild_loading import (
    chunked,
    iter_conversation_id_pages_async,
    iter_conversation_id_pages_sync,
    load_async_batch,
    load_sync_batch,
)
from polylogue.storage.action_events.rebuild_materialization import materialize_batch
from polylogue.storage.action_events.rebuild_sql import (
    ACTION_EVENT_REPAIR_CANDIDATE_IDS_SQL,
    ACTION_EVENT_VALID_SOURCE_IDS_SQL,
)
from polylogue.storage.action_events.rebuild_storage import (
    replace_action_events_async,
    replace_action_events_sync,
)
from polylogue.storage.runtime import ACTION_EVENT_MATERIALIZER_VERSION


def _row_int(row: sqlite3.Row | None, key: int | str) -> int:
    if row is None:
        return 0
    try:
        return int(row[key])
    except (TypeError, ValueError):
        return 0


def rebuild_action_event_read_model_sync(
    conn: sqlite3.Connection,
    *,
    conversation_ids: Sequence[str] | None = None,
    page_size: int = 200,
) -> int:
    replaced = 0
    if conversation_ids is None:
        conn.execute("DELETE FROM action_events")
        for chunk in iter_conversation_id_pages_sync(conn, page_size=page_size):
            conversations, messages, blocks = load_sync_batch(conn, chunk)
            materialized = materialize_batch(conversations, messages, blocks)
            for conversation_id in chunk:
                records = materialized.get(conversation_id, [])
                replace_action_events_sync(conn, conversation_id, records)
                replaced += len(records)
        return replaced
    elif not conversation_ids:
        conn.execute("DELETE FROM action_events")
        return 0

    for raw_chunk in chunked(list(conversation_ids), size=page_size):
        chunk = list(raw_chunk)
        conversations, messages, blocks = load_sync_batch(conn, chunk)
        materialized = materialize_batch(conversations, messages, blocks)
        for conversation_id in chunk:
            records = materialized.get(conversation_id, [])
            replace_action_events_sync(conn, conversation_id, records)
            replaced += len(records)
    return replaced


async def rebuild_action_event_read_model_async(
    conn: aiosqlite.Connection,
    *,
    conversation_ids: Sequence[str] | None = None,
    page_size: int = 200,
    progress_callback: Callable[[int, str | None], None] | None = None,
    progress_desc: Callable[[int, int], str] | None = None,
) -> int:
    if conversation_ids is None:
        await conn.execute("DELETE FROM action_events")
        total = _row_int(await (await conn.execute("SELECT COUNT(*) FROM conversations")).fetchone(), 0)
    elif not conversation_ids:
        await conn.execute("DELETE FROM action_events")
        return 0
    else:
        total = len(conversation_ids)

    replaced = 0
    processed = 0
    if conversation_ids is None:
        async for chunk in iter_conversation_id_pages_async(conn, page_size=page_size):
            conversations, messages, blocks = await load_async_batch(conn, chunk)
            materialized = materialize_batch(conversations, messages, blocks)
            for conversation_id in chunk:
                records = materialized.get(conversation_id, [])
                await replace_action_events_async(conn, conversation_id, records)
                replaced += len(records)
            processed += len(chunk)
            if progress_callback is not None:
                desc = progress_desc(processed, total) if progress_desc is not None else None
                progress_callback(len(chunk), desc)
    else:
        for raw_chunk in chunked(list(conversation_ids), size=page_size):
            chunk = list(raw_chunk)
            conversations, messages, blocks = await load_async_batch(conn, chunk)
            materialized = materialize_batch(conversations, messages, blocks)
            for conversation_id in chunk:
                records = materialized.get(conversation_id, [])
                await replace_action_events_async(conn, conversation_id, records)
                replaced += len(records)
            processed += len(chunk)
            if progress_callback is not None:
                desc = progress_desc(processed, total) if progress_desc is not None else None
                progress_callback(len(chunk), desc)
    return replaced


def action_event_repair_candidates_sync(conn: sqlite3.Connection) -> list[str]:
    rows = conn.execute(
        ACTION_EVENT_REPAIR_CANDIDATE_IDS_SQL,
        (ACTION_EVENT_MATERIALIZER_VERSION,),
    ).fetchall()
    return [str(row["conversation_id"]) for row in rows]


async def action_event_repair_candidates_async(conn: aiosqlite.Connection) -> list[str]:
    rows = await (
        await conn.execute(
            ACTION_EVENT_REPAIR_CANDIDATE_IDS_SQL,
            (ACTION_EVENT_MATERIALIZER_VERSION,),
        )
    ).fetchall()
    return [str(row["conversation_id"]) for row in rows]


def valid_action_event_source_ids_sync(conn: sqlite3.Connection) -> list[str]:
    rows = conn.execute(ACTION_EVENT_VALID_SOURCE_IDS_SQL).fetchall()
    return [str(row["conversation_id"]) for row in rows]


async def valid_action_event_source_ids_async(conn: aiosqlite.Connection) -> list[str]:
    rows = await (await conn.execute(ACTION_EVENT_VALID_SOURCE_IDS_SQL)).fetchall()
    return [str(row["conversation_id"]) for row in rows]


__all__ = [
    "action_event_repair_candidates_async",
    "action_event_repair_candidates_sync",
    "rebuild_action_event_read_model_async",
    "rebuild_action_event_read_model_sync",
    "valid_action_event_source_ids_async",
    "valid_action_event_source_ids_sync",
]
