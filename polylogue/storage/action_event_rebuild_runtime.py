"""Runtime entry points for action-event read-model repair and rebuild."""

from __future__ import annotations

import sqlite3
from collections.abc import Callable, Sequence

import aiosqlite

from polylogue.storage.action_event_rebuild_loading import (
    chunked,
    load_async_batch,
    load_sync_batch,
)
from polylogue.storage.action_event_rebuild_materialization import materialize_batch
from polylogue.storage.action_event_rebuild_sql import (
    ACTION_EVENT_CONVERSATION_IDS_SQL,
    ACTION_EVENT_REPAIR_CANDIDATE_IDS_SQL,
    ACTION_EVENT_VALID_SOURCE_IDS_SQL,
)
from polylogue.storage.action_event_rebuild_storage import (
    replace_action_events_async,
    replace_action_events_sync,
)
from polylogue.storage.store import ACTION_EVENT_MATERIALIZER_VERSION


def rebuild_action_event_read_model_sync(
    conn: sqlite3.Connection,
    *,
    conversation_ids: Sequence[str] | None = None,
    page_size: int = 200,
) -> int:
    if conversation_ids is None:
        conn.execute("DELETE FROM action_events")
        conversation_ids = [
            row["conversation_id"]
            for row in conn.execute(ACTION_EVENT_CONVERSATION_IDS_SQL).fetchall()
        ]
    if not conversation_ids:
        conn.execute("DELETE FROM action_events")
        return 0

    replaced = 0
    for chunk in chunked(list(conversation_ids), size=page_size):
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
        rows = await (await conn.execute(ACTION_EVENT_CONVERSATION_IDS_SQL)).fetchall()
        conversation_ids = [row["conversation_id"] for row in rows]
    if not conversation_ids:
        await conn.execute("DELETE FROM action_events")
        return 0

    replaced = 0
    total = len(conversation_ids)
    processed = 0
    for chunk in chunked(list(conversation_ids), size=page_size):
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
