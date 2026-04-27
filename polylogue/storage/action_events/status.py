"""Status and readiness helpers for the action-event read model."""

from __future__ import annotations

import sqlite3
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TypeAlias

import aiosqlite

from polylogue.storage.action_events.artifacts import ActionEventArtifactState
from polylogue.storage.runtime import ACTION_EVENT_MATERIALIZER_VERSION

_ACTION_EVENTS_EXISTS_SQL = "SELECT name FROM sqlite_master WHERE type='table' AND name='action_events'"
_ACTION_EVENT_DOC_COUNT_SQL = "SELECT COUNT(*) FROM action_events"
_ACTION_EVENT_MISMATCH_COUNT_SQL = """
    SELECT COUNT(*)
    FROM action_events
    WHERE materializer_version != ?
"""
_ACTION_EVENT_MATERIALIZED_CONVERSATION_COUNT_SQL = """
    SELECT COUNT(DISTINCT conversation_id)
    FROM action_events
"""
_ACTION_EVENT_VALID_SOURCE_CONVERSATION_COUNT_SQL = """
    SELECT COUNT(DISTINCT cb.conversation_id)
    FROM content_blocks cb
    JOIN conversations c ON c.conversation_id = cb.conversation_id
    WHERE cb.type = 'tool_use'
"""
_ACTION_EVENT_ORPHAN_SOURCE_CONVERSATION_COUNT_SQL = """
    SELECT COUNT(DISTINCT cb.conversation_id)
    FROM content_blocks cb
    LEFT JOIN conversations c ON c.conversation_id = cb.conversation_id
    WHERE cb.type = 'tool_use' AND c.conversation_id IS NULL
"""
_ACTION_EVENT_ORPHAN_TOOL_BLOCK_COUNT_SQL = """
    SELECT COUNT(*)
    FROM content_blocks cb
    LEFT JOIN conversations c ON c.conversation_id = cb.conversation_id
    WHERE cb.type = 'tool_use' AND c.conversation_id IS NULL
"""

SqlRow: TypeAlias = sqlite3.Row | tuple[object, ...]
StatusValue: TypeAlias = int | bool
ReadModelStatus: TypeAlias = dict[str, StatusValue]


@dataclass(frozen=True, slots=True)
class _SourceCounts:
    valid_conversations: int
    orphan_conversations: int
    orphan_tool_blocks: int


@dataclass(frozen=True, slots=True)
class _ActionEventCounts:
    exists: bool
    materialized_rows: int
    materialized_conversations: int
    stale_rows: int
    source: _SourceCounts


def _coerce_int(value: object) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return 0
    return 0


def _sql_row(row: object) -> SqlRow | None:
    if row is None:
        return None
    if isinstance(row, (sqlite3.Row, tuple)):
        return row
    return None


def _row_int(row: SqlRow | None, key: int | str) -> int:
    if row is None:
        return 0
    try:
        if isinstance(key, str):
            if not isinstance(row, sqlite3.Row):
                return 0
            return _coerce_int(row[key])
        return _coerce_int(row[key])
    except (IndexError, KeyError, TypeError, ValueError):
        return 0


def _mapping_int(mapping: Mapping[str, object], key: str) -> int:
    return _coerce_int(mapping.get(key, 0))


def _count_sync(
    conn: sqlite3.Connection,
    sql: str,
    params: Sequence[object] = (),
) -> int:
    return _row_int(conn.execute(sql, params).fetchone(), 0)


async def _count_async(
    conn: aiosqlite.Connection,
    sql: str,
    params: Sequence[object] = (),
) -> int:
    row = _sql_row(await (await conn.execute(sql, params)).fetchone())
    return _row_int(row, 0)


def _table_exists_sync(conn: sqlite3.Connection) -> bool:
    return conn.execute(_ACTION_EVENTS_EXISTS_SQL).fetchone() is not None


async def _table_exists_async(conn: aiosqlite.Connection) -> bool:
    return bool(await (await conn.execute(_ACTION_EVENTS_EXISTS_SQL)).fetchone())


def _materialized_counts_sync(conn: sqlite3.Connection, *, exists: bool) -> tuple[int, int]:
    if not exists:
        return (0, 0)
    return (
        _count_sync(conn, _ACTION_EVENT_DOC_COUNT_SQL),
        _count_sync(conn, _ACTION_EVENT_MATERIALIZED_CONVERSATION_COUNT_SQL),
    )


async def _materialized_counts_async(conn: aiosqlite.Connection, *, exists: bool) -> tuple[int, int]:
    if not exists:
        return (0, 0)
    return (
        await _count_async(conn, _ACTION_EVENT_DOC_COUNT_SQL),
        await _count_async(conn, _ACTION_EVENT_MATERIALIZED_CONVERSATION_COUNT_SQL),
    )


def _source_counts_sync(
    conn: sqlite3.Connection,
    *,
    materialized_conversations: int,
    verify_source_alignment: bool,
) -> _SourceCounts:
    if not verify_source_alignment:
        return _SourceCounts(
            valid_conversations=materialized_conversations,
            orphan_conversations=0,
            orphan_tool_blocks=0,
        )
    return _SourceCounts(
        valid_conversations=_count_sync(conn, _ACTION_EVENT_VALID_SOURCE_CONVERSATION_COUNT_SQL),
        orphan_conversations=_count_sync(conn, _ACTION_EVENT_ORPHAN_SOURCE_CONVERSATION_COUNT_SQL),
        orphan_tool_blocks=_count_sync(conn, _ACTION_EVENT_ORPHAN_TOOL_BLOCK_COUNT_SQL),
    )


async def _source_counts_async(
    conn: aiosqlite.Connection,
    *,
    materialized_conversations: int,
    verify_source_alignment: bool,
) -> _SourceCounts:
    if not verify_source_alignment:
        return _SourceCounts(
            valid_conversations=materialized_conversations,
            orphan_conversations=0,
            orphan_tool_blocks=0,
        )
    return _SourceCounts(
        valid_conversations=await _count_async(conn, _ACTION_EVENT_VALID_SOURCE_CONVERSATION_COUNT_SQL),
        orphan_conversations=await _count_async(conn, _ACTION_EVENT_ORPHAN_SOURCE_CONVERSATION_COUNT_SQL),
        orphan_tool_blocks=await _count_async(conn, _ACTION_EVENT_ORPHAN_TOOL_BLOCK_COUNT_SQL),
    )


def _stale_count_sync(
    conn: sqlite3.Connection,
    *,
    exists: bool,
    verify_source_alignment: bool,
) -> int:
    if not exists or not verify_source_alignment:
        return 0
    return _count_sync(conn, _ACTION_EVENT_MISMATCH_COUNT_SQL, (ACTION_EVENT_MATERIALIZER_VERSION,))


async def _stale_count_async(
    conn: aiosqlite.Connection,
    *,
    exists: bool,
    verify_source_alignment: bool,
) -> int:
    if not exists or not verify_source_alignment:
        return 0
    return await _count_async(conn, _ACTION_EVENT_MISMATCH_COUNT_SQL, (ACTION_EVENT_MATERIALIZER_VERSION,))


def _counts_sync(
    conn: sqlite3.Connection,
    *,
    verify_source_alignment: bool,
) -> _ActionEventCounts:
    exists = _table_exists_sync(conn)
    materialized_rows, materialized_conversations = _materialized_counts_sync(conn, exists=exists)
    return _ActionEventCounts(
        exists=exists,
        materialized_rows=materialized_rows,
        materialized_conversations=materialized_conversations,
        stale_rows=_stale_count_sync(
            conn,
            exists=exists,
            verify_source_alignment=verify_source_alignment,
        ),
        source=_source_counts_sync(
            conn,
            materialized_conversations=materialized_conversations,
            verify_source_alignment=verify_source_alignment,
        ),
    )


async def _counts_async(
    conn: aiosqlite.Connection,
    *,
    verify_source_alignment: bool,
) -> _ActionEventCounts:
    exists = await _table_exists_async(conn)
    materialized_rows, materialized_conversations = await _materialized_counts_async(conn, exists=exists)
    return _ActionEventCounts(
        exists=exists,
        materialized_rows=materialized_rows,
        materialized_conversations=materialized_conversations,
        stale_rows=await _stale_count_async(
            conn,
            exists=exists,
            verify_source_alignment=verify_source_alignment,
        ),
        source=await _source_counts_async(
            conn,
            materialized_conversations=materialized_conversations,
            verify_source_alignment=verify_source_alignment,
        ),
    )


def _artifact_state(
    counts: _ActionEventCounts,
    fts_status: Mapping[str, object],
) -> ActionEventArtifactState:
    return ActionEventArtifactState(
        source_conversations=counts.source.valid_conversations,
        materialized_conversations=counts.materialized_conversations,
        materialized_rows=counts.materialized_rows,
        fts_rows=_mapping_int(fts_status, "action_count"),
        stale_rows=counts.stale_rows,
        orphan_rows=counts.source.orphan_tool_blocks,
        matches_version=counts.stale_rows == 0,
    )


def _read_model_status(
    counts: _ActionEventCounts,
    state: ActionEventArtifactState,
    fts_status: Mapping[str, object],
) -> ReadModelStatus:
    source_conversations = counts.source.valid_conversations + counts.source.orphan_conversations
    return {
        "exists": counts.exists,
        "count": state.materialized_rows,
        "stale_count": state.stale_rows,
        "matches_version": state.matches_version,
        "source_conversation_count": source_conversations,
        "valid_source_conversation_count": counts.source.valid_conversations,
        "orphan_source_conversation_count": counts.source.orphan_conversations,
        "orphan_tool_block_count": state.orphan_rows,
        "materialized_conversation_count": state.materialized_conversations,
        "rows_ready": state.rows_ready,
        "action_fts_exists": bool(fts_status.get("exists", False)),
        "action_fts_count": state.fts_rows,
        "action_fts_pending_rows": state.pending_fts_rows,
        "action_fts_stale_rows": state.excess_fts_rows,
        "action_fts_ready": state.fts_ready,
        "ready": state.ready,
    }


def action_event_artifact_state_sync(
    conn: sqlite3.Connection,
    *,
    verify_source_alignment: bool = True,
) -> ActionEventArtifactState:
    from polylogue.storage.fts.fts_lifecycle import fts_index_status_sync

    counts = _counts_sync(conn, verify_source_alignment=verify_source_alignment)
    return _artifact_state(counts, fts_index_status_sync(conn))


async def action_event_artifact_state_async(
    conn: aiosqlite.Connection,
    *,
    verify_source_alignment: bool = True,
) -> ActionEventArtifactState:
    from polylogue.storage.fts.fts_lifecycle import fts_index_status_async

    counts = await _counts_async(conn, verify_source_alignment=verify_source_alignment)
    return _artifact_state(counts, await fts_index_status_async(conn))


def action_event_read_model_status_sync(
    conn: sqlite3.Connection,
    *,
    verify_source_alignment: bool = True,
) -> ReadModelStatus:
    from polylogue.storage.fts.fts_lifecycle import fts_index_status_sync

    counts = _counts_sync(conn, verify_source_alignment=verify_source_alignment)
    fts_status = fts_index_status_sync(conn)
    return _read_model_status(counts, _artifact_state(counts, fts_status), fts_status)


async def action_event_read_model_status_async(
    conn: aiosqlite.Connection,
    *,
    verify_source_alignment: bool = True,
) -> ReadModelStatus:
    from polylogue.storage.fts.fts_lifecycle import fts_index_status_async

    counts = await _counts_async(conn, verify_source_alignment=verify_source_alignment)
    fts_status = await fts_index_status_async(conn)
    return _read_model_status(counts, _artifact_state(counts, fts_status), fts_status)


__all__ = [
    "action_event_artifact_state_async",
    "action_event_artifact_state_sync",
    "action_event_read_model_status_async",
    "action_event_read_model_status_sync",
]
