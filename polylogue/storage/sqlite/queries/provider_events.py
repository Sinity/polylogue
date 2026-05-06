"""Provider-event archive queries."""

from __future__ import annotations

import sqlite3
from collections import defaultdict
from collections.abc import Sequence

import aiosqlite

from polylogue.core.common import SQL_PROVIDER_EVENT_INSERT as _PROVIDER_EVENT_INSERT_SQL
from polylogue.storage.runtime import ProviderEventRecord, _json_or_none
from polylogue.storage.sqlite.queries.mappers import _row_to_provider_event


async def get_provider_events(
    conn: aiosqlite.Connection,
    conversation_id: str,
) -> list[ProviderEventRecord]:
    rows = await (
        await conn.execute(
            """
            SELECT *
            FROM provider_events
            WHERE conversation_id = ?
            ORDER BY event_index
            """,
            (conversation_id,),
        )
    ).fetchall()
    return [_row_to_provider_event(row) for row in rows]


async def get_provider_events_batch(
    conn: aiosqlite.Connection,
    conversation_ids: Sequence[str],
) -> dict[str, list[ProviderEventRecord]]:
    if not conversation_ids:
        return {}
    placeholders = ", ".join("?" for _ in conversation_ids)
    rows = await (
        await conn.execute(
            f"""
            SELECT *
            FROM provider_events
            WHERE conversation_id IN ({placeholders})
            ORDER BY conversation_id, event_index
            """,
            tuple(conversation_ids),
        )
    ).fetchall()
    result: dict[str, list[ProviderEventRecord]] = {conversation_id: [] for conversation_id in conversation_ids}
    for row in rows:
        record = _row_to_provider_event(row)
        result.setdefault(str(record.conversation_id), []).append(record)
    return result


async def get_provider_event_compaction_counts(
    conn: aiosqlite.Connection,
    conversation_ids: Sequence[str],
) -> dict[str, int]:
    if not conversation_ids:
        return {}
    placeholders = ", ".join("?" for _ in conversation_ids)
    rows = await (
        await conn.execute(
            f"""
            SELECT conversation_id, COUNT(*) AS compaction_count
            FROM provider_events
            WHERE conversation_id IN ({placeholders})
              AND event_type = 'compaction'
            GROUP BY conversation_id
            """,
            tuple(conversation_ids),
        )
    ).fetchall()
    result = dict.fromkeys(conversation_ids, 0)
    for row in rows:
        result[str(row["conversation_id"])] = int(row["compaction_count"] or 0)
    return result


def sync_provider_events_batch(
    conn: sqlite3.Connection,
    conversation_ids: Sequence[str],
) -> dict[str, list[ProviderEventRecord]]:
    if not conversation_ids:
        return {}
    placeholders = ", ".join("?" for _ in conversation_ids)
    rows = conn.execute(
        f"""
        SELECT *
        FROM provider_events
        WHERE conversation_id IN ({placeholders})
        ORDER BY conversation_id, event_index
        """,
        tuple(conversation_ids),
    ).fetchall()
    result: dict[str, list[ProviderEventRecord]] = defaultdict(list)
    for conversation_id in conversation_ids:
        result.setdefault(conversation_id, [])
    for row in rows:
        record = _row_to_provider_event(row)
        result[str(record.conversation_id)].append(record)
    return dict(result)


def sync_provider_event_compaction_counts(
    conn: sqlite3.Connection,
    conversation_ids: Sequence[str],
) -> dict[str, int]:
    if not conversation_ids:
        return {}
    placeholders = ", ".join("?" for _ in conversation_ids)
    rows = conn.execute(
        f"""
        SELECT conversation_id, COUNT(*) AS compaction_count
        FROM provider_events
        WHERE conversation_id IN ({placeholders})
          AND event_type = 'compaction'
        GROUP BY conversation_id
        """,
        tuple(conversation_ids),
    ).fetchall()
    result = dict.fromkeys(conversation_ids, 0)
    for row in rows:
        result[str(row["conversation_id"])] = int(row["compaction_count"] or 0)
    return result


async def replace_provider_events(
    conn: aiosqlite.Connection,
    conversation_id: str,
    records: list[ProviderEventRecord],
    transaction_depth: int,
) -> None:
    await conn.execute("DELETE FROM provider_events WHERE conversation_id = ?", (conversation_id,))
    if records:
        await conn.executemany(
            _PROVIDER_EVENT_INSERT_SQL,
            [
                (
                    record.event_id,
                    record.conversation_id,
                    record.provider_name,
                    record.event_index,
                    record.event_type,
                    record.timestamp,
                    record.sort_key,
                    _json_or_none(record.payload) or "{}",
                    record.source_message_id,
                    record.raw_id,
                    record.materializer_version,
                )
                for record in records
            ],
        )
    if transaction_depth == 0:
        await conn.commit()


__all__ = [
    "get_provider_events",
    "get_provider_events_batch",
    "get_provider_event_compaction_counts",
    "replace_provider_events",
    "sync_provider_event_compaction_counts",
    "sync_provider_events_batch",
]
