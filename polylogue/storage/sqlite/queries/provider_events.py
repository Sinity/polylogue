"""Provider-event archive queries."""

from __future__ import annotations

import sqlite3
from collections import defaultdict
from collections.abc import Sequence

import aiosqlite

from polylogue.core.common import SQL_PROVIDER_EVENT_INSERT as _PROVIDER_EVENT_INSERT_SQL
from polylogue.storage.runtime import ProviderEventRecord
from polylogue.storage.sqlite.provider_event_model import project_provider_event_payload
from polylogue.storage.sqlite.queries.mappers import _row_to_provider_event

_PROVIDER_EVENT_SELECT = """
            SELECT
                pe.*,
                pc.summary AS compaction_summary,
                pc.trigger AS compaction_trigger,
                pc.pre_tokens AS compaction_pre_tokens,
                pc.preserved_segment_id AS compaction_preserved_segment_id,
                pc.is_modern AS compaction_is_modern,
                pc.replacement_history_count AS compaction_replacement_history_count,
                tc.cwd AS turn_context_cwd,
                tc.model AS turn_context_model,
                tc.effort AS turn_context_effort,
                tc.approval_policy AS turn_context_approval_policy,
                tc.sandbox_policy AS turn_context_sandbox_policy,
                tc.summary AS turn_context_summary,
                ptc.call_id AS tool_call_id,
                ptc.tool_name AS tool_call_tool_name,
                ptc.status AS tool_call_status,
                ptc.input_chars AS tool_call_input_chars,
                ptc.output_chars AS tool_call_output_chars,
                ptc.has_input_body AS tool_call_has_input_body,
                ptc.has_output_body AS tool_call_has_output_body,
                pr.summary AS reasoning_summary,
                pr.encrypted_content_hash AS reasoning_encrypted_content_hash,
                pr.encrypted_content_bytes AS reasoning_encrypted_content_bytes,
                pgs.ghost_commit AS ghost_snapshot_ghost_commit
            FROM provider_events pe
            LEFT JOIN provider_event_compactions pc ON pc.event_id = pe.event_id
            LEFT JOIN provider_event_turn_contexts tc ON tc.event_id = pe.event_id
            LEFT JOIN provider_event_tool_calls ptc ON ptc.event_id = pe.event_id
            LEFT JOIN provider_event_reasoning pr ON pr.event_id = pe.event_id
            LEFT JOIN provider_event_ghost_snapshots pgs ON pgs.event_id = pe.event_id
"""


def _provider_event_select_sql(where_clause: str) -> str:
    return f"""
{_PROVIDER_EVENT_SELECT}
            {where_clause}
            ORDER BY pe.session_id, pe.event_index
            """


def _provider_event_compaction_count_sql(placeholders: str) -> str:
    return f"""
            SELECT session_id, COUNT(*) AS compaction_count
            FROM provider_events
            WHERE session_id IN ({placeholders})
              AND event_type = 'compaction'
            GROUP BY session_id
            """


async def get_provider_events(
    conn: aiosqlite.Connection,
    session_id: str,
) -> list[ProviderEventRecord]:
    query = _provider_event_select_sql("WHERE pe.session_id = ?").replace(
        "ORDER BY pe.session_id, pe.event_index",
        "ORDER BY pe.event_index",
    )
    rows = await (await conn.execute(query, (session_id,))).fetchall()
    return [_row_to_provider_event(row) for row in rows]


async def get_provider_events_batch(
    conn: aiosqlite.Connection,
    session_ids: Sequence[str],
) -> dict[str, list[ProviderEventRecord]]:
    if not session_ids:
        return {}
    placeholders = ", ".join("?" for _ in session_ids)
    query = _provider_event_select_sql(f"WHERE pe.session_id IN ({placeholders})")
    rows = await (await conn.execute(query, tuple(session_ids))).fetchall()
    result: dict[str, list[ProviderEventRecord]] = {session_id: [] for session_id in session_ids}
    for row in rows:
        record = _row_to_provider_event(row)
        result.setdefault(str(record.session_id), []).append(record)
    return result


async def get_provider_event_compaction_counts(
    conn: aiosqlite.Connection,
    session_ids: Sequence[str],
) -> dict[str, int]:
    if not session_ids:
        return {}
    placeholders = ", ".join("?" for _ in session_ids)
    query = _provider_event_compaction_count_sql(placeholders)
    rows = await (await conn.execute(query, tuple(session_ids))).fetchall()
    result = dict.fromkeys(session_ids, 0)
    for row in rows:
        result[str(row["session_id"])] = int(row["compaction_count"] or 0)
    return result


def sync_provider_events_batch(
    conn: sqlite3.Connection,
    session_ids: Sequence[str],
) -> dict[str, list[ProviderEventRecord]]:
    if not session_ids:
        return {}
    placeholders = ", ".join("?" for _ in session_ids)
    query = _provider_event_select_sql(f"WHERE pe.session_id IN ({placeholders})")
    rows = conn.execute(query, tuple(session_ids)).fetchall()
    result: dict[str, list[ProviderEventRecord]] = defaultdict(list)
    for session_id in session_ids:
        result.setdefault(session_id, [])
    for row in rows:
        record = _row_to_provider_event(row)
        result[str(record.session_id)].append(record)
    return dict(result)


def sync_provider_event_compaction_counts(
    conn: sqlite3.Connection,
    session_ids: Sequence[str],
) -> dict[str, int]:
    if not session_ids:
        return {}
    placeholders = ", ".join("?" for _ in session_ids)
    query = _provider_event_compaction_count_sql(placeholders)
    rows = conn.execute(query, tuple(session_ids)).fetchall()
    result = dict.fromkeys(session_ids, 0)
    for row in rows:
        result[str(row["session_id"])] = int(row["compaction_count"] or 0)
    return result


async def replace_provider_events(
    conn: aiosqlite.Connection,
    session_id: str,
    records: list[ProviderEventRecord],
    transaction_depth: int,
) -> None:
    await conn.execute("DELETE FROM provider_events WHERE session_id = ?", (session_id,))
    for record in records:
        projection = project_provider_event_payload(record.event_type, record.payload)
        await conn.execute(
            _PROVIDER_EVENT_INSERT_SQL,
            (
                record.event_id,
                record.session_id,
                record.source_name,
                record.event_index,
                record.event_type,
                projection.normalized_kind,
                record.timestamp,
                record.sort_key,
                record.source_message_id,
                record.raw_id,
                record.materializer_version,
            ),
        )
        if projection.compaction is not None:
            await conn.execute(
                """
                INSERT OR REPLACE INTO provider_event_compactions (
                    event_id, summary, trigger, pre_tokens, preserved_segment_id,
                    is_modern, replacement_history_count
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (record.event_id, *projection.compaction),
            )
        if projection.turn_context is not None:
            await conn.execute(
                """
                INSERT OR REPLACE INTO provider_event_turn_contexts (
                    event_id, cwd, model, effort, approval_policy, sandbox_policy, summary
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (record.event_id, *projection.turn_context),
            )
        if projection.tool_call is not None:
            await conn.execute(
                """
                INSERT OR REPLACE INTO provider_event_tool_calls (
                    event_id, call_id, tool_name, status, input_chars, output_chars,
                    has_input_body, has_output_body
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (record.event_id, *projection.tool_call),
            )
        if projection.reasoning is not None:
            await conn.execute(
                """
                INSERT OR REPLACE INTO provider_event_reasoning (
                    event_id, summary, encrypted_content_hash, encrypted_content_bytes
                ) VALUES (?, ?, ?, ?)
                """,
                (record.event_id, *projection.reasoning),
            )
        if projection.ghost_snapshot is not None:
            await conn.execute(
                """
                INSERT OR REPLACE INTO provider_event_ghost_snapshots (
                    event_id, ghost_commit
                ) VALUES (?, ?)
                """,
                (record.event_id, *projection.ghost_snapshot),
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
