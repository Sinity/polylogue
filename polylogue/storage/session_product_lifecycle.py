"""Lifecycle helpers for durable semantic/session product read models."""

from __future__ import annotations

import sqlite3
from collections import defaultdict
from collections.abc import Iterable, Sequence

import aiosqlite

from polylogue.archive_product_builders import (
    build_day_session_summary_records,
    build_session_tag_rollup_records,
    date_from_iso,
)
from polylogue.lib.threads import build_session_threads
from polylogue.storage.action_event_rows import attach_blocks_to_messages
from polylogue.storage.backends.queries.attachments import get_attachments_batch
from polylogue.storage.backends.queries.mappers import (
    _row_to_content_block,
    _row_to_conversation,
    _row_to_message,
    _row_to_session_profile_record,
)
from polylogue.storage.backends.queries.session_products import (
    replace_day_session_summaries,
    replace_session_phases,
    replace_session_profile,
    replace_session_tag_rollup_rows,
    replace_session_work_events,
    replace_work_thread,
)
from polylogue.storage.hydrators import conversation_from_records
from polylogue.storage.session_product_rows import (
    build_session_product_records,
    build_work_thread_record,
    hydrate_session_profile,
)
from polylogue.storage.store import (
    SESSION_PRODUCT_MATERIALIZER_VERSION,
    AttachmentRecord,
    ContentBlockRecord,
    ConversationRecord,
    DaySessionSummaryRecord,
    MessageRecord,
    SessionProfileRecord,
    SessionTagRollupRecord,
    _json_array_or_none,
    _json_or_none,
)

_SESSION_PROFILES_EXISTS_SQL = "SELECT name FROM sqlite_master WHERE type='table' AND name='session_profiles'"
_SESSION_PROFILES_FTS_EXISTS_SQL = "SELECT name FROM sqlite_master WHERE type='table' AND name='session_profiles_fts'"
_SESSION_WORK_EVENTS_EXISTS_SQL = "SELECT name FROM sqlite_master WHERE type='table' AND name='session_work_events'"
_SESSION_WORK_EVENTS_FTS_EXISTS_SQL = "SELECT name FROM sqlite_master WHERE type='table' AND name='session_work_events_fts'"
_SESSION_PHASES_EXISTS_SQL = "SELECT name FROM sqlite_master WHERE type='table' AND name='session_phases'"
_WORK_THREADS_EXISTS_SQL = "SELECT name FROM sqlite_master WHERE type='table' AND name='work_threads'"
_WORK_THREADS_FTS_EXISTS_SQL = "SELECT name FROM sqlite_master WHERE type='table' AND name='work_threads_fts'"
_SESSION_TAG_ROLLUPS_EXISTS_SQL = "SELECT name FROM sqlite_master WHERE type='table' AND name='session_tag_rollups'"
_DAY_SESSION_SUMMARIES_EXISTS_SQL = "SELECT name FROM sqlite_master WHERE type='table' AND name='day_session_summaries'"
_SESSION_PROFILE_COUNT_SQL = "SELECT COUNT(*) FROM session_profiles"
_SESSION_PROFILE_FTS_DOC_COUNT_SQL = "SELECT COUNT(DISTINCT conversation_id) FROM session_profiles_fts"
_SESSION_PROFILE_FTS_DUPLICATE_COUNT_SQL = "SELECT COUNT(*) - COUNT(DISTINCT conversation_id) FROM session_profiles_fts"
_SESSION_WORK_EVENT_COUNT_SQL = "SELECT COUNT(*) FROM session_work_events"
_SESSION_WORK_EVENT_FTS_DOC_COUNT_SQL = "SELECT COUNT(DISTINCT event_id) FROM session_work_events_fts"
_SESSION_WORK_EVENT_FTS_DUPLICATE_COUNT_SQL = "SELECT COUNT(*) - COUNT(DISTINCT event_id) FROM session_work_events_fts"
_SESSION_PHASE_COUNT_SQL = "SELECT COUNT(*) FROM session_phases"
_WORK_THREAD_COUNT_SQL = "SELECT COUNT(*) FROM work_threads"
_WORK_THREAD_FTS_DOC_COUNT_SQL = "SELECT COUNT(DISTINCT thread_id) FROM work_threads_fts"
_WORK_THREAD_FTS_DUPLICATE_COUNT_SQL = "SELECT COUNT(*) - COUNT(DISTINCT thread_id) FROM work_threads_fts"
_SESSION_TAG_ROLLUP_COUNT_SQL = "SELECT COUNT(*) FROM session_tag_rollups"
_DAY_SESSION_SUMMARY_COUNT_SQL = "SELECT COUNT(*) FROM day_session_summaries"
_TOTAL_CONVERSATIONS_SQL = "SELECT COUNT(*) FROM conversations"
_ROOT_THREAD_COUNT_SQL = """
    SELECT COUNT(*)
    FROM conversations c
    LEFT JOIN conversations parent ON c.parent_conversation_id = parent.conversation_id
    WHERE parent.conversation_id IS NULL
"""
_MISSING_SESSION_PROFILE_COUNT_SQL = """
    SELECT COUNT(*)
    FROM conversations c
    LEFT JOIN session_profiles sp ON sp.conversation_id = c.conversation_id
    WHERE sp.conversation_id IS NULL
"""
_PROFILE_BUCKET_DAY_SQL = (
    "COALESCE(sp.canonical_session_date, "
    "date(COALESCE(sp.first_message_at, json_extract(sp.payload_json, '$.created_at'), sp.source_updated_at, sp.last_message_at)))"
)
_STALE_SESSION_PROFILE_COUNT_SQL = """
    SELECT COUNT(*)
    FROM conversations c
    JOIN session_profiles sp ON sp.conversation_id = c.conversation_id
    WHERE sp.materializer_version != ?
       OR ABS(COALESCE(sp.source_sort_key, 0.0) - COALESCE(c.sort_key, 0.0)) > 0.000001
"""
_ORPHAN_SESSION_PROFILE_COUNT_SQL = """
    SELECT COUNT(*)
    FROM session_profiles sp
    LEFT JOIN conversations c ON c.conversation_id = sp.conversation_id
    WHERE c.conversation_id IS NULL
"""
_EXPECTED_WORK_EVENT_COUNT_SQL = "SELECT COALESCE(SUM(work_event_count), 0) FROM session_profiles"
_EXPECTED_PHASE_COUNT_SQL = "SELECT COALESCE(SUM(phase_count), 0) FROM session_profiles"
_STALE_WORK_EVENT_COUNT_SQL = """
    SELECT COUNT(*)
    FROM session_work_events swe
    JOIN conversations c ON c.conversation_id = swe.conversation_id
    WHERE swe.materializer_version != ?
       OR ABS(COALESCE(swe.source_sort_key, 0.0) - COALESCE(c.sort_key, 0.0)) > 0.000001
"""
_ORPHAN_SESSION_WORK_EVENT_COUNT_SQL = """
    SELECT COUNT(*)
    FROM session_work_events swe
    LEFT JOIN conversations c ON c.conversation_id = swe.conversation_id
    WHERE c.conversation_id IS NULL
"""
_STALE_SESSION_PHASE_COUNT_SQL = """
    SELECT COUNT(*)
    FROM session_phases sph
    JOIN conversations c ON c.conversation_id = sph.conversation_id
    WHERE sph.materializer_version != ?
       OR ABS(COALESCE(sph.source_sort_key, 0.0) - COALESCE(c.sort_key, 0.0)) > 0.000001
"""
_ORPHAN_SESSION_PHASE_COUNT_SQL = """
    SELECT COUNT(*)
    FROM session_phases sph
    LEFT JOIN conversations c ON c.conversation_id = sph.conversation_id
    WHERE c.conversation_id IS NULL
"""
_STALE_WORK_THREAD_COUNT_SQL = """
    WITH RECURSIVE roots(root_id) AS (
        SELECT c.conversation_id
        FROM conversations c
        LEFT JOIN conversations parent ON c.parent_conversation_id = parent.conversation_id
        WHERE parent.conversation_id IS NULL
    ),
    descendants(root_id, conversation_id) AS (
        SELECT root_id, root_id FROM roots
        UNION ALL
        SELECT d.root_id, c.conversation_id
        FROM conversations c
        JOIN descendants d ON c.parent_conversation_id = d.conversation_id
    )
    SELECT COUNT(*)
    FROM work_threads wt
    WHERE wt.materializer_version != ?
       OR EXISTS (
            SELECT 1
            FROM descendants d
            JOIN session_profiles sp ON sp.conversation_id = d.conversation_id
            WHERE d.root_id = wt.thread_id
              AND sp.materialized_at > wt.materialized_at
       )
"""
_ORPHAN_WORK_THREAD_COUNT_SQL = """
    SELECT COUNT(*)
    FROM work_threads wt
    LEFT JOIN conversations c ON c.conversation_id = wt.root_id
    WHERE c.conversation_id IS NULL
"""
_EXPECTED_DAY_SESSION_SUMMARY_COUNT_SQL = f"""
    SELECT COUNT(*) FROM (
        SELECT sp.provider_name, {_PROFILE_BUCKET_DAY_SQL} AS bucket_day
        FROM session_profiles sp
        WHERE {_PROFILE_BUCKET_DAY_SQL} IS NOT NULL
        GROUP BY sp.provider_name, bucket_day
    )
"""
_STALE_DAY_SESSION_SUMMARY_COUNT_SQL = f"""
    WITH expected AS (
        SELECT
            sp.provider_name AS provider_name,
            {_PROFILE_BUCKET_DAY_SQL} AS bucket_day,
            MAX(sp.materialized_at) AS max_profile_materialized_at
        FROM session_profiles sp
        WHERE {_PROFILE_BUCKET_DAY_SQL} IS NOT NULL
        GROUP BY sp.provider_name, bucket_day
    )
    SELECT COUNT(*)
    FROM day_session_summaries dss
    LEFT JOIN expected e
      ON e.provider_name = dss.provider_name
     AND e.bucket_day = dss.day
    WHERE dss.materializer_version != ?
       OR e.bucket_day IS NULL
       OR COALESCE(e.max_profile_materialized_at, '') > COALESCE(dss.materialized_at, '')
"""
_EXPECTED_SESSION_TAG_ROLLUP_COUNT_SQL = f"""
    WITH tag_rows AS (
        SELECT sp.provider_name, {_PROFILE_BUCKET_DAY_SQL} AS bucket_day, tag.value AS tag
        FROM session_profiles sp, json_each(COALESCE(sp.tags_json, '[]')) tag
        WHERE {_PROFILE_BUCKET_DAY_SQL} IS NOT NULL AND tag.value IS NOT NULL AND tag.value != ''
        UNION
        SELECT sp.provider_name, {_PROFILE_BUCKET_DAY_SQL} AS bucket_day, tag.value AS tag
        FROM session_profiles sp, json_each(COALESCE(sp.auto_tags_json, '[]')) tag
        WHERE {_PROFILE_BUCKET_DAY_SQL} IS NOT NULL AND tag.value IS NOT NULL AND tag.value != ''
    )
    SELECT COUNT(*) FROM (
        SELECT provider_name, bucket_day, tag
        FROM tag_rows
        GROUP BY provider_name, bucket_day, tag
    )
"""
_STALE_SESSION_TAG_ROLLUP_COUNT_SQL = f"""
    WITH tag_rows AS (
        SELECT
            sp.provider_name AS provider_name,
            {_PROFILE_BUCKET_DAY_SQL} AS bucket_day,
            tag.value AS tag,
            sp.materialized_at AS profile_materialized_at
        FROM session_profiles sp, json_each(COALESCE(sp.tags_json, '[]')) tag
        WHERE {_PROFILE_BUCKET_DAY_SQL} IS NOT NULL AND tag.value IS NOT NULL AND tag.value != ''
        UNION ALL
        SELECT
            sp.provider_name AS provider_name,
            {_PROFILE_BUCKET_DAY_SQL} AS bucket_day,
            tag.value AS tag,
            sp.materialized_at AS profile_materialized_at
        FROM session_profiles sp, json_each(COALESCE(sp.auto_tags_json, '[]')) tag
        WHERE {_PROFILE_BUCKET_DAY_SQL} IS NOT NULL AND tag.value IS NOT NULL AND tag.value != ''
    ),
    expected AS (
        SELECT provider_name, bucket_day, tag, MAX(profile_materialized_at) AS max_profile_materialized_at
        FROM tag_rows
        GROUP BY provider_name, bucket_day, tag
    )
    SELECT COUNT(*)
    FROM session_tag_rollups str
    LEFT JOIN expected e
      ON e.provider_name = str.provider_name
     AND e.bucket_day = str.bucket_day
     AND e.tag = str.tag
    WHERE str.materializer_version != ?
       OR e.tag IS NULL
       OR COALESCE(e.max_profile_materialized_at, '') > COALESCE(str.materialized_at, '')
"""
_SESSION_PROFILE_REPAIR_CANDIDATES_SQL = """
    SELECT c.conversation_id
    FROM conversations c
    LEFT JOIN session_profiles sp ON sp.conversation_id = c.conversation_id
    WHERE sp.conversation_id IS NULL
       OR sp.materializer_version != ?
       OR COALESCE(sp.source_updated_at, '') != COALESCE(c.updated_at, '')
    ORDER BY c.conversation_id
"""
_ROOT_THREAD_IDS_SQL = """
    SELECT c.conversation_id
    FROM conversations c
    LEFT JOIN conversations parent ON c.parent_conversation_id = parent.conversation_id
    WHERE parent.conversation_id IS NULL
    ORDER BY c.conversation_id
"""
_ALL_CONVERSATION_IDS_SQL = "SELECT conversation_id FROM conversations ORDER BY COALESCE(sort_key, 0) DESC, conversation_id"
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


def _chunked(items: Sequence[str], *, size: int) -> Iterable[Sequence[str]]:
    for index in range(0, len(items), size):
        yield items[index : index + size]


def _sync_attachment_batch(
    conn: sqlite3.Connection,
    conversation_ids: Sequence[str],
) -> dict[str, list[AttachmentRecord]]:
    if not conversation_ids:
        return {}
    placeholders = ", ".join("?" for _ in conversation_ids)
    rows = conn.execute(
        f"""
        SELECT a.*, r.message_id, r.conversation_id
        FROM attachments a
        JOIN attachment_refs r ON a.attachment_id = r.attachment_id
        WHERE r.conversation_id IN ({placeholders})
        """,
        tuple(conversation_ids),
    ).fetchall()
    result: dict[str, list[AttachmentRecord]] = {conversation_id: [] for conversation_id in conversation_ids}
    for row in rows:
        conversation_id = str(row["conversation_id"])
        result.setdefault(conversation_id, []).append(
            AttachmentRecord(
                attachment_id=row["attachment_id"],
                conversation_id=conversation_id,
                message_id=row["message_id"],
                mime_type=row["mime_type"],
                size_bytes=row["size_bytes"],
                path=row["path"],
                provider_meta=None,
            )
        )
    return result


def _load_sync_batch(
    conn: sqlite3.Connection,
    conversation_ids: Sequence[str],
) -> tuple[list[ConversationRecord], list[MessageRecord], dict[str, list[AttachmentRecord]], list[ContentBlockRecord]]:
    placeholders = ", ".join("?" for _ in conversation_ids)
    conversations = [
        _row_to_conversation(row)
        for row in conn.execute(
            f"SELECT * FROM conversations WHERE conversation_id IN ({placeholders})",
            tuple(conversation_ids),
        ).fetchall()
    ]
    messages = [
        _row_to_message(row)
        for row in conn.execute(
            f"""
            SELECT *
            FROM messages
            WHERE conversation_id IN ({placeholders})
            ORDER BY conversation_id, sort_key, message_id
            """,
            tuple(conversation_ids),
        ).fetchall()
    ]
    blocks = [
        _row_to_content_block(row)
        for row in conn.execute(
            f"""
            SELECT *
            FROM content_blocks
            WHERE conversation_id IN ({placeholders})
            ORDER BY conversation_id, message_id, block_index
            """,
            tuple(conversation_ids),
        ).fetchall()
    ]
    return conversations, messages, _sync_attachment_batch(conn, conversation_ids), blocks


async def _load_async_batch(
    conn: aiosqlite.Connection,
    conversation_ids: Sequence[str],
) -> tuple[list[ConversationRecord], list[MessageRecord], dict[str, list[AttachmentRecord]], list[ContentBlockRecord]]:
    placeholders = ", ".join("?" for _ in conversation_ids)
    conversations = [
        _row_to_conversation(row)
        for row in await (
            await conn.execute(
                f"SELECT * FROM conversations WHERE conversation_id IN ({placeholders})",
                tuple(conversation_ids),
            )
        ).fetchall()
    ]
    messages = [
        _row_to_message(row)
        for row in await (
            await conn.execute(
                f"""
                SELECT *
                FROM messages
                WHERE conversation_id IN ({placeholders})
                ORDER BY conversation_id, sort_key, message_id
                """,
                tuple(conversation_ids),
            )
        ).fetchall()
    ]
    blocks = [
        _row_to_content_block(row)
        for row in await (
            await conn.execute(
                f"""
                SELECT *
                FROM content_blocks
                WHERE conversation_id IN ({placeholders})
                ORDER BY conversation_id, message_id, block_index
                """,
                tuple(conversation_ids),
            )
        ).fetchall()
    ]
    attachments = await get_attachments_batch(conn, list(conversation_ids))
    return conversations, messages, attachments, blocks


def _hydrate_conversations(
    conversations: list[ConversationRecord],
    messages: list[MessageRecord],
    attachments_by_conversation: dict[str, list[AttachmentRecord]],
    blocks: list[ContentBlockRecord],
) -> list[object]:
    messages_by_conversation: dict[str, list[MessageRecord]] = defaultdict(list)
    blocks_by_conversation: dict[str, list[ContentBlockRecord]] = defaultdict(list)
    for message in messages:
        messages_by_conversation[str(message.conversation_id)].append(message)
    for block in blocks:
        blocks_by_conversation[str(block.conversation_id)].append(block)

    hydrated: list[object] = []
    for conversation in conversations:
        conversation_id = str(conversation.conversation_id)
        attached_messages = attach_blocks_to_messages(
            messages_by_conversation.get(conversation_id, []),
            blocks_by_conversation.get(conversation_id, []),
        )
        hydrated.append(
            conversation_from_records(
                conversation,
                attached_messages,
                attachments_by_conversation.get(conversation_id, []),
            )
        )
    return hydrated


def _replace_session_profile_sync(conn: sqlite3.Connection, record: SessionProfileRecord) -> None:
    conn.execute("DELETE FROM session_profiles WHERE conversation_id = ?", (record.conversation_id,))
    conn.execute(
        """
        INSERT INTO session_profiles (
            conversation_id,
            materializer_version,
            materialized_at,
            source_updated_at,
            source_sort_key,
            provider_name,
            title,
            first_message_at,
            last_message_at,
            canonical_session_date,
            primary_work_kind,
            repo_paths_json,
            canonical_projects_json,
            tags_json,
            auto_tags_json,
            message_count,
            work_event_count,
            phase_count,
            word_count,
            tool_use_count,
            thinking_count,
            total_cost_usd,
            total_duration_ms,
            engaged_duration_ms,
            wall_duration_ms,
            payload_json,
            search_text
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            record.conversation_id,
            record.materializer_version,
            record.materialized_at,
            record.source_updated_at,
            record.source_sort_key,
            record.provider_name,
            record.title,
            record.first_message_at,
            record.last_message_at,
            record.canonical_session_date,
            record.primary_work_kind,
            _json_array_or_none(record.repo_paths),
            _json_array_or_none(record.canonical_projects),
            _json_array_or_none(record.tags),
            _json_array_or_none(record.auto_tags),
            record.message_count,
            record.work_event_count,
            record.phase_count,
            record.word_count,
            record.tool_use_count,
            record.thinking_count,
            record.total_cost_usd,
            record.total_duration_ms,
            record.engaged_duration_ms,
            record.wall_duration_ms,
            _json_or_none(record.payload),
            record.search_text,
        ),
    )


def _replace_session_work_events_sync(
    conn: sqlite3.Connection,
    conversation_id: str,
    records: Sequence[object],
) -> None:
    conn.execute("DELETE FROM session_work_events WHERE conversation_id = ?", (conversation_id,))
    if records:
        conn.executemany(
            """
            INSERT INTO session_work_events (
                event_id,
                conversation_id,
                materializer_version,
                materialized_at,
                source_updated_at,
                source_sort_key,
                provider_name,
                event_index,
                kind,
                confidence,
                start_index,
                end_index,
                start_time,
                end_time,
                duration_ms,
                canonical_session_date,
                summary,
                file_paths_json,
                tools_used_json,
                payload_json,
                search_text
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    record.event_id,
                    record.conversation_id,
                    record.materializer_version,
                    record.materialized_at,
                    record.source_updated_at,
                    record.source_sort_key,
                    record.provider_name,
                    record.event_index,
                    record.kind,
                    record.confidence,
                    record.start_index,
                    record.end_index,
                    record.start_time,
                    record.end_time,
                    record.duration_ms,
                    record.canonical_session_date,
                    record.summary,
                    _json_array_or_none(record.file_paths),
                    _json_array_or_none(record.tools_used),
                    _json_or_none(record.payload),
                    record.search_text,
                )
                for record in records
            ],
        )


def _replace_session_phases_sync(
    conn: sqlite3.Connection,
    conversation_id: str,
    records: Sequence[object],
) -> None:
    conn.execute("DELETE FROM session_phases WHERE conversation_id = ?", (conversation_id,))
    if records:
        conn.executemany(
            """
            INSERT INTO session_phases (
                phase_id,
                conversation_id,
                materializer_version,
                materialized_at,
                source_updated_at,
                source_sort_key,
                provider_name,
                phase_index,
                kind,
                start_index,
                end_index,
                start_time,
                end_time,
                duration_ms,
                canonical_session_date,
                tool_counts_json,
                word_count,
                payload_json,
                search_text
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    record.phase_id,
                    record.conversation_id,
                    record.materializer_version,
                    record.materialized_at,
                    record.source_updated_at,
                    record.source_sort_key,
                    record.provider_name,
                    record.phase_index,
                    record.kind,
                    record.start_index,
                    record.end_index,
                    record.start_time,
                    record.end_time,
                    record.duration_ms,
                    record.canonical_session_date,
                    _json_or_none(record.tool_counts),
                    record.word_count,
                    _json_or_none(record.payload),
                    record.search_text,
                )
                for record in records
            ],
        )


def _replace_work_thread_sync(conn: sqlite3.Connection, thread_id: str, record: object | None) -> None:
    conn.execute("DELETE FROM work_threads WHERE thread_id = ?", (thread_id,))
    if record is not None:
        conn.execute(
            """
            INSERT INTO work_threads (
                thread_id,
                root_id,
                materializer_version,
                materialized_at,
                start_time,
                end_time,
                dominant_project,
                session_ids_json,
                session_count,
                depth,
                branch_count,
                total_messages,
                total_cost_usd,
                wall_duration_ms,
                work_event_breakdown_json,
                payload_json,
                search_text
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record.thread_id,
                record.root_id,
                record.materializer_version,
                record.materialized_at,
                record.start_time,
                record.end_time,
                record.dominant_project,
                _json_array_or_none(record.session_ids),
                record.session_count,
                record.depth,
                record.branch_count,
                record.total_messages,
                record.total_cost_usd,
                record.wall_duration_ms,
                _json_or_none(record.work_event_breakdown or {}),
                _json_or_none(record.payload),
                record.search_text,
            ),
        )


def _replace_session_tag_rollup_rows_sync(
    conn: sqlite3.Connection,
    *,
    provider_name: str,
    bucket_day: str,
    records: Sequence[SessionTagRollupRecord],
) -> None:
    conn.execute(
        "DELETE FROM session_tag_rollups WHERE provider_name = ? AND bucket_day = ?",
        (provider_name, bucket_day),
    )
    if records:
        conn.executemany(
            """
            INSERT INTO session_tag_rollups (
                tag,
                bucket_day,
                provider_name,
                materializer_version,
                materialized_at,
                source_updated_at,
                source_sort_key,
                conversation_count,
                explicit_count,
                auto_count,
                project_breakdown_json,
                search_text
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    record.tag,
                    record.bucket_day,
                    record.provider_name,
                    record.materializer_version,
                    record.materialized_at,
                    record.source_updated_at,
                    record.source_sort_key,
                    record.conversation_count,
                    record.explicit_count,
                    record.auto_count,
                    _json_or_none(record.project_breakdown),
                    record.search_text,
                )
                for record in records
            ],
        )


def _replace_day_session_summaries_sync(
    conn: sqlite3.Connection,
    *,
    provider_name: str,
    day: str,
    records: Sequence[DaySessionSummaryRecord],
) -> None:
    conn.execute(
        "DELETE FROM day_session_summaries WHERE provider_name = ? AND day = ?",
        (provider_name, day),
    )
    if records:
        conn.executemany(
            """
            INSERT INTO day_session_summaries (
                day,
                provider_name,
                materializer_version,
                materialized_at,
                source_updated_at,
                source_sort_key,
                conversation_count,
                total_cost_usd,
                total_duration_ms,
                total_wall_duration_ms,
                total_messages,
                total_words,
                work_event_breakdown_json,
                projects_active_json,
                payload_json,
                search_text
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    record.day,
                    record.provider_name,
                    record.materializer_version,
                    record.materialized_at,
                    record.source_updated_at,
                    record.source_sort_key,
                    record.conversation_count,
                    record.total_cost_usd,
                    record.total_duration_ms,
                    record.total_wall_duration_ms,
                    record.total_messages,
                    record.total_words,
                    _json_or_none(record.work_event_breakdown),
                    _json_array_or_none(record.projects_active),
                    _json_or_none(record.payload),
                    record.search_text,
                )
                for record in records
            ],
        )


def _load_sync_provider_day_profile_records(
    conn: sqlite3.Connection,
    *,
    provider_name: str,
    bucket_day: str,
) -> list[SessionProfileRecord]:
    rows = conn.execute(
        f"""
        SELECT *
        FROM session_profiles sp
        WHERE sp.provider_name = ?
          AND {_PROFILE_BUCKET_DAY_SQL} = ?
        ORDER BY COALESCE(sp.source_sort_key, 0) DESC, sp.conversation_id
        """,
        (provider_name, bucket_day),
    ).fetchall()
    return [_row_to_session_profile_record(row) for row in rows]


async def _load_async_provider_day_profile_records(
    conn: aiosqlite.Connection,
    *,
    provider_name: str,
    bucket_day: str,
) -> list[SessionProfileRecord]:
    rows = await (
        await conn.execute(
            f"""
            SELECT *
            FROM session_profiles sp
            WHERE sp.provider_name = ?
              AND {_PROFILE_BUCKET_DAY_SQL} = ?
            ORDER BY COALESCE(sp.source_sort_key, 0) DESC, sp.conversation_id
            """,
            (provider_name, bucket_day),
        )
    ).fetchall()
    return [_row_to_session_profile_record(row) for row in rows]


def _refresh_sync_provider_day_aggregates(
    conn: sqlite3.Connection,
    groups: set[tuple[str, str]],
) -> None:
    for provider_name, bucket_day in groups:
        profile_records = _load_sync_provider_day_profile_records(
            conn,
            provider_name=provider_name,
            bucket_day=bucket_day,
        )
        profiles = [hydrate_session_profile(record) for record in profile_records]
        day_rows = build_day_session_summary_records(profiles) if profiles else []
        tag_rows = build_session_tag_rollup_records(profiles) if profiles else []
        _replace_day_session_summaries_sync(
            conn,
            provider_name=provider_name,
            day=bucket_day,
            records=[row for row in day_rows if row.provider_name == provider_name and row.day == bucket_day],
        )
        _replace_session_tag_rollup_rows_sync(
            conn,
            provider_name=provider_name,
            bucket_day=bucket_day,
            records=[
                row
                for row in tag_rows
                if row.provider_name == provider_name and row.bucket_day == bucket_day
            ],
        )


async def _refresh_async_provider_day_aggregates(
    conn: aiosqlite.Connection,
    groups: set[tuple[str, str]],
    *,
    transaction_depth: int,
) -> None:
    for provider_name, bucket_day in groups:
        profile_records = await _load_async_provider_day_profile_records(
            conn,
            provider_name=provider_name,
            bucket_day=bucket_day,
        )
        profiles = [hydrate_session_profile(record) for record in profile_records]
        day_rows = build_day_session_summary_records(profiles) if profiles else []
        tag_rows = build_session_tag_rollup_records(profiles) if profiles else []
        await replace_day_session_summaries(
            conn,
            provider_name=provider_name,
            day=bucket_day,
            records=[row for row in day_rows if row.provider_name == provider_name and row.day == bucket_day],
            transaction_depth=transaction_depth,
        )
        await replace_session_tag_rollup_rows(
            conn,
            provider_name=provider_name,
            bucket_day=bucket_day,
            records=[
                row
                for row in tag_rows
                if row.provider_name == provider_name and row.bucket_day == bucket_day
            ],
            transaction_depth=transaction_depth,
        )


def _profile_provider_day(record: SessionProfileRecord | None) -> tuple[str, str] | None:
    if record is None:
        return None
    if record.canonical_session_date:
        return (record.provider_name, record.canonical_session_date)
    day_candidates = [
        record.first_message_at,
        str(record.payload.get("created_at")) if isinstance(record.payload, dict) and record.payload.get("created_at") else None,
        record.source_updated_at,
        record.last_message_at,
    ]
    bucket_day: str | None = None
    for candidate in day_candidates:
        if not candidate:
            continue
        try:
            bucket_day = date_from_iso(str(candidate)[:10]).isoformat()
            break
        except ValueError:
            continue
    if bucket_day is None:
        return None
    return (record.provider_name, bucket_day)


def thread_root_id_sync(conn: sqlite3.Connection, conversation_id: str) -> str | None:
    row = conn.execute(_THREAD_ROOT_ID_SQL, (conversation_id,)).fetchone()
    return str(row["conversation_id"]) if row else None


async def thread_root_id_async(conn: aiosqlite.Connection, conversation_id: str) -> str | None:
    row = await (await conn.execute(_THREAD_ROOT_ID_SQL, (conversation_id,))).fetchone()
    return str(row["conversation_id"]) if row else None


def thread_conversation_ids_sync(conn: sqlite3.Connection, root_id: str) -> list[str]:
    rows = conn.execute(_THREAD_CONVERSATION_IDS_SQL, (root_id,)).fetchall()
    return [str(row["conversation_id"]) for row in rows]


async def thread_conversation_ids_async(conn: aiosqlite.Connection, root_id: str) -> list[str]:
    rows = await (await conn.execute(_THREAD_CONVERSATION_IDS_SQL, (root_id,))).fetchall()
    return [str(row["conversation_id"]) for row in rows]


def session_profile_repair_candidate_ids_sync(conn: sqlite3.Connection) -> list[str]:
    rows = conn.execute(
        _SESSION_PROFILE_REPAIR_CANDIDATES_SQL,
        (SESSION_PRODUCT_MATERIALIZER_VERSION,),
    ).fetchall()
    return [str(row["conversation_id"]) for row in rows]


async def session_profile_repair_candidate_ids_async(conn: aiosqlite.Connection) -> list[str]:
    rows = await (
        await conn.execute(
            _SESSION_PROFILE_REPAIR_CANDIDATES_SQL,
            (SESSION_PRODUCT_MATERIALIZER_VERSION,),
        )
    ).fetchall()
    return [str(row["conversation_id"]) for row in rows]


def session_product_status_sync(conn: sqlite3.Connection) -> dict[str, int | bool]:
    session_profiles_exists = bool(conn.execute(_SESSION_PROFILES_EXISTS_SQL).fetchone())
    session_profiles_fts_exists = bool(conn.execute(_SESSION_PROFILES_FTS_EXISTS_SQL).fetchone())
    session_work_events_exists = bool(conn.execute(_SESSION_WORK_EVENTS_EXISTS_SQL).fetchone())
    session_work_events_fts_exists = bool(conn.execute(_SESSION_WORK_EVENTS_FTS_EXISTS_SQL).fetchone())
    session_phases_exists = bool(conn.execute(_SESSION_PHASES_EXISTS_SQL).fetchone())
    work_threads_exists = bool(conn.execute(_WORK_THREADS_EXISTS_SQL).fetchone())
    work_threads_fts_exists = bool(conn.execute(_WORK_THREADS_FTS_EXISTS_SQL).fetchone())
    session_tag_rollups_exists = bool(conn.execute(_SESSION_TAG_ROLLUPS_EXISTS_SQL).fetchone())
    day_session_summaries_exists = bool(conn.execute(_DAY_SESSION_SUMMARIES_EXISTS_SQL).fetchone())

    total_conversations = int(conn.execute(_TOTAL_CONVERSATIONS_SQL).fetchone()[0] or 0)
    root_threads = int(conn.execute(_ROOT_THREAD_COUNT_SQL).fetchone()[0] or 0)
    profile_count = int(conn.execute(_SESSION_PROFILE_COUNT_SQL).fetchone()[0] or 0) if session_profiles_exists else 0
    profile_fts_count = int(conn.execute(_SESSION_PROFILE_FTS_DOC_COUNT_SQL).fetchone()[0] or 0) if session_profiles_fts_exists else 0
    profile_fts_duplicate_count = int(
        conn.execute(_SESSION_PROFILE_FTS_DUPLICATE_COUNT_SQL).fetchone()[0] or 0
    ) if session_profiles_fts_exists else 0
    work_event_count = int(conn.execute(_SESSION_WORK_EVENT_COUNT_SQL).fetchone()[0] or 0) if session_work_events_exists else 0
    work_event_fts_count = int(
        conn.execute(_SESSION_WORK_EVENT_FTS_DOC_COUNT_SQL).fetchone()[0] or 0
    ) if session_work_events_fts_exists else 0
    work_event_fts_duplicate_count = int(
        conn.execute(_SESSION_WORK_EVENT_FTS_DUPLICATE_COUNT_SQL).fetchone()[0] or 0
    ) if session_work_events_fts_exists else 0
    phase_count = int(conn.execute(_SESSION_PHASE_COUNT_SQL).fetchone()[0] or 0) if session_phases_exists else 0
    thread_count = int(conn.execute(_WORK_THREAD_COUNT_SQL).fetchone()[0] or 0) if work_threads_exists else 0
    thread_fts_count = int(conn.execute(_WORK_THREAD_FTS_DOC_COUNT_SQL).fetchone()[0] or 0) if work_threads_fts_exists else 0
    thread_fts_duplicate_count = int(
        conn.execute(_WORK_THREAD_FTS_DUPLICATE_COUNT_SQL).fetchone()[0] or 0
    ) if work_threads_fts_exists else 0
    tag_rollup_count = int(conn.execute(_SESSION_TAG_ROLLUP_COUNT_SQL).fetchone()[0] or 0) if session_tag_rollups_exists else 0
    day_summary_count = int(conn.execute(_DAY_SESSION_SUMMARY_COUNT_SQL).fetchone()[0] or 0) if day_session_summaries_exists else 0
    missing_profile_count = int(conn.execute(_MISSING_SESSION_PROFILE_COUNT_SQL).fetchone()[0] or 0) if session_profiles_exists else total_conversations
    stale_profile_count = int(
        conn.execute(_STALE_SESSION_PROFILE_COUNT_SQL, (SESSION_PRODUCT_MATERIALIZER_VERSION,)).fetchone()[0] or 0
    ) if session_profiles_exists else total_conversations
    orphan_profile_count = int(conn.execute(_ORPHAN_SESSION_PROFILE_COUNT_SQL).fetchone()[0] or 0) if session_profiles_exists else 0
    expected_work_event_count = int(conn.execute(_EXPECTED_WORK_EVENT_COUNT_SQL).fetchone()[0] or 0) if session_profiles_exists else 0
    expected_phase_count = int(conn.execute(_EXPECTED_PHASE_COUNT_SQL).fetchone()[0] or 0) if session_profiles_exists else 0
    stale_work_event_count = int(
        conn.execute(_STALE_WORK_EVENT_COUNT_SQL, (SESSION_PRODUCT_MATERIALIZER_VERSION,)).fetchone()[0] or 0
    ) if session_work_events_exists else expected_work_event_count
    orphan_work_event_count = int(conn.execute(_ORPHAN_SESSION_WORK_EVENT_COUNT_SQL).fetchone()[0] or 0) if session_work_events_exists else 0
    stale_phase_count = int(
        conn.execute(_STALE_SESSION_PHASE_COUNT_SQL, (SESSION_PRODUCT_MATERIALIZER_VERSION,)).fetchone()[0] or 0
    ) if session_phases_exists else expected_phase_count
    orphan_phase_count = int(conn.execute(_ORPHAN_SESSION_PHASE_COUNT_SQL).fetchone()[0] or 0) if session_phases_exists else 0
    stale_thread_count = int(
        conn.execute(_STALE_WORK_THREAD_COUNT_SQL, (SESSION_PRODUCT_MATERIALIZER_VERSION,)).fetchone()[0] or 0
    ) if work_threads_exists else root_threads
    orphan_thread_count = int(conn.execute(_ORPHAN_WORK_THREAD_COUNT_SQL).fetchone()[0] or 0) if work_threads_exists else 0
    expected_tag_rollup_count = int(
        conn.execute(_EXPECTED_SESSION_TAG_ROLLUP_COUNT_SQL).fetchone()[0] or 0
    ) if session_profiles_exists else 0
    stale_tag_rollup_count = int(
        conn.execute(_STALE_SESSION_TAG_ROLLUP_COUNT_SQL, (SESSION_PRODUCT_MATERIALIZER_VERSION,)).fetchone()[0] or 0
    ) if session_tag_rollups_exists else expected_tag_rollup_count
    expected_day_summary_count = int(
        conn.execute(_EXPECTED_DAY_SESSION_SUMMARY_COUNT_SQL).fetchone()[0] or 0
    ) if session_profiles_exists else 0
    stale_day_summary_count = int(
        conn.execute(_STALE_DAY_SESSION_SUMMARY_COUNT_SQL, (SESSION_PRODUCT_MATERIALIZER_VERSION,)).fetchone()[0] or 0
    ) if day_session_summaries_exists else expected_day_summary_count
    return {
        "total_conversations": total_conversations,
        "root_threads": root_threads,
        "profile_count": profile_count,
        "profile_fts_count": profile_fts_count,
        "profile_fts_duplicate_count": profile_fts_duplicate_count,
        "work_event_count": work_event_count,
        "work_event_fts_count": work_event_fts_count,
        "work_event_fts_duplicate_count": work_event_fts_duplicate_count,
        "phase_count": phase_count,
        "thread_count": thread_count,
        "thread_fts_count": thread_fts_count,
        "thread_fts_duplicate_count": thread_fts_duplicate_count,
        "tag_rollup_count": tag_rollup_count,
        "day_summary_count": day_summary_count,
        "missing_profile_count": missing_profile_count,
        "stale_profile_count": stale_profile_count,
        "orphan_profile_count": orphan_profile_count,
        "expected_work_event_count": expected_work_event_count,
        "stale_work_event_count": stale_work_event_count,
        "orphan_work_event_count": orphan_work_event_count,
        "expected_phase_count": expected_phase_count,
        "stale_phase_count": stale_phase_count,
        "orphan_phase_count": orphan_phase_count,
        "stale_thread_count": stale_thread_count,
        "orphan_thread_count": orphan_thread_count,
        "expected_tag_rollup_count": expected_tag_rollup_count,
        "stale_tag_rollup_count": stale_tag_rollup_count,
        "expected_day_summary_count": expected_day_summary_count,
        "stale_day_summary_count": stale_day_summary_count,
        "profiles_ready": session_profiles_exists and missing_profile_count == 0 and stale_profile_count == 0 and orphan_profile_count == 0,
        "profiles_fts_ready": session_profiles_fts_exists and profile_fts_count == profile_count and profile_fts_duplicate_count == 0,
        "work_events_ready": session_work_events_exists and work_event_count == expected_work_event_count and stale_work_event_count == 0 and orphan_work_event_count == 0,
        "work_events_fts_ready": session_work_events_fts_exists and work_event_fts_count == work_event_count and work_event_fts_duplicate_count == 0,
        "phases_ready": session_phases_exists and phase_count == expected_phase_count and stale_phase_count == 0 and orphan_phase_count == 0,
        "threads_ready": work_threads_exists and thread_count == root_threads and stale_thread_count == 0 and orphan_thread_count == 0,
        "threads_fts_ready": work_threads_fts_exists and thread_fts_count == thread_count and thread_fts_duplicate_count == 0,
        "tag_rollups_ready": session_tag_rollups_exists and tag_rollup_count == expected_tag_rollup_count and stale_tag_rollup_count == 0,
        "day_summaries_ready": day_session_summaries_exists and day_summary_count == expected_day_summary_count and stale_day_summary_count == 0,
        "week_summaries_ready": day_session_summaries_exists and day_summary_count == expected_day_summary_count and stale_day_summary_count == 0,
    }


async def session_product_status_async(conn: aiosqlite.Connection) -> dict[str, int | bool]:
    def _to_int(row: tuple[object, ...] | None) -> int:
        return int(row[0] or 0) if row else 0

    session_profiles_exists = bool(await (await conn.execute(_SESSION_PROFILES_EXISTS_SQL)).fetchone())
    session_profiles_fts_exists = bool(await (await conn.execute(_SESSION_PROFILES_FTS_EXISTS_SQL)).fetchone())
    session_work_events_exists = bool(await (await conn.execute(_SESSION_WORK_EVENTS_EXISTS_SQL)).fetchone())
    session_work_events_fts_exists = bool(await (await conn.execute(_SESSION_WORK_EVENTS_FTS_EXISTS_SQL)).fetchone())
    session_phases_exists = bool(await (await conn.execute(_SESSION_PHASES_EXISTS_SQL)).fetchone())
    work_threads_exists = bool(await (await conn.execute(_WORK_THREADS_EXISTS_SQL)).fetchone())
    work_threads_fts_exists = bool(await (await conn.execute(_WORK_THREADS_FTS_EXISTS_SQL)).fetchone())
    session_tag_rollups_exists = bool(await (await conn.execute(_SESSION_TAG_ROLLUPS_EXISTS_SQL)).fetchone())
    day_session_summaries_exists = bool(await (await conn.execute(_DAY_SESSION_SUMMARIES_EXISTS_SQL)).fetchone())
    total_conversations = _to_int(await (await conn.execute(_TOTAL_CONVERSATIONS_SQL)).fetchone())
    root_threads = _to_int(await (await conn.execute(_ROOT_THREAD_COUNT_SQL)).fetchone())
    profile_count = _to_int(await (await conn.execute(_SESSION_PROFILE_COUNT_SQL)).fetchone()) if session_profiles_exists else 0
    profile_fts_count = _to_int(await (await conn.execute(_SESSION_PROFILE_FTS_DOC_COUNT_SQL)).fetchone()) if session_profiles_fts_exists else 0
    profile_fts_duplicate_count = _to_int(
        await (await conn.execute(_SESSION_PROFILE_FTS_DUPLICATE_COUNT_SQL)).fetchone()
    ) if session_profiles_fts_exists else 0
    work_event_count = _to_int(await (await conn.execute(_SESSION_WORK_EVENT_COUNT_SQL)).fetchone()) if session_work_events_exists else 0
    work_event_fts_count = _to_int(
        await (await conn.execute(_SESSION_WORK_EVENT_FTS_DOC_COUNT_SQL)).fetchone()
    ) if session_work_events_fts_exists else 0
    work_event_fts_duplicate_count = _to_int(
        await (await conn.execute(_SESSION_WORK_EVENT_FTS_DUPLICATE_COUNT_SQL)).fetchone()
    ) if session_work_events_fts_exists else 0
    phase_count = _to_int(await (await conn.execute(_SESSION_PHASE_COUNT_SQL)).fetchone()) if session_phases_exists else 0
    thread_count = _to_int(await (await conn.execute(_WORK_THREAD_COUNT_SQL)).fetchone()) if work_threads_exists else 0
    thread_fts_count = _to_int(await (await conn.execute(_WORK_THREAD_FTS_DOC_COUNT_SQL)).fetchone()) if work_threads_fts_exists else 0
    thread_fts_duplicate_count = _to_int(
        await (await conn.execute(_WORK_THREAD_FTS_DUPLICATE_COUNT_SQL)).fetchone()
    ) if work_threads_fts_exists else 0
    tag_rollup_count = _to_int(await (await conn.execute(_SESSION_TAG_ROLLUP_COUNT_SQL)).fetchone()) if session_tag_rollups_exists else 0
    day_summary_count = _to_int(await (await conn.execute(_DAY_SESSION_SUMMARY_COUNT_SQL)).fetchone()) if day_session_summaries_exists else 0
    missing_profile_count = _to_int(await (await conn.execute(_MISSING_SESSION_PROFILE_COUNT_SQL)).fetchone()) if session_profiles_exists else total_conversations
    stale_profile_count = _to_int(
        await (await conn.execute(_STALE_SESSION_PROFILE_COUNT_SQL, (SESSION_PRODUCT_MATERIALIZER_VERSION,))).fetchone()
    ) if session_profiles_exists else total_conversations
    orphan_profile_count = _to_int(await (await conn.execute(_ORPHAN_SESSION_PROFILE_COUNT_SQL)).fetchone()) if session_profiles_exists else 0
    expected_work_event_count = _to_int(await (await conn.execute(_EXPECTED_WORK_EVENT_COUNT_SQL)).fetchone()) if session_profiles_exists else 0
    expected_phase_count = _to_int(await (await conn.execute(_EXPECTED_PHASE_COUNT_SQL)).fetchone()) if session_profiles_exists else 0
    stale_work_event_count = _to_int(
        await (await conn.execute(_STALE_WORK_EVENT_COUNT_SQL, (SESSION_PRODUCT_MATERIALIZER_VERSION,))).fetchone()
    ) if session_work_events_exists else expected_work_event_count
    orphan_work_event_count = _to_int(await (await conn.execute(_ORPHAN_SESSION_WORK_EVENT_COUNT_SQL)).fetchone()) if session_work_events_exists else 0
    stale_phase_count = _to_int(
        await (await conn.execute(_STALE_SESSION_PHASE_COUNT_SQL, (SESSION_PRODUCT_MATERIALIZER_VERSION,))).fetchone()
    ) if session_phases_exists else expected_phase_count
    orphan_phase_count = _to_int(await (await conn.execute(_ORPHAN_SESSION_PHASE_COUNT_SQL)).fetchone()) if session_phases_exists else 0
    stale_thread_count = _to_int(
        await (await conn.execute(_STALE_WORK_THREAD_COUNT_SQL, (SESSION_PRODUCT_MATERIALIZER_VERSION,))).fetchone()
    ) if work_threads_exists else root_threads
    orphan_thread_count = _to_int(await (await conn.execute(_ORPHAN_WORK_THREAD_COUNT_SQL)).fetchone()) if work_threads_exists else 0
    expected_tag_rollup_count = _to_int(await (await conn.execute(_EXPECTED_SESSION_TAG_ROLLUP_COUNT_SQL)).fetchone()) if session_profiles_exists else 0
    stale_tag_rollup_count = _to_int(
        await (await conn.execute(_STALE_SESSION_TAG_ROLLUP_COUNT_SQL, (SESSION_PRODUCT_MATERIALIZER_VERSION,))).fetchone()
    ) if session_tag_rollups_exists else expected_tag_rollup_count
    expected_day_summary_count = _to_int(await (await conn.execute(_EXPECTED_DAY_SESSION_SUMMARY_COUNT_SQL)).fetchone()) if session_profiles_exists else 0
    stale_day_summary_count = _to_int(
        await (await conn.execute(_STALE_DAY_SESSION_SUMMARY_COUNT_SQL, (SESSION_PRODUCT_MATERIALIZER_VERSION,))).fetchone()
    ) if day_session_summaries_exists else expected_day_summary_count
    return {
        "total_conversations": total_conversations,
        "root_threads": root_threads,
        "profile_count": profile_count,
        "profile_fts_count": profile_fts_count,
        "profile_fts_duplicate_count": profile_fts_duplicate_count,
        "work_event_count": work_event_count,
        "work_event_fts_count": work_event_fts_count,
        "work_event_fts_duplicate_count": work_event_fts_duplicate_count,
        "phase_count": phase_count,
        "thread_count": thread_count,
        "thread_fts_count": thread_fts_count,
        "thread_fts_duplicate_count": thread_fts_duplicate_count,
        "tag_rollup_count": tag_rollup_count,
        "day_summary_count": day_summary_count,
        "missing_profile_count": missing_profile_count,
        "stale_profile_count": stale_profile_count,
        "orphan_profile_count": orphan_profile_count,
        "expected_work_event_count": expected_work_event_count,
        "stale_work_event_count": stale_work_event_count,
        "orphan_work_event_count": orphan_work_event_count,
        "expected_phase_count": expected_phase_count,
        "stale_phase_count": stale_phase_count,
        "orphan_phase_count": orphan_phase_count,
        "stale_thread_count": stale_thread_count,
        "orphan_thread_count": orphan_thread_count,
        "expected_tag_rollup_count": expected_tag_rollup_count,
        "stale_tag_rollup_count": stale_tag_rollup_count,
        "expected_day_summary_count": expected_day_summary_count,
        "stale_day_summary_count": stale_day_summary_count,
        "profiles_ready": session_profiles_exists and missing_profile_count == 0 and stale_profile_count == 0 and orphan_profile_count == 0,
        "profiles_fts_ready": session_profiles_fts_exists and profile_fts_count == profile_count and profile_fts_duplicate_count == 0,
        "work_events_ready": session_work_events_exists and work_event_count == expected_work_event_count and stale_work_event_count == 0 and orphan_work_event_count == 0,
        "work_events_fts_ready": session_work_events_fts_exists and work_event_fts_count == work_event_count and work_event_fts_duplicate_count == 0,
        "phases_ready": session_phases_exists and phase_count == expected_phase_count and stale_phase_count == 0 and orphan_phase_count == 0,
        "threads_ready": work_threads_exists and thread_count == root_threads and stale_thread_count == 0 and orphan_thread_count == 0,
        "threads_fts_ready": work_threads_fts_exists and thread_fts_count == thread_count and thread_fts_duplicate_count == 0,
        "tag_rollups_ready": session_tag_rollups_exists and tag_rollup_count == expected_tag_rollup_count and stale_tag_rollup_count == 0,
        "day_summaries_ready": day_session_summaries_exists and day_summary_count == expected_day_summary_count and stale_day_summary_count == 0,
        "week_summaries_ready": day_session_summaries_exists and day_summary_count == expected_day_summary_count and stale_day_summary_count == 0,
    }


def rebuild_session_products_sync(
    conn: sqlite3.Connection,
    *,
    conversation_ids: Sequence[str] | None = None,
    page_size: int = 100,
) -> dict[str, int]:
    if conversation_ids is None:
        conn.execute("DELETE FROM session_work_events")
        conn.execute("DELETE FROM session_phases")
        conn.execute("DELETE FROM session_profiles")
        conn.execute("DELETE FROM session_tag_rollups")
        conn.execute("DELETE FROM day_session_summaries")
        conversation_ids = [str(row["conversation_id"]) for row in conn.execute(_ALL_CONVERSATION_IDS_SQL).fetchall()]
    if not conversation_ids:
        conn.execute("DELETE FROM work_threads")
        conn.execute("DELETE FROM session_phases")
        conn.execute("DELETE FROM session_tag_rollups")
        conn.execute("DELETE FROM day_session_summaries")
        conn.commit()
        return {"profiles": 0, "work_events": 0, "phases": 0, "threads": 0, "tag_rollups": 0, "day_summaries": 0}

    profile_count = 0
    work_event_count = 0
    phase_count = 0
    for chunk in _chunked(list(conversation_ids), size=page_size):
        conversations, messages, attachments, blocks = _load_sync_batch(conn, chunk)
        for conversation in _hydrate_conversations(conversations, messages, attachments, blocks):
            profile_record, event_records, phase_records = build_session_product_records(conversation)
            _replace_session_profile_sync(conn, profile_record)
            _replace_session_work_events_sync(conn, profile_record.conversation_id, event_records)
            _replace_session_phases_sync(conn, profile_record.conversation_id, phase_records)
            profile_count += 1
            work_event_count += len(event_records)
            phase_count += len(phase_records)

    conn.execute("DELETE FROM work_threads")
    thread_records = _build_all_thread_records_sync(conn)
    for record in thread_records:
        _replace_work_thread_sync(conn, record.thread_id, record)
    all_profile_records = [
        _row_to_session_profile_record(row)
        for row in conn.execute(
            "SELECT * FROM session_profiles ORDER BY COALESCE(source_sort_key, 0) DESC, conversation_id"
        ).fetchall()
    ]
    all_profiles = [hydrate_session_profile(record) for record in all_profile_records]
    conn.execute("DELETE FROM session_tag_rollups")
    tag_rows = build_session_tag_rollup_records(all_profiles)
    for provider_name, bucket_day in sorted({(row.provider_name, row.bucket_day) for row in tag_rows}):
        _replace_session_tag_rollup_rows_sync(
            conn,
            provider_name=provider_name,
            bucket_day=bucket_day,
            records=[
                row
                for row in tag_rows
                if row.provider_name == provider_name and row.bucket_day == bucket_day
            ],
        )
    conn.execute("DELETE FROM day_session_summaries")
    day_rows = build_day_session_summary_records(all_profiles)
    for provider_name, bucket_day in sorted({(row.provider_name, row.day) for row in day_rows}):
        _replace_day_session_summaries_sync(
            conn,
            provider_name=provider_name,
            day=bucket_day,
            records=[row for row in day_rows if row.provider_name == provider_name and row.day == bucket_day],
        )
    conn.commit()
    return {
        "profiles": profile_count,
        "work_events": work_event_count,
        "phases": phase_count,
        "threads": len(thread_records),
        "tag_rollups": len(tag_rows),
        "day_summaries": len(day_rows),
    }


async def rebuild_session_products_async(
    conn: aiosqlite.Connection,
    *,
    conversation_ids: Sequence[str] | None = None,
    page_size: int = 100,
    transaction_depth: int = 0,
) -> dict[str, int]:
    if conversation_ids is None:
        await conn.execute("DELETE FROM session_work_events")
        await conn.execute("DELETE FROM session_phases")
        await conn.execute("DELETE FROM session_profiles")
        await conn.execute("DELETE FROM session_tag_rollups")
        await conn.execute("DELETE FROM day_session_summaries")
        rows = await (await conn.execute(_ALL_CONVERSATION_IDS_SQL)).fetchall()
        conversation_ids = [str(row["conversation_id"]) for row in rows]
    if not conversation_ids:
        await conn.execute("DELETE FROM work_threads")
        await conn.execute("DELETE FROM session_phases")
        await conn.execute("DELETE FROM session_tag_rollups")
        await conn.execute("DELETE FROM day_session_summaries")
        return {"profiles": 0, "work_events": 0, "phases": 0, "threads": 0, "tag_rollups": 0, "day_summaries": 0}

    profile_count = 0
    work_event_count = 0
    phase_count = 0
    for chunk in _chunked(list(conversation_ids), size=page_size):
        conversations, messages, attachments, blocks = await _load_async_batch(conn, chunk)
        for conversation in _hydrate_conversations(conversations, messages, attachments, blocks):
            profile_record, event_records, phase_records = build_session_product_records(conversation)
            await replace_session_profile(conn, profile_record, transaction_depth)
            await replace_session_work_events(conn, profile_record.conversation_id, event_records, transaction_depth)
            await replace_session_phases(conn, profile_record.conversation_id, phase_records, transaction_depth)
            profile_count += 1
            work_event_count += len(event_records)
            phase_count += len(phase_records)

    await conn.execute("DELETE FROM work_threads")
    thread_records = await _build_all_thread_records_async(conn)
    for record in thread_records:
        await replace_work_thread(conn, record.thread_id, record, transaction_depth)
    rows = await (
        await conn.execute(
            "SELECT * FROM session_profiles ORDER BY COALESCE(source_sort_key, 0) DESC, conversation_id"
        )
    ).fetchall()
    all_profiles = [hydrate_session_profile(_row_to_session_profile_record(row)) for row in rows]
    await conn.execute("DELETE FROM session_tag_rollups")
    tag_rows = build_session_tag_rollup_records(all_profiles)
    for provider_name, bucket_day in sorted({(row.provider_name, row.bucket_day) for row in tag_rows}):
        await replace_session_tag_rollup_rows(
            conn,
            provider_name=provider_name,
            bucket_day=bucket_day,
            records=[
                row
                for row in tag_rows
                if row.provider_name == provider_name and row.bucket_day == bucket_day
            ],
            transaction_depth=transaction_depth,
        )
    await conn.execute("DELETE FROM day_session_summaries")
    day_rows = build_day_session_summary_records(all_profiles)
    for provider_name, bucket_day in sorted({(row.provider_name, row.day) for row in day_rows}):
        await replace_day_session_summaries(
            conn,
            provider_name=provider_name,
            day=bucket_day,
            records=[row for row in day_rows if row.provider_name == provider_name and row.day == bucket_day],
            transaction_depth=transaction_depth,
        )
    return {
        "profiles": profile_count,
        "work_events": work_event_count,
        "phases": phase_count,
        "threads": len(thread_records),
        "tag_rollups": len(tag_rows),
        "day_summaries": len(day_rows),
    }


def _build_all_thread_records_sync(conn: sqlite3.Connection) -> list[object]:
    root_ids = [str(row["conversation_id"]) for row in conn.execute(_ROOT_THREAD_IDS_SQL).fetchall()]
    records: list[object] = []
    for root_id in root_ids:
        profile_records = _load_thread_profile_records_sync(conn, root_id)
        if not profile_records:
            continue
        profiles = [hydrate_session_profile(record) for record in profile_records]
        threads = build_session_threads(profiles)
        for thread in threads:
            if thread.thread_id == root_id:
                records.append(build_work_thread_record(thread))
                break
    return records


async def _build_all_thread_records_async(conn: aiosqlite.Connection) -> list[object]:
    rows = await (await conn.execute(_ROOT_THREAD_IDS_SQL)).fetchall()
    root_ids = [str(row["conversation_id"]) for row in rows]
    records: list[object] = []
    for root_id in root_ids:
        profile_records = await _load_thread_profile_records_async(conn, root_id)
        if not profile_records:
            continue
        profiles = [hydrate_session_profile(record) for record in profile_records]
        threads = build_session_threads(profiles)
        for thread in threads:
            if thread.thread_id == root_id:
                records.append(build_work_thread_record(thread))
                break
    return records


def _load_thread_profile_records_sync(conn: sqlite3.Connection, root_id: str) -> list[SessionProfileRecord]:
    conversation_ids = thread_conversation_ids_sync(conn, root_id)
    if not conversation_ids:
        return []
    placeholders = ", ".join("?" for _ in conversation_ids)
    rows = conn.execute(
        f"SELECT * FROM session_profiles WHERE conversation_id IN ({placeholders})",
        tuple(conversation_ids),
    ).fetchall()
    return [_row_to_session_profile_record(row) for row in rows]


async def _load_thread_profile_records_async(conn: aiosqlite.Connection, root_id: str) -> list[SessionProfileRecord]:
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


async def refresh_session_products_for_conversation_async(
    conn: aiosqlite.Connection,
    conversation_id: str,
    *,
    transaction_depth: int = 0,
) -> dict[str, int]:
    old_profile_record = await (
        await conn.execute(
            "SELECT * FROM session_profiles WHERE conversation_id = ?",
            (conversation_id,),
        )
    ).fetchone()
    conversations, messages, attachments, blocks = await _load_async_batch(conn, [conversation_id])
    hydrated = _hydrate_conversations(conversations, messages, attachments, blocks)
    if not hydrated:
        await conn.execute("DELETE FROM session_profiles WHERE conversation_id = ?", (conversation_id,))
        await replace_session_work_events(conn, conversation_id, [], transaction_depth)
        await replace_session_phases(conn, conversation_id, [], transaction_depth)
        old_group = _profile_provider_day(_row_to_session_profile_record(old_profile_record)) if old_profile_record else None
        if old_group is not None:
            await _refresh_async_provider_day_aggregates(conn, {old_group}, transaction_depth=transaction_depth)
        return {"profiles": 0, "work_events": 0, "phases": 0, "threads": 0, "tag_rollups": 0, "day_summaries": 0}

    profile_record, event_records, phase_records = build_session_product_records(hydrated[0])
    await replace_session_profile(conn, profile_record, transaction_depth)
    await replace_session_work_events(conn, conversation_id, event_records, transaction_depth)
    await replace_session_phases(conn, conversation_id, phase_records, transaction_depth)

    root_id = await thread_root_id_async(conn, conversation_id)
    thread_count = 0
    if root_id is not None:
        profile_records = await _load_thread_profile_records_async(conn, root_id)
        profiles = [hydrate_session_profile(record) for record in profile_records]
        threads = build_session_threads(profiles)
        record = next((build_work_thread_record(thread) for thread in threads if thread.thread_id == root_id), None)
        await replace_work_thread(conn, root_id, record, transaction_depth)
        thread_count = 1 if record is not None else 0

    affected_groups = {
        group
        for group in (
            _profile_provider_day(_row_to_session_profile_record(old_profile_record)) if old_profile_record else None,
            _profile_provider_day(profile_record),
        )
        if group is not None
    }
    await _refresh_async_provider_day_aggregates(conn, affected_groups, transaction_depth=transaction_depth)

    return {
        "profiles": 1,
        "work_events": len(event_records),
        "phases": len(phase_records),
        "threads": thread_count,
        "tag_rollups": len(affected_groups),
        "day_summaries": len(affected_groups),
    }


async def refresh_thread_after_conversation_delete_async(
    conn: aiosqlite.Connection,
    root_id: str | None,
    *,
    transaction_depth: int = 0,
) -> int:
    if root_id is None:
        return 0
    profile_records = await _load_thread_profile_records_async(conn, root_id)
    profiles = [hydrate_session_profile(record) for record in profile_records]
    threads = build_session_threads(profiles)
    record = next((build_work_thread_record(thread) for thread in threads if thread.thread_id == root_id), None)
    await replace_work_thread(conn, root_id, record, transaction_depth)
    return 1 if record is not None else 0


async def delete_session_products_for_conversation_async(
    conn: aiosqlite.Connection,
    conversation_id: str,
    *,
    transaction_depth: int = 0,
) -> dict[str, int]:
    cursor = await conn.execute(
        "SELECT * FROM session_profiles WHERE conversation_id = ?",
        (conversation_id,),
    )
    row = await cursor.fetchone()
    old_group = _profile_provider_day(_row_to_session_profile_record(row)) if row else None
    await conn.execute("DELETE FROM session_profiles WHERE conversation_id = ?", (conversation_id,))
    await replace_session_work_events(conn, conversation_id, [], transaction_depth)
    await replace_session_phases(conn, conversation_id, [], transaction_depth)
    if old_group is not None:
        provider_name, bucket_day = old_group
        await conn.execute(
            "DELETE FROM day_session_summaries WHERE provider_name = ? AND day = ?",
            (provider_name, bucket_day),
        )
        await conn.execute(
            "DELETE FROM session_tag_rollups WHERE provider_name = ? AND bucket_day = ?",
            (provider_name, bucket_day),
        )
        await _refresh_async_provider_day_aggregates(conn, {old_group}, transaction_depth=transaction_depth)
    return {
        "profiles": 1 if row is not None else 0,
        "work_events": 0,
        "phases": 0,
        "threads": 0,
        "tag_rollups": 1 if old_group is not None else 0,
        "day_summaries": 1 if old_group is not None else 0,
    }


__all__ = [
    "delete_session_products_for_conversation_async",
    "rebuild_session_products_async",
    "rebuild_session_products_sync",
    "refresh_session_products_for_conversation_async",
    "refresh_thread_after_conversation_delete_async",
    "session_product_status_async",
    "session_product_status_sync",
    "session_profile_repair_candidate_ids_async",
    "session_profile_repair_candidate_ids_sync",
    "thread_conversation_ids_async",
    "thread_conversation_ids_sync",
    "thread_root_id_async",
    "thread_root_id_sync",
]
