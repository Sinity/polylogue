"""Shared support primitives for durable session-product lifecycle flows."""

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
    replace_session_tag_rollup_rows,
)
from polylogue.storage.hydrators import conversation_from_records
from polylogue.storage.session_product_rows import (
    build_session_product_records,
    build_work_thread_record,
    hydrate_session_profile,
)
from polylogue.storage.store import (
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

_PROFILE_BUCKET_DAY_SQL = (
    "COALESCE(sp.canonical_session_date, "
    "date(COALESCE(sp.first_message_at, json_extract(sp.payload_json, '$.created_at'), sp.source_updated_at, sp.last_message_at)))"
)
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


def chunked(items: Sequence[str], *, size: int) -> Iterable[Sequence[str]]:
    for index in range(0, len(items), size):
        yield items[index : index + size]


def sync_attachment_batch(
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


def load_sync_batch(
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
    return conversations, messages, sync_attachment_batch(conn, conversation_ids), blocks


async def load_async_batch(
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


def hydrate_conversations(
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


def replace_session_profile_sync(conn: sqlite3.Connection, record: SessionProfileRecord) -> None:
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


def replace_session_work_events_sync(
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


def replace_session_phases_sync(
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


def replace_work_thread_sync(conn: sqlite3.Connection, thread_id: str, record: object | None) -> None:
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


def replace_session_tag_rollup_rows_sync(
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


def replace_day_session_summaries_sync(
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


def load_sync_provider_day_profile_records(
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


async def load_async_provider_day_profile_records(
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


def refresh_sync_provider_day_aggregates(
    conn: sqlite3.Connection,
    groups: set[tuple[str, str]],
) -> None:
    for provider_name, bucket_day in groups:
        profile_records = load_sync_provider_day_profile_records(
            conn,
            provider_name=provider_name,
            bucket_day=bucket_day,
        )
        profiles = [hydrate_session_profile(record) for record in profile_records]
        day_rows = build_day_session_summary_records(profiles) if profiles else []
        tag_rows = build_session_tag_rollup_records(profiles) if profiles else []
        replace_day_session_summaries_sync(
            conn,
            provider_name=provider_name,
            day=bucket_day,
            records=[row for row in day_rows if row.provider_name == provider_name and row.day == bucket_day],
        )
        replace_session_tag_rollup_rows_sync(
            conn,
            provider_name=provider_name,
            bucket_day=bucket_day,
            records=[
                row
                for row in tag_rows
                if row.provider_name == provider_name and row.bucket_day == bucket_day
            ],
        )


async def refresh_async_provider_day_aggregates(
    conn: aiosqlite.Connection,
    groups: set[tuple[str, str]],
    *,
    transaction_depth: int,
) -> None:
    for provider_name, bucket_day in groups:
        profile_records = await load_async_provider_day_profile_records(
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


def profile_provider_day(record: SessionProfileRecord | None) -> tuple[str, str] | None:
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


def load_thread_profile_records_sync(conn: sqlite3.Connection, root_id: str) -> list[SessionProfileRecord]:
    conversation_ids = thread_conversation_ids_sync(conn, root_id)
    if not conversation_ids:
        return []
    placeholders = ", ".join("?" for _ in conversation_ids)
    rows = conn.execute(
        f"SELECT * FROM session_profiles WHERE conversation_id IN ({placeholders})",
        tuple(conversation_ids),
    ).fetchall()
    return [_row_to_session_profile_record(row) for row in rows]


async def load_thread_profile_records_async(conn: aiosqlite.Connection, root_id: str) -> list[SessionProfileRecord]:
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
    "_ALL_CONVERSATION_IDS_SQL",
    "_PROFILE_BUCKET_DAY_SQL",
    "_THREAD_ROOT_ID_SQL",
    "build_all_thread_records_async",
    "build_all_thread_records_sync",
    "build_session_product_records",
    "chunked",
    "hydrate_conversations",
    "hydrate_session_profile",
    "load_async_batch",
    "load_async_provider_day_profile_records",
    "load_sync_batch",
    "load_sync_provider_day_profile_records",
    "load_thread_profile_records_async",
    "load_thread_profile_records_sync",
    "profile_provider_day",
    "refresh_async_provider_day_aggregates",
    "refresh_sync_provider_day_aggregates",
    "replace_day_session_summaries_sync",
    "replace_session_phases_sync",
    "replace_session_profile_sync",
    "replace_session_tag_rollup_rows_sync",
    "replace_session_work_events_sync",
    "replace_work_thread_sync",
    "thread_conversation_ids_async",
    "thread_conversation_ids_sync",
    "thread_root_id_async",
    "thread_root_id_sync",
]
