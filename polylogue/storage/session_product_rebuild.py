"""Batch loading, hydration, and full rebuild flows for durable session-product read models."""

from __future__ import annotations

import sqlite3
from collections import defaultdict
from collections.abc import Iterable, Sequence

import aiosqlite

from polylogue.archive_product_rollups import build_session_tag_rollup_records
from polylogue.archive_product_summaries import build_day_session_summary_records
from polylogue.lib.session_profile import build_session_analysis, build_session_profile
from polylogue.storage.action_event_rows import attach_blocks_to_messages
from polylogue.storage.backends.queries.attachments import get_attachments_batch
from polylogue.storage.backends.queries.mappers import (
    _row_to_content_block,
    _row_to_conversation,
    _row_to_message,
    _row_to_session_profile_record,
)
from polylogue.storage.hydrators import conversation_from_records
from polylogue.storage.session_product_profiles import (
    build_session_profile_record,
    hydrate_session_profile,
    now_iso,
)
from polylogue.storage.session_product_storage import (
    replace_day_session_summaries_sync,
    replace_session_phases_sync,
    replace_session_profile_sync,
    replace_session_tag_rollup_rows_sync,
    replace_session_work_events_sync,
    replace_work_thread_sync,
)
from polylogue.storage.session_product_threads import (
    build_all_thread_records_async,
    build_all_thread_records_sync,
)
from polylogue.storage.session_product_timeline_rows import (
    build_session_phase_records,
    build_session_work_event_records,
)
from polylogue.storage.store import (
    AttachmentRecord,
    ContentBlockRecord,
    ConversationRecord,
    MessageRecord,
)

_ALL_CONVERSATION_IDS_SQL = "SELECT conversation_id FROM conversations ORDER BY COALESCE(sort_key, 0) DESC, conversation_id"
_ALL_SESSION_PROFILE_ROWS_SQL = """
SELECT *
FROM session_profiles
ORDER BY COALESCE(source_sort_key, 0) DESC, conversation_id
"""


def chunked(items: Sequence[str], *, size: int) -> Iterable[Sequence[str]]:
    for index in range(0, len(items), size):
        yield items[index : index + size]


def iter_conversation_id_pages_sync(
    conn: sqlite3.Connection,
    *,
    page_size: int,
) -> Iterable[list[str]]:
    cursor = conn.execute(_ALL_CONVERSATION_IDS_SQL)
    while True:
        rows = cursor.fetchmany(page_size)
        if not rows:
            break
        yield [str(row["conversation_id"]) for row in rows]


def iter_hydrated_session_profiles_sync(
    conn: sqlite3.Connection,
    *,
    page_size: int,
):
    cursor = conn.execute(_ALL_SESSION_PROFILE_ROWS_SQL)
    while True:
        rows = cursor.fetchmany(page_size)
        if not rows:
            break
        for row in rows:
            yield hydrate_session_profile(_row_to_session_profile_record(row))


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


def build_session_product_records(
    conversation,
) -> tuple[object, list[object], list[object]]:
    analysis = build_session_analysis(conversation)
    profile = build_session_profile(conversation, analysis=analysis)
    materialized_at = now_iso()
    return (
        build_session_profile_record(profile, analysis=analysis, materialized_at=materialized_at),
        build_session_work_event_records(profile, materialized_at=materialized_at),
        build_session_phase_records(profile, materialized_at=materialized_at),
    )


def rebuild_session_products_sync(
    conn: sqlite3.Connection,
    *,
    conversation_ids: Sequence[str] | None = None,
    page_size: int = 100,
) -> dict[str, int]:
    conversation_chunks: Iterable[Sequence[str]]
    if conversation_ids is None:
        conn.execute("DELETE FROM session_work_events")
        conn.execute("DELETE FROM session_phases")
        conn.execute("DELETE FROM session_profiles")
        conn.execute("DELETE FROM session_tag_rollups")
        conn.execute("DELETE FROM day_session_summaries")
        conversation_chunks = iter_conversation_id_pages_sync(conn, page_size=page_size)
    else:
        conversation_chunks = chunked(conversation_ids, size=page_size)
    if conversation_ids is not None and not conversation_ids:
        conn.execute("DELETE FROM work_threads")
        conn.execute("DELETE FROM session_phases")
        conn.execute("DELETE FROM session_tag_rollups")
        conn.execute("DELETE FROM day_session_summaries")
        conn.commit()
        return {"profiles": 0, "work_events": 0, "phases": 0, "threads": 0, "tag_rollups": 0, "day_summaries": 0}

    profile_count = 0
    work_event_count = 0
    phase_count = 0
    saw_conversation_ids = False
    for chunk in conversation_chunks:
        saw_conversation_ids = True
        conversations, messages, attachments, blocks = load_sync_batch(conn, chunk)
        for conversation in hydrate_conversations(conversations, messages, attachments, blocks):
            profile_record, event_records, phase_records = build_session_product_records(conversation)
            replace_session_profile_sync(conn, profile_record)
            replace_session_work_events_sync(conn, profile_record.conversation_id, event_records)
            replace_session_phases_sync(conn, profile_record.conversation_id, phase_records)
            profile_count += 1
            work_event_count += len(event_records)
            phase_count += len(phase_records)
    if not saw_conversation_ids:
        conn.execute("DELETE FROM work_threads")
        conn.execute("DELETE FROM session_phases")
        conn.execute("DELETE FROM session_tag_rollups")
        conn.execute("DELETE FROM day_session_summaries")
        conn.commit()
        return {"profiles": 0, "work_events": 0, "phases": 0, "threads": 0, "tag_rollups": 0, "day_summaries": 0}

    conn.execute("DELETE FROM work_threads")
    thread_records = build_all_thread_records_sync(conn)
    for record in thread_records:
        replace_work_thread_sync(conn, record.thread_id, record)
    conn.execute("DELETE FROM session_tag_rollups")
    tag_rows = build_session_tag_rollup_records(iter_hydrated_session_profiles_sync(conn, page_size=page_size))
    for provider_name, bucket_day in sorted({(row.provider_name, row.bucket_day) for row in tag_rows}):
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
    conn.execute("DELETE FROM day_session_summaries")
    day_rows = build_day_session_summary_records(iter_hydrated_session_profiles_sync(conn, page_size=page_size))
    for provider_name, bucket_day in sorted({(row.provider_name, row.day) for row in day_rows}):
        replace_day_session_summaries_sync(
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
    from polylogue.storage.backends.queries.session_product_profile_writes import (
        replace_session_profile,
    )
    from polylogue.storage.backends.queries.session_product_summary_queries import (
        replace_day_session_summaries,
        replace_session_tag_rollup_rows,
    )
    from polylogue.storage.backends.queries.session_product_thread_queries import replace_work_thread
    from polylogue.storage.backends.queries.session_product_timeline_writes import (
        replace_session_phases,
        replace_session_work_events,
    )

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
    for chunk in chunked(list(conversation_ids), size=page_size):
        conversations, messages, attachments, blocks = await load_async_batch(conn, chunk)
        for conversation in hydrate_conversations(conversations, messages, attachments, blocks):
            profile_record, event_records, phase_records = build_session_product_records(conversation)
            await replace_session_profile(conn, profile_record, transaction_depth)
            await replace_session_work_events(conn, profile_record.conversation_id, event_records, transaction_depth)
            await replace_session_phases(conn, profile_record.conversation_id, phase_records, transaction_depth)
            profile_count += 1
            work_event_count += len(event_records)
            phase_count += len(phase_records)

    await conn.execute("DELETE FROM work_threads")
    thread_records = await build_all_thread_records_async(conn)
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


__all__ = [
    "_ALL_CONVERSATION_IDS_SQL",
    "_ALL_SESSION_PROFILE_ROWS_SQL",
    "build_session_product_records",
    "chunked",
    "hydrate_conversations",
    "iter_conversation_id_pages_sync",
    "iter_hydrated_session_profiles_sync",
    "load_async_batch",
    "load_sync_batch",
    "rebuild_session_products_async",
    "rebuild_session_products_sync",
    "sync_attachment_batch",
]
