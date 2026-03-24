"""Full rebuild flows for durable session-product read models."""

from __future__ import annotations

import sqlite3
from collections.abc import Sequence

import aiosqlite

from polylogue.archive_product_builders import (
    build_day_session_summary_records,
    build_session_tag_rollup_records,
)
from polylogue.storage.backends.queries.mappers import _row_to_session_profile_record

from .session_product_support import (
    _ALL_CONVERSATION_IDS_SQL,
    build_all_thread_records_async,
    build_all_thread_records_sync,
    build_session_product_records,
    chunked,
    hydrate_conversations,
    hydrate_session_profile,
    load_async_batch,
    load_sync_batch,
    replace_day_session_summaries_sync,
    replace_session_phases_sync,
    replace_session_profile_sync,
    replace_session_tag_rollup_rows_sync,
    replace_session_work_events_sync,
    replace_work_thread_sync,
)


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
    for chunk in chunked(list(conversation_ids), size=page_size):
        conversations, messages, attachments, blocks = load_sync_batch(conn, chunk)
        for conversation in hydrate_conversations(conversations, messages, attachments, blocks):
            profile_record, event_records, phase_records = build_session_product_records(conversation)
            replace_session_profile_sync(conn, profile_record)
            replace_session_work_events_sync(conn, profile_record.conversation_id, event_records)
            replace_session_phases_sync(conn, profile_record.conversation_id, phase_records)
            profile_count += 1
            work_event_count += len(event_records)
            phase_count += len(phase_records)

    conn.execute("DELETE FROM work_threads")
    thread_records = build_all_thread_records_sync(conn)
    for record in thread_records:
        replace_work_thread_sync(conn, record.thread_id, record)
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
    day_rows = build_day_session_summary_records(all_profiles)
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
    from polylogue.storage.backends.queries.session_product_profile_queries import replace_session_profile
    from polylogue.storage.backends.queries.session_product_summary_queries import (
        replace_day_session_summaries,
        replace_session_tag_rollup_rows,
    )
    from polylogue.storage.backends.queries.session_product_thread_queries import replace_work_thread
    from polylogue.storage.backends.queries.session_product_timeline_queries import (
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
    "rebuild_session_products_async",
    "rebuild_session_products_sync",
]
