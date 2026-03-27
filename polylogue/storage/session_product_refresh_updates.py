"""Update-side upkeep for durable session products."""

from __future__ import annotations

import aiosqlite

from polylogue.storage.backends.queries.mappers import _row_to_session_profile_record
from polylogue.storage.session_product_aggregates import (
    profile_provider_day,
    refresh_async_provider_day_aggregates,
)
from polylogue.storage.session_product_batches import hydrate_conversations, load_async_batch
from polylogue.storage.session_product_profiles import hydrate_session_profile
from polylogue.storage.session_product_rebuild import build_session_product_records
from polylogue.storage.session_product_threads import (
    load_thread_profile_records_async,
    thread_root_id_async,
)


async def refresh_session_products_for_conversation_async(
    conn: aiosqlite.Connection,
    conversation_id: str,
    *,
    transaction_depth: int = 0,
) -> dict[str, int]:
    from polylogue.lib.threads import build_session_threads
    from polylogue.storage.backends.queries.session_product_profile_writes import (
        replace_session_profile,
    )
    from polylogue.storage.backends.queries.session_product_thread_queries import (
        replace_work_thread,
    )
    from polylogue.storage.backends.queries.session_product_timeline_writes import (
        replace_session_phases,
        replace_session_work_events,
    )
    from polylogue.storage.session_product_threads import build_work_thread_record

    old_profile_record = await (
        await conn.execute(
            "SELECT * FROM session_profiles WHERE conversation_id = ?",
            (conversation_id,),
        )
    ).fetchone()
    conversations, messages, attachments, blocks = await load_async_batch(conn, [conversation_id])
    hydrated = hydrate_conversations(conversations, messages, attachments, blocks)
    if not hydrated:
        await conn.execute("DELETE FROM session_profiles WHERE conversation_id = ?", (conversation_id,))
        await replace_session_work_events(conn, conversation_id, [], transaction_depth)
        await replace_session_phases(conn, conversation_id, [], transaction_depth)
        old_group = (
            profile_provider_day(_row_to_session_profile_record(old_profile_record))
            if old_profile_record
            else None
        )
        if old_group is not None:
            await refresh_async_provider_day_aggregates(
                conn,
                {old_group},
                transaction_depth=transaction_depth,
            )
        return {
            "profiles": 0,
            "work_events": 0,
            "phases": 0,
            "threads": 0,
            "tag_rollups": 0,
            "day_summaries": 0,
        }

    profile_record, event_records, phase_records = build_session_product_records(hydrated[0])
    await replace_session_profile(conn, profile_record, transaction_depth)
    await replace_session_work_events(conn, conversation_id, event_records, transaction_depth)
    await replace_session_phases(conn, conversation_id, phase_records, transaction_depth)

    root_id = await thread_root_id_async(conn, conversation_id)
    thread_count = 0
    if root_id is not None:
        profile_records = await load_thread_profile_records_async(conn, root_id)
        profiles = [hydrate_session_profile(record) for record in profile_records]
        threads = build_session_threads(profiles)
        record = next(
            (build_work_thread_record(thread) for thread in threads if thread.thread_id == root_id),
            None,
        )
        await replace_work_thread(conn, root_id, record, transaction_depth)
        thread_count = 1 if record is not None else 0

    affected_groups = {
        group
        for group in (
            profile_provider_day(_row_to_session_profile_record(old_profile_record))
            if old_profile_record
            else None,
            profile_provider_day(profile_record),
        )
        if group is not None
    }
    await refresh_async_provider_day_aggregates(
        conn,
        affected_groups,
        transaction_depth=transaction_depth,
    )

    return {
        "profiles": 1,
        "work_events": len(event_records),
        "phases": len(phase_records),
        "threads": thread_count,
        "tag_rollups": len(affected_groups),
        "day_summaries": len(affected_groups),
    }


__all__ = ["refresh_session_products_for_conversation_async"]
