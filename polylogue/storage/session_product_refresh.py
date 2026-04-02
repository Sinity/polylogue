"""Delete-side and update-side upkeep for durable session products."""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass

import aiosqlite

from polylogue.storage.backends.queries.mappers import _row_to_session_profile_record
from polylogue.storage.session_product_aggregates import (
    profile_provider_day,
    refresh_async_provider_day_aggregates,
)
from polylogue.storage.session_product_profiles import hydrate_session_profile
from polylogue.storage.session_product_rebuild import (
    build_session_product_records,
    chunked,
    hydrate_conversations,
    load_async_batch,
)
from polylogue.storage.session_product_threads import (
    load_thread_profile_records_async,
    thread_root_id_async,
    thread_root_ids_async,
)


@dataclass(slots=True)
class _SessionProductRefreshUpdate:
    counts: dict[str, int]
    thread_root_id: str | None
    affected_groups: set[tuple[str, str]]


@dataclass(slots=True)
class _SessionProductBulkRefreshUpdate:
    counts: dict[str, int]
    thread_root_ids: set[str]
    affected_groups: set[tuple[str, str]]
    chunk_observations: list[dict[str, object]]


def _empty_refresh_counts() -> dict[str, int]:
    return {
        "profiles": 0,
        "work_events": 0,
        "phases": 0,
        "threads": 0,
        "tag_rollups": 0,
        "day_summaries": 0,
    }


async def _refresh_thread_root_async(
    conn: aiosqlite.Connection,
    root_id: str | None,
    *,
    transaction_depth: int,
) -> int:
    from polylogue.lib.threads import build_session_threads
    from polylogue.storage.backends.queries.session_product_thread_queries import (
        replace_work_thread,
    )
    from polylogue.storage.session_product_threads import build_work_thread_record

    if root_id is None:
        return 0

    profile_records = await load_thread_profile_records_async(conn, root_id)
    profiles = [hydrate_session_profile(record) for record in profile_records]
    threads = build_session_threads(profiles)
    record = next(
        (build_work_thread_record(thread) for thread in threads if thread.thread_id == root_id),
        None,
    )
    await replace_work_thread(conn, root_id, record, transaction_depth)
    return 1 if record is not None else 0


async def refresh_thread_after_conversation_delete_async(
    conn: aiosqlite.Connection,
    root_id: str | None,
    *,
    transaction_depth: int = 0,
) -> int:
    return await _refresh_thread_root_async(
        conn,
        root_id,
        transaction_depth=transaction_depth,
    )


async def delete_session_products_for_conversation_async(
    conn: aiosqlite.Connection,
    conversation_id: str,
    *,
    transaction_depth: int = 0,
) -> dict[str, int]:
    from polylogue.storage.backends.queries.session_product_timeline_writes import (
        replace_session_phases,
        replace_session_work_events,
    )

    cursor = await conn.execute(
        "SELECT * FROM session_profiles WHERE conversation_id = ?",
        (conversation_id,),
    )
    row = await cursor.fetchone()
    old_group = profile_provider_day(_row_to_session_profile_record(row)) if row else None
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
        await refresh_async_provider_day_aggregates(
            conn,
            {old_group},
            transaction_depth=transaction_depth,
        )
    return {
        "profiles": 1 if row is not None else 0,
        "work_events": 0,
        "phases": 0,
        "threads": 0,
        "tag_rollups": 1 if old_group is not None else 0,
        "day_summaries": 1 if old_group is not None else 0,
    }


async def refresh_session_products_for_conversation_async(
    conn: aiosqlite.Connection,
    conversation_id: str,
    *,
    transaction_depth: int = 0,
) -> dict[str, int]:
    update = await _apply_session_product_conversation_update_async(
        conn,
        conversation_id,
        transaction_depth=transaction_depth,
    )
    thread_count = await _refresh_thread_root_async(
        conn,
        update.thread_root_id,
        transaction_depth=transaction_depth,
    )
    await refresh_async_provider_day_aggregates(
        conn,
        update.affected_groups,
        transaction_depth=transaction_depth,
    )
    result = dict(update.counts)
    result["threads"] = thread_count
    result["tag_rollups"] = len(update.affected_groups)
    result["day_summaries"] = len(update.affected_groups)
    return result


async def _apply_session_product_conversation_update_async(
    conn: aiosqlite.Connection,
    conversation_id: str,
    *,
    transaction_depth: int,
) -> _SessionProductRefreshUpdate:
    from polylogue.storage.backends.queries.session_product_profile_writes import (
        replace_session_profile,
    )
    from polylogue.storage.backends.queries.session_product_timeline_writes import (
        replace_session_phases,
        replace_session_work_events,
    )

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
        return _SessionProductRefreshUpdate(
            counts=_empty_refresh_counts(),
            thread_root_id=None,
            affected_groups={old_group} if old_group is not None else set(),
        )

    profile_record, event_records, phase_records = build_session_product_records(hydrated[0])
    await replace_session_profile(conn, profile_record, transaction_depth)
    await replace_session_work_events(conn, conversation_id, event_records, transaction_depth)
    await replace_session_phases(conn, conversation_id, phase_records, transaction_depth)

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
    return _SessionProductRefreshUpdate(
        counts={
            "profiles": 1,
            "work_events": len(event_records),
            "phases": len(phase_records),
            "threads": 0,
            "tag_rollups": 0,
            "day_summaries": 0,
        },
        thread_root_id=await thread_root_id_async(conn, conversation_id),
        affected_groups=affected_groups,
    )


async def _load_existing_session_profile_records_async(
    conn: aiosqlite.Connection,
    conversation_ids: Sequence[str],
) -> dict[str, object]:
    if not conversation_ids:
        return {}
    placeholders = ", ".join("?" for _ in conversation_ids)
    rows = await (
        await conn.execute(
            f"SELECT * FROM session_profiles WHERE conversation_id IN ({placeholders})",
            tuple(conversation_ids),
        )
    ).fetchall()
    return {
        str(row["conversation_id"]): _row_to_session_profile_record(row)
        for row in rows
    }


async def _apply_session_product_conversation_updates_async(
    conn: aiosqlite.Connection,
    conversation_ids: Sequence[str],
    *,
    transaction_depth: int,
    page_size: int = 100,
) -> _SessionProductBulkRefreshUpdate:
    from polylogue.storage.backends.queries.session_product_profile_writes import (
        replace_session_profiles_bulk,
    )
    from polylogue.storage.backends.queries.session_product_timeline_writes import (
        replace_session_phases_bulk,
        replace_session_work_events_bulk,
    )

    counts = _empty_refresh_counts()
    thread_root_ids: set[str] = set()
    affected_groups: set[tuple[str, str]] = set()
    chunk_observations: list[dict[str, object]] = []
    conversation_id_list = list(conversation_ids)

    for chunk in chunked(conversation_id_list, size=page_size):
        chunk_started = time.perf_counter()
        load_started = time.perf_counter()
        old_profile_records = await _load_existing_session_profile_records_async(conn, chunk)
        conversations, messages, attachments, blocks = await load_async_batch(conn, chunk)
        root_ids_by_conversation = await thread_root_ids_async(conn, chunk)
        load_elapsed_ms = round((time.perf_counter() - load_started) * 1000.0, 1)
        profile_records_to_write: list[object] = []
        work_event_records_to_write: list[object] = []
        phase_records_to_write: list[object] = []
        hydrate_started = time.perf_counter()
        hydrated_by_id = {
            str(conversation.id): conversation
            for conversation in hydrate_conversations(conversations, messages, attachments, blocks)
        }
        hydrate_elapsed_ms = round((time.perf_counter() - hydrate_started) * 1000.0, 1)

        build_started = time.perf_counter()
        for conversation_id in chunk:
            old_profile_record = old_profile_records.get(conversation_id)
            hydrated_conversation = hydrated_by_id.get(conversation_id)
            if hydrated_conversation is None:
                old_group = profile_provider_day(old_profile_record)
                if old_group is not None:
                    affected_groups.add(old_group)
                continue

            profile_record, event_records, phase_records = build_session_product_records(
                hydrated_conversation
            )
            profile_records_to_write.append(profile_record)
            work_event_records_to_write.extend(event_records)
            phase_records_to_write.extend(phase_records)

            counts["profiles"] += 1
            counts["work_events"] += len(event_records)
            counts["phases"] += len(phase_records)
            affected_groups.update(
                group
                for group in (
                    profile_provider_day(old_profile_record),
                    profile_provider_day(profile_record),
                )
                if group is not None
            )
            root_id = root_ids_by_conversation.get(conversation_id)
            if root_id is not None:
                thread_root_ids.add(root_id)
        build_elapsed_ms = round((time.perf_counter() - build_started) * 1000.0, 1)

        write_started = time.perf_counter()
        await replace_session_profiles_bulk(
            conn,
            chunk,
            profile_records_to_write,
            transaction_depth,
        )
        await replace_session_work_events_bulk(
            conn,
            chunk,
            work_event_records_to_write,
            transaction_depth,
        )
        await replace_session_phases_bulk(
            conn,
            chunk,
            phase_records_to_write,
            transaction_depth,
        )
        write_elapsed_ms = round((time.perf_counter() - write_started) * 1000.0, 1)
        chunk_observation = {
            "conversation_count": len(chunk),
            "hydrated_count": len(hydrated_by_id),
            "profiles_written": len(profile_records_to_write),
            "work_events_written": len(work_event_records_to_write),
            "phases_written": len(phase_records_to_write),
            "load_ms": load_elapsed_ms,
            "hydrate_ms": hydrate_elapsed_ms,
            "build_ms": build_elapsed_ms,
            "write_ms": write_elapsed_ms,
            "total_ms": round((time.perf_counter() - chunk_started) * 1000.0, 1),
        }
        if chunk_observation["total_ms"] >= 500.0:
            chunk_observation["slow"] = True
        chunk_observations.append(chunk_observation)

    return _SessionProductBulkRefreshUpdate(
        counts=counts,
        thread_root_ids=thread_root_ids,
        affected_groups=affected_groups,
        chunk_observations=chunk_observations,
    )


__all__ = [
    "delete_session_products_for_conversation_async",
    "refresh_session_products_for_conversation_async",
    "refresh_thread_after_conversation_delete_async",
]
