"""Delete-side and update-side upkeep for durable session insights."""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass

import aiosqlite

from polylogue.storage.backends.queries.mappers import _row_to_session_profile_record
from polylogue.storage.insights.session.aggregates import (
    profile_provider_day,
    refresh_async_provider_day_aggregates,
)
from polylogue.storage.insights.session.rebuild import (
    SessionInsightRecordBundle,
    build_session_product_records,
    hydrate_conversations,
    load_async_batch,
)
from polylogue.storage.insights.session.runtime import (
    ProviderDayGroup,
    SessionInsightCounts,
    SessionInsightRefreshChunkPayload,
)
from polylogue.storage.insights.session.threads import (
    build_thread_records_for_roots_async,
    thread_root_id_async,
    thread_root_ids_async,
)
from polylogue.storage.runtime import SessionPhaseRecord, SessionProfileRecord, SessionWorkEventRecord

# Keep incremental refreshes on the same bounded chunk size as full rebuilds.
# Hydrating 100 conversations at once inflates RSS badly on pathological archives.
_SESSION_INSIGHT_REFRESH_PAGE_SIZE = 10
_SESSION_INSIGHT_REFRESH_MESSAGE_BUDGET = 5_000


@dataclass(slots=True)
class _SessionInsightRefreshUpdate:
    counts: SessionInsightCounts
    thread_root_id: str | None
    affected_groups: set[ProviderDayGroup]


@dataclass(slots=True)
class _SessionInsightBulkRefreshUpdate:
    counts: SessionInsightCounts
    thread_root_ids: set[str]
    affected_groups: set[ProviderDayGroup]
    chunk_observations: list[SessionInsightRefreshChunkObservation]


@dataclass(slots=True, frozen=True)
class _SessionInsightRefreshChunk:
    conversation_ids: tuple[str, ...]
    estimated_message_count: int
    max_estimated_conversation_messages: int


@dataclass(slots=True, frozen=True)
class SessionInsightRefreshChunkObservation:
    conversation_count: int
    estimated_message_count: int
    max_estimated_conversation_messages: int
    hydrated_count: int
    profiles_written: int
    work_events_written: int
    phases_written: int
    load_ms: float
    hydrate_ms: float
    build_ms: float
    write_ms: float
    total_ms: float
    slow: bool = False

    def to_observation(self) -> SessionInsightRefreshChunkPayload:
        observation: SessionInsightRefreshChunkPayload = {
            "conversation_count": self.conversation_count,
            "estimated_message_count": self.estimated_message_count,
            "max_estimated_conversation_messages": self.max_estimated_conversation_messages,
            "hydrated_count": self.hydrated_count,
            "profiles_written": self.profiles_written,
            "work_events_written": self.work_events_written,
            "phases_written": self.phases_written,
            "load_ms": self.load_ms,
            "hydrate_ms": self.hydrate_ms,
            "build_ms": self.build_ms,
            "write_ms": self.write_ms,
            "total_ms": self.total_ms,
            "slow": self.slow,
        }
        return observation


def _empty_refresh_counts() -> SessionInsightCounts:
    return SessionInsightCounts()


async def _refresh_thread_root_async(
    conn: aiosqlite.Connection,
    root_id: str | None,
    *,
    transaction_depth: int,
) -> int:
    return await _refresh_thread_roots_async(
        conn,
        [root_id] if root_id is not None else [],
        transaction_depth=transaction_depth,
    )


async def _refresh_thread_roots_async(
    conn: aiosqlite.Connection,
    root_ids: Sequence[str],
    *,
    transaction_depth: int,
) -> int:
    from polylogue.storage.backends.queries.session_insight_thread_queries import (
        replace_work_thread,
    )

    normalized_root_ids = tuple(dict.fromkeys(str(root_id) for root_id in root_ids if str(root_id)))
    if not normalized_root_ids:
        return 0

    thread_records = await build_thread_records_for_roots_async(conn, normalized_root_ids)
    refreshed = 0
    for root_id in normalized_root_ids:
        record = thread_records.get(root_id)
        await replace_work_thread(conn, root_id, record, transaction_depth)
        if record is not None:
            refreshed += 1
    return refreshed


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
) -> SessionInsightCounts:
    from polylogue.storage.backends.queries.session_insight_timeline_writes import (
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
    counts = _empty_refresh_counts()
    counts.add(
        profiles=1 if row is not None else 0,
        tag_rollups=1 if old_group is not None else 0,
        day_summaries=1 if old_group is not None else 0,
    )
    return counts


async def refresh_session_products_for_conversation_async(
    conn: aiosqlite.Connection,
    conversation_id: str,
    *,
    transaction_depth: int = 0,
) -> SessionInsightCounts:
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
    update.counts.add(
        threads=thread_count,
        tag_rollups=len(update.affected_groups),
        day_summaries=len(update.affected_groups),
    )
    return update.counts


async def _apply_session_product_conversation_update_async(
    conn: aiosqlite.Connection,
    conversation_id: str,
    *,
    transaction_depth: int,
) -> _SessionInsightRefreshUpdate:
    from polylogue.storage.backends.queries.session_insight_profile_writes import (
        replace_session_profile,
    )
    from polylogue.storage.backends.queries.session_insight_timeline_writes import (
        replace_session_phases,
        replace_session_work_events,
    )

    old_profile_record = await (
        await conn.execute(
            "SELECT * FROM session_profiles WHERE conversation_id = ?",
            (conversation_id,),
        )
    ).fetchone()
    batch = await load_async_batch(conn, [conversation_id])
    hydrated = hydrate_conversations(batch)
    if not hydrated:
        await conn.execute("DELETE FROM session_profiles WHERE conversation_id = ?", (conversation_id,))
        await replace_session_work_events(conn, conversation_id, [], transaction_depth)
        await replace_session_phases(conn, conversation_id, [], transaction_depth)
        old_group = (
            profile_provider_day(_row_to_session_profile_record(old_profile_record)) if old_profile_record else None
        )
        return _SessionInsightRefreshUpdate(
            counts=_empty_refresh_counts(),
            thread_root_id=None,
            affected_groups={old_group} if old_group is not None else set(),
        )

    record_bundle = build_session_product_records(hydrated[0])
    await replace_session_profile(conn, record_bundle.profile_record, transaction_depth)
    await replace_session_work_events(
        conn,
        conversation_id,
        record_bundle.work_event_records,
        transaction_depth,
    )
    await replace_session_phases(
        conn,
        conversation_id,
        record_bundle.phase_records,
        transaction_depth,
    )

    affected_groups = {
        group
        for group in (
            profile_provider_day(_row_to_session_profile_record(old_profile_record)) if old_profile_record else None,
            profile_provider_day(record_bundle.profile_record),
        )
        if group is not None
    }
    return _SessionInsightRefreshUpdate(
        counts=SessionInsightCounts(
            profiles=1,
            work_events=record_bundle.work_event_count,
            phases=record_bundle.phase_count,
        ),
        thread_root_id=await thread_root_id_async(conn, conversation_id),
        affected_groups=affected_groups,
    )


async def _load_existing_session_profile_records_async(
    conn: aiosqlite.Connection,
    conversation_ids: Sequence[str],
) -> dict[str, SessionProfileRecord]:
    if not conversation_ids:
        return {}
    placeholders = ", ".join("?" for _ in conversation_ids)
    rows = await (
        await conn.execute(
            f"SELECT * FROM session_profiles WHERE conversation_id IN ({placeholders})",
            tuple(conversation_ids),
        )
    ).fetchall()
    return {str(row["conversation_id"]): _row_to_session_profile_record(row) for row in rows}


async def _load_message_counts_async(
    conn: aiosqlite.Connection,
    conversation_ids: Sequence[str],
) -> dict[str, int]:
    if not conversation_ids:
        return {}
    placeholders = ", ".join("?" for _ in conversation_ids)
    rows = await (
        await conn.execute(
            f"""
            SELECT conversation_id, message_count
            FROM conversation_stats
            WHERE conversation_id IN ({placeholders})
            """,
            tuple(conversation_ids),
        )
    ).fetchall()
    return {str(row["conversation_id"]): int(row["message_count"] or 0) for row in rows}


def _chunk_conversation_ids_by_message_budget(
    conversation_ids: Sequence[str],
    *,
    message_counts: dict[str, int],
    page_size: int,
    message_budget: int,
) -> list[_SessionInsightRefreshChunk]:
    chunks: list[_SessionInsightRefreshChunk] = []
    current_chunk: list[str] = []
    current_messages = 0
    current_max_messages = 0

    for conversation_id in conversation_ids:
        estimated_messages = max(int(message_counts.get(conversation_id, 0) or 0), 1)
        if current_chunk and (
            len(current_chunk) >= page_size or current_messages + estimated_messages > message_budget
        ):
            chunks.append(
                _SessionInsightRefreshChunk(
                    conversation_ids=tuple(current_chunk),
                    estimated_message_count=current_messages,
                    max_estimated_conversation_messages=current_max_messages,
                )
            )
            current_chunk = []
            current_messages = 0
            current_max_messages = 0
        current_chunk.append(conversation_id)
        current_messages += estimated_messages
        current_max_messages = max(current_max_messages, estimated_messages)

    if current_chunk:
        chunks.append(
            _SessionInsightRefreshChunk(
                conversation_ids=tuple(current_chunk),
                estimated_message_count=current_messages,
                max_estimated_conversation_messages=current_max_messages,
            )
        )
    return chunks


def _flatten_record_bundles(
    bundles: Sequence[SessionInsightRecordBundle],
) -> tuple[list[SessionProfileRecord], list[SessionWorkEventRecord], list[SessionPhaseRecord]]:
    profile_records: list[SessionProfileRecord] = []
    work_event_records: list[SessionWorkEventRecord] = []
    phase_records: list[SessionPhaseRecord] = []
    for bundle in bundles:
        profile_records.append(bundle.profile_record)
        work_event_records.extend(bundle.work_event_records)
        phase_records.extend(bundle.phase_records)
    return profile_records, work_event_records, phase_records


def _refresh_chunk_observation(
    *,
    chunk: _SessionInsightRefreshChunk,
    hydrated_count: int,
    profiles_written: int,
    work_events_written: int,
    phases_written: int,
    load_ms: float,
    hydrate_ms: float,
    build_ms: float,
    write_ms: float,
    total_ms: float,
) -> SessionInsightRefreshChunkObservation:
    return SessionInsightRefreshChunkObservation(
        conversation_count=len(chunk.conversation_ids),
        estimated_message_count=chunk.estimated_message_count,
        max_estimated_conversation_messages=chunk.max_estimated_conversation_messages,
        hydrated_count=hydrated_count,
        profiles_written=profiles_written,
        work_events_written=work_events_written,
        phases_written=phases_written,
        load_ms=load_ms,
        hydrate_ms=hydrate_ms,
        build_ms=build_ms,
        write_ms=write_ms,
        total_ms=total_ms,
        slow=total_ms >= 500.0,
    )


async def _apply_session_product_conversation_updates_async(
    conn: aiosqlite.Connection,
    conversation_ids: Sequence[str],
    *,
    transaction_depth: int,
    page_size: int = _SESSION_INSIGHT_REFRESH_PAGE_SIZE,
) -> _SessionInsightBulkRefreshUpdate:
    from polylogue.storage.backends.queries.session_insight_profile_writes import (
        replace_session_profiles_bulk,
    )
    from polylogue.storage.backends.queries.session_insight_timeline_writes import (
        replace_session_phases_bulk,
        replace_session_work_events_bulk,
    )

    counts = _empty_refresh_counts()
    thread_root_ids: set[str] = set()
    affected_groups: set[ProviderDayGroup] = set()
    chunk_observations: list[SessionInsightRefreshChunkObservation] = []
    conversation_id_list = list(conversation_ids)
    message_counts = await _load_message_counts_async(conn, conversation_id_list)
    conversation_chunks = _chunk_conversation_ids_by_message_budget(
        conversation_id_list,
        message_counts=message_counts,
        page_size=page_size,
        message_budget=_SESSION_INSIGHT_REFRESH_MESSAGE_BUDGET,
    )

    for chunk in conversation_chunks:
        chunk_started = time.perf_counter()
        load_started = time.perf_counter()
        old_profile_records = await _load_existing_session_profile_records_async(conn, chunk.conversation_ids)
        batch = await load_async_batch(conn, chunk.conversation_ids)
        root_ids_by_conversation = await thread_root_ids_async(conn, chunk.conversation_ids)
        load_elapsed_ms = round((time.perf_counter() - load_started) * 1000.0, 1)
        hydrate_started = time.perf_counter()
        hydrated_by_id = {str(conversation.id): conversation for conversation in hydrate_conversations(batch)}
        hydrate_elapsed_ms = round((time.perf_counter() - hydrate_started) * 1000.0, 1)

        build_started = time.perf_counter()
        record_bundles: list[SessionInsightRecordBundle] = []
        for conversation_id in chunk.conversation_ids:
            old_profile_record = old_profile_records.get(conversation_id)
            hydrated_conversation = hydrated_by_id.get(conversation_id)
            if hydrated_conversation is None:
                old_group = profile_provider_day(old_profile_record)
                if old_group is not None:
                    affected_groups.add(old_group)
                continue

            record_bundle = build_session_product_records(hydrated_conversation)
            record_bundles.append(record_bundle)

            counts.add(
                profiles=1,
                work_events=record_bundle.work_event_count,
                phases=record_bundle.phase_count,
            )
            affected_groups.update(
                group
                for group in (
                    profile_provider_day(old_profile_record),
                    profile_provider_day(record_bundle.profile_record),
                )
                if group is not None
            )
            root_id = root_ids_by_conversation.get(conversation_id)
            if root_id is not None:
                thread_root_ids.add(root_id)
        build_elapsed_ms = round((time.perf_counter() - build_started) * 1000.0, 1)

        write_started = time.perf_counter()
        profile_records_to_write, work_event_records_to_write, phase_records_to_write = _flatten_record_bundles(
            record_bundles
        )
        await replace_session_profiles_bulk(
            conn,
            chunk.conversation_ids,
            profile_records_to_write,
            transaction_depth,
        )
        await replace_session_work_events_bulk(
            conn,
            chunk.conversation_ids,
            work_event_records_to_write,
            transaction_depth,
        )
        await replace_session_phases_bulk(
            conn,
            chunk.conversation_ids,
            phase_records_to_write,
            transaction_depth,
        )
        write_elapsed_ms = round((time.perf_counter() - write_started) * 1000.0, 1)
        chunk_total_ms = round((time.perf_counter() - chunk_started) * 1000.0, 1)
        chunk_observations.append(
            _refresh_chunk_observation(
                chunk=chunk,
                hydrated_count=len(hydrated_by_id),
                profiles_written=len(profile_records_to_write),
                work_events_written=len(work_event_records_to_write),
                phases_written=len(phase_records_to_write),
                load_ms=load_elapsed_ms,
                hydrate_ms=hydrate_elapsed_ms,
                build_ms=build_elapsed_ms,
                write_ms=write_elapsed_ms,
                total_ms=chunk_total_ms,
            )
        )

    return _SessionInsightBulkRefreshUpdate(
        counts=counts,
        thread_root_ids=thread_root_ids,
        affected_groups=affected_groups,
        chunk_observations=chunk_observations,
    )


__all__ = [
    "SessionInsightRefreshChunkObservation",
    "delete_session_products_for_conversation_async",
    "refresh_async_provider_day_aggregates",
    "refresh_session_products_for_conversation_async",
    "refresh_thread_after_conversation_delete_async",
]
