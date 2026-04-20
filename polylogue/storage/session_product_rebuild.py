"""Batch loading, hydration, and full rebuild flows for durable session-product read models."""

from __future__ import annotations

import sqlite3
from collections import defaultdict
from collections.abc import AsyncIterator, Iterable, Iterator, Sequence
from dataclasses import dataclass

import aiosqlite

from polylogue.lib.conversation_models import Conversation
from polylogue.lib.session_profile import SessionProfile, build_session_analysis, build_session_profile
from polylogue.protocols import ProgressCallback
from polylogue.storage.action_event_rows import attach_blocks_to_messages
from polylogue.storage.backends.queries.attachments import get_attachments_batch
from polylogue.storage.backends.queries.mappers import (
    _parse_json,
    _row_get,
    _row_to_content_block,
    _row_to_message,
    _row_to_session_profile_record,
)
from polylogue.storage.hydrators import conversation_from_records
from polylogue.storage.session_product_aggregates import (
    list_async_provider_day_groups,
    list_sync_provider_day_groups,
    refresh_async_provider_day_aggregates,
    refresh_sync_provider_day_aggregates,
)
from polylogue.storage.session_product_profiles import (
    build_session_profile_record,
    hydrate_session_profile,
    now_iso,
)
from polylogue.storage.session_product_runtime import SessionProductCounts
from polylogue.storage.session_product_storage import (
    replace_session_phases_sync,
    replace_session_profile_sync,
    replace_session_work_events_sync,
    replace_work_thread_sync,
)
from polylogue.storage.session_product_threads import (
    build_thread_records_for_roots_async,
    build_thread_records_for_roots_sync,
    iter_root_id_pages_async,
    iter_root_id_pages_sync,
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
    SessionPhaseRecord,
    SessionProfileRecord,
    SessionWorkEventRecord,
)
from polylogue.types import ConversationId

_ALL_CONVERSATION_IDS_SQL = (
    "SELECT conversation_id FROM conversations ORDER BY COALESCE(sort_key, 0) DESC, conversation_id"
)
_ALL_SESSION_PROFILE_ROWS_SQL = """
SELECT *
FROM session_profiles
ORDER BY COALESCE(source_sort_key, 0) DESC, conversation_id
"""
# Full rebuilds must tolerate pathological historical provider_meta blobs and
# very large conversation payloads without letting a single chunk inflate RSS
# into multi-GB territory.
_SESSION_PRODUCT_REBUILD_PAGE_SIZE = 1
_SESSION_PRODUCT_CONVERSATION_SQL_TEMPLATE = """
SELECT
    conversation_id,
    provider_name,
    provider_conversation_id,
    title,
    created_at,
    updated_at,
    sort_key,
    content_hash,
    metadata,
    version,
    parent_conversation_id,
    branch_type,
    raw_id,
    json_extract(provider_meta, '$.cwd') AS provider_meta_cwd,
    json_extract(provider_meta, '$.gitBranch') AS provider_meta_git_branch,
    json_extract(provider_meta, '$.git') AS provider_meta_git,
    json_extract(provider_meta, '$.context_compactions') AS provider_meta_context_compactions
FROM conversations
WHERE conversation_id IN ({placeholders})
"""


@dataclass(slots=True)
class SessionProductArchiveBatch:
    conversations: list[ConversationRecord]
    messages: list[MessageRecord]
    attachments_by_conversation: dict[str, list[AttachmentRecord]]
    blocks: list[ContentBlockRecord]


@dataclass(slots=True)
class SessionProductRecordBundle:
    profile_record: SessionProfileRecord
    work_event_records: list[SessionWorkEventRecord]
    phase_records: list[SessionPhaseRecord]

    @property
    def conversation_id(self) -> ConversationId:
        return self.profile_record.conversation_id

    @property
    def work_event_count(self) -> int:
        return len(self.work_event_records)

    @property
    def phase_count(self) -> int:
        return len(self.phase_records)


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
) -> Iterator[SessionProfile]:
    cursor = conn.execute(_ALL_SESSION_PROFILE_ROWS_SQL)
    while True:
        rows = cursor.fetchmany(page_size)
        if not rows:
            break
        for row in rows:
            yield hydrate_session_profile(_row_to_session_profile_record(row))


async def iter_conversation_id_pages_async(
    conn: aiosqlite.Connection,
    *,
    page_size: int,
) -> AsyncIterator[list[str]]:
    cursor = await conn.execute(_ALL_CONVERSATION_IDS_SQL)
    while True:
        rows = await cursor.fetchmany(page_size)
        if not rows:
            break
        yield [str(row["conversation_id"]) for row in rows]


def _row_to_session_product_conversation(row: sqlite3.Row) -> ConversationRecord:
    provider_meta: dict[str, object] = {}
    cwd_value = _row_get(row, "provider_meta_cwd")
    if isinstance(cwd_value, str) and cwd_value:
        provider_meta["cwd"] = cwd_value
    git_branch = _row_get(row, "provider_meta_git_branch")
    if isinstance(git_branch, str) and git_branch:
        provider_meta["gitBranch"] = git_branch
    git_raw = _row_get(row, "provider_meta_git")
    if isinstance(git_raw, str) and git_raw:
        parsed_git = _parse_json(git_raw, field="provider_meta.git", record_id=row["conversation_id"])
        if isinstance(parsed_git, dict) and parsed_git:
            provider_meta["git"] = parsed_git
    compactions_raw = _row_get(row, "provider_meta_context_compactions")
    if isinstance(compactions_raw, str) and compactions_raw:
        parsed_compactions = _parse_json(
            compactions_raw,
            field="provider_meta.context_compactions",
            record_id=row["conversation_id"],
        )
        if isinstance(parsed_compactions, list) and parsed_compactions:
            provider_meta["context_compactions"] = parsed_compactions

    return ConversationRecord(
        conversation_id=row["conversation_id"],
        provider_name=row["provider_name"],
        provider_conversation_id=row["provider_conversation_id"],
        title=row["title"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        sort_key=_row_get(row, "sort_key"),
        content_hash=row["content_hash"],
        provider_meta=provider_meta or None,
        metadata=_parse_json(row["metadata"], field="metadata", record_id=row["conversation_id"]),
        version=row["version"],
        parent_conversation_id=_row_get(row, "parent_conversation_id"),
        branch_type=_row_get(row, "branch_type"),
        raw_id=_row_get(row, "raw_id"),
    )


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
                conversation_id=ConversationId(conversation_id),
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
) -> SessionProductArchiveBatch:
    placeholders = ", ".join("?" for _ in conversation_ids)
    conversations = [
        _row_to_session_product_conversation(row)
        for row in conn.execute(
            _SESSION_PRODUCT_CONVERSATION_SQL_TEMPLATE.format(placeholders=placeholders),
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
    return SessionProductArchiveBatch(
        conversations=conversations,
        messages=messages,
        attachments_by_conversation=sync_attachment_batch(conn, conversation_ids),
        blocks=blocks,
    )


async def load_async_batch(
    conn: aiosqlite.Connection,
    conversation_ids: Sequence[str],
) -> SessionProductArchiveBatch:
    placeholders = ", ".join("?" for _ in conversation_ids)
    conversations = [
        _row_to_session_product_conversation(row)
        for row in await (
            await conn.execute(
                _SESSION_PRODUCT_CONVERSATION_SQL_TEMPLATE.format(placeholders=placeholders),
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
    return SessionProductArchiveBatch(
        conversations=conversations,
        messages=messages,
        attachments_by_conversation=attachments,
        blocks=blocks,
    )


def hydrate_conversations(
    batch: SessionProductArchiveBatch,
) -> list[Conversation]:
    messages_by_conversation: dict[str, list[MessageRecord]] = defaultdict(list)
    blocks_by_conversation: dict[str, list[ContentBlockRecord]] = defaultdict(list)
    for message in batch.messages:
        messages_by_conversation[str(message.conversation_id)].append(message)
    for block in batch.blocks:
        blocks_by_conversation[str(block.conversation_id)].append(block)

    hydrated: list[Conversation] = []
    for conversation in batch.conversations:
        conversation_id = str(conversation.conversation_id)
        attached_messages = attach_blocks_to_messages(
            messages_by_conversation.get(conversation_id, []),
            blocks_by_conversation.get(conversation_id, []),
        )
        hydrated.append(
            conversation_from_records(
                conversation,
                attached_messages,
                batch.attachments_by_conversation.get(conversation_id, []),
            )
        )
    return hydrated


def build_session_product_records(
    conversation: Conversation,
) -> SessionProductRecordBundle:
    analysis = build_session_analysis(conversation)
    profile = build_session_profile(conversation, analysis=analysis)
    materialized_at = now_iso()
    return SessionProductRecordBundle(
        profile_record=build_session_profile_record(
            profile,
            analysis=analysis,
            materialized_at=materialized_at,
        ),
        work_event_records=build_session_work_event_records(profile, materialized_at=materialized_at),
        phase_records=build_session_phase_records(profile, materialized_at=materialized_at),
    )


def build_session_product_record_bundles(
    conversations: Iterable[Conversation],
) -> list[SessionProductRecordBundle]:
    return [build_session_product_records(conversation) for conversation in conversations]


def _count_record_bundles(
    bundles: Sequence[SessionProductRecordBundle],
) -> tuple[int, int, int]:
    return (
        len(bundles),
        sum(bundle.work_event_count for bundle in bundles),
        sum(bundle.phase_count for bundle in bundles),
    )


def _materialize_progress_desc(
    *,
    profile_count: int,
    progress_total: int | None,
) -> str:
    if progress_total is not None:
        return f"Materializing: {profile_count}/{progress_total}"
    return f"Materializing: {profile_count}"


def _empty_rebuild_counts() -> SessionProductCounts:
    return SessionProductCounts()


def _finalize_rebuild_counts(
    *,
    profiles: int,
    work_events: int,
    phases: int,
    threads: int,
    tag_rollups: int,
    day_summaries: int,
) -> SessionProductCounts:
    return SessionProductCounts(
        profiles=profiles,
        work_events=work_events,
        phases=phases,
        threads=threads,
        tag_rollups=tag_rollups,
        day_summaries=day_summaries,
    )


def rebuild_session_products_sync(
    conn: sqlite3.Connection,
    *,
    conversation_ids: Sequence[str] | None = None,
    page_size: int = _SESSION_PRODUCT_REBUILD_PAGE_SIZE,
    progress_callback: ProgressCallback | None = None,
    progress_total: int | None = None,
) -> SessionProductCounts:
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
        return _empty_rebuild_counts()

    profile_count = 0
    work_event_count = 0
    phase_count = 0
    saw_conversation_ids = False
    for chunk in conversation_chunks:
        saw_conversation_ids = True
        batch = load_sync_batch(conn, chunk)
        record_bundles = build_session_product_record_bundles(hydrate_conversations(batch))
        chunk_profiles, chunk_work_events, chunk_phases = _count_record_bundles(record_bundles)
        for bundle in record_bundles:
            replace_session_profile_sync(conn, bundle.profile_record)
            replace_session_work_events_sync(
                conn,
                bundle.conversation_id,
                bundle.work_event_records,
            )
            replace_session_phases_sync(
                conn,
                bundle.conversation_id,
                bundle.phase_records,
            )
        profile_count += chunk_profiles
        work_event_count += chunk_work_events
        phase_count += chunk_phases
        if progress_callback is not None and chunk_profiles:
            progress_callback(
                chunk_profiles,
                desc=_materialize_progress_desc(
                    profile_count=profile_count,
                    progress_total=progress_total,
                ),
            )
    if not saw_conversation_ids:
        conn.execute("DELETE FROM work_threads")
        conn.execute("DELETE FROM session_phases")
        conn.execute("DELETE FROM session_tag_rollups")
        conn.execute("DELETE FROM day_session_summaries")
        conn.commit()
        return _empty_rebuild_counts()

    conn.execute("DELETE FROM work_threads")
    thread_count = 0
    for root_chunk in iter_root_id_pages_sync(conn):
        records_by_root = build_thread_records_for_roots_sync(conn, root_chunk)
        for root_id in root_chunk:
            record = records_by_root.get(root_id)
            if record is None:
                continue
            replace_work_thread_sync(conn, record.thread_id, record)
            thread_count += 1
    conn.execute("DELETE FROM day_session_summaries")
    conn.execute("DELETE FROM session_tag_rollups")
    provider_day_groups = set(list_sync_provider_day_groups(conn))
    refresh_sync_provider_day_aggregates(conn, provider_day_groups)
    tag_rollup_count = conn.execute("SELECT COUNT(*) FROM session_tag_rollups").fetchone()[0]
    day_summary_count = conn.execute("SELECT COUNT(*) FROM day_session_summaries").fetchone()[0]
    conn.commit()
    return _finalize_rebuild_counts(
        profiles=profile_count,
        work_events=work_event_count,
        phases=phase_count,
        threads=thread_count,
        tag_rollups=int(tag_rollup_count),
        day_summaries=int(day_summary_count),
    )


async def rebuild_session_products_async(
    conn: aiosqlite.Connection,
    *,
    conversation_ids: Sequence[str] | None = None,
    page_size: int = _SESSION_PRODUCT_REBUILD_PAGE_SIZE,
    transaction_depth: int = 0,
    progress_callback: ProgressCallback | None = None,
    progress_total: int | None = None,
) -> SessionProductCounts:
    from polylogue.storage.backends.queries.session_product_profile_writes import (
        replace_session_profile,
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
    elif not conversation_ids:
        await conn.execute("DELETE FROM work_threads")
        await conn.execute("DELETE FROM session_phases")
        await conn.execute("DELETE FROM session_tag_rollups")
        await conn.execute("DELETE FROM day_session_summaries")
        return _empty_rebuild_counts()

    profile_count = 0
    work_event_count = 0
    phase_count = 0
    if conversation_ids is None:
        async for chunk in iter_conversation_id_pages_async(conn, page_size=page_size):
            batch = await load_async_batch(conn, chunk)
            record_bundles = build_session_product_record_bundles(hydrate_conversations(batch))
            chunk_profiles, chunk_work_events, chunk_phases = _count_record_bundles(record_bundles)
            for bundle in record_bundles:
                await replace_session_profile(conn, bundle.profile_record, transaction_depth)
                await replace_session_work_events(
                    conn,
                    bundle.conversation_id,
                    bundle.work_event_records,
                    transaction_depth,
                )
                await replace_session_phases(
                    conn,
                    bundle.conversation_id,
                    bundle.phase_records,
                    transaction_depth,
                )
            profile_count += chunk_profiles
            work_event_count += chunk_work_events
            phase_count += chunk_phases
            if progress_callback is not None and chunk_profiles:
                progress_callback(
                    chunk_profiles,
                    desc=_materialize_progress_desc(
                        profile_count=profile_count,
                        progress_total=progress_total,
                    ),
                )
    else:
        for chunk_ids in chunked(list(conversation_ids), size=page_size):
            batch = await load_async_batch(conn, chunk_ids)
            record_bundles = build_session_product_record_bundles(hydrate_conversations(batch))
            chunk_profiles, chunk_work_events, chunk_phases = _count_record_bundles(record_bundles)
            for bundle in record_bundles:
                await replace_session_profile(conn, bundle.profile_record, transaction_depth)
                await replace_session_work_events(
                    conn,
                    bundle.conversation_id,
                    bundle.work_event_records,
                    transaction_depth,
                )
                await replace_session_phases(
                    conn,
                    bundle.conversation_id,
                    bundle.phase_records,
                    transaction_depth,
                )
            profile_count += chunk_profiles
            work_event_count += chunk_work_events
            phase_count += chunk_phases
            if progress_callback is not None and chunk_profiles:
                progress_callback(
                    chunk_profiles,
                    desc=_materialize_progress_desc(
                        profile_count=profile_count,
                        progress_total=progress_total,
                    ),
                )

    await conn.execute("DELETE FROM work_threads")
    thread_count = 0
    async for root_chunk in iter_root_id_pages_async(conn):
        records_by_root = await build_thread_records_for_roots_async(conn, root_chunk)
        for root_id in root_chunk:
            record = records_by_root.get(root_id)
            if record is None:
                continue
            await replace_work_thread(conn, record.thread_id, record, transaction_depth)
            thread_count += 1
    await conn.execute("DELETE FROM day_session_summaries")
    await conn.execute("DELETE FROM session_tag_rollups")
    provider_day_groups = set(await list_async_provider_day_groups(conn))
    await refresh_async_provider_day_aggregates(
        conn,
        provider_day_groups,
        transaction_depth=transaction_depth,
    )
    tag_rollup_row = await (await conn.execute("SELECT COUNT(*) FROM session_tag_rollups")).fetchone()
    day_summary_row = await (await conn.execute("SELECT COUNT(*) FROM day_session_summaries")).fetchone()
    tag_rollup_count = int(tag_rollup_row[0]) if tag_rollup_row is not None else 0
    day_summary_count = int(day_summary_row[0]) if day_summary_row is not None else 0
    return _finalize_rebuild_counts(
        profiles=profile_count,
        work_events=work_event_count,
        phases=phase_count,
        threads=thread_count,
        tag_rollups=int(tag_rollup_count),
        day_summaries=int(day_summary_count),
    )


__all__ = [
    "_ALL_CONVERSATION_IDS_SQL",
    "_ALL_SESSION_PROFILE_ROWS_SQL",
    "SessionProductArchiveBatch",
    "SessionProductRecordBundle",
    "build_session_product_record_bundles",
    "build_session_product_records",
    "chunked",
    "hydrate_conversations",
    "iter_conversation_id_pages_async",
    "iter_conversation_id_pages_sync",
    "iter_hydrated_session_profiles_sync",
    "load_async_batch",
    "load_sync_batch",
    "rebuild_session_products_async",
    "rebuild_session_products_sync",
    "sync_attachment_batch",
]
