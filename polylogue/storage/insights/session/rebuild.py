"""Batch loading, hydration, and full rebuild flows for durable session-insight read models."""

from __future__ import annotations

import sqlite3
from collections import defaultdict
from collections.abc import AsyncIterator, Iterable, Iterator, Sequence
from dataclasses import dataclass

import aiosqlite

from polylogue.archive.conversation.branch_type import BranchType
from polylogue.archive.conversation.models import Conversation
from polylogue.archive.session.session_profile import SessionProfile, build_session_analysis, build_session_profile

# Re-export canonical chunked from polylogue.core.common.
from polylogue.core.common import chunked
from polylogue.core.memory import release_process_memory
from polylogue.protocols import ProgressCallback
from polylogue.storage.action_events.rows import attach_blocks_to_messages
from polylogue.storage.hydrators import conversation_from_records
from polylogue.storage.insights.session.aggregates import (
    list_async_provider_day_groups,
    list_sync_provider_day_groups,
    profile_provider_day,
    refresh_async_provider_day_aggregates,
    refresh_sync_provider_day_aggregates,
)
from polylogue.storage.insights.session.latency_profiles import (
    build_latency_profile_facts,
    build_session_latency_profile_record,
)
from polylogue.storage.insights.session.profiles import (
    build_session_profile_record,
    hydrate_session_profile,
    now_iso,
)
from polylogue.storage.insights.session.runtime import SessionInsightCounts
from polylogue.storage.insights.session.storage import (
    replace_session_latency_profiles_bulk_sync,
    replace_session_phases_bulk_sync,
    replace_session_profiles_bulk_sync,
    replace_session_work_events_bulk_sync,
    replace_work_thread_sync,
    replace_work_threads_bulk_sync,
)
from polylogue.storage.insights.session.threads import (
    build_thread_records_for_roots_async,
    build_thread_records_for_roots_sync,
    iter_root_id_pages_async,
    iter_root_id_pages_sync,
    thread_root_ids_sync,
)
from polylogue.storage.insights.session.timeline_rows import (
    build_session_phase_records,
    build_session_work_event_records,
)
from polylogue.storage.runtime import (
    AttachmentRecord,
    ContentBlockRecord,
    ConversationRecord,
    MessageRecord,
    ProviderEventRecord,
    SessionLatencyProfileRecord,
    SessionPhaseRecord,
    SessionProfileRecord,
    SessionWorkEventRecord,
)
from polylogue.storage.sqlite.queries.attachments import get_attachments_batch
from polylogue.storage.sqlite.queries.mappers import (
    _json_object,
    _parse_json,
    _row_float,
    _row_get,
    _row_text,
    _row_to_content_block,
    _row_to_message,
    _row_to_session_profile_record,
)
from polylogue.storage.sqlite.queries.provider_events import (
    get_provider_event_compaction_counts,
    get_provider_events_batch,
    sync_provider_event_compaction_counts,
    sync_provider_events_batch,
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
# into multi-GB territory. The message-budget chunker
# (_chunk_conversation_ids_by_message_budget_sync) caps total messages per
# chunk, so the page size only controls per-conversation SQL round-trips.
# A page size of 1 caused ~17K round-trips for ~4K conversations on the
# scale_small fixture; 50 keeps RSS bounded by the message budget while
# cutting round-trips by ~50x (#1314).
_SESSION_INSIGHT_REBUILD_PAGE_SIZE = 50
_SESSION_INSIGHT_REBUILD_MESSAGE_BUDGET = 5_000
_SESSION_INSIGHT_RELEASE_MESSAGE_THRESHOLD = 1_000
_SESSION_INSIGHT_MESSAGE_TEXT_PREVIEW_CHARS = 16_384
_SESSION_INSIGHT_BLOCK_TEXT_PREVIEW_CHARS = 4_096
_SESSION_INSIGHT_CONVERSATION_SQL_TEMPLATE = """
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
    json_extract(provider_meta, '$.total_cost_usd') AS provider_meta_total_cost_usd,
    json_extract(provider_meta, '$.total_duration_ms') AS provider_meta_total_duration_ms,
    json_extract(provider_meta, '$.models_used') AS provider_meta_models_used
FROM conversations
WHERE conversation_id IN ({placeholders})
"""
_SESSION_INSIGHT_MESSAGE_SQL_TEMPLATE = """
SELECT
    message_id,
    conversation_id,
    provider_message_id,
    role,
    CASE
        WHEN text IS NULL THEN NULL
        ELSE substr(text, 1, ?)
    END AS text,
    sort_key,
    content_hash,
    version,
    parent_message_id,
    branch_index,
    provider_name,
    word_count,
    has_tool_use,
    has_thinking,
    has_paste,
    input_tokens,
    output_tokens,
    cache_read_tokens,
    cache_write_tokens,
    model_name,
    message_type
FROM messages
WHERE conversation_id IN ({placeholders})
ORDER BY conversation_id, sort_key, message_id
"""
_SESSION_INSIGHT_BLOCK_SQL_TEMPLATE = """
SELECT
    block_id,
    message_id,
    conversation_id,
    block_index,
    type,
    CASE
        WHEN text IS NULL THEN NULL
        ELSE substr(text, 1, ?)
    END AS text,
    tool_name,
    tool_id,
    tool_input,
    metadata,
    semantic_type
FROM content_blocks
WHERE conversation_id IN ({placeholders})
  AND message_id IN (
      SELECT message_id
      FROM messages
      WHERE conversation_id IN ({placeholders})
        AND (
            has_tool_use != 0
            OR has_thinking != 0
            OR message_type IN ('tool_use', 'tool_result', 'thinking')
        )
  )
ORDER BY conversation_id, message_id, block_index
"""


@dataclass(slots=True)
class SessionInsightArchiveBatch:
    conversations: list[ConversationRecord]
    messages: list[MessageRecord]
    attachments_by_conversation: dict[str, list[AttachmentRecord]]
    provider_events_by_conversation: dict[str, list[ProviderEventRecord]]
    compaction_counts_by_conversation: dict[str, int]
    blocks: list[ContentBlockRecord]


@dataclass(slots=True)
class SessionInsightRecordBundle:
    profile_record: SessionProfileRecord
    latency_profile_record: SessionLatencyProfileRecord
    work_event_records: list[SessionWorkEventRecord]
    phase_records: list[SessionPhaseRecord]
    repo_observations: tuple[object, ...] = ()
    """Repo identity observations for ``conversation_repo_observations`` (#1253).

    Typed as ``RepoObservation`` from
    ``polylogue.storage.insights.session.repo_identity``; kept as
    ``object`` here to avoid a circular import at module load time.
    """

    @property
    def conversation_id(self) -> ConversationId:
        return self.profile_record.conversation_id

    @property
    def work_event_count(self) -> int:
        return len(self.work_event_records)

    @property
    def phase_count(self) -> int:
        return len(self.phase_records)


@dataclass(slots=True, frozen=True)
class _SessionInsightRebuildChunk:
    conversation_ids: tuple[str, ...]
    estimated_message_count: int
    max_estimated_conversation_messages: int


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
        rows = list(await cursor.fetchmany(page_size))
        if not rows:
            break
        yield [str(row["conversation_id"]) for row in rows]


def _row_to_session_insight_conversation(row: sqlite3.Row) -> ConversationRecord:
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
    total_cost = _row_get(row, "provider_meta_total_cost_usd")
    if isinstance(total_cost, int | float):
        provider_meta["total_cost_usd"] = float(total_cost)
    total_duration = _row_get(row, "provider_meta_total_duration_ms")
    if isinstance(total_duration, int | float):
        provider_meta["total_duration_ms"] = int(total_duration)
    models_used = _row_get(row, "provider_meta_models_used")
    if isinstance(models_used, str) and models_used:
        parsed_models = _parse_json(models_used, field="provider_meta.models_used", record_id=row["conversation_id"])
        if isinstance(parsed_models, list):
            provider_meta["models_used"] = [item for item in parsed_models if isinstance(item, str)]
    parent_conversation_id = _row_text(row, "parent_conversation_id")
    branch_type = _row_text(row, "branch_type")

    return ConversationRecord(
        conversation_id=row["conversation_id"],
        provider_name=row["provider_name"],
        provider_conversation_id=row["provider_conversation_id"],
        title=row["title"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        sort_key=_row_float(row, "sort_key"),
        content_hash=row["content_hash"],
        provider_meta=provider_meta or None,
        metadata=_json_object(_parse_json(row["metadata"], field="metadata", record_id=row["conversation_id"])),
        version=row["version"],
        parent_conversation_id=ConversationId(parent_conversation_id) if parent_conversation_id is not None else None,
        branch_type=BranchType(branch_type) if branch_type is not None else None,
        raw_id=_row_text(row, "raw_id"),
    )


def _load_message_counts_sync(
    conn: sqlite3.Connection,
    conversation_ids: Sequence[str],
) -> dict[str, int]:
    if not conversation_ids:
        return {}
    placeholders = ", ".join("?" for _ in conversation_ids)
    rows = conn.execute(
        f"""
        SELECT conversation_id, message_count
        FROM conversation_stats
        WHERE conversation_id IN ({placeholders})
        """,
        tuple(conversation_ids),
    ).fetchall()
    return {str(row["conversation_id"]): int(row["message_count"] or 0) for row in rows}


def _chunk_conversation_ids_by_message_budget_sync(
    conversation_ids: Sequence[str],
    *,
    message_counts: dict[str, int],
    page_size: int,
    message_budget: int,
) -> list[_SessionInsightRebuildChunk]:
    chunks: list[_SessionInsightRebuildChunk] = []
    current_chunk: list[str] = []
    current_messages = 0
    current_max_messages = 0

    for conversation_id in conversation_ids:
        estimated_messages = max(int(message_counts.get(conversation_id, 0) or 0), 1)
        if current_chunk and (
            len(current_chunk) >= page_size or current_messages + estimated_messages > message_budget
        ):
            chunks.append(
                _SessionInsightRebuildChunk(
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
            _SessionInsightRebuildChunk(
                conversation_ids=tuple(current_chunk),
                estimated_message_count=current_messages,
                max_estimated_conversation_messages=current_max_messages,
            )
        )
    return chunks


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
) -> SessionInsightArchiveBatch:
    placeholders = ", ".join("?" for _ in conversation_ids)
    conversations = [
        _row_to_session_insight_conversation(row)
        for row in conn.execute(
            _SESSION_INSIGHT_CONVERSATION_SQL_TEMPLATE.format(placeholders=placeholders),
            tuple(conversation_ids),
        ).fetchall()
    ]
    messages = [
        _row_to_message(row)
        for row in conn.execute(
            _SESSION_INSIGHT_MESSAGE_SQL_TEMPLATE.format(placeholders=placeholders),
            (_SESSION_INSIGHT_MESSAGE_TEXT_PREVIEW_CHARS, *conversation_ids),
        ).fetchall()
    ]
    blocks = [
        _row_to_content_block(row)
        for row in conn.execute(
            _SESSION_INSIGHT_BLOCK_SQL_TEMPLATE.format(placeholders=placeholders),
            (_SESSION_INSIGHT_BLOCK_TEXT_PREVIEW_CHARS, *conversation_ids, *conversation_ids),
        ).fetchall()
    ]
    return SessionInsightArchiveBatch(
        conversations=conversations,
        messages=messages,
        attachments_by_conversation=sync_attachment_batch(conn, conversation_ids),
        provider_events_by_conversation=sync_provider_events_batch(conn, conversation_ids),
        compaction_counts_by_conversation=sync_provider_event_compaction_counts(conn, conversation_ids),
        blocks=blocks,
    )


async def load_async_batch(
    conn: aiosqlite.Connection,
    conversation_ids: Sequence[str],
) -> SessionInsightArchiveBatch:
    placeholders = ", ".join("?" for _ in conversation_ids)
    conversations = [
        _row_to_session_insight_conversation(row)
        for row in await (
            await conn.execute(
                _SESSION_INSIGHT_CONVERSATION_SQL_TEMPLATE.format(placeholders=placeholders),
                tuple(conversation_ids),
            )
        ).fetchall()
    ]
    messages = [
        _row_to_message(row)
        for row in await (
            await conn.execute(
                _SESSION_INSIGHT_MESSAGE_SQL_TEMPLATE.format(placeholders=placeholders),
                (_SESSION_INSIGHT_MESSAGE_TEXT_PREVIEW_CHARS, *conversation_ids),
            )
        ).fetchall()
    ]
    blocks = [
        _row_to_content_block(row)
        for row in await (
            await conn.execute(
                _SESSION_INSIGHT_BLOCK_SQL_TEMPLATE.format(placeholders=placeholders),
                (_SESSION_INSIGHT_BLOCK_TEXT_PREVIEW_CHARS, *conversation_ids, *conversation_ids),
            )
        ).fetchall()
    ]
    attachments = await get_attachments_batch(conn, list(conversation_ids))
    provider_events = await get_provider_events_batch(conn, list(conversation_ids))
    compaction_counts = await get_provider_event_compaction_counts(conn, list(conversation_ids))
    return SessionInsightArchiveBatch(
        conversations=conversations,
        messages=messages,
        attachments_by_conversation=attachments,
        provider_events_by_conversation=provider_events,
        compaction_counts_by_conversation=compaction_counts,
        blocks=blocks,
    )


def hydrate_conversations(
    batch: SessionInsightArchiveBatch,
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
                batch.provider_events_by_conversation.get(conversation_id, []),
            )
        )
    return hydrated


def build_session_insight_records(
    conversation: Conversation,
    *,
    compaction_count: int | None = None,
) -> SessionInsightRecordBundle:
    from polylogue.storage.insights.session.repo_identity import attribution_to_observations

    analysis = build_session_analysis(conversation)
    profile = build_session_profile(conversation, analysis=analysis, compaction_count=compaction_count)
    materialized_at = now_iso()
    provider_meta: dict[str, object] = (
        conversation.provider_meta if isinstance(conversation.provider_meta, dict) else {}
    )
    git_meta = provider_meta.get("git")
    git_repository_url: str | None = None
    if isinstance(git_meta, dict):
        repository_url = git_meta.get("repository_url")
        if isinstance(repository_url, str) and repository_url.strip():
            git_repository_url = repository_url.strip()
    repo_observations = attribution_to_observations(
        analysis.attribution,
        git_repository_url=git_repository_url,
    )
    latency_facts = build_latency_profile_facts(conversation, profile)
    return SessionInsightRecordBundle(
        profile_record=build_session_profile_record(
            profile,
            analysis=analysis,
            materialized_at=materialized_at,
        ),
        latency_profile_record=build_session_latency_profile_record(
            conversation,
            profile,
            latency_facts,
            materialized_at=materialized_at,
        ),
        work_event_records=build_session_work_event_records(profile, materialized_at=materialized_at),
        phase_records=build_session_phase_records(profile, materialized_at=materialized_at),
        repo_observations=repo_observations,
    )


def build_session_insight_record_bundles(
    conversations: Iterable[Conversation],
    *,
    compaction_counts_by_conversation: dict[str, int] | None = None,
) -> list[SessionInsightRecordBundle]:
    compaction_counts = compaction_counts_by_conversation or {}
    return [
        build_session_insight_records(
            conversation,
            compaction_count=compaction_counts.get(str(conversation.id)),
        )
        for conversation in conversations
    ]


def _count_record_bundles(
    bundles: Sequence[SessionInsightRecordBundle],
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


def _empty_rebuild_counts() -> SessionInsightCounts:
    return SessionInsightCounts()


def _finalize_rebuild_counts(
    *,
    profiles: int,
    work_events: int,
    phases: int,
    threads: int,
    tag_rollups: int,
    day_summaries: int,
) -> SessionInsightCounts:
    return SessionInsightCounts(
        profiles=profiles,
        work_events=work_events,
        phases=phases,
        threads=threads,
        tag_rollups=tag_rollups,
        day_summaries=day_summaries,
    )


def _session_profile_records_for_conversation_ids_sync(
    conn: sqlite3.Connection,
    conversation_ids: Sequence[str],
) -> list[SessionProfileRecord]:
    if not conversation_ids:
        return []
    placeholders = ", ".join("?" for _ in conversation_ids)
    rows = conn.execute(
        f"SELECT * FROM session_profiles WHERE conversation_id IN ({placeholders})",
        tuple(conversation_ids),
    ).fetchall()
    return [_row_to_session_profile_record(row) for row in rows]


def _refresh_thread_roots_sync(
    conn: sqlite3.Connection,
    root_ids: Sequence[str],
) -> int:
    normalized_root_ids = tuple(dict.fromkeys(str(root_id) for root_id in root_ids if str(root_id)))
    if not normalized_root_ids:
        return 0
    records_by_root = build_thread_records_for_roots_sync(conn, normalized_root_ids)
    replace_work_threads_bulk_sync(conn, {root_id: records_by_root.get(root_id) for root_id in normalized_root_ids})
    return len(normalized_root_ids)


def rebuild_session_insights_sync(
    conn: sqlite3.Connection,
    *,
    conversation_ids: Sequence[str] | None = None,
    page_size: int = _SESSION_INSIGHT_REBUILD_PAGE_SIZE,
    progress_callback: ProgressCallback | None = None,
    progress_total: int | None = None,
) -> SessionInsightCounts:
    conversation_chunks: Iterable[Sequence[str]]
    previous_profile_groups: set[tuple[str, str]] = set()
    if conversation_ids is None:
        conn.execute("DELETE FROM session_work_events")
        conn.execute("DELETE FROM session_phases")
        conn.execute("DELETE FROM session_latency_profiles")
        conn.execute("DELETE FROM session_profiles")
        conn.execute("DELETE FROM session_tag_rollups")
        conn.execute("DELETE FROM day_session_summaries")
        conn.execute("DELETE FROM conversation_repo_observations")
        conn.execute("DELETE FROM repo_identities")
        conversation_chunks = iter_conversation_id_pages_sync(conn, page_size=page_size)
    else:
        conversation_ids = tuple(dict.fromkeys(str(conversation_id) for conversation_id in conversation_ids))
        previous_profile_groups = {
            group
            for record in _session_profile_records_for_conversation_ids_sync(conn, conversation_ids)
            if (group := profile_provider_day(record)) is not None
        }
        message_counts = _load_message_counts_sync(conn, conversation_ids)
        conversation_chunks = (
            chunk.conversation_ids
            for chunk in _chunk_conversation_ids_by_message_budget_sync(
                conversation_ids,
                message_counts=message_counts,
                page_size=page_size,
                message_budget=_SESSION_INSIGHT_REBUILD_MESSAGE_BUDGET,
            )
        )
    if conversation_ids is not None and not conversation_ids:
        conn.commit()
        return _empty_rebuild_counts()

    profile_count = 0
    work_event_count = 0
    phase_count = 0
    refreshed_profile_groups: set[tuple[str, str]] = set()
    saw_conversation_ids = False
    for chunk in conversation_chunks:
        saw_conversation_ids = True
        batch = load_sync_batch(conn, chunk)
        record_bundles = build_session_insight_record_bundles(
            hydrate_conversations(batch),
            compaction_counts_by_conversation=batch.compaction_counts_by_conversation,
        )
        chunk_profiles, chunk_work_events, chunk_phases = _count_record_bundles(record_bundles)
        replace_session_profiles_bulk_sync(conn, [bundle.profile_record for bundle in record_bundles])
        replace_session_latency_profiles_bulk_sync(
            conn,
            [bundle.latency_profile_record for bundle in record_bundles],
        )
        replace_session_work_events_bulk_sync(
            conn,
            {bundle.conversation_id: bundle.work_event_records for bundle in record_bundles},
        )
        replace_session_phases_bulk_sync(
            conn,
            {bundle.conversation_id: bundle.phase_records for bundle in record_bundles},
        )
        from polylogue.storage.insights.session.repo_identity import (
            RepoObservation as _RepoObservationSync,
        )
        from polylogue.storage.insights.session.repo_identity import (
            refresh_conversation_repo_observations_sync as _refresh_repo_obs_sync,
        )

        for bundle in record_bundles:
            _refresh_repo_obs_sync(
                conn,
                str(bundle.conversation_id),
                tuple(obs for obs in bundle.repo_observations if isinstance(obs, _RepoObservationSync)),
            )
        for bundle in record_bundles:
            group = profile_provider_day(bundle.profile_record)
            if group is not None:
                refreshed_profile_groups.add(group)
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
        if len(batch.messages) >= _SESSION_INSIGHT_RELEASE_MESSAGE_THRESHOLD:
            del batch, record_bundles
            release_process_memory()
    if not saw_conversation_ids:
        if conversation_ids is None:
            conn.execute("DELETE FROM work_threads")
            conn.execute("DELETE FROM session_phases")
            conn.execute("DELETE FROM session_tag_rollups")
            conn.execute("DELETE FROM day_session_summaries")
        conn.commit()
        return _empty_rebuild_counts()

    if conversation_ids is not None:
        affected_roots = tuple(thread_root_ids_sync(conn, conversation_ids).values())
        thread_count = _refresh_thread_roots_sync(conn, affected_roots)
        refresh_sync_provider_day_aggregates(
            conn,
            previous_profile_groups | refreshed_profile_groups,
        )
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


async def rebuild_session_insights_async(
    conn: aiosqlite.Connection,
    *,
    conversation_ids: Sequence[str] | None = None,
    page_size: int = _SESSION_INSIGHT_REBUILD_PAGE_SIZE,
    transaction_depth: int = 0,
    progress_callback: ProgressCallback | None = None,
    progress_total: int | None = None,
) -> SessionInsightCounts:
    from polylogue.storage.sqlite.queries.session_insight_profile_writes import (
        replace_session_latency_profile,
        replace_session_profile,
    )
    from polylogue.storage.sqlite.queries.session_insight_thread_queries import replace_work_thread
    from polylogue.storage.sqlite.queries.session_insight_timeline_writes import (
        replace_session_phases,
        replace_session_work_events,
    )

    if conversation_ids is None:
        await conn.execute("DELETE FROM session_work_events")
        await conn.execute("DELETE FROM session_phases")
        await conn.execute("DELETE FROM session_latency_profiles")
        await conn.execute("DELETE FROM session_profiles")
        await conn.execute("DELETE FROM session_tag_rollups")
        await conn.execute("DELETE FROM day_session_summaries")
        await conn.execute("DELETE FROM conversation_repo_observations")
        await conn.execute("DELETE FROM repo_identities")
    elif not conversation_ids:
        await conn.execute("DELETE FROM work_threads")
        await conn.execute("DELETE FROM session_phases")
        await conn.execute("DELETE FROM session_latency_profiles")
        await conn.execute("DELETE FROM session_tag_rollups")
        await conn.execute("DELETE FROM day_session_summaries")
        return _empty_rebuild_counts()

    profile_count = 0
    work_event_count = 0
    phase_count = 0
    if conversation_ids is None:
        async for chunk in iter_conversation_id_pages_async(conn, page_size=page_size):
            batch = await load_async_batch(conn, chunk)
            record_bundles = build_session_insight_record_bundles(
                hydrate_conversations(batch),
                compaction_counts_by_conversation=batch.compaction_counts_by_conversation,
            )
            chunk_profiles, chunk_work_events, chunk_phases = _count_record_bundles(record_bundles)
            for bundle in record_bundles:
                await replace_session_profile(conn, bundle.profile_record, transaction_depth)
                await replace_session_latency_profile(conn, bundle.latency_profile_record, transaction_depth)
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
                from polylogue.storage.insights.session.repo_identity import (
                    RepoObservation as _RepoObservation,
                )
                from polylogue.storage.insights.session.repo_identity import (
                    refresh_conversation_repo_observations as _refresh_repo_obs,
                )

                await _refresh_repo_obs(
                    conn,
                    str(bundle.conversation_id),
                    tuple(obs for obs in bundle.repo_observations if isinstance(obs, _RepoObservation)),
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
            if len(batch.messages) >= _SESSION_INSIGHT_RELEASE_MESSAGE_THRESHOLD:
                del batch, record_bundles
                release_process_memory()
    else:
        for chunk_ids in chunked(list(conversation_ids), size=page_size):
            batch = await load_async_batch(conn, chunk_ids)
            record_bundles = build_session_insight_record_bundles(
                hydrate_conversations(batch),
                compaction_counts_by_conversation=batch.compaction_counts_by_conversation,
            )
            chunk_profiles, chunk_work_events, chunk_phases = _count_record_bundles(record_bundles)
            for bundle in record_bundles:
                await replace_session_profile(conn, bundle.profile_record, transaction_depth)
                await replace_session_latency_profile(conn, bundle.latency_profile_record, transaction_depth)
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
                from polylogue.storage.insights.session.repo_identity import (
                    RepoObservation as _RepoObservation,
                )
                from polylogue.storage.insights.session.repo_identity import (
                    refresh_conversation_repo_observations as _refresh_repo_obs,
                )

                await _refresh_repo_obs(
                    conn,
                    str(bundle.conversation_id),
                    tuple(obs for obs in bundle.repo_observations if isinstance(obs, _RepoObservation)),
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
            if len(batch.messages) >= _SESSION_INSIGHT_RELEASE_MESSAGE_THRESHOLD:
                del batch, record_bundles
                release_process_memory()

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
    "SessionInsightArchiveBatch",
    "SessionInsightRecordBundle",
    "build_session_insight_record_bundles",
    "build_session_insight_records",
    "chunked",
    "hydrate_conversations",
    "iter_conversation_id_pages_async",
    "iter_conversation_id_pages_sync",
    "iter_hydrated_session_profiles_sync",
    "load_async_batch",
    "load_sync_batch",
    "rebuild_session_insights_async",
    "rebuild_session_insights_sync",
    "sync_attachment_batch",
]
