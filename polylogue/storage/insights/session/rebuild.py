"""Batch loading, hydration, and full rebuild flows for durable session-insight read models."""

from __future__ import annotations

import sqlite3
from collections import defaultdict
from collections.abc import AsyncIterator, Iterable, Iterator, Sequence
from dataclasses import dataclass

import aiosqlite

from polylogue.archive.session.branch_type import BranchType
from polylogue.archive.session.domain_models import Session
from polylogue.archive.session.session_profile import SessionProfile, build_session_analysis, build_session_profile

# Re-export canonical chunked from polylogue.core.common.
from polylogue.core.common import chunked
from polylogue.core.memory import release_process_memory
from polylogue.protocols import ProgressCallback
from polylogue.storage.action_events.rows import attach_blocks_to_messages
from polylogue.storage.hydrators import session_from_records
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
    thread_root_ids_async,
    thread_root_ids_sync,
)
from polylogue.storage.insights.session.timeline_rows import (
    build_session_phase_records,
    build_session_work_event_records,
)
from polylogue.storage.runtime import (
    AttachmentRecord,
    ContentBlockRecord,
    MessageRecord,
    ProviderEventRecord,
    SessionLatencyProfileRecord,
    SessionPhaseRecord,
    SessionProfileRecord,
    SessionRecord,
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
from polylogue.types import SessionId

_ALL_SESSION_IDS_SQL = "SELECT session_id FROM sessions ORDER BY COALESCE(sort_key, 0) DESC, session_id"
_ALL_SESSION_PROFILE_ROWS_SQL = """
SELECT *
FROM session_profiles
ORDER BY COALESCE(source_sort_key, 0) DESC, session_id
"""
# Full rebuilds must tolerate pathological historical provider_meta blobs and
# very large session payloads without letting a single chunk inflate RSS
# into multi-GB territory. The message-budget chunker
# (_chunk_session_ids_by_message_budget_sync) caps total messages per
# chunk, so the page size only controls per-session SQL round-trips.
# A page size of 1 caused ~17K round-trips for ~4K sessions on the
# scale_small fixture; 50 keeps RSS bounded by the message budget while
# cutting round-trips by ~50x (#1314).
_SESSION_INSIGHT_REBUILD_PAGE_SIZE = 50
_SESSION_INSIGHT_REBUILD_MESSAGE_BUDGET = 5_000
_SESSION_INSIGHT_RELEASE_MESSAGE_THRESHOLD = 1_000
_SESSION_INSIGHT_MESSAGE_TEXT_PREVIEW_CHARS = 16_384
_SESSION_INSIGHT_BLOCK_TEXT_PREVIEW_CHARS = 4_096
_SESSION_INSIGHT_SESSION_SQL_TEMPLATE = """
SELECT
    session_id,
    source_name,
    provider_session_id,
    title,
    created_at,
    updated_at,
    sort_key,
    content_hash,
    metadata,
    version,
    parent_session_id,
    branch_type,
    raw_id,
    json_extract(provider_meta, '$.cwd') AS provider_meta_cwd,
    json_extract(provider_meta, '$.gitBranch') AS provider_meta_git_branch,
    json_extract(provider_meta, '$.git') AS provider_meta_git,
    json_extract(provider_meta, '$.total_cost_usd') AS provider_meta_total_cost_usd,
    json_extract(provider_meta, '$.total_duration_ms') AS provider_meta_total_duration_ms,
    json_extract(provider_meta, '$.models_used') AS provider_meta_models_used
FROM sessions
WHERE session_id IN ({placeholders})
"""
_SESSION_INSIGHT_MESSAGE_SQL_TEMPLATE = """
SELECT
    message_id,
    session_id,
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
    source_name,
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
WHERE session_id IN ({placeholders})
ORDER BY session_id, sort_key, message_id
"""
_SESSION_INSIGHT_BLOCK_SQL_TEMPLATE = """
SELECT
    block_id,
    message_id,
    session_id,
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
WHERE session_id IN ({placeholders})
  AND message_id IN (
      SELECT message_id
      FROM messages
      WHERE session_id IN ({placeholders})
        AND (
            has_tool_use != 0
            OR has_thinking != 0
            OR message_type IN ('tool_use', 'tool_result', 'thinking')
        )
  )
ORDER BY session_id, message_id, block_index
"""


@dataclass(slots=True)
class SessionInsightArchiveBatch:
    sessions: list[SessionRecord]
    messages: list[MessageRecord]
    attachments_by_session: dict[str, list[AttachmentRecord]]
    provider_events_by_session: dict[str, list[ProviderEventRecord]]
    compaction_counts_by_session: dict[str, int]
    blocks: list[ContentBlockRecord]


@dataclass(slots=True)
class SessionInsightRecordBundle:
    profile_record: SessionProfileRecord
    latency_profile_record: SessionLatencyProfileRecord
    work_event_records: list[SessionWorkEventRecord]
    phase_records: list[SessionPhaseRecord]
    repo_observations: tuple[object, ...] = ()
    """Repo identity observations for ``session_repo_observations`` (#1253).

    Typed as ``RepoObservation`` from
    ``polylogue.storage.insights.session.repo_identity``; kept as
    ``object`` here to avoid a circular import at module load time.
    """

    @property
    def session_id(self) -> SessionId:
        return self.profile_record.session_id

    @property
    def work_event_count(self) -> int:
        return len(self.work_event_records)

    @property
    def phase_count(self) -> int:
        return len(self.phase_records)


@dataclass(slots=True, frozen=True)
class _SessionInsightRebuildChunk:
    session_ids: tuple[str, ...]
    estimated_message_count: int
    max_estimated_session_messages: int


def iter_session_id_pages_sync(
    conn: sqlite3.Connection,
    *,
    page_size: int,
) -> Iterable[list[str]]:
    cursor = conn.execute(_ALL_SESSION_IDS_SQL)
    while True:
        rows = cursor.fetchmany(page_size)
        if not rows:
            break
        yield [str(row["session_id"]) for row in rows]


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


async def iter_session_id_pages_async(
    conn: aiosqlite.Connection,
    *,
    page_size: int,
) -> AsyncIterator[list[str]]:
    cursor = await conn.execute(_ALL_SESSION_IDS_SQL)
    while True:
        rows = list(await cursor.fetchmany(page_size))
        if not rows:
            break
        yield [str(row["session_id"]) for row in rows]


def _row_to_session_insight_session(row: sqlite3.Row) -> SessionRecord:
    provider_meta: dict[str, object] = {}
    cwd_value = _row_get(row, "provider_meta_cwd")
    if isinstance(cwd_value, str) and cwd_value:
        provider_meta["cwd"] = cwd_value
    git_branch = _row_get(row, "provider_meta_git_branch")
    if isinstance(git_branch, str) and git_branch:
        provider_meta["gitBranch"] = git_branch
    git_raw = _row_get(row, "provider_meta_git")
    if isinstance(git_raw, str) and git_raw:
        parsed_git = _parse_json(git_raw, field="provider_meta.git", record_id=row["session_id"])
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
        parsed_models = _parse_json(models_used, field="provider_meta.models_used", record_id=row["session_id"])
        if isinstance(parsed_models, list):
            provider_meta["models_used"] = [item for item in parsed_models if isinstance(item, str)]
    parent_session_id = _row_text(row, "parent_session_id")
    branch_type = _row_text(row, "branch_type")

    return SessionRecord(
        session_id=row["session_id"],
        source_name=row["source_name"],
        provider_session_id=row["provider_session_id"],
        title=row["title"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        sort_key=_row_float(row, "sort_key"),
        content_hash=row["content_hash"],
        provider_meta=provider_meta or None,
        metadata=_json_object(_parse_json(row["metadata"], field="metadata", record_id=row["session_id"])),
        version=row["version"],
        parent_session_id=SessionId(parent_session_id) if parent_session_id is not None else None,
        branch_type=BranchType(branch_type) if branch_type is not None else None,
        raw_id=_row_text(row, "raw_id"),
    )


def _load_message_counts_sync(
    conn: sqlite3.Connection,
    session_ids: Sequence[str],
) -> dict[str, int]:
    if not session_ids:
        return {}
    placeholders = ", ".join("?" for _ in session_ids)
    rows = conn.execute(
        f"""
        SELECT session_id, message_count
        FROM session_stats
        WHERE session_id IN ({placeholders})
        """,
        tuple(session_ids),
    ).fetchall()
    return {str(row["session_id"]): int(row["message_count"] or 0) for row in rows}


def _chunk_session_ids_by_message_budget_sync(
    session_ids: Sequence[str],
    *,
    message_counts: dict[str, int],
    page_size: int,
    message_budget: int,
) -> list[_SessionInsightRebuildChunk]:
    chunks: list[_SessionInsightRebuildChunk] = []
    current_chunk: list[str] = []
    current_messages = 0
    current_max_messages = 0

    for session_id in session_ids:
        estimated_messages = max(int(message_counts.get(session_id, 0) or 0), 1)
        if current_chunk and (
            len(current_chunk) >= page_size or current_messages + estimated_messages > message_budget
        ):
            chunks.append(
                _SessionInsightRebuildChunk(
                    session_ids=tuple(current_chunk),
                    estimated_message_count=current_messages,
                    max_estimated_session_messages=current_max_messages,
                )
            )
            current_chunk = []
            current_messages = 0
            current_max_messages = 0
        current_chunk.append(session_id)
        current_messages += estimated_messages
        current_max_messages = max(current_max_messages, estimated_messages)

    if current_chunk:
        chunks.append(
            _SessionInsightRebuildChunk(
                session_ids=tuple(current_chunk),
                estimated_message_count=current_messages,
                max_estimated_session_messages=current_max_messages,
            )
        )
    return chunks


def sync_attachment_batch(
    conn: sqlite3.Connection,
    session_ids: Sequence[str],
) -> dict[str, list[AttachmentRecord]]:
    if not session_ids:
        return {}
    placeholders = ", ".join("?" for _ in session_ids)
    rows = conn.execute(
        f"""
        SELECT a.*, r.message_id, r.session_id
        FROM attachments a
        JOIN attachment_refs r ON a.attachment_id = r.attachment_id
        WHERE r.session_id IN ({placeholders})
        """,
        tuple(session_ids),
    ).fetchall()
    result: dict[str, list[AttachmentRecord]] = {session_id: [] for session_id in session_ids}
    for row in rows:
        session_id = str(row["session_id"])
        result.setdefault(session_id, []).append(
            AttachmentRecord(
                attachment_id=row["attachment_id"],
                session_id=SessionId(session_id),
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
    session_ids: Sequence[str],
) -> SessionInsightArchiveBatch:
    placeholders = ", ".join("?" for _ in session_ids)
    sessions = [
        _row_to_session_insight_session(row)
        for row in conn.execute(
            _SESSION_INSIGHT_SESSION_SQL_TEMPLATE.format(placeholders=placeholders),
            tuple(session_ids),
        ).fetchall()
    ]
    messages = [
        _row_to_message(row)
        for row in conn.execute(
            _SESSION_INSIGHT_MESSAGE_SQL_TEMPLATE.format(placeholders=placeholders),
            (_SESSION_INSIGHT_MESSAGE_TEXT_PREVIEW_CHARS, *session_ids),
        ).fetchall()
    ]
    blocks = [
        _row_to_content_block(row)
        for row in conn.execute(
            _SESSION_INSIGHT_BLOCK_SQL_TEMPLATE.format(placeholders=placeholders),
            (_SESSION_INSIGHT_BLOCK_TEXT_PREVIEW_CHARS, *session_ids, *session_ids),
        ).fetchall()
    ]
    return SessionInsightArchiveBatch(
        sessions=sessions,
        messages=messages,
        attachments_by_session=sync_attachment_batch(conn, session_ids),
        provider_events_by_session=sync_provider_events_batch(conn, session_ids),
        compaction_counts_by_session=sync_provider_event_compaction_counts(conn, session_ids),
        blocks=blocks,
    )


async def load_async_batch(
    conn: aiosqlite.Connection,
    session_ids: Sequence[str],
) -> SessionInsightArchiveBatch:
    placeholders = ", ".join("?" for _ in session_ids)
    sessions = [
        _row_to_session_insight_session(row)
        for row in await (
            await conn.execute(
                _SESSION_INSIGHT_SESSION_SQL_TEMPLATE.format(placeholders=placeholders),
                tuple(session_ids),
            )
        ).fetchall()
    ]
    messages = [
        _row_to_message(row)
        for row in await (
            await conn.execute(
                _SESSION_INSIGHT_MESSAGE_SQL_TEMPLATE.format(placeholders=placeholders),
                (_SESSION_INSIGHT_MESSAGE_TEXT_PREVIEW_CHARS, *session_ids),
            )
        ).fetchall()
    ]
    blocks = [
        _row_to_content_block(row)
        for row in await (
            await conn.execute(
                _SESSION_INSIGHT_BLOCK_SQL_TEMPLATE.format(placeholders=placeholders),
                (_SESSION_INSIGHT_BLOCK_TEXT_PREVIEW_CHARS, *session_ids, *session_ids),
            )
        ).fetchall()
    ]
    attachments = await get_attachments_batch(conn, list(session_ids))
    provider_events = await get_provider_events_batch(conn, list(session_ids))
    compaction_counts = await get_provider_event_compaction_counts(conn, list(session_ids))
    return SessionInsightArchiveBatch(
        sessions=sessions,
        messages=messages,
        attachments_by_session=attachments,
        provider_events_by_session=provider_events,
        compaction_counts_by_session=compaction_counts,
        blocks=blocks,
    )


def hydrate_sessions(
    batch: SessionInsightArchiveBatch,
) -> list[Session]:
    messages_by_session: dict[str, list[MessageRecord]] = defaultdict(list)
    blocks_by_session: dict[str, list[ContentBlockRecord]] = defaultdict(list)
    for message in batch.messages:
        messages_by_session[str(message.session_id)].append(message)
    for block in batch.blocks:
        blocks_by_session[str(block.session_id)].append(block)

    hydrated: list[Session] = []
    for session in batch.sessions:
        session_id = str(session.session_id)
        attached_messages = attach_blocks_to_messages(
            messages_by_session.get(session_id, []),
            blocks_by_session.get(session_id, []),
        )
        hydrated.append(
            session_from_records(
                session,
                attached_messages,
                batch.attachments_by_session.get(session_id, []),
                batch.provider_events_by_session.get(session_id, []),
            )
        )
    return hydrated


def build_session_insight_records(
    session: Session,
    *,
    compaction_count: int | None = None,
    logical_session_id: str | None = None,
) -> SessionInsightRecordBundle:
    from polylogue.storage.insights.session.repo_identity import attribution_to_observations

    analysis = build_session_analysis(session)
    profile = build_session_profile(session, analysis=analysis, compaction_count=compaction_count)
    materialized_at = now_iso()
    provider_meta: dict[str, object] = session.provider_meta if isinstance(session.provider_meta, dict) else {}
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
    latency_facts = build_latency_profile_facts(session, profile)
    return SessionInsightRecordBundle(
        profile_record=build_session_profile_record(
            profile,
            analysis=analysis,
            logical_session_id=logical_session_id,
            materialized_at=materialized_at,
        ),
        latency_profile_record=build_session_latency_profile_record(
            session,
            profile,
            latency_facts,
            materialized_at=materialized_at,
        ),
        work_event_records=build_session_work_event_records(profile, materialized_at=materialized_at),
        phase_records=build_session_phase_records(profile, materialized_at=materialized_at),
        repo_observations=repo_observations,
    )


def build_session_insight_record_bundles(
    sessions: Iterable[Session],
    *,
    compaction_counts_by_session: dict[str, int] | None = None,
    logical_session_ids_by_session: dict[str, str] | None = None,
) -> list[SessionInsightRecordBundle]:
    compaction_counts = compaction_counts_by_session or {}
    logical_ids = logical_session_ids_by_session or {}
    return [
        build_session_insight_records(
            session,
            compaction_count=compaction_counts.get(str(session.id)),
            logical_session_id=logical_ids.get(str(session.id)),
        )
        for session in sessions
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
) -> SessionInsightCounts:
    return SessionInsightCounts(
        profiles=profiles,
        work_events=work_events,
        phases=phases,
        threads=threads,
        tag_rollups=tag_rollups,
    )


def _session_profile_records_for_session_ids_sync(
    conn: sqlite3.Connection,
    session_ids: Sequence[str],
) -> list[SessionProfileRecord]:
    if not session_ids:
        return []
    placeholders = ", ".join("?" for _ in session_ids)
    rows = conn.execute(
        f"SELECT * FROM session_profiles WHERE session_id IN ({placeholders})",
        tuple(session_ids),
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


def _delete_tables_with_progress_sync(
    conn: sqlite3.Connection,
    *,
    tables: Sequence[str],
    progress_callback: ProgressCallback | None,
) -> None:
    """Delete every row in ``tables`` while emitting one progress event per table.

    The progress event's ``desc`` names the table so an operator log can tell
    "stuck on session_work_events" from "stuck on session_profiles". The
    ``amount`` argument is the number of rows deleted; callers that only care
    about table-step granularity can ignore it.
    """
    for table in tables:
        deleted = conn.execute(f"DELETE FROM {table}").rowcount
        if progress_callback is not None:
            progress_callback(
                int(max(deleted, 0)),
                desc=f"rebuild: cleared {table}",
            )


def rebuild_session_insights_sync(
    conn: sqlite3.Connection,
    *,
    session_ids: Sequence[str] | None = None,
    page_size: int = _SESSION_INSIGHT_REBUILD_PAGE_SIZE,
    progress_callback: ProgressCallback | None = None,
    progress_total: int | None = None,
) -> SessionInsightCounts:
    session_chunks: Iterable[Sequence[str]]
    previous_profile_groups: set[tuple[str, str]] = set()
    if session_ids is None:
        # #1607: the seven DELETEs hold the write lock for seconds-to-minutes
        # at archive scale and were previously silent. Emit a progress
        # heartbeat per table so the operator sees forward motion instead
        # of "polylogue qa --rebuild-insights" hanging with no output.
        # The full rebuild stays inside one transaction (the implicit tx
        # started by the first DELETE spans every subsequent DML through
        # the final ``conn.commit()`` at the bottom of this function), so
        # a SIGKILL mid-rebuild rolls the WAL back to the prior state on
        # the next open — the prior insights are intact, not emptied.
        _delete_tables_with_progress_sync(
            conn,
            tables=(
                "session_work_events",
                "session_phases",
                "session_latency_profiles",
                "session_profiles",
                "session_tag_rollups",
                "session_repo_observations",
                "repo_identities",
            ),
            progress_callback=progress_callback,
        )
        session_chunks = iter_session_id_pages_sync(conn, page_size=page_size)
    else:
        session_ids = tuple(dict.fromkeys(str(session_id) for session_id in session_ids))
        previous_profile_groups = {
            group
            for record in _session_profile_records_for_session_ids_sync(conn, session_ids)
            if (group := profile_provider_day(record)) is not None
        }
        message_counts = _load_message_counts_sync(conn, session_ids)
        session_chunks = (
            chunk.session_ids
            for chunk in _chunk_session_ids_by_message_budget_sync(
                session_ids,
                message_counts=message_counts,
                page_size=page_size,
                message_budget=_SESSION_INSIGHT_REBUILD_MESSAGE_BUDGET,
            )
        )
    if session_ids is not None and not session_ids:
        conn.commit()
        return _empty_rebuild_counts()

    profile_count = 0
    work_event_count = 0
    phase_count = 0
    refreshed_profile_groups: set[tuple[str, str]] = set()
    saw_session_ids = False
    for chunk in session_chunks:
        saw_session_ids = True
        batch = load_sync_batch(conn, chunk)
        root_ids_by_session = thread_root_ids_sync(conn, chunk)
        record_bundles = build_session_insight_record_bundles(
            hydrate_sessions(batch),
            compaction_counts_by_session=batch.compaction_counts_by_session,
            logical_session_ids_by_session=root_ids_by_session,
        )
        chunk_profiles, chunk_work_events, chunk_phases = _count_record_bundles(record_bundles)
        replace_session_profiles_bulk_sync(conn, [bundle.profile_record for bundle in record_bundles])
        replace_session_latency_profiles_bulk_sync(
            conn,
            [bundle.latency_profile_record for bundle in record_bundles],
        )
        replace_session_work_events_bulk_sync(
            conn,
            {bundle.session_id: bundle.work_event_records for bundle in record_bundles},
        )
        replace_session_phases_bulk_sync(
            conn,
            {bundle.session_id: bundle.phase_records for bundle in record_bundles},
        )
        from polylogue.storage.insights.session.repo_identity import (
            RepoObservation as _RepoObservationSync,
        )
        from polylogue.storage.insights.session.repo_identity import (
            refresh_session_repo_observations_sync as _refresh_repo_obs_sync,
        )

        for bundle in record_bundles:
            _refresh_repo_obs_sync(
                conn,
                str(bundle.session_id),
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
    if not saw_session_ids:
        if session_ids is None:
            conn.execute("DELETE FROM work_threads")
            conn.execute("DELETE FROM session_phases")
            conn.execute("DELETE FROM session_tag_rollups")
        conn.commit()
        return _empty_rebuild_counts()

    if session_ids is not None:
        affected_roots = tuple(thread_root_ids_sync(conn, session_ids).values())
        thread_count = _refresh_thread_roots_sync(conn, affected_roots)
        refresh_sync_provider_day_aggregates(
            conn,
            previous_profile_groups | refreshed_profile_groups,
        )
        tag_rollup_count = conn.execute("SELECT COUNT(*) FROM session_tag_rollups").fetchone()[0]
        conn.commit()
        return _finalize_rebuild_counts(
            profiles=profile_count,
            work_events=work_event_count,
            phases=phase_count,
            threads=thread_count,
            tag_rollups=int(tag_rollup_count),
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
    conn.execute("DELETE FROM session_tag_rollups")
    provider_day_groups = set(list_sync_provider_day_groups(conn))
    refresh_sync_provider_day_aggregates(conn, provider_day_groups)
    tag_rollup_count = conn.execute("SELECT COUNT(*) FROM session_tag_rollups").fetchone()[0]
    conn.commit()
    return _finalize_rebuild_counts(
        profiles=profile_count,
        work_events=work_event_count,
        phases=phase_count,
        threads=thread_count,
        tag_rollups=int(tag_rollup_count),
    )


async def rebuild_session_insights_async(
    conn: aiosqlite.Connection,
    *,
    session_ids: Sequence[str] | None = None,
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

    if session_ids is None:
        # See _delete_tables_with_progress_sync for the sync analogue and
        # the #1607 context. The implicit transaction spans every DML
        # through the eventual COMMIT, so SIGKILL mid-rebuild rolls back
        # to prior state on next open.
        for table in (
            "session_work_events",
            "session_phases",
            "session_latency_profiles",
            "session_profiles",
            "session_tag_rollups",
            "session_repo_observations",
            "repo_identities",
        ):
            cursor = await conn.execute(f"DELETE FROM {table}")
            if progress_callback is not None:
                rowcount = getattr(cursor, "rowcount", 0) or 0
                progress_callback(int(rowcount if rowcount > 0 else 0), desc=f"rebuild: cleared {table}")
    elif not session_ids:
        await conn.execute("DELETE FROM work_threads")
        await conn.execute("DELETE FROM session_phases")
        await conn.execute("DELETE FROM session_latency_profiles")
        await conn.execute("DELETE FROM session_tag_rollups")
        return _empty_rebuild_counts()

    profile_count = 0
    work_event_count = 0
    phase_count = 0
    if session_ids is None:
        async for chunk in iter_session_id_pages_async(conn, page_size=page_size):
            batch = await load_async_batch(conn, chunk)
            root_ids_by_session = await thread_root_ids_async(conn, chunk)
            record_bundles = build_session_insight_record_bundles(
                hydrate_sessions(batch),
                compaction_counts_by_session=batch.compaction_counts_by_session,
                logical_session_ids_by_session=root_ids_by_session,
            )
            chunk_profiles, chunk_work_events, chunk_phases = _count_record_bundles(record_bundles)
            for bundle in record_bundles:
                await replace_session_profile(conn, bundle.profile_record, transaction_depth)
                await replace_session_latency_profile(conn, bundle.latency_profile_record, transaction_depth)
                await replace_session_work_events(
                    conn,
                    bundle.session_id,
                    bundle.work_event_records,
                    transaction_depth,
                )
                await replace_session_phases(
                    conn,
                    bundle.session_id,
                    bundle.phase_records,
                    transaction_depth,
                )
                from polylogue.storage.insights.session.repo_identity import (
                    RepoObservation as _RepoObservation,
                )
                from polylogue.storage.insights.session.repo_identity import (
                    refresh_session_repo_observations as _refresh_repo_obs,
                )

                await _refresh_repo_obs(
                    conn,
                    str(bundle.session_id),
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
        for chunk_ids in chunked(list(session_ids), size=page_size):
            batch = await load_async_batch(conn, chunk_ids)
            root_ids_by_session = await thread_root_ids_async(conn, chunk_ids)
            record_bundles = build_session_insight_record_bundles(
                hydrate_sessions(batch),
                compaction_counts_by_session=batch.compaction_counts_by_session,
                logical_session_ids_by_session=root_ids_by_session,
            )
            chunk_profiles, chunk_work_events, chunk_phases = _count_record_bundles(record_bundles)
            for bundle in record_bundles:
                await replace_session_profile(conn, bundle.profile_record, transaction_depth)
                await replace_session_latency_profile(conn, bundle.latency_profile_record, transaction_depth)
                await replace_session_work_events(
                    conn,
                    bundle.session_id,
                    bundle.work_event_records,
                    transaction_depth,
                )
                await replace_session_phases(
                    conn,
                    bundle.session_id,
                    bundle.phase_records,
                    transaction_depth,
                )
                from polylogue.storage.insights.session.repo_identity import (
                    RepoObservation as _RepoObservation,
                )
                from polylogue.storage.insights.session.repo_identity import (
                    refresh_session_repo_observations as _refresh_repo_obs,
                )

                await _refresh_repo_obs(
                    conn,
                    str(bundle.session_id),
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
    await conn.execute("DELETE FROM session_tag_rollups")
    provider_day_groups = set(await list_async_provider_day_groups(conn))
    await refresh_async_provider_day_aggregates(
        conn,
        provider_day_groups,
        transaction_depth=transaction_depth,
    )
    tag_rollup_row = await (await conn.execute("SELECT COUNT(*) FROM session_tag_rollups")).fetchone()
    tag_rollup_count = int(tag_rollup_row[0]) if tag_rollup_row is not None else 0
    return _finalize_rebuild_counts(
        profiles=profile_count,
        work_events=work_event_count,
        phases=phase_count,
        threads=thread_count,
        tag_rollups=int(tag_rollup_count),
    )


__all__ = [
    "_ALL_SESSION_IDS_SQL",
    "_ALL_SESSION_PROFILE_ROWS_SQL",
    "SessionInsightArchiveBatch",
    "SessionInsightRecordBundle",
    "build_session_insight_record_bundles",
    "build_session_insight_records",
    "chunked",
    "hydrate_sessions",
    "iter_session_id_pages_async",
    "iter_session_id_pages_sync",
    "iter_hydrated_session_profiles_sync",
    "load_async_batch",
    "load_sync_batch",
    "rebuild_session_insights_async",
    "rebuild_session_insights_sync",
    "sync_attachment_batch",
]
