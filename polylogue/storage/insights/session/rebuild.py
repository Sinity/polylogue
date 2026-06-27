"""Batch loading, hydration, and full rebuild flows for durable session-insight read models."""

from __future__ import annotations

import sqlite3
import time
from collections import defaultdict
from collections.abc import AsyncIterator, Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass

import aiosqlite

from polylogue.archive.session.branch_type import BranchType
from polylogue.archive.session.domain_models import Session
from polylogue.archive.session.session_profile import SessionProfile, build_session_analysis, build_session_profile

# Re-export canonical chunked from polylogue.core.common.
from polylogue.core.common import chunked
from polylogue.core.enums import SessionKind
from polylogue.core.memory import release_process_memory
from polylogue.protocols import ProgressCallback
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
from polylogue.storage.insights.session.run_projection_rows import (
    build_session_context_snapshot_records,
    build_session_observed_event_records,
    build_session_run_records,
)
from polylogue.storage.insights.session.runtime import SessionInsightCounts
from polylogue.storage.insights.session.storage import (
    _epoch_ms_or_none,
    replace_session_context_snapshots_bulk_sync,
    replace_session_latency_profiles_bulk_sync,
    replace_session_observed_events_bulk_sync,
    replace_session_phases_bulk_sync,
    replace_session_profiles_bulk_sync,
    replace_session_runs_bulk_sync,
    replace_session_work_events_bulk_sync,
    replace_threads_bulk_sync,
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
    BlockRecord,
    MessageRecord,
    SessionContextSnapshotRecord,
    SessionEventRecord,
    SessionLatencyProfileRecord,
    SessionObservedEventRecord,
    SessionPhaseRecord,
    SessionProfileRecord,
    SessionRecord,
    SessionRunRecord,
    SessionWorkEventRecord,
    ThreadRecord,
)
from polylogue.storage.runtime.store_constants import SESSION_INSIGHT_MATERIALIZER_VERSION
from polylogue.storage.sqlite.queries.attachments import get_attachments_batch
from polylogue.storage.sqlite.queries.mappers import (
    _json_object,
    _parse_json,
    _row_float,
    _row_text,
    _row_to_content_block,
    _row_to_message,
    _row_to_session_profile_record,
)
from polylogue.storage.sqlite.queries.session_events import (
    get_session_event_compaction_counts,
    get_session_events_batch,
    sync_session_event_compaction_counts,
    sync_session_events_batch,
)
from polylogue.types import SessionId

_ALL_SESSION_IDS_SQL = "SELECT session_id FROM sessions ORDER BY COALESCE(sort_key_ms, 0) DESC, session_id"
_ALL_SESSION_PROFILE_ROWS_SQL = """
SELECT *
FROM session_profiles
ORDER BY COALESCE(source_sort_key, 0) DESC, session_id
"""
# Full rebuilds must tolerate very large session payloads without letting a
# single chunk inflate RSS into multi-GB territory. The message-budget chunker
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
    origin,
    native_id,
    title,
    session_kind,
    CASE WHEN created_at_ms IS NULL THEN NULL ELSE strftime('%Y-%m-%dT%H:%M:%f+00:00', created_at_ms / 1000.0, 'unixepoch') END AS created_at,
    CASE WHEN updated_at_ms IS NULL THEN NULL ELSE strftime('%Y-%m-%dT%H:%M:%f+00:00', updated_at_ms / 1000.0, 'unixepoch') END AS updated_at,
    CAST(sort_key_ms AS REAL) / 1000.0 AS sort_key,
    hex(content_hash) AS content_hash,
    '{{}}' AS metadata,
    1 AS version,
    parent_session_id,
    branch_type,
    raw_id,
    (SELECT json_group_array(path) FROM session_working_dirs swd WHERE swd.session_id = sessions.session_id ORDER BY position) AS working_directories_json,
    git_branch,
    git_repository_url
FROM sessions
WHERE session_id IN ({placeholders})
"""
_SESSION_INSIGHT_MESSAGE_SQL_TEMPLATE = """
SELECT
    m.message_id,
    m.session_id,
    m.native_id AS provider_message_id,
    m.role,
    (
        SELECT substr(b.text, 1, ?)
        FROM blocks b
        WHERE b.message_id = m.message_id
          AND b.block_type = 'text'
          AND b.text IS NOT NULL
        ORDER BY b.position
        LIMIT 1
    ) AS text,
    CAST(m.occurred_at_ms AS REAL) / 1000.0 AS sort_key,
    hex(m.content_hash) AS content_hash,
    1 AS version,
    m.parent_message_id,
    m.variant_index AS branch_index,
    s.origin AS source_name,
    m.word_count,
    m.has_tool_use,
    m.has_thinking,
    m.has_paste,
    m.input_tokens,
    m.output_tokens,
    m.cache_read_tokens,
    m.cache_write_tokens,
    m.model_name,
    m.message_type
FROM messages m
JOIN sessions s ON s.session_id = m.session_id
WHERE m.session_id IN ({placeholders})
ORDER BY m.session_id, m.position, m.message_id
"""
_SESSION_INSIGHT_BLOCK_SQL_TEMPLATE = """
SELECT
    block_id,
    message_id,
    session_id,
    position AS block_index,
    block_type AS type,
    CASE
        WHEN text IS NULL THEN NULL
        ELSE substr(text, 1, ?)
    END AS text,
    tool_name,
    tool_id,
    tool_input,
    NULL AS metadata,
    semantic_type
FROM blocks
WHERE session_id IN ({placeholders})
  AND block_type != 'text'
ORDER BY session_id, message_id, position
"""


@dataclass(slots=True)
class SessionInsightArchiveBatch:
    sessions: list[SessionRecord]
    messages: list[MessageRecord]
    attachments_by_session: dict[str, list[AttachmentRecord]]
    session_events_by_session: dict[str, list[SessionEventRecord]]
    compaction_counts_by_session: dict[str, int]
    blocks: list[BlockRecord]


@dataclass(slots=True)
class SessionInsightRecordBundle:
    profile_record: SessionProfileRecord
    latency_profile_record: SessionLatencyProfileRecord
    work_event_records: list[SessionWorkEventRecord]
    phase_records: list[SessionPhaseRecord]
    run_records: list[SessionRunRecord]
    observed_event_records: list[SessionObservedEventRecord]
    context_snapshot_records: list[SessionContextSnapshotRecord]
    repo_observations: tuple[object, ...] = ()
    """Repo observations for ``session_repos`` (#1253).

    Typed as ``RepoObservation`` from
    ``polylogue.storage.insights.session.repo_observations``; kept as
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

    @property
    def run_count(self) -> int:
        return len(self.run_records)

    @property
    def observed_event_count(self) -> int:
        return len(self.observed_event_records)

    @property
    def context_snapshot_count(self) -> int:
        return len(self.context_snapshot_records)


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
    parent_session_id = _row_text(row, "parent_session_id")
    branch_type = _row_text(row, "branch_type")

    return SessionRecord(
        session_id=row["session_id"],
        origin=row["origin"],
        native_id=row["native_id"],
        title=row["title"],
        session_kind=SessionKind.normalize(_row_text(row, "session_kind")),
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        sort_key=_row_float(row, "sort_key"),
        content_hash=row["content_hash"],
        metadata=_json_object(_parse_json(row["metadata"], field="metadata", record_id=row["session_id"])),
        version=row["version"],
        parent_session_id=SessionId(parent_session_id) if parent_session_id is not None else None,
        branch_type=BranchType(branch_type) if branch_type is not None else None,
        raw_id=_row_text(row, "raw_id"),
        working_directories_json=_row_text(row, "working_directories_json"),
        git_branch=_row_text(row, "git_branch"),
        git_repository_url=_row_text(row, "git_repository_url"),
    )


def attach_blocks_to_messages(
    messages: Sequence[MessageRecord],
    content_blocks: Sequence[BlockRecord],
) -> list[MessageRecord]:
    grouped: dict[str, list[BlockRecord]] = defaultdict(list)
    for block in content_blocks:
        grouped[str(block.message_id)].append(block)
    return [
        message.model_copy(update={"blocks": list(grouped.get(str(message.message_id), []))}) for message in messages
    ]


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
        FROM sessions
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
        SELECT
            a.attachment_id,
            a.display_name,
            a.media_type AS mime_type,
            a.byte_count AS size_bytes,
            COALESCE(r.source_url, a.display_name) AS path,
            r.source_url,
            r.caption,
            r.upload_origin,
            r.message_id,
            r.session_id
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
                display_name=row["display_name"],
                source_url=row["source_url"],
                caption=row["caption"],
                upload_origin=row["upload_origin"],
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
            (_SESSION_INSIGHT_BLOCK_TEXT_PREVIEW_CHARS, *session_ids),
        ).fetchall()
    ]
    return SessionInsightArchiveBatch(
        sessions=sessions,
        messages=messages,
        attachments_by_session=sync_attachment_batch(conn, session_ids),
        session_events_by_session=sync_session_events_batch(conn, session_ids),
        compaction_counts_by_session=sync_session_event_compaction_counts(conn, session_ids),
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
                (_SESSION_INSIGHT_BLOCK_TEXT_PREVIEW_CHARS, *session_ids),
            )
        ).fetchall()
    ]
    attachments = await get_attachments_batch(conn, list(session_ids))
    session_events = await get_session_events_batch(conn, list(session_ids))
    compaction_counts = await get_session_event_compaction_counts(conn, list(session_ids))
    return SessionInsightArchiveBatch(
        sessions=sessions,
        messages=messages,
        attachments_by_session=attachments,
        session_events_by_session=session_events,
        compaction_counts_by_session=compaction_counts,
        blocks=blocks,
    )


def hydrate_sessions(
    batch: SessionInsightArchiveBatch,
) -> list[Session]:
    messages_by_session: dict[str, list[MessageRecord]] = defaultdict(list)
    blocks_by_session: dict[str, list[BlockRecord]] = defaultdict(list)
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
                batch.session_events_by_session.get(session_id, []),
            )
        )
    return hydrated


def build_session_insight_records(
    session: Session,
    *,
    compaction_count: int | None = None,
    logical_session_id: str | None = None,
    stage_timing_add: Callable[[str, float], None] | None = None,
) -> SessionInsightRecordBundle:
    from polylogue.storage.insights.session.repo_observations import attribution_to_observations

    def add_timing(name: str, started_at: float) -> None:
        if stage_timing_add is not None:
            stage_timing_add(name, started_at)

    t0 = time.perf_counter()
    analysis = build_session_analysis(session, stage_timing_add=stage_timing_add)
    add_timing("build_records.analysis", t0)
    t0 = time.perf_counter()
    profile = build_session_profile(
        session,
        analysis=analysis,
        compaction_count=compaction_count,
        stage_timing_add=stage_timing_add,
    )
    add_timing("build_records.profile", t0)
    t0 = time.perf_counter()
    materialized_at = now_iso()
    repo_observations = attribution_to_observations(
        analysis.attribution,
        git_repository_url=session.git_repository_url,
    )
    add_timing("build_records.repo_observations", t0)
    t0 = time.perf_counter()
    latency_facts = build_latency_profile_facts(session, profile)
    add_timing("build_records.latency_facts", t0)
    t0 = time.perf_counter()
    profile_record = build_session_profile_record(
        profile,
        analysis=analysis,
        logical_session_id=logical_session_id,
        materialized_at=materialized_at,
    )
    add_timing("build_records.profile_record", t0)
    t0 = time.perf_counter()
    latency_profile_record = build_session_latency_profile_record(
        session,
        profile,
        latency_facts,
        materialized_at=materialized_at,
    )
    add_timing("build_records.latency_profile_record", t0)
    t0 = time.perf_counter()
    work_event_records = build_session_work_event_records(profile, materialized_at=materialized_at)
    add_timing("build_records.work_event_records", t0)
    t0 = time.perf_counter()
    phase_records = build_session_phase_records(profile, materialized_at=materialized_at)
    add_timing("build_records.phase_records", t0)
    t0 = time.perf_counter()
    # The run projection is computed from the same hydrated Session via the
    # recovery digest, with no cross-session links (session_links=()), so the
    # materialized rows match the runtime query path exactly. compile_recovery_digest
    # always yields a main run, so RunProjection's ">=1 run" invariant holds for
    # every session, including empty ones. A projection failure must surface, not
    # be swallowed, so the rebuild fails loudly on malformed evidence.
    from polylogue.insights.transforms import compile_recovery_digest

    run_projection = compile_recovery_digest(session, session_links=()).run_projection
    source_updated_at = profile_record.source_updated_at
    run_records = build_session_run_records(
        run_projection,
        materialized_at=materialized_at,
        source_updated_at=source_updated_at,
    )
    observed_event_records = build_session_observed_event_records(
        run_projection,
        materialized_at=materialized_at,
        source_updated_at=source_updated_at,
    )
    context_snapshot_records = build_session_context_snapshot_records(
        run_projection,
        materialized_at=materialized_at,
        source_updated_at=source_updated_at,
    )
    add_timing("build_records.run_projection_records", t0)
    return SessionInsightRecordBundle(
        profile_record=profile_record,
        latency_profile_record=latency_profile_record,
        work_event_records=work_event_records,
        phase_records=phase_records,
        run_records=run_records,
        observed_event_records=observed_event_records,
        context_snapshot_records=context_snapshot_records,
        repo_observations=repo_observations,
    )


def build_session_insight_record_bundles(
    sessions: Iterable[Session],
    *,
    compaction_counts_by_session: dict[str, int] | None = None,
    logical_session_ids_by_session: dict[str, str] | None = None,
    stage_timing_add: Callable[[str, float], None] | None = None,
) -> list[SessionInsightRecordBundle]:
    compaction_counts = compaction_counts_by_session or {}
    logical_ids = logical_session_ids_by_session or {}
    return [
        build_session_insight_records(
            session,
            compaction_count=compaction_counts.get(str(session.id)),
            logical_session_id=logical_ids.get(str(session.id)),
            stage_timing_add=stage_timing_add,
        )
        for session in sessions
    ]


def _stamp_bundle_materialization(conn: sqlite3.Connection, bundle: SessionInsightRecordBundle) -> None:
    """Stamp insight_materialization for one rebuilt session bundle.

    The canonical bulk rebuild and the api archive rebuild both materialize the
    same per-session insight bundle; stamping here keeps insight_materialization
    populated on the daemon convergence path (not only the api path), so
    materialization tracking is coherent regardless of which entrypoint ran.
    Uses the no-commit primitive so the stamp participates in the rebuild's
    single transaction.
    """
    from polylogue.storage.sqlite.archive_tiers.write import apply_insight_materialization

    profile = bundle.profile_record
    session_id = str(profile.session_id)
    materialized_at_ms = _epoch_ms_or_none(profile.materialized_at) or 0
    source_updated_at_ms = _epoch_ms_or_none(profile.source_updated_at)
    source_sort_key_ms = int(profile.source_sort_key * 1000) if profile.source_sort_key is not None else None
    input_high_water_mark_ms = _epoch_ms_or_none(profile.input_high_water_mark)
    for insight_type, materializer_version, input_row_count in (
        ("session_profile", profile.materializer_version, profile.input_row_count),
        ("latency", profile.materializer_version, bundle.latency_profile_record.input_row_count),
        ("work_events", SESSION_INSIGHT_MATERIALIZER_VERSION, len(bundle.work_event_records)),
        ("phases", SESSION_INSIGHT_MATERIALIZER_VERSION, len(bundle.phase_records)),
        ("runs", SESSION_INSIGHT_MATERIALIZER_VERSION, len(bundle.run_records)),
        ("observed_events", SESSION_INSIGHT_MATERIALIZER_VERSION, len(bundle.observed_event_records)),
        ("context_snapshots", SESSION_INSIGHT_MATERIALIZER_VERSION, len(bundle.context_snapshot_records)),
    ):
        apply_insight_materialization(
            conn,
            insight_type=insight_type,
            session_id=session_id,
            materializer_version=materializer_version,
            materialized_at_ms=materialized_at_ms,
            source_updated_at_ms=source_updated_at_ms,
            source_sort_key_ms=source_sort_key_ms,
            input_high_water_mark_ms=input_high_water_mark_ms,
            input_high_water_mark_source=profile.input_high_water_mark_source,
            input_row_count=input_row_count,
        )


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


def _materialize_thread_spine_sync(
    conn: sqlite3.Connection,
    records_by_root: Mapping[str, ThreadRecord | None],
    *,
    materialized_at_ms: int,
) -> None:
    """Restore the topology-owned thread spine and stamp readiness markers.

    ``replace_threads_bulk_sync`` (the analytics writer) deliberately never
    touches ``created_at_ms``/``session_count``/``depth`` or the
    ``thread_sessions`` membership join (#1743 P13). This writer re-derives that
    spine from the same evidence topology uses at ingest — the root session's
    timestamps and the parent-chain membership in ``record.session_ids`` — so a
    full insight rebuild (which clears the ``threads`` table) restores
    everything topology would otherwise own, without depending on the
    ``root_session_id`` column. It also stamps the per-member ``'thread'``
    ``insight_materialization`` marker that readiness ``stale_thread_count``
    counts: every session in every refreshed thread, not just the root.
    """
    from polylogue.storage.sqlite.archive_tiers.write import apply_insight_materialization

    for root_id, record in records_by_root.items():
        if record is None:
            continue
        root = str(root_id)
        session_ids = [str(session_id) for session_id in record.session_ids]
        root_row = conn.execute(
            "SELECT created_at_ms, updated_at_ms FROM sessions WHERE session_id = ?",
            (root,),
        ).fetchone()
        created_at_ms = int(root_row["created_at_ms"] or root_row["updated_at_ms"] or 0) if root_row else 0
        updated_at_ms = int(root_row["updated_at_ms"] or 0) if root_row else 0
        conn.execute(
            "UPDATE threads SET created_at_ms = ?, session_count = ?, depth = ? WHERE thread_id = ?",
            (created_at_ms, len(session_ids), record.depth, root),
        )
        conn.execute("DELETE FROM thread_sessions WHERE thread_id = ?", (root,))
        hwm_ms = updated_at_ms or created_at_ms or None
        for position, session_id in enumerate(session_ids):
            conn.execute(
                "INSERT INTO thread_sessions (thread_id, session_id, position) VALUES (?, ?, ?)",
                (root, session_id, position),
            )
            apply_insight_materialization(
                conn,
                insight_type="thread",
                session_id=session_id,
                materializer_version=1,
                materialized_at_ms=materialized_at_ms,
                source_updated_at_ms=updated_at_ms or None,
                source_sort_key_ms=hwm_ms,
                input_high_water_mark_ms=hwm_ms,
                input_high_water_mark_source="provider_ts" if hwm_ms else "fallback_date",
                input_row_count=len(session_ids),
            )


def _refresh_thread_roots_sync(
    conn: sqlite3.Connection,
    root_ids: Sequence[str],
    *,
    materialized_at_ms: int,
) -> int:
    normalized_root_ids = tuple(dict.fromkeys(str(root_id) for root_id in root_ids if str(root_id)))
    if not normalized_root_ids:
        return 0
    records_by_root = build_thread_records_for_roots_sync(conn, normalized_root_ids)
    mapping: dict[str, ThreadRecord | None] = {root_id: records_by_root.get(root_id) for root_id in normalized_root_ids}
    replace_threads_bulk_sync(conn, mapping)
    _materialize_thread_spine_sync(conn, mapping, materialized_at_ms=materialized_at_ms)
    return sum(1 for root_id in normalized_root_ids if records_by_root.get(root_id) is not None)


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
    stage_timings_s: dict[str, float] | None = None,
    stage_timing_prefix: str = "insights",
) -> SessionInsightCounts:
    def add_timing(name: str, started_at: float) -> None:
        if stage_timings_s is None:
            return
        key = f"{stage_timing_prefix}.{name}"
        stage_timings_s[key] = stage_timings_s.get(key, 0.0) + (time.perf_counter() - started_at)

    session_chunks: Iterable[Sequence[str]]
    previous_profile_groups: set[tuple[str, str]] = set()
    thread_materialized_at_ms = _epoch_ms_or_none(now_iso()) or 0
    if session_ids is None:
        # #1607: the seven DELETEs hold the write lock for seconds-to-minutes
        # at archive scale and were previously silent. Emit a progress
        # heartbeat per table so the operator sees forward motion instead
        # of full insight rebuilds appearing to hang with no output.
        # The full rebuild stays inside one transaction (the implicit tx
        # started by the first DELETE spans every subsequent DML through
        # the final ``conn.commit()`` at the bottom of this function), so
        # a SIGKILL mid-rebuild rolls the WAL back to the prior state on
        # the next open — the prior insights are intact, not emptied.
        # thread_sessions before threads keeps the FK cascade ordering explicit
        # even if foreign_keys are off; both are repopulated per root in the
        # thread-refresh phase below. The 'thread' materialization markers are
        # cleared then re-stamped there so readiness stale_thread_count reflects
        # exactly the rebuilt membership.
        _delete_tables_with_progress_sync(
            conn,
            tables=(
                "session_work_events",
                "session_phases",
                "session_runs",
                "session_observed_events",
                "session_context_snapshots",
                "session_latency_profiles",
                "session_profiles",
                "session_tag_rollups",
                "thread_sessions",
                "threads",
            ),
            progress_callback=progress_callback,
        )
        session_chunks = iter_session_id_pages_sync(conn, page_size=page_size)
    else:
        session_ids = tuple(dict.fromkeys(str(session_id) for session_id in session_ids))
        t0 = time.perf_counter()
        previous_profile_groups = {
            group
            for record in _session_profile_records_for_session_ids_sync(conn, session_ids)
            if (group := profile_provider_day(record)) is not None
        }
        add_timing("existing_profile_groups", t0)
        t0 = time.perf_counter()
        message_counts = _load_message_counts_sync(conn, session_ids)
        add_timing("message_counts", t0)
        t0 = time.perf_counter()
        session_chunks = (
            chunk.session_ids
            for chunk in _chunk_session_ids_by_message_budget_sync(
                session_ids,
                message_counts=message_counts,
                page_size=page_size,
                message_budget=_SESSION_INSIGHT_REBUILD_MESSAGE_BUDGET,
            )
        )
        add_timing("chunk_plan", t0)
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
        t0 = time.perf_counter()
        batch = load_sync_batch(conn, chunk)
        add_timing("load_batch", t0)
        t0 = time.perf_counter()
        root_ids_by_session = thread_root_ids_sync(conn, chunk)
        add_timing("thread_root_lookup", t0)
        t0 = time.perf_counter()
        hydrated_sessions = hydrate_sessions(batch)
        add_timing("hydrate", t0)
        t0 = time.perf_counter()
        record_bundles = build_session_insight_record_bundles(
            hydrated_sessions,
            compaction_counts_by_session=batch.compaction_counts_by_session,
            logical_session_ids_by_session=root_ids_by_session,
            stage_timing_add=add_timing,
        )
        add_timing("build_records", t0)
        t0 = time.perf_counter()
        chunk_profiles, chunk_work_events, chunk_phases = _count_record_bundles(record_bundles)
        add_timing("count_records", t0)
        t0 = time.perf_counter()
        replace_session_profiles_bulk_sync(conn, [bundle.profile_record for bundle in record_bundles])
        add_timing("write_profiles", t0)
        t0 = time.perf_counter()
        replace_session_latency_profiles_bulk_sync(
            conn,
            [bundle.latency_profile_record for bundle in record_bundles],
        )
        add_timing("write_latency_profiles", t0)
        t0 = time.perf_counter()
        replace_session_work_events_bulk_sync(
            conn,
            {bundle.session_id: bundle.work_event_records for bundle in record_bundles},
        )
        add_timing("write_work_events", t0)
        t0 = time.perf_counter()
        replace_session_phases_bulk_sync(
            conn,
            {bundle.session_id: bundle.phase_records for bundle in record_bundles},
        )
        add_timing("write_phases", t0)
        t0 = time.perf_counter()
        replace_session_runs_bulk_sync(
            conn,
            {bundle.session_id: bundle.run_records for bundle in record_bundles},
        )
        replace_session_observed_events_bulk_sync(
            conn,
            {bundle.session_id: bundle.observed_event_records for bundle in record_bundles},
        )
        replace_session_context_snapshots_bulk_sync(
            conn,
            {bundle.session_id: bundle.context_snapshot_records for bundle in record_bundles},
        )
        add_timing("write_run_projection", t0)
        t0 = time.perf_counter()
        for bundle in record_bundles:
            _stamp_bundle_materialization(conn, bundle)
            group = profile_provider_day(bundle.profile_record)
            if group is not None:
                refreshed_profile_groups.add(group)
        add_timing("stamp_materialization", t0)
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
            conn.execute("DELETE FROM thread_sessions")
            conn.execute("DELETE FROM threads")
            conn.execute("DELETE FROM session_phases")
            conn.execute("DELETE FROM session_tag_rollups")
            conn.execute("DELETE FROM insight_materialization WHERE insight_type = 'thread'")
        conn.commit()
        return _empty_rebuild_counts()

    if session_ids is not None:
        t0 = time.perf_counter()
        affected_roots = tuple(thread_root_ids_sync(conn, session_ids).values())
        add_timing("affected_thread_roots", t0)
        t0 = time.perf_counter()
        thread_count = _refresh_thread_roots_sync(
            conn,
            affected_roots,
            materialized_at_ms=thread_materialized_at_ms,
        )
        add_timing("refresh_threads", t0)
        t0 = time.perf_counter()
        refresh_sync_provider_day_aggregates(
            conn,
            previous_profile_groups | refreshed_profile_groups,
        )
        add_timing("provider_day_aggregates", t0)
        t0 = time.perf_counter()
        tag_rollup_count = conn.execute("SELECT COUNT(*) FROM session_tag_rollups").fetchone()[0]
        add_timing("tag_rollup_count", t0)
        t0 = time.perf_counter()
        conn.commit()
        add_timing("commit", t0)
        return _finalize_rebuild_counts(
            profiles=profile_count,
            work_events=work_event_count,
            phases=phase_count,
            threads=thread_count,
            tag_rollups=int(tag_rollup_count),
        )

    # threads / thread_sessions were cleared in the DELETE phase above; re-stamp
    # the 'thread' markers so readiness reflects only the rebuilt membership.
    conn.execute("DELETE FROM insight_materialization WHERE insight_type = 'thread'")
    thread_count = 0
    for root_chunk in iter_root_id_pages_sync(conn):
        records_by_root = build_thread_records_for_roots_sync(conn, root_chunk)
        mapping: dict[str, ThreadRecord | None] = {root_id: records_by_root.get(root_id) for root_id in root_chunk}
        replace_threads_bulk_sync(conn, mapping)
        _materialize_thread_spine_sync(conn, mapping, materialized_at_ms=thread_materialized_at_ms)
        thread_count += sum(1 for root_id in root_chunk if records_by_root.get(root_id) is not None)
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
    from polylogue.storage.sqlite.queries.session_insight_run_projection_writes import (
        replace_session_context_snapshots,
        replace_session_observed_events,
        replace_session_runs,
    )
    from polylogue.storage.sqlite.queries.session_insight_thread_queries import replace_thread
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
            "session_runs",
            "session_observed_events",
            "session_context_snapshots",
            "session_latency_profiles",
            "session_profiles",
            "session_tag_rollups",
        ):
            cursor = await conn.execute(f"DELETE FROM {table}")
            if progress_callback is not None:
                rowcount = getattr(cursor, "rowcount", 0) or 0
                progress_callback(int(rowcount if rowcount > 0 else 0), desc=f"rebuild: cleared {table}")
    elif not session_ids:
        await conn.execute("DELETE FROM threads")
        await conn.execute("DELETE FROM session_phases")
        await conn.execute("DELETE FROM session_runs")
        await conn.execute("DELETE FROM session_observed_events")
        await conn.execute("DELETE FROM session_context_snapshots")
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
                await replace_session_runs(
                    conn,
                    bundle.session_id,
                    bundle.run_records,
                    transaction_depth,
                )
                await replace_session_observed_events(
                    conn,
                    bundle.session_id,
                    bundle.observed_event_records,
                    transaction_depth,
                )
                await replace_session_context_snapshots(
                    conn,
                    bundle.session_id,
                    bundle.context_snapshot_records,
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
                await replace_session_runs(
                    conn,
                    bundle.session_id,
                    bundle.run_records,
                    transaction_depth,
                )
                await replace_session_observed_events(
                    conn,
                    bundle.session_id,
                    bundle.observed_event_records,
                    transaction_depth,
                )
                await replace_session_context_snapshots(
                    conn,
                    bundle.session_id,
                    bundle.context_snapshot_records,
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
            if len(batch.messages) >= _SESSION_INSIGHT_RELEASE_MESSAGE_THRESHOLD:
                del batch, record_bundles
                release_process_memory()

    await conn.execute("DELETE FROM threads")
    thread_count = 0
    async for root_chunk in iter_root_id_pages_async(conn):
        records_by_root = await build_thread_records_for_roots_async(conn, root_chunk)
        for root_id in root_chunk:
            record = records_by_root.get(root_id)
            if record is None:
                continue
            await replace_thread(conn, record.thread_id, record, transaction_depth)
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
