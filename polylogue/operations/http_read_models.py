"""Bounded archive read models used by HTTP projections."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from polylogue.analysis.archive import SessionCostInsight, ThreadInsight
    from polylogue.analysis.topology import SessionTopology
    from polylogue.archive.session.session_profile import SessionProfile
    from polylogue.storage.runtime import SessionProfileRecord, SessionRecord
    from polylogue.storage.sqlite.archive_tiers.archive import ArchiveSessionSummary, ArchiveStore


def read_configured_sources() -> dict[str, object]:
    """Project configured live sources for the HTTP source-list endpoint."""

    from polylogue.sources.live.watcher import default_sources

    sources = default_sources()
    return {"sources": [{"name": s.name, "root": str(s.root), "exists": s.exists()} for s in sources]}


@dataclass(frozen=True, slots=True)
class ArchiveOverview:
    total_sessions: int
    total_messages: int
    origins: dict[str, int]
    recent: tuple[ArchiveSessionSummary, ...]


def read_archive_overview(archive: ArchiveStore) -> ArchiveOverview:
    """Read the cockpit's aggregate and six-row page from one pinned reader."""

    origin_rows = archive._conn.execute(
        "SELECT origin, COUNT(*) FROM sessions GROUP BY origin ORDER BY origin"
    ).fetchall()
    total_sessions = int(archive.count_sessions())
    total_messages = int(archive._conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0])
    recent = tuple(archive.list_summaries(limit=6, offset=0))
    return ArchiveOverview(
        total_sessions=total_sessions,
        total_messages=total_messages,
        origins={str(row[0]): int(row[1]) for row in origin_rows},
        recent=recent,
    )


@dataclass(frozen=True, slots=True)
class SessionCostRead:
    session_id: str
    origin: str
    insight: SessionCostInsight | None


def read_session_cost(archive: ArchiveStore, session_id: str) -> SessionCostRead | None:
    """Read a session's cost insight from the supplied archive generation."""

    try:
        resolved_id = archive.resolve_session_id(session_id)
        summary = archive.read_summary(resolved_id)
    except KeyError:
        return None
    insights = archive.list_session_cost_insights(session_id=resolved_id, limit=1)
    return SessionCostRead(resolved_id, summary.origin, insights[0] if insights else None)


def read_session_raw(archive: ArchiveStore, session_id: str) -> dict[str, object] | None:
    """Read the legacy raw-session header and bounded artifact page."""

    from polylogue.archive.hydration import archive_envelope_to_session

    try:
        resolved_id = archive.resolve_session_id(session_id)
        envelope = archive.read_session(resolved_id)
    except KeyError:
        return None
    session = archive_envelope_to_session(envelope)
    artifacts, total = archive.raw_artifacts_for_session(resolved_id)
    return {
        "id": str(session.id),
        "origin": session.origin,
        "title": session.display_title,
        "working_directories": list(getattr(session, "working_directories", ()) or ()),
        "git_branch": getattr(session, "git_branch", None),
        "git_repository_url": getattr(session, "git_repository_url", None),
        "branch_type": str(session.branch_type) if session.branch_type else None,
        "parent_id": str(session.parent_id) if session.parent_id else None,
        "session_id": getattr(session, "session_id", None),
        "raw_artifacts": artifacts,
        "raw_artifacts_total": total,
    }


def read_raw_artifacts(archive: ArchiveStore, session_id: str) -> dict[str, object]:
    """Retain the raw-artifact endpoint's existing tuple-shaped result."""

    return {"raw_artifacts": archive.raw_artifacts_for_session(session_id)}


@dataclass(frozen=True, slots=True)
class SessionAttachmentRead:
    session_id: str
    attachments: tuple[tuple[Any, str], ...]


def read_session_attachments(archive: ArchiveStore, session_id: str) -> SessionAttachmentRead | None:
    """Read exactly the message-owned attachments in the composed session."""

    from polylogue.archive.hydration import archive_envelope_to_session

    try:
        resolved_id = archive.resolve_session_id(session_id)
        session = archive_envelope_to_session(archive.read_session(resolved_id))
    except KeyError:
        return None
    return SessionAttachmentRead(
        session_id=str(session.id),
        attachments=tuple(
            (attachment, str(message.id)) for message in session.messages for attachment in message.attachments or ()
        ),
    )


@dataclass(frozen=True, slots=True)
class SessionEvidenceRead:
    session_id: str
    origin: str
    tool_calls: int
    outcomes: tuple[int, int, int]
    lineage_refs: tuple[tuple[str, str, str], ...]
    lineage_unreadable: bool
    lineage_error: sqlite3.Error | None
    cost: SessionCostRead


def read_session_evidence(archive: ArchiveStore, session_id: str) -> SessionEvidenceRead | None:
    """Read composed-message action counts, lineage refs, and cost in one snapshot."""

    try:
        resolved_id = archive.resolve_session_id(session_id)
        summary = archive.read_summary(resolved_id)
        envelope = archive.read_session(resolved_id)
    except KeyError:
        return None
    message_ids_json = json.dumps([message.message_id for message in envelope.messages])
    tool_calls = int(
        archive._conn.execute(
            "SELECT COUNT(*) FROM blocks WHERE block_type = 'tool_use' "
            "AND message_id IN (SELECT value FROM json_each(?))",
            (message_ids_json,),
        ).fetchone()[0]
    )
    outcome_row = archive._conn.execute(
        """SELECT
                  COALESCE(SUM(outcome = 'ok'), 0),
                  COALESCE(SUM(outcome = 'failed'), 0),
                  COALESCE(SUM(outcome = 'unknown'), 0)
           FROM (
             SELECT CASE result_state
               WHEN 'outcome_success' THEN 'ok'
               WHEN 'outcome_error' THEN 'failed'
               ELSE 'unknown'
             END AS outcome
             FROM actions WHERE message_id IN (SELECT value FROM json_each(?))
           )""",
        (message_ids_json,),
    ).fetchone()
    lineage_error: sqlite3.Error | None = None
    try:
        rows = archive._conn.execute(
            "SELECT dst_origin || ':' || dst_native_id, link_type, status FROM session_links WHERE src_session_id = ? ORDER BY link_type, dst_origin, dst_native_id",
            (resolved_id,),
        ).fetchall()
    except sqlite3.Error as exc:
        rows = []
        lineage_error = exc
    cost = read_session_cost(archive, resolved_id)
    assert cost is not None
    return SessionEvidenceRead(
        session_id=resolved_id,
        origin=summary.origin,
        tool_calls=tool_calls,
        outcomes=(int(outcome_row[0]), int(outcome_row[1]), int(outcome_row[2])),
        lineage_refs=tuple((str(row[0]), str(row[1]), str(row[2])) for row in rows),
        lineage_unreadable=lineage_error is not None,
        lineage_error=lineage_error,
        cost=cost,
    )


def read_attachment_library_page(
    archive: ArchiveStore,
    *,
    limit: int,
    offset: int,
    mime_filter: str,
    state_filter: str,
    session_filter: str,
) -> list[tuple[Any, str, str | None]]:
    """Read attachment references in the same stable order as the repository query."""

    from polylogue.storage.sqlite.queries.attachment_records import (
        AttachmentLibraryWindow,
        attachment_library_page_sql,
    )

    segments: tuple[tuple[str, int | None, int | None], ...] = ()
    if session_filter:
        from polylogue.core.errors import DatabaseError
        from polylogue.storage.sqlite.archive_tiers.write import _composed_transcript_plan

        plan = _composed_transcript_plan(archive._conn, session_filter)
        if not plan.lineage_complete:
            raise DatabaseError(
                f"attachment library membership has incomplete lineage: {plan.lineage_truncation_reason}"
            )
        segments = tuple(
            (segment.session_id, segment.upto_position, segment.upto_variant_index) for segment in plan.segments
        )
    sql, args = attachment_library_page_sql(
        limit=None if state_filter else limit,
        offset=0 if state_filter else offset,
        mime_filter=mime_filter,
        session_filter=session_filter,
        segments=segments,
    )
    window = AttachmentLibraryWindow(
        limit=limit,
        offset=offset if state_filter else 0,
        state_filter=state_filter,
        blob_store=archive._read_blob_store,
    )
    if limit <= 0:
        return []
    cursor = archive._conn.execute(sql, args)
    try:
        for row in cursor:
            if window.consume((row,)):
                break
    finally:
        cursor.close()
    return window.rows


class _PinnedTopologySource:
    """Adapt one pinned archive reader to the canonical bounded graph builder."""

    def __init__(self, archive: ArchiveStore) -> None:
        self.archive = archive

    async def get_session(self, session_id: str) -> SessionRecord | None:
        from polylogue.storage.sqlite.queries.mappers import _row_to_session
        from polylogue.storage.sqlite.queries.sessions_reads import _SESSION_RECORD_SELECT

        try:
            resolved_id = self.archive.resolve_session_id(session_id)
        except KeyError:
            return None
        row = self.archive._conn.execute(
            f"SELECT {_SESSION_RECORD_SELECT} FROM sessions WHERE session_id = ?",
            (resolved_id,),
        ).fetchone()
        return _row_to_session(row) if row is not None else None

    async def list_session_links_for_session(
        self, session_id: str, *, limit: int | None = None
    ) -> list[dict[str, object]]:
        from polylogue.storage.sqlite.queries.session_links import SESSION_LINK_COLUMNS

        bounded = "" if limit is None else " LIMIT ?"
        params: tuple[object, ...] = (session_id,) if limit is None else (session_id, limit)
        rows = self.archive._conn.execute(
            f"SELECT {SESSION_LINK_COLUMNS} FROM session_links WHERE src_session_id = ? "
            "ORDER BY link_type, dst_origin, dst_native_id" + bounded,
            params,
        ).fetchall()
        return [dict(row) for row in rows]

    async def list_session_links_to_session(self, session_id: str, *, limit: int | None) -> list[dict[str, object]]:
        from polylogue.storage.sqlite.queries.session_links import SESSION_LINK_COLUMNS

        bound = "" if limit is None else " LIMIT ?"
        args: tuple[object, ...] = (session_id,) if limit is None else (session_id, limit)
        rows = self.archive._conn.execute(
            f"SELECT {SESSION_LINK_COLUMNS} FROM session_links WHERE resolved_dst_session_id = ? "
            "ORDER BY src_session_id, dst_origin, dst_native_id, link_type" + bound,
            args,
        ).fetchall()
        return [dict(row) for row in rows]


async def read_session_topology(
    archive: ArchiveStore,
    session_id: str,
    *,
    node_offset: int = 0,
    node_limit: int = 200,
    edge_limit: int = 500,
) -> SessionTopology | None:
    """Derive the canonical bounded topology from one pinned archive."""

    from polylogue.storage.derived.topology.derivation import derive_session_topology_async

    return await derive_session_topology_async(
        _PinnedTopologySource(archive),
        session_id,
        node_offset=node_offset,
        node_limit=node_limit,
        edge_limit=edge_limit,
    )


@dataclass(frozen=True, slots=True)
class SessionInsightsRead:
    session_id: str
    origin: str
    updated_at: str | None
    profile_record: SessionProfileRecord | None
    profile: SessionProfile | None
    profile_error: BaseException | None
    partition_status: str | None
    profile_row_matches: bool
    threads: tuple[ThreadInsight, ...]
    threads_error: BaseException | None


def read_session_insights(
    archive: ArchiveStore,
    session_id: str,
    includes: tuple[str, ...],
) -> SessionInsightsRead | None:
    """Read requested insight facts and exact profile binding from one snapshot."""

    from polylogue.analysis.archive import ArchiveInsightUnavailableError
    from polylogue.storage.derived.session.derivation import bound_session_profile_partitions
    from polylogue.storage.derived.session.profiles import hydrate_session_profile
    from polylogue.storage.runtime import SESSION_INSIGHT_MATERIALIZER_VERSION

    try:
        resolved_id = archive.resolve_session_id(session_id)
        summary = archive.read_summary(resolved_id)
    except KeyError:
        return None
    profile_record = None
    profile = None
    profile_error: BaseException | None = None
    partition_status: str | None = None
    profile_row_matches = False
    if "profile" in includes:
        try:
            profile_record = archive.get_session_profile_record(resolved_id)
            if profile_record is not None:
                profile = hydrate_session_profile(profile_record)
                partition_status = bound_session_profile_partitions(
                    archive._conn,
                    (resolved_id,),
                    materializer_version=SESSION_INSIGHT_MATERIALIZER_VERSION,
                )[resolved_id]
                row = archive._conn.execute(
                    "SELECT input_content_hash, materializer_version FROM session_profiles WHERE session_id = ?",
                    (resolved_id,),
                ).fetchone()
                stored_binding = None if row is None or row[0] is None else str(row[0])
                stored_version = None if row is None else int(row[1])
                profile_row_matches = (
                    profile_record.input_content_hash is not None
                    and profile_record.input_content_hash == stored_binding
                    and profile_record.materializer_version == stored_version
                )
        except ArchiveInsightUnavailableError as exc:
            profile_record = None
            profile = None
            profile_error = exc
    threads: tuple[ThreadInsight, ...] = ()
    threads_error: BaseException | None = None
    if "threads" in includes:
        try:
            all_threads = archive.list_thread_insights(limit=None)
            threads = tuple(thread for thread in all_threads if resolved_id in (thread.thread.session_ids or ()))
        except ArchiveInsightUnavailableError as exc:
            threads_error = exc
    return SessionInsightsRead(
        session_id=resolved_id,
        origin=summary.origin,
        updated_at=summary.updated_at,
        profile_record=profile_record,
        profile=profile,
        profile_error=profile_error,
        partition_status=partition_status,
        profile_row_matches=profile_row_matches,
        threads=threads,
        threads_error=threads_error,
    )
