"""Small archive-root façade over archive source/index/user tiers."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from types import TracebackType
from typing import Any, Literal, TypedDict

from polylogue.archive.query.path_prefix import escaped_sql_path_prefix_patterns
from polylogue.archive.query.predicate import (
    QueryBoolPredicate,
    QueryExistsPredicate,
    QueryFieldPredicate,
    QueryLineagePredicate,
    QueryNotPredicate,
    QueryPredicate,
    QuerySequencePredicate,
    QueryTextPredicate,
)
from polylogue.archive.semantic.pricing import (
    CostBasisPayload,
    CostEstimatePayload,
    CostEstimateStatus,
    CostUnavailableReason,
    _normalize_model,
)
from polylogue.archive.stats import ArchiveStats
from polylogue.core.dates import parse_date
from polylogue.core.enums import Provider
from polylogue.core.sources import origin_from_provider
from polylogue.insights.archive import (
    ArchiveCoverageInsight,
    ArchiveDebtInsight,
    ArchiveEnrichmentProvenance,
    ArchiveInferenceProvenance,
    ArchiveInsightProvenance,
    CostRollupInsight,
    SessionCostInsight,
    SessionEnrichmentPayload,
    SessionEvidencePayload,
    SessionInferencePayload,
    SessionLatencyProfileInsight,
    SessionLatencyProfilePayload,
    SessionPhaseEvidencePayload,
    SessionPhaseInferencePayload,
    SessionPhaseInsight,
    SessionProfileInsight,
    SessionTagRollupInsight,
    SessionWorkEventInsight,
    ThreadInsight,
    WorkEventEvidencePayload,
    WorkEventInferencePayload,
)
from polylogue.insights.archive_models import ThreadMemberEvidencePayload, ThreadPayload
from polylogue.insights.archive_rollups import aggregate_cost_rollup_insights
from polylogue.insights.audit import InsightRigorAuditQuery, InsightRigorAuditReport, _audit_one
from polylogue.insights.confidence import from_score as confidence_from_score
from polylogue.insights.feedback import LearningCorrection, parse_correction_kind
from polylogue.insights.readiness import (
    InsightProviderCoverage,
    InsightReadinessEntry,
    InsightReadinessQuery,
    InsightReadinessReport,
    InsightReadinessVerdict,
    InsightStorageArtifact,
    InsightVersionCoverage,
    known_insight_readiness_names,
    normalize_insight_readiness_name,
)
from polylogue.insights.rigor import list_rigor_contracts
from polylogue.insights.tool_usage import ToolUsageInsight, ToolUsageInsightQuery, build_tool_usage_insight
from polylogue.sources.parsers.base import ParsedSession
from polylogue.storage.insights.session.records import SessionProfileRecord
from polylogue.storage.insights.session.runtime import SessionInsightStatusSnapshot
from polylogue.storage.runtime.store_constants import SESSION_INSIGHT_MATERIALIZER_VERSION
from polylogue.storage.search.query_support import normalize_fts5_query
from polylogue.storage.sqlite.archive_tiers.bootstrap import (
    initialize_active_archive_root,
    initialize_archive_database,
)
from polylogue.storage.sqlite.archive_tiers.source_write import (
    write_source_raw_session,
    write_source_raw_session_blob_ref,
)
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from polylogue.storage.sqlite.archive_tiers.user_write import (
    ArchiveBlackboardNoteEnvelope,
    AssertionKind,
    assertion_id_for_annotation,
    assertion_id_for_correction,
    assertion_id_for_mark,
    assertion_id_for_recall_pack,
    assertion_id_for_saved_view,
    assertion_id_for_workspace,
    list_archive_blackboard_note_envelopes,
    mark_assertion_status,
    upsert_annotation,
    upsert_assertion,
    upsert_blackboard_note,
    upsert_mark,
    upsert_recall_pack,
    upsert_saved_view,
    upsert_workspace,
)
from polylogue.storage.sqlite.archive_tiers.write import (
    ArchiveInsightMaterialization,
    ArchiveSessionEnvelope,
    ArchiveSessionPhase,
    ArchiveSessionWorkEvent,
    read_archive_session_envelope,
    read_insight_materialization,
    read_session_phases,
    read_session_work_events,
    rebuild_archive_messages_fts,
    search_archive_blocks,
    upsert_session_tag,
    write_parsed_session_to_archive,
)
from polylogue.storage.sqlite.connection_profile import (
    READ_CONNECTION_PRAGMA_STATEMENTS,
    WRITE_CONNECTION_PRAGMA_STATEMENTS,
)
from polylogue.storage.sqlite.queries.tool_usage import ToolUsageProviderCoverageRow, ToolUsageRow
from polylogue.types import SessionId


class IndexStatus(TypedDict):
    """block-FTS index existence and indexed-document count."""

    exists: bool
    count: int


@dataclass(frozen=True, slots=True)
class ArchiveSessionSummary:
    """archive summary projection over archive sessions."""

    session_id: str
    native_id: str
    origin: str
    provider: Provider
    title: str | None
    created_at: str | None
    updated_at: str | None
    message_count: int
    word_count: int
    tags: tuple[str, ...]
    working_directories: tuple[str, ...] = ()
    git_branch: str | None = None
    git_repository_url: str | None = None


@dataclass(frozen=True, slots=True)
class ArchiveSessionSearchHit:
    """Search hit projection over archive block FTS."""

    rank: int
    session_id: str
    block_id: str
    message_id: str
    origin: str
    provider: Provider
    title: str | None
    snippet: str


@dataclass(frozen=True, slots=True)
class ArchiveMessageQueryRow:
    """Terminal query projection over archive messages."""

    message_id: str
    session_id: str
    origin: str
    title: str | None
    role: str
    message_type: str
    position: int
    word_count: int
    text: str


@dataclass(frozen=True, slots=True)
class ArchiveActionQueryRow:
    """Terminal query projection over normalized tool/action rows."""

    session_id: str
    message_id: str
    origin: str
    title: str | None
    tool_use_block_id: str
    tool_result_block_id: str | None
    tool_name: str | None
    semantic_type: str | None
    tool_command: str | None
    tool_path: str | None
    output_text: str | None


@dataclass(frozen=True, slots=True)
class ArchiveBlockQueryRow:
    """Terminal query projection over archive content blocks."""

    block_id: str
    message_id: str
    session_id: str
    origin: str
    title: str | None
    block_type: str
    position: int
    text: str | None
    tool_name: str | None
    semantic_type: str | None
    tool_command: str | None
    tool_path: str | None


class ArchiveStore:
    """Minimal archive-root façade for archive source/index/user tiers."""

    def __init__(self, archive_root: Path, *, initialize: bool = True, read_only: bool = False) -> None:
        self.archive_root = archive_root
        self.source_db_path = archive_root / "source.db"
        self.index_db_path = archive_root / "index.db"
        self.embeddings_db_path = archive_root / "embeddings.db"
        self.user_db_path = archive_root / "user.db"
        self.ops_db_path = archive_root / "ops.db"
        self._read_only = read_only
        if initialize:
            initialize_active_archive_root(archive_root)
        if read_only:
            self._conn = sqlite3.connect(f"file:{self.index_db_path}?mode=ro", uri=True)
            pragma_statements = READ_CONNECTION_PRAGMA_STATEMENTS
        else:
            self._conn = sqlite3.connect(self.index_db_path)
            pragma_statements = WRITE_CONNECTION_PRAGMA_STATEMENTS
        self._conn.row_factory = sqlite3.Row
        # Apply the canonical connection profile rather than a bare connection.
        # A bare sqlite3.connect defaults to a 5s busy_timeout with no WAL
        # tuning; under daemon write contention (live ingest and convergence
        # stages both writing index.db) that window is exceeded and ingest
        # writes fail with "database is locked", marking files failed and
        # collapsing throughput. The write profile raises busy_timeout to 30s so
        # writers queue instead of failing (docs/internals.md: SQLite tuning is
        # profile-driven, not backend-local).
        for statement in pragma_statements:
            self._conn.execute(statement)
        self._user_tier_attached = False
        self._tags_relation = "session_tags"
        self._attach_user_tier_if_present()

    @classmethod
    def open_existing(cls, archive_root: Path, *, read_only: bool = True) -> ArchiveStore:
        """Open archive tier files, bootstrapping the five-tier root for writes."""
        initialize = not read_only or cls._needs_tier_bootstrap(archive_root)
        return cls(archive_root, initialize=initialize, read_only=read_only)

    @staticmethod
    def _needs_tier_bootstrap(archive_root: Path) -> bool:
        return any(
            not (archive_root / filename).exists()
            for filename in ("source.db", "index.db", "embeddings.db", "user.db", "ops.db")
        )

    def close(self) -> None:
        self._conn.close()

    def write_parsed(self, session: ParsedSession, *, content_hash: str | None = None) -> str:
        """Write a parsed session to index.db."""
        return write_parsed_session_to_archive(self._conn, session, content_hash=content_hash)

    def write_raw_and_parsed(
        self,
        session: ParsedSession,
        *,
        payload: bytes,
        source_path: str,
        acquired_at_ms: int,
        source_index: int = 0,
    ) -> tuple[str, str]:
        """Write raw acquisition bytes and the parsed session they produced."""
        source_conn = sqlite3.connect(self.source_db_path)
        try:
            source_conn.execute("PRAGMA foreign_keys = ON")
            raw_id = write_source_raw_session(
                source_conn,
                origin=origin_from_provider(session.source_name),
                source_path=source_path,
                source_index=source_index,
                native_id=session.provider_session_id,
                payload=payload,
                acquired_at_ms=acquired_at_ms,
            )
        finally:
            source_conn.close()
        session_id = write_parsed_session_to_archive(
            self._conn,
            session,
            raw_id=raw_id,
            merge_append=source_index < 0,
        )
        return raw_id, session_id

    def write_raw_blob_and_parsed(
        self,
        session: ParsedSession,
        *,
        blob_hash_hex: str,
        blob_size: int,
        source_path: str,
        acquired_at_ms: int,
        source_index: int = 0,
    ) -> tuple[str, str]:
        """Write parsed session metadata for an already-materialized raw blob."""
        source_conn = sqlite3.connect(self.source_db_path)
        try:
            source_conn.execute("PRAGMA foreign_keys = ON")
            raw_id = write_source_raw_session_blob_ref(
                source_conn,
                origin=origin_from_provider(session.source_name),
                source_path=source_path,
                source_index=source_index,
                native_id=session.provider_session_id,
                blob_hash=bytes.fromhex(blob_hash_hex),
                blob_size=blob_size,
                acquired_at_ms=acquired_at_ms,
            )
        finally:
            source_conn.close()
        session_id = write_parsed_session_to_archive(
            self._conn,
            session,
            raw_id=raw_id,
            merge_append=source_index < 0,
        )
        return raw_id, session_id

    def read_session(self, session_id: str) -> ArchiveSessionEnvelope:
        """Read a session envelope from index.db."""
        return read_archive_session_envelope(self._conn, session_id)

    def get_session_tree(self, session_id: str) -> list[ArchiveSessionEnvelope]:
        """Return the rooted archive session tree containing ``session_id``."""
        try:
            resolved_session_id = self.resolve_session_id(session_id)
        except KeyError:
            return []
        root_session_id = self._root_session_id_for_tree(resolved_session_id)
        rows = self._conn.execute(
            """
            SELECT session_id
            FROM sessions
            WHERE session_id = ?
               OR root_session_id = ?
            ORDER BY
                CASE WHEN session_id = ? THEN 0 ELSE 1 END,
                COALESCE(sort_key_ms, created_at_ms, updated_at_ms, 0),
                session_id
            """,
            (root_session_id, root_session_id, root_session_id),
        ).fetchall()
        return [read_archive_session_envelope(self._conn, str(row["session_id"])) for row in rows]

    def _root_session_id_for_tree(self, session_id: str) -> str:
        row = self._conn.execute(
            "SELECT root_session_id, parent_session_id FROM sessions WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        if row is None:
            raise KeyError(session_id)
        if row["root_session_id"]:
            return str(row["root_session_id"])

        current_id = session_id
        seen: set[str] = set()
        while current_id not in seen:
            seen.add(current_id)
            parent_row = self._conn.execute(
                "SELECT parent_session_id FROM sessions WHERE session_id = ?",
                (current_id,),
            ).fetchone()
            if parent_row is None or not parent_row["parent_session_id"]:
                return current_id
            current_id = str(parent_row["parent_session_id"])
        return session_id

    def raw_artifacts_for_session(
        self,
        session_id: str,
        *,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[list[dict[str, object]], int]:
        """Return raw acquisition surface rows for one archive session."""
        try:
            resolved_session_id = self.resolve_session_id(session_id)
        except KeyError:
            return [], 0
        raw_row = self._conn.execute(
            "SELECT raw_id FROM sessions WHERE session_id = ?",
            (resolved_session_id,),
        ).fetchone()
        if raw_row is None or raw_row["raw_id"] is None or not self.source_db_path.exists():
            return [], 0
        raw_id = str(raw_row["raw_id"])
        source_conn = sqlite3.connect(f"file:{self.source_db_path}?mode=ro", uri=True)
        source_conn.row_factory = sqlite3.Row
        try:
            total = int(
                source_conn.execute("SELECT COUNT(*) FROM raw_sessions WHERE raw_id = ?", (raw_id,)).fetchone()[0]
            )
            rows = source_conn.execute(
                """
                SELECT raw_id, origin, source_path, blob_size, acquired_at_ms,
                       parsed_at_ms, validation_status
                FROM raw_sessions
                WHERE raw_id = ?
                ORDER BY acquired_at_ms DESC, raw_id
                LIMIT ? OFFSET ?
                """,
                (raw_id, max(limit, 0), max(offset, 0)),
            ).fetchall()
        finally:
            source_conn.close()
        return [
            {
                "raw_id": str(row["raw_id"]),
                "source_name": _provider_for_origin(str(row["origin"])).value,
                "source_path": str(row["source_path"]),
                "blob_size": int(row["blob_size"] or 0),
                "acquired_at": _iso_from_ms(row["acquired_at_ms"]),
                "parsed_at": _iso_from_ms(row["parsed_at_ms"]),
                "validation_status": row["validation_status"],
            }
            for row in rows
        ], total

    def get_session_work_event_insights(self, session_id: str) -> list[SessionWorkEventInsight]:
        """Read archive work-event insights for one session."""
        try:
            resolved_session_id = self.resolve_session_id(session_id)
        except KeyError:
            return []
        return self.list_session_work_event_insights(session_id=resolved_session_id)

    def list_session_work_event_insights(
        self,
        *,
        session_id: str | None = None,
        provider: str | None = None,
        heuristic_label: str | None = None,
        since_ms: int | None = None,
        until_ms: int | None = None,
        limit: int | None = 50,
        offset: int = 0,
    ) -> list[SessionWorkEventInsight]:
        """List archive work-event insights with the public insight contract."""
        where: list[str] = []
        params: list[object] = []
        if session_id is not None:
            where.append("we.session_id = ?")
            params.append(self.resolve_session_id(session_id))
        origin = _origin_for_provider_value(provider)
        if origin is not None:
            where.append("s.origin = ?")
            params.append(origin)
        if heuristic_label is not None:
            where.append("we.work_event_type = ?")
            params.append(heuristic_label)
        if since_ms is not None:
            where.append("COALESCE(we.started_at_ms, s.sort_key_ms) >= ?")
            params.append(since_ms)
        if until_ms is not None:
            where.append("COALESCE(we.started_at_ms, s.sort_key_ms) <= ?")
            params.append(until_ms)
        clause = "WHERE " + " AND ".join(where) if where else ""
        pagination = "" if limit is None else " LIMIT ? OFFSET ?"
        if limit is not None:
            params.extend([max(int(limit), 0), max(int(offset), 0)])
        rows = self._conn.execute(
            f"""
            SELECT we.session_id, we.position
            FROM session_work_events we
            JOIN sessions s ON s.session_id = we.session_id
            {clause}
            ORDER BY COALESCE(we.started_at_ms, s.sort_key_ms) DESC, we.session_id, we.position
            {pagination}
            """,
            tuple(params),
        ).fetchall()
        events_by_session = {str(row["session_id"]) for row in rows}
        indexed: dict[tuple[str, int], SessionWorkEventInsight] = {}
        for event_session_id in events_by_session:
            materialization = _read_archive_materialization(self._conn, "work_events", event_session_id)
            session_origin = _session_origin(self._conn, event_session_id)
            for event in read_session_work_events(self._conn, session_id=event_session_id).values():
                if heuristic_label is None or event.work_event_type == heuristic_label:
                    indexed[(event.session_id, event.position)] = _work_event_insight_from_archive_row(
                        event,
                        origin=session_origin,
                        materialization=materialization,
                    )
        return [indexed[(str(row["session_id"]), int(row["position"]))] for row in rows]

    def get_session_phase_insights(self, session_id: str) -> list[SessionPhaseInsight]:
        """Read archive phase insights for one session."""
        try:
            resolved_session_id = self.resolve_session_id(session_id)
        except KeyError:
            return []
        return self.list_session_phase_insights(session_id=resolved_session_id)

    def list_session_phase_insights(
        self,
        *,
        session_id: str | None = None,
        provider: str | None = None,
        kind: str | None = None,
        since_ms: int | None = None,
        until_ms: int | None = None,
        limit: int | None = 50,
        offset: int = 0,
    ) -> list[SessionPhaseInsight]:
        """List archive phase insights with the public insight contract."""
        where: list[str] = []
        params: list[object] = []
        if session_id is not None:
            where.append("sp.session_id = ?")
            params.append(self.resolve_session_id(session_id))
        origin = _origin_for_provider_value(provider)
        if origin is not None:
            where.append("s.origin = ?")
            params.append(origin)
        if kind is not None:
            where.append("sp.phase_type = ?")
            params.append(kind)
        if since_ms is not None:
            where.append("COALESCE(sp.started_at_ms, s.sort_key_ms) >= ?")
            params.append(since_ms)
        if until_ms is not None:
            where.append("COALESCE(sp.started_at_ms, s.sort_key_ms) <= ?")
            params.append(until_ms)
        clause = "WHERE " + " AND ".join(where) if where else ""
        pagination = "" if limit is None else " LIMIT ? OFFSET ?"
        if limit is not None:
            params.extend([max(int(limit), 0), max(int(offset), 0)])
        rows = self._conn.execute(
            f"""
            SELECT sp.session_id, sp.position
            FROM session_phases sp
            JOIN sessions s ON s.session_id = sp.session_id
            {clause}
            ORDER BY COALESCE(sp.started_at_ms, s.sort_key_ms) DESC, sp.session_id, sp.position
            {pagination}
            """,
            tuple(params),
        ).fetchall()
        phases_by_session = {str(row["session_id"]) for row in rows}
        indexed: dict[tuple[str, int], SessionPhaseInsight] = {}
        for phase_session_id in phases_by_session:
            materialization = _read_archive_materialization(self._conn, "phases", phase_session_id)
            session_origin = _session_origin(self._conn, phase_session_id)
            for phase in read_session_phases(self._conn, session_id=phase_session_id).values():
                if kind is None or phase.phase_type == kind:
                    indexed[(phase.session_id, phase.position)] = _phase_insight_from_archive_row(
                        phase,
                        origin=session_origin,
                        materialization=materialization,
                    )
        return [indexed[(str(row["session_id"]), int(row["position"]))] for row in rows]

    def get_thread_insight(self, thread_id: str) -> ThreadInsight | None:
        """Read one archive thread projection as a public thread insight."""
        row = self._conn.execute(
            "SELECT thread_id FROM threads WHERE thread_id = ?",
            (thread_id,),
        ).fetchone()
        if row is None:
            return None
        return self._thread_insight_from_id(str(row["thread_id"]))

    def list_thread_insights(
        self,
        *,
        query: str | None = None,
        since_ms: int | None = None,
        until_ms: int | None = None,
        limit: int | None = 50,
        offset: int = 0,
    ) -> list[ThreadInsight]:
        """List threads as public thread insights."""
        where: list[str] = []
        params: list[object] = []
        if query:
            like = f"%{query.strip().lower()}%"
            where.append(
                """
                (
                    lower(t.thread_id) LIKE ?
                    OR EXISTS (
                        SELECT 1
                        FROM thread_sessions qts
                        JOIN sessions qs ON qs.session_id = qts.session_id
                        WHERE qts.thread_id = t.thread_id
                          AND (
                            lower(qs.session_id) LIKE ?
                            OR lower(COALESCE(qs.title, '')) LIKE ?
                            OR lower(COALESCE(qs.git_repository_url, '')) LIKE ?
                            OR lower(COALESCE(qs.git_branch, '')) LIKE ?
                          )
                    )
                )
                """.strip()
            )
            params.extend([like, like, like, like, like])
        if since_ms is not None:
            where.append("t.created_at_ms >= ?")
            params.append(since_ms)
        if until_ms is not None:
            where.append("t.created_at_ms <= ?")
            params.append(until_ms)
        clause = "WHERE " + " AND ".join(where) if where else ""
        pagination = "" if limit is None else " LIMIT ? OFFSET ?"
        if limit is not None:
            params.extend([max(int(limit), 0), max(int(offset), 0)])
        rows = self._conn.execute(
            f"""
            SELECT t.thread_id
            FROM threads t
            {clause}
            ORDER BY t.created_at_ms DESC, t.thread_id
            {pagination}
            """,
            tuple(params),
        ).fetchall()
        return [insight for row in rows if (insight := self._thread_insight_from_id(str(row["thread_id"]))) is not None]

    def _thread_insight_from_id(self, thread_id: str) -> ThreadInsight | None:
        row = self._conn.execute(
            """
            SELECT thread_id, created_at_ms, session_count
            FROM threads
            WHERE thread_id = ?
            """,
            (thread_id,),
        ).fetchone()
        if row is None:
            return None
        session_rows = self._conn.execute(
            """
            SELECT s.session_id, s.parent_session_id, s.origin, s.title,
                   s.message_count, s.word_count, s.tool_use_count,
                   s.created_at_ms, s.updated_at_ms, s.git_repository_url,
                   s.git_branch, sp.cost_usd
            FROM thread_sessions ts
            JOIN sessions s ON s.session_id = ts.session_id
            LEFT JOIN session_profiles sp ON sp.session_id = s.session_id
            WHERE ts.thread_id = ?
            ORDER BY ts.position, s.sort_key_ms, s.session_id
            """,
            (thread_id,),
        ).fetchall()
        session_ids = tuple(str(session["session_id"]) for session in session_rows)
        provider_breakdown: dict[str, int] = {}
        for session in session_rows:
            provider = _provider_for_origin(str(session["origin"])).value
            provider_breakdown[provider] = provider_breakdown.get(provider, 0) + 1
        start_ms = min(
            (int(session["created_at_ms"]) for session in session_rows if session["created_at_ms"] is not None),
            default=None,
        )
        end_ms = max(
            (int(session["updated_at_ms"]) for session in session_rows if session["updated_at_ms"] is not None),
            default=None,
        )
        dominant_repo = _dominant_repo(session_rows)
        member_evidence = tuple(
            ThreadMemberEvidencePayload(
                session_id=str(session["session_id"]),
                parent_id=str(session["parent_session_id"]) if session["parent_session_id"] else None,
                role="root" if str(session["session_id"]) == str(row["thread_id"]) else "child",
                depth=_thread_member_depth(session_rows, str(session["session_id"])),
                confidence=1.0,
                support_signals=("archive_thread_sessions",),
                evidence=(f"position={index}",),
            )
            for index, session in enumerate(session_rows)
        )
        payload = ThreadPayload(
            start_time=_iso_from_ms(start_ms),
            end_time=_iso_from_ms(end_ms),
            dominant_repo=dominant_repo,
            session_ids=session_ids,
            session_count=len(session_ids),
            depth=max((member.depth for member in member_evidence), default=0),
            branch_count=sum(1 for session in session_rows if session["parent_session_id"] is not None),
            total_messages=sum(int(session["message_count"] or 0) for session in session_rows),
            total_cost_usd=sum(float(session["cost_usd"] or 0.0) for session in session_rows),
            wall_duration_ms=(end_ms - start_ms) if start_ms is not None and end_ms is not None else 0,
            provider_breakdown=provider_breakdown,
            confidence=1.0 if session_rows else 0.0,
            support_signals=("archive_threads", "archive_thread_sessions"),
            member_evidence=member_evidence,
        )
        materialization = _read_archive_materialization(self._conn, "thread", thread_id)
        return ThreadInsight(
            thread_id=str(row["thread_id"]),
            root_id=str(row["thread_id"]),
            dominant_repo=dominant_repo,
            provenance=_archive_provenance(materialization),
            thread=payload,
        )

    def list_session_cost_insights(
        self,
        *,
        session_id: str | None = None,
        provider: str | None = None,
        status: str | None = None,
        model: str | None = None,
        since_ms: int | None = None,
        until_ms: int | None = None,
        limit: int | None = 50,
        offset: int = 0,
    ) -> list[SessionCostInsight]:
        """List archive session cost insights from sessions plus session_profiles."""
        if model is not None:
            return []
        where: list[str] = []
        params: list[object] = []
        if session_id is not None:
            try:
                resolved_session_id = self.resolve_session_id(session_id)
            except KeyError:
                # Unknown session id: no cost insight exists. Returning [] lets
                # the daemon cost endpoint run its existence check and answer
                # 404 instead of surfacing this as an opaque 500.
                return []
            where.append("s.session_id = ?")
            params.append(resolved_session_id)
        origin = _origin_for_provider_value(provider)
        if origin is not None:
            where.append("s.origin = ?")
            params.append(origin)
        if since_ms is not None:
            where.append("s.sort_key_ms >= ?")
            params.append(since_ms)
        if until_ms is not None:
            where.append("s.sort_key_ms <= ?")
            params.append(until_ms)
        clause = "WHERE " + " AND ".join(where) if where else ""
        pagination = "" if limit is None else " LIMIT ? OFFSET ?"
        if limit is not None:
            params.extend([max(int(limit), 0), max(int(offset), 0)])
        rows = self._conn.execute(
            f"""
            SELECT s.session_id, s.origin, s.title, s.created_at_ms, s.updated_at_ms,
                   s.sort_key_ms, sp.cost_credits, sp.cost_usd, sp.cost_is_estimated,
                   sp.cost_provenance,
                   (
                       SELECT smu.model_name
                       FROM session_model_usage smu
                       WHERE smu.session_id = s.session_id
                       ORDER BY smu.input_tokens + smu.output_tokens DESC, smu.model_name
                       LIMIT 1
                   ) AS model_name
            FROM sessions s
            LEFT JOIN session_profiles sp ON sp.session_id = s.session_id
            {clause}
            ORDER BY s.sort_key_ms DESC, s.session_id
            {pagination}
            """,
            tuple(params),
        ).fetchall()
        insights = [_session_cost_insight_from_archive_row(self._conn, row) for row in rows]
        if status is not None:
            insights = [insight for insight in insights if insight.estimate.status == status]
        return insights

    def list_cost_rollup_insights(
        self,
        *,
        provider: str | None = None,
        model: str | None = None,
        since_ms: int | None = None,
        until_ms: int | None = None,
        limit: int | None = None,
        offset: int = 0,
    ) -> list[CostRollupInsight]:
        """Aggregate archive session-cost insights into public cost rollups."""
        session_costs = self.list_session_cost_insights(
            provider=provider,
            model=model,
            since_ms=since_ms,
            until_ms=until_ms,
            limit=None,
            offset=0,
        )
        rollups = aggregate_cost_rollup_insights(session_costs, materialized_at=datetime.now(UTC).isoformat())
        if offset:
            rollups = rollups[offset:]
        if limit is not None:
            rollups = rollups[: max(int(limit), 0)]
        return rollups

    def list_archive_debt_insights(
        self,
        *,
        category: str | None = None,
        only_actionable: bool = False,
        limit: int | None = None,
        offset: int = 0,
    ) -> list[ArchiveDebtInsight]:
        """Report consistency debt."""
        insights = [
            _archive_messages_fts_debt(self._conn),
            _archive_profile_rows_debt(self._conn),
            _archive_profile_counts_debt(self._conn),
            _archive_materialization_debt(self._conn),
            _archive_source_raw_link_debt(self._conn, self.source_db_path),
            _archive_user_overlay_debt(self._conn, self.user_db_path),
        ]
        insights.sort(key=lambda insight: (insight.category, insight.debt_name))
        if category is not None:
            insights = [insight for insight in insights if insight.category == category]
        if only_actionable:
            insights = [insight for insight in insights if not insight.healthy]
        if offset:
            insights = insights[offset:]
        if limit is not None:
            insights = insights[: max(int(limit), 0)]
        return insights

    def get_session_latency_profile_insight(self, session_id: str) -> SessionLatencyProfileInsight | None:
        """Project one latency profile from timestamped messages."""
        try:
            resolved_session_id = self.resolve_session_id(session_id)
        except KeyError:
            return None
        row = self._conn.execute(
            """
            SELECT session_id, origin, title, sort_key_ms
            FROM sessions
            WHERE session_id = ?
            """,
            (resolved_session_id,),
        ).fetchone()
        return None if row is None else _session_latency_profile_from_archive_row(self._conn, row)

    def list_session_latency_profile_insights(
        self,
        *,
        session_id: str | None = None,
        provider: str | None = None,
        only_stuck: bool = False,
        since_ms: int | None = None,
        until_ms: int | None = None,
        limit: int | None = 50,
        offset: int = 0,
    ) -> list[SessionLatencyProfileInsight]:
        """Project archive latency profiles from sessions plus timestamped messages."""
        where: list[str] = []
        params: list[object] = []
        if session_id is not None:
            where.append("s.session_id = ?")
            params.append(self.resolve_session_id(session_id))
        origin = _origin_for_provider_value(provider)
        if origin is not None:
            where.append("s.origin = ?")
            params.append(origin)
        if since_ms is not None:
            where.append("s.sort_key_ms >= ?")
            params.append(since_ms)
        if until_ms is not None:
            where.append("s.sort_key_ms <= ?")
            params.append(until_ms)
        clause = "WHERE " + " AND ".join(where) if where else ""
        pagination = "" if limit is None else " LIMIT ? OFFSET ?"
        if limit is not None:
            params.extend([max(int(limit), 0), max(int(offset), 0)])
        rows = self._conn.execute(
            f"""
            SELECT s.session_id, s.origin, s.title, s.sort_key_ms
            FROM sessions s
            {clause}
            ORDER BY s.sort_key_ms DESC, s.session_id
            {pagination}
            """,
            tuple(params),
        ).fetchall()
        insights = [_session_latency_profile_from_archive_row(self._conn, row) for row in rows]
        if only_stuck:
            insights = [insight for insight in insights if insight.latency.stuck_tool_count > 0]
        return insights

    def find_stuck_session_latency_profile_insights(
        self,
        *,
        provider: str | None = None,
        since_ms: int | None = None,
        until_ms: int | None = None,
        limit: int | None = 50,
    ) -> list[SessionLatencyProfileInsight]:
        """Return archive latency profiles with stuck tools.

        currently lacks session event start/end pairs, so stuck
        tool detection remains conservative and this returns only profiles
        whose projected stuck count is non-zero.
        """
        return self.list_session_latency_profile_insights(
            provider=provider,
            only_stuck=True,
            since_ms=since_ms,
            until_ms=until_ms,
            limit=limit,
            offset=0,
        )

    def _fetch_session_profile_row(self, session_id: str) -> sqlite3.Row | None:
        """Resolve *session_id* and fetch its joined session/profile row, or None."""
        try:
            resolved_session_id = self.resolve_session_id(session_id)
        except KeyError:
            return None
        rows = self._conn.execute(
            """
            SELECT s.session_id, s.origin, s.root_session_id, s.title, s.created_at_ms, s.updated_at_ms,
                   s.message_count, s.word_count, s.tool_use_count, s.thinking_count,
                   sp.workflow_shape, sp.workflow_shape_confidence, sp.terminal_state,
                   sp.terminal_state_confidence, sp.duration_ms, sp.substantive_count,
                   sp.attachment_count, sp.work_event_count, sp.phase_count,
                   sp.tool_calls_per_minute, sp.cost_usd, sp.cost_is_estimated,
                   sp.cost_provenance,
                   sp.total_cost_usd, sp.total_duration_ms,
                   sp.evidence_payload_json, sp.inference_payload_json, sp.enrichment_payload_json
            FROM session_profiles sp
            JOIN sessions s ON s.session_id = sp.session_id
            WHERE sp.session_id = ?
            """,
            (resolved_session_id,),
        ).fetchall()
        return rows[0] if rows else None

    def get_session_profile_insight(self, session_id: str, *, tier: str = "merged") -> SessionProfileInsight | None:
        """Read one archive session profile insight."""
        row = self._fetch_session_profile_row(session_id)
        if row is None:
            return None
        return _session_profile_insight_from_archive_row(self._conn, row, tier=tier)

    def get_session_profile_record(self, session_id: str) -> SessionProfileRecord | None:
        """Read one archive session profile as a domain :class:`SessionProfileRecord`.

        Mirrors :meth:`get_session_profile_insight` but rehydrates the full
        record needed by ``hydrate_session_profile`` (domain ``SessionProfile``)
        and the provenance-based staleness check. The materialization HWM
        provenance is pulled from ``read_insight_materialization`` so the
        downstream ``is_stale`` comparison is grounded in the same source the
        daemon's ``/insights`` profile panel consumes.

        Returns ``None`` when the session id does not resolve or has no
        materialized profile.
        """
        row = self._fetch_session_profile_row(session_id)
        if row is None:
            return None
        return _session_profile_record_from_archive_row(self._conn, row)

    def list_session_profile_insights(
        self,
        *,
        provider: str | None = None,
        workflow_shape: str | None = None,
        terminal_state: str | None = None,
        since_ms: int | None = None,
        until_ms: int | None = None,
        tier: str = "merged",
        limit: int | None = 50,
        offset: int = 0,
        min_wallclock_seconds: float | None = None,
        max_wallclock_seconds: float | None = None,
        sort: str | None = None,
    ) -> list[SessionProfileInsight]:
        """List archive session profile insights.

        ``min_wallclock_seconds`` / ``max_wallclock_seconds`` filter on the
        session's message-timestamp span (last minus first message), and
        ``sort='wallclock'`` orders by that span descending.
        """
        # Wallclock span = newest minus oldest message timestamp for the session.
        wall_expr = (
            "(SELECT MAX(m.occurred_at_ms) - MIN(m.occurred_at_ms) "
            "FROM messages m WHERE m.session_id = s.session_id AND m.occurred_at_ms IS NOT NULL)"
        )
        where: list[str] = []
        params: list[object] = []
        origin = _origin_for_provider_value(provider)
        if origin is not None:
            where.append("s.origin = ?")
            params.append(origin)
        if workflow_shape is not None:
            where.append("sp.workflow_shape = ?")
            params.append(workflow_shape)
        if terminal_state is not None:
            where.append("sp.terminal_state = ?")
            params.append(terminal_state)
        if since_ms is not None:
            where.append("s.sort_key_ms >= ?")
            params.append(since_ms)
        if until_ms is not None:
            where.append("s.sort_key_ms <= ?")
            params.append(until_ms)
        if min_wallclock_seconds is not None:
            where.append(f"COALESCE({wall_expr}, 0) >= ?")
            params.append(int(min_wallclock_seconds * 1000))
        if max_wallclock_seconds is not None:
            where.append(f"COALESCE({wall_expr}, 0) <= ?")
            params.append(int(max_wallclock_seconds * 1000))
        clause = "WHERE " + " AND ".join(where) if where else ""
        order_by = f"{wall_expr} DESC, s.session_id" if sort == "wallclock" else "s.sort_key_ms DESC, s.session_id"
        pagination = "" if limit is None else " LIMIT ? OFFSET ?"
        if limit is not None:
            params.extend([max(int(limit), 0), max(int(offset), 0)])
        rows = self._conn.execute(
            f"""
            SELECT s.session_id, s.origin, s.root_session_id, s.title, s.created_at_ms, s.updated_at_ms,
                   s.message_count, s.word_count, s.tool_use_count, s.thinking_count,
                   sp.workflow_shape, sp.workflow_shape_confidence, sp.terminal_state,
                   sp.terminal_state_confidence, sp.duration_ms, sp.substantive_count,
                   sp.attachment_count, sp.work_event_count, sp.phase_count,
                   sp.tool_calls_per_minute, sp.cost_usd, sp.cost_is_estimated,
                   sp.cost_provenance,
                   sp.total_cost_usd, sp.total_duration_ms,
                   sp.evidence_payload_json, sp.inference_payload_json, sp.enrichment_payload_json
            FROM session_profiles sp
            JOIN sessions s ON s.session_id = sp.session_id
            {clause}
            ORDER BY {order_by}
            {pagination}
            """,
            tuple(params),
        ).fetchall()
        return [_session_profile_insight_from_archive_row(self._conn, row, tier=tier) for row in rows]

    def read_summary(self, session_id: str) -> ArchiveSessionSummary:
        """Read one session summary by exact session id."""
        row = self._conn.execute(
            f"""
            SELECT s.session_id, s.native_id, s.origin, s.title, s.created_at_ms, s.updated_at_ms,
                   s.message_count, s.word_count, s.git_branch, s.git_repository_url,
                   COALESCE(
                       (
                           SELECT json_group_array(swd.path)
                           FROM session_working_dirs swd
                           WHERE swd.session_id = s.session_id
                           ORDER BY swd.position, swd.path
                       ),
                       '[]'
                   ) AS working_directories_json,
                   COALESCE(
                       json_group_array(st.tag) FILTER (WHERE st.tag IS NOT NULL),
                       '[]'
                   ) AS tags_json
            FROM sessions s
            LEFT JOIN {self._tags_relation} st
              ON st.session_id = s.session_id
             AND st.tag_source = 'user'
            WHERE s.session_id = ?
            GROUP BY s.session_id
            """,
            (session_id,),
        ).fetchone()
        if row is None:
            raise KeyError(session_id)
        return _summary_from_row(row)

    def resolve_session_id(self, token: str) -> str:
        """Resolve an exact or prefix session id token."""
        exact = self._conn.execute(
            "SELECT session_id FROM sessions WHERE session_id = ?",
            (token,),
        ).fetchone()
        if exact is not None:
            return str(exact["session_id"])
        if ":" in token:
            provider_token, native_id = token.split(":", 1)
            origin_id = f"{origin_from_provider(Provider.from_string(provider_token)).value}:{native_id}"
            exact = self._conn.execute(
                "SELECT session_id FROM sessions WHERE session_id = ?",
                (origin_id,),
            ).fetchone()
            if exact is not None:
                return str(exact["session_id"])
        rows = self._conn.execute(
            """
            SELECT session_id
            FROM sessions
            WHERE session_id LIKE ?
            ORDER BY session_id
            LIMIT 2
            """,
            (f"{token}%",),
        ).fetchall()
        if not rows:
            raise KeyError(token)
        if len(rows) > 1:
            raise ValueError(f"session id prefix {token!r} is ambiguous")
        return str(rows[0]["session_id"])

    def search_blocks(self, query: str) -> list[str]:
        """Search indexed block text and return block ids."""
        return search_archive_blocks(self._conn, query)

    def rebuild_index(self) -> int:
        """Rebuild the block FTS index from index.db blocks."""
        rebuilt_rows = rebuild_archive_messages_fts(self._conn)
        self._conn.commit()
        return rebuilt_rows

    def index_status(self) -> IndexStatus:
        """Return ``{exists, count}`` for the archive block FTS index.

        The block FTS index (``messages_fts`` over ``blocks``) is trigger-maintained, so a
        missing table means it was never built and the count is the
        indexed-block total.
        """
        if not _table_exists(self._conn, "messages_fts"):
            return IndexStatus(exists=False, count=0)
        return IndexStatus(exists=True, count=_count_scalar(self._conn, "SELECT COUNT(*) FROM messages_fts"))

    def add_user_tags(self, session_ids: tuple[str, ...], tags: tuple[str, ...]) -> int:
        """Add user tags to archive user.db and return changed row count."""
        user_db_path = self.user_db_path
        initialize_archive_database(user_db_path, ArchiveTier.USER)
        changed = 0
        user_conn = sqlite3.connect(user_db_path)
        user_conn.row_factory = sqlite3.Row
        try:
            for session_id in tuple(dict.fromkeys(self.resolve_session_id(session_id) for session_id in session_ids)):
                for tag in tags:
                    normalized_tag = tag.strip().lower()
                    if not normalized_tag:
                        raise ValueError("tag cannot be empty")
                    existing = user_conn.execute(
                        """
                        SELECT 1
                        FROM session_tags
                        WHERE session_id = ? AND tag = ? AND tag_source = 'user'
                        """,
                        (session_id, normalized_tag),
                    ).fetchone()
                    if existing is not None:
                        continue
                    upsert_session_tag(
                        user_conn,
                        session_id=session_id,
                        tag=tag,
                        tag_source="user",
                        method="cli",
                        evidence={"source": "archive_query"},
                    )
                    changed += 1
        finally:
            user_conn.close()
        self._attach_user_tier_if_present()
        return changed

    def remove_user_tags(self, session_ids: tuple[str, ...], tags: tuple[str, ...]) -> int:
        """Remove user tags from archive user.db and return deleted row count."""
        resolved_session_ids = tuple(dict.fromkeys(self.resolve_session_id(session_id) for session_id in session_ids))
        if not resolved_session_ids or not self.user_db_path.exists():
            return 0
        removed = 0
        user_conn = sqlite3.connect(self.user_db_path)
        try:
            with user_conn:
                for session_id in resolved_session_ids:
                    for tag in tags:
                        normalized_tag = tag.strip().lower()
                        if not normalized_tag:
                            raise ValueError("tag cannot be empty")
                        cursor = user_conn.execute(
                            """
                            DELETE FROM session_tags
                            WHERE session_id = ? AND tag = ? AND tag_source = 'user'
                            """,
                            (session_id, normalized_tag),
                        )
                        removed += max(int(cursor.rowcount), 0)
        finally:
            user_conn.close()
        self._attach_user_tier_if_present()
        return removed

    def list_user_tags(self, *, origin: str | None = None) -> dict[str, int]:
        """Return user tag counts over archive sessions."""
        where = "WHERE st.tag_source = 'user'"
        params: list[object] = []
        if origin is not None:
            where += " AND s.origin = ?"
            params.append(origin)
        rows = self._conn.execute(
            f"""
            SELECT st.tag, COUNT(DISTINCT s.session_id) AS count
            FROM sessions s
            JOIN {self._tags_relation} st
              ON st.session_id = s.session_id
            {where}
            GROUP BY st.tag
            ORDER BY count DESC, st.tag
            """,
            tuple(params),
        ).fetchall()
        return {str(row["tag"]): int(row["count"] or 0) for row in rows}

    def list_session_tag_rollup_insights(
        self,
        *,
        provider: str | None = None,
        query: str | None = None,
        since_ms: int | None = None,
        until_ms: int | None = None,
        limit: int | None = 100,
        offset: int = 0,
    ) -> list[SessionTagRollupInsight]:
        """Aggregate archive session tags into public tag-rollup insights."""
        where: list[str] = []
        params: list[object] = []
        origin = _origin_for_provider_value(provider)
        if origin is not None:
            where.append("s.origin = ?")
            params.append(origin)
        if query:
            where.append("lower(st.tag) LIKE ?")
            params.append(f"%{query.strip().lower()}%")
        if since_ms is not None:
            where.append("s.sort_key_ms >= ?")
            params.append(since_ms)
        if until_ms is not None:
            where.append("s.sort_key_ms <= ?")
            params.append(until_ms)
        clause = "WHERE " + " AND ".join(where) if where else ""
        filter_params = tuple(params)
        pagination = "" if limit is None else " LIMIT ? OFFSET ?"
        if limit is not None:
            params.extend([max(int(limit), 0), max(int(offset), 0)])
        rows = self._conn.execute(
            f"""
            SELECT st.tag,
                   COUNT(DISTINCT s.session_id) AS session_count,
                   COUNT(DISTINCT COALESCE(s.root_session_id, s.session_id)) AS logical_session_count,
                   COUNT(DISTINCT CASE WHEN st.tag_source = 'user' THEN s.session_id END) AS explicit_count,
                   COUNT(DISTINCT CASE WHEN st.tag_source = 'auto' THEN s.session_id END) AS auto_count,
                   MAX(s.sort_key_ms) AS source_sort_key_ms
            FROM sessions s
            JOIN {self._tags_relation} st ON st.session_id = s.session_id
            {clause}
            GROUP BY st.tag
            ORDER BY session_count DESC, st.tag
            {pagination}
            """,
            tuple(params),
        ).fetchall()
        return [
            SessionTagRollupInsight(
                tag=str(row["tag"]),
                session_count=int(row["session_count"] or 0),
                logical_session_count=int(row["logical_session_count"] or 0),
                explicit_count=int(row["explicit_count"] or 0),
                auto_count=int(row["auto_count"] or 0),
                provider_breakdown=_tag_provider_breakdown(
                    self._conn, str(row["tag"]), clause, filter_params, self._tags_relation
                ),
                repo_breakdown=_tag_repo_breakdown(
                    self._conn, str(row["tag"]), clause, filter_params, self._tags_relation
                ),
                provenance=ArchiveInsightProvenance(
                    materializer_version=1,
                    materialized_at=_iso_from_ms(row["source_sort_key_ms"]) or "1970-01-01T00:00:00Z",
                    source_updated_at=_iso_from_ms(row["source_sort_key_ms"]),
                    source_sort_key=(
                        float(row["source_sort_key_ms"]) / 1000.0 if row["source_sort_key_ms"] is not None else None
                    ),
                ),
            )
            for row in rows
        ]

    def list_tool_usage_insights(self, query: ToolUsageInsightQuery | None = None) -> list[ToolUsageInsight]:
        """Aggregate tool-usage insights from action rows."""
        request = query or ToolUsageInsightQuery()
        insight = build_tool_usage_insight(
            rows=self._tool_usage_rows(),
            coverage_rows=self._tool_usage_provider_coverage_rows(),
            query=request,
            materialized_at=datetime.now(UTC).isoformat(),
        )
        return [insight]

    def _tool_usage_rows(self) -> list[ToolUsageRow]:
        rows = self._conn.execute(
            """
            SELECT
                s.origin AS origin,
                COALESCE(NULLIF(LOWER(a.tool_name), ''), 'unknown') AS normalized_tool_name,
                COALESCE(NULLIF(a.semantic_type, ''), 'tool_use') AS action_kind,
                COUNT(*) AS call_count,
                COUNT(DISTINCT s.session_id) AS session_count,
                COUNT(DISTINCT a.message_id) AS message_count,
                COUNT(DISTINCT a.tool_use_block_id) AS distinct_tool_ids,
                SUM(CASE WHEN a.tool_path IS NOT NULL AND a.tool_path != '' THEN 1 ELSE 0 END) AS affected_path_calls,
                SUM(CASE WHEN a.output_text IS NOT NULL AND a.output_text != '' THEN 1 ELSE 0 END) AS output_text_calls
            FROM actions a
            JOIN sessions s ON s.session_id = a.session_id
            GROUP BY s.origin, normalized_tool_name, action_kind
            ORDER BY call_count DESC, s.origin ASC, normalized_tool_name ASC
            """
        ).fetchall()
        return [
            {
                "source_name": _provider_for_origin(str(row["origin"])).value,
                "normalized_tool_name": str(row["normalized_tool_name"] or "unknown"),
                "action_kind": str(row["action_kind"] or "tool_use"),
                "call_count": int(row["call_count"] or 0),
                "session_count": int(row["session_count"] or 0),
                "message_count": int(row["message_count"] or 0),
                "distinct_tool_ids": int(row["distinct_tool_ids"] or 0),
                "affected_path_calls": int(row["affected_path_calls"] or 0),
                "output_text_calls": int(row["output_text_calls"] or 0),
            }
            for row in rows
        ]

    def _tool_usage_provider_coverage_rows(self) -> list[ToolUsageProviderCoverageRow]:
        rows = self._conn.execute(
            """
            SELECT
                s.origin AS origin,
                COUNT(DISTINCT s.session_id) AS session_count,
                COUNT(a.tool_use_block_id) AS action_count,
                COUNT(DISTINCT COALESCE(NULLIF(LOWER(a.tool_name), ''), 'unknown')) AS distinct_tool_count,
                COUNT(DISTINCT COALESCE(NULLIF(a.semantic_type, ''), 'tool_use')) AS distinct_action_kind_count,
                COUNT(a.tool_use_block_id) AS has_tool_id_signal,
                SUM(CASE WHEN a.tool_path IS NOT NULL AND a.tool_path != '' THEN 1 ELSE 0 END) AS has_affected_paths_signal,
                SUM(CASE WHEN a.output_text IS NOT NULL AND a.output_text != '' THEN 1 ELSE 0 END) AS has_output_text_signal
            FROM sessions s
            LEFT JOIN actions a ON a.session_id = s.session_id
            GROUP BY s.origin
            ORDER BY action_count DESC, session_count DESC, s.origin ASC
            """
        ).fetchall()
        return [
            {
                "source_name": _provider_for_origin(str(row["origin"])).value,
                "session_count": int(row["session_count"] or 0),
                "action_count": int(row["action_count"] or 0),
                "distinct_tool_count": int(row["distinct_tool_count"] or 0),
                "distinct_action_kind_count": int(row["distinct_action_kind_count"] or 0),
                "has_tool_id_signal": int(row["has_tool_id_signal"] or 0),
                "has_affected_paths_signal": int(row["has_affected_paths_signal"] or 0),
                "has_output_text_signal": int(row["has_output_text_signal"] or 0),
            }
            for row in rows
        ]

    def list_archive_coverage_insights(
        self,
        *,
        group_by: str = "provider",
        provider: str | None = None,
        since_ms: int | None = None,
        until_ms: int | None = None,
        limit: int | None = None,
        offset: int = 0,
    ) -> list[ArchiveCoverageInsight]:
        """Aggregate archive coverage from index tables."""
        origin = _origin_for_provider_value(provider)
        if group_by == "provider":
            return self._provider_coverage_insights(origin=origin, limit=limit, offset=offset)
        if group_by == "day":
            return self._time_bucket_coverage_insights(
                bucket_format="%Y-%m-%d",
                group_by="day",
                origin=origin,
                since_ms=since_ms,
                until_ms=until_ms,
                limit=limit,
                offset=offset,
            )
        if group_by == "week":
            return self._time_bucket_coverage_insights(
                bucket_format="%Y-W%W",
                group_by="week",
                origin=origin,
                since_ms=since_ms,
                until_ms=until_ms,
                limit=limit,
                offset=offset,
            )
        raise ValueError("archive coverage group_by must be one of: provider, day, week")

    def _provider_coverage_insights(
        self,
        *,
        origin: str | None,
        limit: int | None,
        offset: int,
    ) -> list[ArchiveCoverageInsight]:
        where = ""
        params: list[object] = []
        if origin is not None:
            where = "WHERE s.origin = ?"
            params.append(origin)
        pagination = "" if limit is None else " LIMIT ? OFFSET ?"
        if limit is not None:
            params.extend([max(int(limit), 0), max(int(offset), 0)])
        rows = self._conn.execute(
            f"""
            SELECT
                s.origin,
                COUNT(*) AS session_count,
                SUM(s.message_count) AS message_count,
                SUM(s.user_message_count) AS user_message_count,
                SUM(s.assistant_message_count) AS assistant_message_count,
                SUM(s.user_word_count) AS user_word_sum,
                SUM(s.assistant_word_count) AS assistant_word_sum,
                SUM(s.tool_use_count) AS tool_use_count,
                SUM(s.thinking_count) AS thinking_count,
                SUM(CASE WHEN s.tool_use_count > 0 THEN 1 ELSE 0 END) AS sessions_with_tools,
                SUM(CASE WHEN s.thinking_count > 0 THEN 1 ELSE 0 END) AS sessions_with_thinking
            FROM sessions s
            {where}
            GROUP BY s.origin
            ORDER BY session_count DESC, s.origin
            {pagination}
            """,
            tuple(params),
        ).fetchall()
        return [_provider_coverage_from_archive_row(row) for row in rows]

    def _time_bucket_coverage_insights(
        self,
        *,
        bucket_format: str,
        group_by: str,
        origin: str | None,
        since_ms: int | None,
        until_ms: int | None,
        limit: int | None,
        offset: int,
    ) -> list[ArchiveCoverageInsight]:
        where: list[str] = []
        params: list[object] = []
        if origin is not None:
            where.append("s.origin = ?")
            params.append(origin)
        if since_ms is not None:
            where.append("s.sort_key_ms >= ?")
            params.append(since_ms)
        if until_ms is not None:
            where.append("s.sort_key_ms <= ?")
            params.append(until_ms)
        clause = "WHERE " + " AND ".join(where) if where else ""
        pagination = "" if limit is None else " LIMIT ? OFFSET ?"
        if limit is not None:
            params.extend([max(int(limit), 0), max(int(offset), 0)])
        rows = self._conn.execute(
            f"""
            SELECT
                strftime('{bucket_format}', s.sort_key_ms / 1000, 'unixepoch') AS bucket,
                COUNT(DISTINCT s.session_id) AS session_count,
                COUNT(DISTINCT COALESCE(s.root_session_id, s.session_id)) AS logical_session_count,
                SUM(s.message_count) AS message_count,
                SUM(s.word_count) AS total_words,
                SUM(COALESCE(sp.cost_usd, 0.0)) AS total_cost_usd,
                SUM(COALESCE(sp.duration_ms, 0)) AS total_duration_ms,
                SUM(COALESCE(sp.duration_ms, 0)) AS total_wall_duration_ms,
                MAX(s.sort_key_ms) AS source_sort_key_ms
            FROM sessions s
            LEFT JOIN session_profiles sp ON sp.session_id = s.session_id
            {clause}
            GROUP BY bucket
            HAVING bucket IS NOT NULL
            ORDER BY bucket DESC
            {pagination}
            """,
            tuple(params),
        ).fetchall()
        return [
            ArchiveCoverageInsight(
                group_by=group_by,
                bucket=str(row["bucket"]),
                session_count=int(row["session_count"] or 0),
                logical_session_count=int(row["logical_session_count"] or 0),
                message_count=int(row["message_count"] or 0),
                total_cost_usd=float(row["total_cost_usd"] or 0.0),
                total_duration_ms=int(row["total_duration_ms"] or 0),
                total_wall_duration_ms=int(row["total_wall_duration_ms"] or 0),
                total_words=int(row["total_words"] or 0),
                work_event_breakdown=_coverage_work_event_breakdown(
                    self._conn,
                    str(row["bucket"]),
                    bucket_format,
                    origin=origin,
                    since_ms=since_ms,
                    until_ms=until_ms,
                ),
                repos_active=_coverage_repos_active(
                    self._conn,
                    str(row["bucket"]),
                    bucket_format,
                    origin=origin,
                    since_ms=since_ms,
                    until_ms=until_ms,
                ),
                provider_breakdown=_coverage_provider_breakdown(
                    self._conn,
                    str(row["bucket"]),
                    bucket_format,
                    origin=origin,
                    since_ms=since_ms,
                    until_ms=until_ms,
                ),
                provenance=ArchiveInsightProvenance(
                    materializer_version=1,
                    materialized_at=_iso_from_ms(row["source_sort_key_ms"]) or "1970-01-01T00:00:00Z",
                    source_updated_at=_iso_from_ms(row["source_sort_key_ms"]),
                    source_sort_key=(
                        float(row["source_sort_key_ms"]) / 1000.0 if row["source_sort_key_ms"] is not None else None
                    ),
                ),
            )
            for row in rows
        ]

    def set_user_metadata(self, session_ids: tuple[str, ...], pairs: tuple[tuple[str, object], ...]) -> int:
        """Set human-owned session metadata in archive user.db."""
        user_db_path = self.user_db_path
        initialize_archive_database(user_db_path, ArchiveTier.USER)
        user_conn = sqlite3.connect(user_db_path)
        try:
            changed = 0
            now_ms = int(datetime.now(UTC).timestamp() * 1000)
            with user_conn:
                for session_id in tuple(
                    dict.fromkeys(self.resolve_session_id(session_id) for session_id in session_ids)
                ):
                    for key, value in pairs:
                        normalized_key = key.strip()
                        if not normalized_key:
                            raise ValueError("metadata key cannot be empty")
                        value_json = json.dumps(value, ensure_ascii=False)
                        existing = user_conn.execute(
                            "SELECT created_at_ms, value_json FROM session_metadata WHERE session_id = ? AND key = ?",
                            (session_id, normalized_key),
                        ).fetchone()
                        if existing is not None and str(existing[1]) == value_json:
                            continue
                        created_at_ms = int(existing[0]) if existing is not None else now_ms
                        before = user_conn.total_changes
                        user_conn.execute(
                            """
                            INSERT INTO session_metadata (session_id, key, value_json, created_at_ms, updated_at_ms)
                            VALUES (?, ?, ?, ?, ?)
                            ON CONFLICT(session_id, key) DO UPDATE SET
                                value_json = excluded.value_json,
                                updated_at_ms = excluded.updated_at_ms
                            """,
                            (
                                session_id,
                                normalized_key,
                                value_json,
                                created_at_ms,
                                now_ms,
                            ),
                        )
                        changed += user_conn.total_changes - before
        finally:
            user_conn.close()
        return changed

    def read_user_metadata(self, session_id: str) -> dict[str, object]:
        """Read human-owned metadata for one archive session."""
        resolved_session_id = self.resolve_session_id(session_id)
        if not self.user_db_path.exists():
            return {}
        user_conn = sqlite3.connect(f"file:{self.user_db_path}?mode=ro", uri=True)
        try:
            rows = user_conn.execute(
                """
                SELECT key, value_json
                FROM session_metadata
                WHERE session_id = ?
                ORDER BY key
                """,
                (resolved_session_id,),
            ).fetchall()
        finally:
            user_conn.close()
        decoded: dict[str, object] = {}
        for key, value_json in rows:
            try:
                decoded[str(key)] = json.loads(str(value_json))
            except json.JSONDecodeError:
                decoded[str(key)] = str(value_json)
        return decoded

    def delete_user_metadata(self, session_id: str, key: str) -> int:
        """Delete one user metadata key from archive user.db."""
        resolved_session_id = self.resolve_session_id(session_id)
        normalized_key = key.strip()
        if not normalized_key:
            raise ValueError("metadata key cannot be empty")
        if not self.user_db_path.exists():
            return 0
        user_conn = sqlite3.connect(self.user_db_path)
        try:
            with user_conn:
                cursor = user_conn.execute(
                    "DELETE FROM session_metadata WHERE session_id = ? AND key = ?",
                    (resolved_session_id, normalized_key),
                )
                return max(int(cursor.rowcount), 0)
        finally:
            user_conn.close()

    def add_mark(self, target_type: str, target_id: str, mark_type: str) -> bool:
        """Add one user mark to archive user.db."""
        storage_target_type = _user_target_type_to_storage(target_type)
        initialize_archive_database(self.user_db_path, ArchiveTier.USER)
        user_conn = sqlite3.connect(self.user_db_path)
        try:
            exists = (
                user_conn.execute(
                    """
                    SELECT 1 FROM marks
                    WHERE target_type = ? AND target_id = ? AND mark_type = ?
                    """,
                    (storage_target_type, target_id, mark_type),
                ).fetchone()
                is not None
            )
            with user_conn:
                upsert_mark(user_conn, storage_target_type, target_id, mark_type)
            return not exists
        finally:
            user_conn.close()

    def remove_mark(self, target_type: str, target_id: str, mark_type: str) -> bool:
        """Remove one user mark from archive user.db."""
        if not self.user_db_path.exists():
            return False
        storage_target_type = _user_target_type_to_storage(target_type)
        user_conn = sqlite3.connect(self.user_db_path)
        try:
            with user_conn:
                cursor = user_conn.execute(
                    """
                    DELETE FROM marks
                    WHERE target_type = ? AND target_id = ? AND mark_type = ?
                    """,
                    (storage_target_type, target_id, mark_type),
                )
                deleted = int(cursor.rowcount) > 0
                if deleted:
                    mark_assertion_status(
                        user_conn,
                        assertion_id_for_mark(storage_target_type, target_id, mark_type),
                        "deleted",
                    )
                return deleted
        finally:
            user_conn.close()

    def list_marks(
        self,
        *,
        mark_type: str | None = None,
        target_type: str | None = None,
        target_id: str | None = None,
    ) -> list[dict[str, str]]:
        """List user marks from archive user.db."""
        if not self.user_db_path.exists():
            return []
        where: list[str] = []
        params: list[object] = []
        if mark_type:
            where.append("mark_type = ?")
            params.append(mark_type)
        if target_type:
            where.append("target_type = ?")
            params.append(_user_target_type_to_storage(target_type))
        if target_id:
            where.append("target_id = ?")
            params.append(target_id)
        sql = "SELECT target_type, target_id, mark_type, created_at_ms FROM marks"
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += " ORDER BY created_at_ms DESC, mark_id"
        user_conn = sqlite3.connect(f"file:{self.user_db_path}?mode=ro", uri=True)
        try:
            rows = user_conn.execute(sql, tuple(params)).fetchall()
        finally:
            user_conn.close()
        return [
            {
                "target_type": _user_target_type_from_storage(str(row[0])),
                "target_id": str(row[1]),
                "session_id": _user_mark_session_id(str(row[0]), str(row[1])),
                "message_id": str(row[1]) if str(row[0]) == "message" else "",
                "mark_type": str(row[2]),
                "created_at": str(row[3]),
            }
            for row in rows
        ]

    def save_annotation(self, annotation_id: str, target_type: str, target_id: str, note_text: str) -> bool:
        """Create or update one annotation in archive user.db."""
        storage_target_type = _user_target_type_to_storage(target_type)
        initialize_archive_database(self.user_db_path, ArchiveTier.USER)
        user_conn = sqlite3.connect(self.user_db_path)
        try:
            exists = (
                user_conn.execute(
                    "SELECT 1 FROM annotations WHERE annotation_id = ?",
                    (annotation_id,),
                ).fetchone()
                is not None
            )
            with user_conn:
                upsert_annotation(
                    user_conn,
                    storage_target_type,
                    target_id,
                    note_text,
                    annotation_id=annotation_id,
                )
            return not exists
        finally:
            user_conn.close()

    def get_annotation(self, annotation_id: str) -> dict[str, str] | None:
        """Read one annotation from archive user.db."""
        rows = self.list_annotations(annotation_id=annotation_id)
        return rows[0] if rows else None

    def list_annotations(
        self,
        *,
        annotation_id: str | None = None,
        target_type: str | None = None,
        target_id: str | None = None,
        session_id: str | None = None,
    ) -> list[dict[str, str]]:
        """List annotations from archive user.db.

        When ``session_id`` is supplied (and no explicit target filter),
        the result includes both the session-target annotation and every
        message-target annotation whose native message id is prefixed by the
        session id (``session_id:message_native_id``). This mirrors the read model
        contract where annotations on messages belonging to a session were
        listed under that session.
        """
        if not self.user_db_path.exists():
            return []
        where: list[str] = []
        params: list[object] = []
        if annotation_id:
            where.append("annotation_id = ?")
            params.append(annotation_id)
        if session_id and target_type is None and target_id is None:
            where.append(
                "((target_type = 'session' AND target_id = ?)"
                " OR (target_type = 'message' AND (target_id = ? OR target_id LIKE ? ESCAPE '\\')))"
            )
            like_prefix = session_id.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_") + ":%"
            params.extend([session_id, session_id, like_prefix])
        if target_type:
            where.append("target_type = ?")
            params.append(_user_target_type_to_storage(target_type))
        if target_id:
            where.append("target_id = ?")
            params.append(target_id)
        sql = "SELECT annotation_id, target_type, target_id, body, created_at_ms, updated_at_ms FROM annotations"
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += " ORDER BY updated_at_ms DESC, annotation_id"
        user_conn = sqlite3.connect(f"file:{self.user_db_path}?mode=ro", uri=True)
        try:
            rows = user_conn.execute(sql, tuple(params)).fetchall()
        finally:
            user_conn.close()
        return [
            {
                "annotation_id": str(row[0]),
                "target_type": _user_target_type_from_storage(str(row[1])),
                "target_id": str(row[2]),
                "session_id": _user_mark_session_id(str(row[1]), str(row[2])),
                "message_id": str(row[2]) if str(row[1]) == "message" else "",
                "note_text": str(row[3]),
                "created_at": str(row[4]),
                "updated_at": str(row[5]),
            }
            for row in rows
        ]

    def delete_annotation(self, annotation_id: str) -> bool:
        """Delete one annotation from archive user.db."""
        if not self.user_db_path.exists():
            return False
        user_conn = sqlite3.connect(self.user_db_path)
        try:
            with user_conn:
                cursor = user_conn.execute("DELETE FROM annotations WHERE annotation_id = ?", (annotation_id,))
                deleted = int(cursor.rowcount) > 0
                if deleted:
                    mark_assertion_status(user_conn, assertion_id_for_annotation(annotation_id), "deleted")
                return deleted
        finally:
            user_conn.close()

    def save_view(self, view_id: str, name: str, query_json: str) -> bool:
        """Create or update one saved view in archive user.db."""
        normalized_name = name.strip()
        if not normalized_name:
            raise ValueError("name must not be empty")
        query = json.loads(query_json)
        if not isinstance(query, dict):
            raise ValueError("query_json must encode an object")
        initialize_archive_database(self.user_db_path, ArchiveTier.USER)
        user_conn = sqlite3.connect(self.user_db_path)
        try:
            existing = user_conn.execute(
                "SELECT created_at_ms FROM saved_views WHERE view_id = ?",
                (view_id,),
            ).fetchone()
            with user_conn:
                upsert_saved_view(user_conn, normalized_name, query, view_id=view_id)
            return existing is None
        finally:
            user_conn.close()

    def get_view(self, view_id: str) -> dict[str, str] | None:
        """Get one saved view by id from archive user.db."""
        rows = self._list_views(where="WHERE view_id = ?", params=(view_id,))
        return rows[0] if rows else None

    def get_view_by_name(self, name: str) -> dict[str, str] | None:
        """Get one saved view by name from archive user.db."""
        rows = self._list_views(where="WHERE name = ?", params=(name,))
        return rows[0] if rows else None

    def list_views(self) -> list[dict[str, str]]:
        """List saved views from archive user.db."""
        return self._list_views()

    def _list_views(self, *, where: str = "", params: tuple[object, ...] = ()) -> list[dict[str, str]]:
        if not self.user_db_path.exists():
            return []
        user_conn = sqlite3.connect(f"file:{self.user_db_path}?mode=ro", uri=True)
        try:
            rows = user_conn.execute(
                f"""
                SELECT view_id, name, query_json, created_at_ms
                FROM saved_views
                {where}
                ORDER BY created_at_ms DESC, view_id
                """,
                params,
            ).fetchall()
        finally:
            user_conn.close()
        return [
            {
                "view_id": str(row[0]),
                "name": str(row[1]),
                "query_json": str(row[2]),
                "created_at": str(row[3]),
            }
            for row in rows
        ]

    def delete_view(self, view_id: str) -> bool:
        """Delete one saved view from archive user.db."""
        if not self.user_db_path.exists():
            return False
        user_conn = sqlite3.connect(self.user_db_path)
        try:
            with user_conn:
                cursor = user_conn.execute("DELETE FROM saved_views WHERE view_id = ?", (view_id,))
                deleted = int(cursor.rowcount) > 0
                if deleted:
                    mark_assertion_status(user_conn, assertion_id_for_saved_view(view_id), "deleted")
                return deleted
        finally:
            user_conn.close()

    def save_recall_pack(
        self,
        pack_id: str,
        label: str,
        session_ids_json: str,
        payload_json: str,
    ) -> bool:
        """Create or update one recall pack in archive user.db."""
        payload = json.loads(payload_json)
        if not isinstance(payload, dict):
            raise ValueError("payload_json must encode an object")
        payload = dict(payload)
        payload["session_ids_json"] = session_ids_json
        initialize_archive_database(self.user_db_path, ArchiveTier.USER)
        user_conn = sqlite3.connect(self.user_db_path)
        try:
            existing = user_conn.execute(
                "SELECT created_at_ms FROM recall_packs WHERE recall_pack_id = ?",
                (pack_id,),
            ).fetchone()
            with user_conn:
                upsert_recall_pack(user_conn, label, payload, recall_pack_id=pack_id)
            return existing is None
        finally:
            user_conn.close()

    def get_recall_pack(self, pack_id: str) -> dict[str, str] | None:
        """Get one recall pack by id from archive user.db."""
        rows = self._list_recall_packs(where="WHERE recall_pack_id = ?", params=(pack_id,))
        return rows[0] if rows else None

    def list_recall_packs(self) -> list[dict[str, str]]:
        """List recall packs from archive user.db."""
        return self._list_recall_packs()

    def _list_recall_packs(self, *, where: str = "", params: tuple[object, ...] = ()) -> list[dict[str, str]]:
        if not self.user_db_path.exists():
            return []
        user_conn = sqlite3.connect(f"file:{self.user_db_path}?mode=ro", uri=True)
        try:
            rows = user_conn.execute(
                f"""
                SELECT recall_pack_id, name, payload_json, created_at_ms
                FROM recall_packs
                {where}
                ORDER BY created_at_ms DESC, recall_pack_id
                """,
                params,
            ).fetchall()
        finally:
            user_conn.close()
        out: list[dict[str, str]] = []
        for row in rows:
            try:
                payload = json.loads(str(row[2]))
            except json.JSONDecodeError:
                payload = {}
            if not isinstance(payload, dict):
                payload = {}
            session_ids_json = payload.pop("session_ids_json", "[]")
            out.append(
                {
                    "pack_id": str(row[0]),
                    "label": str(row[1]),
                    "session_ids_json": str(session_ids_json),
                    "payload_json": json.dumps(payload, sort_keys=True, separators=(",", ":")),
                    "created_at": str(row[3]),
                }
            )
        return out

    def delete_recall_pack(self, pack_id: str) -> bool:
        """Delete one recall pack from archive user.db."""
        if not self.user_db_path.exists():
            return False
        user_conn = sqlite3.connect(self.user_db_path)
        try:
            with user_conn:
                cursor = user_conn.execute("DELETE FROM recall_packs WHERE recall_pack_id = ?", (pack_id,))
                deleted = int(cursor.rowcount) > 0
                if deleted:
                    mark_assertion_status(user_conn, assertion_id_for_recall_pack(pack_id), "deleted")
                return deleted
        finally:
            user_conn.close()

    def save_workspace(
        self,
        *,
        workspace_id: str,
        name: str,
        mode: str,
        open_targets_json: str,
        layout_json: str,
        active_target_json: str,
    ) -> bool:
        """Create or update one reader workspace in archive user.db."""
        settings: dict[str, object] = {
            "mode": mode,
            "open_targets_json": open_targets_json,
            "layout_json": layout_json,
            "active_target_json": active_target_json,
        }
        initialize_archive_database(self.user_db_path, ArchiveTier.USER)
        user_conn = sqlite3.connect(self.user_db_path)
        try:
            existing = user_conn.execute(
                "SELECT created_at_ms FROM workspaces WHERE workspace_id = ?",
                (workspace_id,),
            ).fetchone()
            with user_conn:
                upsert_workspace(user_conn, name, settings, workspace_id=workspace_id)
            return existing is None
        finally:
            user_conn.close()

    def get_workspace(self, workspace_id: str) -> dict[str, str] | None:
        """Get one workspace by id from archive user.db."""
        rows = self._list_workspaces(where="WHERE workspace_id = ?", params=(workspace_id,))
        return rows[0] if rows else None

    def list_workspaces(self) -> list[dict[str, str]]:
        """List workspaces from archive user.db."""
        return self._list_workspaces()

    def _list_workspaces(self, *, where: str = "", params: tuple[object, ...] = ()) -> list[dict[str, str]]:
        if not self.user_db_path.exists():
            return []
        user_conn = sqlite3.connect(f"file:{self.user_db_path}?mode=ro", uri=True)
        try:
            rows = user_conn.execute(
                f"""
                SELECT workspace_id, name, settings_json, created_at_ms, updated_at_ms
                FROM workspaces
                {where}
                ORDER BY updated_at_ms DESC, workspace_id
                """,
                params,
            ).fetchall()
        finally:
            user_conn.close()
        out: list[dict[str, str]] = []
        for row in rows:
            try:
                settings = json.loads(str(row[2]))
            except json.JSONDecodeError:
                settings = {}
            if not isinstance(settings, dict):
                settings = {}
            out.append(
                {
                    "workspace_id": str(row[0]),
                    "name": str(row[1]),
                    "mode": str(settings.get("mode") or ""),
                    "open_targets_json": str(settings.get("open_targets_json") or "[]"),
                    "layout_json": str(settings.get("layout_json") or "{}"),
                    "active_target_json": str(settings.get("active_target_json") or "{}"),
                    "created_at": str(row[3]),
                    "updated_at": str(row[4]),
                }
            )
        return out

    def delete_workspace(self, workspace_id: str) -> bool:
        """Delete one workspace from archive user.db."""
        if not self.user_db_path.exists():
            return False
        user_conn = sqlite3.connect(self.user_db_path)
        try:
            with user_conn:
                cursor = user_conn.execute("DELETE FROM workspaces WHERE workspace_id = ?", (workspace_id,))
                deleted = int(cursor.rowcount) > 0
                if deleted:
                    mark_assertion_status(user_conn, assertion_id_for_workspace(workspace_id), "deleted")
                return deleted
        finally:
            user_conn.close()

    def record_correction(
        self,
        session_id: str,
        kind: str,
        payload: dict[str, str],
        *,
        note: str | None = None,
    ) -> LearningCorrection:
        """Record one learning correction in archive user.db."""
        resolved_session_id = self.resolve_session_id(session_id)
        correction_kind = parse_correction_kind(kind)
        initialize_archive_database(self.user_db_path, ArchiveTier.USER)
        now_ms = int(datetime.now(UTC).timestamp() * 1000)
        correction_id = f"correction:{resolved_session_id}:{correction_kind.value}"
        stored_payload = {"payload": dict(payload), "note": note}
        user_conn = sqlite3.connect(self.user_db_path)
        try:
            existing = user_conn.execute(
                """
                SELECT created_at_ms
                FROM corrections
                WHERE target_type = 'session' AND target_id = ? AND correction_type = ?
                """,
                (resolved_session_id, correction_kind.value),
            ).fetchone()
            created_at_ms = int(existing[0]) if existing is not None else now_ms
            with user_conn:
                user_conn.execute(
                    """
                    INSERT INTO corrections (
                        correction_id, target_type, target_id, correction_type,
                        payload_json, created_at_ms, updated_at_ms
                    ) VALUES (?, 'session', ?, ?, ?, ?, ?)
                    ON CONFLICT(target_type, target_id, correction_type) DO UPDATE SET
                        payload_json = excluded.payload_json,
                        updated_at_ms = excluded.updated_at_ms
                    """,
                    (
                        correction_id,
                        resolved_session_id,
                        correction_kind.value,
                        json.dumps(stored_payload, sort_keys=True),
                        created_at_ms,
                        now_ms,
                    ),
                )
                upsert_assertion(
                    user_conn,
                    assertion_id=assertion_id_for_correction(correction_id),
                    target_ref=f"insight:{resolved_session_id}",
                    kind=AssertionKind.CORRECTION,
                    key=correction_kind.value,
                    value=stored_payload,
                    body_text=note,
                    author_kind="user",
                    now_ms=now_ms,
                )
        finally:
            user_conn.close()
        listed = self.list_corrections(session_id=resolved_session_id, kind=correction_kind.value)
        if not listed:
            raise KeyError(correction_id)
        return listed[0]

    def list_corrections(self, *, session_id: str | None = None, kind: str | None = None) -> list[LearningCorrection]:
        """List learning corrections from archive user.db."""
        if not self.user_db_path.exists():
            return []
        resolved_session_id = self.resolve_session_id(session_id) if session_id else None
        correction_kind = parse_correction_kind(kind).value if kind is not None else None
        where = ["target_type = 'session'"]
        params: list[object] = []
        if resolved_session_id is not None:
            where.append("target_id = ?")
            params.append(resolved_session_id)
        if correction_kind is not None:
            where.append("correction_type = ?")
            params.append(correction_kind)
        user_conn = sqlite3.connect(f"file:{self.user_db_path}?mode=ro", uri=True)
        try:
            rows = user_conn.execute(
                f"""
                SELECT target_id, correction_type, payload_json, updated_at_ms
                FROM corrections
                WHERE {" AND ".join(where)}
                ORDER BY updated_at_ms DESC, correction_id
                """,
                tuple(params),
            ).fetchall()
        finally:
            user_conn.close()
        return [_learning_correction_from_archive_row(row) for row in rows]

    def delete_correction(self, session_id: str, kind: str) -> bool:
        """Delete one learning correction from archive user.db."""
        resolved_session_id = self.resolve_session_id(session_id)
        correction_kind = parse_correction_kind(kind)
        if not self.user_db_path.exists():
            return False
        user_conn = sqlite3.connect(self.user_db_path)
        try:
            with user_conn:
                row = user_conn.execute(
                    """
                    SELECT correction_id
                    FROM corrections
                    WHERE target_type = 'session' AND target_id = ? AND correction_type = ?
                    """,
                    (resolved_session_id, correction_kind.value),
                ).fetchone()
                cursor = user_conn.execute(
                    """
                    DELETE FROM corrections
                    WHERE target_type = 'session' AND target_id = ? AND correction_type = ?
                    """,
                    (resolved_session_id, correction_kind.value),
                )
                deleted = int(cursor.rowcount) > 0
                if deleted and row is not None:
                    mark_assertion_status(user_conn, assertion_id_for_correction(str(row[0])), "deleted")
                return deleted
        finally:
            user_conn.close()

    def clear_corrections(self, session_id: str) -> int:
        """Delete all learning corrections for one archive session."""
        resolved_session_id = self.resolve_session_id(session_id)
        if not self.user_db_path.exists():
            return 0
        user_conn = sqlite3.connect(self.user_db_path)
        try:
            with user_conn:
                rows = user_conn.execute(
                    "SELECT correction_id FROM corrections WHERE target_type = 'session' AND target_id = ?",
                    (resolved_session_id,),
                ).fetchall()
                cursor = user_conn.execute(
                    "DELETE FROM corrections WHERE target_type = 'session' AND target_id = ?",
                    (resolved_session_id,),
                )
                deleted_count = max(int(cursor.rowcount), 0)
                if deleted_count:
                    for row in rows:
                        mark_assertion_status(user_conn, assertion_id_for_correction(str(row[0])), "deleted")
                return deleted_count
        finally:
            user_conn.close()

    def post_blackboard_note(
        self,
        body: str,
        *,
        target_type: str | None = None,
        target_id: str | None = None,
        note_id: str | None = None,
        author_ref: str | None = None,
        author_kind: str = "user",
        evidence_refs: tuple[str, ...] = (),
        staleness: dict[str, object] | None = None,
        context_policy: dict[str, object] | None = None,
    ) -> ArchiveBlackboardNoteEnvelope:
        """Insert-or-update one blackboard note in archive user.db."""
        initialize_archive_database(self.user_db_path, ArchiveTier.USER)
        user_conn = sqlite3.connect(self.user_db_path)
        try:
            envelope = upsert_blackboard_note(
                user_conn,
                body,
                target_type=target_type,
                target_id=target_id,
                note_id=note_id,
                author_ref=author_ref,
                author_kind=author_kind,
                evidence_refs=evidence_refs,
                staleness=staleness,
                context_policy=context_policy,
            )
            user_conn.commit()
            return envelope
        finally:
            user_conn.close()

    def list_blackboard_notes(self, *, limit: int | None = None) -> list[ArchiveBlackboardNoteEnvelope]:
        """List blackboard notes from archive user.db, newest first.

        Mirrored assertion rows own note body/timestamps for write-through
        notes; legacy rows still provide stable note ids and target fields.
        Structured-field decoding (kind/title/scope) is a presentation concern
        handled by ``polylogue.archive.blackboard``.
        """
        if not self.user_db_path.exists():
            return []
        user_conn = sqlite3.connect(f"file:{self.user_db_path}?mode=ro", uri=True)
        try:
            return list_archive_blackboard_note_envelopes(user_conn, limit=limit)
        finally:
            user_conn.close()

    def delete_sessions(self, session_ids: tuple[str, ...]) -> int:
        """Delete rebuildable archive sessions by id.

        User-tier overlays are intentionally left in ``user.db``; the user
        overlay orphan checker owns follow-up visibility for those durable rows.
        """
        resolved_session_ids = tuple(dict.fromkeys(self.resolve_session_id(session_id) for session_id in session_ids))
        if not resolved_session_ids:
            return 0
        conn = sqlite3.connect(self.index_db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        deleted = 0
        try:
            with conn:
                for session_id in resolved_session_ids:
                    cursor = conn.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
                    deleted += max(int(cursor.rowcount), 0)
        finally:
            conn.close()
        return deleted

    def _attach_user_tier_if_present(self) -> None:
        if self._user_tier_attached or not self.user_db_path.exists():
            return
        user_db_uri = f"file:{self.user_db_path}?mode=ro" if self._read_only else str(self.user_db_path)
        self._conn.execute("ATTACH DATABASE ? AS user_tier", (user_db_uri,))
        self._user_tier_attached = True
        self._tags_relation = _all_session_tags_sql()

    def count_sessions(
        self,
        *,
        origin: str | None = None,
        origins: tuple[str, ...] = (),
        excluded_origins: tuple[str, ...] = (),
        tags: tuple[str, ...] = (),
        excluded_tags: tuple[str, ...] = (),
        repo_names: tuple[str, ...] = (),
        has_types: tuple[str, ...] = (),
        has_tool_use: bool = False,
        has_thinking: bool = False,
        has_paste: bool = False,
        tool_terms: tuple[str, ...] = (),
        excluded_tool_terms: tuple[str, ...] = (),
        action_terms: tuple[str, ...] = (),
        excluded_action_terms: tuple[str, ...] = (),
        action_sequence: tuple[str, ...] = (),
        action_text_terms: tuple[str, ...] = (),
        referenced_paths: tuple[str, ...] = (),
        cwd_prefix: str | None = None,
        typed_only: bool = False,
        message_type: str | None = None,
        title: str | None = None,
        min_messages: int | None = None,
        max_messages: int | None = None,
        min_words: int | None = None,
        max_words: int | None = None,
        since_ms: int | None = None,
        until_ms: int | None = None,
        since_session_id: str | None = None,
        boolean_predicate: QueryPredicate | None = None,
    ) -> int:
        """Count sessions in the archive index."""
        where, params = _session_filter_clause(
            "s",
            origin=origin,
            origins=origins,
            excluded_origins=excluded_origins,
            tags=tags,
            excluded_tags=excluded_tags,
            repo_names=repo_names,
            has_types=has_types,
            has_tool_use=has_tool_use,
            has_thinking=has_thinking,
            has_paste=has_paste,
            tool_terms=tool_terms,
            excluded_tool_terms=excluded_tool_terms,
            action_terms=action_terms,
            excluded_action_terms=excluded_action_terms,
            action_sequence=action_sequence,
            action_text_terms=action_text_terms,
            referenced_paths=referenced_paths,
            cwd_prefix=cwd_prefix,
            typed_only=typed_only,
            message_type=message_type,
            title=title,
            min_messages=min_messages,
            max_messages=max_messages,
            min_words=min_words,
            max_words=max_words,
            since_ms=since_ms,
            until_ms=until_ms,
            boolean_predicate=boolean_predicate,
            tags_relation=self._tags_relation,
        )
        where, params = _with_since_session_filter(self._conn, where, params, "s", since_session_id=since_session_id)
        return int(self._conn.execute(f"SELECT COUNT(*) FROM sessions s {where}", params).fetchone()[0])

    def session_insight_status(self) -> SessionInsightStatusSnapshot:
        """Return readiness for session insight tables."""
        total_sessions = _count_rows(self._conn, "sessions")
        profile_rows = _count_rows(self._conn, "session_profiles")
        work_event_rows = _count_rows(self._conn, "session_work_events")
        phase_rows = _count_rows(self._conn, "session_phases")
        thread_rows = _count_rows(self._conn, "threads")
        tag_rows = _count_rows(self._conn, "session_tags")
        root_threads = _count_scalar(
            self._conn,
            """
            SELECT COUNT(*)
            FROM sessions
            WHERE parent_session_id IS NULL
               OR parent_session_id = ''
               OR parent_session_id = session_id
            """,
        )
        missing_profiles = _count_scalar(
            self._conn,
            """
            SELECT COUNT(*)
            FROM sessions AS s
            WHERE NOT EXISTS (
                SELECT 1 FROM session_profiles AS p WHERE p.session_id = s.session_id
            )
            """,
        )
        orphan_profiles = _count_scalar(
            self._conn,
            """
            SELECT COUNT(*)
            FROM session_profiles AS p
            WHERE NOT EXISTS (
                SELECT 1 FROM sessions AS s WHERE s.session_id = p.session_id
            )
            """,
        )
        orphan_work_events = _count_scalar(
            self._conn,
            """
            SELECT COUNT(*)
            FROM session_work_events AS e
            WHERE NOT EXISTS (
                SELECT 1 FROM sessions AS s WHERE s.session_id = e.session_id
            )
            """,
        )
        orphan_phases = _count_scalar(
            self._conn,
            """
            SELECT COUNT(*)
            FROM session_phases AS ph
            WHERE NOT EXISTS (
                SELECT 1 FROM sessions AS s WHERE s.session_id = ph.session_id
            )
            """,
        )
        expected_work_events = _count_scalar(
            self._conn,
            "SELECT COALESCE(SUM(work_event_count), 0) FROM session_profiles",
        )
        expected_phases = _count_scalar(
            self._conn,
            "SELECT COALESCE(SUM(phase_count), 0) FROM session_profiles",
        )
        stale_work_event_profiles = _count_scalar(
            self._conn,
            """
            SELECT COUNT(*)
            FROM session_profiles AS p
            WHERE p.work_event_count != (
                SELECT COUNT(*) FROM session_work_events AS e WHERE e.session_id = p.session_id
            )
            """,
        )
        stale_phase_profiles = _count_scalar(
            self._conn,
            """
            SELECT COUNT(*)
            FROM session_profiles AS p
            WHERE p.phase_count != (
                SELECT COUNT(*) FROM session_phases AS ph WHERE ph.session_id = p.session_id
            )
            """,
        )
        missing_materialization = _archive_missing_materialization_counts(self._conn)
        return SessionInsightStatusSnapshot(
            total_sessions=total_sessions,
            root_threads=root_threads,
            profile_row_count=profile_rows,
            work_event_inference_count=work_event_rows,
            phase_inference_count=phase_rows,
            thread_count=thread_rows,
            tag_rollup_count=tag_rows,
            missing_profile_row_count=missing_profiles,
            orphan_profile_row_count=orphan_profiles,
            expected_work_event_inference_count=expected_work_events,
            stale_work_event_inference_count=stale_work_event_profiles,
            orphan_work_event_inference_count=orphan_work_events,
            expected_phase_inference_count=expected_phases,
            stale_phase_inference_count=stale_phase_profiles,
            orphan_phase_inference_count=orphan_phases,
            stale_thread_count=missing_materialization["thread"],
            expected_tag_rollup_count=tag_rows,
            profile_rows_ready=missing_profiles == 0 and orphan_profiles == 0,
            work_event_inference_rows_ready=(
                missing_materialization["work_events"] == 0
                and stale_work_event_profiles == 0
                and orphan_work_events == 0
            ),
            phase_inference_rows_ready=(
                missing_materialization["phases"] == 0 and stale_phase_profiles == 0 and orphan_phases == 0
            ),
            threads_ready=missing_materialization["thread"] == 0 and thread_rows == root_threads,
            tag_rollups_ready=tag_rows > 0 or total_sessions == 0,
        )

    def insight_readiness_report(self, query: InsightReadinessQuery | None = None) -> InsightReadinessReport:
        """Return public insight readiness from tables."""
        request = query or InsightReadinessQuery()
        selected = (
            tuple(normalize_insight_readiness_name(insight) for insight in request.insights)
            if request.insights
            else known_insight_readiness_names()
        )
        status = self.session_insight_status()
        origin_filter = _origin_for_provider_value(request.provider)
        since_ms = _epoch_ms_from_iso(request.since)
        until_ms = _epoch_ms_from_iso(request.until)
        total_sessions = self.count_sessions(origin=origin_filter, since_ms=since_ms, until_ms=until_ms)
        coverage = self._archive_session_provider_coverage(origin=origin_filter, since_ms=since_ms, until_ms=until_ms)
        entries = tuple(
            entry
            for name in selected
            if (
                entry := self._insight_readiness_entry(
                    name,
                    status=status,
                    total_sessions=total_sessions,
                    provider_coverage=coverage,
                    origin=origin_filter,
                    since_ms=since_ms,
                    until_ms=until_ms,
                )
            )
            is not None
        )
        return InsightReadinessReport(
            checked_at=datetime.now(UTC).isoformat(),
            aggregate_verdict=_insight_readiness_aggregate_verdict(entries),
            total_sessions=total_sessions,
            provider=request.provider,
            since=request.since,
            until=request.until,
            insights=entries,
        )

    def audit_insight_rigor(self, query: InsightRigorAuditQuery | None = None) -> InsightRigorAuditReport:
        """Audit insight rigor over read models."""
        request = query or InsightRigorAuditQuery()
        targeted = set(request.insights) if request.insights else None
        entries = []
        for contract in list_rigor_contracts():
            if targeted is not None and contract.insight_name not in targeted:
                continue
            rows = self._rigor_audit_rows(contract.insight_name, limit=max(request.sample_limit, 0))
            entries.append(_audit_one(rows, contract))
        return InsightRigorAuditReport(sample_limit=request.sample_limit, entries=tuple(entries))

    def _rigor_audit_rows(self, insight_name: str, *, limit: int) -> list[object]:
        if insight_name == "session_profiles":
            return list(self.list_session_profile_insights(limit=limit))
        if insight_name == "session_work_events":
            return list(self.list_session_work_event_insights(limit=limit))
        if insight_name == "session_phases":
            return list(self.list_session_phase_insights(limit=limit))
        if insight_name == "threads":
            return list(self.list_thread_insights(limit=limit))
        if insight_name == "session_tag_rollups":
            return list(self.list_session_tag_rollup_insights(limit=limit))
        return []

    def _archive_session_provider_coverage(
        self, *, origin: str | None, since_ms: int | None, until_ms: int | None
    ) -> tuple[InsightProviderCoverage, ...]:
        """Per-provider session distribution for insight readiness coverage."""
        where: list[str] = []
        params: list[object] = []
        if origin is not None:
            where.append("origin = ?")
            params.append(origin)
        if since_ms is not None:
            where.append("COALESCE(updated_at_ms, created_at_ms) >= ?")
            params.append(since_ms)
        if until_ms is not None:
            where.append("COALESCE(updated_at_ms, created_at_ms) <= ?")
            params.append(until_ms)
        clause = "WHERE " + " AND ".join(where) if where else ""
        rows = self._conn.execute(
            f"SELECT origin, COUNT(*) AS n, MIN(created_at_ms) AS lo, MAX(updated_at_ms) AS hi "
            f"FROM sessions {clause} GROUP BY origin ORDER BY n DESC, origin",
            tuple(params),
        ).fetchall()
        return tuple(
            InsightProviderCoverage(
                source_name=_provider_for_origin(str(row["origin"])).value,
                row_count=int(row["n"]),
                min_time=_iso_from_ms(row["lo"]),
                max_time=_iso_from_ms(row["hi"]),
            )
            for row in rows
        )

    def _readiness_session_filter(
        self, *, origin: str | None, since_ms: int | None, until_ms: int | None
    ) -> tuple[str, list[object]]:
        """Build a ``WHERE`` fragment over the joined ``sessions`` (alias ``s``)."""
        clauses: list[str] = []
        params: list[object] = []
        if origin is not None:
            clauses.append("s.origin = ?")
            params.append(origin)
        if since_ms is not None:
            clauses.append("COALESCE(s.updated_at_ms, s.created_at_ms) >= ?")
            params.append(since_ms)
        if until_ms is not None:
            clauses.append("COALESCE(s.updated_at_ms, s.created_at_ms) <= ?")
            params.append(until_ms)
        return (" AND " + " AND ".join(clauses)) if clauses else "", params

    def _archive_materialization_signals(
        self,
        insight_type: str,
        *,
        origin: str | None,
        since_ms: int | None,
        until_ms: int | None,
    ) -> tuple[tuple[InsightVersionCoverage, ...], int, int]:
        """Derive version coverage, incompatible count, and native staleness.

        Reads the ``insight_materialization`` high-water marks for ``insight_type``
        joined to ``sessions``. A row is *incompatible* (legacy) when its
        ``materializer_version`` is below ``SESSION_INSIGHT_MATERIALIZER_VERSION``;
        it is *stale* when its captured ``source_sort_key_ms`` no longer matches the
        live session ``sort_key_ms`` (the native source high-water mark). The
        ``session_profiles.materializer_version``/``source_sort_key`` columns are not
        used here: they are not reliably populated by the canonical rebuild path,
        so the materialization ledger is the authoritative provenance source.
        """
        if not _table_exists(self._conn, "insight_materialization"):
            return ((), 0, 0)
        clause, params = self._readiness_session_filter(origin=origin, since_ms=since_ms, until_ms=until_ms)
        version_rows = self._conn.execute(
            "SELECT im.materializer_version AS version, COUNT(*) AS n "
            "FROM insight_materialization AS im "
            "JOIN sessions AS s ON s.session_id = im.session_id "
            f"WHERE im.insight_type = ?{clause} "
            "GROUP BY im.materializer_version ORDER BY im.materializer_version",
            (insight_type, *params),
        ).fetchall()
        versions = {str(int(row["version"])): int(row["n"]) for row in version_rows}
        incompatible_count = sum(
            count for version, count in versions.items() if int(version) < SESSION_INSIGHT_MATERIALIZER_VERSION
        )
        version_coverage = (
            (
                InsightVersionCoverage(
                    field="materializer_version",
                    current_version=SESSION_INSIGHT_MATERIALIZER_VERSION,
                    versions=versions,
                    incompatible_count=incompatible_count,
                ),
            )
            if versions
            else ()
        )
        stale_row = self._conn.execute(
            "SELECT COUNT(*) AS n "
            "FROM insight_materialization AS im "
            "JOIN sessions AS s ON s.session_id = im.session_id "
            f"WHERE im.insight_type = ?{clause} "
            "AND COALESCE(im.source_sort_key_ms, -1) != COALESCE(s.sort_key_ms, -1)",
            (insight_type, *params),
        ).fetchone()
        stale_count = int(stale_row["n"]) if stale_row is not None else 0
        return (version_coverage, incompatible_count, stale_count)

    def _archive_fallback_coverage(
        self,
        table_name: str,
        column_paths: tuple[tuple[str, str], ...],
        *,
        origin: str | None,
        since_ms: int | None,
        until_ms: int | None,
    ) -> tuple[int, dict[str, int]]:
        """Count rows whose enrichment provenance carries fallback reasons.

        Each insight row stores its fallback markers as JSON arrays under
        ``$.fallback_reasons`` inside one or more payload columns (e.g.
        ``inference_payload_json`` and ``enrichment_payload_json`` on
        ``session_profiles``). A row is *degraded* when any declared
        ``(column, path)`` holds a non-empty array; the row is counted at most
        once regardless of how many columns flag it. ``reason_totals`` sums
        occurrences per reason across every inspected column.
        """
        if not _table_exists(self._conn, table_name):
            return (0, {})
        clause, params = self._readiness_session_filter(origin=origin, since_ms=since_ms, until_ms=until_ms)
        any_terms = " OR ".join(
            f"json_array_length(COALESCE(json_extract(t.{column}, '{path}'), '[]')) > 0"
            for column, path in column_paths
        )
        degraded_row = self._conn.execute(
            f"SELECT COUNT(*) AS n FROM {table_name} AS t "
            "JOIN sessions AS s ON s.session_id = t.session_id "
            f"WHERE ({any_terms}){clause}",
            tuple(params),
        ).fetchone()
        degraded_count = int(degraded_row["n"]) if degraded_row is not None else 0
        reason_totals: dict[str, int] = {}
        for column, path in column_paths:
            rows = self._conn.execute(
                "SELECT value AS reason, COUNT(*) AS occurrences "
                f"FROM {table_name} AS t "
                "JOIN sessions AS s ON s.session_id = t.session_id, "
                f"json_each(COALESCE(json_extract(t.{column}, '{path}'), '[]')) "
                f"WHERE 1=1{clause} GROUP BY value",
                tuple(params),
            ).fetchall()
            for row in rows:
                reason = str(row["reason"])
                reason_totals[reason] = reason_totals.get(reason, 0) + int(row["occurrences"])
        return (degraded_count, dict(sorted(reason_totals.items())))

    def _insight_readiness_entry(
        self,
        name: str,
        *,
        status: SessionInsightStatusSnapshot,
        total_sessions: int,
        provider_coverage: tuple[InsightProviderCoverage, ...] = (),
        origin: str | None = None,
        since_ms: int | None = None,
        until_ms: int | None = None,
    ) -> InsightReadinessEntry | None:
        specs = {
            "session_profiles": (
                "Session Profiles",
                "session_profiles",
                status.profile_row_count,
                total_sessions,
                status.missing_profile_row_count,
                0,
                status.orphan_profile_row_count,
                {"profile_rows_ready": status.profile_rows_ready},
                ("session_profiles",),
            ),
            "session_work_events": (
                "Work Events",
                "session_work_events",
                status.work_event_inference_count,
                status.expected_work_event_inference_count,
                0,
                status.stale_work_event_inference_count,
                status.orphan_work_event_inference_count,
                {"work_event_inference_rows_ready": status.work_event_inference_rows_ready},
                ("session_work_events",),
            ),
            "session_phases": (
                "Session Phases",
                "session_phases",
                status.phase_inference_count,
                status.expected_phase_inference_count,
                0,
                status.stale_phase_inference_count,
                status.orphan_phase_inference_count,
                {"phase_inference_rows_ready": status.phase_inference_rows_ready},
                ("session_phases",),
            ),
            "threads": (
                "Threads",
                "threads",
                status.thread_count,
                status.root_threads,
                0,
                status.stale_thread_count,
                status.orphan_thread_count,
                {"threads_ready": status.threads_ready},
                ("threads", "thread_sessions"),
            ),
            "session_tag_rollups": (
                "Session Tag Rollups",
                "session_tags",
                status.tag_rollup_count,
                status.expected_tag_rollup_count,
                0,
                status.stale_tag_rollup_count,
                0,
                {"tag_rollups_ready": status.tag_rollups_ready},
                ("session_tags",),
            ),
            "archive_coverage": (
                "Archive Coverage",
                "sessions",
                total_sessions,
                total_sessions,
                0,
                0,
                0,
                {},
                ("sessions",),
            ),
        }
        spec = specs.get(name)
        if spec is None:
            return None
        (
            display_name,
            table_name,
            row_count,
            expected_row_count,
            missing_count,
            stale_count,
            orphan_count,
            ready_flags,
            artifact_names,
        ) = spec
        table_present = _table_exists(self._conn, table_name)
        artifacts = tuple(
            InsightStorageArtifact(
                name=artifact,
                present=_table_exists(self._conn, artifact),
                ready=ready_flags[next(iter(ready_flags))] if len(ready_flags) == 1 else None,
            )
            for artifact in artifact_names
        )
        # Provenance-backed insights (profiles, work events, phases) carry their
        # materializer version and source high-water mark in the
        # ``insight_materialization`` ledger; the #1278 fallback taxonomy lives in
        # each session profile's ``provenance_json``. Threads/tags/coverage have no
        # such ledger entry and keep the status-derived staleness only.
        version_coverage: tuple[InsightVersionCoverage, ...] = ()
        incompatible_count = 0
        materialization_type = _INSIGHT_MATERIALIZATION_TYPE.get(name)
        if materialization_type is not None and table_present:
            version_coverage, incompatible_count, native_stale = self._archive_materialization_signals(
                materialization_type, origin=origin, since_ms=since_ms, until_ms=until_ms
            )
            stale_count = native_stale
        degraded_count = 0
        fallback_reason_counts: dict[str, int] = {}
        fallback = _INSIGHT_FALLBACK_PAYLOAD.get(name)
        if fallback is not None and table_present:
            fallback_table, fallback_column_paths = fallback
            degraded_count, fallback_reason_counts = self._archive_fallback_coverage(
                fallback_table,
                fallback_column_paths,
                origin=origin,
                since_ms=since_ms,
                until_ms=until_ms,
            )
        verdict = _archive_insight_readiness_verdict(
            table_present=table_present,
            row_count=row_count,
            expected_row_count=expected_row_count,
            missing_count=missing_count,
            stale_count=stale_count,
            orphan_count=orphan_count,
            incompatible_count=incompatible_count,
            degraded_count=degraded_count,
            ready_flags=ready_flags,
            total_sessions=total_sessions,
        )
        return InsightReadinessEntry(
            insight_name=name,
            display_name=display_name,
            verdict=verdict,
            row_count=row_count,
            expected_row_count=expected_row_count,
            missing_count=missing_count,
            stale_count=stale_count,
            orphan_count=orphan_count,
            incompatible_count=incompatible_count,
            degraded_count=degraded_count,
            fallback_reason_counts=fallback_reason_counts,
            storage_artifacts=artifacts,
            ready_flags=ready_flags,
            provider_coverage=provider_coverage,
            version_coverage=version_coverage,
            evidence=_archive_insight_readiness_evidence(
                row_count=row_count,
                expected_row_count=expected_row_count,
                missing_count=missing_count,
                stale_count=stale_count,
                orphan_count=orphan_count,
                incompatible_count=incompatible_count,
                degraded_count=degraded_count,
                fallback_reason_counts=fallback_reason_counts,
                ready_flags=ready_flags,
            ),
        )

    def list_summaries(
        self,
        *,
        limit: int = 50,
        offset: int = 0,
        origin: str | None = None,
        origins: tuple[str, ...] = (),
        excluded_origins: tuple[str, ...] = (),
        tags: tuple[str, ...] = (),
        excluded_tags: tuple[str, ...] = (),
        repo_names: tuple[str, ...] = (),
        has_types: tuple[str, ...] = (),
        has_tool_use: bool = False,
        has_thinking: bool = False,
        has_paste: bool = False,
        tool_terms: tuple[str, ...] = (),
        excluded_tool_terms: tuple[str, ...] = (),
        action_terms: tuple[str, ...] = (),
        excluded_action_terms: tuple[str, ...] = (),
        action_sequence: tuple[str, ...] = (),
        action_text_terms: tuple[str, ...] = (),
        referenced_paths: tuple[str, ...] = (),
        cwd_prefix: str | None = None,
        typed_only: bool = False,
        message_type: str | None = None,
        title: str | None = None,
        min_messages: int | None = None,
        max_messages: int | None = None,
        min_words: int | None = None,
        max_words: int | None = None,
        session_id: str | None = None,
        since_ms: int | None = None,
        until_ms: int | None = None,
        since_session_id: str | None = None,
        boolean_predicate: QueryPredicate | None = None,
        sample: bool = False,
        sort: str | None = None,
        reverse: bool = False,
    ) -> list[ArchiveSessionSummary]:
        """List session summaries ordered like the normal archive recency view."""
        where, params = _session_filter_clause(
            "s",
            origin=origin,
            origins=origins,
            excluded_origins=excluded_origins,
            tags=tags,
            excluded_tags=excluded_tags,
            repo_names=repo_names,
            has_types=has_types,
            has_tool_use=has_tool_use,
            has_thinking=has_thinking,
            has_paste=has_paste,
            tool_terms=tool_terms,
            excluded_tool_terms=excluded_tool_terms,
            action_terms=action_terms,
            excluded_action_terms=excluded_action_terms,
            action_sequence=action_sequence,
            action_text_terms=action_text_terms,
            referenced_paths=referenced_paths,
            cwd_prefix=cwd_prefix,
            typed_only=typed_only,
            message_type=message_type,
            title=title,
            min_messages=min_messages,
            max_messages=max_messages,
            min_words=min_words,
            max_words=max_words,
            since_ms=since_ms,
            until_ms=until_ms,
            boolean_predicate=boolean_predicate,
            tags_relation=self._tags_relation,
        )
        where, params = _with_since_session_filter(self._conn, where, params, "s", since_session_id=since_session_id)
        if session_id is not None:
            resolved_id = self.resolve_session_id(session_id)
            where = f"{where} AND s.session_id = ?" if where else "WHERE s.session_id = ?"
            params.append(resolved_id)
        order_by = _summary_order_by(sample=sample, sort=sort, reverse=reverse)
        params.extend([limit, 0 if sample else offset])
        rows = self._conn.execute(
            f"""
            SELECT s.session_id, s.native_id, s.origin, s.title, s.created_at_ms, s.updated_at_ms,
                   s.message_count, s.word_count, s.git_branch, s.git_repository_url,
                   COALESCE(
                       (
                           SELECT json_group_array(swd.path)
                           FROM session_working_dirs swd
                           WHERE swd.session_id = s.session_id
                           ORDER BY swd.position, swd.path
                       ),
                       '[]'
                   ) AS working_directories_json,
                   COALESCE(
                       json_group_array(st.tag) FILTER (WHERE st.tag IS NOT NULL),
                       '[]'
                   ) AS tags_json
            FROM sessions s
            LEFT JOIN {self._tags_relation} st
              ON st.session_id = s.session_id
             AND st.tag_source = 'user'
            {where}
            GROUP BY s.session_id
            {order_by}
            LIMIT ? OFFSET ?
            """,
            params,
        ).fetchall()
        return [_summary_from_row(row) for row in rows]

    def search_summaries(
        self,
        query: str,
        *,
        limit: int = 20,
        offset: int = 0,
        sort: str | None = None,
        reverse: bool = False,
        session_id: str | None = None,
        origin: str | None = None,
        origins: tuple[str, ...] = (),
        excluded_origins: tuple[str, ...] = (),
        tags: tuple[str, ...] = (),
        excluded_tags: tuple[str, ...] = (),
        repo_names: tuple[str, ...] = (),
        has_types: tuple[str, ...] = (),
        has_tool_use: bool = False,
        has_thinking: bool = False,
        has_paste: bool = False,
        tool_terms: tuple[str, ...] = (),
        excluded_tool_terms: tuple[str, ...] = (),
        action_terms: tuple[str, ...] = (),
        excluded_action_terms: tuple[str, ...] = (),
        action_sequence: tuple[str, ...] = (),
        action_text_terms: tuple[str, ...] = (),
        referenced_paths: tuple[str, ...] = (),
        cwd_prefix: str | None = None,
        typed_only: bool = False,
        message_type: str | None = None,
        title: str | None = None,
        min_messages: int | None = None,
        max_messages: int | None = None,
        min_words: int | None = None,
        max_words: int | None = None,
        since_ms: int | None = None,
        until_ms: int | None = None,
        since_session_id: str | None = None,
        boolean_predicate: QueryPredicate | None = None,
    ) -> list[ArchiveSessionSearchHit]:
        """Search archive block text and return session-level hits with snippets."""
        match_query = normalize_fts5_query(query)
        if match_query is None:
            # Empty / whitespace / asterisk-only query: no FTS expression to
            # run. Mirror the read model lexical path and return no hits rather
            # than raising ``fts5: syntax error``.
            return []
        # A real query needs the block FTS index. Surface a degraded index as a
        # sanitized DatabaseError (→ 503 "Search index") instead of a raw
        # ``no such table`` 500 or a misleading empty-result 200.
        _ensure_messages_fts_ready(self._conn)
        where, filter_params = _session_filter_clause(
            "s",
            origin=origin,
            origins=origins,
            excluded_origins=excluded_origins,
            tags=tags,
            excluded_tags=excluded_tags,
            repo_names=repo_names,
            has_types=has_types,
            has_tool_use=has_tool_use,
            has_thinking=has_thinking,
            has_paste=has_paste,
            tool_terms=tool_terms,
            excluded_tool_terms=excluded_tool_terms,
            action_terms=action_terms,
            excluded_action_terms=excluded_action_terms,
            action_sequence=action_sequence,
            action_text_terms=action_text_terms,
            referenced_paths=referenced_paths,
            cwd_prefix=cwd_prefix,
            typed_only=typed_only,
            message_type=message_type,
            title=title,
            min_messages=min_messages,
            max_messages=max_messages,
            min_words=min_words,
            max_words=max_words,
            since_ms=since_ms,
            until_ms=until_ms,
            boolean_predicate=boolean_predicate,
            tags_relation=self._tags_relation,
            prefix="AND",
        )
        where, filter_params = _with_since_session_filter(
            self._conn,
            where,
            filter_params,
            "s",
            since_session_id=since_session_id,
            prefix="AND",
        )
        if session_id is not None:
            where = f"{where} AND s.session_id = ?"
            filter_params.append(session_id)
        order_by = _search_order_by(sort=sort, reverse=reverse)
        params: list[object] = [match_query, *filter_params]
        params.extend([limit, offset])
        rows = self._conn.execute(
            f"""
            SELECT b.block_id, b.message_id, b.session_id, s.origin, s.native_id, s.title,
                   b.search_text AS fallback_text,
                   snippet(messages_fts, 4, '[', ']', '...', 12) AS snippet,
                   rank
            FROM messages_fts
            JOIN blocks b ON b.rowid = messages_fts.rowid
            JOIN sessions s ON s.session_id = b.session_id
            WHERE messages_fts MATCH ?
            {where}
            {order_by}
            LIMIT ? OFFSET ?
            """,
            params,
        ).fetchall()
        return [
            ArchiveSessionSearchHit(
                rank=index,
                session_id=str(row["session_id"]),
                block_id=str(row["block_id"]),
                message_id=str(row["message_id"]),
                origin=str(row["origin"]),
                provider=_provider_for_origin(str(row["origin"])),
                title=str(row["title"]) if row["title"] is not None else None,
                snippet=_highlight_search_snippet(
                    str(row["snippet"] or ""),
                    fallback=str(row["fallback_text"] or ""),
                    query=match_query,
                ),
            )
            for index, row in enumerate(rows, start=offset + 1)
        ]

    def count_search_sessions(
        self,
        query: str,
        *,
        session_id: str | None = None,
        origin: str | None = None,
        origins: tuple[str, ...] = (),
        excluded_origins: tuple[str, ...] = (),
        tags: tuple[str, ...] = (),
        excluded_tags: tuple[str, ...] = (),
        repo_names: tuple[str, ...] = (),
        has_types: tuple[str, ...] = (),
        has_tool_use: bool = False,
        has_thinking: bool = False,
        has_paste: bool = False,
        tool_terms: tuple[str, ...] = (),
        excluded_tool_terms: tuple[str, ...] = (),
        action_terms: tuple[str, ...] = (),
        excluded_action_terms: tuple[str, ...] = (),
        action_sequence: tuple[str, ...] = (),
        action_text_terms: tuple[str, ...] = (),
        referenced_paths: tuple[str, ...] = (),
        cwd_prefix: str | None = None,
        typed_only: bool = False,
        message_type: str | None = None,
        title: str | None = None,
        min_messages: int | None = None,
        max_messages: int | None = None,
        min_words: int | None = None,
        max_words: int | None = None,
        since_ms: int | None = None,
        until_ms: int | None = None,
        since_session_id: str | None = None,
        boolean_predicate: QueryPredicate | None = None,
    ) -> int:
        """Count distinct sessions matching the archive block FTS search."""
        match_query = normalize_fts5_query(query)
        if match_query is None:
            return 0
        where, filter_params = _session_filter_clause(
            "s",
            origin=origin,
            origins=origins,
            excluded_origins=excluded_origins,
            tags=tags,
            excluded_tags=excluded_tags,
            repo_names=repo_names,
            has_types=has_types,
            has_tool_use=has_tool_use,
            has_thinking=has_thinking,
            has_paste=has_paste,
            tool_terms=tool_terms,
            excluded_tool_terms=excluded_tool_terms,
            action_terms=action_terms,
            excluded_action_terms=excluded_action_terms,
            action_sequence=action_sequence,
            action_text_terms=action_text_terms,
            referenced_paths=referenced_paths,
            cwd_prefix=cwd_prefix,
            typed_only=typed_only,
            message_type=message_type,
            title=title,
            min_messages=min_messages,
            max_messages=max_messages,
            min_words=min_words,
            max_words=max_words,
            since_ms=since_ms,
            until_ms=until_ms,
            boolean_predicate=boolean_predicate,
            tags_relation=self._tags_relation,
            prefix="AND",
        )
        where, filter_params = _with_since_session_filter(
            self._conn,
            where,
            filter_params,
            "s",
            since_session_id=since_session_id,
            prefix="AND",
        )
        if session_id is not None:
            where = f"{where} AND s.session_id = ?"
            filter_params.append(session_id)
        row = self._conn.execute(
            f"""
            SELECT COUNT(DISTINCT b.session_id)
            FROM messages_fts
            JOIN blocks b ON b.rowid = messages_fts.rowid
            JOIN sessions s ON s.session_id = b.session_id
            WHERE messages_fts MATCH ?
            {where}
            """,
            [match_query, *filter_params],
        ).fetchone()
        return int(row[0] if row is not None else 0)

    def semantic_summaries(
        self,
        scored_message_ids: list[tuple[str, float]],
        *,
        limit: int = 20,
        offset: int = 0,
        session_id: str | None = None,
        origin: str | None = None,
        origins: tuple[str, ...] = (),
        excluded_origins: tuple[str, ...] = (),
        tags: tuple[str, ...] = (),
        excluded_tags: tuple[str, ...] = (),
        repo_names: tuple[str, ...] = (),
        has_types: tuple[str, ...] = (),
        has_tool_use: bool = False,
        has_thinking: bool = False,
        has_paste: bool = False,
        tool_terms: tuple[str, ...] = (),
        excluded_tool_terms: tuple[str, ...] = (),
        action_terms: tuple[str, ...] = (),
        excluded_action_terms: tuple[str, ...] = (),
        action_sequence: tuple[str, ...] = (),
        action_text_terms: tuple[str, ...] = (),
        referenced_paths: tuple[str, ...] = (),
        cwd_prefix: str | None = None,
        typed_only: bool = False,
        message_type: str | None = None,
        title: str | None = None,
        min_messages: int | None = None,
        max_messages: int | None = None,
        min_words: int | None = None,
        max_words: int | None = None,
        since_ms: int | None = None,
        until_ms: int | None = None,
        since_session_id: str | None = None,
        boolean_predicate: QueryPredicate | None = None,
    ) -> list[ArchiveSessionSearchHit]:
        """Resolve vector-ranked message ids into filtered session-level hits."""
        if not scored_message_ids:
            return []
        message_ids = tuple(message_id for message_id, _score in scored_message_ids)
        where, params = _session_filter_clause(
            "s",
            origin=origin,
            origins=origins,
            excluded_origins=excluded_origins,
            tags=tags,
            excluded_tags=excluded_tags,
            repo_names=repo_names,
            has_types=has_types,
            has_tool_use=has_tool_use,
            has_thinking=has_thinking,
            has_paste=has_paste,
            tool_terms=tool_terms,
            excluded_tool_terms=excluded_tool_terms,
            action_terms=action_terms,
            excluded_action_terms=excluded_action_terms,
            action_sequence=action_sequence,
            action_text_terms=action_text_terms,
            referenced_paths=referenced_paths,
            cwd_prefix=cwd_prefix,
            typed_only=typed_only,
            message_type=message_type,
            title=title,
            min_messages=min_messages,
            max_messages=max_messages,
            min_words=min_words,
            max_words=max_words,
            since_ms=since_ms,
            until_ms=until_ms,
            boolean_predicate=boolean_predicate,
            tags_relation=self._tags_relation,
        )
        where, params = _with_since_session_filter(self._conn, where, params, "s", since_session_id=since_session_id)
        placeholders = ", ".join("?" for _ in message_ids)
        where = f"{where} AND m.message_id IN ({placeholders})" if where else f"WHERE m.message_id IN ({placeholders})"
        params.extend(message_ids)
        if session_id is not None:
            where = f"{where} AND s.session_id = ?"
            params.append(session_id)
        rows = self._conn.execute(
            f"""
            SELECT m.message_id, m.session_id, s.origin, s.native_id, s.title,
                   b.block_id, b.text
            FROM messages m
            JOIN sessions s ON s.session_id = m.session_id
            LEFT JOIN blocks b
              ON b.message_id = m.message_id
             AND b.position = (
                 SELECT MIN(position)
                 FROM blocks
                 WHERE message_id = m.message_id
                   AND text IS NOT NULL
             )
            {where}
            """,
            params,
        ).fetchall()
        rows_by_message_id = {str(row["message_id"]): row for row in rows}
        deduped: list[ArchiveSessionSearchHit] = []
        seen_sessions: set[str] = set()
        for message_id, _score in scored_message_ids:
            row = rows_by_message_id.get(message_id)
            if row is None:
                continue
            session_id = str(row["session_id"])
            if session_id in seen_sessions:
                continue
            seen_sessions.add(session_id)
            text = str(row["text"] or "")
            deduped.append(
                ArchiveSessionSearchHit(
                    rank=len(deduped) + 1,
                    session_id=session_id,
                    block_id=str(row["block_id"] or message_id),
                    message_id=message_id,
                    origin=str(row["origin"]),
                    provider=_provider_for_origin(str(row["origin"])),
                    title=str(row["title"]) if row["title"] is not None else None,
                    snippet=text[:160],
                )
            )
        page = deduped[offset : offset + limit]
        return [replace(hit, rank=offset + index) for index, hit in enumerate(page, start=1)]

    def query_messages(
        self,
        predicate: QueryPredicate,
        *,
        limit: int = 50,
        offset: int = 0,
    ) -> list[ArchiveMessageQueryRow]:
        """Return message rows matching a unit-scoped query predicate."""

        normalized_limit = max(int(limit), 0)
        normalized_offset = max(int(offset), 0)
        clause, params = _structural_predicate_clause("message", "m", predicate)
        rows = self._conn.execute(
            f"""
            SELECT
                m.message_id,
                m.session_id,
                s.origin,
                s.title,
                m.role,
                m.message_type,
                m.position,
                m.word_count,
                COALESCE((
                    SELECT group_concat(ordered.search_text, char(10))
                    FROM (
                        SELECT b.search_text
                        FROM blocks b
                        WHERE b.message_id = m.message_id
                          AND b.search_text IS NOT NULL
                        ORDER BY b.position, b.block_id
                    ) AS ordered
                ), '') AS text
            FROM messages m
            JOIN sessions s ON s.session_id = m.session_id
            WHERE {clause}
            ORDER BY COALESCE(m.occurred_at_ms, s.sort_key_ms, 0), m.message_id
            LIMIT ? OFFSET ?
            """,
            [*params, normalized_limit, normalized_offset],
        ).fetchall()
        return [
            ArchiveMessageQueryRow(
                message_id=str(row["message_id"]),
                session_id=str(row["session_id"]),
                origin=str(row["origin"]),
                title=str(row["title"]) if row["title"] is not None else None,
                role=str(row["role"]),
                message_type=str(row["message_type"]),
                position=int(row["position"]),
                word_count=int(row["word_count"]),
                text=str(row["text"] or ""),
            )
            for row in rows
        ]

    def query_actions(
        self,
        predicate: QueryPredicate,
        *,
        limit: int = 50,
        offset: int = 0,
    ) -> list[ArchiveActionQueryRow]:
        """Return action rows matching a unit-scoped query predicate."""

        normalized_limit = max(int(limit), 0)
        normalized_offset = max(int(offset), 0)
        clause, params = _structural_predicate_clause("action", "a", predicate)
        rows = self._conn.execute(
            f"""
            SELECT
                a.session_id,
                a.message_id,
                s.origin,
                s.title,
                a.tool_use_block_id,
                a.tool_result_block_id,
                a.tool_name,
                a.semantic_type,
                a.tool_command,
                a.tool_path,
                a.output_text
            FROM actions a
            JOIN sessions s ON s.session_id = a.session_id
            JOIN messages m ON m.message_id = a.message_id
            WHERE {clause}
            ORDER BY COALESCE(m.occurred_at_ms, s.sort_key_ms, 0), a.tool_use_block_id
            LIMIT ? OFFSET ?
            """,
            [*params, normalized_limit, normalized_offset],
        ).fetchall()
        return [
            ArchiveActionQueryRow(
                session_id=str(row["session_id"]),
                message_id=str(row["message_id"]),
                origin=str(row["origin"]),
                title=str(row["title"]) if row["title"] is not None else None,
                tool_use_block_id=str(row["tool_use_block_id"]),
                tool_result_block_id=str(row["tool_result_block_id"])
                if row["tool_result_block_id"] is not None
                else None,
                tool_name=str(row["tool_name"]) if row["tool_name"] is not None else None,
                semantic_type=str(row["semantic_type"]) if row["semantic_type"] is not None else None,
                tool_command=str(row["tool_command"]) if row["tool_command"] is not None else None,
                tool_path=str(row["tool_path"]) if row["tool_path"] is not None else None,
                output_text=str(row["output_text"]) if row["output_text"] is not None else None,
            )
            for row in rows
        ]

    def query_blocks(
        self,
        predicate: QueryPredicate,
        *,
        limit: int = 50,
        offset: int = 0,
    ) -> list[ArchiveBlockQueryRow]:
        """Return content-block rows matching a unit-scoped query predicate."""

        normalized_limit = max(int(limit), 0)
        normalized_offset = max(int(offset), 0)
        clause, params = _structural_predicate_clause("block", "b", predicate)
        rows = self._conn.execute(
            f"""
            SELECT
                b.block_id,
                b.message_id,
                b.session_id,
                s.origin,
                s.title,
                b.block_type,
                b.position,
                b.text,
                b.tool_name,
                b.semantic_type,
                b.tool_command,
                b.tool_path
            FROM blocks b
            JOIN sessions s ON s.session_id = b.session_id
            JOIN messages m ON m.message_id = b.message_id
            WHERE {clause}
            ORDER BY COALESCE(m.occurred_at_ms, s.sort_key_ms, 0), b.block_id
            LIMIT ? OFFSET ?
            """,
            [*params, normalized_limit, normalized_offset],
        ).fetchall()
        return [
            ArchiveBlockQueryRow(
                block_id=str(row["block_id"]),
                message_id=str(row["message_id"]),
                session_id=str(row["session_id"]),
                origin=str(row["origin"]),
                title=str(row["title"]) if row["title"] is not None else None,
                block_type=str(row["block_type"]),
                position=int(row["position"]),
                text=str(row["text"]) if row["text"] is not None else None,
                tool_name=str(row["tool_name"]) if row["tool_name"] is not None else None,
                semantic_type=str(row["semantic_type"]) if row["semantic_type"] is not None else None,
                tool_command=str(row["tool_command"]) if row["tool_command"] is not None else None,
                tool_path=str(row["tool_path"]) if row["tool_path"] is not None else None,
            )
            for row in rows
        ]

    def stats(
        self,
        *,
        origin: str | None = None,
        origins: tuple[str, ...] = (),
        excluded_origins: tuple[str, ...] = (),
        tags: tuple[str, ...] = (),
        excluded_tags: tuple[str, ...] = (),
        repo_names: tuple[str, ...] = (),
        has_types: tuple[str, ...] = (),
        has_tool_use: bool = False,
        has_thinking: bool = False,
        has_paste: bool = False,
        tool_terms: tuple[str, ...] = (),
        excluded_tool_terms: tuple[str, ...] = (),
        action_terms: tuple[str, ...] = (),
        excluded_action_terms: tuple[str, ...] = (),
        action_sequence: tuple[str, ...] = (),
        action_text_terms: tuple[str, ...] = (),
        referenced_paths: tuple[str, ...] = (),
        cwd_prefix: str | None = None,
        typed_only: bool = False,
        message_type: str | None = None,
        title: str | None = None,
        min_messages: int | None = None,
        max_messages: int | None = None,
        min_words: int | None = None,
        max_words: int | None = None,
        since_ms: int | None = None,
        until_ms: int | None = None,
        since_session_id: str | None = None,
        session_ids: tuple[str, ...] = (),
    ) -> ArchiveStats:
        """Return archive-level stats from filtered archive index sessions."""
        where, params = _session_filter_clause(
            "s",
            origin=origin,
            origins=origins,
            excluded_origins=excluded_origins,
            tags=tags,
            excluded_tags=excluded_tags,
            repo_names=repo_names,
            has_types=has_types,
            has_tool_use=has_tool_use,
            has_thinking=has_thinking,
            has_paste=has_paste,
            tool_terms=tool_terms,
            excluded_tool_terms=excluded_tool_terms,
            action_terms=action_terms,
            excluded_action_terms=excluded_action_terms,
            action_sequence=action_sequence,
            action_text_terms=action_text_terms,
            referenced_paths=referenced_paths,
            cwd_prefix=cwd_prefix,
            typed_only=typed_only,
            message_type=message_type,
            title=title,
            min_messages=min_messages,
            max_messages=max_messages,
            min_words=min_words,
            max_words=max_words,
            since_ms=since_ms,
            until_ms=until_ms,
            tags_relation=self._tags_relation,
        )
        where, params = _with_since_session_filter(self._conn, where, params, "s", since_session_id=since_session_id)
        where, params = _with_session_id_filter(where, params, "s", session_ids=session_ids)
        row = self._conn.execute(
            f"""
            SELECT COUNT(*) AS total_sessions,
                   COALESCE(SUM(s.message_count), 0) AS total_messages
            FROM sessions s
            {where}
            """,
            params,
        ).fetchone()
        provider_rows = self._conn.execute(
            f"""
            SELECT s.origin, COUNT(*) AS count
            FROM sessions s
            {where}
            GROUP BY s.origin
            ORDER BY count DESC, s.origin
            """,
            params,
        ).fetchall()
        attachment_row = self._conn.execute(
            f"""
            SELECT COUNT(DISTINCT ar.attachment_id) AS total_attachments
            FROM sessions s
            JOIN attachment_refs ar ON ar.session_id = s.session_id
            {where}
            """,
            params,
        ).fetchone()
        return ArchiveStats(
            total_sessions=int(row["total_sessions"] or 0) if row is not None else 0,
            total_messages=int(row["total_messages"] or 0) if row is not None else 0,
            total_attachments=int(attachment_row["total_attachments"] or 0) if attachment_row is not None else 0,
            origins={str(provider_row["origin"]): int(provider_row["count"] or 0) for provider_row in provider_rows},
            db_size_bytes=self.index_db_path.stat().st_size if self.index_db_path.exists() else 0,
        )

    def stats_by(
        self,
        group_by: str,
        *,
        origin: str | None = None,
        origins: tuple[str, ...] = (),
        excluded_origins: tuple[str, ...] = (),
        tags: tuple[str, ...] = (),
        excluded_tags: tuple[str, ...] = (),
        repo_names: tuple[str, ...] = (),
        has_types: tuple[str, ...] = (),
        has_tool_use: bool = False,
        has_thinking: bool = False,
        has_paste: bool = False,
        tool_terms: tuple[str, ...] = (),
        excluded_tool_terms: tuple[str, ...] = (),
        action_terms: tuple[str, ...] = (),
        excluded_action_terms: tuple[str, ...] = (),
        action_sequence: tuple[str, ...] = (),
        action_text_terms: tuple[str, ...] = (),
        referenced_paths: tuple[str, ...] = (),
        cwd_prefix: str | None = None,
        typed_only: bool = False,
        message_type: str | None = None,
        title: str | None = None,
        min_messages: int | None = None,
        max_messages: int | None = None,
        min_words: int | None = None,
        max_words: int | None = None,
        since_ms: int | None = None,
        until_ms: int | None = None,
        since_session_id: str | None = None,
        session_ids: tuple[str, ...] = (),
    ) -> dict[str, int]:
        """Return filtered session counts grouped by a archive dimension."""
        where, params = _session_filter_clause(
            "s",
            origin=origin,
            origins=origins,
            excluded_origins=excluded_origins,
            tags=tags,
            excluded_tags=excluded_tags,
            repo_names=repo_names,
            has_types=has_types,
            has_tool_use=has_tool_use,
            has_thinking=has_thinking,
            has_paste=has_paste,
            tool_terms=tool_terms,
            excluded_tool_terms=excluded_tool_terms,
            action_terms=action_terms,
            excluded_action_terms=excluded_action_terms,
            action_sequence=action_sequence,
            action_text_terms=action_text_terms,
            referenced_paths=referenced_paths,
            cwd_prefix=cwd_prefix,
            typed_only=typed_only,
            message_type=message_type,
            title=title,
            min_messages=min_messages,
            max_messages=max_messages,
            min_words=min_words,
            max_words=max_words,
            since_ms=since_ms,
            until_ms=until_ms,
            tags_relation=self._tags_relation,
        )
        where, params = _with_since_session_filter(self._conn, where, params, "s", since_session_id=since_session_id)
        where, params = _with_session_id_filter(where, params, "s", session_ids=session_ids)
        rows = self._conn.execute(_stats_by_sql(group_by, where, tags_relation=self._tags_relation), params).fetchall()
        results = {str(row["group_key"]): int(row["count"] or 0) for row in rows if row["group_key"] is not None}
        return results

    def __enter__(self) -> ArchiveStore:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()


def _summary_from_row(row: sqlite3.Row) -> ArchiveSessionSummary:
    import json

    raw_tags = json.loads(str(row["tags_json"] or "[]"))
    tags = tuple(str(tag) for tag in raw_tags if tag is not None)
    raw_working_dirs = json.loads(str(row["working_directories_json"] or "[]"))
    working_directories = tuple(str(path) for path in raw_working_dirs if path)
    origin = str(row["origin"])
    return ArchiveSessionSummary(
        session_id=str(row["session_id"]),
        native_id=str(row["native_id"]),
        origin=origin,
        provider=_provider_for_origin(origin),
        title=str(row["title"]) if row["title"] is not None else None,
        created_at=_iso_from_ms(row["created_at_ms"]),
        updated_at=_iso_from_ms(row["updated_at_ms"]),
        message_count=int(row["message_count"] or 0),
        word_count=int(row["word_count"] or 0),
        tags=tags,
        working_directories=working_directories,
        git_branch=str(row["git_branch"]) if row["git_branch"] is not None else None,
        git_repository_url=str(row["git_repository_url"]) if row["git_repository_url"] is not None else None,
    )


def _highlight_search_snippet(snippet: str, *, fallback: str, query: str) -> str:
    """Return bracket-highlighted text when contentless FTS omits markers."""
    import re

    text = snippet or fallback
    if "[" in text and "]" in text:
        return text
    terms = [term.strip('"') for term in re.findall(r'"[^"]+"|[\w.-]+', query) if term.strip('"')]
    for term in sorted(terms, key=len, reverse=True):
        pattern = re.compile(re.escape(term), re.IGNORECASE)
        if pattern.search(text):
            return str(pattern.sub(lambda match: f"[{match.group(0)}]", text, count=1))
    return text


def _summary_order_by(*, sample: bool, sort: str | None, reverse: bool) -> str:
    if sample or sort == "random":
        return "ORDER BY RANDOM()"
    direction = "ASC" if reverse else "DESC"
    if sort in {None, "date"}:
        return f"ORDER BY s.sort_key_ms IS NULL, s.sort_key_ms {direction}, s.session_id {direction}"
    if sort == "messages":
        return f"ORDER BY s.message_count {direction}, s.sort_key_ms {direction}, s.session_id {direction}"
    if sort == "words":
        return f"ORDER BY s.word_count {direction}, s.sort_key_ms {direction}, s.session_id {direction}"
    if sort == "longest":
        return f"""
            ORDER BY (
                SELECT COALESCE(MAX(m.word_count), 0)
                FROM messages m
                WHERE m.session_id = s.session_id
            ) {direction}, s.sort_key_ms {direction}, s.session_id {direction}
        """
    if sort == "tokens":
        return f"""
            ORDER BY (
                SELECT COALESCE(SUM(m.input_tokens + m.output_tokens + m.cache_read_tokens + m.cache_write_tokens), 0)
                FROM messages m
                WHERE m.session_id = s.session_id
            ) {direction}, s.sort_key_ms {direction}, s.session_id {direction}
        """
    raise ValueError("archive root query sort must be one of date, messages, words, longest, tokens, random.")


def _search_order_by(*, sort: str | None, reverse: bool) -> str:
    if sort is None:
        return "ORDER BY rank DESC" if reverse else "ORDER BY rank"
    return _summary_order_by(sample=False, sort=sort, reverse=reverse)


def _with_session_id_filter(
    where: str,
    params: list[object],
    table_alias: str,
    *,
    session_ids: tuple[str, ...],
) -> tuple[str, list[object]]:
    if not session_ids:
        return where, params
    placeholders = ", ".join("?" for _ in session_ids)
    clause = f"{table_alias}.session_id IN ({placeholders})"
    merged_params = [*params, *session_ids]
    if where:
        return f"{where} AND {clause}", merged_params
    return f"WHERE {clause}", merged_params


def _with_since_session_filter(
    conn: sqlite3.Connection,
    where: str,
    params: list[object],
    table_alias: str,
    *,
    since_session_id: str | None,
    prefix: str = "WHERE",
) -> tuple[str, list[object]]:
    if since_session_id is None:
        return where, params
    reference = _since_session_reference(conn, since_session_id)
    if reference is None:
        clause = "0 = 1"
        if where:
            return f"{where} AND {clause}", params
        return f"{prefix} {clause}", params
    ref_session_id, ref_sort_key_ms, ref_paths = reference
    clauses = [f"{table_alias}.session_id != ?"]
    merged_params: list[object] = [*params, ref_session_id]
    if ref_sort_key_ms is not None:
        clauses.append(f"{table_alias}.sort_key_ms > ?")
        merged_params.append(ref_sort_key_ms)
    if ref_paths:
        path_clauses: list[str] = []
        for ref_path in ref_paths:
            exact_prefix, child_prefix = escaped_sql_path_prefix_patterns(ref_path)
            path_clauses.append(
                f"""
                EXISTS (
                    SELECT 1
                    FROM session_working_dirs since_cwd
                    WHERE since_cwd.session_id = {table_alias}.session_id
                      AND (
                        REPLACE(since_cwd.path, char(92), '/') = ?
                        OR REPLACE(since_cwd.path, char(92), '/') LIKE ? ESCAPE '\\'
                      )
                )
                """.strip()
            )
            merged_params.extend([exact_prefix, child_prefix])
        clauses.append("(" + " OR ".join(path_clauses) + ")")
    clause = " AND ".join(clauses)
    if where:
        return f"{where} AND {clause}", merged_params
    return f"{prefix} {clause}", merged_params


def _since_session_reference(
    conn: sqlite3.Connection,
    token: str,
) -> tuple[str, int | None, tuple[str, ...]] | None:
    rows = conn.execute(
        """
        SELECT s.session_id,
               COALESCE(
                   (SELECT MAX(m.occurred_at_ms) FROM messages m WHERE m.session_id = s.session_id),
                   s.sort_key_ms
               ) AS anchor_ms
        FROM sessions s
        WHERE s.session_id = ? OR s.session_id LIKE ?
        ORDER BY CASE WHEN s.session_id = ? THEN 0 ELSE 1 END, s.session_id
        LIMIT 2
        """,
        (token, f"{token}%", token),
    ).fetchall()
    if not rows:
        return None
    row = rows[0]
    session_id = str(row["session_id"])
    path_rows = conn.execute(
        """
        SELECT path
        FROM session_working_dirs
        WHERE session_id = ?
        ORDER BY position, path
        """,
        (session_id,),
    ).fetchall()
    paths = tuple(str(path_row["path"]) for path_row in path_rows if path_row["path"])
    anchor_value = row["anchor_ms"]
    return session_id, int(anchor_value) if anchor_value is not None else None, paths


def _all_session_tags_sql() -> str:
    return """
        (
            SELECT session_id, tag, tag_source, method, confidence, evidence_json
            FROM session_tags
            WHERE tag_source = 'auto'
            UNION ALL
            SELECT session_id, tag, tag_source, method, confidence, evidence_json
            FROM user_tier.session_tags
        )
    """


def _user_target_type_to_storage(target_type: str) -> str:
    if target_type == "session":
        return "session"
    if target_type == "content_block":
        return "block"
    return target_type


def _user_target_type_from_storage(target_type: str) -> str:
    if target_type == "session":
        return "session"
    if target_type == "block":
        return "content_block"
    return target_type


def _user_mark_session_id(target_type: str, target_id: str) -> str:
    if target_type == "session":
        return target_id
    if target_type == "message":
        session_id, _sep, _message_native_id = target_id.rpartition(":")
        return session_id
    return ""


def _learning_correction_from_archive_row(row: sqlite3.Row | tuple[object, ...]) -> LearningCorrection:
    session_id = str(row[0])
    kind = parse_correction_kind(str(row[1]))
    try:
        stored = json.loads(str(row[2]))
    except json.JSONDecodeError:
        stored = {}
    if isinstance(stored, dict) and isinstance(stored.get("payload"), dict):
        payload = {str(key): str(value) for key, value in dict(stored["payload"]).items()}
        note_raw = stored.get("note")
        note = str(note_raw) if note_raw is not None else None
    elif isinstance(stored, dict):
        payload = {str(key): str(value) for key, value in stored.items()}
        note = None
    else:
        payload = {}
        note = None
    raw_updated_at_ms = row[3]
    updated_at_ms = int(str(raw_updated_at_ms or 0))
    return LearningCorrection(
        session_id=session_id,
        kind=kind,
        payload=payload,
        note=note,
        created_at=datetime.fromtimestamp(updated_at_ms / 1000.0, tz=UTC),
    )


def _origin_for_provider_value(provider: str | None) -> str | None:
    if provider is None:
        return None
    return origin_from_provider(Provider.from_string(provider)).value


def _session_origin(conn: sqlite3.Connection, session_id: str) -> str:
    row = conn.execute("SELECT origin FROM sessions WHERE session_id = ?", (session_id,)).fetchone()
    return str(row["origin"]) if row is not None else "unknown-export"


def _read_archive_materialization(
    conn: sqlite3.Connection,
    insight_type: str,
    session_id: str,
) -> ArchiveInsightMaterialization:
    try:
        return read_insight_materialization(conn, insight_type, session_id)
    except KeyError:
        return ArchiveInsightMaterialization(
            insight_type=insight_type,
            session_id=session_id,
            materializer_version=1,
            materialized_at_ms=0,
            source_updated_at_ms=None,
            source_sort_key_ms=None,
            input_high_water_mark_ms=None,
            input_row_count=0,
        )


def _archive_provenance(materialization: ArchiveInsightMaterialization) -> ArchiveInsightProvenance:
    return ArchiveInsightProvenance(
        materializer_version=materialization.materializer_version,
        materialized_at=_iso_from_ms(materialization.materialized_at_ms) or "1970-01-01T00:00:00Z",
        source_updated_at=_iso_from_ms(materialization.source_updated_at_ms),
        source_sort_key=(
            materialization.source_sort_key_ms / 1000.0 if materialization.source_sort_key_ms is not None else None
        ),
    )


def _archive_inference_provenance(materialization: ArchiveInsightMaterialization) -> ArchiveInferenceProvenance:
    base = _archive_provenance(materialization)
    return ArchiveInferenceProvenance(
        materializer_version=base.materializer_version,
        materialized_at=base.materialized_at,
        source_updated_at=base.source_updated_at,
        source_sort_key=base.source_sort_key,
        inference_version=materialization.materializer_version,
        inference_family="archive",
    )


def _archive_enrichment_provenance(materialization: ArchiveInsightMaterialization) -> ArchiveEnrichmentProvenance:
    base = _archive_provenance(materialization)
    return ArchiveEnrichmentProvenance(
        materializer_version=base.materializer_version,
        materialized_at=base.materialized_at,
        source_updated_at=base.source_updated_at,
        source_sort_key=base.source_sort_key,
        enrichment_version=materialization.materializer_version,
        enrichment_family="archive",
    )


def _work_event_insight_from_archive_row(
    event: ArchiveSessionWorkEvent,
    *,
    origin: str,
    materialization: ArchiveInsightMaterialization,
) -> SessionWorkEventInsight:
    evidence_payload = {
        **event.evidence,
        "start_index": event.start_index,
        "end_index": event.end_index,
        "start_time": _iso_from_ms(event.started_at_ms),
        "end_time": _iso_from_ms(event.ended_at_ms),
        "duration_ms": event.duration_ms,
        "file_paths": event.file_paths,
        "tools_used": event.tools_used,
    }
    inference_payload = {
        **event.inference,
        "heuristic_label": event.work_event_type,
        "summary": event.summary,
        "confidence": event.confidence,
        "support_level": confidence_from_score(event.confidence),
    }
    return SessionWorkEventInsight(
        event_id=event.event_id,
        session_id=event.session_id,
        source_name=_provider_for_origin(origin).value,
        event_index=event.position,
        provenance=_archive_provenance(materialization),
        inference_provenance=_archive_inference_provenance(materialization),
        evidence=WorkEventEvidencePayload.model_validate(evidence_payload),
        inference=WorkEventInferencePayload.model_validate(inference_payload),
    )


def _phase_insight_from_archive_row(
    phase: ArchiveSessionPhase,
    *,
    origin: str,
    materialization: ArchiveInsightMaterialization,
) -> SessionPhaseInsight:
    evidence_payload = {
        **phase.evidence,
        "start_time": _iso_from_ms(phase.started_at_ms),
        "end_time": _iso_from_ms(phase.ended_at_ms),
        "message_range": (phase.start_index, phase.end_index),
        "duration_ms": phase.duration_ms,
        "tool_counts": phase.tool_counts,
        "word_count": phase.word_count,
    }
    inference_payload = {
        **phase.inference,
        "confidence": phase.confidence,
        "support_level": confidence_from_score(phase.confidence),
    }
    return SessionPhaseInsight(
        phase_id=phase.phase_id,
        session_id=phase.session_id,
        source_name=_provider_for_origin(origin).value,
        phase_index=phase.position,
        provenance=_archive_provenance(materialization),
        inference_provenance=_archive_inference_provenance(materialization),
        evidence=SessionPhaseEvidencePayload.model_validate(evidence_payload),
        inference=SessionPhaseInferencePayload.model_validate(inference_payload),
    )


@dataclass(frozen=True)
class _SessionProfileComponents:
    """Extracted session-profile payloads shared by the insight and record builders."""

    materialization: ArchiveInsightMaterialization
    evidence: SessionEvidencePayload
    inference: SessionInferencePayload
    enrichment: SessionEnrichmentPayload | None


def _session_profile_components_from_archive_row(
    conn: sqlite3.Connection,
    row: sqlite3.Row,
) -> _SessionProfileComponents:
    """Build the evidence/inference/enrichment payloads from a session profile row.

    This is the shared extraction used by both
    :func:`_session_profile_insight_from_archive_row` (tier-gated insight projection)
    and :meth:`ArchiveStore.get_session_profile_record` (full domain-record
    hydration). All three payloads are always materialized here; the insight
    builder applies tier gating on top.

    Reads the typed *_payload_json columns written by the canonical
    session-profile writer (replace_session_profiles_bulk_sync).  The legacy
    provenance_json column has been dropped from the DDL.
    """
    from polylogue.storage.sqlite.queries.mappers_insight_fallback import parse_payload_model

    session_id = str(row["session_id"])
    materialization = _read_archive_materialization(conn, "session_profile", session_id)
    workflow_shape = str(row["workflow_shape"] or "unknown")
    workflow_confidence = float(row["workflow_shape_confidence"] or 0.0)
    terminal_state = str(row["terminal_state"] or "unknown")
    terminal_confidence = float(row["terminal_state_confidence"] or 0.0)

    evidence = parse_payload_model(row, "evidence_payload_json", record_id=session_id, model=SessionEvidencePayload)
    if evidence is None:
        # Fallback for rows written before the typed-column migration: build
        # a minimal payload from the direct session/profile row columns.
        evidence = SessionEvidencePayload.model_validate(
            {
                "created_at": _iso_from_ms(row["created_at_ms"]),
                "updated_at": _iso_from_ms(row["updated_at_ms"]),
                "message_count": int(row["message_count"] or 0),
                "substantive_count": int(row["substantive_count"] or 0),
                "attachment_count": int(row["attachment_count"] or 0),
                "tool_use_count": int(row["tool_use_count"] or 0),
                "thinking_count": int(row["thinking_count"] or 0),
                "word_count": int(row["word_count"] or 0),
                "total_cost_usd": float(row["total_cost_usd"] or row["cost_usd"] or 0.0),
                "total_duration_ms": int(row["total_duration_ms"] or row["duration_ms"] or 0),
                "workflow_shape": workflow_shape,
                "workflow_shape_confidence": workflow_confidence,
                "terminal_state": terminal_state,
                "terminal_state_confidence": terminal_confidence,
                "cost_is_estimated": bool(row["cost_is_estimated"]),
                "cost_provenance": str(row["cost_provenance"] or "unknown"),
                "logical_session_id": str(row["root_session_id"] or session_id),
                "tool_calls_per_minute": float(row["tool_calls_per_minute"] or 0.0),
            }
        )

    inference = parse_payload_model(row, "inference_payload_json", record_id=session_id, model=SessionInferencePayload)
    if inference is None:
        inference = SessionInferencePayload.model_validate(
            {
                "work_event_count": int(row["work_event_count"] or 0),
                "phase_count": int(row["phase_count"] or 0),
                "engaged_duration_ms": int(row["total_duration_ms"] or row["duration_ms"] or 0),
                "engaged_minutes": float(row["total_duration_ms"] or row["duration_ms"] or 0) / 60000.0,
                "workflow_shape": workflow_shape,
                "workflow_shape_confidence": workflow_confidence,
                "terminal_state": terminal_state,
                "terminal_state_confidence": terminal_confidence,
                "support_level": confidence_from_score(max(workflow_confidence, terminal_confidence)),
            }
        )
    else:
        # The denormalized native session_profiles columns are the authoritative
        # ranking signals; reconcile the JSON-derived payload onto them so resume
        # ranking and aggregation read the queryable native columns rather than a
        # divergent payload copy.
        inference = inference.model_copy(
            update={
                "workflow_shape": workflow_shape,
                "workflow_shape_confidence": workflow_confidence,
                "terminal_state": terminal_state,
                "terminal_state_confidence": terminal_confidence,
            }
        )

    enrichment = parse_payload_model(
        row, "enrichment_payload_json", record_id=session_id, model=SessionEnrichmentPayload
    )
    return _SessionProfileComponents(
        materialization=materialization,
        evidence=evidence,
        inference=inference,
        enrichment=enrichment,
    )


def _session_profile_insight_from_archive_row(
    conn: sqlite3.Connection,
    row: sqlite3.Row,
    *,
    tier: str,
) -> SessionProfileInsight:
    session_id = str(row["session_id"])
    components = _session_profile_components_from_archive_row(conn, row)
    materialization = components.materialization
    include_evidence = tier in {"merged", "evidence"}
    include_inference = tier in {"merged", "inference"}
    include_enrichment = tier == "merged"
    evidence = components.evidence if include_evidence else None
    inference = components.inference if include_inference else None
    enrichment = None
    enrichment_provenance = None
    if include_enrichment and components.enrichment is not None:
        enrichment = components.enrichment
        enrichment_provenance = _archive_enrichment_provenance(materialization)
    return SessionProfileInsight(
        semantic_tier=tier,
        session_id=session_id,
        logical_session_id=str(row["root_session_id"] or session_id),
        source_name=_provider_for_origin(str(row["origin"])).value,
        title=str(row["title"]) if row["title"] is not None else None,
        provenance=_archive_provenance(materialization),
        evidence=evidence,
        inference_provenance=_archive_inference_provenance(materialization) if include_inference else None,
        inference=inference,
        enrichment_provenance=enrichment_provenance,
        enrichment=enrichment,
    )


def _session_profile_record_from_archive_row(
    conn: sqlite3.Connection,
    row: sqlite3.Row,
) -> SessionProfileRecord:
    """Build the full domain :class:`SessionProfileRecord` from a session profile row.

    Reuses the same payload extraction as the insight projection and pulls the
    materialization HWM provenance from ``read_insight_materialization`` so the
    record carries the fields ``hydrate_session_profile`` and the
    ``is_stale`` staleness check expect. The FTS-only ``*_search_text`` fields
    are not stored in the archive ``session_profiles`` row and are not read by
    ``hydrate_session_profile``; they are synthesized as non-empty strings here
    purely to satisfy the model's required-non-empty validators.
    """
    session_id = str(row["session_id"])
    components = _session_profile_components_from_archive_row(conn, row)
    materialization = components.materialization
    evidence = components.evidence
    inference = components.inference
    enrichment = components.enrichment if components.enrichment is not None else SessionEnrichmentPayload()
    logical_session_id = str(row["root_session_id"] or session_id)
    source_name = _provider_for_origin(str(row["origin"])).value
    title = str(row["title"]) if row["title"] is not None else None
    workflow_shape = str(row["workflow_shape"] or "unknown")
    materialized_at = _iso_from_ms(materialization.materialized_at_ms) or "1970-01-01T00:00:00Z"
    # search_text* are FTS-only and not consumed by hydrate_session_profile;
    # synthesize a stable non-empty string so the record validates.
    search_text = title or workflow_shape or session_id
    return SessionProfileRecord(
        session_id=SessionId(session_id),
        logical_session_id=SessionId(logical_session_id),
        materializer_version=materialization.materializer_version,
        materialized_at=materialized_at,
        source_updated_at=_iso_from_ms(materialization.source_updated_at_ms),
        source_sort_key=(
            materialization.source_sort_key_ms / 1000.0 if materialization.source_sort_key_ms is not None else None
        ),
        input_high_water_mark=_iso_from_ms(materialization.input_high_water_mark_ms),
        input_high_water_mark_source=None,
        input_row_count=materialization.input_row_count,
        source_name=source_name,
        title=title,
        first_message_at=evidence.first_message_at,
        last_message_at=evidence.last_message_at,
        canonical_session_date=evidence.canonical_session_date,
        repo_paths=evidence.repo_paths,
        repo_names=inference.repo_names,
        tags=evidence.tags,
        auto_tags=inference.auto_tags,
        message_count=int(row["message_count"] or 0),
        substantive_count=int(row["substantive_count"] or 0),
        attachment_count=int(row["attachment_count"] or 0),
        work_event_count=int(row["work_event_count"] or 0),
        phase_count=int(row["phase_count"] or 0),
        word_count=int(row["word_count"] or 0),
        tool_use_count=int(row["tool_use_count"] or 0),
        thinking_count=int(row["thinking_count"] or 0),
        total_cost_usd=evidence.total_cost_usd,
        total_duration_ms=evidence.total_duration_ms,
        engaged_duration_ms=inference.engaged_duration_ms,
        tool_active_duration_ms=evidence.tool_active_duration_ms,
        wall_duration_ms=evidence.wall_duration_ms,
        workflow_shape=workflow_shape,
        workflow_shape_confidence=float(row["workflow_shape_confidence"] or 0.0),
        terminal_state=str(row["terminal_state"] or "unknown"),
        terminal_state_confidence=float(row["terminal_state_confidence"] or 0.0),
        cost_is_estimated=bool(row["cost_is_estimated"]),
        thinking_duration_ms=evidence.thinking_duration_ms,
        output_duration_ms=evidence.output_duration_ms,
        tool_duration_ms=evidence.tool_duration_ms,
        tool_calls_per_minute=float(row["tool_calls_per_minute"] or 0.0),
        timing_provenance=evidence.timing_provenance,
        total_input_tokens=evidence.total_input_tokens,
        total_output_tokens=evidence.total_output_tokens,
        total_cache_read_tokens=evidence.total_cache_read_tokens,
        total_cache_write_tokens=evidence.total_cache_write_tokens,
        total_credit_cost=evidence.total_credit_cost,
        cost_provenance=str(row["cost_provenance"] or "unknown"),
        evidence_payload=evidence,
        inference_payload=inference,
        search_text=search_text,
        evidence_search_text=search_text,
        inference_search_text=search_text,
        enrichment_payload=enrichment,
        enrichment_search_text=search_text,
    )


def _session_cost_insight_from_archive_row(conn: sqlite3.Connection, row: sqlite3.Row) -> SessionCostInsight:
    session_id = str(row["session_id"])
    source_name = _provider_for_origin(str(row["origin"])).value
    total_usd = float(row["cost_usd"] or 0.0)
    cost_provenance = str(row["cost_provenance"] or "")
    try:
        raw_model_name = row["model_name"]
    except (IndexError, KeyError):
        raw_model_name = None
    model_name = str(raw_model_name) if raw_model_name is not None else None
    normalized_model = _normalize_model(model_name) if model_name else None
    status: CostEstimateStatus
    unavailable_reason: CostUnavailableReason | None
    provenance: tuple[str, ...]
    if total_usd > 0:
        status = "exact" if cost_provenance == "exact" else "priced"
        confidence = 1.0 if status == "exact" else (0.7 if row["cost_is_estimated"] else 0.9)
        basis = (
            CostBasisPayload(provider_reported_usd=total_usd)
            if status == "exact"
            else CostBasisPayload(catalog_priced_usd=total_usd)
        )
        missing_reasons: tuple[str, ...] = ()
        unavailable_reason = None
        provenance = ("archive_session_profiles", cost_provenance or status)
    else:
        status = "unavailable"
        confidence = 0.0
        basis = CostBasisPayload()
        missing_reasons = ("archive_profile_no_cost",)
        unavailable_reason = "no_tokens"
        provenance = ("archive_session_profiles",)
    materialization = _read_archive_materialization(conn, "session_profile", session_id)
    return SessionCostInsight(
        session_id=session_id,
        source_name=source_name,
        title=str(row["title"]) if row["title"] is not None else None,
        created_at=_iso_from_ms(row["created_at_ms"]),
        updated_at=_iso_from_ms(row["updated_at_ms"]),
        estimate=CostEstimatePayload(
            source_name=source_name,
            session_id=session_id,
            model_name=model_name,
            normalized_model=normalized_model,
            status=status,
            confidence=confidence,
            total_usd=total_usd,
            basis=basis,
            missing_reasons=missing_reasons,
            unavailable_reason=unavailable_reason,
            provenance=provenance,
        ),
        provenance=_archive_provenance(materialization),
    )


def _json_object_from_text(value: object) -> dict[str, object]:
    try:
        decoded = json.loads(str(value or "{}"))
    except json.JSONDecodeError:
        return {}
    return decoded if isinstance(decoded, dict) else {}


def _stats_by_sql(group_by: str, where: str, *, tags_relation: str = "session_tags") -> str:
    if group_by in {"provider", "origin"}:
        return f"""
            SELECT s.origin AS group_key, COUNT(DISTINCT s.session_id) AS count
            FROM sessions s
            {where}
            GROUP BY s.origin
            ORDER BY count DESC, group_key
        """
    if group_by in {"day", "month", "year"}:
        formats = {"day": "%Y-%m-%d", "month": "%Y-%m", "year": "%Y"}
        return f"""
            SELECT strftime('{formats[group_by]}', s.sort_key_ms / 1000, 'unixepoch') AS group_key,
                   COUNT(DISTINCT s.session_id) AS count
            FROM sessions s
            {where}
            GROUP BY group_key
            HAVING group_key IS NOT NULL
            ORDER BY group_key DESC
        """
    if group_by == "tag":
        return f"""
            SELECT st.tag AS group_key, COUNT(DISTINCT s.session_id) AS count
            FROM sessions s
            JOIN {tags_relation} st ON st.session_id = s.session_id
            {where}
            GROUP BY st.tag
            ORDER BY count DESC, group_key
        """
    if group_by == "repo":
        return f"""
            SELECT COALESCE(NULLIF(r.repo_name, ''), NULLIF(r.root_path, ''), NULLIF(r.origin_url, '')) AS group_key,
                   COUNT(DISTINCT s.session_id) AS count
            FROM sessions s
            JOIN session_repos sr ON sr.session_id = s.session_id
            JOIN repos r ON r.repo_id = sr.repo_id
            {where}
            GROUP BY group_key
            HAVING group_key IS NOT NULL
            ORDER BY count DESC, group_key
        """
    if group_by == "tool":
        return f"""
            SELECT COALESCE(NULLIF(LOWER(a.tool_name), ''), 'unknown') AS group_key,
                   COUNT(DISTINCT s.session_id) AS count
            FROM sessions s
            JOIN actions a ON a.session_id = s.session_id
            {where}
            GROUP BY group_key
            ORDER BY count DESC, group_key
        """
    if group_by == "action":
        return f"""
            SELECT COALESCE(NULLIF(a.semantic_type, ''), 'unknown') AS group_key,
                   COUNT(DISTINCT s.session_id) AS count
            FROM sessions s
            JOIN actions a ON a.session_id = s.session_id
            {where}
            GROUP BY group_key
            ORDER BY count DESC, group_key
        """
    if group_by == "work-kind":
        return f"""
            SELECT COALESCE(NULLIF(sp.workflow_shape, ''), 'unknown') AS group_key,
                   COUNT(DISTINCT s.session_id) AS count
            FROM sessions s
            JOIN session_profiles sp ON sp.session_id = s.session_id
            {where}
            GROUP BY group_key
            ORDER BY count DESC, group_key
        """
    raise ValueError(
        f"Unknown group_by {group_by!r}; expected one of: provider, origin, day, month, year, tag, repo, tool, action, work-kind"
    )


def _clause_without_prefix(where: str, *, prefix: str) -> str:
    stripped = where.strip()
    marker = f"{prefix} "
    if stripped.startswith(marker):
        return stripped[len(marker) :].strip()
    return stripped


def _date_ms(value: str, *, field: str) -> int:
    parsed = parse_date(value)
    if parsed is None:
        raise ValueError(f"invalid {field}: {value}")
    return int(parsed.timestamp() * 1000)


def _field_predicate_clause(
    table_alias: str,
    predicate: QueryFieldPredicate,
    *,
    tags_relation: str,
) -> tuple[str, list[object]]:
    field = predicate.field
    values = predicate.values
    kwargs: dict[str, Any] = {}
    if field == "id":
        if not values:
            return "", []
        return f"{table_alias}.session_id = ?", [values[-1]]
    if field == "repo":
        kwargs["repo_names"] = values
    elif field == "origin":
        kwargs["origins"] = values
    elif field == "tag":
        kwargs["tags"] = values
    elif field == "path":
        kwargs["referenced_paths"] = values
    elif field == "cwd":
        kwargs["cwd_prefix"] = values[-1] if values else None
    elif field == "tool":
        kwargs["tool_terms"] = values
    elif field == "action":
        kwargs["action_terms"] = values
    elif field == "has":
        has_types: list[str] = []
        for value in values:
            if value == "paste":
                kwargs["has_paste"] = True
            elif value == "tools":
                kwargs["has_tool_use"] = True
            elif value == "thinking":
                kwargs["has_thinking"] = True
            else:
                has_types.append(value)
        kwargs["has_types"] = tuple(has_types)
    elif field == "title":
        kwargs["title"] = " ".join(values)
    elif field == "date":
        if values:
            if predicate.op == ">=":
                kwargs["since_ms"] = _date_ms(values[-1], field="date")
            elif predicate.op == "<=":
                kwargs["until_ms"] = _date_ms(values[-1], field="date")
            else:
                raise ValueError("unsupported Boolean query operator for date")
    elif field == "since":
        if values:
            kwargs["since_ms"] = _date_ms(values[-1], field="since")
    elif field == "until":
        if values:
            kwargs["until_ms"] = _date_ms(values[-1], field="until")
    elif field == "messages":
        if not values:
            return "", []
        count_value = int(values[-1])
        if predicate.op == ">=":
            kwargs["min_messages"] = count_value
        elif predicate.op == "<=":
            kwargs["max_messages"] = count_value
        else:
            kwargs["min_messages"] = count_value
            kwargs["max_messages"] = count_value
    elif field == "words":
        if not values:
            return "", []
        count_value = int(values[-1])
        if predicate.op == ">=":
            kwargs["min_words"] = count_value
        elif predicate.op == "<=":
            kwargs["max_words"] = count_value
        else:
            kwargs["min_words"] = count_value
            kwargs["max_words"] = count_value
    else:
        raise ValueError(f"unsupported Boolean query field: {field}")
    where, params = _session_filter_clause(table_alias, tags_relation=tags_relation, prefix="WHERE", **kwargs)
    return _clause_without_prefix(where, prefix="WHERE"), params


def _in_or_equals_clause(column: str, values: tuple[str, ...], *, lower: bool = False) -> tuple[str, list[object]]:
    normalized = tuple(value.strip().lower() if lower else value.strip() for value in values if value.strip())
    if not normalized:
        return "", []
    expression = f"lower({column})" if lower else column
    if len(normalized) == 1:
        return f"{expression} = ?", [normalized[0]]
    placeholders = ", ".join("?" for _ in normalized)
    return f"{expression} IN ({placeholders})", list(normalized)


def _count_predicate_clause(column: str, predicate: QueryFieldPredicate) -> tuple[str, list[object]]:
    if not predicate.values:
        return "", []
    value = int(predicate.values[-1])
    if predicate.op == ">=":
        return f"{column} >= ?", [value]
    if predicate.op == "<=":
        return f"{column} <= ?", [value]
    return f"{column} = ?", [value]


def _like_clause(
    expression: str,
    values: tuple[str, ...],
    *,
    joiner: Literal["AND", "OR"] = "OR",
) -> tuple[str, list[object]]:
    normalized = tuple(value.strip().lower() for value in values if value.strip())
    if not normalized:
        return "", []
    clauses = [f"lower({expression}) LIKE ?" for _ in normalized]
    joined = f" {joiner} ".join(clauses)
    return (f"({joined})" if len(clauses) > 1 else joined), [f"%{value}%" for value in normalized]


def _message_field_predicate_clause(message_alias: str, predicate: QueryFieldPredicate) -> tuple[str, list[object]]:
    field = predicate.field
    if field == "role":
        return _in_or_equals_clause(f"{message_alias}.role", predicate.values, lower=True)
    if field == "type":
        return _in_or_equals_clause(f"{message_alias}.message_type", predicate.values, lower=True)
    if field == "words":
        return _count_predicate_clause(f"{message_alias}.word_count", predicate)
    if field in {"text", "command", "path", "output", "tool", "action"}:
        action_clause = ""
        params: list[object] = []
        if field == "text":
            block_clause, params = _like_clause("COALESCE(filter_blocks.search_text, '')", predicate.values)
            action_clause = f"""
                EXISTS (
                    SELECT 1
                    FROM blocks filter_blocks
                    WHERE filter_blocks.message_id = {message_alias}.message_id
                      AND {block_clause}
                )
            """.strip()
        elif field == "tool":
            inner_clause, params = _in_or_equals_clause("filter_actions.tool_name", predicate.values, lower=True)
            action_clause = f"""
                EXISTS (
                    SELECT 1
                    FROM actions filter_actions
                    WHERE filter_actions.message_id = {message_alias}.message_id
                      AND {inner_clause}
                )
            """.strip()
        elif field == "action":
            inner_clause, params = _in_or_equals_clause("filter_actions.semantic_type", predicate.values, lower=True)
            action_clause = f"""
                EXISTS (
                    SELECT 1
                    FROM actions filter_actions
                    WHERE filter_actions.message_id = {message_alias}.message_id
                      AND {inner_clause}
                )
            """.strip()
        else:
            action_column = {
                "command": "COALESCE(filter_actions.tool_command, '')",
                "path": "REPLACE(COALESCE(filter_actions.tool_path, ''), char(92), '/')",
                "output": "COALESCE(filter_actions.output_text, '')",
            }[field]
            inner_clause, params = _like_clause(action_column, predicate.values)
            action_clause = f"""
                EXISTS (
                    SELECT 1
                    FROM actions filter_actions
                    WHERE filter_actions.message_id = {message_alias}.message_id
                      AND {inner_clause}
                )
            """.strip()
        return action_clause, params
    raise ValueError(f"unsupported message predicate field: {field}")


def _action_field_predicate_clause(action_alias: str, predicate: QueryFieldPredicate) -> tuple[str, list[object]]:
    field = predicate.field
    if field == "tool":
        return _in_or_equals_clause(f"{action_alias}.tool_name", predicate.values, lower=True)
    if field in {"action", "type"}:
        return _in_or_equals_clause(f"{action_alias}.semantic_type", predicate.values, lower=True)
    if field == "command":
        return _like_clause(f"COALESCE({action_alias}.tool_command, '')", predicate.values)
    if field == "path":
        return _like_clause(f"REPLACE(COALESCE({action_alias}.tool_path, ''), char(92), '/')", predicate.values)
    if field == "output":
        return _like_clause(f"COALESCE({action_alias}.output_text, '')", predicate.values)
    if field == "text":
        return _like_clause(
            f"""
            COALESCE({action_alias}.tool_name, '') || ' ' ||
            COALESCE({action_alias}.semantic_type, '') || ' ' ||
            COALESCE({action_alias}.tool_command, '') || ' ' ||
            COALESCE({action_alias}.tool_path, '') || ' ' ||
            COALESCE({action_alias}.tool_input, '') || ' ' ||
            COALESCE({action_alias}.output_text, '')
            """.strip(),
            predicate.values,
        )
    raise ValueError(f"unsupported action predicate field: {field}")


def _block_field_predicate_clause(block_alias: str, predicate: QueryFieldPredicate) -> tuple[str, list[object]]:
    field = predicate.field
    if field == "type":
        return _in_or_equals_clause(f"{block_alias}.block_type", predicate.values, lower=True)
    if field == "text":
        return _like_clause(f"COALESCE({block_alias}.search_text, '')", predicate.values)
    if field == "tool":
        return _in_or_equals_clause(f"{block_alias}.tool_name", predicate.values, lower=True)
    if field in {"action", "command", "path"}:
        column = {
            "action": f"{block_alias}.semantic_type",
            "command": f"COALESCE({block_alias}.tool_command, '')",
            "path": f"REPLACE(COALESCE({block_alias}.tool_path, ''), char(92), '/')",
        }[field]
        if field == "action":
            return _in_or_equals_clause(column, predicate.values, lower=True)
        return _like_clause(column, predicate.values)
    raise ValueError(f"unsupported block predicate field: {field}")


def _structural_predicate_clause(
    unit: str,
    row_alias: str,
    predicate: QueryPredicate,
) -> tuple[str, list[object]]:
    if isinstance(predicate, QueryFieldPredicate):
        if unit == "message":
            return _message_field_predicate_clause(row_alias, predicate)
        if unit == "action":
            return _action_field_predicate_clause(row_alias, predicate)
        if unit == "block":
            return _block_field_predicate_clause(row_alias, predicate)
    if isinstance(predicate, QueryNotPredicate):
        clause, params = _structural_predicate_clause(unit, row_alias, predicate.child)
        return (f"NOT ({clause})" if clause else "", params)
    if isinstance(predicate, QueryBoolPredicate):
        child_clauses: list[str] = []
        merged_params: list[object] = []
        for child in predicate.children:
            clause, child_params = _structural_predicate_clause(unit, row_alias, child)
            if clause:
                child_clauses.append(f"({clause})")
                merged_params.extend(child_params)
        if not child_clauses:
            return "", merged_params
        joiner = " OR " if predicate.op == "or" else " AND "
        return joiner.join(child_clauses), merged_params
    raise ValueError(f"unsupported nested structural predicate for {unit}: {predicate!r}")


def _exists_predicate_clause(table_alias: str, predicate: QueryExistsPredicate) -> tuple[str, list[object]]:
    if predicate.unit == "message":
        row_alias = "exists_messages"
        child_clause, params = _structural_predicate_clause(predicate.unit, row_alias, predicate.child)
        return (
            f"""
            EXISTS (
                SELECT 1
                FROM messages {row_alias}
                WHERE {row_alias}.session_id = {table_alias}.session_id
                  AND {child_clause}
            )
            """.strip(),
            params,
        )
    if predicate.unit == "action":
        row_alias = "exists_actions"
        child_clause, params = _structural_predicate_clause(predicate.unit, row_alias, predicate.child)
        return (
            f"""
            EXISTS (
                SELECT 1
                FROM actions {row_alias}
                WHERE {row_alias}.session_id = {table_alias}.session_id
                  AND {child_clause}
            )
            """.strip(),
            params,
        )
    if predicate.unit == "block":
        row_alias = "exists_blocks"
        child_clause, params = _structural_predicate_clause(predicate.unit, row_alias, predicate.child)
        return (
            f"""
            EXISTS (
                SELECT 1
                FROM blocks {row_alias}
                WHERE {row_alias}.session_id = {table_alias}.session_id
                  AND {child_clause}
            )
            """.strip(),
            params,
        )
    raise ValueError(f"unsupported structural query unit: {predicate.unit}")


def _fts_predicate_clause(table_alias: str, predicate: QueryTextPredicate) -> tuple[str, list[object]]:
    match_query = normalize_fts5_query(predicate.text)
    if match_query is None:
        raise ValueError("FTS predicate requires non-empty text")
    return (
        f"""
        EXISTS (
            SELECT 1
            FROM messages_fts
            JOIN blocks filter_fts_blocks
              ON filter_fts_blocks.rowid = messages_fts.rowid
            WHERE filter_fts_blocks.session_id = {table_alias}.session_id
              AND messages_fts MATCH ?
        )
        """.strip(),
        [match_query],
    )


def _lineage_predicate_clause(table_alias: str, predicate: QueryLineagePredicate) -> tuple[str, list[object]]:
    seed_session_id = predicate.seed_session_id.strip()
    if not seed_session_id:
        raise ValueError("lineage predicate requires a session id")
    return (
        f"""
        COALESCE({table_alias}.root_session_id, {table_alias}.session_id) = (
            SELECT COALESCE(seed.root_session_id, seed.session_id)
            FROM sessions seed
            WHERE seed.session_id = ?
        )
        """.strip(),
        [seed_session_id],
    )


def _boolean_predicate_clause(
    table_alias: str,
    predicate: QueryPredicate,
    *,
    tags_relation: str,
) -> tuple[str, list[object]]:
    if isinstance(predicate, QueryFieldPredicate):
        return _field_predicate_clause(table_alias, predicate, tags_relation=tags_relation)
    if isinstance(predicate, QueryExistsPredicate):
        return _exists_predicate_clause(table_alias, predicate)
    if isinstance(predicate, QuerySequencePredicate):
        return _action_sequence_clause(table_alias, predicate.action_terms)
    if isinstance(predicate, QueryTextPredicate):
        return _fts_predicate_clause(table_alias, predicate)
    if isinstance(predicate, QueryLineagePredicate):
        return _lineage_predicate_clause(table_alias, predicate)
    if isinstance(predicate, QueryNotPredicate):
        clause, params = _boolean_predicate_clause(table_alias, predicate.child, tags_relation=tags_relation)
        return (f"NOT ({clause})" if clause else "", params)
    if isinstance(predicate, QueryBoolPredicate):
        child_clauses: list[str] = []
        merged_params: list[object] = []
        for child in predicate.children:
            clause, child_params = _boolean_predicate_clause(table_alias, child, tags_relation=tags_relation)
            if clause:
                child_clauses.append(f"({clause})")
                merged_params.extend(child_params)
        if not child_clauses:
            return "", merged_params
        joiner = " OR " if predicate.op == "or" else " AND "
        return joiner.join(child_clauses), merged_params
    raise TypeError(f"unsupported Boolean query predicate: {predicate!r}")


def _session_filter_clause(
    table_alias: str,
    *,
    origin: str | None = None,
    origins: tuple[str, ...] = (),
    excluded_origins: tuple[str, ...] = (),
    tags: tuple[str, ...] = (),
    excluded_tags: tuple[str, ...] = (),
    repo_names: tuple[str, ...] = (),
    has_types: tuple[str, ...] = (),
    has_tool_use: bool = False,
    has_thinking: bool = False,
    has_paste: bool = False,
    tool_terms: tuple[str, ...] = (),
    excluded_tool_terms: tuple[str, ...] = (),
    action_terms: tuple[str, ...] = (),
    excluded_action_terms: tuple[str, ...] = (),
    action_sequence: tuple[str, ...] = (),
    action_text_terms: tuple[str, ...] = (),
    referenced_paths: tuple[str, ...] = (),
    cwd_prefix: str | None = None,
    typed_only: bool = False,
    message_type: str | None = None,
    title: str | None = None,
    min_messages: int | None = None,
    max_messages: int | None = None,
    min_words: int | None = None,
    max_words: int | None = None,
    since_ms: int | None = None,
    until_ms: int | None = None,
    boolean_predicate: QueryPredicate | None = None,
    tags_relation: str = "session_tags",
    prefix: str = "WHERE",
) -> tuple[str, list[object]]:
    clauses: list[str] = []
    params: list[object] = []
    if origin is not None:
        clauses.append(f"{table_alias}.origin = ?")
        params.append(origin)
    if origins:
        placeholders = ", ".join("?" for _ in origins)
        clauses.append(f"{table_alias}.origin IN ({placeholders})")
        params.extend(origins)
    if excluded_origins:
        placeholders = ", ".join("?" for _ in excluded_origins)
        clauses.append(f"{table_alias}.origin NOT IN ({placeholders})")
        params.extend(excluded_origins)
    if tags:
        placeholders = ", ".join("?" for _ in tags)
        clauses.append(
            f"""
            EXISTS (
                SELECT 1
                FROM {tags_relation} filter_tags
                WHERE filter_tags.session_id = {table_alias}.session_id
                  AND filter_tags.tag IN ({placeholders})
            )
            """.strip()
        )
        params.extend(tags)
    if excluded_tags:
        placeholders = ", ".join("?" for _ in excluded_tags)
        clauses.append(
            f"""
            NOT EXISTS (
                SELECT 1
                FROM {tags_relation} excluded_filter_tags
                WHERE excluded_filter_tags.session_id = {table_alias}.session_id
                  AND excluded_filter_tags.tag IN ({placeholders})
            )
            """.strip()
        )
        params.extend(excluded_tags)
    if repo_names:
        placeholders = ", ".join("?" for _ in repo_names)
        clauses.append(
            f"""
            EXISTS (
                SELECT 1
                FROM session_repos filter_session_repos
                JOIN repos filter_repos
                  ON filter_repos.repo_id = filter_session_repos.repo_id
                WHERE filter_session_repos.session_id = {table_alias}.session_id
                  AND filter_repos.repo_name IN ({placeholders})
            )
            """.strip()
        )
        params.extend(repo_names)
    if has_types:
        placeholders = ", ".join("?" for _ in has_types)
        clauses.append(
            f"""
            EXISTS (
                SELECT 1
                FROM blocks filter_blocks
                WHERE filter_blocks.session_id = {table_alias}.session_id
                  AND filter_blocks.block_type IN ({placeholders})
            )
            """.strip()
        )
        params.extend(has_types)
    if has_tool_use:
        clauses.append(f"{table_alias}.tool_use_count > 0")
    if has_thinking:
        clauses.append(f"{table_alias}.thinking_count > 0")
    if has_paste:
        clauses.append(f"{table_alias}.paste_count > 0")
    if typed_only:
        clauses.append(f"{table_alias}.paste_count = 0")
    for term in tool_terms:
        normalized = term.strip().lower()
        if not normalized:
            continue
        if normalized == "none":
            clauses.append(
                f"NOT EXISTS (SELECT 1 FROM actions filter_actions WHERE filter_actions.session_id = {table_alias}.session_id)"
            )
        else:
            clauses.append(
                f"""
                EXISTS (
                    SELECT 1
                    FROM actions filter_actions
                    WHERE filter_actions.session_id = {table_alias}.session_id
                      AND lower(filter_actions.tool_name) = ?
                )
                """.strip()
            )
            params.append(normalized)
    for term in excluded_tool_terms:
        normalized = term.strip().lower()
        if not normalized:
            continue
        if normalized == "none":
            clauses.append(
                f"EXISTS (SELECT 1 FROM actions filter_actions WHERE filter_actions.session_id = {table_alias}.session_id)"
            )
        else:
            clauses.append(
                f"""
                NOT EXISTS (
                    SELECT 1
                    FROM actions filter_actions
                    WHERE filter_actions.session_id = {table_alias}.session_id
                      AND lower(filter_actions.tool_name) = ?
                )
                """.strip()
            )
            params.append(normalized)
    for term in action_terms:
        normalized = term.strip().lower()
        if not normalized:
            continue
        if normalized == "none":
            clauses.append(
                f"NOT EXISTS (SELECT 1 FROM actions filter_actions WHERE filter_actions.session_id = {table_alias}.session_id)"
            )
        else:
            clauses.append(
                f"""
                EXISTS (
                    SELECT 1
                    FROM actions filter_actions
                    WHERE filter_actions.session_id = {table_alias}.session_id
                      AND filter_actions.semantic_type = ?
                )
                """.strip()
            )
            params.append(normalized)
    for term in excluded_action_terms:
        normalized = term.strip().lower()
        if not normalized:
            continue
        if normalized == "none":
            clauses.append(
                f"EXISTS (SELECT 1 FROM actions filter_actions WHERE filter_actions.session_id = {table_alias}.session_id)"
            )
        else:
            clauses.append(
                f"""
                NOT EXISTS (
                    SELECT 1
                    FROM actions filter_actions
                    WHERE filter_actions.session_id = {table_alias}.session_id
                      AND filter_actions.semantic_type = ?
                )
                """.strip()
            )
            params.append(normalized)
    if action_sequence:
        sequence_clause, sequence_params = _action_sequence_clause(table_alias, action_sequence)
        clauses.append(sequence_clause)
        params.extend(sequence_params)
    for term in action_text_terms:
        normalized = term.strip().lower()
        if not normalized:
            continue
        clauses.append(
            f"""
            EXISTS (
                SELECT 1
                FROM actions filter_actions
                WHERE filter_actions.session_id = {table_alias}.session_id
                  AND lower(
                      COALESCE(filter_actions.tool_name, '') || ' ' ||
                      COALESCE(filter_actions.semantic_type, '') || ' ' ||
                      COALESCE(filter_actions.tool_command, '') || ' ' ||
                      COALESCE(filter_actions.tool_path, '') || ' ' ||
                      COALESCE(filter_actions.tool_input, '') || ' ' ||
                      COALESCE(filter_actions.output_text, '')
                  ) LIKE ?
            )
            """.strip()
        )
        params.append(f"%{normalized}%")
    for term in referenced_paths:
        normalized = term.strip().replace("\\", "/").lower()
        if not normalized:
            continue
        escaped = normalized.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
        clauses.append(
            f"""
            EXISTS (
                SELECT 1
                FROM actions filter_actions
                WHERE filter_actions.session_id = {table_alias}.session_id
                  AND REPLACE(LOWER(COALESCE(filter_actions.tool_path, '')), char(92), '/') LIKE ? ESCAPE '\\'
            )
            """.strip()
        )
        params.append(f"%{escaped}%")
    if cwd_prefix:
        exact_prefix, child_prefix = escaped_sql_path_prefix_patterns(cwd_prefix)
        clauses.append(
            f"""
            EXISTS (
                SELECT 1
                FROM session_working_dirs filter_cwd
                WHERE filter_cwd.session_id = {table_alias}.session_id
                  AND (
                    REPLACE(filter_cwd.path, char(92), '/') = ?
                    OR REPLACE(filter_cwd.path, char(92), '/') LIKE ? ESCAPE '\\'
                  )
            )
            """.strip()
        )
        params.extend([exact_prefix, child_prefix])
    if message_type:
        clauses.append(
            f"""
            EXISTS (
                SELECT 1
                FROM messages filter_messages
                WHERE filter_messages.session_id = {table_alias}.session_id
                  AND filter_messages.message_type = ?
            )
            """.strip()
        )
        params.append(message_type)
    if title:
        clauses.append(f"{table_alias}.title LIKE ? ESCAPE '\\'")
        escaped_title = title.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
        params.append(f"%{escaped_title}%")
    if min_messages is not None:
        clauses.append(f"{table_alias}.message_count >= ?")
        params.append(min_messages)
    if max_messages is not None:
        clauses.append(f"{table_alias}.message_count <= ?")
        params.append(max_messages)
    if min_words is not None:
        clauses.append(f"{table_alias}.word_count >= ?")
        params.append(min_words)
    if max_words is not None:
        clauses.append(f"{table_alias}.word_count <= ?")
        params.append(max_words)
    if since_ms is not None:
        clauses.append(f"COALESCE({table_alias}.updated_at_ms, {table_alias}.created_at_ms) >= ?")
        params.append(since_ms)
    if until_ms is not None:
        clauses.append(f"COALESCE({table_alias}.updated_at_ms, {table_alias}.created_at_ms) <= ?")
        params.append(until_ms)
    if boolean_predicate is not None:
        boolean_clause, boolean_params = _boolean_predicate_clause(
            table_alias,
            boolean_predicate,
            tags_relation=tags_relation,
        )
        if boolean_clause:
            clauses.append(f"({boolean_clause})")
            params.extend(boolean_params)
    if not clauses:
        return "", params
    return f"{prefix} " + " AND ".join(clauses), params


def _action_sequence_clause(table_alias: str, action_sequence: tuple[str, ...]) -> tuple[str, list[object]]:
    joins: list[str] = []
    predicates: list[str] = []
    params: list[object] = []
    for index, term in enumerate(action_sequence):
        action_alias = f"seq_a{index}"
        message_alias = f"seq_m{index}"
        block_alias = f"seq_b{index}"
        joins.append(
            f"""
            JOIN actions {action_alias}
              ON {action_alias}.session_id = {table_alias}.session_id
            JOIN messages {message_alias}
              ON {message_alias}.message_id = {action_alias}.message_id
            JOIN blocks {block_alias}
              ON {block_alias}.block_id = {action_alias}.tool_use_block_id
            """.strip()
        )
        predicates.append(f"{action_alias}.semantic_type = ?")
        params.append(term)
        if index > 0:
            predicates.append(_action_after_predicate(index - 1, index))
    sql = (
        "EXISTS ("
        "SELECT 1 FROM sessions sequence_root "
        f"{' '.join(joins)} "
        f"WHERE sequence_root.session_id = {table_alias}.session_id "
        f"AND {' AND '.join(predicates)}"
        ")"
    )
    return sql, params


def _action_after_predicate(previous: int, current: int) -> str:
    prev_message = f"seq_m{previous}"
    curr_message = f"seq_m{current}"
    prev_block = f"seq_b{previous}"
    curr_block = f"seq_b{current}"
    return (
        "("
        f"{curr_message}.position > {prev_message}.position "
        f"OR ({curr_message}.position = {prev_message}.position "
        f"AND {curr_message}.variant_index > {prev_message}.variant_index) "
        f"OR ({curr_message}.position = {prev_message}.position "
        f"AND {curr_message}.variant_index = {prev_message}.variant_index "
        f"AND {curr_block}.position > {prev_block}.position)"
        ")"
    )


def _count_rows(conn: sqlite3.Connection, table: str) -> int:
    return _count_scalar(conn, f"SELECT COUNT(*) FROM {table}")


def _count_scalar(conn: sqlite3.Connection, sql: str, params: tuple[object, ...] = ()) -> int:
    row = conn.execute(sql, params).fetchone()
    return int(row[0] or 0) if row is not None else 0


def _table_exists(conn: sqlite3.Connection, table_name: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ? LIMIT 1",
        (table_name,),
    ).fetchone()
    return row is not None


def _ensure_messages_fts_ready(conn: sqlite3.Connection) -> None:
    """Raise ``DatabaseError`` unless message FTS is built and complete.

    Mirrors the archive FTS readiness contract for the split-
    file archive: a missing ``messages_fts`` virtual table means the search index
    was never built, and an FTS row count below the text-bearing block count
    means a bulk write suspended the triggers and never restored them. Both are
    reported as a sanitized ``DatabaseError`` so the reader degrades to a 503
    "Search index" response instead of surfacing a raw ``no such table`` /
    empty-result 200.
    """
    from polylogue.errors import DatabaseError

    repair_hint = "Run `polylogue check --repair` to rebuild the search index."
    if not _table_exists(conn, "messages_fts"):
        raise DatabaseError(f"Search index not built. {repair_hint}")
    text_blocks = _count_scalar(conn, "SELECT COUNT(*) FROM blocks WHERE search_text != ''")
    fts_rows = _count_scalar(conn, "SELECT COUNT(*) FROM messages_fts")
    if fts_rows < text_blocks:
        raise DatabaseError(f"Search index is incomplete. {repair_hint}")


def _epoch_ms_from_iso(value: str | None) -> int | None:
    if value is None:
        return None
    return int(datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp() * 1000)


# Insights whose provenance is tracked in the ``insight_materialization`` ledger
# (materializer version + source high-water mark). Threads use a separate version
# namespace and are intentionally excluded from the version-compatibility check.
_INSIGHT_MATERIALIZATION_TYPE: dict[str, str] = {
    "session_profiles": "session_profile",
    "session_work_events": "work_events",
    "session_phases": "phases",
}

# Insights whose #1278 fallback markers are stored as JSON arrays inside payload
# columns: (table_name, ((column, json_path), ...)). Session profiles carry the
# inference and enrichment fallback reasons under ``$.fallback_reasons`` in their
# respective ``inference_payload_json`` / ``enrichment_payload_json`` columns.
_INSIGHT_FALLBACK_PAYLOAD: dict[str, tuple[str, tuple[tuple[str, str], ...]]] = {
    "session_profiles": (
        "session_profiles",
        (
            ("inference_payload_json", "$.fallback_reasons"),
            ("enrichment_payload_json", "$.fallback_reasons"),
        ),
    ),
}


def _archive_insight_readiness_verdict(
    *,
    table_present: bool,
    row_count: int,
    expected_row_count: int | None,
    missing_count: int,
    stale_count: int,
    orphan_count: int,
    incompatible_count: int,
    degraded_count: int,
    ready_flags: dict[str, bool],
    total_sessions: int,
) -> InsightReadinessVerdict:
    if not table_present:
        return "missing"
    if incompatible_count:
        return "incompatible"
    if stale_count or orphan_count:
        return "stale"
    if missing_count or (expected_row_count is not None and row_count < expected_row_count):
        return "partial"
    if row_count == 0:
        # An empty archive (no sessions at all) reports every surface as empty.
        # In a populated archive a surface with 0 expected rows is vacuously
        # ready (e.g. no tags to roll up); a surface that should hold rows was
        # already caught by the partial branch above.
        if total_sessions > 0 and expected_row_count == 0:
            return "ready"
        return "empty"
    if degraded_count:
        return "degraded"
    if ready_flags and all(ready_flags.values()):
        return "ready"
    if not ready_flags:
        return "ready"
    return "unknown"


def _insight_readiness_aggregate_verdict(entries: tuple[InsightReadinessEntry, ...]) -> InsightReadinessVerdict:
    verdicts = {entry.verdict for entry in entries}
    for verdict in ("incompatible", "stale", "partial", "missing", "degraded", "unknown", "empty"):
        if verdict in verdicts:
            return verdict
    return "ready"


def _archive_insight_readiness_evidence(
    *,
    row_count: int,
    expected_row_count: int | None,
    missing_count: int,
    stale_count: int,
    orphan_count: int,
    incompatible_count: int,
    degraded_count: int,
    fallback_reason_counts: dict[str, int],
    ready_flags: dict[str, bool],
) -> tuple[str, ...]:
    values = [f"rows={row_count}"]
    if expected_row_count is not None:
        values.append(f"expected={expected_row_count}")
    if missing_count:
        values.append(f"missing={missing_count}")
    if stale_count:
        values.append(f"stale={stale_count}")
    if orphan_count:
        values.append(f"orphan={orphan_count}")
    if incompatible_count:
        values.append(f"incompatible={incompatible_count}")
    if degraded_count:
        values.append(f"degraded={degraded_count}")
    values.extend(f"fallback_reason={reason}={count}" for reason, count in fallback_reason_counts.items())
    values.extend(f"{key}={value}" for key, value in sorted(ready_flags.items()))
    return tuple(values)


def _provider_coverage_from_archive_row(row: sqlite3.Row) -> ArchiveCoverageInsight:
    session_count = int(row["session_count"] or 0)
    message_count = int(row["message_count"] or 0)
    user_message_count = int(row["user_message_count"] or 0)
    assistant_message_count = int(row["assistant_message_count"] or 0)
    user_word_sum = int(row["user_word_sum"] or 0)
    assistant_word_sum = int(row["assistant_word_sum"] or 0)
    sessions_with_tools = int(row["sessions_with_tools"] or 0)
    sessions_with_thinking = int(row["sessions_with_thinking"] or 0)
    source_name = _provider_for_origin(str(row["origin"])).value
    return ArchiveCoverageInsight(
        group_by="provider",
        bucket=source_name,
        source_name=source_name,
        session_count=session_count,
        message_count=message_count,
        user_message_count=user_message_count,
        assistant_message_count=assistant_message_count,
        avg_messages_per_session=(message_count / session_count if session_count else 0.0),
        avg_user_words=(user_word_sum / user_message_count if user_message_count else 0.0),
        avg_assistant_words=(assistant_word_sum / assistant_message_count if assistant_message_count else 0.0),
        tool_use_count=int(row["tool_use_count"] or 0),
        thinking_count=int(row["thinking_count"] or 0),
        total_sessions_with_tools=sessions_with_tools,
        total_sessions_with_thinking=sessions_with_thinking,
        tool_use_percentage=(sessions_with_tools / session_count) * 100 if session_count else 0.0,
        thinking_percentage=((sessions_with_thinking / session_count) * 100 if session_count else 0.0),
    )


def _archive_debt(
    *,
    name: str,
    category: str,
    issue_count: int,
    detail: str,
    destructive: bool = False,
) -> ArchiveDebtInsight:
    return ArchiveDebtInsight(
        debt_name=name,
        category=category,
        maintenance_target=name,
        destructive=destructive,
        issue_count=issue_count,
        healthy=issue_count == 0,
        detail=detail,
    )


def _archive_messages_fts_debt(conn: sqlite3.Connection) -> ArchiveDebtInsight:
    text_blocks = _count_scalar(conn, "SELECT COUNT(*) FROM blocks WHERE search_text != ''")
    fts_rows = _count_scalar(conn, "SELECT COUNT(*) FROM messages_fts")
    issue_count = abs(text_blocks - fts_rows)
    detail = "archive message FTS synchronized" if issue_count == 0 else f"{issue_count:,} message FTS row mismatch"
    return _archive_debt(
        name="archive_messages_fts",
        category="derived_repair",
        issue_count=issue_count,
        detail=detail,
    )


def _archive_profile_rows_debt(conn: sqlite3.Connection) -> ArchiveDebtInsight:
    missing = _count_scalar(
        conn,
        """
        SELECT COUNT(*)
        FROM sessions AS s
        WHERE NOT EXISTS (
            SELECT 1 FROM session_profiles AS p WHERE p.session_id = s.session_id
        )
        """,
    )
    orphaned = _count_scalar(
        conn,
        """
        SELECT COUNT(*)
        FROM session_profiles AS p
        WHERE NOT EXISTS (
            SELECT 1 FROM sessions AS s WHERE s.session_id = p.session_id
        )
        """,
    )
    issue_count = missing + orphaned
    detail = (
        "archive session profile rows complete"
        if issue_count == 0
        else f"{missing:,} missing and {orphaned:,} orphaned archive session profile rows"
    )
    return _archive_debt(
        name="archive_session_profile_rows",
        category="derived_repair",
        issue_count=issue_count,
        detail=detail,
    )


def _archive_profile_counts_debt(conn: sqlite3.Connection) -> ArchiveDebtInsight:
    work_event_mismatch = _count_scalar(
        conn,
        """
        SELECT COUNT(*)
        FROM session_profiles AS p
        WHERE p.work_event_count != (
            SELECT COUNT(*) FROM session_work_events AS e WHERE e.session_id = p.session_id
        )
        """,
    )
    phase_mismatch = _count_scalar(
        conn,
        """
        SELECT COUNT(*)
        FROM session_profiles AS p
        WHERE p.phase_count != (
            SELECT COUNT(*) FROM session_phases AS ph WHERE ph.session_id = p.session_id
        )
        """,
    )
    issue_count = work_event_mismatch + phase_mismatch
    detail = (
        "archive profile derived counts match timeline rows"
        if issue_count == 0
        else f"{work_event_mismatch:,} work-event and {phase_mismatch:,} phase count mismatches"
    )
    return _archive_debt(
        name="archive_profile_counts",
        category="derived_repair",
        issue_count=issue_count,
        detail=detail,
    )


def _archive_materialization_debt(conn: sqlite3.Connection) -> ArchiveDebtInsight:
    missing = _archive_missing_materialization_counts(conn)
    issue_count = sum(missing.values())
    detail = (
        "archive insight materialization rows complete"
        if issue_count == 0
        else "missing archive materialization rows: "
        + ", ".join(f"{key}={value}" for key, value in sorted(missing.items()) if value)
    )
    return _archive_debt(
        name="archive_insight_materialization",
        category="derived_repair",
        issue_count=issue_count,
        detail=detail,
    )


def _archive_source_raw_link_debt(conn: sqlite3.Connection, source_db_path: Path) -> ArchiveDebtInsight:
    raw_links = _count_scalar(conn, "SELECT COUNT(*) FROM sessions WHERE raw_id IS NOT NULL")
    if not source_db_path.exists():
        issue_count = raw_links
        detail = (
            "archive sessions have no source raw links"
            if raw_links == 0
            else f"source.db missing while {raw_links:,} sessions carry raw_id links"
        )
        return _archive_debt(
            name="archive_source_raw_links",
            category="source_ingest",
            issue_count=issue_count,
            detail=detail,
        )
    source_uri = f"file:{source_db_path}?mode=ro"
    conn.execute("ATTACH DATABASE ? AS source_debt", (source_uri,))
    try:
        missing = _count_scalar(
            conn,
            """
            SELECT COUNT(*)
            FROM sessions AS s
            WHERE s.raw_id IS NOT NULL
              AND NOT EXISTS (
                SELECT 1 FROM source_debt.raw_sessions AS r WHERE r.raw_id = s.raw_id
              )
            """,
        )
    finally:
        conn.execute("DETACH DATABASE source_debt")
    detail = "archive source raw links resolve" if missing == 0 else f"{missing:,} sessions reference missing raw rows"
    return _archive_debt(
        name="archive_source_raw_links",
        category="source_ingest",
        issue_count=missing,
        detail=detail,
    )


def _archive_user_overlay_debt(conn: sqlite3.Connection, user_db_path: Path) -> ArchiveDebtInsight:
    if not user_db_path.exists():
        return _archive_debt(
            name="archive_user_overlay_orphans",
            category="archive_cleanup",
            issue_count=0,
            detail="archive user tier absent; no overlay orphan check needed",
        )
    conn.execute("ATTACH DATABASE ? AS user_debt", (f"file:{user_db_path}?mode=ro",))
    try:
        checks = (
            "SELECT COUNT(*) FROM user_debt.session_tags u "
            "WHERE NOT EXISTS (SELECT 1 FROM sessions s WHERE s.session_id = u.session_id)",
            "SELECT COUNT(*) FROM user_debt.session_metadata u "
            "WHERE NOT EXISTS (SELECT 1 FROM sessions s WHERE s.session_id = u.session_id)",
            "SELECT COUNT(*) FROM user_debt.suppressions u "
            "WHERE NOT EXISTS (SELECT 1 FROM sessions s WHERE s.session_id = u.session_id)",
            "SELECT COUNT(*) FROM user_debt.corrections u "
            "WHERE u.target_type = 'session' "
            "AND NOT EXISTS (SELECT 1 FROM sessions s WHERE s.session_id = u.target_id)",
            "SELECT COUNT(*) FROM user_debt.marks u "
            "WHERE u.target_type = 'session' "
            "AND NOT EXISTS (SELECT 1 FROM sessions s WHERE s.session_id = u.target_id)",
            "SELECT COUNT(*) FROM user_debt.annotations u "
            "WHERE u.target_type = 'session' "
            "AND NOT EXISTS (SELECT 1 FROM sessions s WHERE s.session_id = u.target_id)",
            "SELECT COUNT(*) FROM user_debt.blackboard_notes u "
            "WHERE u.target_type = 'session' "
            "AND NOT EXISTS (SELECT 1 FROM sessions s WHERE s.session_id = u.target_id)",
        )
        issue_count = sum(_count_scalar(conn, sql) for sql in checks)
    finally:
        conn.execute("DETACH DATABASE user_debt")
    detail = (
        "archive user overlays resolve to index sessions"
        if issue_count == 0
        else f"{issue_count:,} archive user overlay rows reference missing sessions"
    )
    return _archive_debt(
        name="archive_user_overlay_orphans",
        category="archive_cleanup",
        issue_count=issue_count,
        detail=detail,
    )


def _session_latency_profile_from_archive_row(
    conn: sqlite3.Connection, row: sqlite3.Row
) -> SessionLatencyProfileInsight:
    session_id = str(row["session_id"])
    response_rows = conn.execute(
        """
        SELECT role, occurred_at_ms
        FROM messages
        WHERE session_id = ?
          AND occurred_at_ms IS NOT NULL
          AND role IN ('user', 'assistant')
        ORDER BY position, variant_index
        """,
        (session_id,),
    ).fetchall()
    agent_response_ms: list[int] = []
    user_response_ms: list[int] = []
    previous_role: str | None = None
    previous_at: int | None = None
    for message in response_rows:
        role = str(message["role"])
        occurred_at = int(message["occurred_at_ms"])
        if previous_role is not None and previous_at is not None:
            delta_ms = max(occurred_at - previous_at, 0)
            if previous_role == "user" and role == "assistant":
                agent_response_ms.append(delta_ms)
            elif previous_role == "assistant" and role == "user" and delta_ms <= 1_800_000:
                user_response_ms.append(delta_ms)
        previous_role = role
        previous_at = occurred_at
    tool_counts = _latency_tool_category_counts(conn, session_id)
    materialization = _read_archive_materialization(conn, "latency", session_id)
    return SessionLatencyProfileInsight(
        session_id=session_id,
        source_name=_provider_for_origin(str(row["origin"])).value,
        title=str(row["title"]) if row["title"] is not None else None,
        provenance=_archive_provenance(materialization),
        latency=SessionLatencyProfilePayload(
            median_tool_call_ms=0,
            p90_tool_call_ms=0,
            max_tool_call_ms=0,
            stuck_tool_count=0,
            median_agent_response_ms=_median_ms(agent_response_ms),
            median_user_response_ms=_median_ms(user_response_ms),
            tool_call_count_by_category=tool_counts,
        ),
    )


def _latency_tool_category_counts(conn: sqlite3.Connection, session_id: str) -> dict[str, int]:
    rows = conn.execute(
        """
        SELECT COALESCE(NULLIF(semantic_type, ''), 'unknown') AS category, COUNT(*) AS count
        FROM actions
        WHERE session_id = ?
        GROUP BY category
        ORDER BY count DESC, category
        """,
        (session_id,),
    ).fetchall()
    return {str(row["category"]): int(row["count"] or 0) for row in rows}


def _median_ms(values: list[int]) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return int((ordered[middle - 1] + ordered[middle]) / 2)


def _coverage_bucket_filter(
    bucket: str,
    bucket_format: str,
    *,
    origin: str | None,
    since_ms: int | None,
    until_ms: int | None,
) -> tuple[str, tuple[object, ...]]:
    clauses = ["strftime(?, s.sort_key_ms / 1000, 'unixepoch') = ?"]
    params: list[object] = [bucket_format, bucket]
    if origin is not None:
        clauses.append("s.origin = ?")
        params.append(origin)
    if since_ms is not None:
        clauses.append("s.sort_key_ms >= ?")
        params.append(since_ms)
    if until_ms is not None:
        clauses.append("s.sort_key_ms <= ?")
        params.append(until_ms)
    return "WHERE " + " AND ".join(clauses), tuple(params)


def _coverage_work_event_breakdown(
    conn: sqlite3.Connection,
    bucket: str,
    bucket_format: str,
    *,
    origin: str | None,
    since_ms: int | None,
    until_ms: int | None,
) -> dict[str, int]:
    where, params = _coverage_bucket_filter(
        bucket,
        bucket_format,
        origin=origin,
        since_ms=since_ms,
        until_ms=until_ms,
    )
    rows = conn.execute(
        f"""
        SELECT e.work_event_type, COUNT(*) AS count
        FROM sessions s
        JOIN session_work_events e ON e.session_id = s.session_id
        {where}
        GROUP BY e.work_event_type
        ORDER BY count DESC, e.work_event_type
        """,
        params,
    ).fetchall()
    return {str(row["work_event_type"]): int(row["count"] or 0) for row in rows}


def _coverage_repos_active(
    conn: sqlite3.Connection,
    bucket: str,
    bucket_format: str,
    *,
    origin: str | None,
    since_ms: int | None,
    until_ms: int | None,
) -> tuple[str, ...]:
    where, params = _coverage_bucket_filter(
        bucket,
        bucket_format,
        origin=origin,
        since_ms=since_ms,
        until_ms=until_ms,
    )
    rows = conn.execute(
        f"""
        SELECT DISTINCT COALESCE(NULLIF(r.repo_name, ''), NULLIF(r.root_path, ''), NULLIF(r.origin_url, '')) AS repo
        FROM sessions s
        JOIN session_repos sr ON sr.session_id = s.session_id
        JOIN repos r ON r.repo_id = sr.repo_id
        {where}
        ORDER BY repo
        """,
        params,
    ).fetchall()
    return tuple(str(row["repo"]) for row in rows if row["repo"])


def _coverage_provider_breakdown(
    conn: sqlite3.Connection,
    bucket: str,
    bucket_format: str,
    *,
    origin: str | None,
    since_ms: int | None,
    until_ms: int | None,
) -> dict[str, int]:
    where, params = _coverage_bucket_filter(
        bucket,
        bucket_format,
        origin=origin,
        since_ms=since_ms,
        until_ms=until_ms,
    )
    rows = conn.execute(
        f"""
        SELECT s.origin, COUNT(DISTINCT s.session_id) AS count
        FROM sessions s
        {where}
        GROUP BY s.origin
        ORDER BY count DESC, s.origin
        """,
        params,
    ).fetchall()
    return {_provider_for_origin(str(row["origin"])).value: int(row["count"] or 0) for row in rows}


def _archive_missing_materialization_counts(conn: sqlite3.Connection) -> dict[str, int]:
    insight_types = ("session_profile", "work_events", "phases", "latency", "thread")
    return {
        insight_type: _count_scalar(
            conn,
            """
            SELECT COUNT(*)
            FROM sessions AS s
            WHERE NOT EXISTS (
                SELECT 1
                FROM insight_materialization AS m
                WHERE m.insight_type = ? AND m.session_id = s.session_id
            )
            """,
            (insight_type,),
        )
        for insight_type in insight_types
    }


def _dominant_repo(rows: list[sqlite3.Row]) -> str | None:
    counts: dict[str, int] = {}
    for row in rows:
        repo = row["git_repository_url"]
        if not isinstance(repo, str) or not repo:
            continue
        counts[repo] = counts.get(repo, 0) + 1
    if not counts:
        return None
    return sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0][0]


def _thread_member_depth(rows: list[sqlite3.Row], session_id: str) -> int:
    parents = {
        str(row["session_id"]): str(row["parent_session_id"]) for row in rows if row["parent_session_id"] is not None
    }
    depth = 0
    current = session_id
    seen: set[str] = set()
    while current in parents and current not in seen:
        seen.add(current)
        current = parents[current]
        depth += 1
    return depth


def _tag_provider_breakdown(
    conn: sqlite3.Connection,
    tag: str,
    clause: str,
    params: tuple[object, ...],
    tags_relation: str,
) -> dict[str, int]:
    tag_clause, tag_params = _with_exact_tag_filter(clause, params, tag)
    rows = conn.execute(
        f"""
        SELECT s.origin, COUNT(DISTINCT s.session_id) AS count
        FROM sessions s
        JOIN {tags_relation} st ON st.session_id = s.session_id
        {tag_clause}
        GROUP BY s.origin
        ORDER BY count DESC, s.origin
        """,
        tag_params,
    ).fetchall()
    return {_provider_for_origin(str(row["origin"])).value: int(row["count"] or 0) for row in rows}


def _tag_repo_breakdown(
    conn: sqlite3.Connection,
    tag: str,
    clause: str,
    params: tuple[object, ...],
    tags_relation: str,
) -> dict[str, int]:
    tag_clause, tag_params = _with_exact_tag_filter(clause, params, tag)
    rows = conn.execute(
        f"""
        SELECT s.git_repository_url AS repo, COUNT(DISTINCT s.session_id) AS count
        FROM sessions s
        JOIN {tags_relation} st ON st.session_id = s.session_id
        {tag_clause}
          AND s.git_repository_url IS NOT NULL
          AND s.git_repository_url != ''
        GROUP BY s.git_repository_url
        ORDER BY count DESC, s.git_repository_url
        """,
        tag_params,
    ).fetchall()
    return {str(row["repo"]): int(row["count"] or 0) for row in rows}


def _with_exact_tag_filter(clause: str, params: tuple[object, ...], tag: str) -> tuple[str, tuple[object, ...]]:
    if clause:
        return f"{clause} AND st.tag = ?", (*params, tag)
    return "WHERE st.tag = ?", (tag,)


def _iso_from_ms(value: object) -> str | None:
    if not isinstance(value, int):
        return None
    return datetime.fromtimestamp(value / 1000, tz=UTC).isoformat().replace("+00:00", "Z")


def _provider_for_origin(origin: str) -> Provider:
    return {
        "claude-code-session": Provider.CLAUDE_CODE,
        "codex-session": Provider.CODEX,
        "gemini-cli-session": Provider.GEMINI_CLI,
        "hermes-session": Provider.HERMES,
        "antigravity-session": Provider.ANTIGRAVITY,
        "chatgpt-export": Provider.CHATGPT,
        "claude-ai-export": Provider.CLAUDE_AI,
        "aistudio-drive": Provider.GEMINI,
        "unknown-export": Provider.UNKNOWN,
    }.get(origin, Provider.UNKNOWN)


__all__ = ["ArchiveStore", "ArchiveSessionSearchHit", "ArchiveSessionSummary"]
