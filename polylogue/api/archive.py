"""Archive/query domain methods for the async Polylogue facade."""

from __future__ import annotations

import builtins
import json
import sqlite3
import uuid
from collections.abc import AsyncIterator, Sequence
from contextlib import suppress
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from polylogue.archive.actions.actions import Action
from polylogue.archive.attachment.models import Attachment
from polylogue.archive.blackboard import BlackboardNote
from polylogue.archive.message.messages import MessageCollection
from polylogue.archive.message.models import Message
from polylogue.archive.message.roles import MessageRoleFilter, Role
from polylogue.archive.message.types import MessageType, validate_message_type_filter
from polylogue.archive.query.spec import normalize_action_sequence, normalize_action_terms, parse_query_date
from polylogue.archive.semantic.content_projection import ContentProjectionSpec
from polylogue.archive.session.branch_type import BranchType
from polylogue.archive.session.domain_models import Session, SessionSummary
from polylogue.core.enums import Origin, Provider
from polylogue.core.json import JSONDocument
from polylogue.core.sources import origin_from_provider, provider_from_origin
from polylogue.core.user_state_targets import TARGET_MESSAGE, TARGET_SESSION
from polylogue.errors import PolylogueError
from polylogue.insights.archive import (
    SessionProfileInsight,
    SessionProfileInsightQuery,
)
from polylogue.insights.archive_models import ArchiveInsightModel
from polylogue.insights.feedback import LearningCorrection, parse_correction_kind
from polylogue.paths import archive_file_set_index_available_for_paths, archive_file_set_root_for_paths
from polylogue.storage.insights.session.records import SessionProfileRecord
from polylogue.storage.insights.session.runtime import SessionInsightStatusSnapshot
from polylogue.storage.search.models import SearchHit, SearchResult
from polylogue.storage.search.query_builders import session_web_url
from polylogue.storage.sqlite.archive_tiers import ARCHIVE_VERSION_BY_TIER
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveSessionSummary, IndexStatus
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from polylogue.storage.sqlite.archive_tiers.write import (
    ArchiveAttachmentRow,
    ArchiveMessageRow,
    ArchiveSessionEnvelope,
)
from polylogue.storage.sqlite.queries.message_query_reads import MessageTypeName
from polylogue.types import SessionId

if TYPE_CHECKING:
    from polylogue.archive.filter.filters import SessionFilter
    from polylogue.archive.message.models import Message
    from polylogue.archive.query.miss_diagnostics import QueryMissDiagnostics
    from polylogue.archive.query.search_hits import SessionSearchHit
    from polylogue.archive.query.spec import SessionQuerySpec
    from polylogue.archive.session.domain_models import Session, SessionSummary
    from polylogue.archive.session.neighbor_candidates import SessionNeighborCandidate
    from polylogue.archive.stats import ArchiveStats as StorageArchiveStats
    from polylogue.config import Config
    from polylogue.insights.audit import InsightRigorAuditQuery, InsightRigorAuditReport
    from polylogue.insights.export_bundles import InsightExportBundleRequest, InsightExportBundleResult
    from polylogue.insights.readiness import InsightReadinessQuery, InsightReadinessReport
    from polylogue.insights.resume import ResumeBrief, ResumeCandidate
    from polylogue.insights.transforms import (
        RecoveryDigest,
        RecoveryReportPreset,
        RecoveryWorkPacket,
        WorkPacketSupport,
    )
    from polylogue.operations import ArchiveStats
    from polylogue.protocols import ProgressCallback, SessionQueryRuntimeStore
    from polylogue.readiness import ReadinessReport
    from polylogue.storage.insights.session.runtime import SessionInsightCounts
    from polylogue.storage.repository import SessionRepository
    from polylogue.storage.search.models import SearchResult
    from polylogue.storage.sqlite.archive_tiers.archive import ArchiveSessionSearchHit, ArchiveSessionSummary
    from polylogue.storage.sqlite.archive_tiers.write import ArchiveSessionEnvelope
    from polylogue.surfaces.payloads import (
        BulkTagMutationResult,
        DeleteSessionResult,
        FacetsResponse,
        MetadataMutationResult,
        QueryUnitEnvelope,
        SearchEnvelope,
        SessionSearchHitPayload,
        TagMutationResult,
    )


class SessionNotFoundError(PolylogueError):
    """Raised when a requested session does not exist in the archive."""

    http_status_code = 404


def _archive_query_date_ms(field: str, value: str | None) -> int | None:
    parsed = parse_query_date(field, value)
    if parsed is None:
        return None
    return int(parsed.timestamp() * 1000)


def _archive_message_type(value: str | None) -> str | None:
    if value is None:
        return None
    return validate_message_type_filter(value).value


def _archive_action_terms(field: str, values: Sequence[str]) -> tuple[str, ...]:
    return normalize_action_terms(field, tuple(values))


def _archive_action_sequence(values: Sequence[str]) -> tuple[str, ...]:
    return normalize_action_sequence("action_sequence", ",".join(values))


def _archive_index_available(config: Config) -> bool:
    return archive_file_set_index_available_for_paths(archive_root_path=config.archive_root, db_anchor=config.db_path)


def _active_archive_root(config: Config) -> Path:
    return archive_file_set_root_for_paths(archive_root_path=config.archive_root, db_anchor=config.db_path)


def _archive_origin_for_provider(provider: str | None) -> str | None:
    if provider is None:
        return None
    return origin_from_provider(Provider.from_string(provider)).value


def _provider_for_archive_origin(origin: str) -> Provider:
    try:
        return provider_from_origin(Origin(origin))
    except ValueError:
        return Provider.UNKNOWN


def _parse_archive_datetime(value: str | None) -> datetime | None:
    if value is None:
        return None
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _archive_query_kwargs(spec: SessionQuerySpec, *, default_limit: int | None) -> dict[str, object]:
    limit = spec.limit if spec.limit is not None else default_limit
    kwargs: dict[str, object] = {
        "offset": spec.offset,
        "origins": spec.origins,
        "excluded_origins": spec.excluded_origins,
        "tags": spec.tags,
        "excluded_tags": spec.excluded_tags,
        "repo_names": spec.repo_names,
        "has_types": spec.has_types,
        "has_tool_use": spec.filter_has_tool_use,
        "has_thinking": spec.filter_has_thinking,
        "has_paste": spec.filter_has_paste,
        "tool_terms": spec.tool_terms,
        "excluded_tool_terms": spec.excluded_tool_terms,
        "action_terms": _archive_action_terms("action", spec.action_terms),
        "excluded_action_terms": _archive_action_terms("exclude_action", spec.excluded_action_terms),
        "action_sequence": _archive_action_sequence(spec.action_sequence),
        "action_text_terms": spec.action_text_terms,
        "referenced_paths": spec.referenced_path,
        "cwd_prefix": spec.cwd_prefix,
        "typed_only": spec.typed_only,
        "message_type": _archive_message_type(spec.message_type),
        "title": spec.title,
        "min_messages": spec.min_messages,
        "max_messages": spec.max_messages,
        "min_words": spec.min_words,
        "max_words": spec.max_words,
        "since_ms": _archive_query_date_ms("since", spec.since),
        "until_ms": _archive_query_date_ms("until", spec.until),
        "since_session_id": spec.since_session_id,
    }
    if limit is not None:
        kwargs["limit"] = limit
    if spec.sort is not None:
        kwargs["sort"] = spec.sort
    if spec.reverse:
        kwargs["reverse"] = True
    if spec.sample is not None:
        kwargs["sample"] = spec.sample
    return kwargs


def _archive_text_query(spec: SessionQuerySpec) -> str | None:
    terms = (*spec.query_terms, *spec.contains_terms)
    if not terms:
        return None
    return " ".join(term for term in terms if term).strip() or None


def _archive_list_summaries_for_spec(
    archive: Any,
    spec: SessionQuerySpec,
    *,
    default_limit: int,
) -> list[ArchiveSessionSummary]:
    query_text = _archive_text_query(spec)
    query_kwargs = _archive_query_kwargs(spec, default_limit=default_limit)
    if query_text is not None:
        return [archive.read_summary(hit.session_id) for hit in archive.search_summaries(query_text, **query_kwargs)]
    return cast(list[ArchiveSessionSummary], archive.list_summaries(**query_kwargs))


def _archive_count_sessions_for_spec(archive: Any, spec: SessionQuerySpec) -> int:
    query_kwargs = _archive_query_kwargs(spec, default_limit=None)
    for key in ("limit", "offset", "sort", "reverse", "sample"):
        query_kwargs.pop(key, None)
    query_text = _archive_text_query(spec)
    if query_text is not None:
        return int(archive.count_search_sessions(query_text, **query_kwargs))
    return int(archive.count_sessions(**query_kwargs))


def _archive_facet_buckets(archive: Any, spec: SessionQuerySpec | None) -> Any:
    from polylogue.archive.query.facets import FacetBuckets

    if spec is None:
        summaries = cast(list[ArchiveSessionSummary], archive.list_summaries(limit=1_000_000))
    else:
        summaries = _archive_list_summaries_for_spec(archive, spec, default_limit=1_000_000)
    origins: dict[str, int] = {}
    tags: dict[str, int] = {}
    total_messages = 0
    session_ids: list[str] = []
    for summary in summaries:
        session_ids.append(summary.session_id)
        total_messages += summary.message_count
        origins[summary.origin] = origins.get(summary.origin, 0) + 1
        for tag in set(summary.tags):
            tags[tag] = tags.get(tag, 0) + 1
    sql_buckets = _archive_aggregate_facet_families(
        archive._conn,
        session_ids=session_ids if spec is not None else None,
    )
    return FacetBuckets(
        providers=origins,
        tags=tags,
        repos=sql_buckets["repos"],
        message_types=sql_buckets["message_types"],
        action_types=sql_buckets["action_types"],
        has_flags=sql_buckets["has_flags"],
        total_sessions=len(summaries),
        total_messages=total_messages,
    )


def _archive_aggregate_facet_families(
    conn: Any,
    *,
    session_ids: list[str] | None,
) -> dict[str, dict[str, int]]:
    result: dict[str, dict[str, int]] = {
        "repos": {},
        "message_types": {},
        "action_types": {},
        "has_flags": {},
    }
    if session_ids is not None and not session_ids:
        return result

    def scoped_rows(scoped_sql: str, global_sql: str) -> list[Any]:
        if session_ids is None:
            return list(conn.execute(global_sql).fetchall())
        rows: list[Any] = []
        for start in range(0, len(session_ids), 900):
            chunk = session_ids[start : start + 900]
            placeholders = ",".join("?" for _ in chunk)
            rows.extend(conn.execute(scoped_sql.format(placeholders), chunk).fetchall())
        return rows

    def keyed(rows: list[Any]) -> dict[str, int]:
        return {str(row[0]): int(row[1] or 0) for row in rows if row[0]}

    result["repos"] = keyed(
        scoped_rows(
            """
            SELECT COALESCE(NULLIF(r.repo_name, ''), NULLIF(r.root_path, ''), NULLIF(r.origin_url, '')) AS key,
                   COUNT(DISTINCT sr.session_id) AS n
            FROM session_repos sr
            JOIN repos r ON r.repo_id = sr.repo_id
            WHERE sr.session_id IN ({})
            GROUP BY key
            """,
            """
            SELECT COALESCE(NULLIF(r.repo_name, ''), NULLIF(r.root_path, ''), NULLIF(r.origin_url, '')) AS key,
                   COUNT(DISTINCT sr.session_id) AS n
            FROM session_repos sr
            JOIN repos r ON r.repo_id = sr.repo_id
            GROUP BY key
            """,
        )
    )
    result["message_types"] = keyed(
        scoped_rows(
            "SELECT message_type, COUNT(*) AS n FROM messages WHERE session_id IN ({}) GROUP BY message_type",
            "SELECT message_type, COUNT(*) AS n FROM messages GROUP BY message_type",
        )
    )
    result["action_types"] = keyed(
        scoped_rows(
            "SELECT semantic_type, COUNT(*) AS n FROM actions WHERE session_id IN ({}) GROUP BY semantic_type",
            "SELECT semantic_type, COUNT(*) AS n FROM actions GROUP BY semantic_type",
        )
    )
    flag_rows = scoped_rows(
        """
        SELECT COALESCE(SUM(has_tool_use), 0), COALESCE(SUM(has_thinking), 0), COALESCE(SUM(has_paste), 0)
        FROM messages
        WHERE session_id IN ({})
        """,
        """
        SELECT COALESCE(SUM(has_tool_use), 0), COALESCE(SUM(has_thinking), 0), COALESCE(SUM(has_paste), 0)
        FROM messages
        """,
    )
    result["has_flags"] = {
        "has_tool_use": sum(int(row[0] or 0) for row in flag_rows),
        "has_thinking": sum(int(row[1] or 0) for row in flag_rows),
        "has_paste": sum(int(row[2] or 0) for row in flag_rows),
    }
    return result


def _archive_health_report(config: Config) -> ReadinessReport:
    from polylogue.readiness import ReadinessCheck, ReadinessReport, VerifyStatus
    from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore

    checks: list[ReadinessCheck] = []
    root = config.archive_root
    checks.append(
        ReadinessCheck(
            "archive_root",
            VerifyStatus.OK if root.exists() else VerifyStatus.WARNING,
            summary=str(root),
        )
    )

    tier_paths = {
        ArchiveTier.SOURCE: root / "source.db",
        ArchiveTier.INDEX: root / "index.db",
        ArchiveTier.EMBEDDINGS: root / "embeddings.db",
        ArchiveTier.USER: root / "user.db",
        ArchiveTier.OPS: root / "ops.db",
    }
    for tier, path in tier_paths.items():
        checks.append(_archive_tier_readiness_check(tier, path))

    try:
        with ArchiveStore.open_existing(root) as archive:
            stats = archive.stats()
            checks.append(
                ReadinessCheck(
                    "archive_index_rows",
                    VerifyStatus.OK,
                    count=stats.total_sessions,
                    summary=f"{stats.total_sessions:,} sessions / {stats.total_messages:,} messages",
                )
            )
            fts_count = _archive_count_table_rows(archive._conn, "messages_fts")
            checks.append(
                ReadinessCheck(
                    "archive_search",
                    VerifyStatus.OK if fts_count is not None else VerifyStatus.WARNING,
                    count=fts_count or 0,
                    summary="messages_fts present" if fts_count is not None else "messages_fts missing",
                )
            )
            insight_status = archive.session_insight_status()
            insights_ready = (
                insight_status.profile_rows_ready
                and insight_status.work_event_inference_rows_ready
                and insight_status.phase_inference_rows_ready
                and insight_status.threads_ready
            )
            checks.append(
                ReadinessCheck(
                    "archive_session_insights",
                    VerifyStatus.OK if insights_ready else VerifyStatus.WARNING,
                    count=insight_status.profile_row_count,
                    summary=(
                        "session insight rows ready"
                        if insights_ready
                        else "session insight rows missing or stale; run rebuild_insights"
                    ),
                )
            )
    except Exception as exc:
        checks.append(ReadinessCheck("archive_index", VerifyStatus.ERROR, summary=str(exc)))

    return ReadinessReport(checks=checks)


def _archive_tier_readiness_check(tier: ArchiveTier, path: Any) -> Any:
    from polylogue.readiness import ReadinessCheck, VerifyStatus

    name = f"archive_{tier.value}"
    if not path.exists():
        return ReadinessCheck(name, VerifyStatus.WARNING, summary=f"missing: {path}")
    try:
        conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        try:
            row = conn.execute("PRAGMA user_version").fetchone()
            version = int(row[0] or 0) if row is not None else 0
        finally:
            conn.close()
    except sqlite3.Error as exc:
        return ReadinessCheck(name, VerifyStatus.ERROR, summary=str(exc))

    expected = ARCHIVE_VERSION_BY_TIER[tier]
    return ReadinessCheck(
        name,
        VerifyStatus.OK if version == expected else VerifyStatus.ERROR,
        summary=f"v{version}/{expected}: {path}",
    )


def _archive_assertion_work_packet_entries(config: Config, session_id: str) -> tuple[Any, ...]:
    """Return assertion-backed work-packet rows for one target session."""

    from polylogue.core.refs import EvidenceRef
    from polylogue.insights.transforms import RecoveryWorkPacketEntry
    from polylogue.storage.sqlite.archive_tiers.user_write import list_assertion_claims

    user_db = _active_archive_root(config) / "user.db"
    if not user_db.exists():
        return ()
    try:
        conn = sqlite3.connect(f"file:{user_db}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
        try:
            claims = list_assertion_claims(conn, target_ref=f"session:{session_id}", limit=20)
        finally:
            conn.close()
    except sqlite3.Error:
        return ()

    entries: list[RecoveryWorkPacketEntry] = []
    fallback_ref = EvidenceRef(session_id=session_id)
    for claim in claims:
        text = claim.body_text
        if not text:
            text = json.dumps(claim.value, sort_keys=True) if claim.value is not None else claim.assertion_id
        entries.append(
            RecoveryWorkPacketEntry(
                section="assertions",
                label=claim.kind,
                text=text,
                support=_archive_assertion_support(claim.kind),
                evidence_refs=_archive_assertion_evidence_refs(claim.evidence_refs, fallback_ref),
                metadata={
                    "assertion_id": claim.assertion_id,
                    "status": claim.status or "",
                    "visibility": claim.visibility or "",
                    "author_ref": claim.author_ref or "",
                },
            )
        )
    return tuple(entries)


def _archive_assertion_support(kind: str) -> WorkPacketSupport:
    if kind in {"blocker", "caveat"}:
        return "caveat"
    if kind == "transform_candidate":
        return "inference"
    return "assertion"


def _archive_assertion_evidence_refs(raw_refs: Sequence[str], fallback: Any) -> tuple[Any, ...]:
    from polylogue.core.refs import EvidenceRef

    parsed: list[EvidenceRef] = []
    for raw_ref in raw_refs:
        try:
            parsed.append(EvidenceRef.parse(raw_ref))
        except ValueError:
            continue
    return tuple(parsed) or (fallback,)


def _archive_count_table_rows(conn: Any, table_name: str) -> int | None:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type IN ('table', 'view') AND name = ? LIMIT 1",
        (table_name,),
    ).fetchone()
    if row is None:
        return None
    count_row = conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()
    return int(count_row[0] or 0) if count_row is not None else 0


def _maybe_parse_json_object(value: str | None) -> dict[str, object] | None:
    """Decode a stored JSON object column back into a mapping for domain blocks."""
    if not value:
        return None
    try:
        parsed = json.loads(value)
    except (ValueError, TypeError):
        return None
    return parsed if isinstance(parsed, dict) else None


def _archive_attachment_to_domain(attachment: ArchiveAttachmentRow) -> Attachment:
    return Attachment(
        id=attachment.attachment_id,
        name=attachment.display_name,
        mime_type=attachment.media_type,
        size_bytes=attachment.byte_count,
        path=None,
        source_url=attachment.source_url,
        caption=attachment.caption,
    )


def _archive_message_to_domain(message: ArchiveMessageRow, *, provider: Provider) -> Message:
    text = "\n\n".join(block.text for block in message.blocks if block.text) or None
    content_blocks: list[dict[str, object]] = [
        {
            key: value
            for key, value in {
                "id": block.block_id,
                "type": block.block_type,
                "text": block.text,
                "tool_name": block.tool_name,
                "tool_id": block.tool_id,
                "semantic_type": block.semantic_type,
                "tool_input": _maybe_parse_json_object(block.tool_input),
                "metadata": _maybe_parse_json_object(block.metadata),
            }.items()
            if value is not None
        }
        for block in message.blocks
    ]
    return Message(
        id=message.message_id,
        role=Role.normalize(message.role),
        text=text,
        timestamp=_parse_archive_datetime(message.occurred_at),
        provider=provider,
        blocks=content_blocks,
        message_type=MessageType.normalize(message.message_type),
        has_tool_use=message.has_tool_use,
        has_thinking=message.has_thinking,
        has_paste=message.has_paste,
        duration_ms=message.duration_ms,
        branch_index=message.variant_index,
        parent_id=message.parent_message_id,
        attachments=[_archive_attachment_to_domain(att) for att in message.attachments],
    )


def _archive_session_to_session(session: ArchiveSessionEnvelope) -> Session:
    provider = _provider_for_archive_origin(session.origin)
    messages = [_archive_message_to_domain(message, provider=provider) for message in session.messages]
    timestamps = [message.timestamp for message in messages if message.timestamp is not None]
    # Prefer the stored session timestamps (sessions.created_at_ms/updated_at_ms);
    # fall back to the message-timestamp envelope only when the session row has
    # none. The summary projection already uses the stored values, so this keeps
    # the full-read and summary-read session timelines consistent.
    stored_created = _parse_archive_datetime(session.created_at)
    stored_updated = _parse_archive_datetime(session.updated_at)
    return Session(
        id=SessionId(session.session_id),
        origin=origin_from_provider(provider),
        title=session.title,
        messages=MessageCollection(messages=messages),
        created_at=stored_created or (min(timestamps) if timestamps else None),
        updated_at=stored_updated or (max(timestamps) if timestamps else None),
        working_directories=tuple(session.working_directories),
        git_branch=session.git_branch,
        git_repository_url=session.git_repository_url,
        parent_id=SessionId(session.parent_session_id) if session.parent_session_id else None,
        branch_type=BranchType(session.branch_type) if session.branch_type else None,
        attachments=[_archive_attachment_to_domain(att) for att in session.orphan_attachments],
    )


def _archive_summary_to_domain(summary: ArchiveSessionSummary) -> SessionSummary:
    return SessionSummary(
        id=SessionId(summary.session_id),
        origin=origin_from_provider(summary.provider),
        title=summary.title,
        created_at=_parse_archive_datetime(summary.created_at),
        updated_at=_parse_archive_datetime(summary.updated_at),
        working_directories=tuple(summary.working_directories),
        git_branch=summary.git_branch,
        git_repository_url=summary.git_repository_url,
        message_count=summary.message_count,
        tags_m2m=summary.tags,
    )


def _archive_search_hit_to_domain(hit: ArchiveSessionSearchHit) -> SearchHit:
    return SearchHit(
        session_id=hit.session_id,
        source_name=hit.provider.value,
        message_id=hit.message_id,
        title=hit.title,
        timestamp=None,
        snippet=hit.snippet,
        session_url=session_web_url(hit.session_id),
    )


def _archive_search_hit_to_payload(
    hit: ArchiveSessionSearchHit, summary: ArchiveSessionSummary
) -> SessionSearchHitPayload:
    from polylogue.surfaces.payloads import (
        SessionSearchHitPayload,
        SessionSearchMatchPayload,
        SessionSummaryPayload,
        TargetRefPayload,
        reader_anchor,
        reader_message_actions,
    )

    return SessionSearchHitPayload(
        session=SessionSummaryPayload.from_summary(
            _archive_summary_to_domain(summary),
            message_count=summary.message_count,
        ),
        match=SessionSearchMatchPayload(
            rank=hit.rank,
            retrieval_lane="dialogue",
            match_surface="message",
            target_ref=TargetRefPayload.message(session_id=hit.session_id, message_id=hit.message_id),
            anchor=reader_anchor("message", hit.message_id),
            actions=reader_message_actions(),
            message_id=hit.message_id,
            snippet=hit.snippet,
            score=None,
        ),
    )


class _ArchiveInsightExportOperations:
    """Async operations adapter for registry-backed archive insight exports."""

    def __init__(self, archive: Any) -> None:
        self._archive = archive

    async def get_insight_readiness_report(self, query: object | None = None) -> InsightReadinessReport:
        from polylogue.insights.readiness import InsightReadinessQuery

        request = query if isinstance(query, InsightReadinessQuery) else None
        return cast("InsightReadinessReport", self._archive.insight_readiness_report(request))

    async def list_session_profile_insights(self, query: object) -> list[ArchiveInsightModel]:
        return list(
            self._archive.list_session_profile_insights(
                provider=str(provider) if (provider := getattr(query, "provider", None)) is not None else None,
                workflow_shape=getattr(query, "workflow_shape", None),
                terminal_state=getattr(query, "terminal_state", None),
                since_ms=_archive_query_date_ms("since", getattr(query, "since", None)),
                until_ms=_archive_query_date_ms("until", getattr(query, "until", None)),
                tier=str(getattr(query, "tier", "merged")),
                limit=getattr(query, "limit", None),
                offset=int(getattr(query, "offset", 0)),
            )
        )

    async def list_session_work_event_insights(self, query: object) -> list[ArchiveInsightModel]:
        return list(
            self._archive.list_session_work_event_insights(
                session_id=getattr(query, "session_id", None),
                provider=str(provider) if (provider := getattr(query, "provider", None)) is not None else None,
                heuristic_label=getattr(query, "heuristic_label", None),
                since_ms=_archive_query_date_ms("since", getattr(query, "since", None)),
                until_ms=_archive_query_date_ms("until", getattr(query, "until", None)),
                limit=getattr(query, "limit", None),
                offset=int(getattr(query, "offset", 0)),
            )
        )

    async def list_session_phase_insights(self, query: object) -> list[ArchiveInsightModel]:
        return list(
            self._archive.list_session_phase_insights(
                session_id=getattr(query, "session_id", None),
                provider=str(provider) if (provider := getattr(query, "provider", None)) is not None else None,
                kind=getattr(query, "kind", None),
                since_ms=_archive_query_date_ms("since", getattr(query, "since", None)),
                until_ms=_archive_query_date_ms("until", getattr(query, "until", None)),
                limit=getattr(query, "limit", None),
                offset=int(getattr(query, "offset", 0)),
            )
        )

    async def list_thread_insights(self, query: object) -> list[ArchiveInsightModel]:
        return list(
            self._archive.list_thread_insights(
                query=getattr(query, "query", None),
                since_ms=_archive_query_date_ms("since", getattr(query, "since", None)),
                until_ms=_archive_query_date_ms("until", getattr(query, "until", None)),
                limit=getattr(query, "limit", None),
                offset=int(getattr(query, "offset", 0)),
            )
        )

    async def list_session_tag_rollup_insights(self, query: object) -> list[ArchiveInsightModel]:
        return list(
            self._archive.list_session_tag_rollup_insights(
                provider=str(provider) if (provider := getattr(query, "provider", None)) is not None else None,
                query=getattr(query, "query", None),
                since_ms=_archive_query_date_ms("since", getattr(query, "since", None)),
                until_ms=_archive_query_date_ms("until", getattr(query, "until", None)),
                limit=getattr(query, "limit", None),
                offset=int(getattr(query, "offset", 0)),
            )
        )

    async def list_archive_coverage_insights(self, query: object) -> list[ArchiveInsightModel]:
        return list(
            self._archive.list_archive_coverage_insights(
                group_by=str(getattr(query, "group_by", "provider")),
                provider=str(provider) if (provider := getattr(query, "provider", None)) is not None else None,
                since_ms=_archive_query_date_ms("since", getattr(query, "since", None)),
                until_ms=_archive_query_date_ms("until", getattr(query, "until", None)),
                limit=getattr(query, "limit", None),
                offset=int(getattr(query, "offset", 0)),
            )
        )


class _ArchiveNeighborRuntime:
    """Repository-shaped read adapter for archive neighbor discovery."""

    def __init__(self, archive: Any) -> None:
        self._archive = archive

    async def resolve_id(self, id_prefix: str, *, strict: bool = False) -> SessionId | None:
        del strict
        try:
            return SessionId(self._archive.resolve_session_id(id_prefix))
        except KeyError:
            return None

    async def get(self, session_id: str) -> Session | None:
        try:
            resolved = self._archive.resolve_session_id(session_id)
            return _archive_session_to_session(self._archive.read_session(resolved))
        except KeyError:
            return None

    async def get_eager(self, session_id: str) -> Session | None:
        return await self.get(session_id)

    async def get_summary(self, session_id: str) -> SessionSummary | None:
        try:
            resolved = self._archive.resolve_session_id(session_id)
            return _archive_summary_to_domain(self._archive.read_summary(resolved))
        except KeyError:
            return None

    async def list_summaries_by_query(self, query: object) -> builtins.list[SessionSummary]:
        return [
            _archive_summary_to_domain(summary)
            for summary in self._archive.list_summaries(**self._query_kwargs(query, default_limit=50))
        ]

    async def list_by_query(self, query: object) -> builtins.list[Session]:
        sessions: builtins.list[Session] = []
        for summary in await self.list_summaries_by_query(query):
            session = await self.get(str(summary.id))
            if session is not None:
                sessions.append(session)
        return sessions

    async def list(
        self,
        limit: int | None = 50,
        offset: int = 0,
        provider: str | None = None,
        providers: builtins.list[str] | None = None,
        since: str | None = None,
        until: str | None = None,
        title_contains: str | None = None,
        referenced_path: builtins.list[str] | None = None,
        cwd_prefix: str | None = None,
        action_terms: builtins.list[str] | None = None,
        excluded_action_terms: builtins.list[str] | None = None,
        tool_terms: builtins.list[str] | None = None,
        excluded_tool_terms: builtins.list[str] | None = None,
        has_tool_use: bool = False,
        has_thinking: bool = False,
        min_messages: int | None = None,
        max_messages: int | None = None,
        min_words: int | None = None,
        max_words: int | None = None,
        message_type: str | None = None,
    ) -> builtins.list[Session]:
        sessions: builtins.list[Session] = []
        for summary in await self.list_summaries(
            limit=limit,
            offset=offset,
            provider=provider,
            providers=providers,
            since=since,
            until=until,
            title_contains=title_contains,
            referenced_path=referenced_path,
            cwd_prefix=cwd_prefix,
            action_terms=action_terms,
            excluded_action_terms=excluded_action_terms,
            tool_terms=tool_terms,
            excluded_tool_terms=excluded_tool_terms,
            has_tool_use=has_tool_use,
            has_thinking=has_thinking,
            min_messages=min_messages,
            max_messages=max_messages,
            min_words=min_words,
            max_words=max_words,
            message_type=message_type,
        ):
            session = await self.get(str(summary.id))
            if session is not None:
                sessions.append(session)
        return sessions

    async def list_summaries(
        self,
        limit: int | None = 50,
        offset: int = 0,
        provider: str | None = None,
        providers: builtins.list[str] | None = None,
        source: str | None = None,
        since: str | None = None,
        until: str | None = None,
        title_contains: str | None = None,
        referenced_path: builtins.list[str] | None = None,
        cwd_prefix: str | None = None,
        action_terms: builtins.list[str] | None = None,
        excluded_action_terms: builtins.list[str] | None = None,
        tool_terms: builtins.list[str] | None = None,
        excluded_tool_terms: builtins.list[str] | None = None,
        has_tool_use: bool = False,
        has_thinking: bool = False,
        min_messages: int | None = None,
        max_messages: int | None = None,
        min_words: int | None = None,
        max_words: int | None = None,
        message_type: str | None = None,
    ) -> builtins.list[SessionSummary]:
        del source
        origin, origins = self._provider_filters(provider=provider, providers=providers)
        return [
            _archive_summary_to_domain(summary)
            for summary in self._archive.list_summaries(
                limit=limit or 50,
                offset=offset,
                origin=origin,
                origins=origins,
                referenced_paths=tuple(referenced_path or ()),
                cwd_prefix=cwd_prefix,
                action_terms=tuple(action_terms or ()),
                excluded_action_terms=tuple(excluded_action_terms or ()),
                tool_terms=tuple(tool_terms or ()),
                excluded_tool_terms=tuple(excluded_tool_terms or ()),
                has_tool_use=has_tool_use,
                has_thinking=has_thinking,
                message_type=_archive_message_type(message_type),
                title=title_contains,
                min_messages=min_messages,
                max_messages=max_messages,
                min_words=min_words,
                max_words=max_words,
                since_ms=_archive_query_date_ms("since", since),
                until_ms=_archive_query_date_ms("until", until),
            )
        ]

    async def count(
        self,
        provider: str | None = None,
        providers: builtins.list[str] | None = None,
        since: str | None = None,
        until: str | None = None,
        title_contains: str | None = None,
        referenced_path: builtins.list[str] | None = None,
        cwd_prefix: str | None = None,
        action_terms: builtins.list[str] | None = None,
        excluded_action_terms: builtins.list[str] | None = None,
        tool_terms: builtins.list[str] | None = None,
        excluded_tool_terms: builtins.list[str] | None = None,
        has_tool_use: bool = False,
        has_thinking: bool = False,
        min_messages: int | None = None,
        max_messages: int | None = None,
        min_words: int | None = None,
        max_words: int | None = None,
        message_type: str | None = None,
    ) -> int:
        origin, origins = self._provider_filters(provider=provider, providers=providers)
        return cast(
            int,
            self._archive.count_sessions(
                origin=origin,
                origins=origins,
                referenced_paths=tuple(referenced_path or ()),
                cwd_prefix=cwd_prefix,
                action_terms=tuple(action_terms or ()),
                excluded_action_terms=tuple(excluded_action_terms or ()),
                tool_terms=tuple(tool_terms or ()),
                excluded_tool_terms=tuple(excluded_tool_terms or ()),
                has_tool_use=has_tool_use,
                has_thinking=has_thinking,
                message_type=_archive_message_type(message_type),
                title=title_contains,
                min_messages=min_messages,
                max_messages=max_messages,
                min_words=min_words,
                max_words=max_words,
                since_ms=_archive_query_date_ms("since", since),
                until_ms=_archive_query_date_ms("until", until),
            ),
        )

    async def search_summary_hits(
        self,
        query: str,
        limit: int = 20,
        providers: builtins.list[str] | None = None,
        since: str | None = None,
    ) -> builtins.list[Any]:
        from polylogue.archive.query.search_hits import session_search_hit_from_summary

        _origin, origins = self._provider_filters(provider=None, providers=providers)
        hits = self._archive.search_summaries(
            query,
            limit=limit,
            origins=origins,
            since_ms=_archive_query_date_ms("since", since),
        )
        results: builtins.list[Any] = []
        for hit in hits:
            try:
                summary = _archive_summary_to_domain(self._archive.read_summary(hit.session_id))
            except KeyError:
                continue
            results.append(
                session_search_hit_from_summary(
                    summary,
                    rank=hit.rank,
                    retrieval_lane="dialogue",
                    match_surface="message",
                    message_id=hit.message_id,
                    snippet=hit.snippet,
                    score=None,
                )
            )
        return results

    async def search(
        self,
        query: str,
        limit: int = 20,
        providers: builtins.list[str] | None = None,
    ) -> builtins.list[Session]:
        sessions: builtins.list[Session] = []
        for hit in await self.search_summary_hits(query, limit=limit, providers=providers):
            session = await self.get(hit.session_id)
            if session is not None:
                sessions.append(session)
        return sessions

    async def search_summaries(
        self,
        query: str,
        limit: int = 20,
        providers: builtins.list[str] | None = None,
    ) -> builtins.list[SessionSummary]:
        return [hit.summary for hit in await self.search_summary_hits(query, limit=limit, providers=providers)]

    def iter_messages(
        self,
        session_id: str,
        *,
        dialogue_only: bool = False,
        message_roles: MessageRoleFilter = (),
        limit: int | None = None,
    ) -> AsyncIterator[Message]:
        async def _iter() -> AsyncIterator[Message]:
            session = await self.get(session_id)
            if session is None:
                return
            count = 0
            for message in session.messages:
                if dialogue_only and message.role not in {Role.USER, Role.ASSISTANT}:
                    continue
                if message_roles and message.role not in message_roles:
                    continue
                if limit is not None and count >= limit:
                    break
                count += 1
                yield message

        return _iter()

    async def search_similar(
        self,
        text: str,
        limit: int = 10,
        vector_provider: object | None = None,
    ) -> builtins.list[Session]:
        del text, limit, vector_provider
        return []

    def _query_kwargs(self, query: object, *, default_limit: int) -> dict[str, object]:
        origin, origins = self._provider_filters(
            provider=getattr(query, "provider", None),
            providers=builtins.list(getattr(query, "providers", ()) or ()),
        )
        return {
            "limit": getattr(query, "limit", None) or default_limit,
            "offset": int(getattr(query, "offset", 0) or 0),
            "origin": origin,
            "origins": origins,
            "referenced_paths": tuple(getattr(query, "referenced_path", ()) or ()),
            "cwd_prefix": getattr(query, "cwd_prefix", None),
            "action_terms": tuple(getattr(query, "action_terms", ()) or ()),
            "excluded_action_terms": tuple(getattr(query, "excluded_action_terms", ()) or ()),
            "tool_terms": tuple(getattr(query, "tool_terms", ()) or ()),
            "excluded_tool_terms": tuple(getattr(query, "excluded_tool_terms", ()) or ()),
            "has_tool_use": bool(getattr(query, "has_tool_use", False)),
            "has_thinking": bool(getattr(query, "has_thinking", False)),
            "has_paste": bool(getattr(query, "has_paste", False)),
            "message_type": _archive_message_type(getattr(query, "message_type", None)),
            "title": getattr(query, "title_contains", None),
            "min_messages": getattr(query, "min_messages", None),
            "max_messages": getattr(query, "max_messages", None),
            "min_words": getattr(query, "min_words", None),
            "max_words": getattr(query, "max_words", None),
            "since_ms": _archive_query_date_ms("since", getattr(query, "since", None)),
            "until_ms": _archive_query_date_ms("until", getattr(query, "until", None)),
        }

    def _provider_filters(
        self,
        *,
        provider: str | None,
        providers: builtins.list[str] | None,
    ) -> tuple[str | None, tuple[str, ...]]:
        origin = _archive_origin_for_provider(provider)
        origins: builtins.list[str] = []
        for provider_value in providers or []:
            candidate = _archive_origin_for_provider(provider_value)
            if candidate is not None:
                origins.append(candidate)
        return origin, tuple(origins)


def _actions_for_session(session: Session) -> tuple[Action, ...]:
    """Derive ordered actions from an archive session's tool blocks.

    Mirrors the ingest-time derivation (``pipeline/services/ingest_worker``):
    each message's content blocks are parsed into tool calls, then promoted
    to ``Action`` records. No storage round-trip — the domain
    session already carries the content blocks the actions are built from.
    """
    from polylogue.archive.actions.actions import build_actions, build_tool_calls_from_content_blocks

    events: builtins.list[Action] = []
    for message in session.messages:
        calls = build_tool_calls_from_content_blocks(
            provider=provider_from_origin(session.origin),
            content_blocks=message.blocks,
        )
        events.extend(build_actions(message, calls))
    return tuple(events)


def _rebuild_archive_session_insights(
    archive: Any,
    *,
    session_ids: Sequence[str] | None = None,
    progress_callback: ProgressCallback | None = None,
) -> SessionInsightCounts:
    """Rebuild durable session insights via the canonical materializer.

    This is a thin adapter over
    ``polylogue.storage.insights.session.rebuild.rebuild_session_insights_sync``
    — the single rebuild stack shared with daemon convergence (#1743 P13). It
    resolves any session-id aliases against the archive, then delegates the
    whole rebuild (profiles, latency, work events, phases, threads +
    thread_sessions + 'thread' markers, tag rollups, provider-day aggregates)
    to the canonical path, which commits internally.
    """
    from polylogue.storage.insights.session.rebuild import rebuild_session_insights_sync
    from polylogue.storage.insights.session.runtime import SessionInsightCounts

    resolved_ids = _archive_rebuild_session_ids(archive, session_ids) if session_ids is not None else None
    if session_ids is not None and not resolved_ids:
        return SessionInsightCounts()
    return rebuild_session_insights_sync(
        archive._conn,
        session_ids=resolved_ids,
        progress_callback=progress_callback,
    )


def _archive_rebuild_session_ids(archive: Any, session_ids: Sequence[str] | None) -> tuple[str, ...]:
    if session_ids is None:
        return tuple(summary.session_id for summary in archive.list_summaries(limit=1_000_000))
    resolved: list[str] = []
    for session_id in session_ids:
        with suppress(KeyError):
            resolved.append(archive.resolve_session_id(str(session_id)))
    return tuple(dict.fromkeys(resolved))


def _archive_message_matches(
    message: Message,
    *,
    message_role: MessageRoleFilter,
    message_type: MessageTypeName | None,
    since_ms: int | None = None,
    until_ms: int | None = None,
) -> bool:
    if message_role and message.role not in message_role:
        return False
    if message_type is not None and message.message_type != MessageType.normalize(message_type):
        return False
    occurred_ms = int(message.timestamp.timestamp() * 1000) if message.timestamp is not None else None
    if since_ms is not None and (occurred_ms is None or occurred_ms < since_ms):
        return False
    return not (until_ms is not None and (occurred_ms is None or occurred_ms > until_ms))


class PolylogueArchiveMixin:
    if TYPE_CHECKING:

        @property
        def config(self) -> Config: ...

        @property
        def repository(self) -> SessionRepository: ...

    async def get_session(
        self,
        session_id: str,
        *,
        content_projection: ContentProjectionSpec | None = None,
    ) -> Session | None:
        from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore

        with ArchiveStore.open_existing(_active_archive_root(self.config)) as archive:
            try:
                resolved_id = archive.resolve_session_id(session_id)
            except KeyError:
                return None
            session = _archive_session_to_session(archive.read_session(resolved_id))
            if content_projection is None or not content_projection.filters_content():
                return session
            return session.with_content_projection(content_projection)

    async def recovery_digest(self, session_id: str) -> RecoveryDigest | None:
        """Compile one resolved session and its child links into a recovery digest."""
        from polylogue.insights.transforms import compile_recovery_digest
        from polylogue.storage.query_models import SessionRecordQuery

        session = await self.get_session(session_id)
        if session is None:
            return None
        resolved_session_id = str(session.id)
        session_links: list[dict[str, object]] = await self.repository.queries.list_session_links_for_session(
            resolved_session_id
        )
        children = await self.repository.queries.list_sessions(SessionRecordQuery(parent_id=resolved_session_id))
        session_links.extend(
            {
                "dst_origin": child.origin.value,
                "dst_native_id": child.native_id,
                "resolved_dst_session_id": str(child.session_id),
                "status": "resolved",
                "link_type": child.branch_type.value if child.branch_type is not None else "child",
            }
            for child in children
        )
        return compile_recovery_digest(session, session_links=session_links)

    async def recovery_report(self, session_id: str, preset: RecoveryReportPreset) -> str | None:
        """Render one deterministic recovery report preset for a session."""
        if preset == "work-packet":
            packet = await self.recovery_work_packet(session_id)
            return None if packet is None else packet.render_markdown()
        digest = await self.recovery_digest(session_id)
        if digest is None:
            return None
        return digest.report_markdown(preset)

    async def recovery_work_packet(self, session_id: str) -> RecoveryWorkPacket | None:
        """Return the storage-free continuation packet DTO for a session."""
        digest = await self.recovery_digest(session_id)
        if digest is None:
            return None
        packet = digest.work_packet()
        assertion_entries = _archive_assertion_work_packet_entries(self.config, digest.session_id)
        if not assertion_entries:
            return packet
        return packet.model_copy(update={"entries": (*packet.entries, *assertion_entries)})

    async def list_read_view_profiles(self) -> list[JSONDocument]:
        """List executable read-view profile metadata."""
        from polylogue.archive.viewport import read_view_profile_payloads

        return list(read_view_profile_payloads())

    async def explain_query_expression(self, expression: str) -> JSONDocument:
        """Explain query DSL parsing, AST metadata, and lowering details."""
        from polylogue.archive.query.expression import explain_expression

        return cast(JSONDocument, explain_expression(expression).to_payload())

    async def query_units(
        self,
        expression: str,
        *,
        limit: int = 50,
        offset: int = 0,
        origin: str | None = None,
        origins: tuple[str, ...] = (),
        excluded_origins: tuple[str, ...] = (),
        tag: str | None = None,
        tags: tuple[str, ...] = (),
        excluded_tags: tuple[str, ...] = (),
        repo: str | None = None,
        repo_names: tuple[str, ...] = (),
        has_types: tuple[str, ...] = (),
        tool_terms: tuple[str, ...] = (),
        excluded_tool_terms: tuple[str, ...] = (),
        action_terms: tuple[str, ...] = (),
        excluded_action_terms: tuple[str, ...] = (),
        action_sequence: tuple[str, ...] = (),
        action_text_terms: tuple[str, ...] = (),
        referenced_paths: tuple[str, ...] = (),
        cwd_prefix: str | None = None,
        title: str | None = None,
        since: str | None = None,
        until: str | None = None,
        has_tool_use: bool = False,
        has_thinking: bool = False,
        has_paste: bool = False,
        typed_only: bool = False,
        min_messages: int | None = None,
        max_messages: int | None = None,
        min_words: int | None = None,
        max_words: int | None = None,
        message_type: str | None = None,
    ) -> QueryUnitEnvelope:
        """Execute a terminal unit-source query."""
        from polylogue.archive.query.expression import ExpressionCompileError, parse_unit_source_expression
        from polylogue.archive.query.unit_results import query_unit_rows, query_unit_session_filters
        from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore

        source = parse_unit_source_expression(expression)
        if source is None:
            raise ExpressionCompileError(
                "query_units requires an explicit messages/actions/blocks/assertions where expression",
                field=None,
            )
        session_filters = query_unit_session_filters(
            origin=origin,
            origins=origins,
            tags=(tag,) if tag else tags,
            excluded_tags=excluded_tags,
            excluded_origins=excluded_origins,
            repo=repo,
            repo_names=repo_names,
            has_types=has_types,
            tool_terms=tool_terms,
            excluded_tool_terms=excluded_tool_terms,
            action_terms=action_terms,
            excluded_action_terms=excluded_action_terms,
            action_sequence=action_sequence,
            action_text_terms=action_text_terms,
            referenced_paths=referenced_paths,
            cwd_prefix=cwd_prefix,
            title=title,
            since=since,
            until=until,
            has_tool_use=has_tool_use,
            has_thinking=has_thinking,
            has_paste=has_paste,
            typed_only=typed_only,
            min_messages=min_messages,
            max_messages=max_messages,
            min_words=min_words,
            max_words=max_words,
            message_type=message_type,
        )
        with ArchiveStore.open_existing(_active_archive_root(self.config)) as archive:
            return query_unit_rows(
                archive, source, query=expression, limit=limit, offset=offset, session_filters=session_filters
            )

    async def query_completions(
        self,
        kind: str,
        *,
        incomplete: str = "",
        unit: str | None = None,
        field: str | None = None,
    ) -> JSONDocument:
        """Return shared query/action completion metadata for adapters."""
        from polylogue.archive.query.completions import query_completion_payload

        return cast(
            JSONDocument,
            query_completion_payload(kind, incomplete=incomplete, unit=unit, field=field),
        )

    async def get_sessions(
        self,
        session_ids: list[str],
        *,
        content_projection: ContentProjectionSpec | None = None,
    ) -> list[Session]:
        rows: list[Session] = []
        for session_id in session_ids:
            row = await self.get_session(session_id, content_projection=content_projection)
            if row is not None:
                rows.append(row)
        return rows

    async def get_actions(self, session_id: str) -> tuple[Action, ...]:
        """Derive a session's actions from its content blocks.

        ``index.db`` exposes an ``actions`` view; these actions
        are derived on read from the session's tool-use/tool-result blocks —
        the same source the archive materializer hashed into durable rows.
        Returns an empty tuple when the session is absent.
        """
        session = await self.get_session(session_id)
        if session is None:
            return ()
        return _actions_for_session(session)

    async def get_actions_batch(
        self,
        session_ids: builtins.list[str],
    ) -> dict[str, tuple[Action, ...]]:
        """Batch counterpart of :meth:`get_actions`.

        Missing sessions are omitted from the result mapping, mirroring
        the archive repository batch reader.
        """
        sessions = await self.get_sessions(session_ids)
        return {str(session.id): _actions_for_session(session) for session in sessions}

    async def list_sessions(
        self,
        origin: str | None = None,
        limit: int | None = None,
        content_projection: ContentProjectionSpec | None = None,
    ) -> list[Session]:
        from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore

        with ArchiveStore.open_existing(_active_archive_root(self.config)) as archive:
            summaries = archive.list_summaries(
                origin=origin,
                limit=50 if limit is None else limit,
            )
            sessions = [_archive_session_to_session(archive.read_session(summary.session_id)) for summary in summaries]
            if content_projection is None or not content_projection.filters_content():
                return sessions
            return [session.with_content_projection(content_projection) for session in sessions]

    async def list_summaries(
        self,
        *,
        limit: int | None = 50,
        offset: int = 0,
        origin: str | None = None,
    ) -> builtins.list[SessionSummary]:
        """List archive session summaries without hydrating full sessions.

        The cheap read path for callers that only need summary fields
        (title, timestamps, origin, model, counts). Use
        :meth:`list_sessions` when full message bodies are required.
        """
        from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore

        with ArchiveStore.open_existing(_active_archive_root(self.config)) as archive:
            return [
                _archive_summary_to_domain(summary)
                for summary in archive.list_summaries(
                    origin=origin,
                    limit=50 if limit is None else limit,
                    offset=offset,
                )
            ]

    async def list_sessions_for_spec(
        self,
        spec: SessionQuerySpec,
        *,
        content_projection: ContentProjectionSpec | None = None,
    ) -> list[Session]:
        """Run a ``SessionQuerySpec`` directly, returning full sessions.

        The spec-based counterpart of :meth:`list_sessions` (which only
        takes origin/limit). A vector provider is resolved only for explicit
        semantic (``similar_text``) or ``hybrid`` specs; plain filter specs run
        without touching the embeddings tier.
        """
        vector_provider = None
        if spec.similar_text or spec.retrieval_lane == "hybrid":
            from polylogue.storage.search_providers import create_vector_provider

            archive_root = _active_archive_root(self.config)
            with suppress(ValueError, ImportError):
                vector_provider = create_vector_provider(self.config, db_path=archive_root / "embeddings.db")
        sessions = await spec.list(self.config, vector_provider=vector_provider)
        if content_projection is None or not content_projection.filters_content():
            return sessions
        return [session.with_content_projection(content_projection) for session in sessions]

    async def search_session_hits(self, spec: SessionQuerySpec) -> builtins.list[SessionSearchHit]:
        """Return archive FTS/hybrid search-hit projections for a query spec.

        The hit projection carries match snippets and ranking metadata the
        :class:`SearchEnvelope` builder needs, distinct from the full
        session hydration of :meth:`list_sessions_for_spec`.
        """
        from polylogue.archive.query.search_hits import search_hits_for_plan

        vector_provider = None
        if spec.similar_text or spec.retrieval_lane == "hybrid":
            from polylogue.storage.search_providers import create_vector_provider

            archive_root = _active_archive_root(self.config)
            with suppress(ValueError, ImportError):
                vector_provider = create_vector_provider(self.config, db_path=archive_root / "embeddings.db")
        return await search_hits_for_plan(spec.to_plan(vector_provider=vector_provider), self.config)

    async def diagnose_query_miss(self, spec: SessionQuerySpec) -> QueryMissDiagnostics:
        """Best-effort explanation for an empty archive query result.

        The diagnostic is duck-typed over this facade: it reads whatever
        archive count/stats methods are available and degrades gracefully when
        a probe is absent.
        """
        from polylogue.archive.query.miss_diagnostics import diagnose_query_miss

        return await diagnose_query_miss(self, spec, config=self.config)

    async def storage_stats(self) -> StorageArchiveStats:
        """Lightweight archive stats without recent-session hydration.

        The cheap counterpart of :meth:`stats`: counts and provider/tag
        breakdowns straight from ``index.db`` for status surfaces.
        """
        from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore

        with ArchiveStore.open_existing(_active_archive_root(self.config)) as archive:
            return archive.stats()

    async def search(
        self,
        query: str,
        *,
        limit: int = 100,
        source: str | None = None,
        since: str | None = None,
    ) -> SearchResult:
        from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore

        with ArchiveStore.open_existing(_active_archive_root(self.config)) as archive:
            return SearchResult(
                hits=[
                    _archive_search_hit_to_domain(hit)
                    for hit in archive.search_summaries(
                        query,
                        limit=limit,
                        origin=source,
                        since_ms=_archive_query_date_ms("since", since),
                    )
                ]
            )

    async def search_envelope(
        self,
        query: str,
        *,
        limit: int = 50,
        offset: int = 0,
        origin: str | None = None,
        since: str | None = None,
        until: str | None = None,
        retrieval_lane: str = "auto",
        sort: str | None = None,
        cursor: str | None = None,
    ) -> SearchEnvelope:
        """Return the canonical :class:`SearchEnvelope` for a query (#1266).

        Pass ``cursor`` (an opaque token previously returned as
        :attr:`SearchEnvelope.next_cursor`) to fetch the next page
        without losing or duplicating hits even when the archive grew
        between requests (#1268).
        """
        from polylogue.archive.query.spec import SessionQuerySpec
        from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
        from polylogue.surfaces.payloads import (
            InvalidSearchCursorError,
            build_search_envelope,
            decode_search_cursor,
            search_cursor_lane_matches_request,
        )

        spec = SessionQuerySpec.from_params(
            {
                "query": query,
                "origin": origin,
                "since": since,
                "until": until,
                "retrieval_lane": retrieval_lane,
                "sort": sort,
                "limit": limit,
                "offset": offset,
                "cursor": cursor,
            },
            strict=True,
        )
        decoded_cursor = decode_search_cursor(spec.cursor) if spec.cursor else None
        if decoded_cursor is not None and not search_cursor_lane_matches_request(
            decoded_cursor.lane,
            spec.retrieval_lane,
        ):
            raise InvalidSearchCursorError(
                f"cursor was minted for retrieval_lane={decoded_cursor.lane!r} "
                f"but this request is {spec.retrieval_lane!r}"
            )
        display_limit = spec.limit or limit
        display_offset = spec.offset
        fetch_offset = decoded_cursor.r if decoded_cursor is not None else display_offset
        fetch_limit = display_limit * 2 if decoded_cursor is not None else display_limit
        origins = spec.origins
        excluded_origins = spec.excluded_origins
        with ArchiveStore.open_existing(_active_archive_root(self.config)) as archive:
            if query.strip():
                hits = archive.search_summaries(
                    query,
                    limit=fetch_limit,
                    offset=fetch_offset,
                    sort=spec.sort,
                    origins=origins,
                    excluded_origins=excluded_origins,
                    since_ms=_archive_query_date_ms("since", since),
                    until_ms=_archive_query_date_ms("until", until),
                )
                total = archive.count_search_sessions(
                    query,
                    origins=origins,
                    excluded_origins=excluded_origins,
                    since_ms=_archive_query_date_ms("since", since),
                    until_ms=_archive_query_date_ms("until", until),
                )
                hit_payloads = tuple(
                    _archive_search_hit_to_payload(hit, archive.read_summary(hit.session_id)) for hit in hits
                )
            else:
                hit_payloads = ()
                total = archive.count_sessions(
                    origins=origins,
                    excluded_origins=excluded_origins,
                    since_ms=_archive_query_date_ms("since", since),
                    until_ms=_archive_query_date_ms("until", until),
                )
        return build_search_envelope(
            hit_payloads,
            total=total,
            limit=display_limit,
            offset=display_offset,
            query=query,
            retrieval_lane="dialogue" if query.strip() else spec.retrieval_lane,
            sort=spec.sort,
            cursor=decoded_cursor,
        )

    async def archive_count_sessions(
        self,
        *,
        origin: str | None = None,
        excluded_origins: Sequence[str] = (),
        tags: Sequence[str] = (),
        excluded_tags: Sequence[str] = (),
        repo_names: Sequence[str] = (),
        has_types: Sequence[str] = (),
        has_tool_use: bool = False,
        has_thinking: bool = False,
        has_paste: bool = False,
        tool_terms: Sequence[str] = (),
        excluded_tool_terms: Sequence[str] = (),
        action_terms: Sequence[str] = (),
        excluded_action_terms: Sequence[str] = (),
        action_sequence: Sequence[str] = (),
        action_text_terms: Sequence[str] = (),
        referenced_paths: Sequence[str] = (),
        cwd_prefix: str | None = None,
        typed_only: bool = False,
        message_type: str | None = None,
        title: str | None = None,
        min_messages: int | None = None,
        max_messages: int | None = None,
        min_words: int | None = None,
        max_words: int | None = None,
        since: str | None = None,
        until: str | None = None,
    ) -> int:
        """Count sessions in the index tier."""
        from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore

        with ArchiveStore.open_existing(_active_archive_root(self.config)) as archive:
            return archive.count_sessions(
                origin=origin,
                excluded_origins=tuple(excluded_origins),
                tags=tuple(tags),
                excluded_tags=tuple(excluded_tags),
                repo_names=tuple(repo_names),
                has_types=tuple(has_types),
                has_tool_use=has_tool_use,
                has_thinking=has_thinking,
                has_paste=has_paste,
                tool_terms=tuple(tool_terms),
                excluded_tool_terms=tuple(excluded_tool_terms),
                action_terms=_archive_action_terms("action", action_terms),
                excluded_action_terms=_archive_action_terms("exclude_action", excluded_action_terms),
                action_sequence=_archive_action_sequence(action_sequence),
                action_text_terms=tuple(action_text_terms),
                referenced_paths=tuple(referenced_paths),
                cwd_prefix=cwd_prefix,
                typed_only=typed_only,
                message_type=_archive_message_type(message_type),
                title=title,
                min_messages=min_messages,
                max_messages=max_messages,
                min_words=min_words,
                max_words=max_words,
                since_ms=_archive_query_date_ms("since", since),
                until_ms=_archive_query_date_ms("until", until),
            )

    async def archive_list_sessions(
        self,
        *,
        origin: str | None = None,
        excluded_origins: Sequence[str] = (),
        tags: Sequence[str] = (),
        excluded_tags: Sequence[str] = (),
        repo_names: Sequence[str] = (),
        has_types: Sequence[str] = (),
        has_tool_use: bool = False,
        has_thinking: bool = False,
        has_paste: bool = False,
        tool_terms: Sequence[str] = (),
        excluded_tool_terms: Sequence[str] = (),
        action_terms: Sequence[str] = (),
        excluded_action_terms: Sequence[str] = (),
        action_sequence: Sequence[str] = (),
        action_text_terms: Sequence[str] = (),
        referenced_paths: Sequence[str] = (),
        cwd_prefix: str | None = None,
        typed_only: bool = False,
        message_type: str | None = None,
        title: str | None = None,
        min_messages: int | None = None,
        max_messages: int | None = None,
        min_words: int | None = None,
        max_words: int | None = None,
        since: str | None = None,
        until: str | None = None,
        limit: int = 50,
        offset: int = 0,
        sample: bool = False,
    ) -> list[ArchiveSessionSummary]:
        """List session summaries."""
        from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore

        with ArchiveStore.open_existing(_active_archive_root(self.config)) as archive:
            return list(
                archive.list_summaries(
                    origin=origin,
                    excluded_origins=tuple(excluded_origins),
                    tags=tuple(tags),
                    excluded_tags=tuple(excluded_tags),
                    repo_names=tuple(repo_names),
                    has_types=tuple(has_types),
                    has_tool_use=has_tool_use,
                    has_thinking=has_thinking,
                    has_paste=has_paste,
                    tool_terms=tuple(tool_terms),
                    excluded_tool_terms=tuple(excluded_tool_terms),
                    action_terms=_archive_action_terms("action", action_terms),
                    excluded_action_terms=_archive_action_terms("exclude_action", excluded_action_terms),
                    action_sequence=_archive_action_sequence(action_sequence),
                    action_text_terms=tuple(action_text_terms),
                    referenced_paths=tuple(referenced_paths),
                    cwd_prefix=cwd_prefix,
                    typed_only=typed_only,
                    message_type=_archive_message_type(message_type),
                    title=title,
                    min_messages=min_messages,
                    max_messages=max_messages,
                    min_words=min_words,
                    max_words=max_words,
                    since_ms=_archive_query_date_ms("since", since),
                    until_ms=_archive_query_date_ms("until", until),
                    limit=limit,
                    offset=offset,
                    sample=sample,
                )
            )

    async def archive_search_sessions(
        self,
        query: str,
        *,
        origin: str | None = None,
        excluded_origins: Sequence[str] = (),
        tags: Sequence[str] = (),
        excluded_tags: Sequence[str] = (),
        repo_names: Sequence[str] = (),
        has_types: Sequence[str] = (),
        has_tool_use: bool = False,
        has_thinking: bool = False,
        has_paste: bool = False,
        tool_terms: Sequence[str] = (),
        excluded_tool_terms: Sequence[str] = (),
        action_terms: Sequence[str] = (),
        excluded_action_terms: Sequence[str] = (),
        action_sequence: Sequence[str] = (),
        action_text_terms: Sequence[str] = (),
        referenced_paths: Sequence[str] = (),
        cwd_prefix: str | None = None,
        typed_only: bool = False,
        message_type: str | None = None,
        title: str | None = None,
        min_messages: int | None = None,
        max_messages: int | None = None,
        min_words: int | None = None,
        max_words: int | None = None,
        since: str | None = None,
        until: str | None = None,
        limit: int = 20,
    ) -> list[ArchiveSessionSearchHit]:
        """Search session block text."""
        from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore

        with ArchiveStore.open_existing(_active_archive_root(self.config)) as archive:
            return list(
                archive.search_summaries(
                    query,
                    origin=origin,
                    excluded_origins=tuple(excluded_origins),
                    tags=tuple(tags),
                    excluded_tags=tuple(excluded_tags),
                    repo_names=tuple(repo_names),
                    has_types=tuple(has_types),
                    has_tool_use=has_tool_use,
                    has_thinking=has_thinking,
                    has_paste=has_paste,
                    tool_terms=tuple(tool_terms),
                    excluded_tool_terms=tuple(excluded_tool_terms),
                    action_terms=_archive_action_terms("action", action_terms),
                    excluded_action_terms=_archive_action_terms("exclude_action", excluded_action_terms),
                    action_sequence=_archive_action_sequence(action_sequence),
                    action_text_terms=tuple(action_text_terms),
                    referenced_paths=tuple(referenced_paths),
                    cwd_prefix=cwd_prefix,
                    typed_only=typed_only,
                    message_type=_archive_message_type(message_type),
                    title=title,
                    min_messages=min_messages,
                    max_messages=max_messages,
                    min_words=min_words,
                    max_words=max_words,
                    since_ms=_archive_query_date_ms("since", since),
                    until_ms=_archive_query_date_ms("until", until),
                    limit=limit,
                )
            )

    async def archive_get_session(self, session_id: str) -> ArchiveSessionEnvelope | None:
        """Read a full session envelope by exact id or prefix."""
        from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore

        with ArchiveStore.open_existing(_active_archive_root(self.config)) as archive:
            try:
                resolved_id = archive.resolve_session_id(session_id)
            except KeyError:
                return None
            return archive.read_session(resolved_id)

    async def get_session_insight_status(self) -> SessionInsightStatusSnapshot:
        from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore

        with ArchiveStore.open_existing(_active_archive_root(self.config)) as archive:
            return archive.session_insight_status()

    async def get_session_profile_insight(
        self,
        session_id: str,
        *,
        tier: str = "merged",
    ) -> SessionProfileInsight | None:
        from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore

        with ArchiveStore.open_existing(_active_archive_root(self.config)) as archive:
            return archive.get_session_profile_insight(session_id, tier=tier)

    async def get_session_profile_record(
        self,
        session_id: str,
    ) -> SessionProfileRecord | None:
        from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore

        with ArchiveStore.open_existing(_active_archive_root(self.config)) as archive:
            return archive.get_session_profile_record(session_id)

    async def list_session_profile_insights(
        self,
        query: SessionProfileInsightQuery | None = None,
    ) -> list[SessionProfileInsight]:
        from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore

        request = query or SessionProfileInsightQuery()
        with ArchiveStore.open_existing(_active_archive_root(self.config)) as archive:
            return archive.list_session_profile_insights(
                provider=request.provider,
                workflow_shape=request.workflow_shape,
                terminal_state=request.terminal_state,
                since_ms=_archive_query_date_ms("since", request.since),
                until_ms=_archive_query_date_ms("until", request.until),
                tier=request.tier,
                limit=request.limit,
                offset=request.offset,
                min_wallclock_seconds=request.min_wallclock_seconds,
                max_wallclock_seconds=request.max_wallclock_seconds,
                sort=request.sort,
            )

    def filter(self) -> SessionFilter:
        from polylogue.archive.filter.filters import SessionFilter
        from polylogue.storage.search_providers import create_vector_provider

        archive_root = _active_archive_root(self.config)
        vector_provider = None
        with suppress(ValueError, ImportError):
            vector_provider = create_vector_provider(self.config, db_path=archive_root / "embeddings.db")

        return SessionFilter(
            archive_root=archive_root,
            config=self.config,
            vector_provider=vector_provider,
        )

    async def stats(self) -> ArchiveStats:
        from polylogue.operations import ArchiveStats as PublicArchiveStats
        from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore

        with ArchiveStore.open_existing(_active_archive_root(self.config)) as archive:
            stats = archive.stats()
            word_row = archive._conn.execute("SELECT COALESCE(SUM(word_count), 0) FROM sessions").fetchone()
            recent = [
                _archive_session_to_session(archive.read_session(summary.session_id))
                for summary in archive.list_summaries(limit=5)
            ]
            return PublicArchiveStats(
                session_count=stats.total_sessions,
                message_count=stats.total_messages,
                word_count=int(word_row[0] or 0) if word_row is not None else 0,
                origins=stats.origins,
                tags={},
                last_sync=None,
                recent=recent,
            )

    async def facets(
        self,
        spec: SessionQuerySpec | None = None,
        *,
        include_idf: bool = True,
    ) -> FacetsResponse:
        """Compute scoped + global facet aggregates over the archive.

        When ``spec`` carries any active filter, the scoped buckets are
        rolled from that filter's summary list and ``scoped_to_query``
        becomes true.  The global buckets always reflect the
        unfiltered archive.  Surfaces (daemon HTTP, MCP, CLI) call into
        this method so the scope vocabulary stays in one place
        (#1269 / slice D of #873).
        """
        from polylogue.archive.query.facets import (
            FacetBuckets as _FacetBuckets,
        )
        from polylogue.archive.query.facets import (
            compute_idf,
        )
        from polylogue.surfaces.payloads import (
            FacetBucketsPayload,
            FacetsResponse,
        )

        def _payload(b: _FacetBuckets) -> FacetBucketsPayload:
            return FacetBucketsPayload(
                origins=dict(b.providers),
                tags=dict(b.tags),
                repos=dict(b.repos),
                message_types=dict(b.message_types),
                action_types=dict(b.action_types),
                has_flags=dict(b.has_flags),
                total_sessions=b.total_sessions,
                total_messages=b.total_messages,
            )

        from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore

        scoped_to_query = spec is not None and spec.has_filters()
        with ArchiveStore.open_existing(_active_archive_root(self.config)) as archive:
            global_buckets = _archive_facet_buckets(archive, None)
            scoped_buckets = _archive_facet_buckets(archive, spec) if scoped_to_query else global_buckets
        idf_map = compute_idf(global_buckets) if include_idf else {}
        active = scoped_buckets if scoped_to_query else global_buckets
        return FacetsResponse.model_validate(
            {
                "scoped_to_query": scoped_to_query,
                "origins": dict(active.providers),
                "tags": dict(active.tags),
                "repos": dict(active.repos),
                "message_types": dict(active.message_types),
                "action_types": dict(active.action_types),
                "has_flags": dict(active.has_flags),
                "total_sessions": active.total_sessions,
                "total_messages": active.total_messages,
                "scoped": _payload(scoped_buckets),
                "global": _payload(global_buckets),
                "idf": idf_map,
            }
        )

    async def health_check(self) -> ReadinessReport:
        """Return the canonical archive readiness report."""
        return _archive_health_report(self.config)

    async def rebuild_insights(
        self,
        session_ids: Sequence[str] | None = None,
        *,
        progress_callback: ProgressCallback | None = None,
    ) -> SessionInsightCounts:
        """Rebuild durable session-insight read models.

        When ``progress_callback`` is supplied, the full-rebuild DELETE phase
        emits a per-table heartbeat (#1607 parity) so a long rebuild shows
        forward motion instead of hanging silently.
        """
        from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore

        with ArchiveStore.open_existing(_active_archive_root(self.config), read_only=False) as archive:
            return _rebuild_archive_session_insights(
                archive,
                session_ids=session_ids,
                progress_callback=progress_callback,
            )

    async def resume_brief(
        self,
        session_id: str,
        *,
        related_limit: int = 6,
    ) -> ResumeBrief | None:
        """Build a compact handoff brief for an archived session."""
        from polylogue.insights.resume import ResumeOperations, build_resume_brief

        return await build_resume_brief(cast(ResumeOperations, self), session_id, related_limit=related_limit)

    async def find_resume_candidates(
        self, *, repo_path: str, cwd: str | None = None, recent_files: Sequence[str] = (), limit: int = 10
    ) -> tuple[ResumeCandidate, ...]:
        from polylogue.insights.resume import ResumeOperations, find_resume_candidates

        return await find_resume_candidates(
            cast(ResumeOperations, self),
            repo_path=repo_path,
            cwd=cwd,
            recent_files=recent_files,
            limit=limit,
        )

    async def insight_readiness_report(
        self,
        query: InsightReadinessQuery | None = None,
    ) -> InsightReadinessReport:
        """Return insight materialization readiness for downstream consumers."""
        from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore

        with ArchiveStore.open_existing(_active_archive_root(self.config)) as archive:
            return archive.insight_readiness_report(query)

    async def insight_rigor_audit(
        self,
        query: InsightRigorAuditQuery | None = None,
    ) -> InsightRigorAuditReport:
        """Per-product rigor profile across materialized insights (#1275)."""
        from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore

        with ArchiveStore.open_existing(_active_archive_root(self.config)) as archive:
            return archive.audit_insight_rigor(query)

    async def get_messages_paginated(
        self,
        session_id: str,
        *,
        message_role: MessageRoleFilter = (),
        message_type: MessageTypeName | None = None,
        limit: int = 50,
        offset: int = 0,
        content_projection: ContentProjectionSpec | None = None,
    ) -> tuple[list[Message], int]:
        """Return paginated ``Message`` objects for a session.

        Raises ``SessionNotFoundError`` if the session does not exist.
        """
        session = await self.get_session(session_id, content_projection=content_projection)
        if session is None:
            raise SessionNotFoundError(session_id)
        messages = [
            message
            for message in session.messages
            if _archive_message_matches(message, message_role=message_role, message_type=message_type)
        ]
        return messages[offset : offset + limit], len(messages)

    async def bulk_get_messages(
        self,
        session_ids: Sequence[str],
        *,
        since: str | None = None,
        until: str | None = None,
        message_role: MessageRoleFilter = (),
        content_projection: ContentProjectionSpec | None = None,
    ) -> dict[str, list[Message]]:
        """Return messages for many sessions using one archive batch read."""
        since_ms = _archive_query_date_ms("since", since)
        until_ms = _archive_query_date_ms("until", until)
        rows: dict[str, list[Message]] = {}
        for session_id in session_ids:
            session = await self.get_session(session_id, content_projection=content_projection)
            if session is None:
                continue
            rows[str(session.id)] = [
                message
                for message in session.messages
                if _archive_message_matches(
                    message,
                    message_role=message_role,
                    message_type=None,
                    since_ms=since_ms,
                    until_ms=until_ms,
                )
            ]
        return rows

    async def get_raw_artifacts_for_session(
        self,
        session_id: str,
        *,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[list[dict[str, object]], int]:
        """Return paginated raw archive artifact rows for a session.

        Delegates to the archive layer rather than accessing
        the private ``_backend`` connection directly.
        """
        from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore

        with ArchiveStore.open_existing(_active_archive_root(self.config)) as archive:
            return archive.raw_artifacts_for_session(session_id, limit=limit, offset=offset)

    async def query_sessions(
        self,
        *,
        origin: str | None = None,
        tag: str | None = None,
        since: str | None = None,
        until: str | None = None,
        sort: str | None = None,
        limit: int | None = None,
        offset: int = 0,
        has_tool_use: bool = False,
        has_thinking: bool = False,
        has_paste: bool = False,
        typed_only: bool = False,
        min_messages: int | None = None,
        max_messages: int | None = None,
        min_words: int | None = None,
        **kwargs: object,
    ) -> builtins.list[dict[str, object]]:
        """Query sessions with full filter support.

        Returns lightweight dicts suitable for the web reader and daemon API.
        For full ``Session`` objects use ``list_sessions``.
        """
        from polylogue.archive.query.spec import SessionQuerySpec

        spec = SessionQuerySpec.from_params(
            {
                "origin": origin,
                "tag": tag,
                "since": since,
                "until": until,
                "sort": sort,
                "limit": limit,
                "offset": offset,
                "filter_has_tool_use": has_tool_use,
                "filter_has_thinking": has_thinking,
                "filter_has_paste": has_paste,
                "typed_only": typed_only,
                "min_messages": min_messages,
                "max_messages": max_messages,
                "min_words": min_words,
                **kwargs,
            },
            strict=True,
        )
        from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore

        with ArchiveStore.open_existing(_active_archive_root(self.config)) as archive:
            archive_summaries = _archive_list_summaries_for_spec(archive, spec, default_limit=50)
        return [
            {
                "id": summary.session_id,
                "title": summary.title,
                "origin": summary.origin,
                "created_at": _parse_archive_datetime(summary.created_at),
                "updated_at": _parse_archive_datetime(summary.updated_at),
                "message_count": summary.message_count,
                "word_count": summary.word_count,
            }
            for summary in archive_summaries
        ]

    async def count_sessions(
        self,
        *,
        origin: str | None = None,
        since: str | None = None,
        until: str | None = None,
        **kwargs: object,
    ) -> int:
        """Count sessions matching the given filters."""
        from polylogue.archive.query.spec import SessionQuerySpec

        query_params = dict(kwargs)
        if "has_tool_use" in query_params and "filter_has_tool_use" not in query_params:
            query_params["filter_has_tool_use"] = query_params.pop("has_tool_use")
        if "has_thinking" in query_params and "filter_has_thinking" not in query_params:
            query_params["filter_has_thinking"] = query_params.pop("has_thinking")
        if "has_paste" in query_params and "filter_has_paste" not in query_params:
            query_params["filter_has_paste"] = query_params.pop("has_paste")
        spec = SessionQuerySpec.from_params(
            {"origin": origin, "since": since, "until": until, **query_params},
            strict=True,
        )
        from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore

        with ArchiveStore.open_existing(_active_archive_root(self.config)) as archive:
            return _archive_count_sessions_for_spec(archive, spec)

    async def export_insight_bundle(
        self,
        request: InsightExportBundleRequest,
    ) -> InsightExportBundleResult:
        """Write a versioned archive-insight export bundle."""
        from polylogue.insights.export_bundles import export_insight_bundle
        from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore

        with ArchiveStore.open_existing(_active_archive_root(self.config)) as archive:
            return await export_insight_bundle(_ArchiveInsightExportOperations(archive), self.config, request)

    async def get_session_summary(self, session_id: str) -> SessionSummary | None:
        """Return a summary record for a single session, or ``None`` if not found."""
        from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore

        with ArchiveStore.open_existing(_active_archive_root(self.config)) as archive:
            try:
                resolved_id = archive.resolve_session_id(session_id)
                return _archive_summary_to_domain(archive.read_summary(resolved_id))
            except KeyError:
                return None

    async def get_session_stats(self, session_id: str) -> dict[str, int]:
        """Return message-count and word-count stats for a single session."""
        from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore

        with ArchiveStore.open_existing(_active_archive_root(self.config)) as archive:
            try:
                resolved_id = archive.resolve_session_id(session_id)
                summary = archive.read_summary(resolved_id)
            except KeyError:
                return {}
            return {
                "messages": summary.message_count,
                "words": summary.word_count,
                "attachments": 0,
            }

    async def get_stats_by(self, group_by: str = "origin") -> dict[str, int]:
        """Group session counts by origin/calendar dimensions."""
        from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore

        with ArchiveStore.open_existing(_active_archive_root(self.config)) as archive:
            return archive.stats_by(group_by)

    async def get_index_status(self) -> IndexStatus:
        """Return archive block-FTS index existence and document count."""
        from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore

        with ArchiveStore.open_existing(_active_archive_root(self.config)) as archive:
            return archive.index_status()

    async def update_index(self, session_ids: list[str]) -> bool:
        """Repair the archive block-FTS index.

        The archive FTS index is trigger-maintained, so it stays in sync on every
        write and there is no per-session update primitive. This exposes the
        operator repair path: a full archive rebuild that reconciles the index
        with ``index.db`` blocks. ``session_ids`` is accepted for surface
        symmetry but the archive rebuild always reconciles the whole index.
        """
        del session_ids
        from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore

        with ArchiveStore.open_existing(_active_archive_root(self.config), read_only=False) as archive:
            archive.rebuild_index()
        return True

    async def neighbor_candidates(
        self,
        *,
        session_id: str | None = None,
        query: str | None = None,
        provider: str | None = None,
        limit: int = 10,
        window_hours: int = 24,
    ) -> list[SessionNeighborCandidate]:
        """Discover explainable neighboring or near-duplicate candidates.

        At least one of ``session_id`` or ``query`` must be provided.
        """
        from polylogue.archive.session.neighbor_candidates import (
            NeighborDiscoveryRequest,
            discover_neighbor_candidates,
        )
        from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore

        with ArchiveStore.open_existing(_active_archive_root(self.config)) as archive:
            return await discover_neighbor_candidates(
                cast("SessionQueryRuntimeStore", _ArchiveNeighborRuntime(archive)),
                NeighborDiscoveryRequest(
                    session_id=session_id,
                    query=query,
                    provider=provider,
                    limit=limit,
                    window_hours=window_hours,
                ),
            )

    async def get_session_tree(self, session_id: str) -> list[Session]:
        """Return the full session tree (parent + children) for a session."""
        from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore

        with ArchiveStore.open_existing(_active_archive_root(self.config)) as archive:
            return [_archive_session_to_session(session) for session in archive.get_session_tree(session_id)]

    async def list_tags(self, *, origin: str | None = None) -> dict[str, int]:
        """List all tags with session counts, optionally filtered by origin."""
        from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore

        with ArchiveStore.open_existing(_active_archive_root(self.config)) as archive:
            return archive.list_user_tags(origin=origin)

    async def delete_session(self, session_id: str) -> bool:
        """Permanently delete a session and all associated data.

        Returns ``True`` if something was deleted, ``False`` if the session
        was not found. Routes through :meth:`ArchiveMutationsMixin
        .delete_session_safe` so resolution and idempotency stay
        centralized (#862).
        """
        result = await self.delete_session_safe(session_id)
        return result.outcome == "deleted"

    async def delete_session_safe(self, session_id: str) -> DeleteSessionResult:
        """Typed delete that returns ``outcome="deleted"`` or ``"not_found"``."""
        from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
        from polylogue.surfaces.payloads import DeleteSessionResult

        with ArchiveStore.open_existing(_active_archive_root(self.config), read_only=False) as archive:
            try:
                resolved = archive.resolve_session_id(session_id)
                deleted = archive.delete_sessions((resolved,))
            except KeyError:
                return DeleteSessionResult(
                    outcome="not_found",
                    session_id=session_id,
                    detail="session_not_found",
                )
        return DeleteSessionResult(
            outcome="deleted" if deleted else "not_found",
            session_id=resolved,
            detail=None if deleted else "session_not_found",
        )

    async def add_tag(self, session_id: str, tag: str) -> TagMutationResult:
        """Add a tag to a session.

        Returns a ``TagMutationResult`` with:
        - ``outcome="added"`` if the tag was newly added
        - ``outcome="no_op"`` if the tag was already present
        """
        from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
        from polylogue.surfaces.payloads import TagMutationResult

        with ArchiveStore.open_existing(_active_archive_root(self.config), read_only=False) as archive:
            try:
                resolved_v1 = archive.resolve_session_id(session_id)
                changed = archive.add_user_tags((resolved_v1,), (tag,))
            except KeyError:
                raise SessionNotFoundError(session_id) from None
        return TagMutationResult(
            outcome="added" if changed else "no_op",
            detail=None if changed else "already_present",
        )

    async def remove_tag(self, session_id: str, tag: str) -> TagMutationResult:
        """Remove a tag from a session.

        Returns a ``TagMutationResult`` with:
        - ``outcome="removed"`` if the tag was removed
        - ``outcome="not_present"`` if the tag was not present
        """
        from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
        from polylogue.surfaces.payloads import TagMutationResult

        with ArchiveStore.open_existing(_active_archive_root(self.config), read_only=False) as archive:
            try:
                resolved_v1 = archive.resolve_session_id(session_id)
                changed = archive.remove_user_tags((resolved_v1,), (tag,))
            except KeyError:
                raise SessionNotFoundError(session_id) from None
        return TagMutationResult(
            outcome="removed" if changed else "not_present",
            detail=None if changed else "tag_not_present",
        )

    async def get_metadata(self, session_id: str) -> dict[str, str]:
        """Return all metadata key-value pairs for a session."""
        from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore

        with ArchiveStore.open_existing(_active_archive_root(self.config)) as archive:
            try:
                doc = archive.read_user_metadata(session_id)
            except KeyError:
                return {}
        return {str(k): str(v) if not isinstance(v, str) else v for k, v in doc.items()}

    async def update_metadata(self, session_id: str, key: str, value: str) -> bool:
        """Set a metadata key on a session.

        Returns ``True`` if the value was changed, ``False`` if it was already set
        to the same value. This is the boolean wrapper over
        :meth:`set_metadata`, so it follows the active archive backend.
        """
        result = await self.set_metadata(session_id, key, value)
        return result.outcome == "set"

    async def set_metadata(self, session_id: str, key: str, value: object) -> MetadataMutationResult:
        """Typed metadata-set returning ``outcome="set"`` or ``"unchanged"``.

        Follows the centralized mutation contract (#862): the key is
        validated before any store call (raising
        :class:`~polylogue.surfaces.payloads.MetadataKeyValidationError`),
        and the ``unchanged`` detail token is the shared ``value_unchanged``.
        """
        from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
        from polylogue.surfaces.payloads import (
            MetadataKeyValidationError,
            MetadataMutationResult,
            validate_metadata_key,
        )

        validation_error = validate_metadata_key(key)
        if validation_error is not None:
            raise MetadataKeyValidationError(validation_error)

        with ArchiveStore.open_existing(_active_archive_root(self.config), read_only=False) as archive:
            try:
                resolved = archive.resolve_session_id(session_id)
                changed = archive.set_user_metadata((resolved,), ((key, value),))
            except KeyError:
                raise SessionNotFoundError(session_id) from None
        return MetadataMutationResult(
            outcome="set" if changed else "unchanged",
            session_id=resolved,
            key=key,
            detail=None if changed else "value_unchanged",
        )

    async def delete_metadata(self, session_id: str, key: str) -> MetadataMutationResult:
        """Typed metadata-delete returning ``outcome="deleted"`` or ``"not_found"``.

        Follows the centralized mutation contract (#862): the key is
        validated before any store call, and the missing-key detail token is
        the shared ``key_not_found``.
        """
        from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
        from polylogue.surfaces.payloads import (
            MetadataKeyValidationError,
            MetadataMutationResult,
            validate_metadata_key,
        )

        validation_error = validate_metadata_key(key)
        if validation_error is not None:
            raise MetadataKeyValidationError(validation_error)

        with ArchiveStore.open_existing(_active_archive_root(self.config), read_only=False) as archive:
            try:
                resolved = archive.resolve_session_id(session_id)
                changed = archive.delete_user_metadata(resolved, key)
            except KeyError:
                raise SessionNotFoundError(session_id) from None
        return MetadataMutationResult(
            outcome="deleted" if changed else "not_found",
            session_id=resolved,
            key=key,
            detail=None if changed else "key_not_found",
        )

    async def bulk_tag_sessions(self, session_ids: list[str], tags: list[str]) -> BulkTagMutationResult:
        """Apply a bulk-tag operation across many sessions (#862).

        Validation (empty inputs and size limits) is enforced inside the
        :class:`ArchiveMutationsMixin` so every surface sees the same
        behavior.
        """
        from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
        from polylogue.surfaces.payloads import BulkTagMutationResult

        if not session_ids:
            raise ValueError("bulk_tag_sessions requires at least one session_id")
        if not tags:
            raise ValueError("bulk_tag_sessions requires at least one tag")
        max_sessions = 100
        max_tags = 20
        if len(session_ids) > max_sessions:
            raise ValueError(f"bulk_tag_sessions supports at most {max_sessions} session_ids")
        if len(tags) > max_tags:
            raise ValueError(f"bulk_tag_sessions supports at most {max_tags} tags")
        affected_count = 0
        with ArchiveStore.open_existing(_active_archive_root(self.config), read_only=False) as archive:
            for session_id in session_ids:
                try:
                    resolved = archive.resolve_session_id(session_id)
                except KeyError:
                    continue
                if archive.add_user_tags((resolved,), tuple(tags)) > 0:
                    affected_count += 1
        return BulkTagMutationResult(
            session_count=len(session_ids),
            tag_count=len(tags),
            affected_count=affected_count,
            skipped_count=len(session_ids) - affected_count,
        )

    # ------------------------------------------------------------------
    # Marks
    # ------------------------------------------------------------------

    async def _resolve_user_state_target(
        self,
        session_id: str,
        *,
        target_type: str = TARGET_SESSION,
        target_id: str | None = None,
        message_id: str | None = None,
    ) -> dict[str, str | None]:
        from polylogue.api.user_state_resolver import resolve_insight_target
        from polylogue.core.user_state_targets import validate_target_kind

        resolved_session_id = await self._resolve_user_state_session_id(session_id)
        if target_type == TARGET_SESSION:
            if target_id:
                resolved_target_id = await self._resolve_user_state_session_id(target_id)
                if resolved_target_id != resolved_session_id:
                    raise ValueError("session target_id must match session_id")
            return {
                "target_type": TARGET_SESSION,
                "target_id": resolved_session_id,
                "session_id": resolved_session_id,
                "message_id": None,
            }
        if target_type == TARGET_MESSAGE:
            if target_id and message_id and target_id != message_id:
                raise ValueError("message target_id must match message_id")
            resolved_message_id = message_id or target_id
            if not resolved_message_id:
                raise ValueError("message target requires message_id or target_id")
            if not await self._user_state_message_exists(resolved_session_id, resolved_message_id):
                raise ValueError(f"message {resolved_message_id!r} is not in session {resolved_session_id!r}")
            return {
                "target_type": TARGET_MESSAGE,
                "target_id": resolved_message_id,
                "session_id": resolved_session_id,
                "message_id": resolved_message_id,
            }
        validate_target_kind(target_type)
        resolved_target = await resolve_insight_target(
            _active_archive_root(self.config),
            target_type=target_type,
            target_id=target_id,
            session_id=resolved_session_id,
            message_id=message_id,
        )
        # Strip the identity_key — the storage layer doesn't carry it,
        # the recall-pack/workspace resolver re-derives it.
        return {
            "target_type": resolved_target["target_type"],
            "target_id": resolved_target["target_id"],
            "session_id": resolved_target["session_id"],
            "message_id": resolved_target.get("message_id"),
        }

    async def _resolve_user_state_session_id(self, session_id: str) -> str:
        archive_resolved = await self._archive_resolve_session_id(session_id)
        if archive_resolved is None:
            raise SessionNotFoundError(session_id)
        return archive_resolved

    async def _user_state_message_exists(self, session_id: str, message_id: str) -> bool:
        return bool(await self._archive_message_exists(session_id, message_id))

    async def _archive_resolve_session_id(self, token: str) -> str | None:
        from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore

        archive_db = _active_archive_root(self.config) / "index.db"
        if not archive_db.exists():
            return None
        try:
            with ArchiveStore.open_existing(_active_archive_root(self.config)) as archive:
                return archive.resolve_session_id(token)
        except KeyError:
            raise SessionNotFoundError(token) from None
        except ValueError:
            raise
        except Exception:
            return None

    async def _archive_message_exists(self, session_id: str, message_id: str) -> bool | None:
        import sqlite3

        archive_db = _active_archive_root(self.config) / "index.db"
        if not archive_db.exists():
            return None
        try:
            with sqlite3.connect(f"file:{archive_db}?mode=ro", uri=True) as conn:
                row = conn.execute(
                    "SELECT 1 FROM messages WHERE session_id = ? AND message_id = ?",
                    (session_id, message_id),
                ).fetchone()
            return row is not None
        except sqlite3.Error:
            return None

    async def add_mark(
        self,
        session_id: str,
        mark_type: str,
        *,
        target_type: str = TARGET_SESSION,
        target_id: str | None = None,
        message_id: str | None = None,
    ) -> bool:
        """Add a mark (star/pin/archive) to a session or message.

        Returns ``True`` if the mark was newly added, ``False`` if it already
        existed.
        """
        from polylogue.core.user_state_targets import validate_mark_type

        mark_type = validate_mark_type(mark_type)
        target = await self._resolve_user_state_target(
            session_id,
            target_type=target_type,
            target_id=target_id,
            message_id=message_id,
        )
        from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore

        with ArchiveStore.open_existing(_active_archive_root(self.config), read_only=False) as archive:
            return archive.add_mark(str(target["target_type"]), str(target["target_id"]), mark_type)

    async def remove_mark(
        self,
        session_id: str,
        mark_type: str,
        *,
        target_type: str = TARGET_SESSION,
        target_id: str | None = None,
        message_id: str | None = None,
    ) -> bool:
        """Remove a mark from a session or message. Returns ``True`` if removed."""
        from polylogue.core.user_state_targets import validate_mark_type

        mark_type = validate_mark_type(mark_type)
        target = await self._resolve_user_state_target(
            session_id,
            target_type=target_type,
            target_id=target_id,
            message_id=message_id,
        )
        from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore

        with ArchiveStore.open_existing(_active_archive_root(self.config), read_only=False) as archive:
            return archive.remove_mark(str(target["target_type"]), str(target["target_id"]), mark_type)

    async def list_marks(
        self,
        *,
        mark_type: str | None = None,
        session_id: str | None = None,
        target_type: str | None = None,
        target_id: str | None = None,
        message_id: str | None = None,
    ) -> list[dict[str, str]]:
        """List marks, optionally filtered by type, target, session, or message."""
        from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore

        resolved_target_type = target_type
        resolved_target_id = target_id
        if message_id is not None:
            resolved_target_type = TARGET_MESSAGE
            resolved_target_id = message_id
        elif session_id is not None and target_id is None:
            try:
                resolved_target_id = await self._resolve_user_state_session_id(session_id)
                resolved_target_type = TARGET_SESSION
            except SessionNotFoundError:
                return []
        with ArchiveStore.open_existing(_active_archive_root(self.config)) as archive:
            return archive.list_marks(
                mark_type=mark_type,
                target_type=resolved_target_type,
                target_id=resolved_target_id,
            )

    async def save_annotation(
        self,
        annotation_id: str,
        session_id: str,
        note_text: str,
        *,
        target_type: str = TARGET_SESSION,
        target_id: str | None = None,
        message_id: str | None = None,
    ) -> bool:
        """Create or update an annotation. Returns ``True`` if newly created."""
        if not annotation_id.strip():
            raise ValueError("annotation_id must not be empty")
        if not note_text.strip():
            raise ValueError("note_text must not be empty")
        target = await self._resolve_user_state_target(
            session_id,
            target_type=target_type,
            target_id=target_id,
            message_id=message_id,
        )
        from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore

        with ArchiveStore.open_existing(_active_archive_root(self.config), read_only=False) as archive:
            return archive.save_annotation(
                annotation_id,
                str(target["target_type"]),
                str(target["target_id"]),
                note_text,
            )

    async def get_annotation(self, annotation_id: str) -> dict[str, str] | None:
        """Get an annotation by ID."""
        from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore

        with ArchiveStore.open_existing(_active_archive_root(self.config)) as archive:
            return archive.get_annotation(annotation_id)

    async def list_annotations(
        self,
        *,
        session_id: str | None = None,
        target_type: str | None = None,
        target_id: str | None = None,
        message_id: str | None = None,
    ) -> list[dict[str, str]]:
        """List annotations, optionally filtered by target, session, or message."""
        from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore

        resolved_target_type = target_type
        resolved_target_id = target_id
        scope_session_id: str | None = None
        if message_id is not None:
            resolved_target_type = "message"
            resolved_target_id = message_id
        elif session_id is not None and target_id is None:
            try:
                scope_session_id = await self._resolve_user_state_session_id(session_id)
            except SessionNotFoundError:
                return []
        with ArchiveStore.open_existing(_active_archive_root(self.config)) as archive:
            return archive.list_annotations(
                target_type=resolved_target_type,
                target_id=resolved_target_id,
                session_id=scope_session_id,
            )

    async def delete_annotation(self, annotation_id: str) -> bool:
        """Delete an annotation. Returns ``True`` if deleted."""
        from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore

        with ArchiveStore.open_existing(_active_archive_root(self.config), read_only=False) as archive:
            return archive.delete_annotation(annotation_id)

    # ------------------------------------------------------------------
    # Saved views
    # ------------------------------------------------------------------

    async def save_view(self, view_id: str, name: str, query_json: str) -> bool:
        """Save a named query view. Returns ``True`` if newly created."""
        from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore

        with ArchiveStore.open_existing(_active_archive_root(self.config), read_only=False) as archive:
            return archive.save_view(view_id, name, query_json)

    async def get_view(self, view_id: str) -> dict[str, str] | None:
        """Get a saved view by ID."""
        from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore

        with ArchiveStore.open_existing(_active_archive_root(self.config)) as archive:
            return archive.get_view(view_id)

    async def get_view_by_name(self, name: str) -> dict[str, str] | None:
        """Get a saved view by name."""
        from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore

        with ArchiveStore.open_existing(_active_archive_root(self.config)) as archive:
            return archive.get_view_by_name(name)

    async def list_views(self) -> list[dict[str, str]]:
        """List all saved views."""
        from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore

        with ArchiveStore.open_existing(_active_archive_root(self.config)) as archive:
            return archive.list_views()

    async def delete_view(self, view_id: str) -> bool:
        """Delete a saved view. Returns ``True`` if deleted."""
        from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore

        with ArchiveStore.open_existing(_active_archive_root(self.config), read_only=False) as archive:
            return archive.delete_view(view_id)

    # ------------------------------------------------------------------
    # Recall packs
    # ------------------------------------------------------------------

    async def _resolve_recall_pack_item(self, item: dict[str, object]) -> dict[str, object]:
        item_type = str(item.get("target_type") or item.get("type") or "session")
        if item_type == "session":
            session_id = str(item.get("session_id") or item.get("target_id") or item.get("id") or "")
            try:
                resolved_id = await self._resolve_user_state_session_id(session_id) if session_id else None
            except SessionNotFoundError:
                resolved_id = None
            if resolved_id is None:
                return {
                    "target_type": "session",
                    "target_id": session_id,
                    "session_id": session_id or None,
                    "status": "missing",
                    "disabled_reason": "session_not_found",
                }
            return {
                "target_type": "session",
                "target_id": resolved_id,
                "session_id": resolved_id,
                "status": "resolved",
                "identity_key": f"session:{resolved_id}",
            }

        if item_type == "message":
            session_id = str(item.get("session_id") or "")
            message_id = str(item.get("message_id") or item.get("target_id") or item.get("id") or "")
            try:
                target = await self._resolve_user_state_target(
                    session_id,
                    target_type="message",
                    message_id=message_id,
                )
            except (SessionNotFoundError, ValueError) as exc:
                return {
                    "target_type": "message",
                    "target_id": message_id,
                    "session_id": session_id or None,
                    "message_id": message_id or None,
                    "status": "missing",
                    "disabled_reason": str(exc) or "message_not_found",
                }
            session_target_id = str(target["session_id"])
            resolved_message_id = str(target["message_id"])
            return {
                "target_type": "message",
                "target_id": resolved_message_id,
                "session_id": session_target_id,
                "message_id": resolved_message_id,
                "status": "resolved",
                "identity_key": f"message:{session_target_id}:{resolved_message_id}",
            }

        if item_type == "annotation":
            annotation_id = str(item.get("annotation_id") or item.get("target_id") or item.get("id") or "")
            row = await self.get_annotation(annotation_id) if annotation_id else None
            if row is None:
                return {
                    "target_type": "annotation",
                    "target_id": annotation_id,
                    "annotation_id": annotation_id or None,
                    "status": "missing",
                    "disabled_reason": "annotation_not_found",
                }
            return {
                "target_type": "annotation",
                "target_id": row["annotation_id"],
                "annotation_id": row["annotation_id"],
                "session_id": row["session_id"],
                "message_id": row["message_id"] or None,
                "annotated_target_type": row["target_type"],
                "annotated_target_id": row["target_id"],
                "note_text": row["note_text"],
                "status": "resolved",
                "identity_key": f"annotation:{row['annotation_id']}",
            }

        if item_type == "mark":
            mark_type = str(item.get("mark_type") or "")
            mark_target_type = str(item.get("mark_target_type") or item.get("target_ref_type") or "session")
            mark_target_id = str(item.get("mark_target_id") or item.get("target_id") or item.get("id") or "")
            session_id = str(item.get("session_id") or "")
            mark_message_id: str | None = str(item.get("message_id") or "") or None
            if not mark_type:
                return {
                    "target_type": "mark",
                    "target_id": mark_target_id,
                    "session_id": session_id or None,
                    "message_id": mark_message_id,
                    "status": "missing",
                    "disabled_reason": "mark_type_missing",
                }
            rows = await self.list_marks(
                mark_type=mark_type,
                session_id=session_id or None,
                target_type=mark_target_type,
                target_id=mark_target_id or None,
                message_id=mark_message_id,
            )
            if not rows:
                return {
                    "target_type": "mark",
                    "target_id": f"{mark_target_type}:{mark_target_id}:{mark_type}",
                    "session_id": session_id or None,
                    "message_id": mark_message_id,
                    "mark_type": mark_type,
                    "mark_target_type": mark_target_type,
                    "mark_target_id": mark_target_id,
                    "status": "missing",
                    "disabled_reason": "mark_not_found",
                }
            row = rows[0]
            return {
                "target_type": "mark",
                "target_id": f"{row['target_type']}:{row['target_id']}:{row['mark_type']}",
                "session_id": row["session_id"],
                "message_id": row["message_id"] or None,
                "mark_type": row["mark_type"],
                "mark_target_type": row["target_type"],
                "mark_target_id": row["target_id"],
                "status": "resolved",
                "identity_key": f"mark:{row['target_type']}:{row['target_id']}:{row['mark_type']}",
            }

        from polylogue.core.user_state_targets import TARGET_KIND_NAMES

        if item_type in TARGET_KIND_NAMES:
            return await self._resolve_recall_pack_insight_item(item, item_type)

        return {
            "target_type": item_type,
            "target_id": str(item.get("target_id") or item.get("id") or ""),
            "status": "unsupported",
            "disabled_reason": "unsupported_target_type",
        }

    async def _resolve_recall_pack_insight_item(
        self,
        item: dict[str, object],
        item_type: str,
    ) -> dict[str, object]:
        """Resolve a recall-pack item for a non-session/message kind (#1113)."""

        session_id = str(item.get("session_id") or "")
        target_id = str(item.get("target_id") or item.get("id") or "")
        message_id_raw = item.get("message_id")
        message_id: str | None = str(message_id_raw) if message_id_raw else None

        # session targets default target_id to the session_id when omitted.
        if item_type == "session" and not target_id and session_id:
            target_id = session_id

        if not session_id:
            return {
                "target_type": item_type,
                "target_id": target_id,
                "session_id": None,
                "message_id": message_id,
                "status": "missing",
                "disabled_reason": "session_id_required",
            }
        try:
            resolved = await self._resolve_user_state_target(
                session_id,
                target_type=item_type,
                target_id=target_id or None,
                message_id=message_id,
            )
        except (SessionNotFoundError, ValueError) as exc:
            return {
                "target_type": item_type,
                "target_id": target_id,
                "session_id": session_id or None,
                "message_id": message_id,
                "status": "missing",
                "disabled_reason": str(exc) or f"{item_type}_not_found",
            }
        from polylogue.core.user_state_targets import identity_key

        resolved_target_id = str(resolved["target_id"])
        resolved_session_id = str(resolved["session_id"])
        resolved_message_id_raw = resolved.get("message_id")
        resolved_message_id: str | None = str(resolved_message_id_raw) if resolved_message_id_raw else None
        return {
            "target_type": item_type,
            "target_id": resolved_target_id,
            "session_id": resolved_session_id,
            "message_id": resolved_message_id,
            "status": "resolved",
            "identity_key": identity_key(
                item_type,
                session_id=resolved_session_id,
                target_id=resolved_target_id,
                message_id=resolved_message_id,
            ),
        }

    async def _build_recall_pack_payload(
        self,
        *,
        label: str,
        payload: dict[str, object],
    ) -> tuple[list[str], str]:
        explicit_items = payload.get("items")
        if not isinstance(explicit_items, list) or not all(isinstance(item, dict) for item in explicit_items):
            raise ValueError("recall pack payload must include an items list of objects")
        raw_items = list(explicit_items)

        items = [await self._resolve_recall_pack_item(item) for item in raw_items]
        resolved_session_ids: list[str] = []
        for item in items:
            session_id = item.get("session_id")
            if (
                item.get("status") == "resolved"
                and isinstance(session_id, str)
                and session_id not in resolved_session_ids
            ):
                resolved_session_ids.append(session_id)

        normalized_payload = {
            "schema_version": 1,
            "label": label,
            "summary": payload.get("summary") or payload.get("reason") or "",
            "items": items,
            "resolved_count": sum(1 for item in items if item.get("status") == "resolved"),
            "degraded_count": sum(1 for item in items if item.get("status") != "resolved"),
        }
        for key, value in payload.items():
            if key not in {"items", "summary", "reason"}:
                normalized_payload[key] = value
        import json

        return resolved_session_ids, json.dumps(normalized_payload, sort_keys=True, separators=(",", ":"))

    async def create_recall_pack(self, pack_id: str, label: str, payload_json: str) -> bool:
        """Save a recall pack. Returns ``True`` if newly created."""
        import json

        payload = json.loads(payload_json)
        if not isinstance(payload, dict):
            raise ValueError("recall pack payload must be a JSON object")
        resolved_session_ids, normalized_payload_json = await self._build_recall_pack_payload(
            label=label,
            payload=payload,
        )
        session_ids_json = json.dumps(resolved_session_ids, sort_keys=True)
        from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore

        with ArchiveStore.open_existing(_active_archive_root(self.config), read_only=False) as archive:
            return archive.save_recall_pack(pack_id, label, session_ids_json, normalized_payload_json)

    async def get_recall_pack(self, pack_id: str) -> dict[str, str] | None:
        """Get a recall pack by ID."""
        from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore

        with ArchiveStore.open_existing(_active_archive_root(self.config)) as archive:
            return archive.get_recall_pack(pack_id)

    async def list_recall_packs(self) -> list[dict[str, str]]:
        """List all recall packs."""
        from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore

        with ArchiveStore.open_existing(_active_archive_root(self.config)) as archive:
            return archive.list_recall_packs()

    async def delete_recall_pack(self, pack_id: str) -> bool:
        """Delete a recall pack. Returns ``True`` if deleted."""
        from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore

        with ArchiveStore.open_existing(_active_archive_root(self.config), read_only=False) as archive:
            return archive.delete_recall_pack(pack_id)

    # ------------------------------------------------------------------
    # Reader workspaces
    # ------------------------------------------------------------------

    async def _build_workspace_targets(
        self, open_targets: Sequence[dict[str, object]]
    ) -> tuple[list[dict[str, object]], str]:
        import json

        items = [await self._resolve_recall_pack_item(item) for item in open_targets]
        return items, json.dumps(items, sort_keys=True, separators=(",", ":"))

    async def _build_workspace_active_target(self, active_target: dict[str, object]) -> str:
        import json

        if not active_target:
            return "{}"
        return json.dumps(await self._resolve_recall_pack_item(active_target), sort_keys=True, separators=(",", ":"))

    async def save_workspace(
        self,
        workspace_id: str,
        name: str,
        mode: str,
        open_targets_json: str,
        layout_json: str,
        active_target_json: str = "{}",
    ) -> bool:
        """Create or update a durable reader workspace."""
        import json

        workspace_id = workspace_id.strip()
        name = name.strip()
        mode = mode.strip()
        if not workspace_id:
            raise ValueError("workspace_id must not be empty")
        if not name:
            raise ValueError("name must not be empty")
        if mode not in {"tabs", "stack", "compare", "timeline"}:
            raise ValueError("mode must be one of: tabs, stack, compare, timeline")

        open_targets = json.loads(open_targets_json)
        if not isinstance(open_targets, list) or not all(isinstance(item, dict) for item in open_targets):
            raise ValueError("open_targets_json must encode a list of objects")
        _, normalized_targets_json = await self._build_workspace_targets(open_targets)

        layout = json.loads(layout_json)
        if not isinstance(layout, dict):
            raise ValueError("layout_json must encode an object")
        normalized_layout_json = json.dumps(layout, sort_keys=True, separators=(",", ":"))

        active_target = json.loads(active_target_json)
        if not isinstance(active_target, dict):
            raise ValueError("active_target_json must encode an object")
        normalized_active_json = await self._build_workspace_active_target(active_target)

        from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore

        with ArchiveStore.open_existing(_active_archive_root(self.config), read_only=False) as archive:
            return archive.save_workspace(
                workspace_id=workspace_id,
                name=name,
                mode=mode,
                open_targets_json=normalized_targets_json,
                layout_json=normalized_layout_json,
                active_target_json=normalized_active_json,
            )

    async def get_workspace(self, workspace_id: str) -> dict[str, str] | None:
        """Get a durable reader workspace by ID."""
        from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore

        with ArchiveStore.open_existing(_active_archive_root(self.config)) as archive:
            return archive.get_workspace(workspace_id)

    async def list_workspaces(self) -> list[dict[str, str]]:
        """List durable reader workspaces."""
        from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore

        with ArchiveStore.open_existing(_active_archive_root(self.config)) as archive:
            return archive.list_workspaces()

    async def delete_workspace(self, workspace_id: str) -> bool:
        """Delete a durable reader workspace. Returns ``True`` if deleted."""
        from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore

        with ArchiveStore.open_existing(_active_archive_root(self.config), read_only=False) as archive:
            return archive.delete_workspace(workspace_id)

    # ------------------------------------------------------------------
    # Learning corrections (#1131)
    #
    # User-recorded overrides that the insight materialization paths
    # consult after computing their base suggestion. Lives outside the
    # content-hash boundary by construction; see
    # :mod:`polylogue.insights.feedback` and
    # :mod:`polylogue.storage.insights.feedback`.
    # ------------------------------------------------------------------

    async def record_correction(
        self,
        session_id: str,
        kind: str,
        payload: dict[str, str],
        *,
        note: str | None = None,
    ) -> LearningCorrection:
        """Record a typed user correction for a session.

        Resolves the session ID first (short IDs are accepted) so
        the durable row is keyed by the canonical ID. Raises
        :class:`SessionNotFoundError` when the target session does
        not exist and
        :class:`~polylogue.insights.feedback.UnknownCorrectionKindError`
        when ``kind`` is not a recognized
        :class:`~polylogue.insights.feedback.CorrectionKind`.
        """

        normalized_payload = {str(key): str(value) for key, value in payload.items()}
        parse_correction_kind(kind)
        from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore

        with ArchiveStore.open_existing(_active_archive_root(self.config), read_only=False) as archive:
            try:
                return archive.record_correction(session_id, kind, normalized_payload, note=note)
            except KeyError as exc:
                raise SessionNotFoundError(session_id) from exc

    async def list_corrections(
        self,
        *,
        session_id: str | None = None,
        kind: str | None = None,
    ) -> list[LearningCorrection]:
        """List stored corrections, optionally filtered by session/kind."""

        if kind is not None:
            parse_correction_kind(kind)
        from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore

        with ArchiveStore.open_existing(_active_archive_root(self.config)) as archive:
            try:
                return archive.list_corrections(session_id=session_id, kind=kind)
            except KeyError as exc:
                raise SessionNotFoundError(str(session_id)) from exc

    async def delete_correction(self, session_id: str, kind: str) -> bool:
        """Delete one correction. Returns ``True`` when a row was removed."""

        parse_correction_kind(kind)
        from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore

        with ArchiveStore.open_existing(_active_archive_root(self.config), read_only=False) as archive:
            try:
                return archive.delete_correction(session_id, kind)
            except KeyError as exc:
                raise SessionNotFoundError(session_id) from exc

    async def clear_corrections(self, session_id: str) -> int:
        """Delete every correction for a session. Returns the count."""

        from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore

        with ArchiveStore.open_existing(_active_archive_root(self.config), read_only=False) as archive:
            try:
                return archive.clear_corrections(session_id)
            except KeyError as exc:
                raise SessionNotFoundError(session_id) from exc

    async def post_blackboard_note(
        self,
        *,
        kind: str,
        title: str,
        content: str,
        scope_repo: str | None = None,
        scope_session: str | None = None,
        scope_issue: int | None = None,
        scope_path: str | None = None,
        related_sessions: tuple[str, ...] = (),
        author_ref: str | None = None,
        author_kind: str = "user",
        evidence_refs: tuple[str, ...] = (),
        staleness: dict[str, object] | None = None,
        context_policy: dict[str, object] | None = None,
    ) -> BlackboardNote:
        """Post a note to the persistent agent blackboard (#1697).

        ``kind`` must be one of :data:`BLACKBOARD_KINDS`; raises ``ValueError``
        otherwise. The structured fields are encoded into the stored body and a
        fresh note id is allocated, so each call appends a distinct note. The
        optional assertion metadata fields are mirrored only into the unified
        assertion row (#1839/#1883), preserving the legacy blackboard row shape.
        """
        from polylogue.archive.blackboard import (
            BLACKBOARD_KINDS,
            build_blackboard_body,
            decode_blackboard_note,
        )
        from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore

        if kind not in BLACKBOARD_KINDS:
            raise ValueError(f"kind must be one of {list(BLACKBOARD_KINDS)}, got {kind!r}")
        body = build_blackboard_body(
            kind=kind,
            title=title,
            content=content,
            scope_repo=scope_repo,
            scope_issue=scope_issue,
            scope_path=scope_path,
            related_sessions=related_sessions,
        )
        note_id = str(uuid.uuid4())
        target_type = "session" if scope_session else None
        with ArchiveStore.open_existing(_active_archive_root(self.config), read_only=False) as archive:
            envelope = archive.post_blackboard_note(
                body,
                target_type=target_type,
                target_id=scope_session,
                note_id=note_id,
                author_ref=author_ref,
                author_kind=author_kind,
                evidence_refs=evidence_refs,
                staleness=staleness,
                context_policy=context_policy,
            )
        return decode_blackboard_note(
            note_id=envelope.note_id,
            body=envelope.body,
            target_type=envelope.target_type,
            target_id=envelope.target_id,
            created_at_ms=envelope.created_at_ms,
            updated_at_ms=envelope.updated_at_ms,
        )

    async def list_blackboard_notes(
        self,
        *,
        kind: str | None = None,
        scope_repo: str | None = None,
        unresolved: bool = False,
        limit: int = 20,
    ) -> list[BlackboardNote]:
        """List blackboard notes, newest first, with optional filters (#1697).

        ``unresolved`` narrows to open-work kinds (:data:`UNRESOLVED_KINDS`).
        Filtering runs on decoded notes, then the result is capped at ``limit``.
        """
        from polylogue.archive.blackboard import (
            UNRESOLVED_KINDS,
            decode_blackboard_note,
        )
        from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore

        with ArchiveStore.open_existing(_active_archive_root(self.config)) as archive:
            envelopes = archive.list_blackboard_notes()
        notes: list[BlackboardNote] = []
        for envelope in envelopes:
            note = decode_blackboard_note(
                note_id=envelope.note_id,
                body=envelope.body,
                target_type=envelope.target_type,
                target_id=envelope.target_id,
                created_at_ms=envelope.created_at_ms,
                updated_at_ms=envelope.updated_at_ms,
            )
            if kind is not None and note.kind != kind:
                continue
            if scope_repo is not None and note.scope_repo != scope_repo:
                continue
            if unresolved and note.kind not in UNRESOLVED_KINDS:
                continue
            notes.append(note)
            if limit > 0 and len(notes) >= limit:
                break
        return notes
