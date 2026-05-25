"""Archive operations shared across facade, CLI, and MCP call sites."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, TypeVar, cast

import structlog

from polylogue.archive.conversation.models import ConversationSummary
from polylogue.archive.query.spec import ConversationQuerySpec
from polylogue.archive.semantic.content_projection import ContentProjectionSpec, project_message_content
from polylogue.archive.semantic.pricing import (
    _normalize_model,
    estimate_conversation_cost,
    generated_at,
)
from polylogue.config import ConfigError
from polylogue.core.timestamps import parse_timestamp
from polylogue.errors import DatabaseError
from polylogue.insights.archive import (
    ArchiveCoverageInsight,
    ArchiveCoverageInsightQuery,
    ArchiveDebtInsight,
    ArchiveDebtInsightQuery,
    ArchiveInsightProvenance,
    ArchiveInsightUnavailableError,
    CostRollupInsight,
    CostRollupInsightQuery,
    DaySessionSummaryInsight,
    SessionCostInsight,
    SessionCostInsightQuery,
    SessionLatencyProfileInsight,
    SessionLatencyProfileInsightQuery,
    SessionPhaseInsight,
    SessionPhaseInsightQuery,
    SessionProfileInsight,
    SessionProfileInsightQuery,
    SessionTagRollupInsight,
    SessionTagRollupQuery,
    SessionWorkEventInsight,
    SessionWorkEventInsightQuery,
    WeekSessionSummaryInsight,
    WorkThreadInsight,
    WorkThreadInsightQuery,
)
from polylogue.insights.archive_rollups import aggregate_cost_rollup_insights, aggregate_session_tag_rollup_insights
from polylogue.insights.archive_summaries import (
    aggregate_day_session_summary_insights,
    aggregate_week_session_summary_insights,
    build_day_session_summary_records,
)
from polylogue.insights.audit import (
    InsightRigorAuditQuery,
    InsightRigorAuditReport,
    build_insight_rigor_audit_report,
)
from polylogue.insights.export_bundles import (
    InsightExportBundleRequest,
    InsightExportBundleResult,
    InsightExportOperations,
    export_insight_bundle,
)
from polylogue.insights.readiness import (
    InsightReadinessQuery,
    InsightReadinessReport,
    build_insight_readiness_report,
)
from polylogue.insights.resume import (
    ResumeBrief,
    ResumeCandidate,
    ResumeOperations,
    build_resume_brief,
    find_resume_candidates,
)
from polylogue.insights.tool_usage import (
    ToolUsageInsight,
    ToolUsageInsightQuery,
    build_tool_usage_insight,
)
from polylogue.maintenance.targets import build_maintenance_target_catalog
from polylogue.operations.completion_aggregates import ArchiveCompletionMixin, CompletionAggregate
from polylogue.operations.mutations import ArchiveMutationsMixin
from polylogue.services import RuntimeServices, build_runtime_services
from polylogue.storage.hydrators import message_from_record
from polylogue.storage.insights.session.profiles import hydrate_session_profile
from polylogue.storage.insights.session.runtime import (
    SessionInsightReadyFlag,
    SessionInsightStatusSnapshot,
)
from polylogue.storage.repair import collect_archive_debt_statuses_sync
from polylogue.storage.runtime.store_constants import SESSION_INSIGHT_MATERIALIZER_VERSION
from polylogue.storage.search import SearchHit, SearchResult
from polylogue.storage.search.query_builders import conversation_web_url
from polylogue.storage.sqlite.connection import connection_context
from polylogue.storage.sqlite.queries.stats import ProviderMetricsRow
from polylogue.types import Provider

logger = structlog.get_logger(__name__)
_MAINTENANCE_TARGET_CATALOG = build_maintenance_target_catalog()
_SESSION_INSIGHT_REPAIR_HINT = _MAINTENANCE_TARGET_CATALOG.repair_hint(("session_insights",), include_run_all=True)
_PROFILE_FTS_STATUS_BY_TIER: dict[str, SessionInsightReadyFlag] = {
    "merged": "profile_merged_fts_ready",
    "evidence": "profile_evidence_fts_ready",
    "inference": "profile_inference_fts_ready",
    "enrichment": "profile_enrichment_fts_ready",
}

if TYPE_CHECKING:
    from polylogue.archive.conversation.models import Conversation
    from polylogue.archive.conversation.neighbor_candidates import ConversationNeighborCandidate
    from polylogue.archive.message.models import Message
    from polylogue.archive.message.roles import MessageRoleFilter
    from polylogue.archive.query.miss_diagnostics import QueryMissDiagnostics
    from polylogue.archive.query.search_hits import ConversationSearchHit
    from polylogue.archive.stats import ArchiveStats as StorageArchiveStats
    from polylogue.config import Config
    from polylogue.pipeline.services.indexing import IndexStatus
    from polylogue.protocols import VectorProvider
    from polylogue.storage.archive_views import ConversationRenderProjection
    from polylogue.storage.insights.session.runtime import SessionInsightCounts
    from polylogue.storage.repository import ConversationRepository
    from polylogue.storage.runtime import RawConversationRecord
    from polylogue.storage.sqlite.async_sqlite import SQLiteBackend
    from polylogue.storage.sqlite.queries.messages import MessageTypeName

_ResultT = TypeVar("_ResultT")
_QueryT = TypeVar("_QueryT")


def _build_search_snippet(text: str, query: str) -> str:
    """Create a deterministic snippet around the earliest query-term match."""
    if not text:
        return ""

    terms = [term.lower() for term in query.split() if term.strip()]
    lowered = text.lower()
    positions = [lowered.find(term) for term in terms if lowered.find(term) >= 0]
    anchor = min(positions) if positions else 0
    start = max(0, anchor - 60)
    end = min(len(text), anchor + 140)
    snippet = text[start:end].strip()
    if start > 0:
        snippet = f"...{snippet}"
    if end < len(text):
        snippet = f"{snippet}..."
    return snippet


def _conversation_search_hit(
    conversation: Conversation,
    *,
    query: str,
) -> SearchHit:
    """Adapt a canonical conversation result into the SearchResult surface."""
    terms = [term.lower() for term in query.split() if term.strip()]
    matching_message = next(
        (msg for msg in conversation.messages if msg.text and any(term in msg.text.lower() for term in terms)),
        next((msg for msg in conversation.messages if msg.text), None),
    )
    message_id = str(matching_message.id) if matching_message else ""
    timestamp = matching_message.timestamp.isoformat() if matching_message and matching_message.timestamp else None
    snippet = _build_search_snippet(matching_message.text or "", query) if matching_message else ""
    return SearchHit(
        conversation_id=str(conversation.id),
        source_name=None,
        message_id=message_id,
        title=conversation.display_title,
        timestamp=timestamp,
        snippet=snippet,
        conversation_url=conversation_web_url(str(conversation.id)),
    )


def _row_int(row: Mapping[str, object], key: str) -> int:
    value = row.get(key, 0)
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return 0
    return 0


def _row_str(row: Mapping[str, object], key: str, *, default: str = "") -> str:
    value = row.get(key)
    return str(value) if value is not None else default


def _default_query(query: _QueryT | None, model: type[_QueryT]) -> _QueryT:
    return query if query is not None else model()


def _query_wants_search(query: object) -> bool:
    wants_search = getattr(query, "wants_search", None)
    if isinstance(wants_search, bool):
        return wants_search
    return bool(getattr(query, "query", None))


def _slice_insights(
    insights: list[_ResultT],
    *,
    offset: int,
    limit: int | None,
) -> list[_ResultT]:
    if offset:
        insights = insights[offset:]
    if limit is not None:
        insights = insights[:limit]
    return insights


def _provider_coverage_insight(row: ProviderMetricsRow) -> ArchiveCoverageInsight:
    conversation_count = row["conversation_count"]
    user_message_count = row["user_message_count"]
    assistant_message_count = row["assistant_message_count"]
    message_count = row["message_count"]
    user_word_sum = row["user_word_sum"]
    assistant_word_sum = row["assistant_word_sum"]
    tool_use_count = row["tool_use_count"]
    thinking_count = row["thinking_count"]
    conversations_with_tools = row["conversations_with_tools"]
    conversations_with_thinking = row["conversations_with_thinking"]
    tool_use_percentage = (conversations_with_tools / conversation_count) * 100 if conversation_count > 0 else 0.0
    thinking_percentage = (conversations_with_thinking / conversation_count) * 100 if conversation_count > 0 else 0.0
    source_name = row["source_name"] or "unknown"
    return ArchiveCoverageInsight(
        group_by="provider",
        bucket=source_name,
        source_name=source_name,
        conversation_count=conversation_count,
        message_count=message_count,
        user_message_count=user_message_count,
        assistant_message_count=assistant_message_count,
        avg_messages_per_conversation=(message_count / conversation_count if conversation_count > 0 else 0.0),
        avg_user_words=(user_word_sum / user_message_count if user_message_count > 0 else 0.0),
        avg_assistant_words=(assistant_word_sum / assistant_message_count if assistant_message_count > 0 else 0.0),
        tool_use_count=tool_use_count,
        thinking_count=thinking_count,
        total_conversations_with_tools=conversations_with_tools,
        total_conversations_with_thinking=conversations_with_thinking,
        tool_use_percentage=tool_use_percentage,
        thinking_percentage=thinking_percentage,
    )


def _day_coverage_insight(insight: DaySessionSummaryInsight) -> ArchiveCoverageInsight:
    summary = insight.summary
    return ArchiveCoverageInsight(
        group_by="day",
        bucket=insight.date,
        conversation_count=summary.session_count,
        logical_session_count=summary.logical_session_count,
        message_count=summary.total_messages,
        total_cost_usd=summary.total_cost_usd,
        total_duration_ms=summary.total_duration_ms,
        total_tool_active_duration_ms=summary.total_tool_active_duration_ms,
        total_wall_duration_ms=summary.total_wall_duration_ms,
        total_words=summary.total_words,
        work_event_breakdown=summary.work_event_breakdown,
        repos_active=summary.repos_active,
        provider_breakdown=summary.providers,
        provenance=insight.provenance,
    )


def _week_coverage_insight(insight: WeekSessionSummaryInsight) -> ArchiveCoverageInsight:
    summary = insight.summary
    provider_breakdown: dict[str, int] = {}
    repos_active: set[str] = set()
    total_words = 0
    total_wall_duration_ms = 0
    work_event_breakdown: dict[str, int] = {}
    for day in summary.day_summaries:
        total_words += day.total_words
        total_wall_duration_ms += day.total_wall_duration_ms
        repos_active.update(day.repos_active)
        for source_name, count in day.providers.items():
            provider_breakdown[source_name] = provider_breakdown.get(source_name, 0) + count
        for label, count in day.work_event_breakdown.items():
            work_event_breakdown[label] = work_event_breakdown.get(label, 0) + count
    return ArchiveCoverageInsight(
        group_by="week",
        bucket=insight.iso_week,
        conversation_count=summary.session_count,
        logical_session_count=summary.logical_session_count,
        message_count=summary.total_messages,
        total_cost_usd=summary.total_cost_usd,
        total_duration_ms=summary.total_duration_ms,
        total_tool_active_duration_ms=summary.total_tool_active_duration_ms,
        total_wall_duration_ms=total_wall_duration_ms,
        total_words=total_words,
        work_event_breakdown=work_event_breakdown,
        repos_active=tuple(sorted(repos_active)),
        provider_breakdown=provider_breakdown,
        provenance=insight.provenance,
    )


def _session_cost_insight(conversation: Conversation, *, materialized_at: str) -> SessionCostInsight:
    estimate = estimate_conversation_cost(conversation)
    source_updated = conversation.updated_at or conversation.created_at
    return SessionCostInsight(
        conversation_id=str(conversation.id),
        source_name=str(conversation.provider),
        title=conversation.title,
        created_at=conversation.created_at.isoformat() if conversation.created_at is not None else None,
        updated_at=conversation.updated_at.isoformat() if conversation.updated_at is not None else None,
        estimate=estimate,
        provenance=ArchiveInsightProvenance(
            materializer_version=SESSION_INSIGHT_MATERIALIZER_VERSION,
            materialized_at=materialized_at,
            source_updated_at=source_updated.isoformat() if source_updated is not None else None,
            source_sort_key=source_updated.timestamp() if source_updated is not None else None,
        ),
    )


def _cost_model_matches(insight: SessionCostInsight, model_filter: str | None) -> bool:
    if not model_filter:
        return True
    normalized_filter = _normalize_model(model_filter)
    return insight.estimate.model_name == model_filter or insight.estimate.normalized_model == normalized_filter


def _cost_status_matches(insight: SessionCostInsight, status_filter: str | None) -> bool:
    return not status_filter or insight.estimate.status == status_filter


def _require_ready_flag(
    status: SessionInsightStatusSnapshot,
    flag: SessionInsightReadyFlag,
    detail: str,
) -> None:
    if status.ready_flag(flag):
        return
    raise ArchiveInsightUnavailableError(f"{detail} {_SESSION_INSIGHT_REPAIR_HINT}")


async def _read_session_insight_status(backend: SQLiteBackend) -> SessionInsightStatusSnapshot:
    return await backend.get_session_insight_status()


class ArchiveSearchMixin:
    """Conversation retrieval and search methods for archive operations."""

    if TYPE_CHECKING:

        @property
        def repository(self) -> ConversationRepository: ...

        @property
        def config(self) -> Config: ...

        @property
        def backend(self) -> SQLiteBackend: ...

    async def _vector_provider_for_query_spec(self, spec: ConversationQuerySpec) -> VectorProvider | None:
        """Return a vector provider for explicit semantic/hybrid query specs.

        ``auto`` is intentionally not promoted here. CLI auto-elevation has
        its own UX path; archive operations only enforce that explicit
        semantic requests and explicit hybrid requests do not silently run
        without usable vectors.
        """
        if not spec.similar_text and spec.retrieval_lane != "hybrid":
            return None

        archive_stats = await self.repository.get_archive_stats()
        retrieval_ready = getattr(archive_stats, "retrieval_ready", None)
        if not isinstance(retrieval_ready, bool):
            embedded_messages = int(getattr(archive_stats, "embedded_messages", 0) or 0)
            stale_messages = int(getattr(archive_stats, "stale_embedding_messages", 0) or 0)
            retrieval_ready = max(embedded_messages - stale_messages, 0) > 0
        if not retrieval_ready:
            status = getattr(archive_stats, "embedding_readiness_status", None) or "none"
            raise DatabaseError(
                "Semantic or hybrid retrieval requires retrieval-ready embeddings "
                f"(current status: {status}). Run `polylogue embed status`, then "
                "`polylogue embed backfill` or let polylogued converge after enabling embeddings."
            )

        from polylogue.storage.search_providers import create_vector_provider

        vector_provider = create_vector_provider(self.config, db_path=self.backend.db_path)
        if vector_provider is None:
            raise DatabaseError(
                "Semantic or hybrid retrieval requires vector search support, but vector provider initialization "
                "failed or embeddings are disabled. Run `polylogue embed status`, then `polylogue embed enable` "
                "if needed."
            )
        return vector_provider

    async def get_conversation(
        self,
        conversation_id: str,
        *,
        content_projection: ContentProjectionSpec | None = None,
    ) -> Conversation | None:
        conversation = await self.repository.view(conversation_id)
        if conversation is None or content_projection is None or not content_projection.filters_content():
            return conversation
        return conversation.with_content_projection(content_projection)

    async def get_conversation_summary(self, conversation_id: str) -> ConversationSummary | None:
        full_id = await self.repository.resolve_id(conversation_id) or conversation_id
        return await self.repository.get_summary(str(full_id))

    async def get_conversation_stats(self, conversation_id: str) -> dict[str, int]:
        full_id = await self.repository.resolve_id(conversation_id) or conversation_id
        return await self.repository.get_conversation_stats(str(full_id))

    async def get_render_projection(self, conversation_id: str) -> ConversationRenderProjection | None:
        """Load the canonical render projection for a conversation."""
        return await self.repository.get_render_projection(conversation_id)

    async def get_conversations(
        self,
        conversation_ids: list[str],
        *,
        content_projection: ContentProjectionSpec | None = None,
    ) -> list[Conversation]:
        conversations = await self.repository.get_many(conversation_ids)
        if content_projection is None or not content_projection.filters_content():
            return conversations
        return [conversation.with_content_projection(content_projection) for conversation in conversations]

    async def list_conversations(
        self,
        *,
        provider: str | None = None,
        limit: int | None = None,
        content_projection: ContentProjectionSpec | None = None,
    ) -> list[Conversation]:
        conversations = await self.repository.list(provider=provider, limit=limit)
        if content_projection is None or not content_projection.filters_content():
            return conversations
        return [conversation.with_content_projection(content_projection) for conversation in conversations]

    async def query_conversations(
        self,
        spec: ConversationQuerySpec,
        *,
        content_projection: ContentProjectionSpec | None = None,
    ) -> list[Conversation]:
        vector_provider = await self._vector_provider_for_query_spec(spec)
        conversations = await spec.list(self.repository, vector_provider=vector_provider)
        if content_projection is None or not content_projection.filters_content():
            return conversations
        return [conversation.with_content_projection(content_projection) for conversation in conversations]

    async def search_conversation_hits(self, spec: ConversationQuerySpec) -> list[ConversationSearchHit]:
        from polylogue.archive.query.search_hits import search_hits_for_plan

        vector_provider = await self._vector_provider_for_query_spec(spec)
        hits = await search_hits_for_plan(spec.to_plan(vector_provider=vector_provider), self.repository)
        if not hits:
            return hits
        counts = await self.repository.get_message_counts_batch([hit.conversation_id for hit in hits])
        return [hit.with_message_count(counts.get(hit.conversation_id)) for hit in hits]

    async def count_conversations(self, spec: ConversationQuerySpec) -> int:
        vector_provider = await self._vector_provider_for_query_spec(spec)
        return await spec.count(self.repository, vector_provider=vector_provider)

    async def neighbor_candidates(
        self,
        *,
        conversation_id: str | None = None,
        query: str | None = None,
        provider: str | None = None,
        limit: int = 10,
        window_hours: int = 24,
    ) -> list[ConversationNeighborCandidate]:
        from polylogue.archive.conversation.neighbor_candidates import (
            NeighborDiscoveryRequest,
            discover_neighbor_candidates,
        )

        candidates = await discover_neighbor_candidates(
            self.repository,
            NeighborDiscoveryRequest(
                conversation_id=conversation_id,
                query=query,
                provider=provider,
                limit=limit,
                window_hours=window_hours,
            ),
        )
        if not candidates:
            return []
        counts = await self.repository.get_message_counts_batch([candidate.conversation_id for candidate in candidates])
        return [candidate.with_message_count(counts.get(candidate.conversation_id)) for candidate in candidates]

    async def diagnose_query_miss(self, spec: ConversationQuerySpec) -> QueryMissDiagnostics:
        from polylogue.archive.query.miss_diagnostics import diagnose_query_miss

        return await diagnose_query_miss(self.repository, spec, config=self.config)

    async def get_session_tree(self, conversation_id: str) -> list[Conversation]:
        return await self.repository.get_session_tree(conversation_id)

    async def get_messages_paginated(
        self,
        conversation_id: str,
        *,
        message_role: MessageRoleFilter = (),
        message_type: MessageTypeName | None = None,
        limit: int = 50,
        offset: int = 0,
        content_projection: ContentProjectionSpec | None = None,
    ) -> tuple[list[Message], int]:
        resolved = await self.repository.resolve_id(conversation_id)
        if not resolved:
            return [], 0
        messages, total = await self.repository.get_messages_paginated(
            str(resolved),
            message_role=message_role,
            message_type=message_type,
            limit=limit,
            offset=offset,
        )
        return project_message_content(messages, content_projection), total

    async def bulk_get_messages(
        self,
        conversation_ids: Sequence[str],
        *,
        since: str | None = None,
        until: str | None = None,
        message_role: MessageRoleFilter = (),
        content_projection: ContentProjectionSpec | None = None,
    ) -> dict[str, list[Message]]:
        """Return messages for multiple conversations with one batch read."""
        ids = [str(conversation_id) for conversation_id in conversation_ids]
        if not ids:
            return {}

        since_ts = parse_timestamp(since)
        until_ts = parse_timestamp(until)
        records_by_id = await self.repository.get_messages_batch(
            ids,
            sort_key_since=since_ts.timestamp() if since_ts is not None else None,
            sort_key_until=until_ts.timestamp() if until_ts is not None else None,
            message_role=message_role,
        )
        messages_by_id: dict[str, list[Message]] = {}
        for conversation_id in ids:
            messages = [
                message_from_record(record, attachments=[], provider=record.source_name)
                for record in records_by_id.get(conversation_id, [])
            ]
            messages_by_id[conversation_id] = project_message_content(messages, content_projection)
        return messages_by_id

    async def get_raw_artifacts_for_conversation(
        self,
        conversation_id: str,
        *,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[list[RawConversationRecord], int]:
        resolved = await self.repository.resolve_id(conversation_id)
        if not resolved:
            return [], 0
        return await self.repository.get_raw_records_for_conversation(
            str(resolved),
            limit=limit,
            offset=offset,
        )

    async def get_stats_by(self, group_by: str = "provider") -> dict[str, int]:
        return await self.repository.get_stats_by(group_by)

    async def list_tags(self, *, provider: str | None = None) -> dict[str, int]:
        return await self.repository.list_tags(provider=provider)

    async def search(
        self,
        query: str,
        *,
        limit: int = 100,
        source: str | None = None,
        since: str | None = None,
    ) -> SearchResult:
        spec = ConversationQuerySpec(
            query_terms=(query,),
            providers=(Provider.from_string(source),) if source else (),
            since=since,
            limit=limit,
        )
        conversations = await self.query_conversations(spec)
        return SearchResult(
            hits=[
                _conversation_search_hit(
                    conversation,
                    query=query,
                )
                for conversation in conversations
            ]
        )


class ArchiveStats:
    """Statistics about the archive for the public facade surface."""

    def __init__(
        self,
        conversation_count: int,
        message_count: int,
        word_count: int,
        providers: dict[str, int],
        tags: dict[str, int],
        last_sync: str | None,
        recent: list[Conversation],
    ) -> None:
        self.conversation_count = conversation_count
        self.message_count = message_count
        self.word_count = word_count
        self.providers = providers
        self.tags = tags
        self.last_sync = last_sync
        self.recent = recent

    def __repr__(self) -> str:
        return (
            f"ArchiveStats(conversations={self.conversation_count}, "
            f"messages={self.message_count}, providers={list(self.providers.keys())})"
        )


class ArchiveStatsMixin:
    """Archive summary, status, and provider-count helpers."""

    if TYPE_CHECKING:

        @property
        def repository(self) -> ConversationRepository: ...

        @property
        def backend(self) -> SQLiteBackend: ...

        async def list_conversations(
            self,
            *,
            provider: str | None = None,
            limit: int | None = None,
        ) -> list[Conversation]: ...

    async def storage_stats(self) -> StorageArchiveStats:
        return await self.repository.get_archive_stats()

    async def summary_stats(self) -> ArchiveStats:
        storage_snapshot = await self.storage_stats()
        aggregate_stats = await self.repository.aggregate_message_stats()
        tags = await self.repository.list_tags()
        recent = await self.list_conversations(limit=5)

        last_sync = None
        try:
            last_sync = await self.backend.get_last_sync_timestamp()
        except Exception as exc:  # pragma: no cover - defensive debug path
            logger.warning("failed to query last sync timestamp", error=str(exc), exc_info=True)

        return ArchiveStats(
            conversation_count=storage_snapshot.total_conversations,
            message_count=storage_snapshot.total_messages,
            word_count=int(aggregate_stats.get("words_approx", 0)),
            providers=storage_snapshot.providers,
            tags=tags,
            last_sync=last_sync,
            recent=recent,
        )

    async def provider_counts(self) -> list[tuple[str, int]]:
        rows = await self.backend.get_provider_conversation_counts()
        return [(row["source_name"] or "unknown", row["conversation_count"]) for row in rows]

    async def get_session_insight_status(self) -> SessionInsightStatusSnapshot:
        return await self.backend.get_session_insight_status()


class ArchiveInsightSessionMixin:
    if TYPE_CHECKING:

        @property
        def repository(self) -> ConversationRepository: ...

        @property
        def backend(self) -> SQLiteBackend: ...

    async def _session_insight_status(self) -> SessionInsightStatusSnapshot:
        return await _read_session_insight_status(self.backend)

    async def get_session_profile_insight(
        self,
        conversation_id: str,
        *,
        tier: str = "merged",
    ) -> SessionProfileInsight | None:
        status = await self._session_insight_status()
        _require_ready_flag(status, "profile_rows_ready", "Session-profile rows are incomplete.")
        record = await self.repository.get_session_profile_record(conversation_id)
        return SessionProfileInsight.from_record(record, tier=tier) if record is not None else None

    async def get_session_latency_profile_insight(
        self,
        conversation_id: str,
    ) -> SessionLatencyProfileInsight | None:
        status = await self._session_insight_status()
        _require_ready_flag(status, "latency_profile_rows_ready", "Session-latency rows are incomplete.")
        record = await self.repository.get_session_latency_profile_record(conversation_id)
        return SessionLatencyProfileInsight.from_record(record) if record is not None else None

    async def find_stuck_session_latency_profile_insights(
        self,
        query: SessionLatencyProfileInsightQuery | None = None,
    ) -> list[SessionLatencyProfileInsight]:
        request = _default_query(query, SessionLatencyProfileInsightQuery)
        status = await self._session_insight_status()
        _require_ready_flag(status, "latency_profile_rows_ready", "Session-latency rows are incomplete.")
        records = await self.repository.find_stuck_session_latency_profile_records(
            since=request.since,
            limit=request.limit or 50,
        )
        return [SessionLatencyProfileInsight.from_record(record) for record in records]

    async def list_session_latency_profile_insights(
        self,
        query: SessionLatencyProfileInsightQuery | None = None,
    ) -> list[SessionLatencyProfileInsight]:
        request = _default_query(query, SessionLatencyProfileInsightQuery)
        status = await self._session_insight_status()
        _require_ready_flag(status, "latency_profile_rows_ready", "Session-latency rows are incomplete.")
        records = await self.repository.list_session_latency_profile_records(
            provider=request.provider,
            since=request.since,
            until=request.until,
            limit=request.limit,
        )
        if request.conversation_id:
            records = [record for record in records if str(record.conversation_id) == request.conversation_id]
        if request.only_stuck:
            records = [record for record in records if record.stuck_tool_count > 0]
        return [SessionLatencyProfileInsight.from_record(record) for record in records]

    async def list_session_profile_insights(
        self,
        query: SessionProfileInsightQuery | None = None,
    ) -> list[SessionProfileInsight]:
        request = _default_query(query, SessionProfileInsightQuery)
        status = await self._session_insight_status()
        _require_ready_flag(status, "profile_rows_ready", "Session-profile rows are incomplete.")
        if _query_wants_search(request):
            _require_ready_flag(
                status,
                _PROFILE_FTS_STATUS_BY_TIER.get(request.tier, "profile_merged_fts_ready"),
                f"Session-profile {request.tier} search index is incomplete.",
            )
        records = await self.repository.list_session_profile_records(
            provider=request.provider,
            since=request.since,
            until=request.until,
            first_message_since=request.first_message_since,
            first_message_until=request.first_message_until,
            session_date_since=request.session_date_since,
            session_date_until=request.session_date_until,
            min_wallclock_seconds=request.min_wallclock_seconds,
            max_wallclock_seconds=request.max_wallclock_seconds,
            workflow_shape=request.workflow_shape,
            terminal_state=request.terminal_state,
            sort=request.sort,
            tier=request.tier,
            limit=request.limit,
            offset=request.offset,
            query=request.query,
        )
        return [SessionProfileInsight.from_record(record, tier=request.tier) for record in records]

    async def get_session_work_event_insights(
        self,
        conversation_id: str,
    ) -> list[SessionWorkEventInsight]:
        status = await self._session_insight_status()
        _require_ready_flag(status, "work_event_inference_rows_ready", "Session work-event rows are incomplete.")
        records = await self.repository.get_session_work_event_records(conversation_id)
        return [SessionWorkEventInsight.from_record(record) for record in records]

    async def list_session_work_event_insights(
        self,
        query: SessionWorkEventInsightQuery | None = None,
    ) -> list[SessionWorkEventInsight]:
        request = _default_query(query, SessionWorkEventInsightQuery)
        status = await self._session_insight_status()
        _require_ready_flag(status, "work_event_inference_rows_ready", "Session work-event rows are incomplete.")
        if _query_wants_search(request):
            _require_ready_flag(
                status,
                "work_event_inference_fts_ready",
                "Session work-event search index is incomplete.",
            )
        records = await self.repository.list_session_work_event_records(
            conversation_id=request.conversation_id,
            provider=request.provider,
            since=request.since,
            until=request.until,
            session_date_since=request.session_date_since,
            session_date_until=request.session_date_until,
            heuristic_label=request.heuristic_label,
            limit=request.limit,
            offset=request.offset,
            query=request.query,
        )
        return [SessionWorkEventInsight.from_record(record) for record in records]

    async def get_session_phase_insights(
        self,
        conversation_id: str,
    ) -> list[SessionPhaseInsight]:
        status = await self._session_insight_status()
        _require_ready_flag(status, "phase_inference_rows_ready", "Session phase rows are incomplete.")
        records = await self.repository.get_session_phase_records(conversation_id)
        return [SessionPhaseInsight.from_record(record) for record in records]

    async def list_session_phase_insights(
        self,
        query: SessionPhaseInsightQuery | None = None,
    ) -> list[SessionPhaseInsight]:
        request = _default_query(query, SessionPhaseInsightQuery)
        status = await self._session_insight_status()
        _require_ready_flag(status, "phase_inference_rows_ready", "Session phase rows are incomplete.")
        records = await self.repository.list_session_phase_records(
            conversation_id=request.conversation_id,
            provider=request.provider,
            since=request.since,
            until=request.until,
            kind=request.kind,
            limit=request.limit,
            offset=request.offset,
        )
        return [SessionPhaseInsight.from_record(record) for record in records]

    async def get_work_thread_insight(self, thread_id: str) -> WorkThreadInsight | None:
        status = await self._session_insight_status()
        _require_ready_flag(status, "threads_ready", "Work-thread rows are incomplete.")
        record = await self.repository.get_work_thread_record(thread_id)
        return WorkThreadInsight.from_record(record) if record is not None else None

    async def list_work_thread_insights(
        self,
        query: WorkThreadInsightQuery | None = None,
    ) -> list[WorkThreadInsight]:
        request = _default_query(query, WorkThreadInsightQuery)
        status = await self._session_insight_status()
        _require_ready_flag(status, "threads_ready", "Work-thread rows are incomplete.")
        if _query_wants_search(request):
            _require_ready_flag(status, "threads_fts_ready", "Work-thread search index is incomplete.")
        records = await self.repository.list_work_thread_records(
            since=request.since,
            until=request.until,
            limit=request.limit,
            offset=request.offset,
            query=request.query,
        )
        return [WorkThreadInsight.from_record(record) for record in records]


class ArchiveInsightAggregateMixin:
    if TYPE_CHECKING:

        @property
        def repository(self) -> ConversationRepository: ...

        @property
        def backend(self) -> SQLiteBackend: ...

        @property
        def config(self) -> Config: ...

    async def list_session_tag_rollup_insights(
        self,
        query: SessionTagRollupQuery | None = None,
    ) -> list[SessionTagRollupInsight]:
        request = _default_query(query, SessionTagRollupQuery)
        status = await _read_session_insight_status(self.backend)
        _require_ready_flag(status, "tag_rollups_ready", "Session tag rollups are incomplete.")
        rows = await self.repository.list_session_tag_rollup_records(
            provider=request.provider,
            since=request.since,
            until=request.until,
            query=request.query,
        )
        insights = aggregate_session_tag_rollup_insights(rows)
        return _slice_insights(insights, offset=request.offset, limit=request.limit)

    async def list_archive_coverage_insights(
        self,
        query: ArchiveCoverageInsightQuery | None = None,
    ) -> list[ArchiveCoverageInsight]:
        request = _default_query(query, ArchiveCoverageInsightQuery)
        if request.group_by == "provider":
            provider_rows = await self.backend.get_provider_metrics_rows()
            insights = [_provider_coverage_insight(row) for row in provider_rows]
            if request.provider:
                insights = [insight for insight in insights if insight.source_name == request.provider]
            return _slice_insights(insights, offset=request.offset, limit=request.limit)

        if request.group_by not in {"day", "week"}:
            raise ValueError("archive coverage group_by must be one of: provider, day, week")

        status = await _read_session_insight_status(self.backend)
        _require_ready_flag(status, "profile_rows_ready", "Session-profile rows are incomplete.")
        records = await self.repository.list_session_profile_records(
            provider=request.provider,
            since=request.since,
            until=request.until,
            limit=None,
        )
        day_rows = build_day_session_summary_records([hydrate_session_profile(record) for record in records])
        if request.group_by == "day":
            day_insights = aggregate_day_session_summary_insights(day_rows)
            return _slice_insights(
                [_day_coverage_insight(insight) for insight in day_insights],
                offset=request.offset,
                limit=request.limit,
            )
        week_insights = aggregate_week_session_summary_insights(day_rows)
        return _slice_insights(
            [_week_coverage_insight(insight) for insight in week_insights],
            offset=request.offset,
            limit=request.limit,
        )

    async def list_tool_usage_insights(
        self,
        query: ToolUsageInsightQuery | None = None,
    ) -> list[ToolUsageInsight]:
        request = _default_query(query, ToolUsageInsightQuery)
        usage_rows = await self.backend.get_tool_usage_rows()
        coverage_rows = await self.backend.get_tool_usage_provider_coverage_rows()
        insight = build_tool_usage_insight(
            rows=usage_rows,
            coverage_rows=coverage_rows,
            query=request,
            materialized_at=generated_at(),
        )
        return [insight]

    async def list_session_cost_insights(
        self,
        query: SessionCostInsightQuery | None = None,
    ) -> list[SessionCostInsight]:
        request = _default_query(query, SessionCostInsightQuery)
        conversations = await self.repository.list(
            provider=request.provider,
            since=request.since,
            until=request.until,
            limit=None,
        )
        materialized_at = generated_at()
        insights = [
            _session_cost_insight(conversation, materialized_at=materialized_at) for conversation in conversations
        ]
        if request.conversation_id:
            insights = [insight for insight in insights if insight.conversation_id == request.conversation_id]
        insights = [insight for insight in insights if _cost_model_matches(insight, request.model)]
        insights = [insight for insight in insights if _cost_status_matches(insight, request.status)]
        insights.sort(key=lambda insight: insight.provenance.source_sort_key or 0.0, reverse=True)
        return _slice_insights(insights, offset=request.offset, limit=request.limit)

    async def list_cost_rollup_insights(
        self,
        query: CostRollupInsightQuery | None = None,
    ) -> list[CostRollupInsight]:
        request = _default_query(query, CostRollupInsightQuery)
        session_costs = await self.list_session_cost_insights(
            SessionCostInsightQuery(
                provider=request.provider,
                since=request.since,
                until=request.until,
                model=request.model,
                limit=None,
            )
        )
        rollups = aggregate_cost_rollup_insights(session_costs, materialized_at=generated_at())
        return _slice_insights(rollups, offset=request.offset, limit=request.limit)

    async def get_insight_readiness_report(
        self,
        query: InsightReadinessQuery | None = None,
    ) -> InsightReadinessReport:
        status = await _read_session_insight_status(self.backend)
        async with self.backend.connection() as conn:
            return await build_insight_readiness_report(conn, status, query)

    async def audit_insight_rigor(
        self,
        query: InsightRigorAuditQuery | None = None,
    ) -> InsightRigorAuditReport:
        """Audit per-product rigor profile across materialized insights (#1275)."""

        return await build_insight_rigor_audit_report(self, query)

    async def export_insight_bundle(
        self,
        request: InsightExportBundleRequest,
    ) -> InsightExportBundleResult:
        return await export_insight_bundle(cast(InsightExportOperations, self), self.config, request)


class ArchiveInsightDebtMixin:
    if TYPE_CHECKING:

        @property
        def config(self) -> Config: ...

    async def list_archive_debt_insights(
        self,
        query: ArchiveDebtInsightQuery | None = None,
    ) -> list[ArchiveDebtInsight]:
        request = _default_query(query, ArchiveDebtInsightQuery)
        with connection_context(self.config.db_path) as conn:
            statuses = collect_archive_debt_statuses_sync(conn)
        insights = [ArchiveDebtInsight.from_status(status) for status in statuses.values()]
        insights.sort(key=lambda insight: (insight.category, insight.debt_name))
        if request.category:
            insights = [insight for insight in insights if insight.category == request.category]
        if request.only_actionable:
            insights = [insight for insight in insights if not insight.healthy]
        return _slice_insights(insights, offset=request.offset, limit=request.limit)


class ArchiveInsightMixin(
    ArchiveInsightSessionMixin,
    ArchiveInsightAggregateMixin,
    ArchiveInsightDebtMixin,
):
    """Versioned archive-insight retrieval methods."""


class ArchiveMaintenanceMixin:
    if TYPE_CHECKING:

        @property
        def backend(self) -> SQLiteBackend: ...

    async def rebuild_session_insights(
        self,
        conversation_ids: Sequence[str] | None = None,
    ) -> SessionInsightCounts:
        """Rebuild durable session-insight read models."""
        from polylogue.storage.insights.session.rebuild import rebuild_session_insights_async

        async with self.backend.bulk_connection(), self.backend.connection() as conn:
            return await rebuild_session_insights_async(
                conn,
                conversation_ids=conversation_ids,
                transaction_depth=self.backend.transaction_depth,
            )

    async def rebuild_index(self) -> bool:
        """Rebuild the full-text search index from persisted message rows.

        Delegates to the shared indexing service through the operation
        contract, so callers (MCP, CLI, daemon) do not instantiate
        ``IndexService`` directly.
        """
        import sqlite3

        from polylogue.pipeline.services.indexing import rebuild_index as _rebuild_index

        try:
            conversation_ids = [cid async for cid in self.backend.iter_conversation_ids()]
            await _rebuild_index(self.backend, conversation_ids=conversation_ids)
            return True
        except sqlite3.DatabaseError:
            return False

    async def update_index(self, conversation_ids: list[str]) -> bool:
        """Repair FTS rows for specific conversations.

        Delegates to the shared indexing-service free functions through
        the operation contract.
        """
        import sqlite3

        from polylogue.pipeline.services.indexing import update_index_for_conversations

        try:
            await update_index_for_conversations(conversation_ids, self.backend)
            return True
        except sqlite3.DatabaseError:
            return False

    async def get_index_status(self) -> IndexStatus:
        """Return FTS5 index existence and document count."""
        import sqlite3

        from polylogue.pipeline.services.indexing import index_status

        try:
            return await index_status(self.backend)
        except sqlite3.DatabaseError:
            return IndexStatus(exists=False, count=0)


class ArchiveResumeMixin:
    async def build_resume_brief(
        self,
        session_id: str,
        *,
        related_limit: int = 6,
    ) -> ResumeBrief | None:
        """Build a compact resume handoff brief for an archived session."""
        return await build_resume_brief(cast(ResumeOperations, self), session_id, related_limit=related_limit)

    async def find_resume_candidates(
        self,
        *,
        repo_path: str,
        cwd: str | None = None,
        recent_files: Sequence[str] = (),
        limit: int = 10,
    ) -> tuple[ResumeCandidate, ...]:
        """Rank logical sessions that match the operator's current context."""
        return await find_resume_candidates(
            cast(ResumeOperations, self),
            repo_path=repo_path,
            cwd=cwd,
            recent_files=recent_files,
            limit=limit,
        )


class ArchiveOperations(
    ArchiveSearchMixin,
    ArchiveStatsMixin,
    ArchiveCompletionMixin,
    ArchiveInsightMixin,
    ArchiveMaintenanceMixin,
    ArchiveResumeMixin,
    ArchiveMutationsMixin,
):
    """Canonical archive-level operations over configured runtime dependencies."""

    def __init__(
        self,
        *,
        services: RuntimeServices | None = None,
        config: Config | None = None,
        repository: ConversationRepository | None = None,
        backend: SQLiteBackend | None = None,
    ) -> None:
        self._services = services
        self._config = config
        self._repository = repository
        self._backend = backend

    @classmethod
    def from_services(cls, services: RuntimeServices) -> ArchiveOperations:
        return cls(services=services)

    @property
    def config(self) -> Config:
        if self._config is None:
            if self._services is None:
                raise ConfigError("ArchiveOperations requires config or runtime services")
            self._config = self._services.get_config()
        return self._config

    @property
    def repository(self) -> ConversationRepository:
        if self._repository is None:
            if self._services is None:
                raise ConfigError("ArchiveOperations requires repository or runtime services")
            self._repository = self._services.get_repository()
        return self._repository

    @property
    def backend(self) -> SQLiteBackend:
        if self._backend is None:
            if self._services is not None:
                self._backend = self._services.get_backend()
            else:
                self._backend = self.repository.backend
        return self._backend


async def _with_operations(
    action: Callable[[ArchiveOperations], Awaitable[_ResultT]],
    *,
    services: RuntimeServices | None = None,
    db_path: Path | None = None,
) -> _ResultT:
    owns_services = services is None
    runtime_services = services or build_runtime_services(db_path=db_path)
    operations = ArchiveOperations.from_services(runtime_services)
    try:
        return await action(operations)
    finally:
        if owns_services:
            await runtime_services.close()


async def get_provider_counts(
    *,
    services: RuntimeServices | None = None,
    db_path: Path | None = None,
) -> list[tuple[str, int]]:
    """Return (provider, conversation_count) pairs for archive summaries."""

    async def _action(operations: ArchiveOperations) -> list[tuple[str, int]]:
        return await operations.provider_counts()

    return await _with_operations(_action, services=services, db_path=db_path)


async def list_archive_coverage_insights(
    *,
    services: RuntimeServices | None = None,
    db_path: Path | None = None,
) -> list[ArchiveCoverageInsight]:
    """Return provider-level archive coverage buckets for summaries."""

    async def _action(operations: ArchiveOperations) -> list[ArchiveCoverageInsight]:
        return await operations.list_archive_coverage_insights()

    return await _with_operations(_action, services=services, db_path=db_path)


async def list_tool_usage_insights(
    *,
    services: RuntimeServices | None = None,
    db_path: Path | None = None,
    query: ToolUsageInsightQuery | None = None,
) -> list[ToolUsageInsight]:
    """Return tool-usage insights with explicit per-provider coverage."""

    async def _action(operations: ArchiveOperations) -> list[ToolUsageInsight]:
        return await operations.list_tool_usage_insights(query)

    return await _with_operations(_action, services=services, db_path=db_path)


__all__ = [
    "ArchiveDebtInsight",
    "ArchiveOperations",
    "ArchiveStats",
    "CompletionAggregate",
    "build_tool_usage_insight",
    "get_provider_counts",
    "list_archive_coverage_insights",
    "list_tool_usage_insights",
]

