"""Archive operations shared across facade, CLI, and MCP call sites."""

from __future__ import annotations

from collections import Counter
from collections.abc import Awaitable, Callable, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, TypeVar, cast

import structlog

from polylogue.archive_product_rollups import aggregate_session_tag_rollup_products
from polylogue.archive_product_summaries import (
    aggregate_day_session_summary_products,
    aggregate_week_session_summary_products,
)
from polylogue.archive_products import (
    ArchiveDebtProduct,
    ArchiveDebtProductQuery,
    ArchiveProductProvenance,
    ArchiveProductUnavailableError,
    CostRollupProduct,
    CostRollupProductQuery,
    DaySessionSummaryProduct,
    DaySessionSummaryProductQuery,
    ProviderAnalyticsProduct,
    ProviderAnalyticsProductQuery,
    SessionCostProduct,
    SessionCostProductQuery,
    SessionEnrichmentProduct,
    SessionEnrichmentProductQuery,
    SessionPhaseProduct,
    SessionPhaseProductQuery,
    SessionProfileProduct,
    SessionProfileProductQuery,
    SessionTagRollupProduct,
    SessionTagRollupQuery,
    SessionWorkEventProduct,
    SessionWorkEventProductQuery,
    WeekSessionSummaryProduct,
    WeekSessionSummaryProductQuery,
    WorkThreadProduct,
    WorkThreadProductQuery,
)
from polylogue.archive_resume import ResumeBrief, ResumeOperations, build_resume_brief
from polylogue.config import ConfigError
from polylogue.lib.conversation.models import ConversationSummary
from polylogue.lib.pricing import CostUsagePayload, _normalize_model, estimate_conversation_cost, generated_at
from polylogue.lib.query.spec import ConversationQuerySpec
from polylogue.lib.semantic.content_projection import ContentProjectionSpec
from polylogue.maintenance.targets import build_maintenance_target_catalog
from polylogue.paths.sanitize import conversation_render_root
from polylogue.product_export_bundles import (
    ProductExportBundleRequest,
    ProductExportBundleResult,
    ProductExportOperations,
    export_product_bundle,
)
from polylogue.product_readiness import (
    ProductReadinessQuery,
    ProductReadinessReport,
    build_product_readiness_report,
)
from polylogue.services import RuntimeServices, build_runtime_services
from polylogue.storage.backends.connection import connection_context
from polylogue.storage.backends.queries.stats import ProviderMetricsRow
from polylogue.storage.products.session.runtime import (
    SessionProductReadyFlag,
    SessionProductStatusSnapshot,
)
from polylogue.storage.repair import collect_archive_debt_statuses_sync
from polylogue.storage.runtime.store_constants import SESSION_PRODUCT_MATERIALIZER_VERSION
from polylogue.storage.search import SearchHit, SearchResult
from polylogue.types import Provider

logger = structlog.get_logger(__name__)
_MAINTENANCE_TARGET_CATALOG = build_maintenance_target_catalog()
_SESSION_PRODUCT_REPAIR_HINT = _MAINTENANCE_TARGET_CATALOG.repair_hint(("session_products",), include_run_all=True)
_PROFILE_FTS_STATUS_BY_TIER: dict[str, SessionProductReadyFlag] = {
    "merged": "profile_merged_fts_ready",
    "evidence": "profile_evidence_fts_ready",
    "inference": "profile_inference_fts_ready",
    "enrichment": "profile_enrichment_fts_ready",
}

if TYPE_CHECKING:
    from polylogue.config import Config
    from polylogue.lib.conversation.models import Conversation
    from polylogue.lib.conversation.neighbor_candidates import ConversationNeighborCandidate
    from polylogue.lib.query.miss_diagnostics import QueryMissDiagnostics
    from polylogue.lib.search_hits import ConversationSearchHit
    from polylogue.lib.stats import ArchiveStats as StorageArchiveStats
    from polylogue.storage.backends.async_sqlite import SQLiteBackend
    from polylogue.storage.products.session.runtime import SessionProductCounts
    from polylogue.storage.repository import ConversationRepository

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
    render_root_path: Path,
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
    conversation_path = (
        conversation_render_root(render_root_path, conversation.provider, str(conversation.id)) / "conversation.md"
    )
    return SearchHit(
        conversation_id=str(conversation.id),
        provider_name=conversation.provider,
        source_name=None,
        message_id=message_id,
        title=conversation.display_title,
        timestamp=timestamp,
        snippet=snippet,
        conversation_path=conversation_path,
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


def _slice_products(
    products: list[_ResultT],
    *,
    offset: int,
    limit: int | None,
) -> list[_ResultT]:
    if offset:
        products = products[offset:]
    if limit is not None:
        products = products[:limit]
    return products


def provider_analytics_product(row: ProviderMetricsRow) -> ProviderAnalyticsProduct:
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
    return ProviderAnalyticsProduct(
        provider_name=row["provider_name"] or "unknown",
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


def _session_cost_product(conversation: Conversation, *, materialized_at: str) -> SessionCostProduct:
    estimate = estimate_conversation_cost(conversation)
    source_updated = conversation.updated_at or conversation.created_at
    return SessionCostProduct(
        conversation_id=str(conversation.id),
        provider_name=str(conversation.provider),
        title=conversation.title,
        created_at=conversation.created_at.isoformat() if conversation.created_at is not None else None,
        updated_at=conversation.updated_at.isoformat() if conversation.updated_at is not None else None,
        estimate=estimate,
        provenance=ArchiveProductProvenance(
            materializer_version=SESSION_PRODUCT_MATERIALIZER_VERSION,
            materialized_at=materialized_at,
            source_updated_at=source_updated.isoformat() if source_updated is not None else None,
            source_sort_key=source_updated.timestamp() if source_updated is not None else None,
        ),
    )


def _cost_model_matches(product: SessionCostProduct, model_filter: str | None) -> bool:
    if not model_filter:
        return True
    normalized_filter = _normalize_model(model_filter)
    return product.estimate.model_name == model_filter or product.estimate.normalized_model == normalized_filter


def _cost_status_matches(product: SessionCostProduct, status_filter: str | None) -> bool:
    return not status_filter or product.estimate.status == status_filter


def _require_ready_flag(
    status: SessionProductStatusSnapshot,
    flag: SessionProductReadyFlag,
    detail: str,
) -> None:
    if status.ready_flag(flag):
        return
    raise ArchiveProductUnavailableError(f"{detail} {_SESSION_PRODUCT_REPAIR_HINT}")


async def _read_session_product_status(backend: SQLiteBackend) -> SessionProductStatusSnapshot:
    return await backend.get_session_product_status()


class ArchiveSearchMixin:
    """Conversation retrieval and search methods for archive operations."""

    if TYPE_CHECKING:

        @property
        def repository(self) -> ConversationRepository: ...

        @property
        def config(self) -> Config: ...

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
        conversations = await spec.list(self.repository)
        if content_projection is None or not content_projection.filters_content():
            return conversations
        return [conversation.with_content_projection(content_projection) for conversation in conversations]

    async def search_conversation_hits(self, spec: ConversationQuerySpec) -> list[ConversationSearchHit]:
        from polylogue.lib.query.search_hits import search_hits_for_plan

        hits = await search_hits_for_plan(spec.to_plan(), self.repository)
        if not hits:
            return hits
        counts = await self.repository.get_message_counts_batch([hit.conversation_id for hit in hits])
        return [hit.with_message_count(counts.get(hit.conversation_id)) for hit in hits]

    async def neighbor_candidates(
        self,
        *,
        conversation_id: str | None = None,
        query: str | None = None,
        provider: str | None = None,
        limit: int = 10,
        window_hours: int = 24,
    ) -> list[ConversationNeighborCandidate]:
        from polylogue.lib.conversation.neighbor_candidates import (
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
        from polylogue.lib.query.miss_diagnostics import diagnose_query_miss

        return await diagnose_query_miss(self.repository, spec, config=self.config)

    async def get_session_tree(self, conversation_id: str) -> list[Conversation]:
        return await self.repository.get_session_tree(conversation_id)

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
                    render_root_path=self.config.render_root,
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
            logger.debug("failed to query last sync timestamp", error=str(exc))

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
        return [(row["provider_name"] or "unknown", row["conversation_count"]) for row in rows]

    async def get_session_product_status(self) -> SessionProductStatusSnapshot:
        return await self.backend.get_session_product_status()


class ArchiveProductSessionMixin:
    if TYPE_CHECKING:

        @property
        def repository(self) -> ConversationRepository: ...

        @property
        def backend(self) -> SQLiteBackend: ...

    async def _session_product_status(self) -> SessionProductStatusSnapshot:
        return await _read_session_product_status(self.backend)

    async def get_session_profile_product(
        self,
        conversation_id: str,
        *,
        tier: str = "merged",
    ) -> SessionProfileProduct | None:
        status = await self._session_product_status()
        _require_ready_flag(status, "profile_rows_ready", "Session-profile rows are incomplete.")
        record = await self.repository.get_session_profile_record(conversation_id)
        return SessionProfileProduct.from_record(record, tier=tier) if record is not None else None

    async def list_session_profile_products(
        self,
        query: SessionProfileProductQuery | None = None,
    ) -> list[SessionProfileProduct]:
        request = _default_query(query, SessionProfileProductQuery)
        status = await self._session_product_status()
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
            sort=request.sort,
            tier=request.tier,
            limit=request.limit,
            offset=request.offset,
            query=request.query,
        )
        return [SessionProfileProduct.from_record(record, tier=request.tier) for record in records]

    async def get_session_enrichment_product(
        self,
        conversation_id: str,
    ) -> SessionEnrichmentProduct | None:
        status = await self._session_product_status()
        _require_ready_flag(status, "profile_rows_ready", "Session-profile rows are incomplete.")
        record = await self.repository.get_session_enrichment_record(conversation_id)
        return SessionEnrichmentProduct.from_record(record) if record is not None else None

    async def list_session_enrichment_products(
        self,
        query: SessionEnrichmentProductQuery | None = None,
    ) -> list[SessionEnrichmentProduct]:
        request = _default_query(query, SessionEnrichmentProductQuery)
        status = await self._session_product_status()
        _require_ready_flag(status, "profile_rows_ready", "Session-profile rows are incomplete.")
        if _query_wants_search(request):
            _require_ready_flag(
                status,
                "profile_enrichment_fts_ready",
                "Session-profile enrichment search index is incomplete.",
            )
        records = await self.repository.list_session_enrichment_records(
            provider=request.provider,
            since=request.since,
            until=request.until,
            first_message_since=request.first_message_since,
            first_message_until=request.first_message_until,
            session_date_since=request.session_date_since,
            session_date_until=request.session_date_until,
            min_wallclock_seconds=request.min_wallclock_seconds,
            max_wallclock_seconds=request.max_wallclock_seconds,
            sort=request.sort,
            limit=request.limit,
            offset=request.offset,
            query=request.query,
        )
        return [SessionEnrichmentProduct.from_record(record) for record in records]

    async def get_session_work_event_products(
        self,
        conversation_id: str,
    ) -> list[SessionWorkEventProduct]:
        status = await self._session_product_status()
        _require_ready_flag(status, "work_event_inference_rows_ready", "Session work-event rows are incomplete.")
        records = await self.repository.get_session_work_event_records(conversation_id)
        return [SessionWorkEventProduct.from_record(record) for record in records]

    async def list_session_work_event_products(
        self,
        query: SessionWorkEventProductQuery | None = None,
    ) -> list[SessionWorkEventProduct]:
        request = _default_query(query, SessionWorkEventProductQuery)
        status = await self._session_product_status()
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
            kind=request.kind,
            limit=request.limit,
            offset=request.offset,
            query=request.query,
        )
        return [SessionWorkEventProduct.from_record(record) for record in records]

    async def get_session_phase_products(
        self,
        conversation_id: str,
    ) -> list[SessionPhaseProduct]:
        status = await self._session_product_status()
        _require_ready_flag(status, "phase_inference_rows_ready", "Session phase rows are incomplete.")
        records = await self.repository.get_session_phase_records(conversation_id)
        return [SessionPhaseProduct.from_record(record) for record in records]

    async def list_session_phase_products(
        self,
        query: SessionPhaseProductQuery | None = None,
    ) -> list[SessionPhaseProduct]:
        request = _default_query(query, SessionPhaseProductQuery)
        status = await self._session_product_status()
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
        return [SessionPhaseProduct.from_record(record) for record in records]

    async def get_work_thread_product(self, thread_id: str) -> WorkThreadProduct | None:
        status = await self._session_product_status()
        _require_ready_flag(status, "threads_ready", "Work-thread rows are incomplete.")
        record = await self.repository.get_work_thread_record(thread_id)
        return WorkThreadProduct.from_record(record) if record is not None else None

    async def list_work_thread_products(
        self,
        query: WorkThreadProductQuery | None = None,
    ) -> list[WorkThreadProduct]:
        request = _default_query(query, WorkThreadProductQuery)
        status = await self._session_product_status()
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
        return [WorkThreadProduct.from_record(record) for record in records]


class ArchiveProductAggregateMixin:
    if TYPE_CHECKING:

        @property
        def repository(self) -> ConversationRepository: ...

        @property
        def backend(self) -> SQLiteBackend: ...

        @property
        def config(self) -> Config: ...

    async def list_session_tag_rollup_products(
        self,
        query: SessionTagRollupQuery | None = None,
    ) -> list[SessionTagRollupProduct]:
        request = _default_query(query, SessionTagRollupQuery)
        status = await _read_session_product_status(self.backend)
        _require_ready_flag(status, "tag_rollups_ready", "Session tag rollups are incomplete.")
        rows = await self.repository.list_session_tag_rollup_records(
            provider=request.provider,
            since=request.since,
            until=request.until,
            query=request.query,
        )
        products = aggregate_session_tag_rollup_products(rows)
        return _slice_products(products, offset=request.offset, limit=request.limit)

    async def list_day_session_summary_products(
        self,
        query: DaySessionSummaryProductQuery | None = None,
    ) -> list[DaySessionSummaryProduct]:
        request = _default_query(query, DaySessionSummaryProductQuery)
        status = await _read_session_product_status(self.backend)
        _require_ready_flag(status, "day_summaries_ready", "Day session summaries are incomplete.")
        rows = await self.repository.list_day_session_summary_records(
            provider=request.provider,
            since=request.since,
            until=request.until,
        )
        products = aggregate_day_session_summary_products(rows)
        return _slice_products(products, offset=request.offset, limit=request.limit)

    async def list_week_session_summary_products(
        self,
        query: WeekSessionSummaryProductQuery | None = None,
    ) -> list[WeekSessionSummaryProduct]:
        request = _default_query(query, WeekSessionSummaryProductQuery)
        status = await _read_session_product_status(self.backend)
        _require_ready_flag(status, "week_summaries_ready", "Week session summaries are incomplete.")
        rows = await self.repository.list_day_session_summary_records(
            provider=request.provider,
            since=request.since,
            until=request.until,
        )
        products = aggregate_week_session_summary_products(rows)
        return _slice_products(products, offset=request.offset, limit=request.limit)

    async def list_provider_analytics_products(
        self,
        query: ProviderAnalyticsProductQuery | None = None,
    ) -> list[ProviderAnalyticsProduct]:
        rows = await self.backend.get_provider_metrics_rows()
        products = [provider_analytics_product(row) for row in rows]
        request = _default_query(query, ProviderAnalyticsProductQuery)
        if request.provider:
            products = [product for product in products if product.provider_name == request.provider]
        return _slice_products(products, offset=request.offset, limit=request.limit)

    async def list_session_cost_products(
        self,
        query: SessionCostProductQuery | None = None,
    ) -> list[SessionCostProduct]:
        request = _default_query(query, SessionCostProductQuery)
        conversations = await self.repository.list(
            provider=request.provider,
            since=request.since,
            until=request.until,
            limit=None,
        )
        materialized_at = generated_at()
        products = [
            _session_cost_product(conversation, materialized_at=materialized_at) for conversation in conversations
        ]
        if request.conversation_id:
            products = [product for product in products if product.conversation_id == request.conversation_id]
        products = [product for product in products if _cost_model_matches(product, request.model)]
        products = [product for product in products if _cost_status_matches(product, request.status)]
        products.sort(key=lambda product: product.provenance.source_sort_key or 0.0, reverse=True)
        return _slice_products(products, offset=request.offset, limit=request.limit)

    async def list_cost_rollup_products(
        self,
        query: CostRollupProductQuery | None = None,
    ) -> list[CostRollupProduct]:
        request = _default_query(query, CostRollupProductQuery)
        session_products = await self.list_session_cost_products(
            SessionCostProductQuery(
                provider=request.provider,
                since=request.since,
                until=request.until,
                model=request.model,
                limit=None,
            )
        )
        grouped: dict[tuple[str, str | None], list[SessionCostProduct]] = {}
        for product in session_products:
            key = (product.provider_name, product.estimate.normalized_model or product.estimate.model_name)
            grouped.setdefault(key, []).append(product)

        rollups: list[CostRollupProduct] = []
        for (provider_name, normalized_model), products in sorted(
            grouped.items(),
            key=lambda item: (item[0][0], item[0][1] or ""),
        ):
            usage = CostUsagePayload()
            status_counts: Counter[str] = Counter()
            total_usd = 0.0
            priced_count = 0
            confidence_total = 0.0
            source_updated_at = max(
                (
                    product.provenance.source_updated_at
                    for product in products
                    if product.provenance.source_updated_at is not None
                ),
                default=None,
            )
            source_sort_key = max(
                (
                    product.provenance.source_sort_key
                    for product in products
                    if product.provenance.source_sort_key is not None
                ),
                default=None,
            )
            model_names = Counter(product.estimate.model_name for product in products if product.estimate.model_name)
            for product in products:
                estimate = product.estimate
                usage = usage.plus(estimate.usage)
                status_counts[estimate.status] += 1
                total_usd += estimate.total_usd
                if estimate.priced:
                    priced_count += 1
                    confidence_total += estimate.confidence
            rollups.append(
                CostRollupProduct(
                    provider_name=provider_name,
                    model_name=model_names.most_common(1)[0][0] if model_names else None,
                    normalized_model=normalized_model,
                    session_count=len(products),
                    priced_session_count=priced_count,
                    unavailable_session_count=status_counts["unavailable"],
                    status_counts=dict(sorted(status_counts.items())),
                    total_usd=total_usd,
                    usage=usage,
                    confidence=(confidence_total / priced_count if priced_count else 0.0),
                    provenance=ArchiveProductProvenance(
                        materializer_version=SESSION_PRODUCT_MATERIALIZER_VERSION,
                        materialized_at=generated_at(),
                        source_updated_at=source_updated_at,
                        source_sort_key=source_sort_key,
                    ),
                )
            )
        rollups.sort(key=lambda product: product.total_usd, reverse=True)
        return _slice_products(rollups, offset=request.offset, limit=request.limit)

    async def get_product_readiness_report(
        self,
        query: ProductReadinessQuery | None = None,
    ) -> ProductReadinessReport:
        status = await _read_session_product_status(self.backend)
        async with self.backend.connection() as conn:
            return await build_product_readiness_report(conn, status, query)

    async def export_product_bundle(
        self,
        request: ProductExportBundleRequest,
    ) -> ProductExportBundleResult:
        return await export_product_bundle(cast(ProductExportOperations, self), self.config, request)


class ArchiveProductDebtMixin:
    if TYPE_CHECKING:

        @property
        def config(self) -> Config: ...

    async def list_archive_debt_products(
        self,
        query: ArchiveDebtProductQuery | None = None,
    ) -> list[ArchiveDebtProduct]:
        request = _default_query(query, ArchiveDebtProductQuery)
        with connection_context(self.config.db_path) as conn:
            statuses = collect_archive_debt_statuses_sync(conn)
        products = [ArchiveDebtProduct.from_status(status) for status in statuses.values()]
        products.sort(key=lambda product: (product.category, product.debt_name))
        if request.category:
            products = [product for product in products if product.category == request.category]
        if request.only_actionable:
            products = [product for product in products if not product.healthy]
        return _slice_products(products, offset=request.offset, limit=request.limit)


class ArchiveProductMixin(
    ArchiveProductSessionMixin,
    ArchiveProductAggregateMixin,
    ArchiveProductDebtMixin,
):
    """Versioned archive-product retrieval methods."""


class ArchiveMaintenanceMixin:
    if TYPE_CHECKING:

        @property
        def backend(self) -> SQLiteBackend: ...

    async def rebuild_session_products(
        self,
        conversation_ids: Sequence[str] | None = None,
    ) -> SessionProductCounts:
        """Rebuild durable session-product read models."""
        from polylogue.storage.products.session.rebuild import rebuild_session_products_async

        async with self.backend.bulk_connection(), self.backend.connection() as conn:
            return await rebuild_session_products_async(
                conn,
                conversation_ids=conversation_ids,
                transaction_depth=self.backend.transaction_depth,
            )


class ArchiveResumeMixin:
    async def build_resume_brief(
        self,
        session_id: str,
        *,
        related_limit: int = 6,
    ) -> ResumeBrief | None:
        """Build a compact resume handoff brief for an archived session."""
        return await build_resume_brief(cast(ResumeOperations, self), session_id, related_limit=related_limit)


class ArchiveOperations(
    ArchiveSearchMixin,
    ArchiveStatsMixin,
    ArchiveProductMixin,
    ArchiveMaintenanceMixin,
    ArchiveResumeMixin,
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


async def list_provider_analytics_products(
    *,
    services: RuntimeServices | None = None,
    db_path: Path | None = None,
) -> list[ProviderAnalyticsProduct]:
    """Return provider-level analytics products for archive summaries."""

    async def _action(operations: ArchiveOperations) -> list[ProviderAnalyticsProduct]:
        return await operations.list_provider_analytics_products()

    return await _with_operations(_action, services=services, db_path=db_path)


__all__ = [
    "ArchiveDebtProduct",
    "ArchiveOperations",
    "ArchiveStats",
    "get_provider_counts",
    "list_provider_analytics_products",
]
