"""Archive operations shared across facade, CLI, and MCP call sites."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import structlog

from polylogue.archive_product_rollups import aggregate_session_tag_rollup_products
from polylogue.archive_product_summaries import (
    aggregate_day_session_summary_products,
    aggregate_week_session_summary_products,
)
from polylogue.archive_products import (
    ArchiveDebtProduct,
    ArchiveDebtProductQuery,
    ArchiveProductUnavailableError,
    DaySessionSummaryProduct,
    DaySessionSummaryProductQuery,
    ProviderAnalyticsProduct,
    ProviderAnalyticsProductQuery,
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
from polylogue.lib.query_spec import ConversationQuerySpec
from polylogue.maintenance_targets import build_maintenance_target_catalog
from polylogue.paths import conversation_render_root
from polylogue.services import RuntimeServices, build_runtime_services
from polylogue.storage.backends.connection import connection_context
from polylogue.storage.repair import collect_archive_debt_statuses_sync
from polylogue.storage.search import SearchHit, SearchResult

logger = structlog.get_logger(__name__)
_MAINTENANCE_TARGET_CATALOG = build_maintenance_target_catalog()
_SESSION_PRODUCT_REPAIR_HINT = _MAINTENANCE_TARGET_CATALOG.repair_hint(("session_products",), include_run_all=True)
_PROFILE_FTS_STATUS_BY_TIER = {
    "merged": "profile_merged_fts_ready",
    "evidence": "profile_evidence_fts_ready",
    "inference": "profile_inference_fts_ready",
    "enrichment": "profile_enrichment_fts_ready",
}

if TYPE_CHECKING:
    from polylogue.config import Config
    from polylogue.lib.conversation_models import Conversation
    from polylogue.storage.backends.async_sqlite import SQLiteBackend
    from polylogue.storage.repository import ConversationRepository


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


def provider_analytics_product(row) -> ProviderAnalyticsProduct:
    conversation_count = row["conversation_count"]
    user_message_count = row["user_message_count"]
    assistant_message_count = row["assistant_message_count"]
    user_word_sum = row["user_word_sum"] or 0
    assistant_word_sum = row["assistant_word_sum"] or 0
    tool_use_percentage = (
        (row["conversations_with_tools"] / conversation_count) * 100 if conversation_count > 0 else 0.0
    )
    thinking_percentage = (
        (row["conversations_with_thinking"] / conversation_count) * 100 if conversation_count > 0 else 0.0
    )
    return ProviderAnalyticsProduct(
        provider_name=row["provider_name"] or "unknown",
        conversation_count=conversation_count,
        message_count=row["message_count"],
        user_message_count=user_message_count,
        assistant_message_count=assistant_message_count,
        avg_messages_per_conversation=(row["message_count"] / conversation_count if conversation_count > 0 else 0.0),
        avg_user_words=(user_word_sum / user_message_count if user_message_count > 0 else 0.0),
        avg_assistant_words=(assistant_word_sum / assistant_message_count if assistant_message_count > 0 else 0.0),
        tool_use_count=row["tool_use_count"],
        thinking_count=row["thinking_count"],
        total_conversations_with_tools=row["conversations_with_tools"],
        total_conversations_with_thinking=row["conversations_with_thinking"],
        tool_use_percentage=tool_use_percentage,
        thinking_percentage=thinking_percentage,
    )


def _require_ready_flag(
    status: dict[str, int | bool],
    flag: str,
    detail: str,
) -> None:
    if bool(status.get(flag, False)):
        return
    raise ArchiveProductUnavailableError(f"{detail} {_SESSION_PRODUCT_REPAIR_HINT}")


async def _read_session_product_status(repository: ConversationRepository) -> dict[str, int | bool]:
    return await repository.queries.get_session_product_status()


class ArchiveSearchMixin:
    """Conversation retrieval and search methods for archive operations."""

    async def get_conversation(self, conversation_id: str):
        return await self.repository.view(conversation_id)

    async def get_conversations(self, conversation_ids: list[str]):
        return await self.repository.get_many(conversation_ids)

    async def list_conversations(
        self,
        *,
        provider: str | None = None,
        limit: int | None = None,
    ):
        return await self.repository.list(provider=provider, limit=limit)

    async def query_conversations(self, spec: ConversationQuerySpec):
        return await spec.list(self.repository)

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
            providers=(source,) if source else (),
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
        recent,
    ):
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

    async def storage_stats(self):
        return await self.repository.get_archive_stats()

    async def summary_stats(self) -> ArchiveStats:
        storage_stats = await self.storage_stats()
        aggregate_stats = await self.repository.queries.aggregate_message_stats()
        tags = await self.repository.list_tags()
        recent = await self.list_conversations(limit=5)

        last_sync = None
        try:
            last_sync = await self.backend.queries.get_last_sync_timestamp()
        except Exception as exc:  # pragma: no cover - defensive debug path
            logger.debug("failed to query last sync timestamp", error=str(exc))

        return ArchiveStats(
            conversation_count=storage_stats.total_conversations,
            message_count=storage_stats.total_messages,
            word_count=int(aggregate_stats.get("words_approx", 0)),
            providers=storage_stats.providers,
            tags=tags,
            last_sync=last_sync,
            recent=recent,
        )

    async def provider_counts(self) -> list[tuple[str, int]]:
        rows = await self.backend.queries.get_provider_conversation_counts()
        return [(row["provider_name"] or "unknown", row["conversation_count"]) for row in rows]

    async def get_session_product_status(self) -> dict[str, int | bool]:
        return await self.repository.get_session_product_status()


class ArchiveProductSessionMixin:
    async def _session_product_status(self) -> dict[str, int | bool]:
        return await _read_session_product_status(self.repository)

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
        request = query or SessionProfileProductQuery()
        status = await self._session_product_status()
        _require_ready_flag(status, "profile_rows_ready", "Session-profile rows are incomplete.")
        if request.query:
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
        request = query or SessionEnrichmentProductQuery()
        status = await self._session_product_status()
        _require_ready_flag(status, "profile_rows_ready", "Session-profile rows are incomplete.")
        if request.query:
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
        request = query or SessionWorkEventProductQuery()
        status = await self._session_product_status()
        _require_ready_flag(status, "work_event_inference_rows_ready", "Session work-event rows are incomplete.")
        if request.query:
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
        request = query or SessionPhaseProductQuery()
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
        request = query or WorkThreadProductQuery()
        status = await self._session_product_status()
        _require_ready_flag(status, "threads_ready", "Work-thread rows are incomplete.")
        if request.query:
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
    async def list_session_tag_rollup_products(
        self,
        query: SessionTagRollupQuery | None = None,
    ) -> list[SessionTagRollupProduct]:
        request = query or SessionTagRollupQuery()
        status = await _read_session_product_status(self.repository)
        _require_ready_flag(status, "tag_rollups_ready", "Session tag rollups are incomplete.")
        rows = await self.repository.list_session_tag_rollup_records(
            provider=request.provider,
            since=request.since,
            until=request.until,
            query=request.query,
        )
        products = aggregate_session_tag_rollup_products(rows)
        if request.offset:
            products = products[request.offset :]
        if request.limit is not None:
            products = products[: request.limit]
        return products

    async def list_day_session_summary_products(
        self,
        query: DaySessionSummaryProductQuery | None = None,
    ) -> list[DaySessionSummaryProduct]:
        request = query or DaySessionSummaryProductQuery()
        status = await _read_session_product_status(self.repository)
        _require_ready_flag(status, "day_summaries_ready", "Day session summaries are incomplete.")
        rows = await self.repository.list_day_session_summary_records(
            provider=request.provider,
            since=request.since,
            until=request.until,
        )
        products = aggregate_day_session_summary_products(rows)
        if request.offset:
            products = products[request.offset :]
        if request.limit is not None:
            products = products[: request.limit]
        return products

    async def list_week_session_summary_products(
        self,
        query: WeekSessionSummaryProductQuery | None = None,
    ) -> list[WeekSessionSummaryProduct]:
        request = query or WeekSessionSummaryProductQuery()
        status = await _read_session_product_status(self.repository)
        _require_ready_flag(status, "week_summaries_ready", "Week session summaries are incomplete.")
        rows = await self.repository.list_day_session_summary_records(
            provider=request.provider,
            since=request.since,
            until=request.until,
        )
        products = aggregate_week_session_summary_products(rows)
        if request.offset:
            products = products[request.offset :]
        if request.limit is not None:
            products = products[: request.limit]
        return products

    async def list_provider_analytics_products(
        self,
        query: ProviderAnalyticsProductQuery | None = None,
    ) -> list[ProviderAnalyticsProduct]:
        rows = await self.backend.queries.get_provider_metrics_rows()
        products = [provider_analytics_product(row) for row in rows]
        request = query or ProviderAnalyticsProductQuery()
        if request.provider:
            products = [product for product in products if product.provider_name == request.provider]
        if request.offset:
            products = products[request.offset :]
        if request.limit is not None:
            products = products[: request.limit]
        return products


class ArchiveProductDebtMixin:
    async def list_archive_debt_products(
        self,
        query: ArchiveDebtProductQuery | None = None,
    ) -> list[ArchiveDebtProduct]:
        request = query or ArchiveDebtProductQuery()
        with connection_context(self.config.db_path) as conn:
            statuses = collect_archive_debt_statuses_sync(conn)
        products = [ArchiveDebtProduct.from_status(status) for status in statuses.values()]
        products.sort(key=lambda product: (product.category, product.debt_name))
        if request.category:
            products = [product for product in products if product.category == request.category]
        if request.only_actionable:
            products = [product for product in products if not product.healthy]
        if request.offset:
            products = products[request.offset :]
        if request.limit is not None:
            products = products[: request.limit]
        return products


class ArchiveProductMixin(
    ArchiveProductSessionMixin,
    ArchiveProductAggregateMixin,
    ArchiveProductDebtMixin,
):
    """Versioned archive-product retrieval methods."""


class ArchiveOperations(ArchiveSearchMixin, ArchiveStatsMixin, ArchiveProductMixin):
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
                raise RuntimeError("ArchiveOperations requires config or runtime services")
            self._config = self._services.get_config()
        return self._config

    @property
    def repository(self) -> ConversationRepository:
        if self._repository is None:
            if self._services is None:
                raise RuntimeError("ArchiveOperations requires repository or runtime services")
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
    action,
    *,
    services: RuntimeServices | None = None,
    db_path: Path | None = None,
):
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
