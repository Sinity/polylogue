"""Archive/query domain methods for the async Polylogue facade."""

from __future__ import annotations

from collections.abc import Sequence
from contextlib import suppress
from typing import TYPE_CHECKING, Protocol

from polylogue.archive_products import (
    SessionEnrichmentProduct,
    SessionEnrichmentProductQuery,
    SessionProfileProduct,
    SessionProfileProductQuery,
)
from polylogue.storage.session_product_runtime import SessionProductStatusSnapshot

if TYPE_CHECKING:
    from polylogue.archive_resume import ResumeBrief
    from polylogue.config import Config
    from polylogue.lib.conversation_models import Conversation
    from polylogue.lib.filters import ConversationFilter
    from polylogue.operations import ArchiveStats
    from polylogue.product_export_bundles import ProductExportBundleRequest, ProductExportBundleResult
    from polylogue.product_readiness import ProductReadinessQuery, ProductReadinessReport
    from polylogue.readiness import ReadinessReport
    from polylogue.storage.repository import ConversationRepository
    from polylogue.storage.search_models import SearchResult
    from polylogue.storage.session_product_runtime import SessionProductCounts

    class _ArchiveOperationsSurface(Protocol):
        async def get_conversation(self, conversation_id: str) -> Conversation | None: ...

        async def get_conversations(self, conversation_ids: list[str]) -> list[Conversation]: ...

        async def list_conversations(
            self,
            *,
            provider: str | None = None,
            limit: int | None = None,
        ) -> list[Conversation]: ...

        async def search(
            self,
            query: str,
            *,
            limit: int = 100,
            source: str | None = None,
            since: str | None = None,
        ) -> SearchResult: ...

        async def get_session_product_status(self) -> SessionProductStatusSnapshot: ...

        async def get_session_profile_product(
            self,
            conversation_id: str,
            *,
            tier: str = "merged",
        ) -> SessionProfileProduct | None: ...

        async def list_session_profile_products(
            self,
            query: SessionProfileProductQuery | None = None,
        ) -> list[SessionProfileProduct]: ...

        async def get_session_enrichment_product(
            self,
            conversation_id: str,
        ) -> SessionEnrichmentProduct | None: ...

        async def list_session_enrichment_products(
            self,
            query: SessionEnrichmentProductQuery | None = None,
        ) -> list[SessionEnrichmentProduct]: ...

        async def summary_stats(self) -> ArchiveStats: ...

        async def rebuild_session_products(
            self,
            conversation_ids: Sequence[str] | None = None,
        ) -> SessionProductCounts: ...

        async def build_resume_brief(
            self,
            session_id: str,
            *,
            related_limit: int = 6,
        ) -> ResumeBrief | None: ...

        async def get_product_readiness_report(
            self,
            query: ProductReadinessQuery | None = None,
        ) -> ProductReadinessReport: ...

        async def export_product_bundle(
            self,
            request: ProductExportBundleRequest,
        ) -> ProductExportBundleResult: ...


class PolylogueArchiveMixin:
    if TYPE_CHECKING:

        @property
        def config(self) -> Config: ...

        @property
        def operations(self) -> _ArchiveOperationsSurface: ...

        @property
        def repository(self) -> ConversationRepository: ...

    async def get_conversation(self, conversation_id: str) -> Conversation | None:
        return await self.operations.get_conversation(conversation_id)

    async def get_conversations(self, conversation_ids: list[str]) -> list[Conversation]:
        return await self.operations.get_conversations(conversation_ids)

    async def list_conversations(
        self,
        provider: str | None = None,
        limit: int | None = None,
    ) -> list[Conversation]:
        return await self.operations.list_conversations(
            provider=provider,
            limit=limit,
        )

    async def search(
        self,
        query: str,
        *,
        limit: int = 100,
        source: str | None = None,
        since: str | None = None,
    ) -> SearchResult:
        return await self.operations.search(
            query,
            limit=limit,
            source=source,
            since=since,
        )

    async def get_session_product_status(self) -> SessionProductStatusSnapshot:
        return await self.operations.get_session_product_status()

    async def get_session_profile_product(
        self,
        conversation_id: str,
        *,
        tier: str = "merged",
    ) -> SessionProfileProduct | None:
        return await self.operations.get_session_profile_product(conversation_id, tier=tier)

    async def list_session_profile_products(
        self,
        query: SessionProfileProductQuery | None = None,
    ) -> list[SessionProfileProduct]:
        return await self.operations.list_session_profile_products(query)

    async def get_session_enrichment_product(
        self,
        conversation_id: str,
    ) -> SessionEnrichmentProduct | None:
        return await self.operations.get_session_enrichment_product(conversation_id)

    async def list_session_enrichment_products(
        self,
        query: SessionEnrichmentProductQuery | None = None,
    ) -> list[SessionEnrichmentProduct]:
        return await self.operations.list_session_enrichment_products(query)

    def filter(self) -> ConversationFilter:
        from polylogue.lib.filters import ConversationFilter
        from polylogue.storage.search_providers import create_vector_provider

        vector_provider = None
        with suppress(ValueError, ImportError):
            vector_provider = create_vector_provider(self.config)

        return ConversationFilter(self.repository, vector_provider=vector_provider)

    async def stats(self) -> ArchiveStats:
        return await self.operations.summary_stats()

    async def get_archive_stats(self) -> ArchiveStats:
        """Return archive summary statistics with an explicit API name."""
        return await self.stats()

    async def health_check(self) -> ReadinessReport:
        """Return the canonical archive readiness report."""
        from polylogue.readiness import get_readiness

        return get_readiness(self.config)

    async def rebuild_products(
        self,
        conversation_ids: Sequence[str] | None = None,
    ) -> SessionProductCounts:
        """Rebuild durable session-product read models."""
        return await self.operations.rebuild_session_products(conversation_ids=conversation_ids)

    async def resume_brief(
        self,
        session_id: str,
        *,
        related_limit: int = 6,
    ) -> ResumeBrief | None:
        """Build a compact handoff brief for an archived session."""
        return await self.operations.build_resume_brief(session_id, related_limit=related_limit)

    async def product_readiness_report(
        self,
        query: ProductReadinessQuery | None = None,
    ) -> ProductReadinessReport:
        """Return product materialization readiness for downstream consumers."""
        return await self.operations.get_product_readiness_report(query)

    async def export_product_bundle(
        self,
        request: ProductExportBundleRequest,
    ) -> ProductExportBundleResult:
        """Write a versioned archive-product export bundle."""
        return await self.operations.export_product_bundle(request)
