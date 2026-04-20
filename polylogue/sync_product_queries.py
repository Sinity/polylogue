"""Product-oriented sync facade methods."""

from __future__ import annotations

from typing import TYPE_CHECKING

from polylogue.archive_products import (
    ArchiveDebtProduct,
    ArchiveDebtProductQuery,
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
from polylogue.storage.session_product_runtime import SessionProductStatusSnapshot
from polylogue.sync_bridge import run_coroutine_sync

if TYPE_CHECKING:
    from polylogue.facade import Polylogue


class SyncProductQueriesMixin:
    """Derived product query helpers for ``SyncPolylogue``."""

    _facade: Polylogue

    def get_session_product_status(self) -> SessionProductStatusSnapshot:
        return run_coroutine_sync(self._facade.get_session_product_status())

    def get_session_profile_product(
        self,
        conversation_id: str,
        *,
        tier: str = "merged",
    ) -> SessionProfileProduct | None:
        return run_coroutine_sync(self._facade.get_session_profile_product(conversation_id, tier=tier))

    def list_session_profile_products(
        self,
        query: SessionProfileProductQuery | None = None,
    ) -> list[SessionProfileProduct]:
        return run_coroutine_sync(self._facade.list_session_profile_products(query))

    def get_session_enrichment_product(self, conversation_id: str) -> SessionEnrichmentProduct | None:
        return run_coroutine_sync(self._facade.get_session_enrichment_product(conversation_id))

    def list_session_enrichment_products(
        self,
        query: SessionEnrichmentProductQuery | None = None,
    ) -> list[SessionEnrichmentProduct]:
        return run_coroutine_sync(self._facade.list_session_enrichment_products(query))

    def list_session_tag_rollup_products(
        self,
        query: SessionTagRollupQuery | None = None,
    ) -> list[SessionTagRollupProduct]:
        return run_coroutine_sync(self._facade.list_session_tag_rollup_products(query))

    def get_session_work_event_products(self, conversation_id: str) -> list[SessionWorkEventProduct]:
        return run_coroutine_sync(self._facade.get_session_work_event_products(conversation_id))

    def list_session_work_event_products(
        self,
        query: SessionWorkEventProductQuery | None = None,
    ) -> list[SessionWorkEventProduct]:
        return run_coroutine_sync(self._facade.list_session_work_event_products(query))

    def get_session_phase_products(self, conversation_id: str) -> list[SessionPhaseProduct]:
        return run_coroutine_sync(self._facade.get_session_phase_products(conversation_id))

    def list_session_phase_products(
        self,
        query: SessionPhaseProductQuery | None = None,
    ) -> list[SessionPhaseProduct]:
        return run_coroutine_sync(self._facade.list_session_phase_products(query))

    def get_work_thread_product(self, thread_id: str) -> WorkThreadProduct | None:
        return run_coroutine_sync(self._facade.get_work_thread_product(thread_id))

    def list_work_thread_products(
        self,
        query: WorkThreadProductQuery | None = None,
    ) -> list[WorkThreadProduct]:
        return run_coroutine_sync(self._facade.list_work_thread_products(query))

    def list_day_session_summary_products(
        self,
        query: DaySessionSummaryProductQuery | None = None,
    ) -> list[DaySessionSummaryProduct]:
        return run_coroutine_sync(self._facade.list_day_session_summary_products(query))

    def list_week_session_summary_products(
        self,
        query: WeekSessionSummaryProductQuery | None = None,
    ) -> list[WeekSessionSummaryProduct]:
        return run_coroutine_sync(self._facade.list_week_session_summary_products(query))

    def list_provider_analytics_products(
        self,
        query: ProviderAnalyticsProductQuery | None = None,
    ) -> list[ProviderAnalyticsProduct]:
        return run_coroutine_sync(self._facade.list_provider_analytics_products(query))

    def list_archive_debt_products(
        self,
        query: ArchiveDebtProductQuery | None = None,
    ) -> list[ArchiveDebtProduct]:
        return run_coroutine_sync(self._facade.list_archive_debt_products(query))


__all__ = ["SyncProductQueriesMixin"]
