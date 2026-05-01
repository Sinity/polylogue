"""Insight-oriented sync facade methods."""

from __future__ import annotations

from typing import TYPE_CHECKING

from polylogue.api.sync.bridge import run_coroutine_sync
from polylogue.insights.archive import (
    ArchiveDebtInsight,
    ArchiveDebtInsightQuery,
    CostRollupInsight,
    CostRollupInsightQuery,
    DaySessionSummaryInsight,
    DaySessionSummaryInsightQuery,
    ProviderAnalyticsInsight,
    ProviderAnalyticsInsightQuery,
    SessionCostInsight,
    SessionCostInsightQuery,
    SessionEnrichmentInsight,
    SessionEnrichmentInsightQuery,
    SessionPhaseInsight,
    SessionPhaseInsightQuery,
    SessionProfileInsight,
    SessionProfileInsightQuery,
    SessionTagRollupInsight,
    SessionTagRollupQuery,
    SessionWorkEventInsight,
    SessionWorkEventInsightQuery,
    WeekSessionSummaryInsight,
    WeekSessionSummaryInsightQuery,
    WorkThreadInsight,
    WorkThreadInsightQuery,
)
from polylogue.storage.insights.session.runtime import SessionInsightStatusSnapshot

if TYPE_CHECKING:
    from polylogue.api import Polylogue


class SyncInsightQueriesMixin:
    """Derived product query helpers for ``SyncPolylogue``."""

    _facade: Polylogue

    def get_session_product_status(self) -> SessionInsightStatusSnapshot:
        return run_coroutine_sync(self._facade.get_session_product_status())

    def get_session_profile_product(
        self,
        conversation_id: str,
        *,
        tier: str = "merged",
    ) -> SessionProfileInsight | None:
        return run_coroutine_sync(self._facade.get_session_profile_product(conversation_id, tier=tier))

    def list_session_profile_products(
        self,
        query: SessionProfileInsightQuery | None = None,
    ) -> list[SessionProfileInsight]:
        return run_coroutine_sync(self._facade.list_session_profile_products(query))

    def get_session_enrichment_product(self, conversation_id: str) -> SessionEnrichmentInsight | None:
        return run_coroutine_sync(self._facade.get_session_enrichment_product(conversation_id))

    def list_session_enrichment_products(
        self,
        query: SessionEnrichmentInsightQuery | None = None,
    ) -> list[SessionEnrichmentInsight]:
        return run_coroutine_sync(self._facade.list_session_enrichment_products(query))

    def list_session_tag_rollup_products(
        self,
        query: SessionTagRollupQuery | None = None,
    ) -> list[SessionTagRollupInsight]:
        return run_coroutine_sync(self._facade.list_session_tag_rollup_products(query))

    def get_session_work_event_products(self, conversation_id: str) -> list[SessionWorkEventInsight]:
        return run_coroutine_sync(self._facade.get_session_work_event_products(conversation_id))

    def list_session_work_event_products(
        self,
        query: SessionWorkEventInsightQuery | None = None,
    ) -> list[SessionWorkEventInsight]:
        return run_coroutine_sync(self._facade.list_session_work_event_products(query))

    def get_session_phase_products(self, conversation_id: str) -> list[SessionPhaseInsight]:
        return run_coroutine_sync(self._facade.get_session_phase_products(conversation_id))

    def list_session_phase_products(
        self,
        query: SessionPhaseInsightQuery | None = None,
    ) -> list[SessionPhaseInsight]:
        return run_coroutine_sync(self._facade.list_session_phase_products(query))

    def get_work_thread_product(self, thread_id: str) -> WorkThreadInsight | None:
        return run_coroutine_sync(self._facade.get_work_thread_product(thread_id))

    def list_work_thread_products(
        self,
        query: WorkThreadInsightQuery | None = None,
    ) -> list[WorkThreadInsight]:
        return run_coroutine_sync(self._facade.list_work_thread_products(query))

    def list_day_session_summary_products(
        self,
        query: DaySessionSummaryInsightQuery | None = None,
    ) -> list[DaySessionSummaryInsight]:
        return run_coroutine_sync(self._facade.list_day_session_summary_products(query))

    def list_week_session_summary_products(
        self,
        query: WeekSessionSummaryInsightQuery | None = None,
    ) -> list[WeekSessionSummaryInsight]:
        return run_coroutine_sync(self._facade.list_week_session_summary_products(query))

    def list_provider_analytics_products(
        self,
        query: ProviderAnalyticsInsightQuery | None = None,
    ) -> list[ProviderAnalyticsInsight]:
        return run_coroutine_sync(self._facade.list_provider_analytics_products(query))

    def list_session_cost_products(
        self,
        query: SessionCostInsightQuery | None = None,
    ) -> list[SessionCostInsight]:
        return run_coroutine_sync(self._facade.list_session_cost_products(query))

    def list_cost_rollup_products(
        self,
        query: CostRollupInsightQuery | None = None,
    ) -> list[CostRollupInsight]:
        return run_coroutine_sync(self._facade.list_cost_rollup_products(query))

    def list_archive_debt_products(
        self,
        query: ArchiveDebtInsightQuery | None = None,
    ) -> list[ArchiveDebtInsight]:
        return run_coroutine_sync(self._facade.list_archive_debt_products(query))


__all__ = ["SyncInsightQueriesMixin"]
