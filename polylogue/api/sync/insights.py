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
    """Derived insight query helpers for ``SyncPolylogue``."""

    _facade: Polylogue

    def get_session_insight_status(self) -> SessionInsightStatusSnapshot:
        return run_coroutine_sync(self._facade.get_session_insight_status())

    def get_session_profile_insight(
        self,
        conversation_id: str,
        *,
        tier: str = "merged",
    ) -> SessionProfileInsight | None:
        return run_coroutine_sync(self._facade.get_session_profile_insight(conversation_id, tier=tier))

    def list_session_profile_insights(
        self,
        query: SessionProfileInsightQuery | None = None,
    ) -> list[SessionProfileInsight]:
        return run_coroutine_sync(self._facade.list_session_profile_insights(query))

    def get_session_enrichment_insight(self, conversation_id: str) -> SessionEnrichmentInsight | None:
        return run_coroutine_sync(self._facade.get_session_enrichment_insight(conversation_id))

    def list_session_enrichment_insights(
        self,
        query: SessionEnrichmentInsightQuery | None = None,
    ) -> list[SessionEnrichmentInsight]:
        return run_coroutine_sync(self._facade.list_session_enrichment_insights(query))

    def list_session_tag_rollup_insights(
        self,
        query: SessionTagRollupQuery | None = None,
    ) -> list[SessionTagRollupInsight]:
        return run_coroutine_sync(self._facade.list_session_tag_rollup_insights(query))

    def get_session_work_event_insights(self, conversation_id: str) -> list[SessionWorkEventInsight]:
        return run_coroutine_sync(self._facade.get_session_work_event_insights(conversation_id))

    def list_session_work_event_insights(
        self,
        query: SessionWorkEventInsightQuery | None = None,
    ) -> list[SessionWorkEventInsight]:
        return run_coroutine_sync(self._facade.list_session_work_event_insights(query))

    def get_session_phase_insights(self, conversation_id: str) -> list[SessionPhaseInsight]:
        return run_coroutine_sync(self._facade.get_session_phase_insights(conversation_id))

    def list_session_phase_insights(
        self,
        query: SessionPhaseInsightQuery | None = None,
    ) -> list[SessionPhaseInsight]:
        return run_coroutine_sync(self._facade.list_session_phase_insights(query))

    def get_work_thread_insight(self, thread_id: str) -> WorkThreadInsight | None:
        return run_coroutine_sync(self._facade.get_work_thread_insight(thread_id))

    def list_work_thread_insights(
        self,
        query: WorkThreadInsightQuery | None = None,
    ) -> list[WorkThreadInsight]:
        return run_coroutine_sync(self._facade.list_work_thread_insights(query))

    def list_day_session_summary_insights(
        self,
        query: DaySessionSummaryInsightQuery | None = None,
    ) -> list[DaySessionSummaryInsight]:
        return run_coroutine_sync(self._facade.list_day_session_summary_insights(query))

    def list_week_session_summary_insights(
        self,
        query: WeekSessionSummaryInsightQuery | None = None,
    ) -> list[WeekSessionSummaryInsight]:
        return run_coroutine_sync(self._facade.list_week_session_summary_insights(query))

    def list_provider_analytics_insights(
        self,
        query: ProviderAnalyticsInsightQuery | None = None,
    ) -> list[ProviderAnalyticsInsight]:
        return run_coroutine_sync(self._facade.list_provider_analytics_insights(query))

    def list_session_cost_insights(
        self,
        query: SessionCostInsightQuery | None = None,
    ) -> list[SessionCostInsight]:
        return run_coroutine_sync(self._facade.list_session_cost_insights(query))

    def list_cost_rollup_insights(
        self,
        query: CostRollupInsightQuery | None = None,
    ) -> list[CostRollupInsight]:
        return run_coroutine_sync(self._facade.list_cost_rollup_insights(query))

    def list_archive_debt_insights(
        self,
        query: ArchiveDebtInsightQuery | None = None,
    ) -> list[ArchiveDebtInsight]:
        return run_coroutine_sync(self._facade.list_archive_debt_insights(query))


__all__ = ["SyncInsightQueriesMixin"]
