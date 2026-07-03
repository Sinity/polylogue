"""Insight-oriented sync facade methods."""

from __future__ import annotations

from typing import TYPE_CHECKING

from polylogue.api.sync.bridge import run_coroutine_sync
from polylogue.insights.archive import (
    ArchiveCoverageInsight,
    ArchiveCoverageInsightQuery,
    ArchiveDebtInsight,
    ArchiveDebtInsightQuery,
    CostRollupInsight,
    CostRollupInsightQuery,
    SessionCostInsight,
    SessionCostInsightQuery,
    SessionPhaseInsight,
    SessionPhaseInsightQuery,
    SessionProfileInsight,
    SessionProfileInsightQuery,
    SessionTagRollupInsight,
    SessionTagRollupQuery,
    SessionWorkEventInsight,
    SessionWorkEventInsightQuery,
    ThreadInsight,
    ThreadInsightQuery,
    UsageTimelineInsight,
    UsageTimelineInsightQuery,
)
from polylogue.insights.tool_usage import ToolUsageInsight, ToolUsageInsightQuery
from polylogue.storage.insights.session.runtime import SessionInsightStatusSnapshot

if TYPE_CHECKING:
    from polylogue.api import Polylogue
    from polylogue.insights.readiness import InsightReadinessQuery, InsightReadinessReport


class SyncInsightQueriesMixin:
    """Derived insight query helpers for ``SyncPolylogue``."""

    _facade: Polylogue

    def get_session_insight_status(self) -> SessionInsightStatusSnapshot:
        return run_coroutine_sync(self._facade.get_session_insight_status())

    def get_session_profile_insight(
        self,
        session_id: str,
        *,
        tier: str = "merged",
    ) -> SessionProfileInsight | None:
        return run_coroutine_sync(self._facade.get_session_profile_insight(session_id, tier=tier))

    def list_session_profile_insights(
        self,
        query: SessionProfileInsightQuery | None = None,
    ) -> list[SessionProfileInsight]:
        return run_coroutine_sync(self._facade.list_session_profile_insights(query))

    def list_session_tag_rollup_insights(
        self,
        query: SessionTagRollupQuery | None = None,
    ) -> list[SessionTagRollupInsight]:
        return run_coroutine_sync(self._facade.list_session_tag_rollup_insights(query))

    def get_session_work_event_insights(self, session_id: str) -> list[SessionWorkEventInsight]:
        return run_coroutine_sync(self._facade.get_session_work_event_insights(session_id))

    def list_session_work_event_insights(
        self,
        query: SessionWorkEventInsightQuery | None = None,
    ) -> list[SessionWorkEventInsight]:
        return run_coroutine_sync(self._facade.list_session_work_event_insights(query))

    def get_session_phase_insights(self, session_id: str) -> list[SessionPhaseInsight]:
        return run_coroutine_sync(self._facade.get_session_phase_insights(session_id))

    def list_session_phase_insights(
        self,
        query: SessionPhaseInsightQuery | None = None,
    ) -> list[SessionPhaseInsight]:
        return run_coroutine_sync(self._facade.list_session_phase_insights(query))

    def get_thread_insight(self, thread_id: str) -> ThreadInsight | None:
        return run_coroutine_sync(self._facade.get_thread_insight(thread_id))

    def list_thread_insights(
        self,
        query: ThreadInsightQuery | None = None,
    ) -> list[ThreadInsight]:
        return run_coroutine_sync(self._facade.list_thread_insights(query))

    def list_archive_coverage_insights(
        self,
        query: ArchiveCoverageInsightQuery | None = None,
    ) -> list[ArchiveCoverageInsight]:
        return run_coroutine_sync(self._facade.list_archive_coverage_insights(query))

    def list_tool_usage_insights(
        self,
        query: ToolUsageInsightQuery | None = None,
    ) -> list[ToolUsageInsight]:
        return run_coroutine_sync(self._facade.list_tool_usage_insights(query))

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

    def list_usage_timeline_insights(
        self,
        query: UsageTimelineInsightQuery | None = None,
    ) -> list[UsageTimelineInsight]:
        return run_coroutine_sync(self._facade.list_usage_timeline_insights(query))

    def list_archive_debt_insights(
        self,
        query: ArchiveDebtInsightQuery | None = None,
    ) -> list[ArchiveDebtInsight]:
        return run_coroutine_sync(self._facade.list_archive_debt_insights(query))

    def insight_readiness_report(
        self,
        query: InsightReadinessQuery | None = None,
    ) -> InsightReadinessReport:
        return run_coroutine_sync(self._facade.insight_readiness_report(query))


__all__ = ["SyncInsightQueriesMixin"]
