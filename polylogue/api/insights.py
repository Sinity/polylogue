"""Durable insight methods for the async Polylogue facade."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from polylogue.archive.session.session_profile import SessionProfile
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
    SessionPhaseInsight,
    SessionPhaseInsightQuery,
    SessionTagRollupInsight,
    SessionTagRollupQuery,
    SessionWorkEventInsight,
    SessionWorkEventInsightQuery,
    WeekSessionSummaryInsight,
    WeekSessionSummaryInsightQuery,
    WorkThreadInsight,
    WorkThreadInsightQuery,
)
from polylogue.insights.classification import SessionClassification, classify_session
from polylogue.insights.productivity import (
    ProductivityRollupInsight,
    ProductivityRollupInsightQuery,
)
from polylogue.insights.tool_usage import ToolUsageInsight, ToolUsageInsightQuery

if TYPE_CHECKING:

    class _InsightOperationsSurface(Protocol):
        async def list_session_tag_rollup_insights(
            self,
            query: SessionTagRollupQuery | None = None,
        ) -> list[SessionTagRollupInsight]: ...

        async def get_session_work_event_insights(
            self,
            conversation_id: str,
        ) -> list[SessionWorkEventInsight]: ...

        async def list_session_work_event_insights(
            self,
            query: SessionWorkEventInsightQuery | None = None,
        ) -> list[SessionWorkEventInsight]: ...

        async def get_session_phase_insights(
            self,
            conversation_id: str,
        ) -> list[SessionPhaseInsight]: ...

        async def list_session_phase_insights(
            self,
            query: SessionPhaseInsightQuery | None = None,
        ) -> list[SessionPhaseInsight]: ...

        async def get_work_thread_insight(self, thread_id: str) -> WorkThreadInsight | None: ...

        async def list_work_thread_insights(
            self,
            query: WorkThreadInsightQuery | None = None,
        ) -> list[WorkThreadInsight]: ...

        async def list_day_session_summary_insights(
            self,
            query: DaySessionSummaryInsightQuery | None = None,
        ) -> list[DaySessionSummaryInsight]: ...

        async def list_week_session_summary_insights(
            self,
            query: WeekSessionSummaryInsightQuery | None = None,
        ) -> list[WeekSessionSummaryInsight]: ...

        async def list_provider_analytics_insights(
            self,
            query: ProviderAnalyticsInsightQuery | None = None,
        ) -> list[ProviderAnalyticsInsight]: ...

        async def list_tool_usage_insights(
            self,
            query: ToolUsageInsightQuery | None = None,
        ) -> list[ToolUsageInsight]: ...

        async def list_productivity_rollup_insights(
            self,
            query: ProductivityRollupInsightQuery | None = None,
        ) -> list[ProductivityRollupInsight]: ...

        async def list_session_cost_insights(
            self,
            query: SessionCostInsightQuery | None = None,
        ) -> list[SessionCostInsight]: ...

        async def list_cost_rollup_insights(
            self,
            query: CostRollupInsightQuery | None = None,
        ) -> list[CostRollupInsight]: ...

        async def list_archive_debt_insights(
            self,
            query: ArchiveDebtInsightQuery | None = None,
        ) -> list[ArchiveDebtInsight]: ...


class _RepositorySurface(Protocol):
    async def get_session_profile(self, conversation_id: str) -> SessionProfile | None: ...


class PolylogueInsightsMixin:
    if TYPE_CHECKING:

        @property
        def operations(self) -> _InsightOperationsSurface: ...

        @property
        def repository(self) -> _RepositorySurface: ...

    async def list_session_tag_rollup_insights(
        self,
        query: SessionTagRollupQuery | None = None,
    ) -> list[SessionTagRollupInsight]:
        return await self.operations.list_session_tag_rollup_insights(query)

    async def get_session_work_event_insights(
        self,
        conversation_id: str,
    ) -> list[SessionWorkEventInsight]:
        return await self.operations.get_session_work_event_insights(conversation_id)

    async def list_session_work_event_insights(
        self,
        query: SessionWorkEventInsightQuery | None = None,
    ) -> list[SessionWorkEventInsight]:
        return await self.operations.list_session_work_event_insights(query)

    async def get_session_phase_insights(
        self,
        conversation_id: str,
    ) -> list[SessionPhaseInsight]:
        return await self.operations.get_session_phase_insights(conversation_id)

    async def list_session_phase_insights(
        self,
        query: SessionPhaseInsightQuery | None = None,
    ) -> list[SessionPhaseInsight]:
        return await self.operations.list_session_phase_insights(query)

    async def get_work_thread_insight(self, thread_id: str) -> WorkThreadInsight | None:
        return await self.operations.get_work_thread_insight(thread_id)

    async def list_work_thread_insights(
        self,
        query: WorkThreadInsightQuery | None = None,
    ) -> list[WorkThreadInsight]:
        return await self.operations.list_work_thread_insights(query)

    async def list_day_session_summary_insights(
        self,
        query: DaySessionSummaryInsightQuery | None = None,
    ) -> list[DaySessionSummaryInsight]:
        return await self.operations.list_day_session_summary_insights(query)

    async def list_week_session_summary_insights(
        self,
        query: WeekSessionSummaryInsightQuery | None = None,
    ) -> list[WeekSessionSummaryInsight]:
        return await self.operations.list_week_session_summary_insights(query)

    async def list_provider_analytics_insights(
        self,
        query: ProviderAnalyticsInsightQuery | None = None,
    ) -> list[ProviderAnalyticsInsight]:
        return await self.operations.list_provider_analytics_insights(query)

    async def list_tool_usage_insights(
        self,
        query: ToolUsageInsightQuery | None = None,
    ) -> list[ToolUsageInsight]:
        return await self.operations.list_tool_usage_insights(query)

    async def list_productivity_rollup_insights(
        self,
        query: ProductivityRollupInsightQuery | None = None,
    ) -> list[ProductivityRollupInsight]:
        return await self.operations.list_productivity_rollup_insights(query)

    async def list_session_cost_insights(
        self,
        query: SessionCostInsightQuery | None = None,
    ) -> list[SessionCostInsight]:
        return await self.operations.list_session_cost_insights(query)

    async def list_cost_rollup_insights(
        self,
        query: CostRollupInsightQuery | None = None,
    ) -> list[CostRollupInsight]:
        return await self.operations.list_cost_rollup_insights(query)

    async def list_archive_debt_insights(
        self,
        query: ArchiveDebtInsightQuery | None = None,
    ) -> list[ArchiveDebtInsight]:
        return await self.operations.list_archive_debt_insights(query)

    async def classify_session(self, conversation_id: str) -> SessionClassification | None:
        """Classify a session into the typed :class:`SessionCategory` taxonomy.

        Computed on-the-fly from the hydrated :class:`SessionProfile`. The
        classifier is pure and deterministic (see
        :mod:`polylogue.insights.classification` for details). Returns
        ``None`` when no profile exists for ``conversation_id``.
        """

        profile = await self.repository.get_session_profile(conversation_id)
        if profile is None:
            return None
        return classify_session(profile)
