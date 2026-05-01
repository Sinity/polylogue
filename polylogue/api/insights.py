"""Durable product methods for the async Polylogue facade."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

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

if TYPE_CHECKING:

    class _InsightOperationsSurface(Protocol):
        async def list_session_tag_rollup_products(
            self,
            query: SessionTagRollupQuery | None = None,
        ) -> list[SessionTagRollupInsight]: ...

        async def get_session_work_event_products(
            self,
            conversation_id: str,
        ) -> list[SessionWorkEventInsight]: ...

        async def list_session_work_event_products(
            self,
            query: SessionWorkEventInsightQuery | None = None,
        ) -> list[SessionWorkEventInsight]: ...

        async def get_session_phase_products(
            self,
            conversation_id: str,
        ) -> list[SessionPhaseInsight]: ...

        async def list_session_phase_products(
            self,
            query: SessionPhaseInsightQuery | None = None,
        ) -> list[SessionPhaseInsight]: ...

        async def get_work_thread_product(self, thread_id: str) -> WorkThreadInsight | None: ...

        async def list_work_thread_products(
            self,
            query: WorkThreadInsightQuery | None = None,
        ) -> list[WorkThreadInsight]: ...

        async def list_day_session_summary_products(
            self,
            query: DaySessionSummaryInsightQuery | None = None,
        ) -> list[DaySessionSummaryInsight]: ...

        async def list_week_session_summary_products(
            self,
            query: WeekSessionSummaryInsightQuery | None = None,
        ) -> list[WeekSessionSummaryInsight]: ...

        async def list_provider_analytics_products(
            self,
            query: ProviderAnalyticsInsightQuery | None = None,
        ) -> list[ProviderAnalyticsInsight]: ...

        async def list_session_cost_products(
            self,
            query: SessionCostInsightQuery | None = None,
        ) -> list[SessionCostInsight]: ...

        async def list_cost_rollup_products(
            self,
            query: CostRollupInsightQuery | None = None,
        ) -> list[CostRollupInsight]: ...

        async def list_archive_debt_products(
            self,
            query: ArchiveDebtInsightQuery | None = None,
        ) -> list[ArchiveDebtInsight]: ...


class PolylogueInsightsMixin:
    if TYPE_CHECKING:

        @property
        def operations(self) -> _InsightOperationsSurface: ...

    async def list_session_tag_rollup_products(
        self,
        query: SessionTagRollupQuery | None = None,
    ) -> list[SessionTagRollupInsight]:
        return await self.operations.list_session_tag_rollup_products(query)

    async def get_session_work_event_products(
        self,
        conversation_id: str,
    ) -> list[SessionWorkEventInsight]:
        return await self.operations.get_session_work_event_products(conversation_id)

    async def list_session_work_event_products(
        self,
        query: SessionWorkEventInsightQuery | None = None,
    ) -> list[SessionWorkEventInsight]:
        return await self.operations.list_session_work_event_products(query)

    async def get_session_phase_products(
        self,
        conversation_id: str,
    ) -> list[SessionPhaseInsight]:
        return await self.operations.get_session_phase_products(conversation_id)

    async def list_session_phase_products(
        self,
        query: SessionPhaseInsightQuery | None = None,
    ) -> list[SessionPhaseInsight]:
        return await self.operations.list_session_phase_products(query)

    async def get_work_thread_product(self, thread_id: str) -> WorkThreadInsight | None:
        return await self.operations.get_work_thread_product(thread_id)

    async def list_work_thread_products(
        self,
        query: WorkThreadInsightQuery | None = None,
    ) -> list[WorkThreadInsight]:
        return await self.operations.list_work_thread_products(query)

    async def list_day_session_summary_products(
        self,
        query: DaySessionSummaryInsightQuery | None = None,
    ) -> list[DaySessionSummaryInsight]:
        return await self.operations.list_day_session_summary_products(query)

    async def list_week_session_summary_products(
        self,
        query: WeekSessionSummaryInsightQuery | None = None,
    ) -> list[WeekSessionSummaryInsight]:
        return await self.operations.list_week_session_summary_products(query)

    async def list_provider_analytics_products(
        self,
        query: ProviderAnalyticsInsightQuery | None = None,
    ) -> list[ProviderAnalyticsInsight]:
        return await self.operations.list_provider_analytics_products(query)

    async def list_session_cost_products(
        self,
        query: SessionCostInsightQuery | None = None,
    ) -> list[SessionCostInsight]:
        return await self.operations.list_session_cost_products(query)

    async def list_cost_rollup_products(
        self,
        query: CostRollupInsightQuery | None = None,
    ) -> list[CostRollupInsight]:
        return await self.operations.list_cost_rollup_products(query)

    async def list_archive_debt_products(
        self,
        query: ArchiveDebtInsightQuery | None = None,
    ) -> list[ArchiveDebtInsight]:
        return await self.operations.list_archive_debt_products(query)
