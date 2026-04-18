"""Durable product methods for the async Polylogue facade."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from polylogue.archive_products import (
    ArchiveDebtProduct,
    ArchiveDebtProductQuery,
    DaySessionSummaryProduct,
    DaySessionSummaryProductQuery,
    ProviderAnalyticsProduct,
    ProviderAnalyticsProductQuery,
    SessionPhaseProduct,
    SessionPhaseProductQuery,
    SessionTagRollupProduct,
    SessionTagRollupQuery,
    SessionWorkEventProduct,
    SessionWorkEventProductQuery,
    WeekSessionSummaryProduct,
    WeekSessionSummaryProductQuery,
    WorkThreadProduct,
    WorkThreadProductQuery,
)

if TYPE_CHECKING:

    class _ProductOperationsSurface(Protocol):
        async def list_session_tag_rollup_products(
            self,
            query: SessionTagRollupQuery | None = None,
        ) -> list[SessionTagRollupProduct]: ...

        async def get_session_work_event_products(
            self,
            conversation_id: str,
        ) -> list[SessionWorkEventProduct]: ...

        async def list_session_work_event_products(
            self,
            query: SessionWorkEventProductQuery | None = None,
        ) -> list[SessionWorkEventProduct]: ...

        async def get_session_phase_products(
            self,
            conversation_id: str,
        ) -> list[SessionPhaseProduct]: ...

        async def list_session_phase_products(
            self,
            query: SessionPhaseProductQuery | None = None,
        ) -> list[SessionPhaseProduct]: ...

        async def get_work_thread_product(self, thread_id: str) -> WorkThreadProduct | None: ...

        async def list_work_thread_products(
            self,
            query: WorkThreadProductQuery | None = None,
        ) -> list[WorkThreadProduct]: ...

        async def list_day_session_summary_products(
            self,
            query: DaySessionSummaryProductQuery | None = None,
        ) -> list[DaySessionSummaryProduct]: ...

        async def list_week_session_summary_products(
            self,
            query: WeekSessionSummaryProductQuery | None = None,
        ) -> list[WeekSessionSummaryProduct]: ...

        async def list_provider_analytics_products(
            self,
            query: ProviderAnalyticsProductQuery | None = None,
        ) -> list[ProviderAnalyticsProduct]: ...

        async def list_archive_debt_products(
            self,
            query: ArchiveDebtProductQuery | None = None,
        ) -> list[ArchiveDebtProduct]: ...


class PolylogueProductsMixin:
    if TYPE_CHECKING:

        @property
        def operations(self) -> _ProductOperationsSurface: ...

    async def list_session_tag_rollup_products(
        self,
        query: SessionTagRollupQuery | None = None,
    ) -> list[SessionTagRollupProduct]:
        return await self.operations.list_session_tag_rollup_products(query)

    async def get_session_work_event_products(
        self,
        conversation_id: str,
    ) -> list[SessionWorkEventProduct]:
        return await self.operations.get_session_work_event_products(conversation_id)

    async def list_session_work_event_products(
        self,
        query: SessionWorkEventProductQuery | None = None,
    ) -> list[SessionWorkEventProduct]:
        return await self.operations.list_session_work_event_products(query)

    async def get_session_phase_products(
        self,
        conversation_id: str,
    ) -> list[SessionPhaseProduct]:
        return await self.operations.get_session_phase_products(conversation_id)

    async def list_session_phase_products(
        self,
        query: SessionPhaseProductQuery | None = None,
    ) -> list[SessionPhaseProduct]:
        return await self.operations.list_session_phase_products(query)

    async def get_work_thread_product(self, thread_id: str) -> WorkThreadProduct | None:
        return await self.operations.get_work_thread_product(thread_id)

    async def list_work_thread_products(
        self,
        query: WorkThreadProductQuery | None = None,
    ) -> list[WorkThreadProduct]:
        return await self.operations.list_work_thread_products(query)

    async def list_day_session_summary_products(
        self,
        query: DaySessionSummaryProductQuery | None = None,
    ) -> list[DaySessionSummaryProduct]:
        return await self.operations.list_day_session_summary_products(query)

    async def list_week_session_summary_products(
        self,
        query: WeekSessionSummaryProductQuery | None = None,
    ) -> list[WeekSessionSummaryProduct]:
        return await self.operations.list_week_session_summary_products(query)

    async def list_provider_analytics_products(
        self,
        query: ProviderAnalyticsProductQuery | None = None,
    ) -> list[ProviderAnalyticsProduct]:
        return await self.operations.list_provider_analytics_products(query)

    async def list_archive_debt_products(
        self,
        query: ArchiveDebtProductQuery | None = None,
    ) -> list[ArchiveDebtProduct]:
        return await self.operations.list_archive_debt_products(query)
