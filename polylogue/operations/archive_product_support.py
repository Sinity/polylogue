"""Archive-product retrieval and aggregation support."""

from __future__ import annotations

from collections.abc import Iterable

from polylogue.archive_product_builders import (
    aggregate_day_session_summary_products,
    aggregate_session_tag_rollup_products,
    aggregate_week_session_summary_products,
)
from polylogue.archive_products import (
    ArchiveDebtProduct,
    ArchiveDebtProductQuery,
    ArchiveDebtTargetLineage,
    DaySessionSummaryProduct,
    DaySessionSummaryProductQuery,
    MaintenanceRunProduct,
    MaintenanceRunProductQuery,
    ProviderAnalyticsProduct,
    ProviderAnalyticsProductQuery,
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
from polylogue.storage.archive_debt import collect_archive_debt_statuses_sync
from polylogue.storage.backends.connection import connection_context
from polylogue.storage.store import MaintenanceRunRecord


def _provider_analytics_product(row) -> ProviderAnalyticsProduct:
    conversation_count = row["conversation_count"]
    user_message_count = row["user_message_count"]
    assistant_message_count = row["assistant_message_count"]
    user_word_sum = row["user_word_sum"] or 0
    assistant_word_sum = row["assistant_word_sum"] or 0
    tool_use_percentage = (
        (row["conversations_with_tools"] / conversation_count) * 100
        if conversation_count > 0
        else 0.0
    )
    thinking_percentage = (
        (row["conversations_with_thinking"] / conversation_count) * 100
        if conversation_count > 0
        else 0.0
    )
    return ProviderAnalyticsProduct(
        provider_name=row["provider_name"] or "unknown",
        conversation_count=conversation_count,
        message_count=row["message_count"],
        user_message_count=user_message_count,
        assistant_message_count=assistant_message_count,
        avg_messages_per_conversation=(
            row["message_count"] / conversation_count if conversation_count > 0 else 0.0
        ),
        avg_user_words=(user_word_sum / user_message_count if user_message_count > 0 else 0.0),
        avg_assistant_words=(
            assistant_word_sum / assistant_message_count if assistant_message_count > 0 else 0.0
        ),
        tool_use_count=row["tool_use_count"],
        thinking_count=row["thinking_count"],
        total_conversations_with_tools=row["conversations_with_tools"],
        total_conversations_with_thinking=row["conversations_with_thinking"],
        tool_use_percentage=tool_use_percentage,
        thinking_percentage=thinking_percentage,
    )


def _target_lineage(records: Iterable[MaintenanceRunRecord]) -> dict[str, ArchiveDebtTargetLineage]:
    lineage_by_target: dict[str, ArchiveDebtTargetLineage] = {}
    for record in records:
        if not record.target_names:
            continue
        for target_name in record.target_names:
            lineage = lineage_by_target.get(target_name)
            if lineage is None:
                lineage = ArchiveDebtTargetLineage()
            if lineage.latest_run_at is None:
                lineage = lineage.model_copy(update={"latest_run_at": record.executed_at, "latest_mode": record.mode})
            if record.preview and lineage.latest_preview_at is None:
                lineage = lineage.model_copy(update={"latest_preview_at": record.executed_at})
            if not record.preview and lineage.latest_apply_at is None:
                lineage = lineage.model_copy(update={"latest_apply_at": record.executed_at})
            if not record.preview and record.success and lineage.latest_successful_apply_at is None:
                lineage = lineage.model_copy(update={"latest_successful_apply_at": record.executed_at})
            lineage_by_target[target_name] = lineage
    return lineage_by_target


def _lineage_governance_stage(*, issue_count: int, lineage: ArchiveDebtTargetLineage | None) -> str:
    if issue_count <= 0:
        return "validated" if lineage and lineage.latest_successful_apply_at else "healthy"
    if lineage is None:
        return "unreviewed"
    if lineage.latest_successful_apply_at:
        return "applied"
    if lineage.latest_preview_at:
        return "previewed"
    return "unreviewed"


class ArchiveProductMixin:
    """Versioned archive-product retrieval methods."""

    async def get_session_profile_product(
        self,
        conversation_id: str,
        *,
        tier: str = "merged",
    ) -> SessionProfileProduct | None:
        record = await self.repository.get_session_profile_record(conversation_id)
        return SessionProfileProduct.from_record(record, tier=tier) if record is not None else None

    async def list_session_profile_products(
        self,
        query: SessionProfileProductQuery | None = None,
    ) -> list[SessionProfileProduct]:
        request = query or SessionProfileProductQuery()
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

    async def list_session_tag_rollup_products(
        self,
        query: SessionTagRollupQuery | None = None,
    ) -> list[SessionTagRollupProduct]:
        request = query or SessionTagRollupQuery()
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

    async def get_session_work_event_products(
        self,
        conversation_id: str,
    ) -> list[SessionWorkEventProduct]:
        records = await self.repository.get_session_work_event_records(conversation_id)
        return [SessionWorkEventProduct.from_record(record) for record in records]

    async def list_session_work_event_products(
        self,
        query: SessionWorkEventProductQuery | None = None,
    ) -> list[SessionWorkEventProduct]:
        request = query or SessionWorkEventProductQuery()
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
        records = await self.repository.get_session_phase_records(conversation_id)
        return [SessionPhaseProduct.from_record(record) for record in records]

    async def list_session_phase_products(
        self,
        query: SessionPhaseProductQuery | None = None,
    ) -> list[SessionPhaseProduct]:
        request = query or SessionPhaseProductQuery()
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
        record = await self.repository.get_work_thread_record(thread_id)
        return WorkThreadProduct.from_record(record) if record is not None else None

    async def list_work_thread_products(
        self,
        query: WorkThreadProductQuery | None = None,
    ) -> list[WorkThreadProduct]:
        request = query or WorkThreadProductQuery()
        records = await self.repository.list_work_thread_records(
            since=request.since,
            until=request.until,
            limit=request.limit,
            offset=request.offset,
            query=request.query,
        )
        return [WorkThreadProduct.from_record(record) for record in records]

    async def list_day_session_summary_products(
        self,
        query: DaySessionSummaryProductQuery | None = None,
    ) -> list[DaySessionSummaryProduct]:
        request = query or DaySessionSummaryProductQuery()
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

    async def list_maintenance_run_products(
        self,
        query: MaintenanceRunProductQuery | None = None,
    ) -> list[MaintenanceRunProduct]:
        request = query or MaintenanceRunProductQuery()
        records = await self.repository.list_maintenance_runs(limit=request.limit)
        return [MaintenanceRunProduct.from_record(record) for record in records]

    async def list_provider_analytics_products(
        self,
        query: ProviderAnalyticsProductQuery | None = None,
    ) -> list[ProviderAnalyticsProduct]:
        rows = await self.backend.queries.get_provider_metrics_rows()
        products = [_provider_analytics_product(row) for row in rows]
        request = query or ProviderAnalyticsProductQuery()
        if request.provider:
            products = [product for product in products if product.provider_name == request.provider]
        if request.offset:
            products = products[request.offset :]
        if request.limit is not None:
            products = products[: request.limit]
        return products

    async def list_archive_debt_products(
        self,
        query: ArchiveDebtProductQuery | None = None,
    ) -> list[ArchiveDebtProduct]:
        request = query or ArchiveDebtProductQuery()
        maintenance_records = await self.repository.list_maintenance_runs(limit=100)
        lineage_by_target = _target_lineage(maintenance_records)
        with connection_context(self.config.db_path) as conn:
            statuses = collect_archive_debt_statuses_sync(conn)
        products = [
            ArchiveDebtProduct.from_status(
                status,
                governance_stage=_lineage_governance_stage(
                    issue_count=status.issue_count,
                    lineage=lineage_by_target.get(status.maintenance_target),
                ),
                lineage=lineage_by_target.get(status.maintenance_target),
            )
            for status in statuses.values()
        ]
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


__all__ = ["ArchiveProductMixin"]
