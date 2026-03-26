"""Aggregate and analytics archive-product mixins."""

from __future__ import annotations

from polylogue.archive_product_builders import (
    aggregate_day_session_summary_products,
    aggregate_session_tag_rollup_products,
    aggregate_week_session_summary_products,
)
from polylogue.archive_products import (
    DaySessionSummaryProduct,
    DaySessionSummaryProductQuery,
    ProviderAnalyticsProduct,
    ProviderAnalyticsProductQuery,
    SessionTagRollupProduct,
    SessionTagRollupQuery,
    WeekSessionSummaryProduct,
    WeekSessionSummaryProductQuery,
)


def provider_analytics_product(row) -> ProviderAnalyticsProduct:
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


class ArchiveProductAggregateMixin:
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


__all__ = ["ArchiveProductAggregateMixin", "provider_analytics_product"]
