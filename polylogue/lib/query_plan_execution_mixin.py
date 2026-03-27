"""Retrieval and execution mixin methods for conversation plans."""

from __future__ import annotations

from typing import TYPE_CHECKING

from polylogue.lib.query_retrieval import (
    action_event_rows_ready,
    can_use_action_event_stats_with,
    candidate_record_query,
    candidate_record_query_for,
    fetch_record_query_for,
    search_limit,
    should_batch_post_filter_fetch,
    uses_action_read_model,
)

if TYPE_CHECKING:
    from polylogue.lib.models import Conversation, ConversationSummary
    from polylogue.storage.query_models import ConversationRecordQuery
    from polylogue.storage.repository import ConversationRepository


class QueryPlanExecutionMixin:
    def _candidate_record_query(self) -> tuple[ConversationRecordQuery, bool]:
        return candidate_record_query(self)

    def fetch_record_query(self) -> ConversationRecordQuery:
        record_query, _ = self._candidate_record_query()
        return record_query.with_limit(self.effective_fetch_limit())

    def _uses_action_read_model(self) -> bool:
        return uses_action_read_model(self)

    async def _action_event_rows_ready(self, repository: ConversationRepository) -> bool:
        return await action_event_rows_ready(self, repository)

    async def can_use_action_event_stats_with(self, repository: ConversationRepository) -> bool:
        return await can_use_action_event_stats_with(self, repository)

    async def _candidate_record_query_for(
        self,
        repository: ConversationRepository,
    ) -> tuple[ConversationRecordQuery, bool]:
        return await candidate_record_query_for(self, repository)

    async def fetch_record_query_for(self, repository: ConversationRepository) -> ConversationRecordQuery:
        return await fetch_record_query_for(self, repository)

    def _should_batch_post_filter_fetch(self) -> bool:
        return should_batch_post_filter_fetch(self)

    def _search_limit(self) -> int:
        return search_limit(self)

    async def list(self, repository: ConversationRepository) -> list[Conversation]:
        from polylogue.lib.query_plan_execution import list_for_plan

        return await list_for_plan(self, repository)

    async def list_summaries(self, repository: ConversationRepository) -> list[ConversationSummary]:
        from polylogue.lib.query_plan_execution import list_summaries_for_plan

        return await list_summaries_for_plan(self, repository)

    async def first(self, repository: ConversationRepository) -> Conversation | None:
        from polylogue.lib.query_plan_execution import first_for_plan

        return await first_for_plan(self, repository)

    async def count(self, repository: ConversationRepository) -> int:
        from polylogue.lib.query_plan_execution import count_for_plan

        return await count_for_plan(self, repository)

    async def delete(self, repository: ConversationRepository) -> int:
        from polylogue.lib.query_plan_execution import delete_for_plan

        return await delete_for_plan(self, repository)


__all__ = ["QueryPlanExecutionMixin"]
