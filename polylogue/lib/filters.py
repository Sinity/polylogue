"""Fluent adapter over the canonical immutable conversation query plan."""

from __future__ import annotations

import builtins
from datetime import datetime
from typing import TYPE_CHECKING

from polylogue.lib.filter_builder import ConversationFilterBuilderMixin
from polylogue.lib.filter_types import SortField
from polylogue.lib.query_plan import ConversationQueryPlan
from polylogue.lib.query_plan_execution import (
    count_for_plan,
    delete_for_plan,
    first_for_plan,
    list_for_plan,
    list_summaries_for_plan,
)

if TYPE_CHECKING:
    from polylogue.lib.conversation_models import Conversation, ConversationSummary
    from polylogue.protocols import ConversationQueryRuntimeStore, VectorProvider


class ConversationFilter(ConversationFilterBuilderMixin):
    """Fluent query shell backed directly by the canonical execution plan."""

    def __init__(
        self,
        repository: ConversationQueryRuntimeStore,
        vector_provider: VectorProvider | None = None,
        *,
        query_plan: ConversationQueryPlan | None = None,
    ) -> None:
        self._repo = repository
        self._plan = query_plan or ConversationQueryPlan(vector_provider=vector_provider)

    @classmethod
    def from_query_plan(
        cls,
        repository: ConversationQueryRuntimeStore,
        query_plan: ConversationQueryPlan,
    ) -> ConversationFilter:
        return cls(repository, vector_provider=query_plan.vector_provider, query_plan=query_plan)

    @property
    def _since_date(self) -> datetime | None:
        return self._plan.since

    @property
    def _until_date(self) -> datetime | None:
        return self._plan.until

    @property
    def _continuation(self) -> bool | None:
        return self._plan.continuation

    @property
    def _sidechain(self) -> bool | None:
        return self._plan.sidechain

    @property
    def _has_branches(self) -> bool | None:
        return self._plan.has_branches

    def build_query_plan(self) -> ConversationQueryPlan:
        return self._plan

    def _sql_pushdown_params(self) -> dict[str, object]:
        return self._plan.sql_pushdown_params()

    def _has_post_filters(self) -> bool:
        return self._plan.has_post_filters()

    def _needs_content_loading(self) -> bool:
        return self._plan.needs_content_loading()

    def can_use_summaries(self) -> bool:
        return self._plan.can_use_summaries()

    def describe(self) -> list[str]:
        return self._plan.describe()

    async def list(self) -> builtins.list[Conversation]:
        return await list_for_plan(self._plan, self._repo)

    async def list_summaries(self) -> builtins.list[ConversationSummary]:
        return await list_summaries_for_plan(self._plan, self._repo)

    async def first(self) -> Conversation | None:
        return await first_for_plan(self._plan, self._repo)

    async def count(self) -> int:
        return await count_for_plan(self._plan, self._repo)

    async def delete(self) -> int:
        return await delete_for_plan(self._plan, self._repo)


__all__ = ["ConversationFilter", "SortField"]
