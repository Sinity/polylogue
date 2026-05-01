"""Canonical immutable conversation-query plan model."""

from __future__ import annotations

import builtins
from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import datetime
from typing import TYPE_CHECKING, TypeVar

from polylogue.archive.query.fields import (
    SqlPushdownParams,
    conversation_record_query_for_plan,
    sql_pushdown_params_for_plan,
)
from polylogue.archive.query.plan_description import describe_plan, effective_fetch_limit, plan_has_filters
from polylogue.archive.query.retrieval import (
    action_event_rows_ready,
    can_use_action_event_stats_with,
    candidate_record_query,
    candidate_record_query_for,
    fetch_record_query_for,
    search_limit,
    should_batch_post_filter_fetch,
    uses_action_read_model,
)
from polylogue.archive.query.runtime import (
    apply_common_filters,
    apply_full_filters,
    matches_action_sequence,
    matches_action_terms,
    matches_action_text_terms,
    matches_referenced_path,
    matches_tool_terms,
    plan_can_count_in_sql,
    plan_can_use_action_event_stats,
    plan_has_post_filters,
    plan_needs_content_loading,
)
from polylogue.archive.query.sorting import SortKey, finalize_results, sort_conversations, sort_generic, sort_summaries
from polylogue.archive.query.support import conversation_has_branches
from polylogue.storage.query_models import ConversationRecordQuery
from polylogue.types import Provider

if TYPE_CHECKING:
    from polylogue.archive.filter.types import SortField
    from polylogue.archive.query.runtime_filters import FilterableConversationLike
    from polylogue.lib.models import Conversation, ConversationSummary
    from polylogue.protocols import ConversationQueryRuntimeStore, VectorProvider

_T = TypeVar("_T")
_FilterableT = TypeVar("_FilterableT", bound="FilterableConversationLike")


# ---------------------------------------------------------------------------
# Record-query translation helpers
# ---------------------------------------------------------------------------


def plan_record_query(plan: ConversationQueryPlan) -> ConversationRecordQuery:
    return conversation_record_query_for_plan(plan)


def plan_sql_pushdown_params(plan: ConversationQueryPlan) -> SqlPushdownParams:
    return sql_pushdown_params_for_plan(plan)


# ---------------------------------------------------------------------------
# Query plan dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ConversationQueryPlan:
    """Canonical immutable execution state for conversation selection."""

    query_terms: tuple[str, ...] = ()
    contains_terms: tuple[str, ...] = ()
    negative_terms: tuple[str, ...] = ()
    retrieval_lane: str = "auto"
    referenced_path: tuple[str, ...] = ()
    cwd_prefix: str | None = None
    action_terms: tuple[str, ...] = ()
    excluded_action_terms: tuple[str, ...] = ()
    action_sequence: tuple[str, ...] = ()
    action_text_terms: tuple[str, ...] = ()
    tool_terms: tuple[str, ...] = ()
    excluded_tool_terms: tuple[str, ...] = ()
    providers: tuple[Provider | str, ...] = ()
    excluded_providers: tuple[Provider | str, ...] = ()
    repo_names: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()
    excluded_tags: tuple[str, ...] = ()
    has_types: tuple[str, ...] = ()
    title: str | None = None
    conversation_id: str | None = None
    parent_id: str | None = None
    since: datetime | None = None
    until: datetime | None = None
    sort: SortField = "date"
    reverse: bool = False
    limit: int | None = None
    sample: int | None = None
    similar_text: str | None = None
    predicates: tuple[Callable[[Conversation], bool], ...] = ()
    continuation: bool | None = None
    sidechain: bool | None = None
    root: bool | None = None
    has_branches: bool | None = None
    filter_has_tool_use: bool = False
    filter_has_thinking: bool = False
    filter_has_paste: bool = False
    typed_only: bool = False
    min_messages: int | None = None
    max_messages: int | None = None
    min_words: int | None = None
    since_session_id: str | None = None
    message_type: str | None = None
    offset: int = 0
    vector_provider: VectorProvider | None = None

    # -- Description / record-query methods (was QueryPlanDescriptionMixin) --

    @property
    def fts_terms(self) -> tuple[str, ...]:
        return self.query_terms + self.contains_terms

    @property
    def sql_pushed(self) -> bool:
        return not self.fts_terms and self.conversation_id is None

    @property
    def record_query(self) -> ConversationRecordQuery:
        return plan_record_query(self)

    def sql_pushdown_params(self) -> SqlPushdownParams:
        return plan_sql_pushdown_params(self)

    def describe(self) -> list[str]:
        return describe_plan(self)

    def has_filters(self) -> bool:
        return plan_has_filters(self)

    def effective_fetch_limit(self) -> int | None:
        return effective_fetch_limit(self)

    def with_limit(self, limit: int | None) -> ConversationQueryPlan:
        return replace(self, limit=limit)

    # -- Runtime filtering and sorting methods (was QueryPlanRuntimeMixin) --

    def has_post_filters(self) -> bool:
        return plan_has_post_filters(self)

    def needs_content_loading(self) -> bool:
        return plan_needs_content_loading(self)

    def can_use_summaries(self) -> bool:
        return not self.needs_content_loading()

    def can_count_in_sql(self) -> bool:
        return plan_can_count_in_sql(self)

    def can_use_action_event_stats(self) -> bool:
        return plan_can_use_action_event_stats(self)

    def _matches_referenced_path(self, conversation: Conversation) -> bool:
        return matches_referenced_path(self, conversation)

    def _matches_action_terms(self, conversation: Conversation) -> bool:
        return matches_action_terms(self, conversation)

    def _matches_tool_terms(self, conversation: Conversation) -> bool:
        return matches_tool_terms(self, conversation)

    def _matches_action_sequence(self, conversation: Conversation) -> bool:
        return matches_action_sequence(self, conversation)

    def _matches_action_text_terms(self, conversation: Conversation) -> bool:
        return matches_action_text_terms(self, conversation)

    def _apply_common_filters(
        self,
        items: builtins.list[_FilterableT],
        *,
        sql_pushed: bool,
    ) -> builtins.list[_FilterableT]:
        return apply_common_filters(self, items, sql_pushed=sql_pushed)

    def _apply_full_filters(self, conversations: list[Conversation], *, sql_pushed: bool) -> list[Conversation]:
        return apply_full_filters(self, conversations, sql_pushed=sql_pushed)

    def _sort_generic(self, items: list[_T], key_fn: Callable[[_T], SortKey]) -> list[_T]:
        return sort_generic(self, items, key_fn)

    def _sort_conversations(self, conversations: list[Conversation]) -> list[Conversation]:
        return sort_conversations(self, conversations)

    def _sort_summaries(self, summaries: list[ConversationSummary]) -> list[ConversationSummary]:
        return sort_summaries(self, summaries)

    def _finalize(self, items: list[_T]) -> list[_T]:
        return finalize_results(self, items)

    # -- Retrieval and execution methods (was QueryPlanExecutionMixin) --

    def _candidate_record_query(self) -> tuple[ConversationRecordQuery, bool]:
        return candidate_record_query(self)

    def fetch_record_query(self) -> ConversationRecordQuery:
        record_query, _ = self._candidate_record_query()
        return record_query.with_limit(self.effective_fetch_limit())

    def _uses_action_read_model(self) -> bool:
        return uses_action_read_model(self)

    async def _action_event_rows_ready(self, repository: ConversationQueryRuntimeStore) -> bool:
        return await action_event_rows_ready(self, repository)

    async def can_use_action_event_stats_with(self, repository: ConversationQueryRuntimeStore) -> bool:
        return await can_use_action_event_stats_with(self, repository)

    async def _candidate_record_query_for(
        self,
        repository: ConversationQueryRuntimeStore,
    ) -> tuple[ConversationRecordQuery, bool]:
        return await candidate_record_query_for(self, repository)

    async def fetch_record_query_for(self, repository: ConversationQueryRuntimeStore) -> ConversationRecordQuery:
        return await fetch_record_query_for(self, repository)

    def _should_batch_post_filter_fetch(self) -> bool:
        return should_batch_post_filter_fetch(self)

    def _search_limit(self) -> int:
        return search_limit(self)

    async def list(self, repository: ConversationQueryRuntimeStore) -> list[Conversation]:
        from polylogue.archive.query.plan_execution import list_for_plan

        return await list_for_plan(self, repository)

    async def list_summaries(self, repository: ConversationQueryRuntimeStore) -> builtins.list[ConversationSummary]:
        from polylogue.archive.query.plan_execution import list_summaries_for_plan

        return await list_summaries_for_plan(self, repository)

    async def first(self, repository: ConversationQueryRuntimeStore) -> Conversation | None:
        from polylogue.archive.query.plan_execution import first_for_plan

        return await first_for_plan(self, repository)

    async def count(self, repository: ConversationQueryRuntimeStore) -> int:
        from polylogue.archive.query.plan_execution import count_for_plan

        return await count_for_plan(self, repository)

    async def delete(self, repository: ConversationQueryRuntimeStore) -> int:
        from polylogue.archive.query.plan_execution import delete_for_plan

        return await delete_for_plan(self, repository)


__all__ = ["ConversationQueryPlan", "conversation_has_branches", "plan_record_query", "plan_sql_pushdown_params"]
