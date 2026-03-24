"""Canonical immutable conversation-query plan model."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import datetime
from typing import TYPE_CHECKING, Any, TypeVar

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
from polylogue.lib.query_runtime import (
    apply_common_filters,
    apply_full_filters,
    matches_action_sequence,
    matches_action_terms,
    matches_action_text_terms,
    matches_path_terms,
    matches_tool_terms,
    plan_can_count_in_sql,
    plan_can_use_action_event_stats,
    plan_has_post_filters,
    plan_needs_content_loading,
)
from polylogue.lib.query_sorting import (
    finalize_results,
    sort_conversations,
    sort_generic,
    sort_summaries,
)
from polylogue.lib.query_support import conversation_has_branches, provider_values
from polylogue.storage.query_models import ConversationRecordQuery
from polylogue.types import Provider

if TYPE_CHECKING:
    from polylogue.lib.filter_types import SortField
    from polylogue.lib.models import Conversation, ConversationSummary
    from polylogue.protocols import VectorProvider
    from polylogue.storage.repository import ConversationRepository

_T = TypeVar("_T")


@dataclass(frozen=True)
class ConversationQueryPlan:
    """Canonical immutable execution state for conversation selection."""

    query_terms: tuple[str, ...] = ()
    contains_terms: tuple[str, ...] = ()
    negative_terms: tuple[str, ...] = ()
    retrieval_lane: str = "auto"
    path_terms: tuple[str, ...] = ()
    action_terms: tuple[str, ...] = ()
    excluded_action_terms: tuple[str, ...] = ()
    action_sequence: tuple[str, ...] = ()
    action_text_terms: tuple[str, ...] = ()
    tool_terms: tuple[str, ...] = ()
    excluded_tool_terms: tuple[str, ...] = ()
    providers: tuple[Provider | str, ...] = ()
    excluded_providers: tuple[Provider | str, ...] = ()
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
    min_messages: int | None = None
    max_messages: int | None = None
    min_words: int | None = None
    vector_provider: VectorProvider | None = None

    @property
    def fts_terms(self) -> tuple[str, ...]:
        return self.query_terms + self.contains_terms

    @property
    def sql_pushed(self) -> bool:
        return not self.fts_terms and self.conversation_id is None

    @property
    def record_query(self) -> ConversationRecordQuery:
        values = provider_values(self.providers)
        provider = values[0] if len(values) == 1 else None
        providers = values if len(values) > 1 else ()
        return ConversationRecordQuery(
            provider=provider,
            providers=providers,
            parent_id=self.parent_id,
            since=self.since.isoformat() if self.since else None,
            until=self.until.isoformat() if self.until else None,
            title_contains=self.title,
            path_terms=self.path_terms,
            action_terms=self.action_terms,
            excluded_action_terms=self.excluded_action_terms,
            tool_terms=self.tool_terms,
            excluded_tool_terms=self.excluded_tool_terms,
            has_tool_use=self.filter_has_tool_use,
            has_thinking=self.filter_has_thinking,
            min_messages=self.min_messages,
            max_messages=self.max_messages,
            min_words=self.min_words,
        )

    def sql_pushdown_params(self) -> dict[str, object]:
        params: dict[str, object] = {}
        values = provider_values(self.providers)
        if len(values) == 1:
            params["provider"] = values[0]
        elif values:
            params["providers"] = list(values)
        if self.parent_id:
            params["parent_id"] = self.parent_id
        if self.since:
            params["since"] = self.since.isoformat()
        if self.until:
            params["until"] = self.until.isoformat()
        if self.title:
            params["title_contains"] = self.title
        if self.path_terms:
            params["path_terms"] = list(self.path_terms)
        if self.action_terms:
            params["action_terms"] = list(self.action_terms)
        if self.excluded_action_terms:
            params["excluded_action_terms"] = list(self.excluded_action_terms)
        if self.action_sequence:
            params["action_sequence"] = list(self.action_sequence)
        if self.action_text_terms:
            params["action_text_terms"] = list(self.action_text_terms)
        if self.tool_terms:
            params["tool_terms"] = list(self.tool_terms)
        if self.excluded_tool_terms:
            params["excluded_tool_terms"] = list(self.excluded_tool_terms)
        if self.filter_has_tool_use:
            params["has_tool_use"] = True
        if self.filter_has_thinking:
            params["has_thinking"] = True
        if self.min_messages is not None:
            params["min_messages"] = self.min_messages
        if self.max_messages is not None:
            params["max_messages"] = self.max_messages
        if self.min_words is not None:
            params["min_words"] = self.min_words
        return params

    def describe(self) -> list[str]:
        parts: list[str] = []
        if self.fts_terms:
            parts.append(f"contains: {', '.join(self.fts_terms)}")
        if self.negative_terms:
            parts.append(f"exclude text: {', '.join(self.negative_terms)}")
        if self.retrieval_lane != "auto":
            parts.append(f"retrieval: {self.retrieval_lane}")
        if self.path_terms:
            parts.append(f"path: {', '.join(self.path_terms)}")
        if self.action_terms:
            parts.append(f"action: {', '.join(self.action_terms)}")
        if self.excluded_action_terms:
            parts.append(f"exclude action: {', '.join(self.excluded_action_terms)}")
        if self.action_sequence:
            parts.append(f"action sequence: {' -> '.join(self.action_sequence)}")
        if self.action_text_terms:
            parts.append(f"action text: {', '.join(self.action_text_terms)}")
        if self.tool_terms:
            parts.append(f"tool: {', '.join(self.tool_terms)}")
        if self.excluded_tool_terms:
            parts.append(f"exclude tool: {', '.join(self.excluded_tool_terms)}")
        if self.providers:
            parts.append(f"provider: {', '.join(provider_values(self.providers))}")
        if self.excluded_providers:
            parts.append(f"exclude provider: {', '.join(provider_values(self.excluded_providers))}")
        if self.tags:
            parts.append(f"tag: {', '.join(self.tags)}")
        if self.excluded_tags:
            parts.append(f"exclude tag: {', '.join(self.excluded_tags)}")
        if self.title:
            parts.append(f"title: {self.title}")
        if self.has_types:
            parts.append(f"has: {', '.join(self.has_types)}")
        if self.filter_has_tool_use:
            parts.append("has_tool_use")
        if self.filter_has_thinking:
            parts.append("has_thinking")
        if self.min_messages is not None:
            parts.append(f"min_messages: {self.min_messages}")
        if self.max_messages is not None:
            parts.append(f"max_messages: {self.max_messages}")
        if self.min_words is not None:
            parts.append(f"min_words: {self.min_words}")
        if self.since:
            parts.append(f"since: {self.since.isoformat()}")
        if self.until:
            parts.append(f"until: {self.until.isoformat()}")
        if self.conversation_id:
            parts.append(f"id: {self.conversation_id}")
        if self.parent_id:
            parts.append(f"parent: {self.parent_id}")
        if self.continuation is True:
            parts.append("continuation")
        if self.continuation is False:
            parts.append("not continuation")
        if self.sidechain is True:
            parts.append("sidechain")
        if self.sidechain is False:
            parts.append("not sidechain")
        if self.root is True:
            parts.append("root")
        if self.root is False:
            parts.append("not root")
        if self.has_branches is True:
            parts.append("has branches")
        if self.has_branches is False:
            parts.append("no branches")
        if self.predicates:
            parts.append(f"custom predicates: {len(self.predicates)}")
        if self.similar_text:
            parts.append(f"similar: {self.similar_text[:30]}")
        return parts

    def has_filters(self) -> bool:
        return any(
            (
                self.fts_terms,
                self.negative_terms,
                self.path_terms,
                self.action_terms,
                self.excluded_action_terms,
                self.action_sequence,
                self.action_text_terms,
                self.tool_terms,
                self.excluded_tool_terms,
                self.providers,
                self.excluded_providers,
                self.tags,
                self.excluded_tags,
                self.has_types,
                self.title is not None,
                self.conversation_id is not None,
                self.parent_id is not None,
                self.since is not None,
                self.until is not None,
                self.similar_text is not None,
                self.continuation is not None,
                self.sidechain is not None,
                self.root is not None,
                self.has_branches is not None,
                self.filter_has_tool_use,
                self.filter_has_thinking,
                self.min_messages is not None,
                self.max_messages is not None,
                self.min_words is not None,
                self.predicates,
            )
        )

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

    def _matches_path_terms(self, conversation: Conversation) -> bool:
        return matches_path_terms(self, conversation)

    def _matches_action_terms(self, conversation: Conversation) -> bool:
        return matches_action_terms(self, conversation)

    def _matches_tool_terms(self, conversation: Conversation) -> bool:
        return matches_tool_terms(self, conversation)

    def _matches_action_sequence(self, conversation: Conversation) -> bool:
        return matches_action_sequence(self, conversation)

    def _matches_action_text_terms(self, conversation: Conversation) -> bool:
        return matches_action_text_terms(self, conversation)

    def effective_fetch_limit(self) -> int | None:
        if self.limit is None:
            return None
        if self.has_post_filters():
            return max(self.limit * 10, 500)
        if self.sample is not None:
            return max(self.sample * 3, 200)
        return max(self.limit * 2, 2)

    def with_limit(self, limit: int | None) -> ConversationQueryPlan:
        return replace(self, limit=limit)

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

    def _apply_common_filters(
        self,
        items: list[_T],
        *,
        sql_pushed: bool,
    ) -> list[_T]:
        return apply_common_filters(self, items, sql_pushed=sql_pushed)

    def _apply_full_filters(
        self,
        conversations: list[Conversation],
        *,
        sql_pushed: bool,
    ) -> list[Conversation]:
        return apply_full_filters(self, conversations, sql_pushed=sql_pushed)

    def _sort_generic(
        self,
        items: list[_T],
        key_fn: Callable[[_T], Any],
    ) -> list[_T]:
        return sort_generic(self, items, key_fn)

    def _sort_conversations(
        self,
        conversations: list[Conversation],
    ) -> list[Conversation]:
        return sort_conversations(self, conversations)

    def _sort_summaries(
        self,
        summaries: list[ConversationSummary],
    ) -> list[ConversationSummary]:
        return sort_summaries(self, summaries)

    def _finalize(self, items: list[_T]) -> list[_T]:
        return finalize_results(self, items)

    async def list(
        self,
        repository: ConversationRepository,
    ) -> list[Conversation]:
        from polylogue.lib.query_plan_execution import list_for_plan

        return await list_for_plan(self, repository)

    async def list_summaries(
        self,
        repository: ConversationRepository,
    ) -> list[ConversationSummary]:
        from polylogue.lib.query_plan_execution import list_summaries_for_plan

        return await list_summaries_for_plan(self, repository)

    async def first(
        self,
        repository: ConversationRepository,
    ) -> Conversation | None:
        from polylogue.lib.query_plan_execution import first_for_plan

        return await first_for_plan(self, repository)

    async def count(
        self,
        repository: ConversationRepository,
    ) -> int:
        from polylogue.lib.query_plan_execution import count_for_plan

        return await count_for_plan(self, repository)

    async def delete(
        self,
        repository: ConversationRepository,
    ) -> int:
        from polylogue.lib.query_plan_execution import delete_for_plan

        return await delete_for_plan(self, repository)


__all__ = ["ConversationQueryPlan", "conversation_has_branches"]
