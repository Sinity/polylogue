"""Fluent filter builder for conversation-level queries.

This module provides the `ConversationFilter` class for building chainable
queries against the conversation repository.  All terminal methods
(`list`, `first`, `count`, `delete`) are async.

Example::

    from polylogue import Polylogue

    async with Polylogue() as p:
        # Get recent Claude conversations
        convs = await p.filter().provider("claude-ai").since("2024-01-01").limit(10).list()

        # Search for errors in ChatGPT
        convs = await p.filter().provider("chatgpt").contains("error").list()

        # Get first matching conversation
        conv = await p.filter().tag("important").first()

        # Count conversations with thinking blocks
        count = await p.filter().has("thinking").count()
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
from typing import TYPE_CHECKING

from polylogue.lib.filter_builder import ConversationFilterBuilderMixin
from polylogue.lib.filter_types import SortField
from polylogue.lib.query_execution import ConversationQueryPlan
from polylogue.types import Provider

if TYPE_CHECKING:
    from polylogue.lib.models import Conversation
    from polylogue.protocols import VectorProvider
    from polylogue.storage.repository import ConversationRepository


class ConversationFilter(ConversationFilterBuilderMixin):
    """Fluent filter builder for conversation-level queries.

    Methods are chainable and return self, allowing concise filter expressions.
    Terminal methods (list, first, count, delete) execute the query.

    Example:
        filter.provider("claude-ai").since("2024-01-01").limit(10).list()
    """

    def __init__(
        self,
        repository: ConversationRepository,
        vector_provider: VectorProvider | None = None,
        *,
        query_plan: ConversationQueryPlan | None = None,
    ) -> None:
        """Initialize filter with repository.

        Args:
            repository: ConversationRepository for executing queries
            vector_provider: Optional VectorProvider for semantic search
        """
        self._repo = repository
        self._vector_provider = vector_provider
        self._predicates: list[Callable[[Conversation], bool]] = []
        self._fts_terms: list[str] = []
        self._negative_fts_terms: list[str] = []
        self._providers: list[Provider] = []
        self._excluded_providers: list[Provider] = []
        self._tags: list[str] = []
        self._excluded_tags: list[str] = []
        self._has_types: list[str] = []
        self._since_date: datetime | None = None
        self._until_date: datetime | None = None
        self._title_pattern: str | None = None
        self._path_terms: list[str] = []
        self._action_terms: list[str] = []
        self._excluded_action_terms: list[str] = []
        self._tool_terms: list[str] = []
        self._excluded_tool_terms: list[str] = []
        self._id_prefix: str | None = None
        self._parent_id: str | None = None
        self._sort_field: SortField = "date"
        self._sort_reverse: bool = False
        self._limit_count: int | None = None
        self._sample_count: int | None = None
        self._similar_text: str | None = None
        self._continuation: bool | None = None
        self._sidechain: bool | None = None
        self._root: bool | None = None
        self._has_branches: bool | None = None
        # SQL-pushable stats filters (via conversation_stats JOIN)
        self._filter_has_tool_use: bool = False
        self._filter_has_thinking: bool = False
        self._min_messages: int | None = None
        self._max_messages: int | None = None
        self._min_words: int | None = None
        if query_plan is not None:
            self._load_query_plan(query_plan)

    @classmethod
    def from_query_plan(
        cls,
        repository: ConversationRepository,
        query_plan: ConversationQueryPlan,
    ) -> ConversationFilter:
        """Build a fluent façade initialized from a canonical query plan."""
        return cls(
            repository,
            vector_provider=query_plan.vector_provider,
            query_plan=query_plan,
        )

    def _load_query_plan(self, query_plan: ConversationQueryPlan) -> None:
        self._fts_terms = list(query_plan.query_terms + query_plan.contains_terms)
        self._negative_fts_terms = list(query_plan.negative_terms)
        self._providers = [Provider.from_string(provider) for provider in query_plan.providers]
        self._excluded_providers = [Provider.from_string(provider) for provider in query_plan.excluded_providers]
        self._tags = list(query_plan.tags)
        self._excluded_tags = list(query_plan.excluded_tags)
        self._has_types = list(query_plan.has_types)
        self._since_date = query_plan.since
        self._until_date = query_plan.until
        self._title_pattern = query_plan.title
        self._path_terms = list(query_plan.path_terms)
        self._action_terms = list(query_plan.action_terms)
        self._excluded_action_terms = list(query_plan.excluded_action_terms)
        self._tool_terms = list(query_plan.tool_terms)
        self._excluded_tool_terms = list(query_plan.excluded_tool_terms)
        self._id_prefix = query_plan.conversation_id
        self._parent_id = query_plan.parent_id
        self._sort_field = query_plan.sort
        self._sort_reverse = query_plan.reverse
        self._limit_count = query_plan.limit
        self._sample_count = query_plan.sample
        self._similar_text = query_plan.similar_text
        self._continuation = query_plan.continuation
        self._sidechain = query_plan.sidechain
        self._root = query_plan.root
        self._has_branches = query_plan.has_branches
        self._filter_has_tool_use = query_plan.filter_has_tool_use
        self._filter_has_thinking = query_plan.filter_has_thinking
        self._min_messages = query_plan.min_messages
        self._max_messages = query_plan.max_messages
        self._min_words = query_plan.min_words
        self._predicates = list(query_plan.predicates)
        self._vector_provider = query_plan.vector_provider

    def build_query_plan(self) -> ConversationQueryPlan:
        """Compile the fluent builder state to the canonical immutable plan."""
        return ConversationQueryPlan(
            query_terms=tuple(self._fts_terms),
            negative_terms=tuple(self._negative_fts_terms),
            providers=tuple(self._providers),
            excluded_providers=tuple(self._excluded_providers),
            tags=tuple(self._tags),
            excluded_tags=tuple(self._excluded_tags),
            has_types=tuple(self._has_types),
            title=self._title_pattern,
            path_terms=tuple(self._path_terms),
            action_terms=tuple(self._action_terms),
            excluded_action_terms=tuple(self._excluded_action_terms),
            tool_terms=tuple(self._tool_terms),
            excluded_tool_terms=tuple(self._excluded_tool_terms),
            conversation_id=self._id_prefix,
            parent_id=self._parent_id,
            since=self._since_date,
            until=self._until_date,
            sort=self._sort_field,
            reverse=self._sort_reverse,
            limit=self._limit_count,
            sample=self._sample_count,
            similar_text=self._similar_text,
            predicates=tuple(self._predicates),
            continuation=self._continuation,
            sidechain=self._sidechain,
            root=self._root,
            has_branches=self._has_branches,
            filter_has_tool_use=self._filter_has_tool_use,
            filter_has_thinking=self._filter_has_thinking,
            min_messages=self._min_messages,
            max_messages=self._max_messages,
            min_words=self._min_words,
            vector_provider=self._vector_provider,
        )

    def _sql_pushdown_params(self) -> dict[str, object]:
        return self.build_query_plan().sql_pushdown_params()

    def _has_post_filters(self) -> bool:
        return self.build_query_plan().has_post_filters()

    def _needs_content_loading(self) -> bool:
        return self.build_query_plan().needs_content_loading()

    def can_use_summaries(self) -> bool:
        return self.build_query_plan().can_use_summaries()

    def describe(self) -> list[str]:
        return self.build_query_plan().describe()

    async def list(self):
        return await self.build_query_plan().list(self._repo)

    async def list_summaries(self):
        return await self.build_query_plan().list_summaries(self._repo)

    async def first(self):
        return await self.build_query_plan().first(self._repo)

    async def count(self) -> int:
        return await self.build_query_plan().count(self._repo)

    async def delete(self) -> int:
        return await self.build_query_plan().delete(self._repo)


__all__ = ["ConversationFilter", "SortField"]
