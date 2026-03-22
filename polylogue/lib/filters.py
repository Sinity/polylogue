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
from polylogue.lib.filter_runtime import ConversationFilterRuntimeMixin
from polylogue.lib.filter_types import SortField
from polylogue.types import Provider

if TYPE_CHECKING:
    from polylogue.lib.models import Conversation
    from polylogue.protocols import VectorProvider
    from polylogue.storage.repository import ConversationRepository


class ConversationFilter(ConversationFilterBuilderMixin, ConversationFilterRuntimeMixin):
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
        self._id_prefix: str | None = None
        self._sort_field: SortField = "date"
        self._sort_reverse: bool = False
        self._limit_count: int | None = None
        self._sample_count: int | None = None
        self._similar_text: str | None = None
        # SQL-pushable stats filters (via conversation_stats JOIN)
        self._filter_has_tool_use: bool = False
        self._filter_has_thinking: bool = False
        self._min_messages: int | None = None
        self._max_messages: int | None = None
        self._min_words: int | None = None
        # SQL-pushable semantic filters (via EXISTS on content_blocks.semantic_type)
        self._filter_has_file_ops: bool = False
        self._filter_has_git_ops: bool = False
        self._filter_has_subagent: bool = False


__all__ = ["ConversationFilter", "SortField"]
