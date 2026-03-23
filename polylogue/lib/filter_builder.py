from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
from typing import TYPE_CHECKING

from polylogue.lib.dates import parse_date
from polylogue.lib.filter_types import SortField
from polylogue.types import Provider

if TYPE_CHECKING:
    from polylogue.lib.filters import ConversationFilter
    from polylogue.lib.models import Conversation


class ConversationFilterBuilderMixin:
    """Fluent mutators for ConversationFilter."""

    def contains(self, text: str) -> ConversationFilter:
        """Filter to conversations containing text (FTS search)."""
        self._fts_terms.append(text)
        return self

    def exclude_text(self, text: str) -> ConversationFilter:
        """Exclude conversations containing text."""
        self._negative_fts_terms.append(text)
        return self

    def provider(self, *names: Provider | str) -> ConversationFilter:
        """Filter to conversations from specific providers."""
        self._providers.extend(
            n if isinstance(n, Provider) else Provider.from_string(n) for n in names
        )
        return self

    def exclude_provider(self, *names: Provider | str) -> ConversationFilter:
        """Exclude conversations from specific providers."""
        self._excluded_providers.extend(
            n if isinstance(n, Provider) else Provider.from_string(n) for n in names
        )
        return self

    def tag(self, *tags: str) -> ConversationFilter:
        """Filter to conversations with specific tags."""
        self._tags.extend(tags)
        return self

    def exclude_tag(self, *tags: str) -> ConversationFilter:
        """Exclude conversations with specific tags."""
        self._excluded_tags.extend(tags)
        return self

    def has(self, *types: str) -> ConversationFilter:
        """Filter to conversations containing specific content types."""
        self._has_types.extend(types)
        return self

    def since(self, date: str | datetime) -> ConversationFilter:
        """Filter to conversations after date."""
        if isinstance(date, str):
            parsed = parse_date(date)
            if parsed is None:
                msg = f"Cannot parse date: {date!r}"
                raise ValueError(msg)
            self._since_date = parsed
        else:
            self._since_date = date
        return self

    def until(self, date: str | datetime) -> ConversationFilter:
        """Filter to conversations before date."""
        if isinstance(date, str):
            parsed = parse_date(date)
            if parsed is None:
                msg = f"Cannot parse date: {date!r}"
                raise ValueError(msg)
            self._until_date = parsed
        else:
            self._until_date = date
        return self

    def title(self, pattern: str) -> ConversationFilter:
        """Filter to conversations with titles containing pattern."""
        self._title_pattern = pattern
        return self

    def path(self, pattern: str) -> ConversationFilter:
        """Filter to conversations that touched a path containing pattern."""
        self._path_terms.append(pattern)
        return self

    def id(self, prefix: str) -> ConversationFilter:
        """Filter to conversations with ID starting with prefix."""
        self._id_prefix = prefix
        return self

    def sort(self, field: SortField) -> ConversationFilter:
        """Set sort field."""
        self._sort_field = field
        return self

    def reverse(self) -> ConversationFilter:
        """Reverse sort order (ascending instead of descending)."""
        self._sort_reverse = True
        return self

    def limit(self, n: int) -> ConversationFilter:
        """Limit number of results."""
        self._limit_count = n
        return self

    def sample(self, n: int) -> ConversationFilter:
        """Randomly sample n conversations from results."""
        self._sample_count = n
        return self

    def similar(self, text: str) -> ConversationFilter:
        """Rank by semantic similarity to text (requires vector index)."""
        self._similar_text = text
        return self

    def where(self, predicate: Callable[[Conversation], bool]) -> ConversationFilter:
        """Add custom filter predicate."""
        self._predicates.append(predicate)
        return self

    def is_continuation(self, value: bool = True) -> ConversationFilter:
        """Filter to continuation conversations (or exclude them if value=False)."""
        self._continuation = value
        return self

    def is_sidechain(self, value: bool = True) -> ConversationFilter:
        """Filter to sidechain conversations (or exclude them if value=False)."""
        self._sidechain = value
        return self

    def is_root(self, value: bool = True) -> ConversationFilter:
        """Filter to root conversations (those with no parent)."""
        self._root = value
        return self

    def has_tool_use(self) -> ConversationFilter:
        """Filter to conversations that contain tool_use or tool_result blocks (SQL pushdown)."""
        self._filter_has_tool_use = True
        return self

    def has_thinking(self) -> ConversationFilter:
        """Filter to conversations that contain thinking blocks (SQL pushdown)."""
        self._filter_has_thinking = True
        return self

    def min_messages(self, n: int) -> ConversationFilter:
        """Filter to conversations with at least n messages (SQL pushdown)."""
        self._min_messages = n
        return self

    def max_messages(self, n: int) -> ConversationFilter:
        """Filter to conversations with at most n messages (SQL pushdown)."""
        self._max_messages = n
        return self

    def min_words(self, n: int) -> ConversationFilter:
        """Filter to conversations with at least n total words (SQL pushdown)."""
        self._min_words = n
        return self

    def has_file_operations(self) -> ConversationFilter:
        """Filter to conversations containing file read/write/edit operations (SQL pushdown)."""
        self._filter_has_file_ops = True
        return self

    def has_git_operations(self) -> ConversationFilter:
        """Filter to conversations containing git operations (SQL pushdown)."""
        self._filter_has_git_ops = True
        return self

    def has_subagent_spawns(self) -> ConversationFilter:
        """Filter to conversations that spawned subagents via Task tool (SQL pushdown)."""
        self._filter_has_subagent = True
        return self

    def parent(self, conversation_id: str) -> ConversationFilter:
        """Filter to conversations that are children of the given parent."""
        self._parent_id = conversation_id
        return self

    def has_branches(self, value: bool = True) -> ConversationFilter:
        """Filter to conversations that have branching messages."""
        self._has_branches = value
        return self


__all__ = ["ConversationFilterBuilderMixin"]
