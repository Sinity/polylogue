"""Fluent filter builder for conversation-level queries.

This module provides the `ConversationFilter` class for building chainable
queries against the conversation repository.

Example:
    from polylogue import Polylogue

    p = Polylogue()

    # Get recent Claude conversations
    convs = p.filter().provider("claude").since("2024-01-01").limit(10).list()

    # Search for errors in ChatGPT
    convs = p.filter().provider("chatgpt").contains("error").list()

    # Get first matching conversation
    conv = p.filter().tag("important").first()

    # Count conversations with thinking blocks
    count = p.filter().has("thinking").count()
"""

from __future__ import annotations

import random
from collections.abc import Callable
from datetime import datetime
from typing import TYPE_CHECKING, Any, Literal

from polylogue.core.timestamps import parse_timestamp

if TYPE_CHECKING:
    from polylogue.lib.models import Conversation
    from polylogue.lib.repository import ConversationRepository
    from polylogue.protocols import VectorProvider

# Sort field options
SortField = Literal["date", "tokens", "messages", "words", "longest", "random"]


class ConversationFilter:
    """Fluent filter builder for conversation-level queries.

    Methods are chainable and return self, allowing concise filter expressions.
    Terminal methods (list, first, count, delete) execute the query.

    Example:
        filter.provider("claude").since("2024-01-01").limit(10).list()
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
        self._providers: list[str] = []
        self._excluded_providers: list[str] = []
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

    # --- Filter methods (return self for chaining) ---

    def contains(self, text: str) -> ConversationFilter:
        """Filter to conversations containing text (FTS search).

        Args:
            text: Text to search for in message content

        Returns:
            self for chaining
        """
        self._fts_terms.append(text)
        return self

    def no_contains(self, text: str) -> ConversationFilter:
        """Exclude conversations containing text.

        Args:
            text: Text to exclude

        Returns:
            self for chaining
        """
        self._negative_fts_terms.append(text)
        return self

    def provider(self, *names: str) -> ConversationFilter:
        """Filter to conversations from specific providers.

        Args:
            *names: Provider names (e.g., "claude", "chatgpt")

        Returns:
            self for chaining
        """
        self._providers.extend(names)
        return self

    def no_provider(self, *names: str) -> ConversationFilter:
        """Exclude conversations from specific providers.

        Args:
            *names: Provider names to exclude

        Returns:
            self for chaining
        """
        self._excluded_providers.extend(names)
        return self

    def tag(self, *tags: str) -> ConversationFilter:
        """Filter to conversations with specific tags.

        Args:
            *tags: Tag names to include

        Returns:
            self for chaining
        """
        self._tags.extend(tags)
        return self

    def no_tag(self, *tags: str) -> ConversationFilter:
        """Exclude conversations with specific tags.

        Args:
            *tags: Tag names to exclude

        Returns:
            self for chaining
        """
        self._excluded_tags.extend(tags)
        return self

    def has(self, *types: str) -> ConversationFilter:
        """Filter to conversations containing specific content types.

        Args:
            *types: Content types like "thinking", "tools", "attachments", "summary"

        Returns:
            self for chaining
        """
        self._has_types.extend(types)
        return self

    def since(self, date: str | datetime) -> ConversationFilter:
        """Filter to conversations after date.

        Args:
            date: Date string (e.g., "2024-01-01") or datetime object

        Returns:
            self for chaining
        """
        if isinstance(date, str):
            self._since_date = parse_timestamp(date)
        else:
            self._since_date = date
        return self

    def until(self, date: str | datetime) -> ConversationFilter:
        """Filter to conversations before date.

        Args:
            date: Date string (e.g., "2024-12-31") or datetime object

        Returns:
            self for chaining
        """
        if isinstance(date, str):
            self._until_date = parse_timestamp(date)
        else:
            self._until_date = date
        return self

    def title(self, pattern: str) -> ConversationFilter:
        """Filter to conversations with titles containing pattern.

        Args:
            pattern: Text pattern to match in title

        Returns:
            self for chaining
        """
        self._title_pattern = pattern
        return self

    def id(self, prefix: str) -> ConversationFilter:
        """Filter to conversations with ID starting with prefix.

        Args:
            prefix: ID prefix to match

        Returns:
            self for chaining
        """
        self._id_prefix = prefix
        return self

    def sort(self, field: SortField) -> ConversationFilter:
        """Set sort field.

        Args:
            field: Sort field - "date", "tokens", "messages", "words", "longest", "random"

        Returns:
            self for chaining
        """
        self._sort_field = field
        return self

    def reverse(self) -> ConversationFilter:
        """Reverse sort order (ascending instead of descending).

        Returns:
            self for chaining
        """
        self._sort_reverse = True
        return self

    def limit(self, n: int) -> ConversationFilter:
        """Limit number of results.

        Args:
            n: Maximum number of conversations to return

        Returns:
            self for chaining
        """
        self._limit_count = n
        return self

    def sample(self, n: int) -> ConversationFilter:
        """Randomly sample n conversations from results.

        Args:
            n: Number of conversations to sample

        Returns:
            self for chaining
        """
        self._sample_count = n
        return self

    def similar(self, text: str) -> ConversationFilter:
        """Rank by semantic similarity to text (requires vector index).

        Args:
            text: Text to compare against

        Returns:
            self for chaining
        """
        self._similar_text = text
        return self

    def where(self, predicate: Callable[[Conversation], bool]) -> ConversationFilter:
        """Add custom filter predicate.

        Args:
            predicate: Function that takes Conversation and returns bool

        Returns:
            self for chaining
        """
        self._predicates.append(predicate)
        return self

    # --- Terminal methods (execute query) ---

    def _apply_filters(self, conversations: list[Conversation]) -> list[Conversation]:
        """Apply in-memory filters to conversation list.

        Args:
            conversations: List of conversations to filter

        Returns:
            Filtered list
        """
        results = list(conversations)

        # Provider filters
        if self._providers:
            results = [c for c in results if c.provider in self._providers]
        if self._excluded_providers:
            results = [c for c in results if c.provider not in self._excluded_providers]

        # Tag filters
        if self._tags:
            results = [c for c in results if any(t in c.tags for t in self._tags)]
        if self._excluded_tags:
            results = [c for c in results if not any(t in c.tags for t in self._excluded_tags)]

        # Date filters
        if self._since_date:
            results = [
                c for c in results
                if c.updated_at and c.updated_at >= self._since_date
            ]
        if self._until_date:
            results = [
                c for c in results
                if c.updated_at and c.updated_at <= self._until_date
            ]

        # Title filter
        if self._title_pattern:
            pattern_lower = self._title_pattern.lower()
            results = [
                c for c in results
                if c.display_title and pattern_lower in c.display_title.lower()
            ]

        # ID prefix filter
        if self._id_prefix:
            results = [c for c in results if str(c.id).startswith(self._id_prefix)]

        # Content type filters
        if self._has_types:
            for content_type in self._has_types:
                if content_type == "thinking":
                    results = [c for c in results if any(m.is_thinking for m in c.messages)]
                elif content_type == "tools":
                    results = [c for c in results if any(m.is_tool_use for m in c.messages)]
                elif content_type == "attachments":
                    results = [c for c in results if any(m.attachments for m in c.messages)]
                elif content_type == "summary":
                    results = [c for c in results if c.summary]

        # Negative FTS (exclude conversations containing text)
        if self._negative_fts_terms:
            for term in self._negative_fts_terms:
                term_lower = term.lower()
                results = [
                    c for c in results
                    if not any(term_lower in m.text.lower() for m in c.messages if m.text)
                ]

        # Custom predicates
        for predicate in self._predicates:
            results = [c for c in results if predicate(c)]

        return results

    def _apply_sort(self, conversations: list[Conversation]) -> list[Conversation]:
        """Apply sorting to conversation list.

        Args:
            conversations: List of conversations to sort

        Returns:
            Sorted list
        """
        if self._sort_field == "random":
            shuffled = list(conversations)
            random.shuffle(shuffled)
            return shuffled

        def sort_key(c: Conversation) -> Any:
            if self._sort_field == "date":
                return c.updated_at or datetime.min
            elif self._sort_field == "messages":
                return len(c.messages)
            elif self._sort_field == "words":
                return sum(m.word_count for m in c.messages)
            elif self._sort_field == "longest":
                return max((m.word_count for m in c.messages), default=0)
            elif self._sort_field == "tokens":
                # Approximate: 1 token ≈ 4 chars
                return sum(len(m.text or "") for m in c.messages) // 4
            return c.updated_at or datetime.min

        return sorted(
            conversations,
            key=sort_key,
            reverse=not self._sort_reverse,  # Default is descending
        )

    def _fetch_candidates(self) -> list[Conversation]:
        """Fetch candidate conversations from repository.

        Uses FTS search if terms specified, otherwise lists all.

        Returns:
            List of candidate conversations
        """
        # If we have FTS terms, use search
        if self._fts_terms:
            # Combine search terms
            query = " ".join(self._fts_terms)
            try:
                return self._repo.search(query)
            except Exception:
                # Fall back to list if search not available
                pass

        # If we have a provider filter and no FTS, use filtered list
        if self._providers and len(self._providers) == 1:
            return self._repo.list(limit=1000, provider=self._providers[0])

        # Default: list all (with reasonable limit)
        return self._repo.list(limit=1000)

    def list(self) -> list[Conversation]:
        """Execute query and return matching conversations.

        Returns:
            List of Conversation objects matching all filters
        """
        # If semantic search is requested, use vector provider
        if self._similar_text:
            candidates = self._repo.search_similar(
                self._similar_text,
                limit=self._limit_count or 10,
                vector_provider=self._vector_provider,
            )
            # Still apply in-memory filters
            filtered = self._apply_filters(candidates)
            return filtered

        # Fetch candidates
        candidates = self._fetch_candidates()

        # Apply in-memory filters
        filtered = self._apply_filters(candidates)

        # Apply sorting
        sorted_results = self._apply_sort(filtered)

        # Apply sampling (before limit)
        if self._sample_count is not None and self._sample_count < len(sorted_results):
            sorted_results = random.sample(sorted_results, self._sample_count)

        # Apply limit
        if self._limit_count is not None:
            sorted_results = sorted_results[:self._limit_count]

        return sorted_results

    def first(self) -> Conversation | None:
        """Execute query and return first matching conversation.

        Returns:
            First matching Conversation or None if no matches
        """
        results = self.limit(1).list()
        return results[0] if results else None

    def count(self) -> int:
        """Execute query and return count of matching conversations.

        Returns:
            Number of matching conversations
        """
        # Fetch and filter without limit
        saved_limit = self._limit_count
        self._limit_count = None
        results = self.list()
        self._limit_count = saved_limit
        return len(results)

    def delete(self) -> int:
        """Delete matching conversations.

        WARNING: This permanently deletes conversations. Use with caution.

        Returns:
            Number of conversations deleted

        Raises:
            NotImplementedError: Deletion via filter chain is not yet implemented.
                Use direct backend methods for now.
        """
        # TODO: Add delete_conversation to StorageBackend protocol
        raise NotImplementedError(
            "Deletion via filter chain not yet implemented. "
            "Use direct backend methods for now."
        )

    def pick(self) -> Conversation | None:
        """Interactive picker for matching conversations.

        If running in a TTY, presents a menu to select from matches.
        Otherwise returns first match.

        Returns:
            Selected Conversation or None if no matches
        """
        import sys

        results = self.list()
        if not results:
            return None

        if not sys.stdout.isatty():
            return results[0]

        # Simple interactive picker
        print(f"\n{len(results)} matching conversations:\n")
        for i, conv in enumerate(results[:20], 1):  # Show max 20
            title = conv.display_title[:50]
            date = conv.updated_at.strftime("%Y-%m-%d") if conv.updated_at else "unknown"
            print(f"  {i:2}. [{conv.provider}] {title} ({date})")

        if len(results) > 20:
            print(f"\n  ... and {len(results) - 20} more")

        try:
            choice = input("\nSelect number (or Enter for first): ").strip()
            if not choice:
                return results[0]
            idx = int(choice) - 1
            if 0 <= idx < len(results):
                return results[idx]
        except (ValueError, EOFError, KeyboardInterrupt):
            pass

        return None
