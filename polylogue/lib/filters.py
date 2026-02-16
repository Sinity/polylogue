"""Fluent filter builder for conversation-level queries.

This module provides the `ConversationFilter` class for building chainable
queries against the conversation repository.  All terminal methods
(`list`, `first`, `count`, `delete`) are async.

Example::

    from polylogue import Polylogue

    async with Polylogue() as p:
        # Get recent Claude conversations
        convs = await p.filter().provider("claude").since("2024-01-01").limit(10).list()

        # Search for errors in ChatGPT
        convs = await p.filter().provider("chatgpt").contains("error").list()

        # Get first matching conversation
        conv = await p.filter().tag("important").first()

        # Count conversations with thinking blocks
        count = await p.filter().has("thinking").count()
"""

from __future__ import annotations

import builtins
import random
from collections.abc import Awaitable, Callable
from datetime import datetime
from typing import TYPE_CHECKING, Any, Literal, TypeVar

from polylogue.lib.dates import parse_date
from polylogue.lib.log import get_logger

logger = get_logger(__name__)

if TYPE_CHECKING:
    from polylogue.lib.models import Conversation, ConversationSummary
    from polylogue.protocols import VectorProvider
    from polylogue.storage.repository import ConversationRepository

# Sort field options
SortField = Literal["date", "tokens", "messages", "words", "longest", "random"]

_T = TypeVar("_T")


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
        """Filter to conversations containing text (FTS search)."""
        self._fts_terms.append(text)
        return self

    def no_contains(self, text: str) -> ConversationFilter:
        """Exclude conversations containing text."""
        self._negative_fts_terms.append(text)
        return self

    def provider(self, *names: str) -> ConversationFilter:
        """Filter to conversations from specific providers."""
        self._providers.extend(names)
        return self

    def no_provider(self, *names: str) -> ConversationFilter:
        """Exclude conversations from specific providers."""
        self._excluded_providers.extend(names)
        return self

    def tag(self, *tags: str) -> ConversationFilter:
        """Filter to conversations with specific tags."""
        self._tags.extend(tags)
        return self

    def no_tag(self, *tags: str) -> ConversationFilter:
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
        if value:
            self._predicates.append(lambda c: c.is_continuation)
        else:
            self._predicates.append(lambda c: not c.is_continuation)
        return self

    def is_sidechain(self, value: bool = True) -> ConversationFilter:
        """Filter to sidechain conversations (or exclude them if value=False)."""
        if value:
            self._predicates.append(lambda c: c.is_sidechain)
        else:
            self._predicates.append(lambda c: not c.is_sidechain)
        return self

    def is_root(self, value: bool = True) -> ConversationFilter:
        """Filter to root conversations (those with no parent)."""
        if value:
            self._predicates.append(lambda c: c.is_root)
        else:
            self._predicates.append(lambda c: not c.is_root)
        return self

    def parent(self, conversation_id: str) -> ConversationFilter:
        """Filter to conversations that are children of the given parent."""
        self._predicates.append(lambda c: c.parent_id == conversation_id)
        return self

    def has_branches(self, value: bool = True) -> ConversationFilter:
        """Filter to conversations that have branching messages."""
        if value:
            self._predicates.append(lambda c: any(m.branch_index > 0 for m in c.messages))
        else:
            self._predicates.append(lambda c: not any(m.branch_index > 0 for m in c.messages))
        return self

    # --- Terminal methods (execute query) ---

    def _apply_common_filters(
        self,
        items: builtins.list[_T],
        *,
        sql_pushed: bool = False,
    ) -> builtins.list[_T]:
        """Apply metadata-level filters shared by Conversation and ConversationSummary.

        Both types expose .provider, .updated_at, .display_title, .tags, .id, .summary
        via duck typing, so this single method handles all metadata filtering.
        """
        results = list(items)

        if not sql_pushed:
            if self._providers:
                results = [x for x in results if x.provider in self._providers]
            if self._since_date:
                results = [x for x in results if x.updated_at and x.updated_at >= self._since_date]
            if self._until_date:
                results = [x for x in results if x.updated_at and x.updated_at <= self._until_date]
            if self._title_pattern:
                pattern_lower = self._title_pattern.lower()
                results = [x for x in results if x.display_title and pattern_lower in x.display_title.lower()]

        if self._excluded_providers:
            excluded_set = set(self._excluded_providers)
            results = [x for x in results if x.provider not in excluded_set]

        if self._tags:
            tag_set = set(self._tags)
            results = [x for x in results if tag_set.intersection(x.tags)]
        if self._excluded_tags:
            excluded_tag_set = set(self._excluded_tags)
            results = [x for x in results if not excluded_tag_set.intersection(x.tags)]

        if self._id_prefix:
            results = [x for x in results if str(x.id).startswith(self._id_prefix)]

        if "summary" in self._has_types:
            results = [x for x in results if x.summary]

        return results

    def _apply_filters(
        self,
        conversations: builtins.list[Conversation],
        *,
        sql_pushed: bool = False,
    ) -> builtins.list[Conversation]:
        """Apply in-memory filters to conversation list.

        Delegates shared metadata filters to _apply_common_filters, then applies
        Conversation-specific filters that require message access.
        """
        results = self._apply_common_filters(conversations, sql_pushed=sql_pushed)

        # Content type filters requiring message access
        if self._has_types:
            for content_type in self._has_types:
                if content_type == "thinking":
                    results = [c for c in results if any(m.is_thinking for m in c.messages)]
                elif content_type == "tools":
                    results = [c for c in results if any(m.is_tool_use for m in c.messages)]
                elif content_type == "attachments":
                    results = [c for c in results if any(m.attachments for m in c.messages)]

        # Negative FTS — single pass combining all terms
        if self._negative_fts_terms:
            neg_terms = [t.lower() for t in self._negative_fts_terms]

            def _has_neg_term(c: Conversation) -> bool:
                for m in c.messages:
                    if not m.text:
                        continue
                    text_lower = m.text.lower()
                    for term in neg_terms:
                        if term in text_lower:
                            return True
                return False

            results = [c for c in results if not _has_neg_term(c)]

        for predicate in self._predicates:
            results = [c for c in results if predicate(c)]

        return results

    def _apply_sort_generic(
        self,
        items: builtins.list[_T],
        sort_key_fn: Callable[[_T], Any],
    ) -> builtins.list[_T]:
        """Apply sorting using the given key function."""
        if self._sort_field == "random":
            shuffled = list(items)
            random.shuffle(shuffled)
            return shuffled
        return sorted(items, key=sort_key_fn, reverse=not self._sort_reverse)

    def _apply_sort(self, conversations: builtins.list[Conversation]) -> builtins.list[Conversation]:
        """Apply sorting to conversation list."""
        from datetime import timezone

        dt_min = datetime.min.replace(tzinfo=timezone.utc)

        def sort_key(c: Conversation) -> Any:
            if self._sort_field == "date":
                return c.updated_at or dt_min
            elif self._sort_field == "messages":
                return len(c.messages)
            elif self._sort_field == "words":
                return sum(m.word_count for m in c.messages)
            elif self._sort_field == "longest":
                return max((m.word_count for m in c.messages), default=0)
            elif self._sort_field == "tokens":
                return sum(len(m.text or "") for m in c.messages) // 4
            return c.updated_at or dt_min

        return self._apply_sort_generic(conversations, sort_key)

    def _sql_pushdown_params(self) -> dict[str, object]:
        """Build kwargs for repository list/list_summaries that push filters to SQL.

        Returns dict with keys matching the repository's list() parameters.
        Only includes non-None values for filters that can be pushed down.
        """
        params: dict[str, object] = {}
        if self._providers:
            if len(self._providers) == 1:
                params["provider"] = self._providers[0]
            else:
                params["providers"] = self._providers
        if self._since_date:
            params["since"] = self._since_date.isoformat()
        if self._until_date:
            params["until"] = self._until_date.isoformat()
        if self._title_pattern:
            params["title_contains"] = self._title_pattern
        return params

    def _has_post_filters(self) -> bool:
        """Check if any filters require in-memory post-processing."""
        return bool(
            self._excluded_providers
            or self._tags
            or self._excluded_tags
            or self._has_types
            or self._predicates
            or self._negative_fts_terms
        )

    def _effective_fetch_limit(self) -> int | None:
        """Calculate how many candidates to fetch from the backend.

        Uses the user's desired limit + a safety margin to compensate for
        post-filter shrinkage. Returns None when post-filters need
        the full dataset, a higher limit when sorting could reduce results.
        """
        if self._limit_count is None:
            if self._has_post_filters():
                # Post-filters (tags, excludes) can match anywhere in the archive,
                # so we must scan the full dataset to avoid missing results.
                return None
            return 1000  # No limit requested — fetch a reasonable max

        if self._has_post_filters():
            # Post-filters may reject many candidates; over-fetch aggressively
            return max(self._limit_count * 10, 500)

        if self._sample_count is not None:
            # Sampling needs a full pool to draw from
            return max(self._sample_count * 3, 200)

        # Simple case: limit is the only constraint, with small safety margin
        return max(self._limit_count * 2, 50)

    async def _fetch_generic(
        self,
        get_by_id: Callable[[str], Awaitable[_T | None]],
        search: Callable[[str, int, builtins.list[str] | None], Awaitable[builtins.list[_T]]],
        list_all: Callable[..., Awaitable[builtins.list[_T]]],
    ) -> builtins.list[_T]:
        """Fetch candidate items from repository using provided accessors.

        Handles three fetch strategies: ID prefix resolution, FTS search,
        and full list with SQL pushdown. Callers provide type-specific
        repository methods as callbacks.
        """
        if self._id_prefix and not self._fts_terms:
            resolved_id = await self._repo.resolve_id(self._id_prefix)
            if resolved_id:
                item = await get_by_id(str(resolved_id))
                return [item] if item else []

        fetch_limit = self._effective_fetch_limit()

        if self._fts_terms:
            query = " ".join(self._fts_terms)
            try:
                search_limit = max(fetch_limit, 100) if fetch_limit is not None else 10000
                return await search(query, search_limit, self._providers or None)
            except Exception as exc:
                logger.debug("FTS search failed, falling back to list: %s", exc)

        sql_params = self._sql_pushdown_params()
        return await list_all(limit=fetch_limit, **sql_params)

    async def _fetch_candidates(self) -> builtins.list[Conversation]:
        """Fetch candidate conversations from repository."""
        return await self._fetch_generic(
            self._repo.get,
            lambda q, lim, provs: self._repo.search(q, limit=lim, providers=provs),
            self._repo.list,
        )

    def _execute_pipeline(
        self,
        candidates: builtins.list[_T],
        apply_filters: Callable[[builtins.list[_T], bool], builtins.list[_T]],
        apply_sort: Callable[[builtins.list[_T]], builtins.list[_T]],
    ) -> builtins.list[_T]:
        """Run the shared filter → sort → sample → limit pipeline."""
        sql_pushed = not self._fts_terms and not self._id_prefix
        filtered = apply_filters(candidates, sql_pushed)
        sorted_results = apply_sort(filtered)

        if self._sample_count is not None and self._sample_count < len(sorted_results):
            sorted_results = random.sample(sorted_results, self._sample_count)
        if self._limit_count is not None:
            sorted_results = sorted_results[: self._limit_count]
        return sorted_results

    async def list(self) -> builtins.list[Conversation]:
        """Execute query and return matching conversations."""
        # Semantic search has its own fetch path
        if self._similar_text:
            candidates = await self._repo.search_similar(
                self._similar_text,
                limit=self._limit_count or 10,
                vector_provider=self._vector_provider,
            )
            return self._apply_filters(candidates)

        candidates = await self._fetch_candidates()
        return self._execute_pipeline(
            candidates,
            lambda items, pushed: self._apply_filters(items, sql_pushed=pushed),
            self._apply_sort,
        )

    async def first(self) -> Conversation | None:
        """Execute query and return first matching conversation.

        Returns:
            First matching Conversation or None if no matches
        """
        results = await self.limit(1).list()
        return results[0] if results else None

    async def count(self) -> int:
        """Execute query and return count of matching conversations.

        When possible, uses SQL COUNT(*) directly for O(1) performance.
        Falls back to loading and counting for complex filter combinations.

        Returns:
            Number of matching conversations
        """
        # Fast path: pure SQL count for simple filter combos (no FTS, no content filters)
        if (
            not self._fts_terms
            and not self._id_prefix
            and not self._similar_text
            and not self._predicates
            and not self._has_types
            and not self._negative_fts_terms
            and not self._excluded_providers
            and not self._tags
            and not self._excluded_tags
        ):
            return await self._repo.count(**self._sql_pushdown_params())

        # Medium path: use summaries (lightweight) if possible
        if self.can_use_summaries():
            saved_limit = self._limit_count
            self._limit_count = None
            try:
                results = await self.list_summaries()
            finally:
                self._limit_count = saved_limit
            return len(results)

        # Slow path: must load full conversations
        saved_limit = self._limit_count
        self._limit_count = None
        try:
            results = await self.list()
        finally:
            self._limit_count = saved_limit
        return len(results)

    async def delete(self) -> int:
        """Delete matching conversations.

        WARNING: This permanently deletes conversations. Use with caution.

        Returns:
            Number of conversations deleted
        """
        # Get all matching conversations
        results = await self.list()
        deleted_count = 0

        for conv in results:
            # Access the backend through the repository
            if await self._repo.backend.delete_conversation(str(conv.id)):
                deleted_count += 1

        return deleted_count

    async def pick(self) -> Conversation | None:
        """Interactive picker for matching conversations.

        If running in a TTY, presents a menu to select from matches.
        Otherwise returns first match.

        Returns:
            Selected Conversation or None if no matches
        """
        import sys

        results = await self.list()
        if not results:
            return None

        if not sys.stdout.isatty():
            return results[0]

        # Simple interactive picker
        print(f"\n{len(results)} matching conversations:\n")
        for i, conv in enumerate(results[:20], 1):  # Show max 20
            title = conv.display_title[:50]
            date = conv.display_date.strftime("%Y-%m-%d") if conv.display_date else "unknown"
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

    # --- Lightweight summary methods (memory-efficient) ---

    def _needs_content_loading(self) -> bool:
        """Check if any filters require loading message content.

        Returns True if we need to load full Conversation objects,
        False if we can use lightweight ConversationSummary objects.
        """
        # These filters require message access
        if self._has_types:
            # 'summary' doesn't need messages, but thinking/tools/attachments do
            needs_messages = any(t in ("thinking", "tools", "attachments") for t in self._has_types)
            if needs_messages:
                return True

        if self._negative_fts_terms:
            return True

        # Custom predicates might need messages - assume they do
        # unless we add a way to mark them as summary-compatible
        if self._predicates:
            return True

        # Sort by messages/words/longest/tokens needs message data
        return self._sort_field in ("messages", "words", "longest", "tokens")

    async def _fetch_summary_candidates(self) -> builtins.list[ConversationSummary]:
        """Fetch candidate conversation summaries (lightweight, no messages)."""
        return await self._fetch_generic(
            self._repo.get_summary,
            lambda q, lim, provs: self._repo.search_summaries(q, limit=lim, providers=provs),
            self._repo.list_summaries,
        )

    def _apply_summary_filters(
        self,
        summaries: builtins.list[ConversationSummary],
        *,
        sql_pushed: bool = False,
    ) -> builtins.list[ConversationSummary]:
        """Apply filters that work on summaries (no message access needed)."""
        return self._apply_common_filters(summaries, sql_pushed=sql_pushed)

    def _apply_summary_sort(self, summaries: builtins.list[ConversationSummary]) -> builtins.list[ConversationSummary]:
        """Apply sorting to summary list (limited to date-based sorts)."""
        from datetime import timezone

        dt_min = datetime.min.replace(tzinfo=timezone.utc)
        return self._apply_sort_generic(summaries, lambda s: s.updated_at or dt_min)

    async def list_summaries(self) -> builtins.list[ConversationSummary]:
        """Execute query and return lightweight summaries (no messages loaded).

        Memory-efficient alternative to list() for cases where you don't need
        message content. Raises ValueError if content-dependent filters are set.
        """
        if self._needs_content_loading():
            raise ValueError(
                "Cannot use list_summaries() with content-dependent filters "
                "(regex, has:thinking, has:tools, etc.). Use list() instead."
            )

        candidates = await self._fetch_summary_candidates()
        return self._execute_pipeline(
            candidates,
            lambda items, pushed: self._apply_summary_filters(items, sql_pushed=pushed),
            self._apply_summary_sort,
        )

    def can_use_summaries(self) -> bool:
        """Check if this filter can use lightweight summaries.

        Returns True if list_summaries() would work, False if list() is required.
        """
        return not self._needs_content_loading()
