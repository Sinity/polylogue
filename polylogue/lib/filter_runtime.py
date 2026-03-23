from __future__ import annotations

import builtins
import random
from collections.abc import Awaitable, Callable
from datetime import datetime
from typing import TYPE_CHECKING, Any, TypeVar

from polylogue.lib.filter_executor import _ExecutionPlan, build_execution_plan, sql_pushdown_params
from polylogue.logging import get_logger

logger = get_logger(__name__)

if TYPE_CHECKING:
    from polylogue.lib.models import Conversation, ConversationSummary

_T = TypeVar("_T")


def _conversation_has_branches(conversation: Conversation) -> bool:
    return any(message.branch_index > 0 for message in conversation.messages)


class ConversationFilterRuntimeMixin:
    """Execution helpers and terminal operations for ConversationFilter."""

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
        """Apply in-memory filters to conversation list."""
        results = self._apply_common_filters(conversations, sql_pushed=sql_pushed)

        if self._has_types:
            for content_type in self._has_types:
                if content_type == "thinking":
                    results = [c for c in results if any(m.is_thinking for m in c.messages)]
                elif content_type == "tools":
                    results = [c for c in results if any(m.is_tool_use for m in c.messages)]
                elif content_type == "attachments":
                    results = [c for c in results if any(m.attachments for m in c.messages)]

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
            if self._sort_field == "messages":
                return len(c.messages)
            if self._sort_field == "words":
                return sum(m.word_count for m in c.messages)
            if self._sort_field == "longest":
                return max((m.word_count for m in c.messages), default=0)
            if self._sort_field == "tokens":
                return sum(len(m.text or "") for m in c.messages) // 4
            return c.updated_at or dt_min

        return self._apply_sort_generic(conversations, sort_key)

    def _describe_active_filters(self) -> list[str]:
        """Return descriptions of all active filters (empty if no filters set)."""
        parts: list[str] = []
        if self._fts_terms:
            parts.append(f"contains: {', '.join(self._fts_terms)}")
        if self._providers:
            parts.append(f"provider: {', '.join(self._providers)}")
        if self._excluded_providers:
            parts.append(f"exclude provider: {', '.join(self._excluded_providers)}")
        if self._tags:
            parts.append(f"tag: {', '.join(self._tags)}")
        if self._excluded_tags:
            parts.append(f"exclude tag: {', '.join(self._excluded_tags)}")
        if self._has_types:
            parts.append(f"has: {', '.join(self._has_types)}")
        if self._filter_has_tool_use:
            parts.append("has_tool_use")
        if self._filter_has_thinking:
            parts.append("has_thinking")
        if self._filter_has_file_ops:
            parts.append("has_file_ops")
        if self._filter_has_git_ops:
            parts.append("has_git_ops")
        if self._filter_has_subagent:
            parts.append("has_subagent")
        if self._min_messages is not None:
            parts.append(f"min_messages: {self._min_messages}")
        if self._max_messages is not None:
            parts.append(f"max_messages: {self._max_messages}")
        if self._min_words is not None:
            parts.append(f"min_words: {self._min_words}")
        if self._since_date:
            parts.append(f"since: {self._since_date.isoformat()}")
        if self._until_date:
            parts.append(f"until: {self._until_date.isoformat()}")
        if self._title_pattern:
            parts.append(f"title: {self._title_pattern}")
        if self._id_prefix:
            parts.append(f"id: {self._id_prefix}")
        if self._negative_fts_terms:
            parts.append(f"exclude text: {', '.join(self._negative_fts_terms)}")
        if self._predicates:
            parts.append(f"custom predicates: {len(self._predicates)}")
        if self._similar_text:
            parts.append(f"similar: {self._similar_text[:30]}")
        return parts

    def _can_count_in_sql(self) -> bool:
        return not (
            self._fts_terms
            or self._id_prefix
            or self._similar_text
            or self._predicates
            or self._has_types
            or self._negative_fts_terms
            or self._excluded_providers
            or self._tags
            or self._excluded_tags
        )

    def describe(self) -> list[str]:
        """Return human-readable descriptions of active filters."""
        return self._describe_active_filters()

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
        """Calculate how many candidates to fetch from the backend."""
        if self._limit_count is None:
            return None
        if self._has_post_filters():
            return max(self._limit_count * 10, 500)
        if self._sample_count is not None:
            return max(self._sample_count * 3, 200)
        return max(self._limit_count * 2, 2)

    def _build_execution_plan(self) -> _ExecutionPlan:
        """Build the canonical internal execution plan for this filter."""
        return build_execution_plan(self)

    def _sql_pushdown_params(self) -> dict[str, object]:
        """Build kwargs for repository list/list_summaries that push filters to SQL."""
        return sql_pushdown_params(self)

    async def _fetch_generic(
        self,
        get_by_id: Callable[[str], Awaitable[_T | None]],
        search: Callable[[str, int, builtins.list[str] | None], Awaitable[builtins.list[_T]]],
        list_all: Callable[..., Awaitable[builtins.list[_T]]],
    ) -> builtins.list[_T]:
        """Fetch candidate items from repository using provided accessors."""
        if self._id_prefix and not self._fts_terms:
            resolved_id = await self._repo.resolve_id(self._id_prefix)
            if resolved_id:
                item = await get_by_id(str(resolved_id))
                return [item] if item else []

        plan = self._build_execution_plan()

        if self._fts_terms:
            query = " ".join(self._fts_terms)
            try:
                search_limit = max(plan.fetch_limit, 100) if plan.fetch_limit is not None else 10000
                return await search(query, search_limit, self._providers or None)
            except Exception as exc:
                logger.debug("FTS search failed, falling back to list: %s", exc)

        return await list_all(limit=plan.fetch_limit, **plan.sql_params)

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
        plan = self._build_execution_plan()
        filtered = apply_filters(candidates, plan.sql_pushed)
        sorted_results = apply_sort(filtered)

        if self._sample_count is not None and self._sample_count < len(sorted_results):
            sorted_results = random.sample(sorted_results, self._sample_count)
        if self._limit_count is not None:
            sorted_results = sorted_results[: self._limit_count]
        return sorted_results

    async def list(self) -> builtins.list[Conversation]:
        """Execute query and return matching conversations."""
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
        """Execute query and return first matching conversation."""
        results = await self.limit(1).list()
        return results[0] if results else None

    async def count(self) -> int:
        """Execute query and return count of matching conversations."""
        if self._can_count_in_sql():
            return await self._repo.count(**self._sql_pushdown_params())

        plan = self._build_execution_plan()
        if plan.can_use_summaries:
            saved_limit, self._limit_count = self._limit_count, None
            try:
                results = await self.list_summaries()
            finally:
                self._limit_count = saved_limit
            return len(results)

        saved_limit, self._limit_count = self._limit_count, None
        try:
            results = await self.list()
        finally:
            self._limit_count = saved_limit
        return len(results)

    async def delete(self) -> int:
        """Delete matching conversations."""
        if self._build_execution_plan().can_use_summaries:
            results: list[Conversation | ConversationSummary] = await self.list_summaries()
        else:
            results = await self.list()
        deleted_count = 0

        for conv in results:
            if await self._repo.delete_conversation(str(conv.id)):
                deleted_count += 1

        return deleted_count

    def _needs_content_loading(self) -> bool:
        """Check if any filters require loading message content."""
        if self._has_types:
            needs_messages = any(t in ("thinking", "tools", "attachments") for t in self._has_types)
            if needs_messages:
                return True
        if self._negative_fts_terms:
            return True
        if self._predicates:
            return True
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

    def _apply_summary_sort(
        self,
        summaries: builtins.list[ConversationSummary],
    ) -> builtins.list[ConversationSummary]:
        """Apply sorting to summary list (limited to date-based sorts)."""
        from datetime import timezone

        dt_min = datetime.min.replace(tzinfo=timezone.utc)
        return self._apply_sort_generic(summaries, lambda s: s.updated_at or dt_min)

    async def list_summaries(self) -> builtins.list[ConversationSummary]:
        """Execute query and return lightweight summaries (no messages loaded)."""
        plan = self._build_execution_plan()
        if plan.needs_content_loading:
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
        """Check if this filter can use lightweight summaries."""
        return self._build_execution_plan().can_use_summaries


__all__ = ["ConversationFilterRuntimeMixin", "_conversation_has_branches"]
