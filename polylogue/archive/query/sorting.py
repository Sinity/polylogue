"""Sorting and finalization helpers for immutable conversation query plans."""

from __future__ import annotations

import random
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Protocol, TypeAlias, TypeVar

if TYPE_CHECKING:
    from polylogue.lib.models import Conversation, ConversationSummary

from polylogue.archive.filter.types import SortField

_T = TypeVar("_T")
SortKey: TypeAlias = datetime | float | int | str


class QuerySortPlan(Protocol):
    """Minimal plan surface required by result sorting/finalization."""

    @property
    def sort(self) -> SortField: ...

    @property
    def reverse(self) -> bool: ...

    @property
    def limit(self) -> int | None: ...

    @property
    def sample(self) -> int | None: ...


@dataclass(frozen=True, slots=True)
class ResultWindow:
    """Typed sampling and limit settings for finalized query results."""

    sample: int | None = None
    limit: int | None = None

    @classmethod
    def from_plan(cls, plan: QuerySortPlan) -> ResultWindow:
        return cls(sample=plan.sample, limit=plan.limit)


def sort_generic(
    plan: QuerySortPlan,
    items: list[_T],
    key_fn: Callable[[_T], SortKey],
) -> list[_T]:
    if plan.sort == "random":
        shuffled = list(items)
        random.shuffle(shuffled)
        return shuffled
    return sorted(items, key=key_fn, reverse=not plan.reverse)


def sort_conversations(
    plan: QuerySortPlan,
    conversations: list[Conversation],
) -> list[Conversation]:
    dt_min = datetime.min.replace(tzinfo=timezone.utc)

    def _key(conversation: Conversation) -> SortKey:
        if plan.sort == "date":
            return conversation.updated_at or dt_min
        if plan.sort == "messages":
            return len(conversation.messages)
        if plan.sort == "words":
            return sum(message.word_count for message in conversation.messages)
        if plan.sort == "longest":
            return max((message.word_count for message in conversation.messages), default=0)
        if plan.sort == "tokens":
            return sum(len(message.text or "") for message in conversation.messages) // 4
        return conversation.updated_at or dt_min

    return sort_generic(plan, conversations, _key)


def sort_summaries(
    plan: QuerySortPlan,
    summaries: list[ConversationSummary],
) -> list[ConversationSummary]:
    dt_min = datetime.min.replace(tzinfo=timezone.utc)
    return sort_generic(plan, summaries, lambda summary: summary.updated_at or dt_min)


def finalize_results(
    plan: QuerySortPlan,
    items: list[_T],
) -> list[_T]:
    return finalize_window(ResultWindow.from_plan(plan), items)


def finalize_window(
    window: ResultWindow,
    items: list[_T],
) -> list[_T]:
    """Apply a typed result window to already sorted query results."""
    results = list(items)
    if window.sample is not None and window.sample < len(results):
        results = random.sample(results, window.sample)
    if window.limit is not None:
        results = results[: window.limit]
    return results


__all__ = [
    "finalize_results",
    "finalize_window",
    "QuerySortPlan",
    "ResultWindow",
    "sort_conversations",
    "sort_generic",
    "sort_summaries",
]
