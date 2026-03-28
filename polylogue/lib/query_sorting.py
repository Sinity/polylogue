"""Sorting and finalization helpers for immutable conversation query plans."""

from __future__ import annotations

import random
from collections.abc import Callable
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, TypeVar

if TYPE_CHECKING:
    from polylogue.lib.models import Conversation, ConversationSummary
    from polylogue.lib.query_plan import ConversationQueryPlan

_T = TypeVar("_T")


def sort_generic(
    plan: ConversationQueryPlan,
    items: list[_T],
    key_fn: Callable[[_T], Any],
) -> list[_T]:
    if plan.sort == "random":
        shuffled = list(items)
        random.shuffle(shuffled)
        return shuffled
    return sorted(items, key=key_fn, reverse=not plan.reverse)


def sort_conversations(
    plan: ConversationQueryPlan,
    conversations: list[Conversation],
) -> list[Conversation]:
    dt_min = datetime.min.replace(tzinfo=timezone.utc)

    def _key(conversation: Conversation) -> Any:
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
    plan: ConversationQueryPlan,
    summaries: list[ConversationSummary],
) -> list[ConversationSummary]:
    dt_min = datetime.min.replace(tzinfo=timezone.utc)
    return sort_generic(plan, summaries, lambda summary: summary.updated_at or dt_min)


def finalize_results(
    plan: ConversationQueryPlan,
    items: list[_T],
) -> list[_T]:
    results = list(items)
    if plan.sample is not None and plan.sample < len(results):
        results = random.sample(results, plan.sample)
    if plan.limit is not None:
        results = results[: plan.limit]
    return results


__all__ = [
    "finalize_results",
    "sort_conversations",
    "sort_generic",
    "sort_summaries",
]
