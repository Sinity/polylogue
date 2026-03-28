"""Runtime filtering and sorting mixin methods for conversation plans."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, TypeVar

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
from polylogue.lib.query_sorting import finalize_results, sort_conversations, sort_generic, sort_summaries

if TYPE_CHECKING:
    from polylogue.lib.models import Conversation, ConversationSummary

_T = TypeVar("_T")


class QueryPlanRuntimeMixin:
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

    def _apply_common_filters(self, items: list[_T], *, sql_pushed: bool) -> list[_T]:
        return apply_common_filters(self, items, sql_pushed=sql_pushed)

    def _apply_full_filters(self, conversations: list[Conversation], *, sql_pushed: bool) -> list[Conversation]:
        return apply_full_filters(self, conversations, sql_pushed=sql_pushed)

    def _sort_generic(self, items: list[_T], key_fn: Callable[[_T], Any]) -> list[_T]:
        return sort_generic(self, items, key_fn)

    def _sort_conversations(self, conversations: list[Conversation]) -> list[Conversation]:
        return sort_conversations(self, conversations)

    def _sort_summaries(self, summaries: list[ConversationSummary]) -> list[ConversationSummary]:
        return sort_summaries(self, summaries)

    def _finalize(self, items: list[_T]) -> list[_T]:
        return finalize_results(self, items)


__all__ = ["QueryPlanRuntimeMixin"]
