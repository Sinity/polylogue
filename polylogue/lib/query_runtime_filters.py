"""Conversation filtering helpers for immutable conversation query plans."""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Protocol, TypeVar, cast

from polylogue.lib.query_runtime_matching import (
    matches_action_sequence,
    matches_action_terms,
    matches_action_text_terms,
    matches_path_terms,
    matches_tool_terms,
)
from polylogue.lib.query_support import conversation_has_branches, provider_values

if TYPE_CHECKING:
    from polylogue.lib.models import Conversation
    from polylogue.lib.query_plan import ConversationQueryPlan


class _FilterableConversationLike(Protocol):
    provider: object
    updated_at: datetime | None
    display_title: str
    parent_id: str | None
    tags: list[str]
    id: str
    summary: str | None
    is_continuation: bool
    is_sidechain: bool
    is_root: bool


_T = TypeVar("_T", bound=_FilterableConversationLike)


def apply_common_filters(
    plan: ConversationQueryPlan,
    items: list[_T],
    *,
    sql_pushed: bool,
) -> list[_T]:
    results = list(items)

    if not sql_pushed:
        provider_set = set(provider_values(plan.providers))
        if provider_set:
            results = [item for item in results if str(item.provider) in provider_set]
        if plan.since:
            results = [item for item in results if item.updated_at and item.updated_at >= plan.since]
        if plan.until:
            results = [item for item in results if item.updated_at and item.updated_at <= plan.until]
        if plan.title:
            lowered = plan.title.lower()
            results = [item for item in results if item.display_title and lowered in item.display_title.lower()]
        if plan.parent_id:
            results = [item for item in results if str(item.parent_id or "") == plan.parent_id]

    if plan.excluded_providers:
        excluded = set(provider_values(plan.excluded_providers))
        results = [item for item in results if str(item.provider) not in excluded]
    if plan.tags:
        tag_set = set(plan.tags)
        results = [item for item in results if tag_set.intersection(item.tags)]
    if plan.excluded_tags:
        excluded_tags = set(plan.excluded_tags)
        results = [item for item in results if not excluded_tags.intersection(item.tags)]
    if plan.conversation_id:
        results = [item for item in results if str(item.id).startswith(plan.conversation_id)]
    if "summary" in plan.has_types:
        results = [item for item in results if item.summary]
    if plan.continuation is True:
        results = [item for item in results if item.is_continuation]
    if plan.continuation is False:
        results = [item for item in results if not item.is_continuation]
    if plan.sidechain is True:
        results = [item for item in results if item.is_sidechain]
    if plan.sidechain is False:
        results = [item for item in results if not item.is_sidechain]
    if plan.root is True:
        results = [item for item in results if item.is_root]
    if plan.root is False:
        results = [item for item in results if not item.is_root]

    return results


def _has_negative_term(conversation: Conversation, negative_terms: list[str]) -> bool:
    for message in conversation.messages:
        if not message.text:
            continue
        lowered = message.text.lower()
        for term in negative_terms:
            if term in lowered:
                return True
    return False


def apply_full_filters(
    plan: ConversationQueryPlan,
    conversations: list[Conversation],
    *,
    sql_pushed: bool,
) -> list[Conversation]:
    results = cast(list[Conversation], apply_common_filters(plan, conversations, sql_pushed=sql_pushed))

    if plan.has_types:
        for content_type in plan.has_types:
            if content_type == "thinking":
                results = [c for c in results if any(m.is_thinking for m in c.messages)]
            elif content_type == "tools":
                results = [c for c in results if any(m.is_tool_use for m in c.messages)]
            elif content_type == "attachments":
                results = [c for c in results if any(m.attachments for m in c.messages)]

    if plan.filter_has_tool_use:
        results = [c for c in results if any(m.is_tool_use for m in c.messages)]
    if plan.filter_has_thinking:
        results = [c for c in results if any(m.is_thinking for m in c.messages)]
    if plan.min_messages is not None:
        results = [c for c in results if len(c.messages) >= plan.min_messages]
    if plan.max_messages is not None:
        results = [c for c in results if len(c.messages) <= plan.max_messages]
    if plan.min_words is not None:
        results = [c for c in results if sum(len((m.text or "").split()) for m in c.messages) >= plan.min_words]

    if plan.negative_terms:
        negative_terms = [term.lower() for term in plan.negative_terms]
        results = [conversation for conversation in results if not _has_negative_term(conversation, negative_terms)]

    if plan.has_branches is True:
        results = [item for item in results if conversation_has_branches(item)]
    if plan.has_branches is False:
        results = [item for item in results if not conversation_has_branches(item)]

    for predicate in plan.predicates:
        results = [conversation for conversation in results if predicate(conversation)]

    if plan.path_terms:
        results = [conversation for conversation in results if matches_path_terms(plan, conversation)]
    if plan.action_terms or plan.excluded_action_terms:
        results = [conversation for conversation in results if matches_action_terms(plan, conversation)]
    if plan.action_sequence:
        results = [conversation for conversation in results if matches_action_sequence(plan, conversation)]
    if plan.action_text_terms:
        results = [conversation for conversation in results if matches_action_text_terms(plan, conversation)]
    if plan.tool_terms or plan.excluded_tool_terms:
        results = [conversation for conversation in results if matches_tool_terms(plan, conversation)]

    return results


__all__ = ["apply_common_filters", "apply_full_filters"]
