"""Conversation filtering helpers for immutable conversation query plans."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from typing import TYPE_CHECKING, Protocol, TypeVar

from polylogue.lib.query.runtime_matching import (
    matches_action_sequence,
    matches_action_terms,
    matches_action_text_terms,
    matches_path_terms,
    matches_tool_terms,
)
from polylogue.lib.query.support import conversation_has_branches, provider_values

if TYPE_CHECKING:
    from polylogue.lib.models import Conversation
    from polylogue.lib.query.plan import ConversationQueryPlan


class FilterableConversationLike(Protocol):
    @property
    def provider(self) -> object: ...

    @property
    def updated_at(self) -> datetime | None: ...

    @property
    def display_title(self) -> str: ...

    @property
    def parent_id(self) -> str | None: ...

    @property
    def tags(self) -> Sequence[str]: ...

    @property
    def id(self) -> str: ...

    @property
    def summary(self) -> str | None: ...

    @property
    def is_continuation(self) -> bool: ...

    @property
    def is_sidechain(self) -> bool: ...

    @property
    def is_root(self) -> bool: ...


_T = TypeVar("_T", bound=FilterableConversationLike)


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
    results = apply_common_filters(plan, conversations, sql_pushed=sql_pushed)

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

    if plan.since_session_id:
        results = _apply_since_session(results, plan.since_session_id)

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


def _apply_since_session(
    conversations: list[Conversation],
    reference_id: str,
) -> list[Conversation]:
    """Filter to conversations in same cwd after the reference session's last message."""
    reference_conv = None
    for c in conversations:
        if str(c.id) == reference_id or str(c.id).startswith(reference_id):
            reference_conv = c
            break

    if reference_conv is None:
        return conversations

    ref_cwds: list[str] = []
    ref_meta = getattr(reference_conv, "provider_meta", None) or {}
    if isinstance(ref_meta, dict):
        wds = ref_meta.get("working_directories") or []
        ref_cwds = [str(wd) for wd in wds if isinstance(wd, str) and wd]

    last_ts = reference_conv.updated_at
    if reference_conv.messages:
        last_msg_ts = max(
            (m.timestamp for m in reference_conv.messages if m.timestamp),
            default=None,
        )
        if last_msg_ts:
            last_ts = last_msg_ts

    results: list[Conversation] = []
    for c in conversations:
        if str(c.id) == str(reference_conv.id):
            continue
        if last_ts and c.updated_at and c.updated_at <= last_ts:
            continue
        if ref_cwds:
            c_meta = getattr(c, "provider_meta", None) or {}
            if isinstance(c_meta, dict):
                c_wds = c_meta.get("working_directories") or []
                c_wd_strs = [str(wd) for wd in c_wds if isinstance(wd, str) and wd]
                if c_wd_strs and not any(str(cwd).startswith(ref_cwd) for cwd in c_wd_strs for ref_cwd in ref_cwds):
                    continue
        results.append(c)

    return results


__all__ = ["FilterableConversationLike", "apply_common_filters", "apply_full_filters"]
