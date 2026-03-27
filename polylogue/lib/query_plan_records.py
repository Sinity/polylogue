"""Record-query translation helpers for immutable conversation query plans."""

from __future__ import annotations

from typing import TYPE_CHECKING

from polylogue.lib.query_support import provider_values
from polylogue.storage.query_models import ConversationRecordQuery

if TYPE_CHECKING:
    from polylogue.lib.query_plan import ConversationQueryPlan


def plan_record_query(plan: ConversationQueryPlan) -> ConversationRecordQuery:
    values = provider_values(plan.providers)
    provider = values[0] if len(values) == 1 else None
    providers = values if len(values) > 1 else ()
    return ConversationRecordQuery(
        provider=provider,
        providers=providers,
        parent_id=plan.parent_id,
        since=plan.since.isoformat() if plan.since else None,
        until=plan.until.isoformat() if plan.until else None,
        title_contains=plan.title,
        path_terms=plan.path_terms,
        action_terms=plan.action_terms,
        excluded_action_terms=plan.excluded_action_terms,
        tool_terms=plan.tool_terms,
        excluded_tool_terms=plan.excluded_tool_terms,
        has_tool_use=plan.filter_has_tool_use,
        has_thinking=plan.filter_has_thinking,
        min_messages=plan.min_messages,
        max_messages=plan.max_messages,
        min_words=plan.min_words,
    )


def plan_sql_pushdown_params(plan: ConversationQueryPlan) -> dict[str, object]:
    params: dict[str, object] = {}
    values = provider_values(plan.providers)
    if len(values) == 1:
        params["provider"] = values[0]
    elif values:
        params["providers"] = list(values)
    if plan.parent_id:
        params["parent_id"] = plan.parent_id
    if plan.since:
        params["since"] = plan.since.isoformat()
    if plan.until:
        params["until"] = plan.until.isoformat()
    if plan.title:
        params["title_contains"] = plan.title
    if plan.path_terms:
        params["path_terms"] = list(plan.path_terms)
    if plan.action_terms:
        params["action_terms"] = list(plan.action_terms)
    if plan.excluded_action_terms:
        params["excluded_action_terms"] = list(plan.excluded_action_terms)
    if plan.action_sequence:
        params["action_sequence"] = list(plan.action_sequence)
    if plan.action_text_terms:
        params["action_text_terms"] = list(plan.action_text_terms)
    if plan.tool_terms:
        params["tool_terms"] = list(plan.tool_terms)
    if plan.excluded_tool_terms:
        params["excluded_tool_terms"] = list(plan.excluded_tool_terms)
    if plan.filter_has_tool_use:
        params["has_tool_use"] = True
    if plan.filter_has_thinking:
        params["has_thinking"] = True
    if plan.min_messages is not None:
        params["min_messages"] = plan.min_messages
    if plan.max_messages is not None:
        params["max_messages"] = plan.max_messages
    if plan.min_words is not None:
        params["min_words"] = plan.min_words
    return params


__all__ = [
    "plan_record_query",
    "plan_sql_pushdown_params",
]
