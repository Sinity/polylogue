from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from polylogue.lib.filters import ConversationFilter


@dataclass(frozen=True)
class _ExecutionPlan:
    sql_params: dict[str, object]
    fetch_limit: int | None
    has_post_filters: bool
    needs_content_loading: bool
    can_use_summaries: bool
    sql_pushed: bool


def sql_pushdown_params(f: ConversationFilter) -> dict[str, object]:
    """Build kwargs for repository list/list_summaries that push filters to SQL."""
    params: dict[str, object] = {}
    if f._providers:
        if len(f._providers) == 1:
            params["provider"] = f._providers[0]
        else:
            params["providers"] = f._providers
    if f._since_date:
        params["since"] = f._since_date.isoformat()
    if f._until_date:
        params["until"] = f._until_date.isoformat()
    if f._title_pattern:
        params["title_contains"] = f._title_pattern
    if f._filter_has_tool_use:
        params["has_tool_use"] = True
    if f._filter_has_thinking:
        params["has_thinking"] = True
    if f._min_messages is not None:
        params["min_messages"] = f._min_messages
    if f._max_messages is not None:
        params["max_messages"] = f._max_messages
    if f._min_words is not None:
        params["min_words"] = f._min_words
    if f._filter_has_file_ops:
        params["has_file_ops"] = True
    if f._filter_has_git_ops:
        params["has_git_ops"] = True
    if f._filter_has_subagent:
        params["has_subagent"] = True
    return params


def build_execution_plan(f: ConversationFilter) -> _ExecutionPlan:
    """Build the canonical execution plan for the given filter."""
    needs_content_loading = f._needs_content_loading()
    has_post_filters = f._has_post_filters()
    params = sql_pushdown_params(f)
    return _ExecutionPlan(
        sql_params=params,
        fetch_limit=f._effective_fetch_limit(),
        has_post_filters=has_post_filters,
        needs_content_loading=needs_content_loading,
        can_use_summaries=not needs_content_loading,
        sql_pushed=not f._fts_terms and not f._id_prefix,
    )


__all__ = ["_ExecutionPlan", "build_execution_plan", "sql_pushdown_params"]
