from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from datetime import datetime
from typing import TYPE_CHECKING

from polylogue.lib.dates import parse_date
from polylogue.lib.filter_types import SortField
from polylogue.types import Provider

if TYPE_CHECKING:
    from polylogue.lib.filters import ConversationFilter
    from polylogue.lib.models import Conversation
    from polylogue.lib.query_plan import ConversationQueryPlan


def _extend_tuple(values: tuple[object, ...], additions: tuple[object, ...]) -> tuple[object, ...]:
    return values + additions


def _replace_plan(
    filter_obj: ConversationFilter,
    **changes: object,
) -> ConversationFilter:
    plan: ConversationQueryPlan = filter_obj._plan
    filter_obj._plan = replace(plan, **changes)
    return filter_obj


class ConversationFilterBuilderMixin:
    """Fluent mutators for ConversationFilter backed by the canonical plan."""

    def contains(self, text: str) -> ConversationFilter:
        return _replace_plan(
            self,
            contains_terms=_extend_tuple(self._plan.contains_terms, (text,)),
        )

    def exclude_text(self, text: str) -> ConversationFilter:
        return _replace_plan(
            self,
            negative_terms=_extend_tuple(self._plan.negative_terms, (text,)),
        )

    def provider(self, *names: Provider | str) -> ConversationFilter:
        providers = tuple(name if isinstance(name, Provider) else Provider.from_string(name) for name in names)
        return _replace_plan(
            self,
            providers=_extend_tuple(self._plan.providers, providers),
        )

    def exclude_provider(self, *names: Provider | str) -> ConversationFilter:
        providers = tuple(name if isinstance(name, Provider) else Provider.from_string(name) for name in names)
        return _replace_plan(
            self,
            excluded_providers=_extend_tuple(self._plan.excluded_providers, providers),
        )

    def tag(self, *tags: str) -> ConversationFilter:
        return _replace_plan(
            self,
            tags=_extend_tuple(self._plan.tags, tuple(tags)),
        )

    def exclude_tag(self, *tags: str) -> ConversationFilter:
        return _replace_plan(
            self,
            excluded_tags=_extend_tuple(self._plan.excluded_tags, tuple(tags)),
        )

    def has(self, *types: str) -> ConversationFilter:
        return _replace_plan(
            self,
            has_types=_extend_tuple(self._plan.has_types, tuple(types)),
        )

    def since(self, date: str | datetime) -> ConversationFilter:
        parsed = parse_date(date) if isinstance(date, str) else date
        if parsed is None:
            msg = f"Cannot parse date: {date!r}"
            raise ValueError(msg)
        return _replace_plan(self, since=parsed)

    def until(self, date: str | datetime) -> ConversationFilter:
        parsed = parse_date(date) if isinstance(date, str) else date
        if parsed is None:
            msg = f"Cannot parse date: {date!r}"
            raise ValueError(msg)
        return _replace_plan(self, until=parsed)

    def title(self, pattern: str) -> ConversationFilter:
        return _replace_plan(self, title=pattern)

    def path(self, pattern: str) -> ConversationFilter:
        return _replace_plan(
            self,
            path_terms=_extend_tuple(self._plan.path_terms, (pattern,)),
        )

    def action(self, *types: str) -> ConversationFilter:
        return _replace_plan(
            self,
            action_terms=_extend_tuple(self._plan.action_terms, tuple(types)),
        )

    def exclude_action(self, *types: str) -> ConversationFilter:
        return _replace_plan(
            self,
            excluded_action_terms=_extend_tuple(self._plan.excluded_action_terms, tuple(types)),
        )

    def tool(self, *names: str) -> ConversationFilter:
        normalized = tuple(name.strip().lower() for name in names if name.strip())
        return _replace_plan(
            self,
            tool_terms=_extend_tuple(self._plan.tool_terms, normalized),
        )

    def exclude_tool(self, *names: str) -> ConversationFilter:
        normalized = tuple(name.strip().lower() for name in names if name.strip())
        return _replace_plan(
            self,
            excluded_tool_terms=_extend_tuple(self._plan.excluded_tool_terms, normalized),
        )

    def id(self, prefix: str) -> ConversationFilter:
        return _replace_plan(self, conversation_id=prefix)

    def sort(self, field: SortField) -> ConversationFilter:
        return _replace_plan(self, sort=field)

    def reverse(self) -> ConversationFilter:
        return _replace_plan(self, reverse=True)

    def limit(self, n: int) -> ConversationFilter:
        return _replace_plan(self, limit=n)

    def sample(self, n: int) -> ConversationFilter:
        return _replace_plan(self, sample=n)

    def similar(self, text: str) -> ConversationFilter:
        return _replace_plan(self, similar_text=text)

    def where(self, predicate: Callable[[Conversation], bool]) -> ConversationFilter:
        return _replace_plan(
            self,
            predicates=_extend_tuple(self._plan.predicates, (predicate,)),
        )

    def is_continuation(self, value: bool = True) -> ConversationFilter:
        return _replace_plan(self, continuation=value)

    def is_sidechain(self, value: bool = True) -> ConversationFilter:
        return _replace_plan(self, sidechain=value)

    def is_root(self, value: bool = True) -> ConversationFilter:
        return _replace_plan(self, root=value)

    def has_tool_use(self) -> ConversationFilter:
        return _replace_plan(self, filter_has_tool_use=True)

    def has_thinking(self) -> ConversationFilter:
        return _replace_plan(self, filter_has_thinking=True)

    def min_messages(self, n: int) -> ConversationFilter:
        return _replace_plan(self, min_messages=n)

    def max_messages(self, n: int) -> ConversationFilter:
        return _replace_plan(self, max_messages=n)

    def min_words(self, n: int) -> ConversationFilter:
        return _replace_plan(self, min_words=n)

    def has_file_operations(self) -> ConversationFilter:
        return self.action("file_read", "file_write", "file_edit")

    def has_git_operations(self) -> ConversationFilter:
        return self.action("git")

    def has_subagent_spawns(self) -> ConversationFilter:
        return self.action("subagent")

    def parent(self, conversation_id: str) -> ConversationFilter:
        return _replace_plan(self, parent_id=conversation_id)

    def has_branches(self, value: bool = True) -> ConversationFilter:
        return _replace_plan(self, has_branches=value)


__all__ = ["ConversationFilterBuilderMixin"]
