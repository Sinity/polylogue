from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from datetime import datetime
from typing import TYPE_CHECKING, Protocol, Self, TypeVar

from polylogue.lib.dates import parse_date
from polylogue.lib.filter.types import SortField
from polylogue.types import Provider

if TYPE_CHECKING:
    from polylogue.lib.models import Conversation
    from polylogue.lib.query.plan import ConversationQueryPlan

_T = TypeVar("_T")


class _HasQueryPlan(Protocol):
    _plan: ConversationQueryPlan


_PlanOwner = TypeVar("_PlanOwner", bound=_HasQueryPlan)
_ReplacePlan: Callable[..., ConversationQueryPlan] = replace


def _extend_tuple(values: tuple[_T, ...], additions: tuple[_T, ...]) -> tuple[_T, ...]:
    return values + additions


def _replace_plan(
    filter_obj: _PlanOwner,
    **changes: object,
) -> _PlanOwner:
    plan = filter_obj._plan
    # dataclasses.replace is typed per field and cannot express this fluent
    # builder's one-field-at-a-time updates.
    filter_obj._plan = _ReplacePlan(plan, **changes)
    return filter_obj


class ConversationFilterBuilderMixin:
    """Fluent mutators for ConversationFilter backed by the canonical plan."""

    _plan: ConversationQueryPlan

    def contains(self, text: str) -> Self:
        return _replace_plan(
            self,
            contains_terms=_extend_tuple(self._plan.contains_terms, (text,)),
        )

    def exclude_text(self, text: str) -> Self:
        return _replace_plan(
            self,
            negative_terms=_extend_tuple(self._plan.negative_terms, (text,)),
        )

    def provider(self, *names: Provider | str) -> Self:
        providers = tuple(name if isinstance(name, Provider) else Provider.from_string(name) for name in names)
        return _replace_plan(
            self,
            providers=_extend_tuple(self._plan.providers, providers),
        )

    def exclude_provider(self, *names: Provider | str) -> Self:
        providers = tuple(name if isinstance(name, Provider) else Provider.from_string(name) for name in names)
        return _replace_plan(
            self,
            excluded_providers=_extend_tuple(self._plan.excluded_providers, providers),
        )

    def tag(self, *tags: str) -> Self:
        return _replace_plan(
            self,
            tags=_extend_tuple(self._plan.tags, tuple(tags)),
        )

    def exclude_tag(self, *tags: str) -> Self:
        return _replace_plan(
            self,
            excluded_tags=_extend_tuple(self._plan.excluded_tags, tuple(tags)),
        )

    def has(self, *types: str) -> Self:
        return _replace_plan(
            self,
            has_types=_extend_tuple(self._plan.has_types, tuple(types)),
        )

    def since(self, date: str | datetime) -> Self:
        parsed = parse_date(date) if isinstance(date, str) else date
        if parsed is None:
            msg = f"Cannot parse date: {date!r}"
            raise ValueError(msg)
        return _replace_plan(self, since=parsed)

    def until(self, date: str | datetime) -> Self:
        parsed = parse_date(date) if isinstance(date, str) else date
        if parsed is None:
            msg = f"Cannot parse date: {date!r}"
            raise ValueError(msg)
        return _replace_plan(self, until=parsed)

    def title(self, pattern: str) -> Self:
        return _replace_plan(self, title=pattern)

    def path(self, pattern: str) -> Self:
        return _replace_plan(
            self,
            path_terms=_extend_tuple(self._plan.path_terms, (pattern,)),
        )

    def cwd_prefix(self, prefix: str) -> Self:
        return _replace_plan(self, cwd_prefix=prefix)

    def action(self, *types: str) -> Self:
        return _replace_plan(
            self,
            action_terms=_extend_tuple(self._plan.action_terms, tuple(types)),
        )

    def exclude_action(self, *types: str) -> Self:
        return _replace_plan(
            self,
            excluded_action_terms=_extend_tuple(self._plan.excluded_action_terms, tuple(types)),
        )

    def tool(self, *names: str) -> Self:
        normalized = tuple(name.strip().lower() for name in names if name.strip())
        return _replace_plan(
            self,
            tool_terms=_extend_tuple(self._plan.tool_terms, normalized),
        )

    def exclude_tool(self, *names: str) -> Self:
        normalized = tuple(name.strip().lower() for name in names if name.strip())
        return _replace_plan(
            self,
            excluded_tool_terms=_extend_tuple(self._plan.excluded_tool_terms, normalized),
        )

    def id(self, prefix: str) -> Self:
        return _replace_plan(self, conversation_id=prefix)

    def sort(self, field: SortField) -> Self:
        return _replace_plan(self, sort=field)

    def reverse(self) -> Self:
        return _replace_plan(self, reverse=True)

    def limit(self, n: int) -> Self:
        return _replace_plan(self, limit=n)

    def sample(self, n: int) -> Self:
        return _replace_plan(self, sample=n)

    def similar(self, text: str) -> Self:
        return _replace_plan(self, similar_text=text)

    def where(self, predicate: Callable[[Conversation], bool]) -> Self:
        return _replace_plan(
            self,
            predicates=_extend_tuple(self._plan.predicates, (predicate,)),
        )

    def is_continuation(self, value: bool = True) -> Self:
        return _replace_plan(self, continuation=value)

    def is_sidechain(self, value: bool = True) -> Self:
        return _replace_plan(self, sidechain=value)

    def is_root(self, value: bool = True) -> Self:
        return _replace_plan(self, root=value)

    def has_tool_use(self) -> Self:
        return _replace_plan(self, filter_has_tool_use=True)

    def has_thinking(self) -> Self:
        return _replace_plan(self, filter_has_thinking=True)

    def min_messages(self, n: int) -> Self:
        return _replace_plan(self, min_messages=n)

    def max_messages(self, n: int) -> Self:
        return _replace_plan(self, max_messages=n)

    def min_words(self, n: int) -> Self:
        return _replace_plan(self, min_words=n)

    def has_file_operations(self) -> Self:
        return self.action("file_read", "file_write", "file_edit")

    def has_git_operations(self) -> Self:
        return self.action("git")

    def has_subagent_spawns(self) -> Self:
        return self.action("subagent")

    def parent(self, conversation_id: str) -> Self:
        return _replace_plan(self, parent_id=conversation_id)

    def has_branches(self, value: bool = True) -> Self:
        return _replace_plan(self, has_branches=value)


__all__ = ["ConversationFilterBuilderMixin"]
