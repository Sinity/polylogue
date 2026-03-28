"""Canonical immutable conversation-query plan model."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING

from polylogue.lib.query_plan_description_mixin import QueryPlanDescriptionMixin
from polylogue.lib.query_plan_execution_mixin import QueryPlanExecutionMixin
from polylogue.lib.query_plan_runtime_mixin import QueryPlanRuntimeMixin
from polylogue.lib.query_support import conversation_has_branches
from polylogue.types import Provider

if TYPE_CHECKING:
    from polylogue.lib.filter_types import SortField
    from polylogue.lib.models import Conversation
    from polylogue.protocols import VectorProvider


@dataclass(frozen=True)
class ConversationQueryPlan(
    QueryPlanDescriptionMixin,
    QueryPlanExecutionMixin,
    QueryPlanRuntimeMixin,
):
    """Canonical immutable execution state for conversation selection."""

    query_terms: tuple[str, ...] = ()
    contains_terms: tuple[str, ...] = ()
    negative_terms: tuple[str, ...] = ()
    retrieval_lane: str = "auto"
    path_terms: tuple[str, ...] = ()
    action_terms: tuple[str, ...] = ()
    excluded_action_terms: tuple[str, ...] = ()
    action_sequence: tuple[str, ...] = ()
    action_text_terms: tuple[str, ...] = ()
    tool_terms: tuple[str, ...] = ()
    excluded_tool_terms: tuple[str, ...] = ()
    providers: tuple[Provider | str, ...] = ()
    excluded_providers: tuple[Provider | str, ...] = ()
    tags: tuple[str, ...] = ()
    excluded_tags: tuple[str, ...] = ()
    has_types: tuple[str, ...] = ()
    title: str | None = None
    conversation_id: str | None = None
    parent_id: str | None = None
    since: datetime | None = None
    until: datetime | None = None
    sort: SortField = "date"
    reverse: bool = False
    limit: int | None = None
    sample: int | None = None
    similar_text: str | None = None
    predicates: tuple[Callable[[Conversation], bool], ...] = ()
    continuation: bool | None = None
    sidechain: bool | None = None
    root: bool | None = None
    has_branches: bool | None = None
    filter_has_tool_use: bool = False
    filter_has_thinking: bool = False
    min_messages: int | None = None
    max_messages: int | None = None
    min_words: int | None = None
    vector_provider: VectorProvider | None = None


__all__ = ["ConversationQueryPlan", "conversation_has_branches"]
