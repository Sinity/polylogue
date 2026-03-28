"""Typed conversation-query specification shared by CLI and MCP surfaces."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING

from polylogue.lib.query_plan import ConversationQueryPlan
from polylogue.lib.query_spec_parsing import (
    QUERY_ACTION_TYPES as _QUERY_ACTION_TYPES,
)
from polylogue.lib.query_spec_parsing import (
    QUERY_RETRIEVAL_LANES as _QUERY_RETRIEVAL_LANES,
)
from polylogue.lib.query_spec_parsing import (
    QuerySpecError,
    build_query_spec_from_params,
    describe_query_spec,
    query_spec_has_filters,
    query_spec_to_plan,
)
from polylogue.types import Provider

if TYPE_CHECKING:
    from polylogue.lib.filters import ConversationFilter
    from polylogue.lib.models import Conversation, ConversationSummary
    from polylogue.protocols import VectorProvider
    from polylogue.storage.repository import ConversationRepository

QUERY_ACTION_TYPES = _QUERY_ACTION_TYPES
QUERY_RETRIEVAL_LANES = _QUERY_RETRIEVAL_LANES


@dataclass(frozen=True)
class ConversationQuerySpec:
    """Canonical selection intent for conversation queries."""

    query_terms: tuple[str, ...] = ()
    contains_terms: tuple[str, ...] = ()
    exclude_text_terms: tuple[str, ...] = ()
    retrieval_lane: str = "auto"
    path_terms: tuple[str, ...] = ()
    action_terms: tuple[str, ...] = ()
    excluded_action_terms: tuple[str, ...] = ()
    action_sequence: tuple[str, ...] = ()
    action_text_terms: tuple[str, ...] = ()
    tool_terms: tuple[str, ...] = ()
    excluded_tool_terms: tuple[str, ...] = ()
    providers: tuple[Provider, ...] = ()
    excluded_providers: tuple[Provider, ...] = ()
    tags: tuple[str, ...] = ()
    excluded_tags: tuple[str, ...] = ()
    has_types: tuple[str, ...] = ()
    title: str | None = None
    conversation_id: str | None = None
    since: str | None = None
    until: str | None = None
    latest: bool = False
    sort: str | None = None
    reverse: bool = False
    limit: int | None = None
    sample: int | None = None
    # Stats-based SQL pushdown filters
    filter_has_tool_use: bool = False
    filter_has_thinking: bool = False
    min_messages: int | None = None
    max_messages: int | None = None
    min_words: int | None = None
    similar_text: str | None = None

    @classmethod
    def from_params(cls, params: Mapping[str, object]) -> ConversationQuerySpec:
        """Build a query spec from CLI-style parameter mapping."""
        return build_query_spec_from_params(cls, params)

    def describe(self) -> list[str]:
        """Human-readable filter descriptions for UX/error output."""
        return describe_query_spec(self)

    def has_filters(self) -> bool:
        """Whether the spec narrows conversation selection."""
        return query_spec_has_filters(self)

    def to_plan(
        self,
        *,
        vector_provider: VectorProvider | None = None,
    ) -> ConversationQueryPlan:
        """Compile the immutable spec to the canonical execution plan."""
        return query_spec_to_plan(self, vector_provider=vector_provider)

    async def list(
        self,
        repository: ConversationRepository,
        *,
        vector_provider: VectorProvider | None = None,
    ) -> list[Conversation]:
        return await self.build_filter(repository, vector_provider=vector_provider).list()

    async def list_summaries(
        self,
        repository: ConversationRepository,
        *,
        vector_provider: VectorProvider | None = None,
    ) -> list[ConversationSummary]:
        return await self.build_filter(repository, vector_provider=vector_provider).list_summaries()

    async def count(
        self,
        repository: ConversationRepository,
        *,
        vector_provider: VectorProvider | None = None,
    ) -> int:
        return await self.build_filter(repository, vector_provider=vector_provider).count()

    async def delete(
        self,
        repository: ConversationRepository,
        *,
        vector_provider: VectorProvider | None = None,
    ) -> int:
        return await self.build_filter(repository, vector_provider=vector_provider).delete()

    def build_filter(
        self,
        repository: ConversationRepository,
        *,
        vector_provider: VectorProvider | None = None,
    ) -> ConversationFilter:
        """Build a fluent filter façade over the canonical execution plan."""
        from polylogue.lib.filters import ConversationFilter

        return ConversationFilter(
            repository,
            vector_provider=vector_provider,
            query_plan=self.to_plan(vector_provider=vector_provider),
        )


__all__ = ["ConversationQuerySpec", "QuerySpecError"]
