"""Typed conversation-query specification shared by CLI and MCP surfaces."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING

from polylogue.types import Provider

if TYPE_CHECKING:
    from polylogue.lib.filters import ConversationFilter
    from polylogue.protocols import VectorProvider
    from polylogue.storage.repository import ConversationRepository


class QuerySpecError(ValueError):
    """Typed query-spec construction/application error."""

    def __init__(self, field: str, value: str) -> None:
        super().__init__(f"invalid {field}: {value}")
        self.field = field
        self.value = value


def _split_csv(value: object) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return tuple(part.strip() for part in value.split(",") if part.strip())
    return tuple(str(item).strip() for item in value if str(item).strip())


def _as_tuple(value: object) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    return tuple(str(item) for item in value)


@dataclass(frozen=True)
class ConversationQuerySpec:
    """Canonical selection intent for conversation queries."""

    query_terms: tuple[str, ...] = ()
    contains_terms: tuple[str, ...] = ()
    exclude_text_terms: tuple[str, ...] = ()
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

    @classmethod
    def from_params(cls, params: Mapping[str, object]) -> ConversationQuerySpec:
        """Build a query spec from CLI-style parameter mapping."""
        return cls(
            query_terms=_as_tuple(params.get("query")),
            contains_terms=_as_tuple(params.get("contains")),
            exclude_text_terms=_as_tuple(params.get("exclude_text")),
            providers=tuple(Provider.from_string(p) for p in _split_csv(params.get("provider"))),
            excluded_providers=tuple(Provider.from_string(p) for p in _split_csv(params.get("exclude_provider"))),
            tags=_split_csv(params.get("tag")),
            excluded_tags=_split_csv(params.get("exclude_tag")),
            has_types=_as_tuple(params.get("has_type")),
            title=str(params["title"]) if params.get("title") else None,
            conversation_id=str(params["conv_id"]) if params.get("conv_id") else None,
            since=str(params["since"]) if params.get("since") else None,
            until=str(params["until"]) if params.get("until") else None,
            latest=bool(params.get("latest")),
            sort=str(params["sort"]) if params.get("sort") else None,
            reverse=bool(params.get("reverse")),
            limit=int(params["limit"]) if params.get("limit") else None,
            sample=int(params["sample"]) if params.get("sample") else None,
            filter_has_tool_use=bool(params.get("filter_has_tool_use")),
            filter_has_thinking=bool(params.get("filter_has_thinking")),
            min_messages=int(params["min_messages"]) if params.get("min_messages") else None,
            max_messages=int(params["max_messages"]) if params.get("max_messages") else None,
            min_words=int(params["min_words"]) if params.get("min_words") else None,
        )

    def describe(self) -> list[str]:
        """Human-readable filter descriptions for UX/error output."""
        parts: list[str] = []
        if self.query_terms:
            parts.append(f"search: {' '.join(self.query_terms)}")
        if self.contains_terms:
            parts.append(f"contains: {', '.join(self.contains_terms)}")
        if self.exclude_text_terms:
            parts.append(f"exclude text: {', '.join(self.exclude_text_terms)}")
        if self.providers:
            parts.append(f"provider: {', '.join(p.value for p in self.providers)}")
        if self.excluded_providers:
            parts.append(f"exclude provider: {', '.join(p.value for p in self.excluded_providers)}")
        if self.tags:
            parts.append(f"tag: {', '.join(self.tags)}")
        if self.excluded_tags:
            parts.append(f"exclude tag: {', '.join(self.excluded_tags)}")
        if self.title:
            parts.append(f"title: {self.title}")
        if self.has_types:
            parts.append(f"has: {', '.join(self.has_types)}")
        if self.filter_has_tool_use:
            parts.append("has: tool_use (sql)")
        if self.filter_has_thinking:
            parts.append("has: thinking (sql)")
        if self.min_messages is not None:
            parts.append(f"min_messages: {self.min_messages}")
        if self.max_messages is not None:
            parts.append(f"max_messages: {self.max_messages}")
        if self.min_words is not None:
            parts.append(f"min_words: {self.min_words}")
        if self.since:
            parts.append(f"since: {self.since}")
        if self.until:
            parts.append(f"until: {self.until}")
        if self.conversation_id:
            parts.append(f"id: {self.conversation_id}")
        return parts

    def has_filters(self) -> bool:
        """Whether the spec narrows conversation selection."""
        return any(
            (
                self.query_terms,
                self.contains_terms,
                self.exclude_text_terms,
                self.providers,
                self.excluded_providers,
                self.tags,
                self.excluded_tags,
                self.has_types,
                self.title is not None,
                self.conversation_id is not None,
                self.since is not None,
                self.until is not None,
                self.latest,
                self.filter_has_tool_use,
                self.filter_has_thinking,
                self.min_messages is not None,
                self.max_messages is not None,
                self.min_words is not None,
            )
        )

    def build_filter(
        self,
        repository: ConversationRepository,
        *,
        vector_provider: VectorProvider | None = None,
    ) -> ConversationFilter:
        """Compile the spec to a canonical ConversationFilter."""
        from polylogue.lib.filters import ConversationFilter

        filter_chain = ConversationFilter(repository, vector_provider=vector_provider)

        if self.conversation_id:
            filter_chain = filter_chain.id(self.conversation_id)

        for term in self.query_terms:
            filter_chain = filter_chain.contains(term)
        for term in self.contains_terms:
            filter_chain = filter_chain.contains(term)
        for term in self.exclude_text_terms:
            filter_chain = filter_chain.exclude_text(term)
        if self.providers:
            filter_chain = filter_chain.provider(*self.providers)
        if self.excluded_providers:
            filter_chain = filter_chain.exclude_provider(*self.excluded_providers)
        if self.tags:
            filter_chain = filter_chain.tag(*self.tags)
        if self.excluded_tags:
            filter_chain = filter_chain.exclude_tag(*self.excluded_tags)
        if self.title:
            filter_chain = filter_chain.title(self.title)
        for content_type in self.has_types:
            filter_chain = filter_chain.has(content_type)
        if self.filter_has_tool_use:
            filter_chain = filter_chain.has_tool_use()
        if self.filter_has_thinking:
            filter_chain = filter_chain.has_thinking()
        if self.min_messages is not None:
            filter_chain = filter_chain.min_messages(self.min_messages)
        if self.max_messages is not None:
            filter_chain = filter_chain.max_messages(self.max_messages)
        if self.min_words is not None:
            filter_chain = filter_chain.min_words(self.min_words)
        if self.since:
            try:
                filter_chain = filter_chain.since(self.since)
            except ValueError as exc:
                raise QuerySpecError("since", self.since) from exc
        if self.until:
            try:
                filter_chain = filter_chain.until(self.until)
            except ValueError as exc:
                raise QuerySpecError("until", self.until) from exc
        if self.latest:
            filter_chain = filter_chain.sort("date").limit(1)
        if self.sort:
            filter_chain = filter_chain.sort(self.sort)
        if self.reverse:
            filter_chain = filter_chain.reverse()
        if self.limit is not None:
            filter_chain = filter_chain.limit(self.limit)
        if self.sample is not None:
            filter_chain = filter_chain.sample(self.sample)

        return filter_chain


__all__ = ["ConversationQuerySpec", "QuerySpecError"]
