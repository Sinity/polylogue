"""Typed conversation-query specification shared by CLI and MCP surfaces."""

from __future__ import annotations

import builtins
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, replace
from datetime import datetime
from typing import TYPE_CHECKING, TypeVar, cast

from polylogue.lib.dates import parse_date
from polylogue.lib.filter_types import SortField
from polylogue.lib.query_plan import ConversationQueryPlan
from polylogue.lib.viewports import ToolCategory
from polylogue.types import Provider

if TYPE_CHECKING:
    from polylogue.lib.filters import ConversationFilter
    from polylogue.lib.models import Conversation, ConversationSummary
    from polylogue.protocols import VectorProvider
    from polylogue.storage.repository import ConversationRepository

_SpecT = TypeVar("_SpecT", bound="ConversationQuerySpec")


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class QuerySpecError(ValueError):
    """Typed query-spec construction/application error."""

    def __init__(self, field: str, value: str) -> None:
        super().__init__(f"invalid {field}: {value}")
        self.field = field
        self.value = value


# ---------------------------------------------------------------------------
# Normalization constants and helpers
# ---------------------------------------------------------------------------

QUERY_ACTION_TYPES = tuple(category.value for category in ToolCategory) + ("none",)
QUERY_SEQUENCE_ACTION_TYPES = tuple(category.value for category in ToolCategory)
QUERY_RETRIEVAL_LANES = ("auto", "dialogue", "actions", "hybrid")


def _iter_values(value: object) -> Iterable[object]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    if isinstance(value, Iterable):
        return value
    return (value,)


def split_csv(value: object) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return tuple(part.strip() for part in value.split(",") if part.strip())
    return tuple(str(item).strip() for item in _iter_values(value) if str(item).strip())


def as_tuple(value: object) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    return tuple(str(item) for item in _iter_values(value))


def parse_query_date(field: str, value: str | None) -> datetime | None:
    if value is None:
        return None
    parsed = parse_date(value)
    if parsed is None:
        raise QuerySpecError(field, value)
    return parsed


def normalize_tool_terms(value: object) -> tuple[str, ...]:
    normalized: list[str] = []
    for term in as_tuple(value):
        candidate = str(term).strip().lower()
        if candidate:
            normalized.append(candidate)
    return tuple(normalized)


def normalize_action_terms(field: str, value: object) -> tuple[str, ...]:
    normalized: list[str] = []
    for term in as_tuple(value):
        candidate = str(term).strip().lower()
        if candidate not in QUERY_ACTION_TYPES:
            raise QuerySpecError(field, term)
        normalized.append(candidate)
    return tuple(normalized)


def normalize_action_sequence(field: str, value: object) -> tuple[str, ...]:
    normalized: list[str] = []
    for term in split_csv(value):
        candidate = str(term).strip().lower()
        if candidate not in QUERY_SEQUENCE_ACTION_TYPES:
            raise QuerySpecError(field, term)
        normalized.append(candidate)
    return tuple(normalized)


def optional_text(value: object) -> str | None:
    if not value:
        return None
    return str(value)


def optional_int(value: object) -> int | None:
    if not value:
        return None
    return int(str(value))


def optional_sort_field(value: object) -> SortField | None:
    candidate = optional_text(value)
    if candidate is None:
        return None
    return cast(SortField, candidate)


# ---------------------------------------------------------------------------
# Description helpers
# ---------------------------------------------------------------------------


def describe_query_spec(spec: ConversationQuerySpec) -> list[str]:
    parts: list[str] = []
    if spec.query_terms:
        parts.append(f"search: {' '.join(spec.query_terms)}")
    if spec.contains_terms:
        parts.append(f"contains: {', '.join(spec.contains_terms)}")
    if spec.exclude_text_terms:
        parts.append(f"exclude text: {', '.join(spec.exclude_text_terms)}")
    if spec.retrieval_lane != "auto":
        parts.append(f"retrieval: {spec.retrieval_lane}")
    if spec.path_terms:
        parts.append(f"path: {', '.join(spec.path_terms)}")
    if spec.action_terms:
        parts.append(f"action: {', '.join(spec.action_terms)}")
    if spec.excluded_action_terms:
        parts.append(f"exclude action: {', '.join(spec.excluded_action_terms)}")
    if spec.action_sequence:
        parts.append(f"action sequence: {' -> '.join(spec.action_sequence)}")
    if spec.action_text_terms:
        parts.append(f"action text: {', '.join(spec.action_text_terms)}")
    if spec.tool_terms:
        parts.append(f"tool: {', '.join(spec.tool_terms)}")
    if spec.excluded_tool_terms:
        parts.append(f"exclude tool: {', '.join(spec.excluded_tool_terms)}")
    if spec.providers:
        parts.append(f"provider: {', '.join(p.value for p in spec.providers)}")
    if spec.excluded_providers:
        parts.append(f"exclude provider: {', '.join(p.value for p in spec.excluded_providers)}")
    if spec.tags:
        parts.append(f"tag: {', '.join(spec.tags)}")
    if spec.excluded_tags:
        parts.append(f"exclude tag: {', '.join(spec.excluded_tags)}")
    if spec.title:
        parts.append(f"title: {spec.title}")
    if spec.has_types:
        parts.append(f"has: {', '.join(spec.has_types)}")
    if spec.filter_has_tool_use:
        parts.append("has: tool_use (sql)")
    if spec.filter_has_thinking:
        parts.append("has: thinking (sql)")
    if spec.min_messages is not None:
        parts.append(f"min_messages: {spec.min_messages}")
    if spec.max_messages is not None:
        parts.append(f"max_messages: {spec.max_messages}")
    if spec.min_words is not None:
        parts.append(f"min_words: {spec.min_words}")
    if spec.similar_text:
        parts.append(f"similar: {spec.similar_text}")
    if spec.since:
        parts.append(f"since: {spec.since}")
    if spec.until:
        parts.append(f"until: {spec.until}")
    if spec.conversation_id:
        parts.append(f"id: {spec.conversation_id}")
    return parts


def query_spec_has_filters(spec: ConversationQuerySpec) -> bool:
    return any(
        (
            spec.query_terms,
            spec.contains_terms,
            spec.exclude_text_terms,
            spec.path_terms,
            spec.action_terms,
            spec.excluded_action_terms,
            spec.action_sequence,
            spec.action_text_terms,
            spec.tool_terms,
            spec.excluded_tool_terms,
            spec.providers,
            spec.excluded_providers,
            spec.tags,
            spec.excluded_tags,
            spec.has_types,
            spec.title is not None,
            spec.conversation_id is not None,
            spec.since is not None,
            spec.until is not None,
            spec.latest,
            spec.filter_has_tool_use,
            spec.filter_has_thinking,
            spec.min_messages is not None,
            spec.max_messages is not None,
            spec.min_words is not None,
            spec.similar_text is not None,
        )
    )


# ---------------------------------------------------------------------------
# Builders (from_params, to_plan)
# ---------------------------------------------------------------------------


def build_query_spec_from_params(
    spec_cls: type[_SpecT],
    params: Mapping[str, object],
) -> _SpecT:
    return spec_cls(
        query_terms=as_tuple(params.get("query")),
        contains_terms=as_tuple(params.get("contains")),
        exclude_text_terms=as_tuple(params.get("exclude_text")),
        retrieval_lane=str(params.get("retrieval_lane") or "auto"),
        path_terms=as_tuple(params.get("path_terms") or params.get("path")),
        action_terms=normalize_action_terms("action", params.get("action")),
        excluded_action_terms=normalize_action_terms("exclude_action", params.get("exclude_action")),
        action_sequence=normalize_action_sequence("action_sequence", params.get("action_sequence")),
        action_text_terms=as_tuple(params.get("action_text")),
        tool_terms=normalize_tool_terms(params.get("tool")),
        excluded_tool_terms=normalize_tool_terms(params.get("exclude_tool")),
        providers=tuple(Provider.from_string(p) for p in split_csv(params.get("provider"))),
        excluded_providers=tuple(Provider.from_string(p) for p in split_csv(params.get("exclude_provider"))),
        tags=split_csv(params.get("tag")),
        excluded_tags=split_csv(params.get("exclude_tag")),
        has_types=as_tuple(params.get("has_type")),
        title=optional_text(params.get("title")),
        conversation_id=optional_text(params.get("conv_id")),
        since=optional_text(params.get("since")),
        until=optional_text(params.get("until")),
        latest=bool(params.get("latest")),
        sort=optional_sort_field(params.get("sort")),
        reverse=bool(params.get("reverse")),
        limit=optional_int(params.get("limit")),
        sample=optional_int(params.get("sample")),
        filter_has_tool_use=bool(params.get("filter_has_tool_use")),
        filter_has_thinking=bool(params.get("filter_has_thinking")),
        min_messages=optional_int(params.get("min_messages")),
        max_messages=optional_int(params.get("max_messages")),
        min_words=optional_int(params.get("min_words")),
        similar_text=optional_text(params.get("similar_text")),
    )


def query_spec_to_plan(
    spec: ConversationQuerySpec,
    *,
    vector_provider: VectorProvider | None = None,
) -> ConversationQueryPlan:
    plan = ConversationQueryPlan(
        query_terms=spec.query_terms,
        contains_terms=spec.contains_terms,
        negative_terms=spec.exclude_text_terms,
        retrieval_lane=spec.retrieval_lane,
        path_terms=spec.path_terms,
        action_terms=spec.action_terms,
        excluded_action_terms=spec.excluded_action_terms,
        action_sequence=spec.action_sequence,
        action_text_terms=spec.action_text_terms,
        tool_terms=spec.tool_terms,
        excluded_tool_terms=spec.excluded_tool_terms,
        providers=spec.providers,
        excluded_providers=spec.excluded_providers,
        tags=spec.tags,
        excluded_tags=spec.excluded_tags,
        has_types=spec.has_types,
        title=spec.title,
        conversation_id=spec.conversation_id,
        since=parse_query_date("since", spec.since),
        until=parse_query_date("until", spec.until),
        sort=spec.sort or "date",
        reverse=spec.reverse,
        limit=spec.limit,
        sample=spec.sample,
        filter_has_tool_use=spec.filter_has_tool_use,
        filter_has_thinking=spec.filter_has_thinking,
        min_messages=spec.min_messages,
        max_messages=spec.max_messages,
        min_words=spec.min_words,
        similar_text=spec.similar_text,
        vector_provider=vector_provider,
    )
    if spec.latest:
        plan = replace(plan, sort="date", limit=1)
    return plan


# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------


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
    sort: SortField | None = None
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
    ) -> builtins.list[Conversation]:
        return await self.build_filter(repository, vector_provider=vector_provider).list()

    async def list_summaries(
        self,
        repository: ConversationRepository,
        *,
        vector_provider: VectorProvider | None = None,
    ) -> builtins.list[ConversationSummary]:
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
        """Build a fluent filter facade over the canonical execution plan."""
        from polylogue.lib.filters import ConversationFilter

        return ConversationFilter(
            repository,
            vector_provider=vector_provider,
            query_plan=self.to_plan(vector_provider=vector_provider),
        )


__all__ = [
    "ConversationQuerySpec",
    "QUERY_ACTION_TYPES",
    "QUERY_RETRIEVAL_LANES",
    "QuerySpecError",
]
