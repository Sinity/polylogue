"""Typed conversation-query specification shared by CLI and MCP surfaces."""

from __future__ import annotations

import builtins
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, replace
from datetime import datetime
from typing import TYPE_CHECKING, TypeVar

from polylogue.archive.query.fields import describe_spec_fields, query_spec_has_selection_filters
from polylogue.archive.query.plan import ConversationQueryPlan
from polylogue.lib.dates import parse_date
from polylogue.lib.filter.types import SortField
from polylogue.lib.viewport.viewports import ToolCategory
from polylogue.types import Provider

if TYPE_CHECKING:
    from polylogue.lib.filter.filters import ConversationFilter
    from polylogue.lib.models import Conversation, ConversationSummary
    from polylogue.protocols import ConversationQueryRuntimeStore, VectorProvider

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
    if candidate == "date":
        return "date"
    if candidate == "tokens":
        return "tokens"
    if candidate == "messages":
        return "messages"
    if candidate == "words":
        return "words"
    if candidate == "longest":
        return "longest"
    if candidate == "random":
        return "random"
    raise QuerySpecError("sort", candidate)


# ---------------------------------------------------------------------------
# Description helpers
# ---------------------------------------------------------------------------


def describe_query_spec(spec: ConversationQuerySpec) -> list[str]:
    return describe_spec_fields(spec)


def query_spec_has_filters(spec: ConversationQuerySpec) -> bool:
    return query_spec_has_selection_filters(spec)


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
        referenced_path=as_tuple(params.get("referenced_path")),
        cwd_prefix=optional_text(params.get("cwd_prefix")),
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
        repo_names=split_csv(params.get("repo")),
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
        filter_has_paste=bool(params.get("filter_has_paste")),
        typed_only=bool(params.get("typed_only")),
        min_messages=optional_int(params.get("min_messages")),
        max_messages=optional_int(params.get("max_messages")),
        min_words=optional_int(params.get("min_words")),
        similar_text=optional_text(params.get("similar_text")),
        since_session_id=optional_text(params.get("since_session_id") or params.get("since_session")),
        message_type=optional_text(params.get("message_type")),
        offset=optional_int(params.get("offset")) or 0,
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
        referenced_path=spec.referenced_path,
        cwd_prefix=spec.cwd_prefix,
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
        repo_names=spec.repo_names,
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
        filter_has_paste=spec.filter_has_paste,
        typed_only=spec.typed_only,
        min_messages=spec.min_messages,
        max_messages=spec.max_messages,
        min_words=spec.min_words,
        similar_text=spec.similar_text,
        since_session_id=spec.since_session_id,
        message_type=spec.message_type,
        offset=spec.offset,
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
    referenced_path: tuple[str, ...] = ()
    cwd_prefix: str | None = None
    action_terms: tuple[str, ...] = ()
    excluded_action_terms: tuple[str, ...] = ()
    action_sequence: tuple[str, ...] = ()
    action_text_terms: tuple[str, ...] = ()
    tool_terms: tuple[str, ...] = ()
    excluded_tool_terms: tuple[str, ...] = ()
    providers: tuple[Provider, ...] = ()
    excluded_providers: tuple[Provider, ...] = ()
    repo_names: tuple[str, ...] = ()
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
    filter_has_paste: bool = False
    typed_only: bool = False
    min_messages: int | None = None
    max_messages: int | None = None
    min_words: int | None = None
    similar_text: str | None = None
    since_session_id: str | None = None
    message_type: str | None = None
    offset: int = 0

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
        repository: ConversationQueryRuntimeStore,
        *,
        vector_provider: VectorProvider | None = None,
    ) -> builtins.list[Conversation]:
        return await self.build_filter(repository, vector_provider=vector_provider).list()

    async def list_summaries(
        self,
        repository: ConversationQueryRuntimeStore,
        *,
        vector_provider: VectorProvider | None = None,
    ) -> builtins.list[ConversationSummary]:
        return await self.build_filter(repository, vector_provider=vector_provider).list_summaries()

    async def count(
        self,
        repository: ConversationQueryRuntimeStore,
        *,
        vector_provider: VectorProvider | None = None,
    ) -> int:
        return await self.build_filter(repository, vector_provider=vector_provider).count()

    async def delete(
        self,
        repository: ConversationQueryRuntimeStore,
        *,
        vector_provider: VectorProvider | None = None,
    ) -> int:
        return await self.build_filter(repository, vector_provider=vector_provider).delete()

    def build_filter(
        self,
        repository: ConversationQueryRuntimeStore,
        *,
        vector_provider: VectorProvider | None = None,
    ) -> ConversationFilter:
        """Build a fluent filter facade over the canonical execution plan."""
        from polylogue.lib.filter.filters import ConversationFilter

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
