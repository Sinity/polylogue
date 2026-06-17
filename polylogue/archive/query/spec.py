"""Typed session-query specification shared by CLI and MCP surfaces."""

from __future__ import annotations

import builtins
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, replace
from datetime import datetime
from typing import TYPE_CHECKING, TypeVar

from polylogue.archive.filter.types import SortField
from polylogue.archive.message.types import validate_message_type_filter
from polylogue.archive.query.fields import describe_spec_fields, query_spec_has_selection_filters
from polylogue.archive.query.plan import SessionQueryPlan
from polylogue.archive.query.predicate import QueryPredicate
from polylogue.archive.viewport.viewports import ToolCategory
from polylogue.core.dates import parse_date
from polylogue.core.enums import Origin
from polylogue.errors import PolylogueError

if TYPE_CHECKING:
    from polylogue.archive.filter.filters import SessionFilter
    from polylogue.archive.models import Session, SessionSummary
    from polylogue.config import Config
    from polylogue.protocols import VectorProvider

_SpecT = TypeVar("_SpecT", bound="SessionQuerySpec")


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class QuerySpecError(PolylogueError):
    """Typed query-spec construction/application error."""

    http_status_code = 400

    def __init__(self, field: str, value: str) -> None:
        super().__init__(f"invalid {field}: {value}")
        self.field = field
        self.value = value


# ---------------------------------------------------------------------------
# Normalization constants and helpers
# ---------------------------------------------------------------------------

QUERY_ACTION_TYPES = tuple(category.value for category in ToolCategory) + ("none",)
QUERY_SEQUENCE_ACTION_TYPES = tuple(category.value for category in ToolCategory)
#: Closed vocabulary of retrieval lanes accepted at the query boundary. This is
#: the single source of truth shared by the CLI ``--retrieval-lane`` choice, the
#: MCP ``retrieval_lane`` parameter, daemon HTTP, and the ``SearchEnvelope``
#: contract. ``similar_text`` (vector-only similarity) is a separate request
#: knob, not a lane — there is deliberately no ``"semantic"`` member.
QUERY_RETRIEVAL_LANES = ("auto", "dialogue", "actions", "hybrid")

#: Shared maximum page size enforced by every read surface (CLI query-first,
#: MCP, daemon HTTP). Surfaces clamp their requested ``limit`` to this ceiling
#: so no single caller can request an unbounded fetch.
MAX_QUERY_LIMIT = 1000


def clamp_query_limit(limit: object, *, default: int = 10) -> int:
    """Clamp a requested ``limit`` into ``[1, MAX_QUERY_LIMIT]``.

    Shared by every read surface so the page-size ceiling is identical across
    CLI, MCP, and daemon HTTP. Non-integer or non-positive input falls back to
    ``default`` (then itself clamped), matching the historical MCP behavior.
    """
    try:
        if isinstance(limit, bool):
            raise TypeError
        if isinstance(limit, int):
            value = limit
        elif isinstance(limit, float | str | bytes | bytearray):
            value = int(limit)
        else:
            value = int(str(limit))
    except (TypeError, ValueError):
        value = default
    return max(1, min(value, MAX_QUERY_LIMIT))


def normalize_retrieval_lane(value: object) -> str:
    """Validate a retrieval-lane token against the closed vocabulary."""
    if value is None:
        return "auto"
    candidate = str(value).strip().lower() or "auto"
    if candidate not in QUERY_RETRIEVAL_LANES:
        raise QuerySpecError("retrieval_lane", str(value))
    return candidate


def optional_non_negative_int(field: str, value: object) -> int | None:
    """Parse an optional integer, rejecting negative values with a typed error."""
    parsed = optional_int(value)
    if parsed is not None and parsed < 0:
        raise QuerySpecError(field, str(value))
    return parsed


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


def optional_message_type(value: object) -> str | None:
    if not value:
        return None
    try:
        return validate_message_type_filter(value).value
    except ValueError as exc:
        raise QuerySpecError("message_type", str(value)) from exc


def optional_int(value: object) -> int | None:
    # Preserve an explicit 0: ``if not value`` would drop it, broadening e.g.
    # ``max_words=0`` (sessions with no words) into an unbounded filter.
    if value is None:
        return None
    if isinstance(value, str) and value.strip() == "":
        return None
    return int(str(value))


# Set of all recognized query-spec parameter names (drives strict-param mode).
_RECOGNIZED_PARAMS: frozenset[str] = frozenset(
    {
        "query",
        "contains",
        "exclude_text",
        "retrieval_lane",
        "referenced_path",
        "cwd_prefix",
        "action",
        "exclude_action",
        "action_sequence",
        "action_text",
        "tool",
        "exclude_tool",
        "origin",
        "exclude_origin",
        "tag",
        "exclude_tag",
        "repo",
        "has_type",
        "title",
        "conv_id",
        "since",
        "until",
        "latest",
        "sort",
        "reverse",
        "limit",
        "sample",
        "filter_has_tool_use",
        "filter_has_thinking",
        "filter_has_paste",
        "typed_only",
        "min_messages",
        "max_messages",
        "min_words",
        "max_words",
        "similar_text",
        "similar_session_id",
        "since_session_id",
        "message_type",
        "offset",
        "cursor",
    }
)


_PATH_COMPONENT_RE = __import__("re").compile(r"^[^./][^/]*(?:/[^./][^/]*)*$")


def _validate_path_component(field: str, value: str | None) -> None:
    """Reject ../ escapes and empty components. Absolute paths are allowed."""
    if value is None:
        return
    if ".." in value.split("/"):
        raise QuerySpecError(field, value)
    if value == "." or value == "":
        raise QuerySpecError(field, value)


def validate_params_known(params: Mapping[str, object], *, strict: bool = False) -> None:
    """Raise on unrecognized param names when strict mode is active."""
    if not strict:
        return
    for key in params:
        if key not in _RECOGNIZED_PARAMS:
            raise QuerySpecError(key, f"unknown query parameter: {key!r}")


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


def describe_query_spec(spec: SessionQuerySpec) -> list[str]:
    return describe_spec_fields(spec)


def query_spec_has_filters(spec: SessionQuerySpec) -> bool:
    return query_spec_has_selection_filters(spec)


# ---------------------------------------------------------------------------
# Builders (from_params, to_plan)
# ---------------------------------------------------------------------------


def build_query_spec_from_params(
    spec_cls: type[_SpecT],
    params: Mapping[str, object],
    *,
    strict_params: bool = False,
) -> _SpecT:
    if strict_params:
        validate_params_known(params, strict=True)
    cwd_prefix = optional_text(params.get("cwd_prefix"))
    _validate_path_component("cwd_prefix", cwd_prefix)
    since_session_id = optional_text(params.get("since_session_id"))
    _validate_path_component("since_session_id", since_session_id)
    return spec_cls(
        query_terms=as_tuple(params.get("query")),
        contains_terms=as_tuple(params.get("contains")),
        exclude_text_terms=as_tuple(params.get("exclude_text")),
        retrieval_lane=normalize_retrieval_lane(params.get("retrieval_lane")),
        referenced_path=as_tuple(params.get("referenced_path")),
        cwd_prefix=cwd_prefix,
        action_terms=normalize_action_terms("action", params.get("action")),
        excluded_action_terms=normalize_action_terms("exclude_action", params.get("exclude_action")),
        action_sequence=normalize_action_sequence("action_sequence", params.get("action_sequence")),
        action_text_terms=as_tuple(params.get("action_text")),
        tool_terms=normalize_tool_terms(params.get("tool")),
        excluded_tool_terms=normalize_tool_terms(params.get("exclude_tool")),
        origins=tuple(Origin.from_string(p).value for p in split_csv(params.get("origin"))),
        excluded_origins=tuple(Origin.from_string(p).value for p in split_csv(params.get("exclude_origin"))),
        tags=split_csv(params.get("tag")),
        excluded_tags=split_csv(params.get("exclude_tag")),
        repo_names=split_csv(params.get("repo")),
        has_types=as_tuple(params.get("has_type")),
        title=optional_text(params.get("title")),
        session_id=optional_text(params.get("conv_id")),
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
        min_messages=optional_non_negative_int("min_messages", params.get("min_messages")),
        max_messages=optional_non_negative_int("max_messages", params.get("max_messages")),
        min_words=optional_non_negative_int("min_words", params.get("min_words")),
        max_words=optional_non_negative_int("max_words", params.get("max_words")),
        similar_text=optional_text(params.get("similar_text")),
        similar_session_id=optional_text(params.get("similar_session_id")),
        since_session_id=since_session_id,
        message_type=optional_message_type(params.get("message_type")),
        offset=optional_int(params.get("offset")) or 0,
        cursor=optional_text(params.get("cursor")),
    )


def query_spec_to_plan(
    spec: SessionQuerySpec,
    *,
    vector_provider: VectorProvider | None = None,
) -> SessionQueryPlan:
    plan = SessionQueryPlan(
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
        origins=spec.origins,
        excluded_origins=spec.excluded_origins,
        tags=spec.tags,
        excluded_tags=spec.excluded_tags,
        repo_names=spec.repo_names,
        has_types=spec.has_types,
        title=spec.title,
        session_id=spec.session_id,
        since=parse_query_date("since", spec.since),
        until=parse_query_date("until", spec.until),
        sort=spec.sort,
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
        max_words=spec.max_words,
        similar_text=spec.similar_text,
        similar_session_id=spec.similar_session_id,
        since_session_id=spec.since_session_id,
        message_type=spec.message_type,
        offset=spec.offset,
        cursor=spec.cursor,
        boolean_predicate=spec.boolean_predicate,
        vector_provider=vector_provider,
    )
    if spec.latest:
        plan = replace(plan, sort="date", limit=1)
    return plan


# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SessionQuerySpec:
    """Canonical selection intent for session queries."""

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
    origins: tuple[str, ...] = ()
    excluded_origins: tuple[str, ...] = ()
    repo_names: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()
    excluded_tags: tuple[str, ...] = ()
    has_types: tuple[str, ...] = ()
    title: str | None = None
    session_id: str | None = None
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
    max_words: int | None = None
    similar_text: str | None = None
    similar_session_id: str | None = None
    since_session_id: str | None = None
    message_type: str | None = None
    offset: int = 0
    cursor: str | None = None
    boolean_predicate: QueryPredicate | None = None

    @classmethod
    def from_params(cls, params: Mapping[str, object], *, strict: bool = False) -> SessionQuerySpec:
        """Build a query spec from CLI-style parameter mapping.

        When *strict* is True, unknown parameter names and path-component
        violations (cwd_prefix, since_session_id) are rejected with
        ``QuerySpecError`` instead of being silently absorbed.
        """
        return build_query_spec_from_params(cls, params, strict_params=strict)

    @classmethod
    def from_expression(cls, expression: str) -> SessionQuerySpec:
        """Compile a DSL query expression string into a ``SessionQuerySpec``.

        This is the Python-facade entry point for the shared expression
        compiler.  Field clauses such as ``repo:polylogue``, ``since:7d``,
        or ``origin:(claude-code-session|codex-session)`` are mapped to the
        corresponding spec fields.  Bare words and quoted phrases go to
        ``query_terms`` (FTS).

        Raises:
            ExpressionCompileError: When an unknown field or structural error
                is encountered.

        Example::

            spec = SessionQuerySpec.from_expression(
                'repo:polylogue since:7d "json envelope"'
            )
        """
        from polylogue.archive.query.expression import compile_expression

        return compile_expression(expression)

    def describe(self) -> list[str]:
        """Human-readable filter descriptions for UX/error output."""
        return describe_query_spec(self)

    def has_filters(self) -> bool:
        """Whether the spec narrows session selection."""
        return query_spec_has_filters(self)

    def to_plan(
        self,
        *,
        vector_provider: VectorProvider | None = None,
    ) -> SessionQueryPlan:
        """Compile the immutable spec to the canonical execution plan."""
        return query_spec_to_plan(self, vector_provider=vector_provider)

    async def list(
        self,
        config: Config,
        *,
        vector_provider: VectorProvider | None = None,
    ) -> builtins.list[Session]:
        return await self.build_filter(config, vector_provider=vector_provider).list()

    async def list_summaries(
        self,
        config: Config,
        *,
        vector_provider: VectorProvider | None = None,
    ) -> builtins.list[SessionSummary]:
        return await self.build_filter(config, vector_provider=vector_provider).list_summaries()

    async def count(
        self,
        config: Config,
        *,
        vector_provider: VectorProvider | None = None,
    ) -> int:
        return await self.build_filter(config, vector_provider=vector_provider).count()

    async def delete(
        self,
        config: Config,
        *,
        vector_provider: VectorProvider | None = None,
    ) -> int:
        return await self.build_filter(config, vector_provider=vector_provider).delete()

    def build_filter(
        self,
        config: Config,
        *,
        vector_provider: VectorProvider | None = None,
    ) -> SessionFilter:
        """Build a fluent filter facade over the canonical execution plan."""
        from polylogue.archive.filter.filters import SessionFilter
        from polylogue.paths import archive_file_set_root_for_paths

        archive_root = archive_file_set_root_for_paths(
            archive_root_path=config.archive_root,
            db_anchor=config.db_path,
        )
        return SessionFilter(
            archive_root=archive_root,
            config=config,
            vector_provider=vector_provider,
            query_plan=self.to_plan(vector_provider=vector_provider),
        )


__all__ = [
    "SessionQuerySpec",
    "MAX_QUERY_LIMIT",
    "QUERY_ACTION_TYPES",
    "QUERY_RETRIEVAL_LANES",
    "QuerySpecError",
    "clamp_query_limit",
    "normalize_retrieval_lane",
    "build_query_spec_from_params",
]
