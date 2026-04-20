"""Typed query request and execution contracts for the CLI surface."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Literal, TypeAlias

from polylogue.lib.query_spec import ConversationQuerySpec

if TYPE_CHECKING:
    from polylogue.lib.models import Conversation, ConversationSummary

QueryParams: TypeAlias = dict[str, object]
QueryParamSource: TypeAlias = Mapping[str, object] | ConversationQuerySpec
QueryResult: TypeAlias = "Conversation | ConversationSummary"

QueryOutputFormat: TypeAlias = str
QueryTransform: TypeAlias = str | None
QueryDeliveryName: TypeAlias = Literal["stdout", "browser", "clipboard"]


def coerce_query_terms(value: object) -> tuple[str, ...]:
    """Coerce a raw Click/query value into canonical query terms."""
    if value is None:
        return ()
    if isinstance(value, str | bytes):
        return (str(value),)
    if isinstance(value, Iterable):
        return tuple(str(term) for term in value)
    return (str(value),)


def coerce_query_spec(params: QueryParamSource) -> ConversationQuerySpec:
    """Build a query spec from raw params or pass through an existing one."""
    if isinstance(params, ConversationQuerySpec):
        return params
    return ConversationQuerySpec.from_params(params)


def describe_query_filters(params: QueryParamSource) -> list[str]:
    """Describe active query filters for CLI feedback."""
    return coerce_query_spec(params).describe()


@dataclass(frozen=True, slots=True)
class QueryDeliveryTarget:
    """Single parsed delivery target for query output."""

    raw: str
    kind: QueryDeliveryName | Literal["path"]
    path: Path | None = None

    @classmethod
    def parse(cls, value: str) -> QueryDeliveryTarget:
        normalized = value.strip() or "stdout"
        if normalized == "stdout":
            return cls(raw=normalized, kind="stdout")
        if normalized == "browser":
            return cls(raw=normalized, kind="browser")
        if normalized == "clipboard":
            return cls(raw=normalized, kind="clipboard")
        return cls(raw=normalized, kind="path", path=Path(normalized))


@dataclass(frozen=True, slots=True)
class QueryOutputSpec:
    """Surface output settings derived from raw CLI params."""

    output_format: QueryOutputFormat
    destinations: tuple[QueryDeliveryTarget, ...]
    fields: str | None
    dialogue_only: bool
    transform: QueryTransform
    list_mode: bool
    print_path: bool

    @classmethod
    def from_params(cls, params: Mapping[str, object]) -> QueryOutputSpec:
        output_dest = str(params.get("output") or "stdout")
        destinations = tuple(QueryDeliveryTarget.parse(part) for part in output_dest.split(",") if part.strip()) or (
            QueryDeliveryTarget.parse("stdout"),
        )
        return cls(
            output_format=str(params.get("output_format") or "markdown"),
            destinations=destinations,
            fields=str(params["fields"]) if params.get("fields") is not None else None,
            dialogue_only=bool(params.get("dialogue_only", False)),
            transform=str(params["transform"]) if params.get("transform") is not None else None,
            list_mode=bool(params.get("list_mode", False)),
            print_path=bool(params.get("print_path", False)),
        )

    def stream_format(self) -> str:
        if self.output_format == "json":
            return "json-lines"
        if self.output_format in {"plaintext", "markdown", "json-lines"}:
            return self.output_format
        return "plaintext"

    def destination_labels(self) -> tuple[str, ...]:
        return tuple(target.raw for target in self.destinations)


@dataclass(frozen=True, slots=True)
class QueryMutationSpec:
    """Mutation-side execution settings derived from raw CLI params."""

    set_meta: tuple[tuple[str, str], ...]
    add_tags: tuple[str, ...]
    delete_matched: bool
    dry_run: bool
    force: bool

    @classmethod
    def from_params(cls, params: Mapping[str, object]) -> QueryMutationSpec:
        return cls(
            set_meta=_iter_set_meta_pairs(params.get("set_meta")),
            add_tags=tuple(str(tag) for tag in _iter_param_values(params.get("add_tag")) or ()),
            delete_matched=bool(params.get("delete_matched", False)),
            dry_run=bool(params.get("dry_run", False)),
            force=bool(params.get("force", False)),
        )

    @property
    def has_operations(self) -> bool:
        return bool(self.set_meta or self.add_tags)


class QueryPlanError(ValueError):
    """Raised when CLI query parameters describe an invalid plan."""


class QueryAction(str, Enum):
    COUNT = "count"
    STREAM = "stream"
    STATS = "stats"
    STATS_BY = "stats-by"
    MODIFY = "modify"
    DELETE = "delete"
    OPEN = "open"
    SHOW = "show"


class QueryRoute(str, Enum):
    COUNT = "count"
    SUMMARY_LIST = "summary-list"
    STREAM = "stream"
    STATS_SQL = "stats-sql"
    SUMMARY_STATS = "summary-stats"
    STATS_BY = "stats-by"
    SUMMARY_MODIFY = "summary-modify"
    SUMMARY_DELETE = "summary-delete"
    MODIFY = "modify"
    DELETE = "delete"
    OPEN = "open"
    SHOW = "show"


@dataclass(frozen=True, slots=True)
class QueryExecutionPlan:
    """Fully typed CLI execution plan for a query request."""

    selection: ConversationQuerySpec
    action: QueryAction
    output: QueryOutputSpec
    mutation: QueryMutationSpec
    stats_dimension: str | None = None

    @classmethod
    def from_params(cls, params: Mapping[str, object]) -> QueryExecutionPlan:
        selection = ConversationQuerySpec.from_params(dict(params))
        output = QueryOutputSpec.from_params(params)
        mutation = QueryMutationSpec.from_params(params)
        stats_dimension = str(params["stats_by"]) if params.get("stats_by") else None

        if params.get("count_only"):
            action = QueryAction.COUNT
        elif params.get("stream"):
            action = QueryAction.STREAM
        elif params.get("stats_only"):
            action = QueryAction.STATS
        elif stats_dimension is not None:
            action = QueryAction.STATS_BY
        elif mutation.delete_matched:
            action = QueryAction.DELETE
        elif mutation.has_operations:
            action = QueryAction.MODIFY
        elif params.get("open_result"):
            action = QueryAction.OPEN
        else:
            action = QueryAction.SHOW

        if action == QueryAction.DELETE and not selection.has_filters():
            raise QueryPlanError(
                "delete requires at least one filter to prevent accidental deletion of the entire archive."
            )

        return cls(
            selection=selection,
            action=action,
            output=output,
            mutation=mutation,
            stats_dimension=stats_dimension,
        )

    def prefers_summary_list(self) -> bool:
        return (
            self.action == QueryAction.SHOW
            and self.output.list_mode
            and self.output.transform is None
            and not self.output.dialogue_only
        )

    def prefers_summary_stats(self) -> bool:
        return self.action == QueryAction.STATS_BY and self.stats_dimension in {"provider", "month", "year", "day"}

    def prefers_summary_mutation(self) -> bool:
        return self.action in {QueryAction.MODIFY, QueryAction.DELETE}


def build_query_execution_plan(params: Mapping[str, object]) -> QueryExecutionPlan:
    """Build a typed execution plan from raw CLI params."""
    return QueryExecutionPlan.from_params(params)


def resolve_query_route(
    plan: QueryExecutionPlan,
    *,
    can_use_summaries: bool,
) -> QueryRoute:
    """Resolve the concrete runtime route for a query execution plan."""
    if plan.action == QueryAction.COUNT:
        return QueryRoute.COUNT
    if plan.prefers_summary_list() and can_use_summaries:
        return QueryRoute.SUMMARY_LIST
    if plan.action == QueryAction.STREAM:
        return QueryRoute.STREAM
    if plan.action == QueryAction.STATS:
        return QueryRoute.STATS_SQL
    if plan.prefers_summary_stats() and can_use_summaries:
        return QueryRoute.SUMMARY_STATS
    if plan.action == QueryAction.STATS_BY:
        return QueryRoute.STATS_BY
    if plan.action == QueryAction.MODIFY:
        return QueryRoute.SUMMARY_MODIFY if can_use_summaries else QueryRoute.MODIFY
    if plan.action == QueryAction.DELETE:
        return QueryRoute.SUMMARY_DELETE if can_use_summaries else QueryRoute.DELETE
    if plan.action == QueryAction.OPEN:
        return QueryRoute.OPEN
    return QueryRoute.SHOW


def result_id(result: QueryResult) -> str:
    return str(result.id)


def result_provider(result: QueryResult) -> str:
    return str(result.provider)


def result_title(result: QueryResult) -> str:
    title = result.display_title
    return title if title else result_id(result)[:20]


def result_date(result: QueryResult) -> datetime | None:
    display_date = getattr(result, "display_date", None)
    if isinstance(display_date, datetime):
        return display_date
    updated_at = getattr(result, "updated_at", None)
    if isinstance(updated_at, datetime):
        return updated_at
    created_at = getattr(result, "created_at", None)
    if isinstance(created_at, datetime):
        return created_at
    return None


def _iter_param_values(value: object) -> Iterable[object] | None:
    if isinstance(value, str | bytes):
        return None
    if isinstance(value, Iterable):
        return value
    return None


def _iter_set_meta_pairs(value: object) -> tuple[tuple[str, str], ...]:
    if isinstance(value, str | bytes):
        return ()
    if not isinstance(value, Iterable):
        return ()
    pairs: list[tuple[str, str]] = []
    for item in value:
        if isinstance(item, str | bytes):
            continue
        if not isinstance(item, Sequence):
            continue
        if len(item) < 2:
            continue
        pairs.append((str(item[0]), str(item[1])))
    return tuple(pairs)


__all__ = [
    "build_query_execution_plan",
    "coerce_query_spec",
    "coerce_query_terms",
    "describe_query_filters",
    "QueryAction",
    "QueryDeliveryTarget",
    "QueryExecutionPlan",
    "QueryMutationSpec",
    "QueryOutputSpec",
    "QueryParamSource",
    "QueryParams",
    "QueryPlanError",
    "QueryResult",
    "QueryRoute",
    "resolve_query_route",
    "result_date",
    "result_id",
    "result_provider",
    "result_title",
]
