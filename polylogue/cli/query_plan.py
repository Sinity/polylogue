"""Typed execution planning for the query-first CLI."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum

from polylogue.lib.query_spec import ConversationQuerySpec


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


@dataclass(frozen=True)
class QueryOutputSpec:
    output_format: str
    destinations: tuple[str, ...]
    fields: str | None
    dialogue_only: bool
    transform: str | None
    list_mode: bool

    def stream_format(self) -> str:
        if self.output_format == "json":
            return "json-lines"
        if self.output_format in {"plaintext", "markdown", "json-lines"}:
            return self.output_format
        return "plaintext"


@dataclass(frozen=True)
class QueryMutationSpec:
    set_meta: tuple[tuple[str, str], ...]
    add_tags: tuple[str, ...]
    delete_matched: bool
    dry_run: bool
    force: bool

    @property
    def has_operations(self) -> bool:
        return bool(self.set_meta or self.add_tags)


@dataclass(frozen=True)
class QueryExecutionPlan:
    selection: ConversationQuerySpec
    action: QueryAction
    output: QueryOutputSpec
    mutation: QueryMutationSpec
    stats_dimension: str | None = None

    def prefers_summary_list(self) -> bool:
        return (
            self.action == QueryAction.SHOW
            and self.output.list_mode
            and self.output.transform is None
            and not self.output.dialogue_only
        )

    def prefers_summary_stats(self) -> bool:
        return self.action == QueryAction.STATS_BY

    def prefers_summary_mutation(self) -> bool:
        return self.action in {QueryAction.MODIFY, QueryAction.DELETE}


def build_query_execution_plan(params: Mapping[str, object]) -> QueryExecutionPlan:
    selection = ConversationQuerySpec.from_params(dict(params))

    output_dest = str(params.get("output") or "stdout")
    destinations = tuple(part.strip() for part in output_dest.split(",") if part.strip()) or ("stdout",)
    output = QueryOutputSpec(
        output_format=str(params.get("output_format") or "markdown"),
        destinations=destinations,
        fields=str(params["fields"]) if params.get("fields") is not None else None,
        dialogue_only=bool(params.get("dialogue_only", False)),
        transform=str(params["transform"]) if params.get("transform") is not None else None,
        list_mode=bool(params.get("list_mode", False)),
    )

    raw_set_meta = params.get("set_meta") or ()
    set_meta = tuple((str(key), str(value)) for key, value in raw_set_meta)
    add_tags = tuple(str(tag) for tag in (params.get("add_tag") or ()))
    mutation = QueryMutationSpec(
        set_meta=set_meta,
        add_tags=add_tags,
        delete_matched=bool(params.get("delete_matched", False)),
        dry_run=bool(params.get("dry_run", False)),
        force=bool(params.get("force", False)),
    )

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
            "--delete requires at least one filter to prevent accidental deletion of the entire archive."
        )

    return QueryExecutionPlan(
        selection=selection,
        action=action,
        output=output,
        mutation=mutation,
        stats_dimension=stats_dimension,
    )
