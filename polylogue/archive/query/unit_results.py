"""Terminal unit-query execution over the archive."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, Protocol, cast

from polylogue.archive.query.expression import (
    ExpressionCompileError,
    QueryUnitPipeline,
    QueryUnitSource,
)
from polylogue.archive.query.metadata import (
    QueryUnitDescriptor,
    query_unit_descriptor,
)
from polylogue.archive.query.spec import (
    normalize_action_sequence,
    normalize_action_terms,
    normalize_tool_terms,
    optional_int,
    optional_message_type,
    optional_text,
    parse_query_date,
    split_csv,
)
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from polylogue.surfaces import payloads as surface_payloads
from polylogue.surfaces.payloads import (
    QueryUnitAggregateRowPayload,
    QueryUnitResultEnvelope,
    build_query_unit_aggregate_envelope,
    build_query_unit_envelope,
)


def _pipeline_stage_payloads(pipeline: QueryUnitPipeline) -> tuple[dict[str, object], ...]:
    """Return the executed pipeline stages, ending in the terminal-action node.

    The terminal node is always appended so every executed page carries the
    full ``select -> shape -> terminal`` chain in its ``pipeline_stages``,
    mirroring the typed AST (#2006).
    """

    return tuple(stage.to_payload() for stage in pipeline.stages) + (pipeline.terminal.to_payload(),)


class UnsupportedTerminalActionError(ExpressionCompileError):
    """Raised when a query-unit terminal action has no registered executor.

    Typed and narrow by construction: an unknown or unwired terminal action
    fails loudly rather than silently broadening to a different terminal (#2006).
    """


@dataclass(frozen=True)
class QueryUnitRequest:
    """Compiled terminal query-unit request shared by daemon, MCP, and API callers."""

    expression: str
    source: QueryUnitSource
    limit: int
    offset: int = 0
    session_filters: Mapping[str, object] | None = None


class _RowPayloadModel(Protocol):
    @classmethod
    def from_row(cls, row: Any) -> Any: ...


def _row_payload_model(descriptor: QueryUnitDescriptor) -> _RowPayloadModel | None:
    """Resolve the descriptor-owned row payload model."""

    model = getattr(surface_payloads, descriptor.payload_model, None)
    if model is None or not hasattr(model, "from_row"):
        return None
    return cast(_RowPayloadModel, model)


def _bool_param(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _epoch_ms(field: str, value: object) -> int | None:
    if isinstance(value, int):
        return value
    if value is None:
        return None
    parsed = parse_query_date(field, str(value))
    if parsed is None:
        return None
    return int(parsed.timestamp() * 1000)


def query_unit_session_filters(**params: object) -> dict[str, object]:
    """Normalize shared session filters for terminal query-unit rows.

    Terminal unit-source execution returns row-level
    results, but callers still need the same surrounding session scope as the
    normal session query surfaces.  This helper is the single cross-surface
    adapter into ``ArchiveStore.query_*``'s ``session_filters`` argument.
    """

    origin = optional_text(params.get("origin"))
    origins = split_csv(params.get("origins"))
    if not origins and origin is None:
        origins = split_csv(params.get("source"))
    excluded_origins = split_csv(params.get("excluded_origins") or params.get("exclude_origin"))
    tags = tuple(tag.lower() for tag in split_csv(params.get("tags") or params.get("tag")))
    excluded_tags = tuple(tag.lower() for tag in split_csv(params.get("excluded_tags") or params.get("exclude_tag")))
    repo_names = split_csv(params.get("repo_names") or params.get("repo"))
    has_types = split_csv(params.get("has_types") or params.get("has_type"))
    since_ms = params.get("since_ms")
    until_ms = params.get("until_ms")
    return {
        "origin": origin,
        "origins": origins,
        "excluded_origins": excluded_origins,
        "tags": tags,
        "excluded_tags": excluded_tags,
        "repo_names": repo_names,
        "has_types": has_types,
        "has_tool_use": _bool_param(params.get("has_tool_use") or params.get("filter_has_tool_use")),
        "has_thinking": _bool_param(params.get("has_thinking") or params.get("filter_has_thinking")),
        "has_paste": _bool_param(params.get("has_paste") or params.get("filter_has_paste")),
        "tool_terms": normalize_tool_terms(params.get("tool_terms") or params.get("tool")),
        "excluded_tool_terms": normalize_tool_terms(params.get("excluded_tool_terms") or params.get("exclude_tool")),
        "action_terms": normalize_action_terms("action", params.get("action_terms") or params.get("action")),
        "excluded_action_terms": normalize_action_terms(
            "exclude_action", params.get("excluded_action_terms") or params.get("exclude_action")
        ),
        "action_sequence": normalize_action_sequence(
            "action_sequence", params.get("action_sequence") or params.get("sequence")
        ),
        "action_text_terms": split_csv(params.get("action_text_terms") or params.get("action_text")),
        "referenced_paths": split_csv(params.get("referenced_paths") or params.get("referenced_path")),
        "cwd_prefix": optional_text(params.get("cwd_prefix")),
        "typed_only": _bool_param(params.get("typed_only")),
        "message_type": optional_message_type(params.get("message_type")),
        "title": optional_text(params.get("title")),
        "min_messages": optional_int(params.get("min_messages")),
        "max_messages": optional_int(params.get("max_messages")),
        "min_words": optional_int(params.get("min_words")),
        "max_words": optional_int(params.get("max_words")),
        "since_ms": int(since_ms) if isinstance(since_ms, int) else _epoch_ms("since", params.get("since")),
        "until_ms": int(until_ms) if isinstance(until_ms, int) else _epoch_ms("until", params.get("until")),
    }


def query_unit_request(
    *,
    expression: str,
    limit: int,
    offset: int = 0,
    session_filters: Mapping[str, object] | None = None,
    **filter_params: object,
) -> QueryUnitRequest:
    """Build a terminal query-unit request from surface parameters."""

    from polylogue.archive.query.expression import ExpressionCompileError, parse_unit_source_expression
    from polylogue.archive.query.metadata import terminal_query_source_list

    source = parse_unit_source_expression(expression)
    if source is None:
        raise ExpressionCompileError(
            f"query_units requires an explicit {terminal_query_source_list()} where expression",
            field=None,
        )
    filters = session_filters if session_filters is not None else query_unit_session_filters(**filter_params)
    return QueryUnitRequest(
        expression=expression,
        source=source,
        limit=limit,
        offset=offset,
        session_filters=filters,
    )


@dataclass(frozen=True)
class TerminalExecutionContext:
    """Resolved inputs for a single terminal-action executor invocation.

    Shared shape across every registered terminal so the dispatcher in
    :func:`_build_sql_envelope` stays a thin registry lookup and each executor
    runs the same ``select -> shape -> terminal`` chain (#2006).
    """

    archive: ArchiveStore
    source: QueryUnitSource
    descriptor: QueryUnitDescriptor
    query: str
    limit: int
    offset: int
    caller_offset: int
    fetch_limit: int
    session_filters: Mapping[str, object] | None


TerminalExecutor = Callable[[TerminalExecutionContext], QueryUnitResultEnvelope]


def _execute_count_terminal(ctx: TerminalExecutionContext) -> QueryUnitResultEnvelope:
    """Terminal ``count`` action: emit the aggregate group rollup page."""

    pipeline = ctx.source.pipeline
    aggregate_sort = (
        cast(Literal["count", "key"], pipeline.sort.field)
        if pipeline.sort is not None and pipeline.sort.field in {"count", "key"}
        else None
    )
    aggregate_sort_direction: Literal["asc", "desc"] = (
        pipeline.sort.direction if aggregate_sort is not None and pipeline.sort is not None else "desc"
    )
    aggregate_rows = ctx.archive.query_unit_counts(
        pipeline.source_unit,
        pipeline.predicate,
        group_by=pipeline.group_by,
        sort=aggregate_sort,
        sort_direction=aggregate_sort_direction,
        limit=ctx.fetch_limit,
        offset=ctx.offset,
        session_filters=ctx.session_filters,
    )
    return build_query_unit_aggregate_envelope(
        tuple(QueryUnitAggregateRowPayload.from_row(row) for row in aggregate_rows[: ctx.limit]),
        unit=ctx.source.unit,
        query=ctx.query,
        limit=ctx.limit,
        offset=ctx.caller_offset,
        has_next=len(aggregate_rows) > ctx.limit,
        pipeline=pipeline.to_payload(),
        pipeline_stages=_pipeline_stage_payloads(pipeline),
    )


def _execute_rows_terminal(ctx: TerminalExecutionContext) -> QueryUnitResultEnvelope:
    """Terminal ``rows`` action: emit the resolved unit-row page."""

    pipeline = ctx.source.pipeline
    sort = pipeline.sort.field if pipeline.sort is not None else None
    sort_direction = pipeline.sort.direction if pipeline.sort is not None else "asc"
    method_name = ctx.descriptor.sql_query_method
    payload_model = _row_payload_model(ctx.descriptor)
    if method_name is None or payload_model is None:
        raise ValueError(f"Query unit {ctx.source.unit!r} is not wired to a SQL executor")
    query_method = cast(Any, getattr(ctx.archive, method_name))
    rows = cast(
        Sequence[Any],
        query_method(
            pipeline.predicate,
            limit=ctx.fetch_limit,
            offset=ctx.offset,
            session_filters=ctx.session_filters,
            sort=sort,
            sort_direction=sort_direction,
        ),
    )
    return build_query_unit_envelope(
        tuple(payload_model.from_row(row) for row in rows[: ctx.limit]),
        unit=ctx.source.unit,
        query=ctx.query,
        limit=ctx.limit,
        offset=ctx.caller_offset,
        has_next=len(rows) > ctx.limit,
        pipeline=pipeline.to_payload(),
        pipeline_stages=_pipeline_stage_payloads(pipeline),
    )


#: Single source of truth mapping a terminal-action name to its executor.
#: The pipeline's ``terminal.action`` selects the executor; one executor runs
#: the full ``select -> shape -> terminal`` chain for every read surface
#: (CLI find, MCP query_units, daemon /api/query-units, Python API) (#2006).
TERMINAL_ACTION_EXECUTORS: dict[str, TerminalExecutor] = {
    "rows": _execute_rows_terminal,
    "count": _execute_count_terminal,
}


def _build_sql_envelope(
    archive: ArchiveStore,
    source: QueryUnitSource,
    descriptor: QueryUnitDescriptor,
    *,
    query: str,
    limit: int,
    offset: int,
    caller_offset: int,
    fetch_limit: int,
    session_filters: Mapping[str, object] | None,
) -> QueryUnitResultEnvelope:
    pipeline = source.pipeline
    action = pipeline.terminal.action
    executor = TERMINAL_ACTION_EXECUTORS.get(action)
    if executor is None:
        registered = ", ".join(sorted(TERMINAL_ACTION_EXECUTORS))
        raise UnsupportedTerminalActionError(
            f"unsupported terminal action {action!r} for {source.unit} rows; registered actions: {registered}",
            field=None,
        )
    return executor(
        TerminalExecutionContext(
            archive=archive,
            source=source,
            descriptor=descriptor,
            query=query,
            limit=limit,
            offset=offset,
            caller_offset=caller_offset,
            fetch_limit=fetch_limit,
            session_filters=session_filters,
        )
    )


def query_unit_rows(
    archive: ArchiveStore,
    source: QueryUnitSource,
    *,
    query: str,
    limit: int,
    offset: int = 0,
    session_filters: Mapping[str, object] | None = None,
) -> QueryUnitResultEnvelope:
    """Execute an explicit unit-source query."""

    caller_offset = offset
    pipeline = source.pipeline
    if pipeline.limit is not None:
        limit = min(limit, pipeline.limit)
    if pipeline.offset is not None:
        offset += pipeline.offset
    fetch_limit = limit + 1
    descriptor = query_unit_descriptor(source.unit)
    if descriptor is None or not descriptor.terminal_supported:
        raise ValueError(f"Unsupported terminal query unit: {source.unit}")
    return _build_sql_envelope(
        archive,
        source,
        descriptor,
        query=query,
        limit=limit,
        offset=offset,
        caller_offset=caller_offset,
        fetch_limit=fetch_limit,
        session_filters=session_filters,
    )


def query_unit_envelope(archive: ArchiveStore, request: QueryUnitRequest) -> QueryUnitResultEnvelope:
    """Execute a compiled terminal query-unit request."""

    return query_unit_rows(
        archive,
        request.source,
        query=request.expression,
        limit=request.limit,
        offset=request.offset,
        session_filters=request.session_filters,
    )


__all__ = [
    "QueryUnitRequest",
    "TERMINAL_ACTION_EXECUTORS",
    "TerminalExecutionContext",
    "TerminalExecutor",
    "UnsupportedTerminalActionError",
    "query_unit_envelope",
    "query_unit_request",
    "query_unit_rows",
    "query_unit_session_filters",
]
