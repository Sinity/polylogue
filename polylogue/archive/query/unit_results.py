"""Terminal unit-query execution over the archive."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from time import monotonic
from typing import Any, Literal, Protocol, cast

from polylogue.archive.query.execution_control import QueryExecutionContext
from polylogue.archive.query.expression import (
    ExpressionCompileError,
    QueryUnitPipeline,
    QueryUnitSource,
)
from polylogue.archive.query.metadata import (
    QueryUnitDescriptor,
    query_unit_descriptor,
)
from polylogue.archive.query.scope import SurfaceSpec, validate_surface_specs
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
from polylogue.archive.query.transaction import (
    QueryContinuation,
    QueryTransactionRequest,
    query_units_transaction_request,
    validate_continuation_epoch,
)
from polylogue.operations.authority import authority_for_reader
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveAggMetricSpec, ArchiveStore
from polylogue.surfaces import payloads as surface_payloads
from polylogue.surfaces.authority import AuthorityEnvelope
from polylogue.surfaces.payloads import (
    QueryUnitAggregateRowPayload,
    QueryUnitProjectedRowPayload,
    QueryUnitResultEnvelope,
    QueryUnitRowPayload,
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


def _projected_rows(
    rows: Sequence[Any], descriptor: QueryUnitDescriptor, selected_fields: Sequence[str]
) -> tuple[QueryUnitProjectedRowPayload, ...]:
    """Build field-only payloads from either storage projections or full rows."""

    if not selected_fields:
        return ()
    return tuple(
        QueryUnitProjectedRowPayload(
            root={
                field: row[field]
                if isinstance(row, Mapping)
                else getattr(row, descriptor.projectable_fields[field], None)
                for field in selected_fields
            }
        )
        for row in rows
    )


def _bool_param(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _query_date_ms(field: str, value: object) -> int | None:
    """Lower one *query-grammar* date expression to epoch milliseconds.

    Deliberately not a timestamp-coercion helper (polylogue-z3sv): the input
    is grammar text such as ``yesterday``, resolved by
    :func:`parse_query_date`, not a wire timestamp.
    """
    if isinstance(value, int):
        return value
    if value is None:
        return None
    parsed = parse_query_date(field, str(value))
    if parsed is None:
        return None
    return int(parsed.timestamp() * 1000)


TerminalFilterKind = Literal["string", "integer", "boolean"]


@dataclass(frozen=True, slots=True)
class TerminalFilterParameter:
    """One surface-facing session filter accepted by terminal query-unit reads.

    ``query_unit_session_filters`` takes ``**params``, so before this table the
    only enumeration of the accepted names lived in hand-typed OpenAPI
    parameter objects, which had already drifted (``origins`` and
    ``exclude_origin`` reach the handler but were undocumented).  This is the
    single declaration of that surface: ``devtools render openapi`` projects it
    into the ``/api/query-units`` parameter list, and
    ``tests/unit/daemon/test_query_unit_filter_declarations.py`` resolves every
    row against the live handler call and the live filter lowering, so a
    declared-but-dead row and a live-but-undeclared parameter both fail.
    """

    #: Public query-parameter name on ``GET /api/query-units``.
    name: str
    #: The ``query_unit_request`` keyword the handler passes it as.
    request_kwarg: str
    #: The ``session_filters`` key it lowers to.
    filter_key: str
    kind: TerminalFilterKind
    description: str


TERMINAL_FILTER_PARAMETERS: tuple[TerminalFilterParameter, ...] = (
    TerminalFilterParameter(
        "origin", "origin", "origin", "string", "Optional session-origin scope for terminal row results."
    ),
    TerminalFilterParameter(
        "origins",
        "origins",
        "origins",
        "string",
        "Optional comma-separated session-origin scope for terminal row results.",
    ),
    TerminalFilterParameter(
        "exclude_origin",
        "exclude_origin",
        "excluded_origins",
        "string",
        "Optional comma-separated session origins excluded from terminal row results.",
    ),
    TerminalFilterParameter("tag", "tag", "tags", "string", "Optional session tag scope for terminal row results."),
    TerminalFilterParameter(
        "exclude_tag",
        "exclude_tag",
        "excluded_tags",
        "string",
        "Optional comma-separated session tags to exclude from terminal row results.",
    ),
    TerminalFilterParameter(
        "repo", "repo", "repo_names", "string", "Optional comma-separated repo-name scope for terminal row results."
    ),
    TerminalFilterParameter(
        "has_type",
        "has_type",
        "has_types",
        "string",
        "Optional comma-separated block types required on the containing session.",
    ),
    TerminalFilterParameter(
        "referenced_path",
        "referenced_path",
        "referenced_paths",
        "string",
        "Optional comma-separated referenced paths required on the containing session.",
    ),
    TerminalFilterParameter(
        "cwd_prefix",
        "cwd_prefix",
        "cwd_prefix",
        "string",
        "Optional working-directory prefix required on the containing session.",
    ),
    TerminalFilterParameter(
        "tool",
        "tool",
        "tool_terms",
        "string",
        "Optional comma-separated tool names required on the containing session.",
    ),
    TerminalFilterParameter(
        "exclude_tool",
        "exclude_tool",
        "excluded_tool_terms",
        "string",
        "Optional comma-separated tool names excluded from the containing session.",
    ),
    TerminalFilterParameter(
        "action",
        "action",
        "action_terms",
        "string",
        "Optional comma-separated action kinds required on the containing session.",
    ),
    TerminalFilterParameter(
        "exclude_action",
        "exclude_action",
        "excluded_action_terms",
        "string",
        "Optional comma-separated action kinds excluded from the containing session.",
    ),
    TerminalFilterParameter(
        "action_sequence",
        "action_sequence",
        "action_sequence",
        "string",
        "Optional action sequence required on the containing session.",
    ),
    TerminalFilterParameter(
        "action_text",
        "action_text",
        "action_text_terms",
        "string",
        "Optional action text required on the containing session.",
    ),
    TerminalFilterParameter(
        "title", "title", "title", "string", "Optional session-title substring scope for terminal row results."
    ),
    TerminalFilterParameter(
        "since", "since", "since_ms", "string", "Optional session lower time bound, using the shared query date parser."
    ),
    TerminalFilterParameter(
        "until", "until", "until_ms", "string", "Optional session upper time bound, using the shared query date parser."
    ),
    TerminalFilterParameter(
        "has_tool_use",
        "has_tool_use",
        "has_tool_use",
        "boolean",
        "Restrict terminal rows to sessions with tool-use evidence.",
    ),
    TerminalFilterParameter(
        "has_paste_evidence",
        "has_paste",
        "has_paste",
        "boolean",
        "Restrict terminal rows to sessions with paste evidence.",
    ),
    TerminalFilterParameter(
        "has_thinking",
        "has_thinking",
        "has_thinking",
        "boolean",
        "Restrict terminal rows to sessions with thinking blocks.",
    ),
    TerminalFilterParameter(
        "typed_only",
        "typed_only",
        "typed_only",
        "boolean",
        "Restrict terminal rows to typed sessions without paste evidence.",
    ),
    TerminalFilterParameter(
        "min_messages",
        "min_messages",
        "min_messages",
        "integer",
        "Restrict terminal rows to sessions with at least this many messages.",
    ),
    TerminalFilterParameter(
        "max_messages",
        "max_messages",
        "max_messages",
        "integer",
        "Restrict terminal rows to sessions with at most this many messages.",
    ),
    TerminalFilterParameter(
        "min_words",
        "min_words",
        "min_words",
        "integer",
        "Restrict terminal rows to sessions with at least this many words.",
    ),
    TerminalFilterParameter(
        "max_words",
        "max_words",
        "max_words",
        "integer",
        "Restrict terminal rows to sessions with at most this many words.",
    ),
    TerminalFilterParameter(
        "message_type",
        "message_type",
        "message_type",
        "string",
        "Restrict terminal rows by session message-type evidence.",
    ),
)

TERMINAL_FILTER_PARAMETER_BY_NAME: dict[str, TerminalFilterParameter] = {
    parameter.name: parameter for parameter in TERMINAL_FILTER_PARAMETERS
}


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
    project_refs = split_csv(params.get("project_refs") or params.get("project"))
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
        "project_refs": project_refs,
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
        "since_ms": int(since_ms) if isinstance(since_ms, int) else _query_date_ms("since", params.get("since")),
        "until_ms": int(until_ms) if isinstance(until_ms, int) else _query_date_ms("until", params.get("until")),
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
    execution_context: QueryExecutionContext | None = None


TerminalExecutor = Callable[[TerminalExecutionContext], QueryUnitResultEnvelope]


def _aggregate_group_fields(group_by: str | None) -> tuple[str, ...]:
    return () if group_by is None else tuple(field.strip() for field in group_by.split(",") if field.strip())


def _record_result_page(
    ctx: TerminalExecutionContext,
    envelope: QueryUnitResultEnvelope,
    *,
    selected_rows_exact: int | None = None,
) -> QueryUnitResultEnvelope:
    if ctx.execution_context is not None:
        ctx.execution_context.record_result_page(
            emitted_rows=len(envelope.items) or len(envelope.projected_items),
            selected_rows_exact=selected_rows_exact,
        )
    return envelope


def _aggregate_pipeline_payload(
    pipeline: QueryUnitPipeline,
    *,
    group_fields: tuple[str, ...],
    denominator: int,
    missing_counts: Mapping[str, int],
    unknown_counts: Mapping[str, int],
    groups: Sequence[tuple[tuple[str, ...], int]],
) -> dict[str, object]:
    payload = pipeline.to_payload()
    if len(group_fields) <= 1:
        return payload
    result = cast(dict[str, object], payload.setdefault("result", {}))
    result.update(
        {
            "group_by": group_fields[0] if len(group_fields) == 1 else list(group_fields),
            "aggregate": ["count", "proportion"],
            "denominator": {"kind": "all_matching_rows", "n": denominator},
            "n": denominator,
            "missing_counts": {field: missing_counts[field] for field in group_fields},
            "unknown_counts": {field: unknown_counts[field] for field in group_fields},
            "groups": [
                {
                    "group": values[0] if len(group_fields) == 1 else dict(zip(group_fields, values, strict=True)),
                    "count": count,
                    "proportion": count / denominator if denominator else 0.0,
                }
                for values, count in groups
            ],
        }
    )
    return payload


def _execute_count_terminal(ctx: TerminalExecutionContext) -> QueryUnitResultEnvelope:
    """Terminal ``count`` action: emit the aggregate group rollup page."""

    pipeline = ctx.source.pipeline
    aggregate_sort: Literal["count", "key"] | None = None
    if pipeline.sort is not None:
        if pipeline.sort.field == "count":
            aggregate_sort = "count"
        elif pipeline.sort.field == "key":
            aggregate_sort = "key"
    aggregate_sort_direction: Literal["asc", "desc"] = (
        pipeline.sort.direction if aggregate_sort is not None and pipeline.sort is not None else "desc"
    )
    group_fields = _aggregate_group_fields(pipeline.group_by)
    if len(group_fields) <= 1:
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
        return _record_result_page(
            ctx,
            build_query_unit_aggregate_envelope(
                tuple(QueryUnitAggregateRowPayload.from_row(row) for row in aggregate_rows[: ctx.limit]),
                unit=ctx.source.unit,
                query=ctx.query,
                limit=ctx.limit,
                offset=ctx.caller_offset,
                has_next=len(aggregate_rows) > ctx.limit,
                pipeline=pipeline.to_payload(),
                pipeline_stages=_pipeline_stage_payloads(pipeline),
            ),
        )

    aggregate_page = ctx.archive.query_unit_multi_counts(
        pipeline.source_unit,
        pipeline.predicate,
        group_by=group_fields,
        sort=aggregate_sort,
        sort_direction=aggregate_sort_direction,
        limit=ctx.fetch_limit,
        offset=ctx.offset,
        session_filters=ctx.session_filters,
    )
    page_groups = [(row.group_values, row.count) for row in aggregate_page.rows]
    missing_counts = dict(zip(group_fields, aggregate_page.missing_counts, strict=True))
    unknown_counts = dict(zip(group_fields, aggregate_page.unknown_counts, strict=True))
    group_by = pipeline.group_by
    aggregate_payload_rows = tuple(
        QueryUnitAggregateRowPayload(
            unit=ctx.source.unit,
            group_by=group_by,
            group_key=(
                values[0]
                if len(group_fields) == 1
                else json.dumps(dict(zip(group_fields, values, strict=True)), sort_keys=True, separators=(",", ":"))
            ),
            count=count,
        )
        for values, count in page_groups[: ctx.limit]
    )
    return _record_result_page(
        ctx,
        build_query_unit_aggregate_envelope(
            aggregate_payload_rows,
            unit=ctx.source.unit,
            query=ctx.query,
            limit=ctx.limit,
            offset=ctx.caller_offset,
            has_next=len(page_groups) > ctx.limit,
            pipeline=_aggregate_pipeline_payload(
                pipeline,
                group_fields=group_fields,
                denominator=aggregate_page.denominator,
                missing_counts=missing_counts,
                unknown_counts=unknown_counts,
                groups=page_groups[: ctx.limit],
            ),
            pipeline_stages=_pipeline_stage_payloads(pipeline),
        ),
        selected_rows_exact=aggregate_page.denominator,
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
    if ctx.source.unit == "message" and pipeline.selected_fields:
        rows = cast(
            Sequence[Any],
            ctx.archive.query_message_projection(
                pipeline.predicate,
                fields=pipeline.selected_fields,
                limit=ctx.fetch_limit,
                offset=ctx.offset,
                session_filters=ctx.session_filters,
                sort=sort,
                sort_direction=sort_direction,
            ),
        )
        page_items: tuple[QueryUnitRowPayload, ...] = ()
        projected_items = _projected_rows(rows[: ctx.limit], ctx.descriptor, pipeline.selected_fields)
    else:
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
        page_items = tuple(payload_model.from_row(row) for row in rows[: ctx.limit])
        projected_items = _projected_rows(rows[: ctx.limit], ctx.descriptor, pipeline.selected_fields)
    return _record_result_page(
        ctx,
        build_query_unit_envelope(
            page_items,
            unit=ctx.source.unit,
            query=ctx.query,
            limit=ctx.limit,
            offset=ctx.caller_offset,
            has_next=len(rows) > ctx.limit,
            pipeline=pipeline.to_payload(),
            pipeline_stages=_pipeline_stage_payloads(pipeline),
            projected_items=projected_items,
        ),
    )


def _execute_agg_terminal(ctx: TerminalExecutionContext) -> QueryUnitResultEnvelope:
    """Terminal ``agg`` action: emit named sum/avg/min/max/percentile metrics per group.

    Every reducer is evaluated by SQLite over the complete predicate-matching
    relation through :meth:`ArchiveStore.query_unit_agg_metrics`, and Python
    retains only the requested aggregate page. The reported metric therefore
    does not change meaning with the size of the match set: there is one
    regime, not a bounded-sample regime above some row count.
    """

    pipeline = ctx.source.pipeline
    agg_metrics = pipeline.agg_metrics
    assert agg_metrics is not None
    group_fields = _aggregate_group_fields(pipeline.group_by)
    page = ctx.archive.query_unit_agg_metrics(
        ctx.source.unit,
        pipeline.predicate,
        group_by=group_fields,
        metrics=tuple(
            ArchiveAggMetricSpec(label=metric.label, fn=metric.fn, field=metric.field) for metric in agg_metrics
        ),
        limit=ctx.fetch_limit,
        offset=ctx.offset,
        session_filters=ctx.session_filters,
    )

    aggregate_rows: list[QueryUnitAggregateRowPayload] = []
    for row in page.rows:
        if not group_fields:
            group_key = None
        elif len(group_fields) == 1:
            group_key = row.group_values[0]
        else:
            group_key = json.dumps(
                dict(zip(group_fields, row.group_values, strict=True)), sort_keys=True, separators=(",", ":")
            )
        aggregate_rows.append(
            QueryUnitAggregateRowPayload(
                unit=cast(Any, ctx.source.unit),
                group_by=pipeline.group_by,
                group_key=group_key,
                count=row.count,
                metrics=dict(row.metrics),
            )
        )

    return _record_result_page(
        ctx,
        build_query_unit_aggregate_envelope(
            tuple(aggregate_rows[: ctx.limit]),
            unit=ctx.source.unit,
            query=ctx.query,
            limit=ctx.limit,
            offset=ctx.caller_offset,
            has_next=len(aggregate_rows) > ctx.limit,
            pipeline=pipeline.to_payload(),
            pipeline_stages=_pipeline_stage_payloads(pipeline),
        ),
        selected_rows_exact=page.total_groups,
    )


#: Single source of truth mapping a terminal-action name to its executor.
#: The pipeline's ``terminal.action`` selects the executor; one executor runs
#: the full ``select -> shape -> terminal`` chain for every read surface
#: (CLI find, MCP query_units, daemon /api/query-units, Python API) (#2006).
TERMINAL_ACTION_SPECS: tuple[SurfaceSpec, ...] = (
    SurfaceSpec("rows", "query-unit", True, lambda predicate: dict(predicate)),
    SurfaceSpec("count", "query-unit", True, lambda predicate: dict(predicate)),
    SurfaceSpec("agg", "query-unit", True, lambda predicate: dict(predicate)),
)
validate_surface_specs(TERMINAL_ACTION_SPECS)

TERMINAL_ACTION_EXECUTORS: dict[str, TerminalExecutor] = {
    "rows": _execute_rows_terminal,
    "count": _execute_count_terminal,
    "agg": _execute_agg_terminal,
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
    execution_context: QueryExecutionContext | None,
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
            execution_context=execution_context,
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
    execution_context: QueryExecutionContext | None = None,
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
        execution_context=execution_context,
    )


def query_unit_envelope(
    archive: ArchiveStore,
    request: QueryUnitRequest,
    *,
    execution_context: QueryExecutionContext | None = None,
    transaction_request: QueryTransactionRequest | None = None,
    authority: AuthorityEnvelope | None = None,
    serving_identity: str = "direct",
) -> QueryUnitResultEnvelope:
    """Execute a compiled terminal query-unit request.

    Callers that already own a :class:`QueryTransaction` pass its canonical
    request so the page, receipt, and continuation share one identity.  The
    optional argument retains the established direct-storage API for callers
    that do not cross a transaction boundary.
    """
    started_at = monotonic()
    canonical_request = transaction_request
    if canonical_request is None:
        canonical_request = query_units_transaction_request(
            expression=request.expression,
            session_filters=request.session_filters or {},
            page_size=request.limit,
            offset=request.offset,
        )
    if canonical_request.operation != "query_units":
        raise ValueError("query-unit envelope requires a query_units transaction")
    if canonical_request.archive_epoch:
        validate_continuation_epoch(canonical_request, archive=archive)
    else:
        # The frame must be captured before the first result statement so
        # every component is served from this reader's one owned snapshot.
        # It is narrowed to the relations this lowered request can read, so
        # an unrelated write cannot 409 the next page.
        from polylogue.archive.query.frame_scope import query_unit_frame_relations
        from polylogue.archive.query.transaction import archive_snapshot_epoch

        canonical_request = canonical_request.with_archive_epoch(
            archive_snapshot_epoch(
                archive,
                relations=query_unit_frame_relations(request.source, request.session_filters),
            )
        )
    envelope = query_unit_rows(
        archive,
        request.source,
        query=request.expression,
        limit=request.limit,
        offset=request.offset,
        session_filters=request.session_filters,
        execution_context=execution_context,
    )
    result_ref = canonical_request.result_ref
    next_offset = getattr(envelope, "next_offset", None)
    continuation = (
        QueryContinuation(
            request=canonical_request.next(offset=next_offset),
            result_ref=result_ref,
        ).encode()
        if next_offset is not None
        else None
    )
    result = envelope.model_copy(
        update={
            "query_ref": canonical_request.query_ref,
            "result_ref": result_ref,
            "continuation": continuation,
            "authority": authority
            or authority_for_reader(
                archive,
                server_identity="daemon" if serving_identity == "daemon" else "direct",
                started_at=started_at,
            ),
        }
    )
    # MCP may clip one physical storage page to its smaller byte budget. Keep
    # the framed request outside the public payload so it can mint a cursor
    # from the retained prefix even when the storage page was final.
    object.__setattr__(result, "_transaction_request", canonical_request)
    return result


__all__ = [
    "TERMINAL_FILTER_PARAMETERS",
    "TERMINAL_FILTER_PARAMETER_BY_NAME",
    "TerminalFilterParameter",
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
