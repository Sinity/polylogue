"""Query execution entrypoint, routing, planning, and shared helpers for the query-first CLI."""

from __future__ import annotations

import inspect
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Any, NoReturn

import click

from polylogue.cli.root_request import RootModeRequest
from polylogue.cli.types import AppEnv
from polylogue.lib.query_spec import ConversationQuerySpec, QuerySpecError
from polylogue.logging import get_logger
from polylogue.sync_bridge import run_coroutine_sync

logger = get_logger(__name__)

if TYPE_CHECKING:
    from polylogue.lib.models import Conversation, ConversationSummary


# ---------------------------------------------------------------------------
# Query helpers (from query_helpers.py)
# ---------------------------------------------------------------------------


def coerce_query_spec(params: dict[str, Any] | ConversationQuerySpec) -> ConversationQuerySpec:
    if isinstance(params, ConversationQuerySpec):
        return params
    return ConversationQuerySpec.from_params(params)


def describe_query_filters(params: dict[str, Any] | ConversationQuerySpec) -> list[str]:
    """Build a human-readable list of active filters from params or spec."""
    return coerce_query_spec(params).describe()


def no_results(
    env: AppEnv,
    params: dict[str, Any] | ConversationQuerySpec,
    *,
    exit_code: int = 2,
) -> NoReturn:
    """Print a helpful no-results message and exit."""
    filters = describe_query_filters(params)
    if filters:
        click.echo("No conversations matched filters:", err=True)
        for item in filters:
            click.echo(f"  {item}", err=True)
        click.echo("Hint: try broadening your filters or use `list` to browse", err=True)
    else:
        click.echo("No conversations matched.", err=True)
    raise SystemExit(exit_code)


def result_id(result: Conversation | ConversationSummary) -> str:
    return str(result.id)


def result_provider(result: Conversation | ConversationSummary) -> str:
    return str(result.provider)


def result_title(result: Conversation | ConversationSummary) -> str:
    title = result.display_title
    return title if title else result_id(result)[:20]


def result_date(result: Conversation | ConversationSummary) -> datetime | None:
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


def summary_to_dict(summary: ConversationSummary, message_count: int) -> dict[str, object]:
    return {
        "id": str(summary.id),
        "provider": str(summary.provider),
        "title": summary.display_title,
        "date": summary.display_date.isoformat() if summary.display_date else None,
        "tags": summary.tags,
        "summary": summary.summary,
        "messages": message_count,
    }


# ---------------------------------------------------------------------------
# Query plan (from query_plan.py)
# ---------------------------------------------------------------------------


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
        return self.action == QueryAction.STATS_BY and self.stats_dimension in {"provider", "month", "year", "day"}

    def prefers_summary_mutation(self) -> bool:
        return self.action in {QueryAction.MODIFY, QueryAction.DELETE}


def resolve_query_route(
    plan: QueryExecutionPlan,
    *,
    can_use_summaries: bool,
) -> QueryRoute:
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
            "delete requires at least one filter to prevent accidental deletion of the entire archive."
        )

    return QueryExecutionPlan(
        selection=selection,
        action=action,
        output=output,
        mutation=mutation,
        stats_dimension=stats_dimension,
    )


# ---------------------------------------------------------------------------
# Query frontdoor (from query_frontdoor.py)
# ---------------------------------------------------------------------------

_ROOT_GLOBAL_OPTIONS = frozenset({"--plain", "--verbose", "-v"})


def _option_arity(group: click.Group) -> dict[str, int]:
    value_options: dict[str, int] = {}
    for param in group.params:
        if isinstance(param, click.Option) and not param.is_flag:
            nargs = param.nargs if param.nargs > 0 else 1
            for opt in param.opts + param.secondary_opts:
                value_options[opt] = nargs
    return value_options


def _matches_option(option: str, token: str) -> bool:
    return token == option or token.startswith(f"{option}=")


def _is_root_global_option(token: str) -> bool:
    return any(_matches_option(option, token) for option in _ROOT_GLOBAL_OPTIONS)


def _iter_option_values(args: list[str], start: int, nargs: int) -> Iterable[str]:
    for offset in range(1, nargs + 1):
        if start + offset < len(args):
            yield args[start + offset]


def _split_query_mode_args(group: click.Group, args: list[str]) -> tuple[list[str], tuple[str, ...], bool]:
    from polylogue.cli.query_verbs import VERB_NAMES

    option_arity = _option_arity(group)
    option_args: list[str] = []
    query_terms: list[str] = []
    index = 0

    while index < len(args):
        arg = args[index]
        if arg == "--":
            query_terms.extend(args[index + 1 :])
            break
        if arg.startswith("-"):
            option_args.append(arg)
            nargs = option_arity.get(arg, 0)
            option_args.extend(_iter_option_values(args, index, nargs))
            index += nargs + 1
            continue
        # Non-verb subcommands (run, doctor, products, etc.) recognized when
        # no bare-word query terms have been collected yet.  Filter options
        # (--provider, --since, …) do NOT prevent subcommand detection — they
        # are consumed by the root group during Click's normal parse phase.
        if not query_terms and arg in group.commands and arg not in VERB_NAMES:
            return args, (), True
        # Verb commands recognized at any position (even after filters/query terms)
        if arg in VERB_NAMES:
            verb_args = option_args + [arg] + list(args[index + 1 :])
            return verb_args, tuple(query_terms), True
        query_terms.append(arg)
        index += 1

    return option_args, tuple(query_terms), False


def handle_query_mode(
    ctx: click.Context,
    *,
    show_stats: Any,
) -> None:
    """Handle query mode: display stats or perform search."""
    env: AppEnv = ctx.obj
    request = RootModeRequest.from_context(ctx)

    if request.should_show_stats():
        show_stats(env, verbose=bool(request.params.get("verbose", False)))
        return

    execute_query(env, request.query_params())


class QueryFirstGroupBase(click.Group):
    """Custom Click group that routes to query mode by default."""

    def parse_args(self, ctx: click.Context, args: list[str]) -> list[str]:
        """Parse args, preserving raw query terms instead of rewriting them as hidden options."""
        parse_args, query_terms, has_subcommand = _split_query_mode_args(self, args)
        ctx.meta["polylogue_has_subcommand"] = has_subcommand
        if not has_subcommand:
            ctx.meta["polylogue_query_terms"] = query_terms
        return list(super().parse_args(ctx, parse_args))

    def invoke(self, ctx: click.Context) -> Any:
        """Invoke the group, dispatching to query or stats mode if no subcommand."""
        if ctx.meta.get("polylogue_has_subcommand", False):
            return super().invoke(ctx)

        assert self.callback is not None, "QueryFirstGroup requires a callback"
        with ctx:
            ctx.invoke(self.callback, **ctx.params)

        self.handle_default_mode(ctx)

    def handle_default_mode(self, ctx: click.Context) -> None:
        """Dispatch no-subcommand mode for subclasses."""
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Query execution (original query.py)
# ---------------------------------------------------------------------------


def project_query_results(results: list[Any], plan: QueryExecutionPlan) -> list[Any]:
    """Apply post-selection transforms consistently before final output."""
    from polylogue.cli import query_actions as _query_actions

    projected = results
    if plan.output.transform is not None:
        projected = _query_actions.apply_transform(projected, plan.output.transform)
    if plan.output.dialogue_only:
        projected = [conversation.dialogue_only() for conversation in projected]
    return projected


def execute_query(env: AppEnv, params: dict[str, Any]) -> None:
    """Execute a query-mode command."""
    run_coroutine_sync(async_execute_query(env, params))


def _create_query_vector_provider(config: object) -> object | None:
    """Best-effort vector provider setup for query execution."""
    from polylogue.storage.search_providers import create_vector_provider

    try:
        return create_vector_provider(config)
    except (ValueError, ImportError):
        return None
    except Exception as exc:
        logger.warning("Vector search setup failed: %s", exc)
        return None


async def _await_if_needed(value: Any) -> Any:
    if inspect.isawaitable(value):
        return await value
    return value


async def async_execute_query(env: AppEnv, params: dict[str, Any]) -> None:
    """Async core of execute_query."""
    from polylogue.cli import query_actions as _query_actions
    from polylogue.cli import query_output as _query_output
    from polylogue.cli.helpers import fail, load_effective_config
    from polylogue.config import ConfigError

    try:
        config = load_effective_config(env)
    except ConfigError as exc:
        fail("query", str(exc))

    repo = env.repository

    vector_provider = _create_query_vector_provider(config)

    try:
        plan = build_query_execution_plan(params)
    except QueryPlanError as exc:
        click.echo(f"Error: {exc}", err=True)
        raise SystemExit(1) from exc

    if plan.selection.similar_text and vector_provider is None:
        click.echo(
            "Error: --similar requires vector search support. Configure VOYAGE_API_KEY and build embeddings with `polylogue embed` first.",
            err=True,
        )
        raise SystemExit(1)
    if plan.selection.similar_text:
        archive_stats = await repo.get_archive_stats()
        if archive_stats.embedded_messages <= 0:
            click.echo(
                "Error: --similar requires existing embeddings. Run `polylogue embed` first.",
                err=True,
            )
            raise SystemExit(1)

    try:
        filter_chain = plan.selection.build_filter(
            repo,
            vector_provider=vector_provider,
        )
    except QuerySpecError as exc:
        if exc.field in {"since", "until"}:
            click.echo(f"Error: Cannot parse date: '{exc.value}'", err=True)
            click.echo(
                "Hint: use ISO format (2025-01-15), relative ('yesterday', 'last week'), or month (2025-01)",
                err=True,
            )
        else:
            click.echo(f"Error: invalid {exc.field}: '{exc.value}'", err=True)
        raise SystemExit(1) from exc

    route = resolve_query_route(plan, can_use_summaries=filter_chain.can_use_summaries())

    if route == QueryRoute.COUNT:
        click.echo(await filter_chain.count())
        return

    if route == QueryRoute.SUMMARY_LIST:
        summary_results = await filter_chain.list_summaries()
        if not summary_results:
            no_results(env, plan.selection)
        await _query_output._output_summary_list(env, summary_results, params, repo)
        return

    if route == QueryRoute.STREAM:
        full_id = await _query_actions.resolve_stream_target(repo, filter_chain, plan.selection)
        if plan.output.transform is not None:
            click.echo(
                "Warning: --transform is ignored in --stream mode (messages are streamed individually).",
                err=True,
            )
        if any(dest != "stdout" for dest in plan.output.destinations):
            click.echo(
                f"Warning: --output {','.join(plan.output.destinations)} is ignored in --stream mode (output goes to stdout).",
                err=True,
            )

        await _query_output.stream_conversation(
            env,
            repo,
            full_id,
            output_format=plan.output.stream_format(),
            dialogue_only=plan.output.dialogue_only,
            message_limit=params.get("limit"),
        )
        return

    if route == QueryRoute.STATS_SQL:
        await _query_output.output_stats_sql(env, filter_chain, repo, output_format=plan.output.output_format)
        return

    if route == QueryRoute.SUMMARY_STATS:
        summaries = await filter_chain.list_summaries()
        msg_counts = await repo.queries.get_message_counts_batch([str(summary.id) for summary in summaries])
        _query_output.output_stats_by_summaries(
            env,
            summaries,
            msg_counts,
            plan.stats_dimension or "all",
            output_format=plan.output.output_format,
        )
        return

    if route == QueryRoute.STATS_BY and plan.stats_dimension in {"action", "tool"}:
        query_plan = filter_chain.build_query_plan()
        if filter_chain.can_use_summaries():
            summaries = await filter_chain.list_summaries()
            await _query_output.output_stats_by_semantic_summaries(
                env,
                summaries,
                repo,
                plan.stats_dimension or "all",
                selection=plan.selection,
                output_format=plan.output.output_format,
            )
            return
        if await _await_if_needed(query_plan.can_use_action_event_stats_with(repo)) is True:
            records_result = repo.queries.list_conversations(query_plan.record_query.with_limit(query_plan.limit))
            if inspect.isawaitable(records_result):
                records = await records_result
                await _query_output.output_stats_by_semantic_query(
                    env,
                    [record.conversation_id for record in records],
                    repo,
                    plan.stats_dimension or "all",
                    selection=plan.selection,
                    output_format=plan.output.output_format,
                )
                return

    if route == QueryRoute.STATS_BY and plan.stats_dimension in {"project", "work-kind"}:
        if filter_chain.can_use_summaries():
            summaries = await filter_chain.list_summaries()
            await _query_output.output_stats_by_profile_summaries(
                env,
                summaries,
                repo,
                plan.stats_dimension or "all",
                output_format=plan.output.output_format,
            )
            return
        query_plan = filter_chain.build_query_plan()
        records = await repo.queries.list_conversations(query_plan.record_query.with_limit(query_plan.limit))
        await _query_output.output_stats_by_profile_query(
            env,
            [record.conversation_id for record in records],
            repo,
            plan.stats_dimension or "all",
            output_format=plan.output.output_format,
        )
        return

    if route in {QueryRoute.SUMMARY_MODIFY, QueryRoute.SUMMARY_DELETE}:
        results = await filter_chain.list_summaries()
    else:
        results = await filter_chain.list()

    if route in {QueryRoute.MODIFY, QueryRoute.SUMMARY_MODIFY}:
        await _query_actions.apply_modifiers(env, results, params, repo)
        return

    if route in {QueryRoute.DELETE, QueryRoute.SUMMARY_DELETE}:
        await _query_actions.delete_conversations(env, results, params, repo)
        return

    results = project_query_results(results, plan)

    if route == QueryRoute.STATS_BY:
        _query_output._output_stats_by(
            env,
            results,
            plan.stats_dimension or "all",
            selection=plan.selection,
            output_format=plan.output.output_format,
        )
        return

    if route == QueryRoute.OPEN:
        _query_output._open_result(env, results, params)
        return

    _query_output.output_results(env, results, params)


__all__ = [
    "QueryAction",
    "QueryExecutionPlan",
    "QueryFirstGroupBase",
    "QueryMutationSpec",
    "QueryOutputSpec",
    "QueryPlanError",
    "QueryRoute",
    "build_query_execution_plan",
    "coerce_query_spec",
    "describe_query_filters",
    "execute_query",
    "handle_query_mode",
    "no_results",
    "resolve_query_route",
    "result_date",
    "result_id",
    "result_provider",
    "result_title",
    "summary_to_dict",
]
