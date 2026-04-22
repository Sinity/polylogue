"""Query execution entrypoint, routing, planning, and shared helpers for the query-first CLI."""

from __future__ import annotations

import inspect
from collections.abc import Awaitable, Iterable, Sequence
from typing import TYPE_CHECKING, Protocol, TypeVar, overload

import click

from polylogue.cli.query_contracts import (
    QueryAction,
    QueryExecutionPlan,
    QueryMutationSpec,
    QueryOutputSpec,
    QueryParams,
    QueryPlanError,
    QueryRoute,
    build_query_execution_plan,
    coerce_query_spec,
    describe_query_filters,
    resolve_query_route,
    result_date,
    result_id,
    result_provider,
    result_title,
)
from polylogue.cli.query_feedback import emit_no_results
from polylogue.cli.root_request import RootModeRequest
from polylogue.cli.types import AppEnv
from polylogue.lib.json import JSONDocument
from polylogue.lib.query_spec import QuerySpecError
from polylogue.logging import get_logger
from polylogue.surface_payloads import ConversationListRowPayload
from polylogue.sync_bridge import run_coroutine_sync

logger = get_logger(__name__)

if TYPE_CHECKING:
    from polylogue.config import Config
    from polylogue.lib.filters import ConversationFilter
    from polylogue.lib.models import Conversation, ConversationSummary
    from polylogue.lib.query_miss_diagnostics import QueryMissDiagnostics
    from polylogue.lib.query_spec import ConversationQuerySpec
    from polylogue.protocols import (
        ConversationArchiveStatsStore,
        ConversationQueryRuntimeStore,
        TagStore,
        VectorProvider,
    )

    class QueryExecutionStore(ConversationArchiveStatsStore, ConversationQueryRuntimeStore, TagStore, Protocol):
        """Repository surface needed by query execution and grouped stats helpers."""


_T = TypeVar("_T")


class ShowStatsCallback(Protocol):
    def __call__(self, env: AppEnv, *, verbose: bool = False) -> None: ...


@overload
async def _resolve_maybe_awaitable(value: Awaitable[_T]) -> _T: ...


@overload
async def _resolve_maybe_awaitable(value: _T) -> _T: ...


async def _resolve_maybe_awaitable(value: Awaitable[_T] | _T) -> _T:
    if inspect.isawaitable(value):
        return await value
    return value


def no_results(
    env: AppEnv,
    params: QueryParams,
    *,
    diagnostics: QueryMissDiagnostics | None = None,
    exit_code: int | None = 2,
) -> None:
    """Emit the canonical query no-results message."""
    emit_no_results(
        env,
        selection=coerce_query_spec(params),
        diagnostics=diagnostics,
        output_format=str(params.get("output_format") or "text"),
        exit_code=exit_code,
    )


def summary_to_dict(summary: ConversationSummary, message_count: int) -> JSONDocument:
    return ConversationListRowPayload.from_summary(
        summary,
        message_count=message_count,
    ).selected()


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


def _command_option_names(command: click.Command) -> frozenset[str]:
    options: set[str] = set()
    for param in command.params:
        if isinstance(param, click.Option):
            options.update(param.opts)
            options.update(param.secondary_opts)
    return frozenset(options)


def _find_root_option_after_verb(
    group: click.Group,
    verb: str,
    trailing_args: list[str],
) -> str | None:
    root_option_names = frozenset(_option_arity(group)) | _ROOT_GLOBAL_OPTIONS
    command = group.commands.get(verb)
    command_option_names = _command_option_names(command) if command is not None else frozenset()
    for token in trailing_args:
        if not token.startswith("-"):
            continue
        for option in root_option_names:
            if not _matches_option(option, token):
                continue
            if option in command_option_names:
                break
            return option
    return None


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
            misplaced_option = _find_root_option_after_verb(group, arg, list(args[index + 1 :]))
            if misplaced_option is not None:
                raise click.UsageError(
                    f"Query filters and root output flags must appear before the verb. Move {misplaced_option} before `{arg}`."
                )
            verb_args = option_args + [arg] + list(args[index + 1 :])
            return verb_args, tuple(query_terms), True
        query_terms.append(arg)
        index += 1

    return option_args, tuple(query_terms), False


def handle_query_mode(
    ctx: click.Context,
    *,
    show_stats: ShowStatsCallback,
) -> None:
    """Handle query mode: display stats or perform search."""
    env: AppEnv = ctx.obj
    request = RootModeRequest.from_context(ctx)

    if request.should_show_stats():
        show_stats(env, verbose=request.verbose)
        return

    execute_query_request(env, request)


class QueryFirstGroupBase(click.Group):
    """Custom Click group that routes to query mode by default."""

    def parse_args(self, ctx: click.Context, args: list[str]) -> list[str]:
        """Parse args, preserving raw query terms instead of rewriting them as hidden options."""
        parse_args, query_terms, has_subcommand = _split_query_mode_args(self, args)
        ctx.meta["polylogue_has_subcommand"] = has_subcommand
        if not has_subcommand:
            ctx.meta["polylogue_query_terms"] = query_terms
        return list(super().parse_args(ctx, parse_args))

    def invoke(self, ctx: click.Context) -> object:
        """Invoke the group, dispatching to query or stats mode if no subcommand."""
        if ctx.meta.get("polylogue_has_subcommand", False):
            return super().invoke(ctx)

        assert self.callback is not None, "QueryFirstGroup requires a callback"
        with ctx:
            ctx.invoke(self.callback, **ctx.params)

        self.handle_default_mode(ctx)
        return None

    def handle_default_mode(self, ctx: click.Context) -> None:
        """Dispatch no-subcommand mode for subclasses."""
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Query execution (original query.py)
# ---------------------------------------------------------------------------


def project_query_results(results: list[Conversation], plan: QueryExecutionPlan) -> list[Conversation]:
    """Apply post-selection transforms consistently before final output."""
    from polylogue.cli import query_actions as _query_actions

    projected = results
    if plan.output.transform is not None:
        projected = _query_actions.apply_transform(projected, plan.output.transform)
    message_roles = plan.output.effective_message_roles()
    if message_roles:
        projected = [conversation.with_roles(message_roles) for conversation in projected]
    return projected


def execute_query(env: AppEnv, params: QueryParams) -> None:
    """Execute a query-mode command."""
    execute_query_request(env, RootModeRequest.from_params(params))


def execute_query_request(env: AppEnv, request: RootModeRequest) -> None:
    """Execute a typed root-mode request."""
    run_coroutine_sync(async_execute_query_request(env, request))


def _create_query_vector_provider(config: Config) -> VectorProvider | None:
    """Best-effort vector provider setup for query execution."""
    from polylogue.storage.search_providers import create_vector_provider

    try:
        return create_vector_provider(config)
    except (ValueError, ImportError):
        return None
    except Exception as exc:
        logger.warning("Vector search setup failed: %s", exc)
        return None


def _stats_dimension(plan: QueryExecutionPlan) -> str:
    return plan.stats_dimension or "all"


async def _diagnose_query_miss(
    repo: QueryExecutionStore,
    selection: ConversationQuerySpec,
    *,
    config: Config,
) -> QueryMissDiagnostics:
    from polylogue.lib.query_miss_diagnostics import diagnose_query_miss

    return await diagnose_query_miss(repo, selection, config=config)


async def _semantic_stats_summaries(
    repo: QueryExecutionStore,
    filter_chain: ConversationFilter,
) -> list[ConversationSummary] | None:
    if filter_chain.can_use_summaries():
        return await filter_chain.list_summaries()

    query_plan = filter_chain.build_query_plan()
    if await _resolve_maybe_awaitable(query_plan.can_use_action_event_stats_with(repo)) is not True:
        return None
    return await repo.list_summaries_by_query(query_plan.record_query.with_limit(query_plan.limit))


async def _profile_stats_summaries(
    repo: QueryExecutionStore,
    filter_chain: ConversationFilter,
) -> list[ConversationSummary]:
    if filter_chain.can_use_summaries():
        return await filter_chain.list_summaries()

    query_plan = filter_chain.build_query_plan()
    return await repo.list_summaries_by_query(query_plan.record_query.with_limit(query_plan.limit))


async def async_execute_query(env: AppEnv, params: QueryParams) -> None:
    """Async compatibility wrapper for raw param execution."""
    await async_execute_query_request(env, RootModeRequest.from_params(params))


async def async_execute_query_request(env: AppEnv, request: RootModeRequest) -> None:
    """Async core of query execution over a typed request."""
    from polylogue.cli import query_actions as _query_actions
    from polylogue.cli import query_output as _query_output
    from polylogue.cli.helpers import fail, load_effective_config
    from polylogue.config import ConfigError

    params = request.query_params()

    try:
        config = load_effective_config(env)
    except ConfigError as exc:
        fail("query", str(exc))

    repo: QueryExecutionStore = env.repository

    vector_provider = _create_query_vector_provider(config)

    try:
        plan = build_query_execution_plan(params)
    except QueryPlanError as exc:
        click.echo(f"Error: {exc}", err=True)
        raise SystemExit(1) from exc

    if plan.selection.similar_text and vector_provider is None:
        click.echo(
            "Error: --similar requires vector search support. Configure VOYAGE_API_KEY and build embeddings with `polylogue run embed` first.",
            err=True,
        )
        raise SystemExit(1)
    if plan.selection.similar_text:
        archive_stats = await repo.get_archive_stats()
        if archive_stats.embedded_messages <= 0:
            click.echo(
                "Error: --similar requires existing embeddings. Run `polylogue run embed` first.",
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
            summary_diagnostics = await _diagnose_query_miss(repo, plan.selection, config=config)
            no_results(env, params, diagnostics=summary_diagnostics)
        await _query_output._output_summary_list(env, summary_results, plan.output, repo)
        return

    if route == QueryRoute.STREAM:
        full_id = await _query_actions.resolve_stream_target(repo, filter_chain, plan.selection)
        if plan.output.transform is not None:
            click.echo(
                "Warning: --transform is ignored in --stream mode (messages are streamed individually).",
                err=True,
            )
        if any(target.kind != "stdout" for target in plan.output.destinations):
            click.echo(
                f"Warning: --output {','.join(plan.output.destination_labels())} is ignored in --stream mode (output goes to stdout).",
                err=True,
            )

        message_limit_param = params.get("limit")
        message_limit = message_limit_param if isinstance(message_limit_param, int) else None
        await _query_output.stream_conversation(
            env,
            repo,
            full_id,
            output_format=plan.output.stream_format(),
            dialogue_only=plan.output.dialogue_only,
            message_roles=plan.output.effective_message_roles(),
            message_limit=message_limit,
        )
        return

    if route == QueryRoute.STATS_SQL:
        await _query_output.output_stats_sql(
            env,
            filter_chain,
            repo,
            selection=plan.selection,
            output_format=plan.output.output_format,
        )
        return

    if route == QueryRoute.SUMMARY_STATS:
        summaries = await filter_chain.list_summaries()
        msg_counts = await repo.get_message_counts_batch([str(summary.id) for summary in summaries])
        _query_output.output_stats_by_summaries(
            env,
            summaries,
            msg_counts,
            _stats_dimension(plan),
            selection=plan.selection,
            output_format=plan.output.output_format,
        )
        return

    if route == QueryRoute.STATS_BY and plan.stats_dimension in {"action", "tool"}:
        semantic_summaries = await _semantic_stats_summaries(repo, filter_chain)
        if semantic_summaries is not None:
            await _query_output.output_stats_by_semantic_summaries(
                env,
                semantic_summaries,
                repo,
                _stats_dimension(plan),
                selection=plan.selection,
                output_format=plan.output.output_format,
            )
            return

    if route == QueryRoute.STATS_BY and plan.stats_dimension in {"repo", "work-kind"}:
        summaries = await _profile_stats_summaries(repo, filter_chain)
        await _query_output.output_stats_by_profile_summaries(
            env,
            summaries,
            repo,
            _stats_dimension(plan),
            selection=plan.selection,
            output_format=plan.output.output_format,
        )
        return

    if route == QueryRoute.OPEN:
        if filter_chain.can_use_summaries():
            open_results: Sequence[Conversation | ConversationSummary] = await filter_chain.list_summaries()
        else:
            open_results = await filter_chain.list()
        open_diagnostics = await _diagnose_query_miss(repo, plan.selection, config=config) if not open_results else None
        _query_output._open_result(
            env,
            open_results,
            plan.output,
            selection=plan.selection,
            diagnostics=open_diagnostics,
        )
        return

    if route in {QueryRoute.MODIFY, QueryRoute.SUMMARY_MODIFY}:
        if route == QueryRoute.SUMMARY_MODIFY:
            matched_results: Sequence[Conversation | ConversationSummary] = await filter_chain.list_summaries()
        else:
            matched_results = await filter_chain.list()
        await _query_actions.apply_modifiers(env, matched_results, plan.mutation, repo)
        return

    if route in {QueryRoute.DELETE, QueryRoute.SUMMARY_DELETE}:
        if route == QueryRoute.SUMMARY_DELETE:
            matched_results = await filter_chain.list_summaries()
        else:
            matched_results = await filter_chain.list()
        await _query_actions.delete_conversations(env, matched_results, plan.mutation, repo)
        return

    if route == QueryRoute.STATS_BY:
        conversation_results = await filter_chain.list()
        projected_results = project_query_results(conversation_results, plan)
        _query_output._output_stats_by(
            env,
            projected_results,
            _stats_dimension(plan),
            selection=plan.selection,
            output_format=plan.output.output_format,
        )
        return

    conversation_results = await filter_chain.list()
    projected_results = project_query_results(conversation_results, plan)
    output_diagnostics = (
        await _diagnose_query_miss(repo, plan.selection, config=config) if not projected_results else None
    )
    _query_output.output_results(
        env,
        projected_results,
        plan.output,
        selection=plan.selection,
        diagnostics=output_diagnostics,
    )


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
    "execute_query_request",
    "handle_query_mode",
    "no_results",
    "resolve_query_route",
    "result_date",
    "result_id",
    "result_provider",
    "result_title",
    "summary_to_dict",
]
