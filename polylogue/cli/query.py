"""Query execution entrypoint for the query-first CLI."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import click

from polylogue.cli import query_actions as _query_actions
from polylogue.cli import query_output as _query_output
from polylogue.cli.query_plan import QueryAction, QueryPlanError, build_query_execution_plan
from polylogue.lib.log import get_logger
from polylogue.lib.query_spec import ConversationQuerySpec, QuerySpecError

logger = get_logger(__name__)

_apply_modifiers = _query_actions._apply_modifiers
_apply_transform = _query_actions._apply_transform
_delete_conversations = _query_actions._delete_conversations
resolve_stream_target = _query_actions.resolve_stream_target

_conv_to_csv = _query_output._conv_to_csv
_copy_to_clipboard = _query_output._copy_to_clipboard
_format_list = _query_output._format_list
_open_in_browser = _query_output._open_in_browser
_open_result = _query_output._open_result
_output_results = _query_output._output_results
_output_stats_by = _query_output._output_stats_by
_output_stats_by_summaries = _query_output._output_stats_by_summaries
_output_stats_sql = _query_output._output_stats_sql
_output_summary_list = _query_output._output_summary_list
_render_conversation_rich = _query_output._render_conversation_rich
_send_output = _query_output._send_output
_write_message_streaming = _query_output._write_message_streaming
stream_conversation = _query_output.stream_conversation

if TYPE_CHECKING:
    from polylogue.cli.types import AppEnv


def _coerce_query_spec(params: dict[str, Any] | ConversationQuerySpec) -> ConversationQuerySpec:
    if isinstance(params, ConversationQuerySpec):
        return params
    return ConversationQuerySpec.from_params(params)


def _describe_filters(params: dict[str, Any] | ConversationQuerySpec) -> list[str]:
    """Build a human-readable list of active filters from params or spec."""
    return _coerce_query_spec(params).describe()


def _no_results(env: AppEnv, params: dict[str, Any] | ConversationQuerySpec, *, exit_code: int = 2) -> None:
    """Print a helpful no-results message and exit."""
    filters = _describe_filters(params)
    if filters:
        click.echo("No conversations matched filters:", err=True)
        for item in filters:
            click.echo(f"  {item}", err=True)
        click.echo("Hint: try broadening your filters or use --list to browse", err=True)
    else:
        click.echo("No conversations matched.", err=True)
    raise SystemExit(exit_code)


def execute_query(env: AppEnv, params: dict[str, Any]) -> None:
    """Execute a query-mode command."""
    import asyncio

    asyncio.run(_async_execute_query(env, params))


async def _async_execute_query(env: AppEnv, params: dict[str, Any]) -> None:
    """Async core of execute_query."""
    from polylogue.cli.helpers import fail, load_effective_config
    from polylogue.config import ConfigError
    from polylogue.storage.search_providers import create_vector_provider

    try:
        config = load_effective_config(env)
    except ConfigError as exc:
        fail("query", str(exc))

    repo = env.repository

    vector_provider = None
    try:
        vector_provider = create_vector_provider(config)
    except (ValueError, ImportError):
        pass
    except Exception as exc:
        logger.warning("Vector search setup failed: %s", exc)

    try:
        plan = build_query_execution_plan(params)
    except QueryPlanError as exc:
        click.echo(f"Error: {exc}", err=True)
        raise SystemExit(1) from exc

    try:
        filter_chain = plan.selection.build_filter(
            repo,
            vector_provider=vector_provider,
        )
    except QuerySpecError as exc:
        click.echo(f"Error: Cannot parse date: '{exc.value}'", err=True)
        click.echo(
            "Hint: use ISO format (2025-01-15), relative ('yesterday', 'last week'), or month (2025-01)",
            err=True,
        )
        raise SystemExit(1) from exc

    if plan.action == QueryAction.COUNT:
        click.echo(await filter_chain.count())
        return

    if plan.prefers_summary_list() and filter_chain.can_use_summaries():
        summary_results = await filter_chain.list_summaries()
        if not summary_results:
            _no_results(env, plan.selection)
        await _output_summary_list(env, summary_results, params, repo)
        return

    if plan.action == QueryAction.STREAM:
        full_id = await resolve_stream_target(repo, filter_chain, plan.selection)
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

        await stream_conversation(
            env,
            repo,
            full_id,
            output_format=plan.output.stream_format(),
            dialogue_only=plan.output.dialogue_only,
            message_limit=params.get("limit"),
        )
        return

    if plan.action == QueryAction.STATS:
        await _output_stats_sql(env, filter_chain, repo)
        return

    if plan.prefers_summary_stats() and filter_chain.can_use_summaries():
        summaries = await filter_chain.list_summaries()
        msg_counts = await repo.get_message_counts_batch([str(summary.id) for summary in summaries])
        _output_stats_by_summaries(env, summaries, msg_counts, plan.stats_dimension or "all")
        return

    if plan.prefers_summary_mutation() and filter_chain.can_use_summaries():
        results = await filter_chain.list_summaries()
    else:
        results = await filter_chain.list()

    if plan.action == QueryAction.MODIFY:
        await _apply_modifiers(env, results, params, repo)
        return

    if plan.action == QueryAction.DELETE:
        await _delete_conversations(env, results, params, repo)
        return

    if plan.output.transform is not None:
        results = _apply_transform(results, plan.output.transform)

    if plan.output.dialogue_only:
        results = [conversation.dialogue_only() for conversation in results]

    if plan.action == QueryAction.STATS_BY:
        _output_stats_by(env, results, plan.stats_dimension or "all")
        return

    if plan.action == QueryAction.OPEN:
        _open_result(env, results, params)
        return

    _output_results(env, results, params)
