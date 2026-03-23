"""Query execution entrypoint for the query-first CLI."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import click

from polylogue.cli import query_actions as _query_actions
from polylogue.cli import query_output as _query_output
from polylogue.cli.query_helpers import no_results
from polylogue.cli.query_plan import (
    QueryPlanError,
    QueryRoute,
    build_query_execution_plan,
    resolve_query_route,
)
from polylogue.lib.query_spec import QuerySpecError
from polylogue.logging import get_logger
from polylogue.sync_bridge import run_coroutine_sync

logger = get_logger(__name__)

if TYPE_CHECKING:
    from polylogue.cli.query_plan import QueryExecutionPlan
    from polylogue.cli.types import AppEnv


def project_query_results(results: list[Any], plan: QueryExecutionPlan) -> list[Any]:
    """Apply post-selection transforms consistently before final output."""
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


async def async_execute_query(env: AppEnv, params: dict[str, Any]) -> None:
    """Async core of execute_query."""
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
        await _query_output.output_stats_sql(env, filter_chain, repo)
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

    if route == QueryRoute.STATS_BY and plan.stats_dimension in {"action", "tool"} and filter_chain.can_use_summaries():
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
