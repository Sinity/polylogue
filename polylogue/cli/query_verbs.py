"""Positional verb subcommands for the query-first CLI.

Each verb takes the filter set from the parent context and executes
a specific action on the matched conversations.
"""

from __future__ import annotations

import click

from polylogue.cli.root_request import RootModeRequest
from polylogue.cli.shell_completion_values import complete_open_targets
from polylogue.cli.types import AppEnv

VERB_NAMES = frozenset({"list", "count", "stats", "open", "delete"})


@click.command("list")
@click.option(
    "--format",
    "-f",
    "output_format",
    type=click.Choice(["markdown", "json", "html", "obsidian", "org", "yaml", "plaintext", "csv"]),
    help="Output format",
)
@click.option("--fields", help="Fields: id, title, provider, date, messages, words, tags, summary")
@click.option("--limit", "-n", type=int, help="Max results")
@click.pass_context
def list_verb(ctx: click.Context, output_format: str | None, fields: str | None, limit: int | None) -> None:
    """List matched conversations."""
    request = _parent_request(ctx).with_param_updates(list_mode=True)
    if output_format:
        request = request.with_param_updates(output_format=output_format)
    if fields:
        request = request.with_param_updates(fields=fields)
    if limit is not None:
        request = request.with_param_updates(limit=limit)
    _execute_query_verb(ctx, request)


@click.command("count")
@click.pass_context
def count_verb(ctx: click.Context) -> None:
    """Print count of matched conversations."""
    _execute_query_verb(ctx, _parent_request(ctx).with_param_updates(count_only=True))


@click.command("stats")
@click.option(
    "--by",
    "stats_by",
    type=click.Choice(["provider", "month", "year", "day", "action", "tool", "repo", "work-kind"]),
    help="Aggregate by dimension",
)
@click.option(
    "--format",
    "-f",
    "output_format",
    type=click.Choice(["markdown", "json", "html", "obsidian", "org", "yaml", "plaintext", "csv"]),
    help="Output format",
)
@click.option("--limit", "-n", type=int, help="Max matched conversations before grouping")
@click.pass_context
def stats_verb(ctx: click.Context, stats_by: str | None, output_format: str | None, limit: int | None) -> None:
    """Show statistics for matched conversations."""
    request = _parent_request(ctx).with_param_updates(stats_only=stats_by is None)
    if stats_by:
        request = request.with_param_updates(stats_by=stats_by)
    if output_format:
        request = request.with_param_updates(output_format=output_format)
    if limit is not None:
        request = request.with_param_updates(limit=limit)
    _execute_query_verb(ctx, request)


@click.command("open")
@click.option("--print-path", is_flag=True, help="Print the matched render path instead of opening it")
@click.argument("target_terms", nargs=-1, shell_complete=complete_open_targets)
@click.pass_context
def open_verb(ctx: click.Context, print_path: bool, target_terms: tuple[str, ...]) -> None:
    """Open matched conversation in browser/editor."""
    request = _parent_request(ctx).with_param_updates(open_result=True, print_path=print_path)
    if not _parent_query_terms(ctx) and len(target_terms) == 1 and ":" in target_terms[0]:
        _execute_query_verb(ctx, request.with_param_updates(conv_id=target_terms[0]))
        return
    _execute_query_verb(ctx, request.append_query_terms(target_terms))


@click.command("delete")
@click.option("--dry-run", is_flag=True, help="Preview without deleting")
@click.option("--force", is_flag=True, help="Skip confirmation")
@click.pass_context
def delete_verb(ctx: click.Context, dry_run: bool, force: bool) -> None:
    """Delete matched conversations."""
    _execute_query_verb(
        ctx,
        _parent_request(ctx).with_param_updates(delete_matched=True, dry_run=dry_run, force=force),
    )


def _parent_query_terms(ctx: click.Context) -> tuple[str, ...]:
    """Load query terms captured on the parent query context."""
    raw_terms = _require_parent_context(ctx).meta.get("polylogue_query_terms", ())
    return tuple(str(term) for term in raw_terms)


def _parent_request(ctx: click.Context) -> RootModeRequest:
    """Build the typed request from the parent query context."""
    return RootModeRequest.from_context(_require_parent_context(ctx))


def _require_parent_context(ctx: click.Context) -> click.Context:
    """Require that the verb runs underneath the query command context."""
    parent = ctx.parent
    if parent is None:
        raise click.UsageError("Query verbs must be invoked from the query command context.")
    return parent


def _execute_query_verb(
    ctx: click.Context,
    request: RootModeRequest,
) -> None:
    """Execute query with verb-modified params."""
    from polylogue.cli.query import execute_query_request

    env: AppEnv = ctx.obj
    execute_query_request(env, request)


QUERY_VERBS = (list_verb, count_verb, stats_verb, open_verb, delete_verb)


__all__ = [
    "QUERY_VERBS",
    "VERB_NAMES",
    "count_verb",
    "delete_verb",
    "list_verb",
    "open_verb",
    "stats_verb",
]
