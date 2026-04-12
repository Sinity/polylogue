"""Positional verb subcommands for the query-first CLI.

Each verb takes the filter set from the parent context and executes
a specific action on the matched conversations.
"""

from __future__ import annotations

from typing import Any

import click

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
    params = _parent_params(ctx)
    params["list_mode"] = True
    if output_format:
        params["output_format"] = output_format
    if fields:
        params["fields"] = fields
    if limit is not None:
        params["limit"] = limit
    _execute_query_verb(ctx, params)


@click.command("count")
@click.pass_context
def count_verb(ctx: click.Context) -> None:
    """Print count of matched conversations."""
    params = _parent_params(ctx)
    params["count_only"] = True
    _execute_query_verb(ctx, params)


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
@click.pass_context
def stats_verb(ctx: click.Context, stats_by: str | None, output_format: str | None) -> None:
    """Show statistics for matched conversations."""
    params = _parent_params(ctx)
    params["stats_only"] = stats_by is None
    if stats_by:
        params["stats_by"] = stats_by
    if output_format:
        params["output_format"] = output_format
    _execute_query_verb(ctx, params)


@click.command("open")
@click.option("--print-path", is_flag=True, help="Print the matched render path instead of opening it")
@click.argument("target_terms", nargs=-1)
@click.pass_context
def open_verb(ctx: click.Context, print_path: bool, target_terms: tuple[str, ...]) -> None:
    """Open matched conversation in browser/editor."""
    params = _parent_params(ctx)
    params["open_result"] = True
    params["print_path"] = print_path
    if not ctx.parent.meta.get("polylogue_query_terms") and len(target_terms) == 1 and ":" in target_terms[0]:
        params["conv_id"] = target_terms[0]
        _execute_query_verb(ctx, params)
        return
    _execute_query_verb(ctx, params, extra_query_terms=target_terms)


@click.command("delete")
@click.option("--dry-run", is_flag=True, help="Preview without deleting")
@click.option("--force", is_flag=True, help="Skip confirmation")
@click.pass_context
def delete_verb(ctx: click.Context, dry_run: bool, force: bool) -> None:
    """Delete matched conversations."""
    params = _parent_params(ctx)
    params["delete_matched"] = True
    params["dry_run"] = dry_run
    params["force"] = force
    _execute_query_verb(ctx, params)


def _parent_params(ctx: click.Context) -> dict[str, Any]:
    """Extract params from parent context."""
    return dict(ctx.parent.params)


def _execute_query_verb(
    ctx: click.Context,
    params: dict[str, Any],
    *,
    extra_query_terms: tuple[str, ...] = (),
) -> None:
    """Execute query with verb-modified params."""
    from polylogue.cli.query import execute_query

    env: AppEnv = ctx.obj
    query_terms = tuple(ctx.parent.meta.get("polylogue_query_terms", ()))
    params["query"] = query_terms + tuple(extra_query_terms)
    execute_query(env, params)


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
