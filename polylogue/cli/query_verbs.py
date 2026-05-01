"""Positional verb subcommands for the query-first CLI.

Each verb takes the filter set from the parent context and executes
a specific action on the matched conversations.
"""

from __future__ import annotations

from typing import cast

import click

from polylogue.cli.root_request import RootModeRequest
from polylogue.cli.shared.types import AppEnv
from polylogue.cli.shell_completion_values import complete_open_targets

VERB_NAMES = frozenset({"list", "count", "stats", "open", "show", "bulk-export", "delete", "messages", "raw"})

_BULK_EXPORT_FORMATS = ("jsonl", "json", "markdown", "yaml", "plaintext", "html", "obsidian", "org")


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


@click.command("show")
@click.argument("target_terms", nargs=-1)
@click.pass_context
def show_verb(ctx: click.Context, target_terms: tuple[str, ...]) -> None:
    """Show matched conversations with default full-content output."""
    request = _parent_request(ctx)
    parent_terms = _parent_query_terms(ctx)
    candidates = parent_terms + target_terms
    if len(candidates) == 1 and ":" in candidates[0]:
        _execute_query_verb(ctx, request.with_query_terms(()).with_param_updates(conv_id=candidates[0]))
        return
    _execute_query_verb(ctx, request.append_query_terms(target_terms))


@click.command("open")
@click.option("--print-path", is_flag=True, help="Print the matched render path instead of opening it")
@click.argument("target_terms", nargs=-1, shell_complete=complete_open_targets)
@click.pass_context
def open_verb(ctx: click.Context, print_path: bool, target_terms: tuple[str, ...]) -> None:
    """Open matched conversation in browser/editor."""
    request = _parent_request(ctx).with_param_updates(open_result=True, print_path=print_path)
    parent_terms = _parent_query_terms(ctx)
    candidates = parent_terms + target_terms
    if len(candidates) == 1 and ":" in candidates[0]:
        _execute_query_verb(ctx, request.with_query_terms(()).with_param_updates(conv_id=candidates[0]))
        return
    _execute_query_verb(ctx, request.append_query_terms(target_terms))


@click.command("bulk-export")
@click.option(
    "--format",
    "-f",
    "output_format",
    type=click.Choice(_BULK_EXPORT_FORMATS),
    default="jsonl",
    show_default=True,
    help="Output format. jsonl emits one conversation JSON per line.",
)
@click.option("--fields", help="Fields for JSON/YAML outputs")
@click.pass_context
def bulk_export_verb(ctx: click.Context, output_format: str, fields: str | None) -> None:
    """Bulk export every matched conversation in one process.

    Reuses the parent filter chain (``--provider``, ``--since``, ``--referenced-path``,
    ``--message-role``, etc.). Default ``--format jsonl`` emits one
    single-line conversation JSON per line, suitable for piping into ``jq``
    or downstream analysis tools. Other formats are concatenated with
    ``\\n---\\n`` separators where appropriate.
    """
    from polylogue.cli.bulk_export import run_bulk_export

    request = _parent_request(ctx)
    env: AppEnv = ctx.obj
    run_bulk_export(env, request, output_format=output_format, fields=fields)


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


@click.command("messages")
@click.argument("conversation_id", required=False)
@click.option("--message-role", "-r", "message_role", multiple=True, help="Filter by message role")
@click.option(
    "--message-type",
    "message_type",
    type=click.Choice(["summary", "tool_use", "tool_result", "thinking"]),
    help="Filter by message content type",
)
@click.option("--limit", "-n", type=int, default=50, help="Max messages to return")
@click.option("--offset", type=int, default=0, help="Offset for pagination")
@click.option("--no-code-blocks", is_flag=True, help="Exclude code blocks")
@click.option("--no-tool-calls", is_flag=True, help="Exclude tool calls")
@click.option("--no-tool-outputs", is_flag=True, help="Exclude tool outputs")
@click.option("--no-file-reads", is_flag=True, help="Exclude file reads")
@click.option("--prose-only", is_flag=True, help="Show only prose text")
@click.option(
    "--format", "-f", "output_format", type=click.Choice(["markdown", "json", "plaintext"]), help="Output format"
)
@click.pass_context
def messages_verb(
    ctx: click.Context,
    conversation_id: str | None,
    message_role: tuple[str, ...],
    message_type: str | None,
    limit: int,
    offset: int,
    no_code_blocks: bool,
    no_tool_calls: bool,
    no_tool_outputs: bool,
    no_file_reads: bool,
    prose_only: bool,
    output_format: str | None,
) -> None:
    """Show paginated messages for a conversation."""
    from polylogue.cli.messages import run_messages

    env: AppEnv = ctx.obj
    if conversation_id is None:
        request = _parent_request(ctx)
        params = dict(request.params)
        conversation_id = cast("str | None", params.get("conv_id"))
        if not conversation_id:
            raise click.UsageError("messages requires a conversation ID (use --id or pass as argument)")
    else:
        request = _parent_request(ctx)

    run_messages(
        env,
        request,
        conversation_id=str(conversation_id),
        message_role=message_role,
        message_type=message_type,
        limit=limit,
        offset=offset,
        no_code_blocks=no_code_blocks,
        no_tool_calls=no_tool_calls,
        no_tool_outputs=no_tool_outputs,
        no_file_reads=no_file_reads,
        prose_only=prose_only,
        output_format=output_format,
    )


@click.command("raw")
@click.argument("conversation_id", required=False)
@click.option("--limit", "-n", type=int, default=50, help="Max raw artifacts to return")
@click.option("--offset", type=int, default=0, help="Offset for pagination")
@click.option(
    "--format", "-f", "output_format", type=click.Choice(["json", "yaml"]), default="json", help="Output format"
)
@click.pass_context
def raw_verb(
    ctx: click.Context,
    conversation_id: str | None,
    limit: int,
    offset: int,
    output_format: str,
) -> None:
    """Show raw archive artifacts for a conversation."""
    from polylogue.cli.messages import run_raw

    env: AppEnv = ctx.obj
    if conversation_id is None:
        request = _parent_request(ctx)
        params = dict(request.params)
        conversation_id = cast("str | None", params.get("conv_id"))
        if not conversation_id:
            raise click.UsageError("raw requires a conversation ID (use --id or pass as argument)")
    else:
        request = _parent_request(ctx)

    run_raw(
        env,
        request,
        conversation_id=str(conversation_id),
        limit=limit,
        offset=offset,
        output_format=output_format,
    )


QUERY_VERBS = (
    list_verb,
    count_verb,
    stats_verb,
    open_verb,
    show_verb,
    bulk_export_verb,
    delete_verb,
    messages_verb,
    raw_verb,
)


__all__ = [
    "QUERY_VERBS",
    "VERB_NAMES",
    "bulk_export_verb",
    "count_verb",
    "delete_verb",
    "list_verb",
    "open_verb",
    "show_verb",
    "stats_verb",
]
