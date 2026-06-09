"""Positional verb subcommands for the query-first CLI.

Each verb takes the filter set from the parent context and executes
a specific action on the matched sessions.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, cast

import click

if TYPE_CHECKING:
    from polylogue.cli.root_request import RootModeRequest

from polylogue.cli.click_option_groups import _LazyChoice
from polylogue.cli.shared.types import AppEnv
from polylogue.cli.verb_names import VERB_NAMES


# Deferred imports: RootModeRequest triggers the archive.query.spec →
# operations.archive chain (~780 ms).  Only import it when a verb
# actually executes, never during --help.
def _get_root_request_class() -> object:  # pragma: no cover — returns RootModeRequest
    from polylogue.cli.root_request import RootModeRequest

    return RootModeRequest


def _get_message_type_class() -> object:  # pragma: no cover — returns MessageType
    from polylogue.archive.message.types import MessageType

    return MessageType


def _get_message_type_choices() -> list[str]:
    return [m.value for m in _get_message_type_class()]  # type: ignore[attr-defined]


def _lazy_open_targets() -> Callable[..., object]:
    def _complete(ctx: click.Context, param: click.Parameter, incomplete: str):  # type: ignore[no-untyped-def]
        from polylogue.cli.shell_completion_values import complete_open_targets

        return complete_open_targets(ctx, param, incomplete)

    return _complete


def _lazy_shell_complete(source: str) -> Callable[..., object]:
    def _complete(ctx: click.Context, param: click.Parameter, incomplete: str):  # type: ignore[no-untyped-def]
        from polylogue.cli.shell_completion_values import complete_query_source

        return complete_query_source(source)(ctx, param, incomplete)  # type: ignore[arg-type]

    _complete.__name__ = f"complete_{source}"
    return _complete


_BULK_EXPORT_FORMATS = ("jsonl", "json", "markdown", "yaml", "plaintext", "html", "obsidian", "org")
_complete_session_id = _lazy_shell_complete("session_id")
_complete_message_type = _lazy_shell_complete("message_type")


@click.command("list")
@click.option(
    "--format",
    "-f",
    "output_format",
    type=click.Choice(["markdown", "json", "ndjson", "html", "obsidian", "org", "yaml", "plaintext", "csv"]),
    help="Output format (ndjson = one JSON document per line, streaming-friendly)",
)
@click.option("--fields", help="Fields: id, title, origin, date, messages, words, tags, summary")
@click.option("--limit", "-l", "-n", type=int, help="Max results")
@click.pass_context
def list_verb(ctx: click.Context, output_format: str | None, fields: str | None, limit: int | None) -> None:
    """List matched sessions."""
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
    """Print count of matched sessions."""
    _execute_query_verb(ctx, _parent_request(ctx).with_param_updates(count_only=True))


@click.command("stats")
@click.option(
    "--by",
    "stats_by",
    type=click.Choice(["origin", "month", "year", "day", "action", "tool", "repo", "work-kind"]),
    help="Aggregate by dimension",
)
@click.option(
    "--format",
    "-f",
    "output_format",
    type=click.Choice(["markdown", "json", "ndjson", "html", "obsidian", "org", "yaml", "plaintext", "csv"]),
    help="Output format (ndjson = one JSON document per row, streaming-friendly)",
)
@click.option("--limit", "-l", "-n", type=int, help="Max matched sessions before grouping")
@click.pass_context
def stats_verb(ctx: click.Context, stats_by: str | None, output_format: str | None, limit: int | None) -> None:
    """Show statistics for matched sessions."""
    request = _parent_request(ctx).with_param_updates(stats_only=stats_by is None)
    if stats_by:
        request = request.with_param_updates(stats_by=stats_by)
    if output_format:
        request = request.with_param_updates(output_format=output_format)
    if limit is not None:
        request = request.with_param_updates(limit=limit)
    _execute_query_verb(ctx, request)


@click.command("recent")
@click.option("--limit", "-n", type=int, default=10, show_default=True, help="Maximum sessions to list.")
@click.option("--origin", "origin_filter", default=None, help="Filter by origin.")
@click.option("--format", "-f", "output_format", default=None, help="Output format: json, ndjson, yaml, csv.")
@click.pass_context
def recent_verb(
    ctx: click.Context,
    limit: int,
    origin_filter: str | None,
    output_format: str | None,
) -> None:
    """List the most recently updated sessions."""
    request = _parent_request(ctx).with_param_updates(
        list_mode=True,
        sort="updated_at",
        reverse=True,
        limit=limit,
    )
    if origin_filter:
        request = request.with_param_updates(origin=origin_filter)
    if output_format:
        request = request.with_param_updates(output_format=output_format)
    _execute_query_verb(ctx, request)


@click.command("show")
@click.argument("target_terms", nargs=-1)
@click.pass_context
def show_verb(ctx: click.Context, target_terms: tuple[str, ...]) -> None:
    """Show matched sessions with default full-content output."""
    request = _parent_request(ctx)
    parent_terms = _parent_query_terms(ctx)
    candidates = parent_terms + target_terms
    if len(candidates) == 1 and ":" in candidates[0]:
        _execute_query_verb(ctx, request.with_query_terms(()).with_param_updates(conv_id=candidates[0]))
        return
    _execute_query_verb(ctx, request.append_query_terms(target_terms))


@click.command("open")
@click.option("--print-url", is_flag=True, help="Print the matched daemon web URL instead of opening it")
@click.argument("target_terms", nargs=-1, shell_complete=_lazy_open_targets())
@click.pass_context
def open_verb(ctx: click.Context, print_url: bool, target_terms: tuple[str, ...]) -> None:
    """Open matched session in the daemon web reader."""
    request = _parent_request(ctx).with_param_updates(open_result=True, print_url=print_url)
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
    help="Output format. jsonl emits one session JSON per line.",
)
@click.option("--fields", help="Fields for JSON/YAML outputs")
@click.pass_context
def bulk_export_verb(ctx: click.Context, output_format: str, fields: str | None) -> None:
    """Bulk export every matched session in one process.

    Reuses the parent filter chain (``--origin``, ``--since``, ``--referenced-path``,
    ``--message-role``, etc.). Default ``--format jsonl`` emits one
    single-line session JSON per line, suitable for piping into ``jq``
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
    """Delete matched sessions."""
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
    return _get_root_request_class().from_context(_require_parent_context(ctx))  # type: ignore[no-any-return,attr-defined]


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


def _resolve_target_session_id(request: RootModeRequest) -> str | None:
    """Verb-tree adapter for the shared latest-resolver helper (#1626, #1642)."""
    from polylogue.cli.shared.latest_resolver import resolve_session_id_from_root_params

    return resolve_session_id_from_root_params(dict(request.params))


@click.command("messages")
@click.argument("session_id", required=False, shell_complete=_complete_session_id)
@click.option("--message-role", "-r", "message_role", multiple=True, help="Filter by message role")
@click.option(
    "--message-type",
    "message_type",
    type=_LazyChoice(_get_message_type_choices, "type"),
    shell_complete=_complete_message_type,
    help="Filter by message content type",
)
@click.option("--limit", "-l", "-n", type=int, default=50, help="Max messages to return")
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
    session_id: str | None,
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
    """Show paginated messages for a session."""
    from polylogue.cli.messages import run_messages

    env: AppEnv = ctx.obj
    request = _parent_request(ctx)
    if session_id is None:
        session_id = _resolve_target_session_id(request)
        if not session_id:
            raise click.UsageError("messages requires a session ID (use --id or pass as argument)")

    run_messages(
        env,
        request,
        session_id=str(session_id),
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
@click.argument("session_id", required=False, shell_complete=_complete_session_id)
@click.option("--limit", "-l", "-n", type=int, default=50, help="Max raw artifacts to return")
@click.option("--offset", type=int, default=0, help="Offset for pagination")
@click.option(
    "--format", "-f", "output_format", type=click.Choice(["json", "yaml"]), default="json", help="Output format"
)
@click.pass_context
def raw_verb(
    ctx: click.Context,
    session_id: str | None,
    limit: int,
    offset: int,
    output_format: str,
) -> None:
    """Show raw archive artifacts for a session."""
    from polylogue.cli.messages import run_raw

    env: AppEnv = ctx.obj
    request = _parent_request(ctx)
    if session_id is None:
        session_id = _resolve_target_session_id(request)
        if not session_id:
            raise click.UsageError("raw requires a session ID (use --id or pass as argument)")

    run_raw(
        env,
        request,
        session_id=str(session_id),
        limit=limit,
        offset=offset,
        output_format=output_format,
    )


@click.command("select")
@click.argument(
    "selector_kind",
    required=False,
    default="session",
    type=click.Choice(["session"]),
)
@click.option("--print", "print_field", type=click.Choice(["id", "title", "origin", "json"]), default="id")
@click.option("--limit", "-l", "-n", type=int, default=50, help="Max candidates to offer")
@click.pass_context
def select_verb(ctx: click.Context, selector_kind: str, print_field: str, limit: int) -> None:
    """Select one matched session and print a field."""
    from polylogue.cli.select import SelectPrintField, run_select

    del selector_kind
    env: AppEnv = ctx.obj
    run_select(
        env,
        _parent_request(ctx),
        limit=limit,
        print_field=cast(SelectPrintField, print_field),
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
    select_verb,
)


__all__ = [
    "QUERY_VERBS",
    "VERB_NAMES",
    "bulk_export_verb",
    "count_verb",
    "delete_verb",
    "list_verb",
    "open_verb",
    "select_verb",
    "show_verb",
    "stats_verb",
]
