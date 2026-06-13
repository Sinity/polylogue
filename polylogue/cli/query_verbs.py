"""Positional verb subcommands for the query-first CLI.

Each verb takes the filter set from the parent context and executes
a specific action on the matched sessions.
"""

from __future__ import annotations

import io
from typing import TYPE_CHECKING

import click

if TYPE_CHECKING:
    from polylogue.archive.session.domain_models import Session, SessionSummary
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


def _lazy_shell_complete(source: str):  # type: ignore[no-untyped-def]
    def _complete(ctx: click.Context, param: click.Parameter, incomplete: str):  # type: ignore[no-untyped-def]
        from polylogue.cli.shell_completion_values import complete_query_source

        return complete_query_source(source)(ctx, param, incomplete)  # type: ignore[arg-type]

    _complete.__name__ = f"complete_{source}"
    return _complete


_complete_session_id = _lazy_shell_complete("session_id")
_complete_message_type = _lazy_shell_complete("message_type")

_READ_VIEWS = ("summary", "conversation", "messages", "raw", "context")
_READ_DESTINATIONS = ("terminal", "stdout", "browser", "clipboard", "file")
_READ_FORMATS = ("text", "markdown", "json", "ndjson", "yaml", "html", "obsidian", "org", "csv")


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


@click.command("read")
@click.option(
    "--view",
    "-v",
    type=click.Choice(_READ_VIEWS),
    default="summary",
    show_default=True,
    help="What to render (summary, conversation, messages, raw, context).",
)
@click.option(
    "--to",
    "destination",
    type=click.Choice(_READ_DESTINATIONS),
    default="terminal",
    show_default=True,
    help="Output destination.",
)
@click.option(
    "--format",
    "-f",
    "output_format",
    type=click.Choice(_READ_FORMATS),
    default=None,
    help="Output format (where applicable).",
)
@click.option("--out", "out_path", type=click.Path(), default=None, help="File path for --to file.")
@click.option("--all", "export_all", is_flag=True, help="Apply to all matched sessions (bulk export).")
# message/raw pagination flags
@click.option("--message-role", "-r", "message_role", multiple=True, help="Filter by message role (--view messages).")
@click.option(
    "--message-type",
    "message_type",
    type=_LazyChoice(_get_message_type_choices, "type"),
    shell_complete=_complete_message_type,
    help="Filter by message content type (--view messages).",
)
@click.option("--limit", "-l", "-n", type=int, default=None, help="Max items to return.")
@click.option("--offset", type=int, default=0, help="Pagination offset.")
@click.option("--no-code-blocks", is_flag=True, help="Exclude code blocks (--view messages).")
@click.option("--no-tool-calls", is_flag=True, help="Exclude tool calls (--view messages).")
@click.option("--no-tool-outputs", is_flag=True, help="Exclude tool outputs (--view messages).")
@click.option("--no-file-reads", is_flag=True, help="Exclude file reads (--view messages).")
@click.option("--prose-only", is_flag=True, help="Show only prose text (--view messages).")
@click.option("--fields", help="Fields for JSON/YAML outputs (--all).")
@click.pass_context
def read_verb(
    ctx: click.Context,
    view: str,
    destination: str,
    output_format: str | None,
    out_path: str | None,
    export_all: bool,
    message_role: tuple[str, ...],
    message_type: str | None,
    limit: int | None,
    offset: int,
    no_code_blocks: bool,
    no_tool_calls: bool,
    no_tool_outputs: bool,
    no_file_reads: bool,
    prose_only: bool,
    fields: str | None,
) -> None:
    """Read matched sessions.

    \b
    Routes to the appropriate renderer based on --view and delivers the
    output to --to (terminal, stdout, browser, clipboard, or file).

    \b
    Examples:
        polylogue --id abc123 read
        polylogue find id:abc then read --view messages
        polylogue find id:abc then read --view raw --format json
        polylogue find id:abc then read --to browser
        polylogue find 'repo:polylogue has:paste' then read --all --format ndjson
        polylogue find 'archive runtime' then read --view context

    \b
    Deferred views (not yet implemented; note in PR body):
        timeline, tools, files, metadata, continuation
    """
    env: AppEnv = ctx.obj
    request = _parent_request(ctx)

    # Bulk-export mode: applies to all matched sessions.
    if export_all:
        _run_read_bulk(
            env, request, output_format=output_format, fields=fields, destination=destination, out_path=out_path
        )
        return

    # Single-session modes: resolve session ID from request.
    session_id = _resolve_target_session_id(request)
    if session_id is None and view in ("messages", "raw"):
        raise click.UsageError(f"read --view {view} requires a session ID (use --id, id:prefix, or --latest).")

    if view == "messages":
        assert session_id is not None
        _run_read_messages(
            env,
            request,
            session_id=session_id,
            message_role=message_role,
            message_type=message_type,
            limit=limit if limit is not None else 50,
            offset=offset,
            no_code_blocks=no_code_blocks,
            no_tool_calls=no_tool_calls,
            no_tool_outputs=no_tool_outputs,
            no_file_reads=no_file_reads,
            prose_only=prose_only,
            output_format=output_format,
            destination=destination,
            out_path=out_path,
        )
        return

    if view == "raw":
        assert session_id is not None
        _run_read_raw(
            env,
            request,
            session_id=session_id,
            limit=limit if limit is not None else 50,
            offset=offset,
            output_format=output_format or "json",
            destination=destination,
            out_path=out_path,
        )
        return

    if view == "context":
        _run_read_context(env, request, destination=destination, out_path=out_path)
        return

    # summary / conversation: standard show/query path with destination routing.
    if destination == "browser":
        _run_read_browser(env, request, output_format=output_format)
        return

    fmt = output_format or "markdown"
    updated = request.with_param_updates(output_format=fmt)
    if destination in ("stdout", "terminal"):
        _execute_query_verb(ctx, updated)
    elif destination == "clipboard":
        _execute_query_verb(ctx, updated.with_param_updates(output="clipboard"))
    elif destination == "file":
        if not out_path:
            raise click.UsageError("--to file requires --out <path>.")
        _execute_query_verb(ctx, updated.with_param_updates(output=out_path))
    else:
        _execute_query_verb(ctx, updated)


def _run_read_browser(env: AppEnv, request: RootModeRequest, *, output_format: str | None) -> None:
    """Open the first matched session in the daemon web reader using /s/{id} URL."""
    import webbrowser
    from urllib.parse import quote

    from polylogue.api.sync.bridge import run_coroutine_sync
    from polylogue.cli.query import _create_query_vector_provider
    from polylogue.cli.query_contracts import build_query_execution_plan
    from polylogue.paths import archive_file_set_root_for_paths

    config = env.config

    async def _find_first() -> str | None:
        plan = build_query_execution_plan(request.query_params())
        archive_root = archive_file_set_root_for_paths(
            archive_root_path=config.archive_root,
            db_anchor=config.db_path,
        )
        vector_provider = _create_query_vector_provider(config, db_path=archive_root / "embeddings.db")
        filter_chain = plan.selection.build_filter(config, vector_provider=vector_provider)
        first_id: str | None = None
        if filter_chain.can_use_summaries():
            summaries: list[SessionSummary] = list(await filter_chain.list_summaries())
            if summaries:
                first_id = str(summaries[0].id)
        else:
            sessions: list[Session] = list(await filter_chain.list())
            if sessions:
                first_id = str(sessions[0].id)
        return first_id

    session_id = run_coroutine_sync(_find_first())
    if session_id is None:
        env.ui.error("No sessions matched.")
        return

    daemon_url = str(getattr(env, "daemon_url", None) or "http://127.0.0.1:8766").rstrip("/")
    web_url = f"{daemon_url}/s/{quote(session_id, safe='')}"
    webbrowser.open(web_url)
    env.ui.console.print(f"Opened: {web_url}")


def _run_read_messages(
    env: AppEnv,
    request: RootModeRequest,
    *,
    session_id: str,
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
    destination: str,
    out_path: str | None,
) -> None:
    """Route messages view to messages renderer with destination handling."""
    from polylogue.cli.messages import run_messages

    # For file/clipboard destinations, capture output via click echo interception then deliver.
    if destination in ("file", "clipboard"):
        buf = io.StringIO()

        def _captured_echo(message: object = None, **_kwargs: object) -> None:
            buf.write(str(message or "") + "\n")

        _orig_echo = click.echo
        click.echo = _captured_echo  # type: ignore[assignment]
        try:
            run_messages(
                env,
                request,
                session_id=session_id,
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
        finally:
            click.echo = _orig_echo
        _deliver_content(env, buf.getvalue(), destination=destination, out_path=out_path)
        return

    run_messages(
        env,
        request,
        session_id=session_id,
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


def _run_read_raw(
    env: AppEnv,
    request: RootModeRequest,
    *,
    session_id: str,
    limit: int,
    offset: int,
    output_format: str,
    destination: str,
    out_path: str | None,
) -> None:
    """Route raw view to raw renderer with destination handling."""
    from polylogue.cli.messages import run_raw

    if destination in ("file", "clipboard", "stdout"):
        buf = io.StringIO()

        def _captured_echo_raw(message: object = None, **_kwargs: object) -> None:
            buf.write(str(message or "") + "\n")

        _orig_echo = click.echo
        click.echo = _captured_echo_raw  # type: ignore[assignment]
        try:
            run_raw(env, request, session_id=session_id, limit=limit, offset=offset, output_format=output_format)
        finally:
            click.echo = _orig_echo
        _deliver_content(env, buf.getvalue(), destination=destination, out_path=out_path)
        return

    run_raw(env, request, session_id=session_id, limit=limit, offset=offset, output_format=output_format)


def _run_read_context(env: AppEnv, request: RootModeRequest, *, destination: str, out_path: str | None) -> None:
    """Route context view to the context command logic."""
    # The context command lives at polylogue/cli/commands/context.py; it is a
    # standalone command with its own option set, not a query verb. We delegate
    # to its compose_context_preamble helper for the archive-backed render.
    from polylogue.cli.query import execute_query_request

    updated = request.with_param_updates(output_format="markdown")
    if destination == "file":
        if not out_path:
            raise click.UsageError("--to file requires --out <path>.")
        execute_query_request(env, updated.with_param_updates(output=out_path))
    elif destination == "clipboard":
        execute_query_request(env, updated.with_param_updates(output="clipboard"))
    else:
        execute_query_request(env, updated)


def _run_read_bulk(
    env: AppEnv,
    request: RootModeRequest,
    *,
    output_format: str | None,
    fields: str | None,
    destination: str,
    out_path: str | None,
) -> None:
    """Bulk export all matched sessions."""
    from polylogue.cli.bulk_export import run_bulk_export

    fmt = output_format or "ndjson"
    # Normalize ndjson alias: bulk_export uses 'jsonl' internally.
    bulk_fmt = "jsonl" if fmt == "ndjson" else fmt

    if destination == "file":
        if not out_path:
            raise click.UsageError("--to file requires --out <path>.")
        from pathlib import Path

        buf = io.StringIO()

        def _captured_echo_bulk(message: object = None, **_kwargs: object) -> None:
            buf.write(str(message or "") + "\n")

        _orig_echo = click.echo
        click.echo = _captured_echo_bulk  # type: ignore[assignment]
        try:
            run_bulk_export(env, request, output_format=bulk_fmt, fields=fields)
        finally:
            click.echo = _orig_echo
        Path(out_path).write_text(buf.getvalue(), encoding="utf-8")
        env.ui.console.print(f"Wrote to {out_path}")
    else:
        run_bulk_export(env, request, output_format=bulk_fmt, fields=fields)


def _deliver_content(env: AppEnv, content: str, *, destination: str, out_path: str | None) -> None:
    """Deliver captured content to the requested destination."""
    if destination == "file":
        from pathlib import Path

        if not out_path:
            raise click.UsageError("--to file requires --out <path>.")
        Path(out_path).write_text(content, encoding="utf-8")
        env.ui.console.print(f"Wrote to {out_path}")
    elif destination == "clipboard":
        from polylogue.cli.query_output import copy_to_clipboard

        copy_to_clipboard(env, content)
    else:
        click.echo(content, nl=False)


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


QUERY_VERBS = (
    list_verb,
    count_verb,
    stats_verb,
    recent_verb,
    read_verb,
    delete_verb,
)


__all__ = [
    "QUERY_VERBS",
    "VERB_NAMES",
    "count_verb",
    "delete_verb",
    "list_verb",
    "read_verb",
    "recent_verb",
    "stats_verb",
]
