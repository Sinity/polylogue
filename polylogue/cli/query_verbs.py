"""Positional verb subcommands for the query-first CLI.

Each verb takes the filter set from the parent context and executes
a specific action on the matched sessions.
"""

from __future__ import annotations

import io
from typing import TYPE_CHECKING, cast

import click

if TYPE_CHECKING:
    from polylogue.archive.session.domain_models import Session, SessionSummary
    from polylogue.archive.session.neighbor_candidates import SessionNeighborCandidate
    from polylogue.cli.root_request import RootModeRequest
    from polylogue.insights.transforms import RecoveryReportPreset

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

_READ_VIEWS = (
    "summary",
    "transcript",
    "messages",
    "raw",
    "context",
    "context-pack",
    "recovery",
    "neighbors",
    "correlation",
)
_READ_DESTINATIONS = ("terminal", "stdout", "browser", "clipboard", "file")
_READ_FORMATS = ("text", "markdown", "json", "ndjson", "yaml", "html", "obsidian", "org", "csv")
_RECOVERY_REPORT_PRESETS = ("continue", "blame")


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
    help=(
        "What to render (summary, transcript, messages, raw, context, context-pack, recovery, neighbors, correlation)."
    ),
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
@click.option(
    "--window-hours",
    type=int,
    default=24,
    show_default=True,
    help="Neighboring time window around the seed session (--view neighbors).",
)
@click.option(
    "--repo-path",
    default=None,
    help="Git repository path for correlation (--view correlation). Defaults to the session's repo/cwd.",
)
@click.option(
    "--since-hours",
    type=int,
    default=2,
    show_default=True,
    help="Hours before/after the session to scan for commits (--view correlation).",
)
@click.option(
    "--confidence-threshold",
    type=float,
    default=0.3,
    show_default=True,
    help="Minimum confidence for file-overlap commit detection (--view correlation).",
)
@click.option(
    "--github-api/--no-github-api",
    default=True,
    help="Cross-reference issue/PR refs with the GitHub API via gh CLI (--view correlation).",
)
@click.option(
    "--otlp",
    is_flag=True,
    default=False,
    help="Add OTLP span evidence to correlation output (--view correlation).",
)
@click.option(
    "--related-limit",
    type=int,
    default=5,
    show_default=True,
    help="Number of related sessions to include (--view context).",
)
@click.option(
    "--report",
    "recovery_report",
    type=click.Choice(_RECOVERY_REPORT_PRESETS),
    default=None,
    help="Render a recovery report preset (--view recovery): continue or blame.",
)
@click.option("--project-path", default=None, help="Filter by cwd prefix pattern (--view context-pack).")
@click.option("--project-repo", default=None, help="Filter by git repo URL or name (--view context-pack).")
@click.option("--since", default=None, help="Start date, ISO 8601 (--view context-pack).")
@click.option("--until", default=None, help="End date, ISO 8601 (--view context-pack).")
@click.option("--pack-origin", "pack_origin", default=None, help="Source-origin filter (--view context-pack).")
@click.option("--query", "pack_query", default=None, help="Free-text query (--view context-pack).")
@click.option(
    "--max-sessions", type=int, default=5, show_default=True, help="Max sessions, 1-20 (--view context-pack)."
)
@click.option(
    "--max-messages",
    type=int,
    default=20,
    show_default=True,
    help="Max messages per session, 1-100 (--view context-pack).",
)
@click.option("--no-redact", is_flag=True, default=False, help="Do not redact filesystem paths (--view context-pack).")
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
    window_hours: int,
    repo_path: str | None,
    since_hours: int,
    confidence_threshold: float,
    github_api: bool,
    otlp: bool,
    related_limit: int,
    recovery_report: str | None,
    project_path: str | None,
    project_repo: str | None,
    since: str | None,
    until: str | None,
    pack_origin: str | None,
    pack_query: str | None,
    max_sessions: int,
    max_messages: int,
    no_redact: bool,
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
        polylogue find id:abc then read --view context --related-limit 5
        polylogue find 'cost tracking' then read --view context-pack --max-sessions 5
        polylogue read --view context-pack --project-repo github.com/Sinity/polylogue --since 2026-01-01
        polylogue find id:abc then read --view recovery
        polylogue find id:abc then read --view neighbors --window-hours 48
        polylogue --latest read --view neighbors --format json
        polylogue find id:abc then read --view correlation --since-hours 4
        polylogue --latest read --view correlation --otlp --format json

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
        _run_read_context(
            env, request, session_id=session_id, related_limit=related_limit, destination=destination, out_path=out_path
        )
        return

    if view == "context-pack":
        from polylogue.cli.commands.context_pack import run_context_pack_view

        run_context_pack_view(
            env,
            project_path=project_path,
            project_repo=project_repo,
            since=since,
            until=until,
            origin=pack_origin,
            query=pack_query,
            max_sessions=max_sessions,
            max_messages=max_messages,
            no_redact=no_redact,
        )
        return

    if view == "recovery":
        effective_format = output_format or request.params.get("output_format")
        _run_read_recovery(
            env,
            session_id=session_id,
            output_format=effective_format if isinstance(effective_format, str) else None,
            report=recovery_report,
            destination=destination,
            out_path=out_path,
        )
        return

    if view == "neighbors":
        _run_read_neighbors(
            env,
            request,
            session_id=session_id,
            limit=limit if limit is not None else 10,
            window_hours=max(1, window_hours),
            output_format=output_format,
            destination=destination,
            out_path=out_path,
        )
        return

    if view == "correlation":
        if session_id is None:
            raise click.UsageError("read --view correlation requires a session ID (use --id, id:prefix, or --latest).")
        from polylogue.cli.commands.correlate import run_correlation_view

        run_correlation_view(
            env,
            session_id=session_id,
            repo_path=repo_path,
            since_hours=since_hours,
            output_format=output_format,
            confidence_threshold=confidence_threshold,
            github_api=github_api,
            otlp=otlp,
        )
        return

    # summary / transcript: standard show/query path with destination routing.
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
        # Preserve the machine no-results contract that the removed
        # `open --print-url --format json` route carried: a JSON consumer gets a
        # structured error envelope and exit 2, not human text on stdout. The
        # JSON intent can arrive on the verb (`read -f json`) or the root
        # (`--format json … read`), so honor both.
        effective_format = output_format or request.params.get("output_format")
        if effective_format == "json":
            from polylogue.cli.shared.machine_errors import error_no_results

            error_no_results("No sessions matched.").emit(exit_code=2)
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


def _run_read_context(
    env: AppEnv,
    request: RootModeRequest,
    *,
    session_id: str | None,
    related_limit: int,
    destination: str,
    out_path: str | None,
) -> None:
    """Compose the context preamble for the seed session (--view context).

    Absorbs the former ``context compose`` command (#1842): resolves the seed
    from the query (``--id``/``id:``/``--latest``) and emits a context-compose
    preamble JSON document. The MCP ``compose_context_preamble`` tool exposes
    the same capability programmatically.
    """
    from polylogue.cli.commands.context import run_context_compose

    if session_id is None:
        raise click.UsageError("read --view context requires a session ID (use --id, id:prefix, or --latest).")
    preamble = run_context_compose(env, session_id=session_id, related_limit=max(1, related_limit))
    _deliver_content(env, preamble + "\n", destination=destination, out_path=out_path)


def _run_read_recovery(
    env: AppEnv,
    *,
    session_id: str | None,
    output_format: str | None,
    report: str | None = None,
    destination: str,
    out_path: str | None,
) -> None:
    """Render the deterministic recovery digest for one archived session (#1880)."""
    from polylogue.api.sync.bridge import run_coroutine_sync
    from polylogue.cli.shared.helper_support import fail
    from polylogue.cli.shared.machine_errors import success
    from polylogue.insights.transforms import compile_recovery_digest
    from polylogue.surfaces.payloads import model_json_document

    if session_id is None:
        fail("read", "read --view recovery requires a session ID (use --id, id:prefix, or --latest).")
    session = run_coroutine_sync(env.polylogue.get_session(session_id))
    if session is None:
        fail("read", f"Session not found: {session_id}")
    digest = compile_recovery_digest(session)
    if report is not None:
        _deliver_content(
            env,
            digest.report_markdown(cast("RecoveryReportPreset", report)),
            destination=destination,
            out_path=out_path,
        )
        return
    if output_format == "json":
        payload = success({"recovery": model_json_document(digest, exclude_none=True)}).to_json()
        _deliver_content(env, payload + "\n", destination=destination, out_path=out_path)
        return
    _deliver_content(env, digest.resume_markdown, destination=destination, out_path=out_path)


def _neighbor_score_label(score: float) -> str:
    return f"{score:.2f}".rstrip("0").rstrip(".")


def _neighbor_candidate_heading(candidate: SessionNeighborCandidate) -> str:
    summary = candidate.summary
    date = f" {summary.display_date.isoformat()}" if summary.display_date else ""
    return (
        f"{candidate.rank}. {candidate.session_id} "
        f"[{summary.origin.value}] {summary.display_title}{date} "
        f"(score {_neighbor_score_label(candidate.score)})"
    )


def _render_neighbors_plain(candidates: list[SessionNeighborCandidate]) -> str:
    if not candidates:
        return "No neighboring candidates found.\n"
    lines = [f"Neighbor candidates ({len(candidates)}):"]
    for candidate in candidates:
        lines.append(_neighbor_candidate_heading(candidate))
        for reason in candidate.reasons:
            evidence = f" ({reason.evidence})" if reason.evidence else ""
            lines.append(f"   - {reason.kind}: {reason.detail}{evidence}")
    return "\n".join(lines) + "\n"


def _run_read_neighbors(
    env: AppEnv,
    request: RootModeRequest,
    *,
    session_id: str | None,
    limit: int,
    window_hours: int,
    output_format: str | None,
    destination: str,
    out_path: str | None,
) -> None:
    """Render explainable neighbor/near-duplicate candidates for a seed session.

    Absorbs the former ``neighbors`` command (#1842): the seed is resolved from
    the query (``--id``/``id:``/``--latest``) and the free-text query terms,
    scoped by the root ``--origin`` filter. The MCP ``neighbor_candidates`` tool
    exposes the same capability programmatically.
    """
    from polylogue.api.sync.bridge import run_coroutine_sync
    from polylogue.archive.session.neighbor_candidates import NeighborDiscoveryError
    from polylogue.cli.shared.helper_support import fail
    from polylogue.cli.shared.machine_errors import emit_success
    from polylogue.core.enums import Origin
    from polylogue.core.sources import provider_from_origin
    from polylogue.surfaces.payloads import SessionNeighborCandidatePayload, model_json_document

    query_seed = " ".join(request.query_terms).strip() or None
    if not session_id and not query_seed:
        fail("read", "read --view neighbors requires a seed (use --id, id:prefix, --latest, or a query).")

    origin = request.params.get("origin")
    provider = provider_from_origin(Origin(str(origin))).value if origin else None

    try:
        candidates = run_coroutine_sync(
            env.polylogue.neighbor_candidates(
                session_id=session_id,
                query=query_seed,
                provider=provider,
                limit=max(1, limit),
                window_hours=max(1, window_hours),
            )
        )
    except NeighborDiscoveryError as exc:
        fail("read", str(exc))

    if output_format == "json":
        emit_success(
            {
                "neighbors": [
                    model_json_document(
                        SessionNeighborCandidatePayload.from_candidate(candidate),
                        exclude_none=True,
                    )
                    for candidate in candidates
                ]
            }
        )
        return

    _deliver_content(env, _render_neighbors_plain(candidates), destination=destination, out_path=out_path)


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
@click.option("--dry-run", is_flag=True, help="Preview what would be deleted without deleting")
@click.option("--yes", "yes_flag", is_flag=True, help="Confirm the deletion (required for actual deletion)")
@click.option("--all", "all_flag", is_flag=True, help="Delete all matched sessions (required when multiple match)")
@click.option("--force", is_flag=True, hidden=True, help="(legacy alias for --yes) Skip confirmation prompt")
@click.pass_context
def delete_verb(ctx: click.Context, dry_run: bool, yes_flag: bool, all_flag: bool, force: bool) -> None:
    """Delete matched sessions.

    \b
    Cardinality rules:
      --dry-run       Preview what would be deleted (no confirmation needed).
      --yes           Confirm deletion for a single matched session.
      --yes --all     Required when the query matches more than one session.

    \b
    Examples:
        polylogue find id:abc then delete --dry-run
        polylogue find id:abc then delete --yes
        polylogue find 'repo:polylogue since:7d' then delete --dry-run
        polylogue find 'repo:polylogue since:7d' then delete --yes --all
    """
    from polylogue.cli.verb_cardinality import CardinalityError, check_cardinality, resolve_session_ids_for_verb

    env: AppEnv = ctx.obj
    request = _parent_request(ctx)

    from polylogue.cli.archive_query import execute_delete_by_session_ids

    # dry-run: skip the cardinality guard and just show a preview. Resolve the
    # SAME full ID set the real delete uses (via resolve_session_ids_for_verb)
    # rather than re-running the query through _execute_query_verb, which caps at
    # the default limit of 20 and would preview fewer sessions than --yes --all
    # actually deletes (#1873). The previewed set must equal the deleted set.
    if dry_run:
        session_ids = resolve_session_ids_for_verb(env, request)
        execute_delete_by_session_ids(env, session_ids, force=True, dry_run=True)
        return

    # Enforce cardinality before any destructive action.
    session_ids = resolve_session_ids_for_verb(env, request)
    try:
        check_cardinality(len(session_ids), allow_all=all_flag, first_only=False, operation="delete")
    except CardinalityError as exc:
        raise click.UsageError(str(exc)) from exc

    # Delete using the pre-resolved IDs so all matched sessions are removed.
    execute_delete_by_session_ids(env, session_ids, force=yes_flag or force)


@click.command("mark")
@click.option("--tag-add", "tags_to_add", multiple=True, metavar="TAG", help="Add a tag to the matched session(s)")
@click.option(
    "--tag-remove", "tags_to_remove", multiple=True, metavar="TAG", help="Remove a tag from the matched session(s)"
)
@click.option("--star", "star", is_flag=True, help="Star the matched session")
@click.option("--unstar", "unstar", is_flag=True, help="Remove star from the matched session")
@click.option("--pin", "pin", is_flag=True, help="Pin the matched session")
@click.option("--unpin", "unpin", is_flag=True, help="Remove pin from the matched session")
@click.option("--archive", "do_archive", is_flag=True, help="Archive-mark the matched session")
@click.option("--unarchive", "do_unarchive", is_flag=True, help="Remove archive-mark from the matched session")
@click.option("--note", "note_text", default=None, metavar="TEXT", help="Add or update a note annotation")
@click.option("--all", "apply_all", is_flag=True, help="Apply to all matched sessions (default: singleton only)")
@click.option("--first", "first_only", is_flag=True, help="Apply to the first matched session only")
@click.pass_context
def mark_verb(
    ctx: click.Context,
    tags_to_add: tuple[str, ...],
    tags_to_remove: tuple[str, ...],
    star: bool,
    unstar: bool,
    pin: bool,
    unpin: bool,
    do_archive: bool,
    do_unarchive: bool,
    note_text: str | None,
    apply_all: bool,
    first_only: bool,
) -> None:
    """Mark matched sessions with tags, notes, or user-state marks.

    \b
    Requires exactly one matched session unless --all is present.
    Use --first to act on the first match when the query is non-specific.

    \b
    Mark types: star, pin, archive (managed via --star/--unstar, --pin/--unpin,
    --archive/--unarchive).  Tags are free-form strings.  Notes are stored as
    durable annotations on the session.

    \b
    Examples:
        polylogue find id:abc then mark --tag-add reviewed
        polylogue find id:abc then mark --star --note "key insight"
        polylogue find id:abc then mark --unstar --tag-remove reviewed
        polylogue find id:abc then mark --pin
        polylogue find 'repo:polylogue since:7d' then mark --tag-add sprint --all
    """
    import uuid

    from polylogue.api.sync.bridge import run_coroutine_sync
    from polylogue.cli.verb_cardinality import CardinalityError, check_cardinality, resolve_session_ids_for_verb

    env: AppEnv = ctx.obj
    request = _parent_request(ctx)

    # Resolve matched sessions and enforce cardinality.
    session_ids = resolve_session_ids_for_verb(env, request)
    try:
        check_cardinality(len(session_ids), allow_all=apply_all, first_only=first_only, operation="mark")
    except CardinalityError as exc:
        raise click.UsageError(str(exc)) from exc

    # Honour --first: act only on the leading result when multiple matched.
    target_ids = session_ids[:1] if first_only and len(session_ids) > 1 else session_ids

    async def _apply_marks() -> None:
        poly = env.polylogue
        for sid in target_ids:
            for tag in tags_to_add:
                await poly.add_tag(sid, tag)
            for tag in tags_to_remove:
                await poly.remove_tag(sid, tag)
            if star:
                await poly.add_mark(sid, "star")
            if unstar:
                await poly.remove_mark(sid, "star")
            if pin:
                await poly.add_mark(sid, "pin")
            if unpin:
                await poly.remove_mark(sid, "pin")
            if do_archive:
                await poly.add_mark(sid, "archive")
            if do_unarchive:
                await poly.remove_mark(sid, "archive")
            if note_text is not None:
                annotation_id = f"note-{uuid.uuid4().hex[:16]}"
                await poly.save_annotation(annotation_id, sid, note_text)

    run_coroutine_sync(_apply_marks())

    # Report.
    count = len(target_ids)
    ops: list[str] = []
    if tags_to_add:
        ops.append(f"added tags: {', '.join(tags_to_add)}")
    if tags_to_remove:
        ops.append(f"removed tags: {', '.join(tags_to_remove)}")
    if star:
        ops.append("starred")
    if unstar:
        ops.append("unstarred")
    if pin:
        ops.append("pinned")
    if unpin:
        ops.append("unpinned")
    if do_archive:
        ops.append("archive-marked")
    if do_unarchive:
        ops.append("archive-mark removed")
    if note_text is not None:
        ops.append("noted")
    if ops:
        click.echo(f"Marked {count} session(s): {'; '.join(ops)}")
    else:
        click.echo("No mark operations specified.")


@click.command("analyze")
@click.option(
    "--by",
    "stats_by",
    type=click.Choice(["origin", "month", "year", "day", "action", "tool", "repo", "work-kind"]),
    default=None,
    help="Group statistics by dimension",
)
@click.option("--facets", "show_facets", is_flag=True, help="Show facet aggregates for the matched result set")
@click.option(
    "--no-idf",
    "no_idf",
    is_flag=True,
    default=False,
    help="With --facets, skip inverse-document-frequency weighting",
)
@click.option(
    "--format",
    "-f",
    "output_format",
    type=click.Choice(["markdown", "json", "ndjson", "html", "plaintext", "csv"]),
    default=None,
    help="Output format",
)
@click.pass_context
def analyze_verb(
    ctx: click.Context,
    stats_by: str | None,
    show_facets: bool,
    no_idf: bool,
    output_format: str | None,
) -> None:
    """Analyze matched sessions: statistics, facets, and aggregates.

    \b
    Applies to the full result set by default (no cardinality restriction).
    Wraps the existing stats and facets surfaces over the matched session set.

    \b
    Examples:
        polylogue find 'repo:polylogue since:7d' then analyze
        polylogue find 'repo:polylogue since:7d' then analyze --by origin
        polylogue find 'repo:polylogue since:7d' then analyze --by month
        polylogue find 'repo:polylogue' then analyze --facets
        polylogue find 'repo:polylogue' then analyze --by day --format json
    """
    import json as _json

    from polylogue.api.sync.bridge import run_coroutine_sync

    env: AppEnv = ctx.obj
    request = _parent_request(ctx)

    if show_facets:
        # Delegate to the Polylogue facets API using the request's query spec.
        spec = request.query_spec()
        response = run_coroutine_sync(env.polylogue.facets(spec, include_idf=not no_idf))
        if output_format == "json":
            click.echo(_json.dumps(response.model_dump(mode="json", by_alias=True), indent=2))
            return
        scope_label = "scoped" if response.scoped_to_query else "global"
        click.echo(f"Facets ({scope_label}) — matched result set:")
        click.echo(f"  sessions: {response.scoped.total_sessions}  messages: {response.scoped.total_messages}")
        if response.scoped.origins:
            click.echo("  Origins:")
            for origin, cnt in sorted(response.scoped.origins.items(), key=lambda kv: -kv[1]):
                click.echo(f"    {origin}: {cnt}")
        if response.scoped.tags:
            click.echo("  Tags:")
            for tag, cnt in sorted(response.scoped.tags.items(), key=lambda kv: -kv[1]):
                click.echo(f"    {tag}: {cnt}")
        if response.idf:
            click.echo("  IDF (higher = rarer, partitions more strongly):")
            for family, values in response.idf.items():
                click.echo(f"    [{family}]")
                for value, weight in sorted(values.items(), key=lambda kv: -kv[1]):
                    click.echo(f"      {value}: {weight:.3f}")
        return

    if stats_by:
        updated = request.with_param_updates(stats_by=stats_by)
        if output_format:
            updated = updated.with_param_updates(output_format=output_format)
        _execute_query_verb(ctx, updated)
        return

    # Default: overall stats for the result set.
    updated = request.with_param_updates(stats_only=True)
    if output_format:
        updated = updated.with_param_updates(output_format=output_format)
    _execute_query_verb(ctx, updated)


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
    mark_verb,
    analyze_verb,
)


__all__ = [
    "QUERY_VERBS",
    "VERB_NAMES",
    "analyze_verb",
    "count_verb",
    "delete_verb",
    "list_verb",
    "mark_verb",
    "read_verb",
    "recent_verb",
    "stats_verb",
]
