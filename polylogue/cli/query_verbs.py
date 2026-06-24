"""Positional verb subcommands for the query-first CLI.

Each verb takes the filter set from the parent context and executes
a specific action on the matched sessions.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, cast

import click
from click.shell_completion import CompletionItem

if TYPE_CHECKING:
    from polylogue.cli.root_request import RootModeRequest
    from polylogue.cli.select import SelectPrintField

from polylogue.api.sync.bridge import run_coroutine_sync
from polylogue.archive.viewport import (
    READ_VIEW_PROFILE_BY_ID,
    READ_VIEW_PROFILES,
    read_view_choices,
    read_view_profile_payloads,
)
from polylogue.cli.click_option_groups import _LazyChoice
from polylogue.cli.read_view_handlers import (
    ReadViewInvocation,
    read_view_option_names,
    read_view_options_for_view,
    run_bulk_export_view,
    run_read_view,
)
from polylogue.cli.shared.types import AppEnv
from polylogue.cli.verb_names import VERB_NAMES
from polylogue.core.enums import AssertionStatus
from polylogue.surfaces.payloads import (
    AssertionClaimListPayload,
    serialize_surface_payload,
)


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


def _get_material_origin_class() -> object:  # pragma: no cover — returns MaterialOrigin
    from polylogue.core.enums import MaterialOrigin

    return MaterialOrigin


def _get_material_origin_choices() -> list[str]:
    return [m.value for m in _get_material_origin_class()]  # type: ignore[attr-defined]


def _lazy_shell_complete(source: str):  # type: ignore[no-untyped-def]
    def _complete(ctx: click.Context, param: click.Parameter, incomplete: str):  # type: ignore[no-untyped-def]
        from polylogue.cli.shell_completion_values import complete_query_source

        return complete_query_source(source)(ctx, param, incomplete)  # type: ignore[arg-type]

    _complete.__name__ = f"complete_{source}"
    return _complete


_complete_session_id = _lazy_shell_complete("session_id")
_complete_material_origin = _lazy_shell_complete("material_origin")
_complete_message_type = _lazy_shell_complete("message_type")

_READ_VIEWS = read_view_choices()
_READ_VIEW_HELP = "What to render (" + ", ".join(_READ_VIEWS) + ")."
_READ_DESTINATIONS = ("terminal", "stdout", "browser", "clipboard", "file")
_READ_FORMATS = tuple(sorted({fmt for profile in READ_VIEW_PROFILES for fmt in profile.formats}))
_RECOVERY_REPORT_PRESETS = ("continue", "blame", "work-packet")


def _explicit_read_view_options(ctx: click.Context) -> frozenset[str]:
    """Return view-specific read options supplied on the command line."""

    return frozenset(
        name
        for name in read_view_option_names()
        if (source := ctx.get_parameter_source(name)) is not None and source.name == "COMMANDLINE"
    )


def _read_view_option_values(
    *,
    limit: int | None,
    offset: int,
    message_role: tuple[str, ...],
    material_origin: tuple[str, ...],
    message_type: str | None,
    no_code_blocks: bool,
    no_tool_calls: bool,
    no_tool_outputs: bool,
    no_file_reads: bool,
    prose_only: bool,
    related_limit: int,
    project_path: str | None,
    project_repo: str | None,
    since: str | None,
    until: str | None,
    pack_origin: str | None,
    pack_query: str | None,
    max_sessions: int,
    max_messages: int,
    no_redact: bool,
    recovery_report: str | None,
    window_hours: int,
    repo_path: str | None,
    since_hours: int,
    confidence_threshold: float,
    github_api: bool,
    otlp: bool,
) -> dict[str, object]:
    """Collect raw Click option values for read-view handler builders."""

    return {
        "limit": limit,
        "offset": offset,
        "message_role": message_role,
        "material_origin": material_origin,
        "message_type": message_type,
        "no_code_blocks": no_code_blocks,
        "no_tool_calls": no_tool_calls,
        "no_tool_outputs": no_tool_outputs,
        "no_file_reads": no_file_reads,
        "prose_only": prose_only,
        "related_limit": related_limit,
        "project_path": project_path,
        "project_repo": project_repo,
        "since": since,
        "until": until,
        "pack_origin": pack_origin,
        "pack_query": pack_query,
        "max_sessions": max_sessions,
        "max_messages": max_messages,
        "no_redact": no_redact,
        "recovery_report": recovery_report,
        "window_hours": window_hours,
        "repo_path": repo_path,
        "since_hours": since_hours,
        "confidence_threshold": confidence_threshold,
        "github_api": github_api,
        "otlp": otlp,
    }


_CONTINUE_CANDIDATE_DEFAULT_LIMIT = 10


def _wants_json(request: RootModeRequest, *, output_format: str | None) -> bool:
    """Return whether the local/root output contract requests JSON."""

    if output_format == "json":
        return True
    root_output = request.params.get("output_format")
    return root_output == "json"


def _emit_continue_candidates(
    env: AppEnv,
    request: RootModeRequest,
    *,
    repo_path: str,
    cwd: str | None,
    recent_files: tuple[str, ...],
    limit: int,
    output_format: str | None,
) -> None:
    """Rank archived sessions for continuation from the current work context."""

    from polylogue.api.sync.bridge import run_coroutine_sync
    from polylogue.cli.shared.machine_errors import emit_success

    candidates = run_coroutine_sync(
        env.polylogue.find_resume_candidates(
            repo_path=repo_path,
            cwd=cwd,
            recent_files=recent_files,
            limit=limit,
        )
    )
    payload = {
        "candidates": [candidate.model_dump(mode="json") for candidate in candidates],
        "total": len(candidates),
    }
    if _wants_json(request, output_format=output_format):
        emit_success(payload)
        return
    for candidate in candidates:
        click.echo(f"{candidate.score:.3f} {candidate.logical_session_id} {candidate.title}")


def _complete_read_view(ctx: click.Context, param: click.Parameter, incomplete: str) -> list[CompletionItem]:
    """Complete read-view ids from the shared view-profile registry."""

    needle = incomplete.lower()
    return [
        CompletionItem(profile.view_id, help=f"{profile.label}: {profile.purpose}")
        for profile in READ_VIEW_PROFILES
        if profile.view_id.startswith(needle)
    ]


def _selected_read_view(ctx: click.Context) -> str | None:
    value = ctx.params.get("view")
    if isinstance(value, str) and value in READ_VIEW_PROFILE_BY_ID:
        return value
    return None


def _complete_read_format(ctx: click.Context, param: click.Parameter, incomplete: str) -> list[CompletionItem]:
    """Complete output formats from read-view profile metadata."""

    del param
    needle = incomplete.lower()
    selected_view = _selected_read_view(ctx)
    profiles = (READ_VIEW_PROFILE_BY_ID[selected_view],) if selected_view is not None else READ_VIEW_PROFILES
    items: dict[str, str] = {}
    for profile in profiles:
        for output_format in profile.formats:
            if needle and not output_format.startswith(needle):
                continue
            items.setdefault(output_format, f"Supported by read --view {profile.view_id}")
    return [CompletionItem(output_format, help=help_text) for output_format, help_text in sorted(items.items())]


def _render_read_view_profiles_plain() -> str:
    lines = ["Read views:"]
    for profile in READ_VIEW_PROFILES:
        handoff = " handoff" if profile.successor_handoff else ""
        lines.append(
            f"  {profile.view_id:<12} {profile.lossiness:<10} evidence={profile.evidence_policy:<10}"
            f" formats={','.join(profile.formats)}{handoff}"
        )
        lines.append(f"      {profile.purpose}")
    return "\n".join(lines)


def _emit_read_view_profiles(output_format: str | None) -> None:
    if output_format == "json":
        from polylogue.cli.shared.machine_errors import emit_success

        emit_success({"read_views": read_view_profile_payloads()})
        return
    if output_format is not None:
        raise click.UsageError("`read --views` only supports terminal text or --format json")
    click.echo(_render_read_view_profiles_plain())


def _summary_all_output_param(destination: str, out_path: str | None) -> str | None:
    """Translate read delivery options to the root query output contract."""
    if destination == "file":
        if not out_path:
            raise click.UsageError("--to file requires --out <path>.")
        return out_path
    if destination in {"browser", "clipboard", "stdout"}:
        return destination
    return None


@click.command("select")
@click.option(
    "--limit", "-n", default=20, show_default=True, type=click.IntRange(min=1), help="Max candidate sessions."
)
@click.option(
    "--print",
    "print_field",
    type=click.Choice(["id", "title", "origin"]),
    default="id",
    show_default=True,
    help="Field to print for the selected session.",
)
@click.option("--json", "json_output", is_flag=True, help="Print the selected session as one JSON object.")
@click.pass_context
def select_verb(ctx: click.Context, limit: int, print_field: str, json_output: bool) -> None:
    """Select one matched session with fzf/prompt fallback."""
    from polylogue.cli.select import run_select

    field: SelectPrintField = "json" if json_output else cast("SelectPrintField", print_field)
    request = _parent_request(ctx)
    if _explain_terminal_action(
        request,
        action="select",
        limit=limit,
        print_field=field,
    ):
        return
    run_select(ctx.obj, request, limit=limit, print_field=field)


@click.command("read")
@click.option(
    "--view",
    "-v",
    type=click.Choice(_READ_VIEWS),
    default="summary",
    show_default=True,
    shell_complete=_complete_read_view,
    help=_READ_VIEW_HELP,
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
    shell_complete=_complete_read_format,
    help="Output format (where applicable).",
)
@click.option("--out", "out_path", type=click.Path(), default=None, help="File path for --to file.")
@click.option("--all", "export_all", is_flag=True, help="Apply to all matched sessions (bulk export).")
# message/raw pagination flags
@click.option("--message-role", "-r", "message_role", multiple=True, help="Filter by message role (--view messages).")
@click.option(
    "--material-origin",
    "material_origin",
    multiple=True,
    type=_LazyChoice(_get_material_origin_choices, "origin"),
    shell_complete=_complete_material_origin,
    help="Filter by material origin (--view messages).",
)
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
    help="Render a recovery report preset (--view recovery): continue, blame, or work-packet.",
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
@click.option("--views", "show_views", is_flag=True, help="List executable read-view profiles and exit.")
@click.argument("ref", required=False)
@click.pass_context
def read_verb(
    ctx: click.Context,
    view: str,
    destination: str,
    output_format: str | None,
    out_path: str | None,
    export_all: bool,
    message_role: tuple[str, ...],
    material_origin: tuple[str, ...],
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
    show_views: bool = False,
    ref: str | None = None,
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
        polylogue read --views
        polylogue read --views --format json
        polylogue find id:abc then read --view neighbors --window-hours 48
        polylogue --latest read --view neighbors --format json
        polylogue find id:abc then read --view correlation --since-hours 4
        polylogue --latest read --view correlation --otlp --format json
        polylogue read session:abc123 --format json

    \b
    Reserved views (not yet implemented):
        timeline, tools, files, metadata, continuation
    """
    env: AppEnv = ctx.obj
    if show_views:
        _emit_read_view_profiles(output_format)
        return
    request = _parent_request(ctx)
    if _explain_terminal_action(
        request,
        action="read",
        view=view,
        destination=destination,
        format=_effective_read_output_format(request, view=view, output_format=output_format) or "default",
        all=export_all,
    ):
        return
    if ref is not None:
        if output_format not in (None, "json"):
            raise click.UsageError("Direct ref reads currently support --format json only.")
        if destination not in ("terminal", "stdout"):
            raise click.UsageError("Direct ref reads write JSON to terminal/stdout only.")
        payload = run_coroutine_sync(env.polylogue.resolve_ref(ref))
        click.echo(serialize_surface_payload(payload, exclude_none=True))
        return

    # Summary all-mode is the command-floor replacement for the old list verb:
    # it preserves the summary-list envelope and fields/limit behavior instead
    # of exporting full transcript payloads.
    if export_all:
        if limit is not None:
            request = request.with_param_updates(limit=limit)
        if view == "summary":
            request = request.with_param_updates(list_mode=True)
            if output_format:
                request = request.with_param_updates(output_format=output_format)
            output = _summary_all_output_param(destination, out_path)
            if output is not None:
                request = request.with_param_updates(output=output)
            if fields:
                request = request.with_param_updates(fields=fields)
            _execute_query_verb(ctx, request)
            return
        run_bulk_export_view(
            env, request, output_format=output_format, fields=fields, destination=destination, out_path=out_path
        )
        return

    session_id = _resolve_target_session_id(request)
    effective_format = _effective_read_output_format(request, view=view, output_format=output_format)
    explicit_options = _explicit_read_view_options(ctx)
    if destination == "browser":
        run_read_view(
            env,
            request,
            ReadViewInvocation(
                view="summary",
                session_id=session_id,
                output_format=effective_format,
                destination=destination,
                out_path=out_path,
                explicit_options=explicit_options,
            ),
        )
        return

    run_read_view(
        env,
        request,
        ReadViewInvocation(
            view=view,
            session_id=session_id,
            output_format=effective_format,
            destination=destination,
            out_path=out_path,
            options=read_view_options_for_view(
                view,
                _read_view_option_values(
                    limit=limit,
                    offset=offset,
                    message_role=message_role,
                    material_origin=material_origin,
                    message_type=message_type,
                    no_code_blocks=no_code_blocks,
                    no_tool_calls=no_tool_calls,
                    no_tool_outputs=no_tool_outputs,
                    no_file_reads=no_file_reads,
                    prose_only=prose_only,
                    related_limit=related_limit,
                    project_path=project_path,
                    project_repo=project_repo,
                    since=since,
                    until=until,
                    pack_origin=pack_origin,
                    pack_query=pack_query,
                    max_sessions=max_sessions,
                    max_messages=max_messages,
                    no_redact=no_redact,
                    recovery_report=recovery_report,
                    window_hours=window_hours,
                    repo_path=repo_path,
                    since_hours=since_hours,
                    confidence_threshold=confidence_threshold,
                    github_api=github_api,
                    otlp=otlp,
                ),
            ),
            explicit_options=explicit_options,
        ),
    )


@click.command("continue")
@click.option(
    "--to",
    "destination",
    type=click.Choice(_READ_DESTINATIONS),
    default="terminal",
    show_default=True,
    help="Output destination.",
)
@click.option("--out", "out_path", type=click.Path(), default=None, help="File path for --to file.")
@click.option(
    "--candidates",
    is_flag=True,
    help="Rank archived sessions that are likely continuation targets for a repository context.",
)
@click.option("--repo", "repo_path", default=None, help="Repository path for --candidates ranking.")
@click.option("--cwd", default=None, help="Current working directory for --candidates prefix matching.")
@click.option(
    "--recent", "recent_files", multiple=True, help="Recently touched file path for --candidates. Repeatable."
)
@click.option(
    "--limit",
    "candidate_limit",
    type=int,
    default=_CONTINUE_CANDIDATE_DEFAULT_LIMIT,
    show_default=True,
    help="Maximum continuation candidates to return.",
)
@click.option(
    "--format",
    "-f",
    "output_format",
    type=click.Choice(["json"]),
    default=None,
    help="Output format. JSON emits the shared ContextImage payload.",
)
@click.pass_context
def continue_verb(
    ctx: click.Context,
    destination: str,
    out_path: str | None,
    candidates: bool,
    repo_path: str | None,
    cwd: str | None,
    recent_files: tuple[str, ...],
    candidate_limit: int,
    output_format: str | None,
) -> None:
    """Compile a successor-agent continuation report for one matched session.

    \b
    Examples:
        polylogue find id:abc then continue
        polylogue find id:abc then continue --format json
        polylogue --latest continue --to clipboard
        polylogue find 'repo:polylogue near:id:abc' then continue --to file --out handoff.md
        polylogue continue --candidates --repo /workspace/polylogue --recent polylogue/cli/query_verbs.py
    """
    env: AppEnv = ctx.obj
    request = _parent_request(ctx)
    if _explain_terminal_action(
        request,
        action="continue",
        destination=destination,
        format=output_format or request.params.get("output_format") or "default",
        candidates=candidates,
    ):
        return
    if candidates:
        if destination not in ("terminal", "stdout") or out_path is not None:
            raise click.UsageError("continue --candidates writes to terminal/stdout; omit --to/--out.")
        if not repo_path:
            raise click.UsageError("continue --candidates requires --repo.")
        _emit_continue_candidates(
            env,
            request,
            repo_path=repo_path,
            cwd=cwd,
            recent_files=recent_files,
            limit=candidate_limit,
            output_format=output_format,
        )
        return
    if repo_path or cwd or recent_files:
        raise click.UsageError("--repo, --cwd, and --recent are only valid with continue --candidates.")
    if candidate_limit != _CONTINUE_CANDIDATE_DEFAULT_LIMIT:
        raise click.UsageError("--limit is only valid with continue --candidates.")
    session_id = _resolve_target_session_id(request)
    if session_id is None:
        raise click.UsageError("continue requires one matched session (use --id, --latest, or a narrowing query).")
    root_format = request.params.get("output_format")
    effective_format = (
        output_format if output_format is not None else root_format if isinstance(root_format, str) else None
    )
    if effective_format == "json":
        if destination not in ("terminal", "stdout", "file"):
            raise click.UsageError("continue --format json supports terminal, stdout, or file destinations only.")
        if destination == "file" and not out_path:
            raise click.UsageError("continue --format json --to file requires --out.")
        from pathlib import Path

        from polylogue.context.compiler import ContextSpec

        image = run_coroutine_sync(
            env.polylogue.compile_context(
                ContextSpec(
                    purpose="continue",
                    seed_refs=(f"session:{session_id}",),
                    read_views=("recovery",),
                )
            )
        )
        rendered = serialize_surface_payload(image, exclude_none=True)
        if destination == "file":
            assert out_path is not None
            Path(out_path).write_text(rendered + "\n", encoding="utf-8")
        else:
            click.echo(rendered)
        return
    run_read_view(
        env,
        request,
        ReadViewInvocation(
            view="recovery",
            session_id=session_id,
            output_format=effective_format,
            destination=destination,
            out_path=out_path,
            options=read_view_options_for_view("recovery", {"recovery_report": "continue"}),
        ),
    )


@click.command("delete")
@click.option("--dry-run", is_flag=True, help="Preview what would be deleted without deleting")
@click.option("--yes", "yes_flag", is_flag=True, help="Confirm the deletion (required for actual deletion)")
@click.option("--all", "all_flag", is_flag=True, help="Delete all matched sessions (required when multiple match)")
@click.pass_context
def delete_verb(ctx: click.Context, dry_run: bool, yes_flag: bool, all_flag: bool) -> None:
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
    if _explain_terminal_action(
        request,
        action="delete",
        dry_run=dry_run,
        yes=yes_flag,
        all=all_flag,
    ):
        return

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
    execute_delete_by_session_ids(env, session_ids, force=yes_flag)


@click.group("mark", invoke_without_command=True)
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
    """Mark query-result sessions with tags, notes, or durable marks.

    This owns session overlays only. Use `mark candidates` for assertion-candidate review;
    target-ref/web annotations are separate surfaces, not hidden here.

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

    if ctx.invoked_subcommand is not None:
        return

    env: AppEnv = ctx.obj
    request = _parent_request(ctx)
    if _explain_terminal_action(
        request,
        action="mark",
        tag_add=tags_to_add,
        tag_remove=tags_to_remove,
        star=star,
        unstar=unstar,
        pin=pin,
        unpin=unpin,
        archive=do_archive,
        unarchive=do_unarchive,
        note=note_text is not None,
        all=apply_all,
        first=first_only,
    ):
        return

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


@mark_verb.group("candidates")
def mark_candidates_group() -> None:
    """Review assertion candidates for a selected session or target ref.

    Candidate commands own claim judgment: list, accept, reject, defer, or
    supersede candidate assertions. They do not apply ordinary session tags,
    stars, pins, archive marks, or notes.
    """


@mark_candidates_group.command("list")
@click.option("--target-ref", default=None, help="Limit candidates to one target object ref.")
@click.option("--limit", "-l", type=int, default=50, show_default=True)
@click.option("--format", "-f", "output_format", type=click.Choice(["json"]), default=None)
@click.pass_obj
def list_mark_candidates_command(env: AppEnv, target_ref: str | None, limit: int, output_format: str | None) -> None:
    """List candidate assertion claims."""
    items = run_coroutine_sync(env.polylogue.list_assertion_candidates(target_ref=target_ref, limit=limit))
    if output_format == "json":
        payload = AssertionClaimListPayload(
            items=tuple(items), total=len(items), limit=limit, statuses=(AssertionStatus.CANDIDATE,)
        )
        click.echo(serialize_surface_payload(payload, exclude_none=True))
        return
    if not items:
        click.echo("No candidate assertions found.")
        return
    for item in items:
        detail = item.body_text or (json.dumps(item.value, sort_keys=True) if item.value is not None else "")
        click.echo(f"{item.assertion_id:<32} {item.kind:<20} {item.target_ref} {detail}")


@mark_candidates_group.command("accept")
@click.argument("candidate_ref")
@click.option("--reason", default=None)
@click.option("--actor-ref", default="user:local", show_default=True)
@click.option("--format", "-f", "output_format", type=click.Choice(["json"]), default=None)
@click.pass_obj
def accept_mark_candidate_command(
    env: AppEnv,
    candidate_ref: str,
    reason: str | None,
    actor_ref: str,
    output_format: str | None,
) -> None:
    """Accept a candidate assertion into an active assertion."""
    _emit_candidate_judgment(
        env,
        candidate_ref=candidate_ref,
        decision="accept",
        reason=reason,
        actor_ref=actor_ref,
        output_format=output_format,
    )


@mark_candidates_group.command("reject")
@click.argument("candidate_ref")
@click.option("--reason", required=True)
@click.option("--actor-ref", default="user:local", show_default=True)
@click.option("--format", "-f", "output_format", type=click.Choice(["json"]), default=None)
@click.pass_obj
def reject_mark_candidate_command(
    env: AppEnv,
    candidate_ref: str,
    reason: str,
    actor_ref: str,
    output_format: str | None,
) -> None:
    """Reject a candidate assertion with a durable reason."""
    _emit_candidate_judgment(
        env,
        candidate_ref=candidate_ref,
        decision="reject",
        reason=reason,
        actor_ref=actor_ref,
        output_format=output_format,
    )


@mark_candidates_group.command("defer")
@click.argument("candidate_ref")
@click.option("--reason", default=None)
@click.option("--actor-ref", default="user:local", show_default=True)
@click.option("--format", "-f", "output_format", type=click.Choice(["json"]), default=None)
@click.pass_obj
def defer_mark_candidate_command(
    env: AppEnv,
    candidate_ref: str,
    reason: str | None,
    actor_ref: str,
    output_format: str | None,
) -> None:
    """Record a candidate assertion deferral without changing candidate status."""
    _emit_candidate_judgment(
        env,
        candidate_ref=candidate_ref,
        decision="defer",
        reason=reason,
        actor_ref=actor_ref,
        output_format=output_format,
    )


@mark_candidates_group.command("supersede")
@click.argument("candidate_ref")
@click.option("--kind", "replacement_kind", required=True, help="Kind for the replacement active assertion.")
@click.option("--body", "replacement_body_text", required=True, help="Body text for the replacement assertion.")
@click.option("--reason", default=None)
@click.option("--actor-ref", default="user:local", show_default=True)
@click.option("--format", "-f", "output_format", type=click.Choice(["json"]), default=None)
@click.pass_obj
def supersede_mark_candidate_command(
    env: AppEnv,
    candidate_ref: str,
    replacement_kind: str,
    replacement_body_text: str,
    reason: str | None,
    actor_ref: str,
    output_format: str | None,
) -> None:
    """Supersede a candidate with an explicit active assertion."""
    _emit_candidate_judgment(
        env,
        candidate_ref=candidate_ref,
        decision="supersede",
        reason=reason,
        actor_ref=actor_ref,
        output_format=output_format,
        replacement_kind=replacement_kind,
        replacement_body_text=replacement_body_text,
    )


def _emit_candidate_judgment(
    env: AppEnv,
    *,
    candidate_ref: str,
    decision: str,
    reason: str | None,
    actor_ref: str,
    output_format: str | None,
    replacement_kind: str | None = None,
    replacement_body_text: str | None = None,
) -> None:
    payload = run_coroutine_sync(
        env.polylogue.judge_assertion_candidate(
            candidate_ref=candidate_ref,
            decision=decision,
            reason=reason,
            actor_ref=actor_ref,
            replacement_kind=replacement_kind,
            replacement_body_text=replacement_body_text,
        )
    )
    if output_format == "json":
        click.echo(serialize_surface_payload(payload, exclude_none=True))
        return
    result_ref = payload.judgment.resulting_assertion_ref or "no active assertion"
    click.echo(f"{decision}: {payload.candidate.assertion_id} -> {result_ref}")


@click.group("analyze", invoke_without_command=True, no_args_is_help=False)
@click.option("--count", "count_only", is_flag=True, help="Print only the matched-session count.")
@click.option(
    "--by",
    "stats_by",
    type=click.Choice(["origin", "month", "year", "day", "action", "tool", "repo", "work-kind"]),
    default=None,
    help="Group statistics by dimension",
)
@click.option(
    "--facets",
    "show_facets",
    is_flag=True,
    help="Show named facet families for the matched result set; cheap families are default.",
)
@click.option(
    "--cost-outlook",
    is_flag=True,
    help="Project the current billing cycle for a configured subscription plan.",
)
@click.option("--plan", "plan_name", default=None, help="Subscription plan name for --cost-outlook.")
@click.option(
    "--method",
    type=click.Choice(["linear", "trailing-7d-mean", "eom-naive"]),
    default="linear",
    help="Projection method for --cost-outlook.",
)
@click.option(
    "--no-idf",
    "no_idf",
    is_flag=True,
    default=False,
    help="With --facets, skip inverse-document-frequency weighting",
)
@click.option(
    "--include-deferred",
    is_flag=True,
    default=False,
    help="With --facets, compute deferred detail families: repos, roles, material origins, message types, actions, flags.",
)
@click.option(
    "--format",
    "-f",
    "output_format",
    type=click.Choice(["markdown", "json", "ndjson", "html", "obsidian", "org", "yaml", "plaintext", "csv"]),
    default=None,
    help="Output format (ndjson = one JSON document per row, streaming-friendly)",
)
@click.option("--limit", "-l", "-n", type=int, help="Max matched sessions before grouping")
@click.pass_context
def analyze_verb(
    ctx: click.Context,
    count_only: bool = False,
    stats_by: str | None = None,
    show_facets: bool = False,
    cost_outlook: bool = False,
    plan_name: str | None = None,
    method: str = "linear",
    no_idf: bool = False,
    include_deferred: bool = False,
    output_format: str | None = None,
    limit: int | None = None,
) -> None:
    """Analyze matched sessions: statistics, facets, and aggregates.

    \b
    Applies to the full result set by default (no cardinality restriction).
    Wraps aggregate and facet views over the matched session set.

    \b
    Examples:
        polylogue find 'repo:polylogue since:7d' then analyze
        polylogue find 'repo:polylogue since:7d' then analyze --count
        polylogue find 'repo:polylogue since:7d' then analyze --by origin
        polylogue find 'repo:polylogue since:7d' then analyze --by month
        polylogue find 'repo:polylogue' then analyze --facets
        polylogue find 'repo:polylogue' then analyze --by day --format json
        polylogue analyze --cost-outlook --plan claude-pro --format json
        polylogue analyze usage --format json
    """
    if ctx.invoked_subcommand is not None:
        if any(
            (
                count_only,
                stats_by is not None,
                show_facets,
                cost_outlook,
                plan_name is not None,
                method != "linear",
                no_idf,
                include_deferred,
                output_format is not None,
                limit is not None,
            )
        ):
            raise click.UsageError(
                "`analyze` options apply to the aggregate analyze view; "
                "put read-model options after `analyze insights <command>` "
                "or subcommand-specific options after `analyze <command>`."
            )
        return

    import json as _json

    from polylogue.api.sync.bridge import run_coroutine_sync
    from polylogue.cli.shared.cost_rendering import render_outlook_plain
    from polylogue.cost.outlook import ProjectionMethod
    from polylogue.cost.plans import PlanLookupError

    env: AppEnv = ctx.obj
    request = _parent_request(ctx)
    if _explain_terminal_action(
        request,
        action="analyze",
        count=count_only,
        by=stats_by,
        facets=show_facets,
        include_deferred=include_deferred,
        cost_outlook=cost_outlook,
        format=output_format or request.params.get("output_format") or "default",
        limit=limit,
    ):
        return

    selected_modes = sum(bool(flag) for flag in (count_only, show_facets, cost_outlook, stats_by is not None))
    if selected_modes > 1:
        raise click.UsageError(
            "Choose only one analyze mode: --count, --facets, --cost-outlook, --by, or default stats."
        )
    if not cost_outlook and plan_name is not None:
        raise click.UsageError("`analyze --plan` requires --cost-outlook.")
    if not cost_outlook and method != "linear":
        raise click.UsageError("`analyze --method` requires --cost-outlook.")

    if count_only:
        if limit is not None:
            raise click.UsageError("`analyze --count` does not support --limit.")
        updated = request.with_param_updates(count_only=True)
        if output_format:
            updated = updated.with_param_updates(output_format=output_format)
        _execute_query_verb(ctx, updated)
        return

    if limit is not None:
        request = request.with_param_updates(limit=limit)

    if show_facets:
        if output_format not in (None, "json"):
            raise click.UsageError("`analyze --facets` only supports terminal text or --format json")
        # Delegate to the Polylogue facets API using the request's query spec.
        spec = request.query_spec()
        response = run_coroutine_sync(
            env.polylogue.facets(spec, include_idf=not no_idf, include_deferred=include_deferred)
        )
        if output_format == "json":
            click.echo(_json.dumps(response.model_dump(mode="json", by_alias=True), indent=2))
            return
        scope_label = "scoped" if response.scoped_to_query else "global"

        def _status_label(family: str) -> str:
            status = response.family_status.get(family)
            return status.label if status is not None and status.label else family.replace("_", " ").title()

        def _emit_bucket(family: str, values: dict[str, int]) -> None:
            visible_values = {key: count for key, count in values.items() if count}
            if not visible_values:
                return
            click.echo(f"  {_status_label(family)}:")
            for value, cnt in sorted(visible_values.items(), key=lambda kv: (-kv[1], kv[0])):
                click.echo(f"    {value}: {cnt}")

        click.echo(f"Facets ({scope_label}) — matched result set:")
        click.echo(f"  sessions: {response.scoped.total_sessions}  messages: {response.scoped.total_messages}")
        click.echo("  Family states:")
        ordered_families = [
            *response.complete_families,
            *(family for family in response.deferred_families if family not in response.complete_families),
        ]
        for family in ordered_families:
            status = response.family_status.get(family)
            if status is None:
                continue
            detail = f" — {status.state}"
            if status.reason:
                detail += f" ({status.reason}; use --include-deferred)"
            if status.canonicalization and family in {"repos", "role_counts", "material_origins"}:
                detail += f"; {status.canonicalization}"
            click.echo(f"    {family}: {_status_label(family)}{detail}")
        _emit_bucket("origins", response.scoped.origins)
        _emit_bucket("tags", response.scoped.tags)
        _emit_bucket("repos", response.scoped.repos)
        _emit_bucket("role_counts", response.scoped.role_counts)
        _emit_bucket("material_origins", response.scoped.material_origins)
        _emit_bucket("message_types", response.scoped.message_types)
        _emit_bucket("action_types", response.scoped.action_types)
        _emit_bucket("has_flags", response.scoped.has_flags)
        _emit_bucket("omitted", response.scoped.omitted)
        if response.idf:
            click.echo("  IDF (higher = rarer, partitions more strongly):")
            for family, values in response.idf.items():
                click.echo(f"    [{family}]")
                for value, weight in sorted(values.items(), key=lambda kv: (-kv[1], kv[0])):
                    click.echo(f"      {value}: {weight:.3f}")
        return

    if cost_outlook:
        if limit is not None:
            raise click.UsageError("`analyze --cost-outlook` does not support --limit.")
        if plan_name is None:
            raise click.UsageError("`analyze --cost-outlook` requires --plan.")
        projection_method = ProjectionMethod(method)
        try:
            outlook = run_coroutine_sync(env.polylogue.cost_outlook(plan_name, method=projection_method))
        except PlanLookupError as exc:
            raise click.ClickException(str(exc)) from exc
        if outlook is None:
            message = (
                f"No cycle window for plan {plan_name!r}: the plan does not declare "
                "a 'cycle_anchor_day'. Configure one under [[cost.subscription.plans]] "
                "or use a plan with a fixed monthly anchor."
            )
            if output_format == "json":
                click.echo(
                    _json.dumps(
                        {"plan_name": plan_name, "outlook": None, "reason": "no_cycle_anchor"},
                        indent=2,
                    )
                )
                return
            env.ui.console.print(f"[yellow]{message}[/yellow]")
            return
        if output_format == "json":
            click.echo(_json.dumps(outlook.model_dump(mode="json"), indent=2, default=str))
            return
        render_outlook_plain(env, outlook)
        return

    if stats_by:
        updated = request.with_param_updates(stats_only=False, stats_by=stats_by)
        if output_format:
            updated = updated.with_param_updates(output_format=output_format)
        _execute_query_verb(ctx, updated)
        return

    # Default: overall stats for the result set.
    updated = request.with_param_updates(stats_only=True)
    if output_format:
        updated = updated.with_param_updates(output_format=output_format)
    _execute_query_verb(ctx, updated)


def _attach_analyze_subcommands() -> None:
    from polylogue.cli.commands.diagnostics import pace_command, tools_command, turns_command, usage_command
    from polylogue.cli.commands.insights import analyze_insights_command

    analyze_verb.add_command(analyze_insights_command)
    analyze_verb.add_command(pace_command)
    analyze_verb.add_command(tools_command)
    analyze_verb.add_command(turns_command)
    analyze_verb.add_command(usage_command)


_attach_analyze_subcommands()


def _parent_query_terms(ctx: click.Context) -> tuple[str, ...]:
    """Load query terms captured on the parent query context."""
    raw_terms = _require_parent_context(ctx).meta.get("polylogue_query_terms", ())
    return tuple(str(term) for term in raw_terms)


def _explain_terminal_action(request: RootModeRequest, **terminal_action: object) -> bool:
    """Render query explain output for a terminal query action when requested."""
    if not request.explain_query:
        return False
    from polylogue.cli.query import explain_query_request

    explain_query_request(request, terminal_action=terminal_action)
    return True


def _effective_read_output_format(request: RootModeRequest, *, view: str, output_format: str | None) -> str | None:
    effective_format = output_format
    if view == "recovery" and effective_format is None:
        root_format = request.params.get("output_format")
        effective_format = root_format if isinstance(root_format, str) else None
    return effective_format


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


def _spec_is_exact_session_ref(spec: object) -> bool:
    return bool(
        getattr(spec, "session_id", None)
        and not any(
            (
                getattr(spec, "query_terms", ()),
                getattr(spec, "contains_terms", ()),
                getattr(spec, "exclude_text_terms", ()),
                getattr(spec, "similar_text", None),
                getattr(spec, "similar_session_id", None),
            )
        )
    )


def _resolve_target_session_id(request: RootModeRequest) -> str | None:
    """Verb-tree adapter for the shared latest-resolver helper (#1626, #1642)."""
    if request.query_terms:
        from dataclasses import replace

        from polylogue.api.sync.bridge import run_coroutine_sync
        from polylogue.config import Config

        explicit = request.params.get("conv_id")
        if isinstance(explicit, str) and explicit:
            return explicit
        spec = request.query_spec()
        if _spec_is_exact_session_ref(spec):
            return cast("str", spec.session_id)
        if not spec.latest and not spec.has_filters():
            return None
        one_match_spec = replace(spec, limit=1)

        async def _resolve() -> str | None:
            from polylogue.api import Polylogue

            async with Polylogue.open(config=cast("Config | None", request.params.get("_config"))) as api:
                summaries = await one_match_spec.list_summaries(api.config)
            return str(summaries[0].id) if summaries else None

        return run_coroutine_sync(_resolve())

    from polylogue.cli.shared.latest_resolver import resolve_session_id_from_root_params

    return resolve_session_id_from_root_params(dict(request.params))


QUERY_VERBS = (
    select_verb,
    read_verb,
    continue_verb,
    delete_verb,
    mark_verb,
    analyze_verb,
)


__all__ = [
    "QUERY_VERBS",
    "VERB_NAMES",
    "analyze_verb",
    "continue_verb",
    "delete_verb",
    "mark_verb",
    "read_verb",
    "select_verb",
]
