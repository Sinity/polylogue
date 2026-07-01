"""Positional verb subcommands for the query-first CLI.

Each verb takes the filter set from the parent context and executes
a specific action on the matched sessions.
"""

from __future__ import annotations

import json
from collections.abc import Awaitable
from typing import TYPE_CHECKING, Any, TypeVar, cast

import click
from click.shell_completion import CompletionItem

if TYPE_CHECKING:
    from polylogue.cli.root_request import RootModeRequest
    from polylogue.cli.select import SelectPrintField
    from polylogue.surfaces.payloads import FacetsResponse
    from polylogue.surfaces.projection_spec import QueryProjectionSpec

from polylogue.archive.viewport import (
    READ_VIEW_PROFILE_BY_ID,
    READ_VIEW_PROFILES,
    read_view_choices,
    read_view_profile_payloads,
)
from polylogue.cli.read_view_registry import READ_VIEW_HANDLER_METADATA, read_view_option_names
from polylogue.cli.shared.types import AppEnv
from polylogue.cli.verb_names import VERB_NAMES
from polylogue.core.enums import AssertionStatus

_FACET_TERMINAL_BUCKET_LIMIT = 12
_FACET_TERMINAL_IDF_LIMIT = 12
_T = TypeVar("_T")


def run_coroutine_sync(coro: Awaitable[_T]) -> _T:
    """Import the sync bridge only when a verb actually needs archive work."""

    from polylogue.api.sync.bridge import run_coroutine_sync as _run_coroutine_sync

    return _run_coroutine_sync(coro)


def _deliver_content(env: AppEnv, content: str, *, destination: str, out_path: str | None) -> None:
    """Import read-view delivery helpers only on real delivery paths."""

    from polylogue.cli.read_views.base import deliver_content

    deliver_content(env, content, destination=destination, out_path=out_path)


def serialize_surface_payload(payload: Any, *, exclude_none: bool = False) -> str:
    """Serialize surface payloads without importing payload models at module load."""

    from polylogue.surfaces.payloads import serialize_surface_payload as _serialize_surface_payload

    return _serialize_surface_payload(payload, exclude_none=exclude_none)


def read_view_options_for_view(view: str, values: dict[str, object]) -> Any:
    """Build typed read-view options without importing handlers at module load."""

    from polylogue.cli.read_view_handlers import read_view_options_for_view as _read_view_options_for_view

    return _read_view_options_for_view(view, values)


def run_query_set_read_view(*args: Any, **kwargs: Any) -> Any:
    """Execute a query-set read view without importing handlers at module load."""

    from polylogue.cli.read_view_handlers import run_query_set_read_view as _run_query_set_read_view

    return _run_query_set_read_view(*args, **kwargs)


def run_read_view(*args: Any, **kwargs: Any) -> Any:
    """Execute a session read view without importing handlers at module load."""

    from polylogue.cli.read_view_handlers import run_read_view as _run_read_view

    return _run_read_view(*args, **kwargs)


def _read_view_invocation(**kwargs: Any) -> Any:
    """Create a read-view invocation without importing handlers at module load."""

    from polylogue.cli.read_view_handlers import ReadViewInvocation

    return ReadViewInvocation(**kwargs)


def _sorted_nonzero_facet_values(values: dict[str, int]) -> list[tuple[str, int]]:
    """Return deterministic non-zero facet rows for terminal output."""

    return sorted(
        ((key, count) for key, count in values.items() if count),
        key=lambda kv: (-kv[1], kv[0]),
    )


def _remaining_facet_value_line(remaining: int) -> str:
    suffix = "value" if remaining == 1 else "values"
    return f"    … {remaining} more {suffix} omitted from terminal view; use --format json for full buckets."


def _emit_facet_bucket(label: str, values: dict[str, int], *, limit: int = _FACET_TERMINAL_BUCKET_LIMIT) -> None:
    """Emit one bounded facet family for terminal output."""

    rows = _sorted_nonzero_facet_values(values)
    if not rows:
        return
    click.echo(f"  {label}:")
    for value, count in rows[:limit]:
        click.echo(f"    {value}: {count}")
    remaining = len(rows) - limit
    if remaining > 0:
        click.echo(_remaining_facet_value_line(remaining))


def _emit_idf_buckets(idf: dict[str, dict[str, float]], *, limit: int = _FACET_TERMINAL_IDF_LIMIT) -> None:
    """Emit bounded IDF details so noisy archives do not flood terminal output."""

    if not idf:
        return
    click.echo("  IDF (higher = rarer, partitions more strongly):")
    for family, values in idf.items():
        rows = sorted(values.items(), key=lambda kv: (-kv[1], kv[0]))
        if not rows:
            continue
        click.echo(f"    [{family}]")
        for value, weight in rows[:limit]:
            click.echo(f"      {value}: {weight:.3f}")
        remaining = len(rows) - limit
        if remaining > 0:
            suffix = "value" if remaining == 1 else "values"
            click.echo(f"      … {remaining} more {suffix} omitted from terminal view; use --format json for full IDF.")


def emit_facets_response(response: FacetsResponse, *, output_format: str | None) -> None:
    """Emit a facets response in the shared terminal or JSON shape."""

    if output_format == "json":
        click.echo(json.dumps(response.model_dump(mode="json", by_alias=True), indent=2))
        return
    scope_label = "scoped" if response.scoped_to_query else "global"

    def _status_label(family: str) -> str:
        status = response.family_status.get(family)
        return status.label if status is not None and status.label else family.replace("_", " ").title()

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
    _emit_facet_bucket(_status_label("origins"), response.scoped.origins)
    _emit_facet_bucket(_status_label("tags"), response.scoped.tags)
    _emit_facet_bucket(_status_label("repos"), response.scoped.repos)
    _emit_facet_bucket(_status_label("role_counts"), response.scoped.role_counts)
    _emit_facet_bucket(_status_label("material_origins"), response.scoped.material_origins)
    _emit_facet_bucket(_status_label("message_types"), response.scoped.message_types)
    _emit_facet_bucket(_status_label("action_types"), response.scoped.action_types)
    _emit_facet_bucket(_status_label("has_flags"), response.scoped.has_flags)
    _emit_facet_bucket("Omitted/noisy facet counts (not canonical facets)", response.scoped.omitted)
    _emit_idf_buckets(response.idf)


# Deferred imports: RootModeRequest triggers the archive.query.spec →
# operations.archive chain (~780 ms).  Only import it when a verb
# actually executes, never during --help.
def _get_root_request_class() -> object:  # pragma: no cover — returns RootModeRequest
    from polylogue.cli.root_request import RootModeRequest

    return RootModeRequest


def _lazy_shell_complete(source: str):  # type: ignore[no-untyped-def]
    def _complete(ctx: click.Context, param: click.Parameter, incomplete: str):  # type: ignore[no-untyped-def]
        from polylogue.cli.shell_completion_values import complete_query_source

        return complete_query_source(source)(ctx, param, incomplete)  # type: ignore[arg-type]

    _complete.__name__ = f"complete_{source}"
    return _complete


_READ_VIEWS = read_view_choices()
_READ_VIEW_HELP = "What to render (" + ", ".join(_READ_VIEWS) + ")."
_READ_DESTINATIONS = ("terminal", "stdout", "browser", "clipboard", "file")
_READ_FORMATS = tuple(sorted({fmt for profile in READ_VIEW_PROFILES for fmt in profile.formats}))
_READ_RENDER_LAYOUTS = ("standard", "context-image")


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
    related_limit: int,
    project_path: str | None,
    project_repo: str | None,
    since: str | None,
    until: str | None,
    context_origin: str | None,
    context_query: str | None,
    max_sessions: int,
    no_redact: bool,
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
        "related_limit": related_limit,
        "project_path": project_path,
        "project_repo": project_repo,
        "since": since,
        "until": until,
        "context_origin": context_origin,
        "context_query": context_query,
        "max_sessions": max_sessions,
        "no_redact": no_redact,
        "window_hours": window_hours,
        "repo_path": repo_path,
        "since_hours": since_hours,
        "confidence_threshold": confidence_threshold,
        "github_api": github_api,
        "otlp": otlp,
    }


_CONTINUE_CANDIDATE_DEFAULT_LIMIT = 10


def _successor_context_unit_queries(session_id: str) -> tuple[str, ...]:
    """Default successor-context recipe expressed through terminal DSL units."""

    session_clause = f"session.id:{session_id}"
    return (
        f"runs where {session_clause}",
        f"observed-events where {session_clause}",
        f"context-snapshots where {session_clause}",
        f"actions where {session_clause}",
    )


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


def _read_view_projection_contract(view_id: str) -> dict[str, object]:
    """Return the shared projection/render contract summary for a read view."""

    from polylogue.surfaces.projection_spec import projection_from_view

    spec = projection_from_view(view_id)
    return {
        "families": [family.value for family in spec.projection.families],
        "body_policy": spec.projection.body_policy.value,
        "render_layout": spec.render.layout,
        "timestamp_policy": spec.render.timestamps.value,
    }


def _render_read_view_profiles_plain() -> str:
    lines = ["Read views:"]
    for profile in READ_VIEW_PROFILES:
        metadata = READ_VIEW_HANDLER_METADATA[profile.view_id]
        options = ", ".join(f"--{name.replace('_', '-')}" for name in sorted(metadata.accepted_options)) or "none"
        scope = "query-set" if metadata.accepts_query_set else metadata.session_policy
        handoff = " handoff" if profile.successor_handoff else ""
        projection = _read_view_projection_contract(profile.view_id)
        lines.append(
            f"  {profile.view_id:<12} {profile.lossiness:<10} evidence={profile.evidence_policy:<10}"
            f" formats={','.join(profile.formats)}{handoff}"
        )
        lines.append(f"      scope={scope}; options={options}")
        lines.append(
            "      projection="
            f"{','.join(cast(list[str], projection['families']))}; "
            f"body={projection['body_policy']}; "
            f"render={projection['render_layout']}; "
            f"timestamps={projection['timestamp_policy']}"
        )
        lines.append(f"      {profile.purpose}")
    return "\n".join(lines)


def _emit_read_view_profiles(output_format: str | None) -> None:
    if output_format == "json":
        from polylogue.cli.shared.machine_errors import emit_success

        payloads: list[dict[str, object]] = []
        for payload in read_view_profile_payloads():
            metadata = READ_VIEW_HANDLER_METADATA[str(payload["view_id"])]
            augmented: dict[str, object] = dict(payload)
            augmented["cli_options"] = sorted(metadata.accepted_options)
            augmented["session_policy"] = metadata.session_policy
            augmented["accepts_query_set"] = metadata.accepts_query_set
            augmented["projection_contract"] = _read_view_projection_contract(str(payload["view_id"]))
            payloads.append(augmented)
        emit_success({"read_views": payloads})
        return
    if output_format is not None:
        raise click.UsageError("`read --views` only supports terminal text or --format json")
    click.echo(_render_read_view_profiles_plain())


def _read_query_text(request: RootModeRequest) -> str | None:
    if not request.query_terms:
        return None
    return " ".join(str(term) for term in request.query_terms)


def _build_read_projection_spec(
    request: RootModeRequest,
    *,
    views: tuple[str, ...],
    output_format: str | None,
    destination: str,
    out_path: str | None,
    max_tokens: int | None,
    selection_limit: int | None,
    render_layout: str = "standard",
    selection_query: str | None = None,
    selection_origin: str | None = None,
    selection_since: str | None = None,
    selection_until: str | None = None,
    selection_project_path: str | None = None,
    selection_project_repo: str | None = None,
    edge_limit: int | None = None,
    body_limit: int | None = None,
    body_offset: int | None = None,
    neighbor_limit: int | None = None,
    neighbor_window_hours: int | None = None,
    redact_paths: bool = True,
    include_assertions: bool = False,
) -> QueryProjectionSpec:
    """Build the typed selection/projection/render contract for read options."""

    from polylogue.surfaces.projection_spec import projection_from_views

    primary_view = views[0] if views else "summary"
    effective_format = (
        _effective_read_output_format(request, view=primary_view, output_format=output_format) or "markdown"
    )
    query_spec = request.query_spec()
    origin = query_spec.origins[0] if len(query_spec.origins) == 1 else None
    return projection_from_views(
        views,
        format=effective_format,
        destination=destination,
        layout=render_layout,
        max_tokens=max_tokens,
        out=out_path,
        query=selection_query if selection_query is not None else _read_query_text(request),
        origin=selection_origin if selection_origin is not None else origin,
        since=selection_since if selection_since is not None else query_spec.since,
        until=selection_until if selection_until is not None else query_spec.until,
        project_path=selection_project_path,
        project_repo=selection_project_repo,
        limit=selection_limit,
        edge_limit=edge_limit,
        body_limit=body_limit,
        body_offset=body_offset,
        neighbor_limit=neighbor_limit,
        neighbor_window_hours=neighbor_window_hours,
        redact_paths=redact_paths,
        include_assertions=include_assertions,
    )


def _read_projection_limits(
    views: tuple[str, ...],
    limit: int | None,
    offset: int,
) -> tuple[int | None, int | None, int | None, int | None, int | None]:
    """Split read --limit between selection cardinality and projection policy."""

    if len(views) == 1 and views[0] == "chronicle":
        return None, limit, None, None, None
    if len(views) == 1 and views[0] in {"messages", "raw"}:
        return None, None, limit, offset if offset else None, None
    if len(views) == 1 and views[0] == "neighbors":
        return None, None, None, None, limit
    return limit, None, None, None, None


def _read_render_layout(views: tuple[str, ...], *, override: str | None = None) -> str:
    """Return the render layout family for read output."""

    if override is not None:
        return override
    if len(views) > 1 or "context-image" in views:
        return "context-image"
    return "standard"


def _projection_spec_with_resolved_session_refs(
    projection_spec: QueryProjectionSpec | None,
    session_ids: tuple[str, ...],
) -> QueryProjectionSpec | None:
    """Return a projection spec that records resolved archive session refs."""

    if projection_spec is None or not session_ids:
        return projection_spec
    refs = tuple(f"session:{session_id}" for session_id in session_ids)
    selection = projection_spec.selection.model_copy(update={"refs": refs})
    return projection_spec.model_copy(update={"selection": selection})


_READ_HELP_OPTION_GROUPS: tuple[tuple[str, frozenset[str]], ...] = (
    ("Projection", frozenset({"view", "show_views", "show_spec", "render_layout"})),
    ("Delivery and format", frozenset({"destination", "output_format", "out_path", "fields"})),
    ("Cardinality and pagination", frozenset({"all_matches", "first_only", "limit", "offset"})),
    (
        "Context-image projection",
        frozenset(
            {
                "project_path",
                "project_repo",
                "since",
                "until",
                "context_origin",
                "context_query",
                "max_sessions",
                "max_tokens",
                "include_assertions",
                "no_redact",
            }
        ),
    ),
    ("Context and neighbor views", frozenset({"related_limit", "window_hours"})),
    ("Correlation view", frozenset({"repo_path", "since_hours", "confidence_threshold", "github_api", "otlp"})),
)


class _ReadCommand(click.Command):
    """Click command with read-option help grouped by ownership."""

    def format_options(self, ctx: click.Context, formatter: click.HelpFormatter) -> None:
        grouped: dict[str, list[tuple[str, str]]] = {heading: [] for heading, _ in _READ_HELP_OPTION_GROUPS}
        other: list[tuple[str, str]] = []

        for param in self.get_params(ctx):
            record = param.get_help_record(ctx)
            if record is None:
                continue
            target = other
            for heading, names in _READ_HELP_OPTION_GROUPS:
                if param.name in names:
                    target = grouped[heading]
                    break
            target.append(record)

        for heading, _names in _READ_HELP_OPTION_GROUPS:
            if grouped[heading]:
                with formatter.section(heading):
                    formatter.write_dl(grouped[heading])
        if other:
            with formatter.section("Other options"):
                formatter.write_dl(other)


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
    help="Field to print for selected or candidate sessions.",
)
@click.option("--json", "json_output", is_flag=True, help="Print selected or candidate sessions as JSON.")
@click.pass_context
def select_verb(ctx: click.Context, limit: int, print_field: str, json_output: bool) -> None:
    """Select one matched session or print bounded candidate identities."""
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


@click.command("read", cls=_ReadCommand)
@click.option(
    "--view",
    "-v",
    type=str,
    default="summary",
    show_default=True,
    metavar="VIEW[,VIEW...]",
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
    "--json",
    "output_format",
    flag_value="json",
    default=None,
    help="Shortcut for --format json.",
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
@click.option(
    "--render-layout",
    type=click.Choice(_READ_RENDER_LAYOUTS),
    default=None,
    help="Render layout for the composed projection spec; defaults from --view.",
)
@click.option("--out", "out_path", type=click.Path(), default=None, help="File path for --to file.")
@click.option("--all", "all_matches", is_flag=True, help="Read all matched sessions.")
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
@click.option("--project-path", default=None, help="Filter by cwd prefix pattern (--view context-image).")
@click.option("--project-repo", default=None, help="Filter by git repo URL or name (--view context-image).")
@click.option("--since", default=None, help="Start date, ISO 8601 (--view context-image).")
@click.option("--until", default=None, help="End date, ISO 8601 (--view context-image).")
@click.option("--context-origin", "context_origin", default=None, help="Source-origin filter (--view context-image).")
@click.option("--query", "context_query", default=None, help="Free-text query (--view context-image).")
@click.option(
    "--max-sessions", type=int, default=5, show_default=True, help="Max sessions, 1-20 (--view context-image)."
)
@click.option(
    "--max-tokens",
    type=int,
    default=None,
    help="Bound accumulated output to a token budget; over-budget segments are reported as omissions.",
)
@click.option(
    "--include-assertions",
    is_flag=True,
    default=False,
    help="Include context-inject assertion claims in the compiled context image.",
)
@click.option("--no-redact", is_flag=True, default=False, help="Do not redact filesystem paths (--view context-image).")
@click.option("--fields", help="Fields for JSON/YAML outputs (--all).")
@click.option("--views", "show_views", is_flag=True, help="List executable read-view profiles, formats, and options.")
@click.option("--spec", "show_spec", is_flag=True, help="Print the composed selection/projection/render spec as JSON.")
@click.option("--first", "first_only", is_flag=True, help="Read the first matched session only.")
@click.argument("ref", required=False)
@click.pass_context
def read_verb(
    ctx: click.Context,
    view: str,
    destination: str,
    output_format: str | None,
    render_layout: str | None,
    out_path: str | None,
    all_matches: bool,
    limit: int | None,
    offset: int,
    window_hours: int,
    repo_path: str | None,
    since_hours: int,
    confidence_threshold: float,
    github_api: bool,
    otlp: bool,
    related_limit: int,
    project_path: str | None,
    project_repo: str | None,
    since: str | None,
    until: str | None,
    context_origin: str | None,
    context_query: str | None,
    max_sessions: int,
    max_tokens: int | None,
    include_assertions: bool,
    no_redact: bool,
    fields: str | None,
    first_only: bool,
    show_spec: bool = False,
    show_views: bool = False,
    ref: str | None = None,
) -> None:
    """Read matched sessions.

    \b
    Routes to the appropriate renderer based on --view and delivers the
    output to --to (terminal, stdout, browser, clipboard, or file).
    Use --views to inspect which options belong to each read view.

    \b
    Examples:
        polylogue --id abc123 read
        polylogue find id:abc then read --view messages
        polylogue find id:abc then read --view raw --format json
        polylogue find id:abc then read --to browser
        polylogue find 'repo:polylogue has:paste' then read --all --format ndjson
        polylogue find id:abc then read --view context --related-limit 5
        polylogue find 'cost tracking' then read --view context-image --max-sessions 5
        polylogue read --view context-image --project-repo github.com/Sinity/polylogue --since 2026-01-01
        polylogue read --views
        polylogue read --views --format json
        polylogue find 'repo:polylogue' then read --view temporal,chronicle --spec
        polylogue find id:abc then read --view neighbors --window-hours 48
        polylogue --latest read --view neighbors --format json
        polylogue find id:abc then read --view correlation --since-hours 4
        polylogue --latest read --view correlation --otlp --format json
        polylogue read session:abc123 --format json
    """
    env: AppEnv = ctx.obj
    if show_views:
        _emit_read_view_profiles(output_format)
        return
    request = _parent_request(ctx)
    view_tokens = [token.strip() for token in view.split(",") if token.strip()] or ["summary"]
    unknown_views = [token for token in view_tokens if token not in _READ_VIEWS]
    if unknown_views:
        raise click.UsageError(
            f"Unknown read view(s): {', '.join(unknown_views)}. Choose from: {', '.join(_READ_VIEWS)}."
        )
    primary_view = view_tokens[0]
    if all_matches and first_only:
        raise click.UsageError("read --all and --first are mutually exclusive.")
    if destination == "file" and not out_path:
        raise click.UsageError("read --to file requires --out.")
    (
        selection_limit,
        projection_edge_limit,
        projection_body_limit,
        projection_body_offset,
        projection_neighbor_limit,
    ) = _read_projection_limits(tuple(view_tokens), limit, offset)
    projection_neighbor_window_hours = window_hours if len(view_tokens) == 1 and view_tokens[0] == "neighbors" else None
    uses_context_image_selector = "context-image" in view_tokens
    context_image_max_sessions = min(max_sessions, limit) if limit is not None else max_sessions
    spec_selection_limit: int | None
    spec_selection_query: str | None
    spec_selection_origin: str | None
    spec_selection_since: str | None
    spec_selection_until: str | None
    spec_selection_project_path: str | None
    spec_selection_project_repo: str | None
    if uses_context_image_selector:
        spec_selection_limit = context_image_max_sessions
        spec_selection_query = context_query
        spec_selection_origin = context_origin
        spec_selection_since = since
        spec_selection_until = until
        spec_selection_project_path = project_path
        spec_selection_project_repo = project_repo
    else:
        spec_selection_limit = selection_limit
        spec_selection_query = None
        spec_selection_origin = None
        spec_selection_since = None
        spec_selection_until = None
        spec_selection_project_path = None
        spec_selection_project_repo = None
    if show_spec:
        spec = _build_read_projection_spec(
            request,
            views=tuple(view_tokens),
            output_format=output_format,
            destination=destination,
            out_path=out_path,
            max_tokens=max_tokens,
            selection_limit=spec_selection_limit,
            render_layout=_read_render_layout(tuple(view_tokens), override=render_layout),
            selection_query=spec_selection_query,
            selection_origin=spec_selection_origin,
            selection_since=spec_selection_since,
            selection_until=spec_selection_until,
            selection_project_path=spec_selection_project_path,
            selection_project_repo=spec_selection_project_repo,
            edge_limit=projection_edge_limit,
            body_limit=projection_body_limit,
            body_offset=projection_body_offset,
            neighbor_limit=projection_neighbor_limit,
            neighbor_window_hours=projection_neighbor_window_hours,
            redact_paths=not no_redact,
            include_assertions=include_assertions,
        )
        _deliver_content(
            env,
            serialize_surface_payload(spec, exclude_none=True) + "\n",
            destination=destination,
            out_path=out_path,
        )
        return
    if _explain_terminal_action(
        request,
        action="read",
        view=view,
        destination=destination,
        format=_effective_read_output_format(request, view=view, output_format=output_format) or "default",
        all=all_matches,
        first=first_only,
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
    if all_matches:
        if limit is not None:
            request = request.with_param_updates(limit=limit)
        if primary_view == "summary":
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
        projection_spec = _build_read_projection_spec(
            request,
            views=tuple(view_tokens),
            output_format=output_format,
            destination=destination,
            out_path=out_path,
            max_tokens=max_tokens,
            selection_limit=spec_selection_limit,
            render_layout=_read_render_layout(tuple(view_tokens), override=render_layout),
            edge_limit=projection_edge_limit,
            body_limit=projection_body_limit,
            body_offset=projection_body_offset,
            neighbor_limit=projection_neighbor_limit,
            neighbor_window_hours=projection_neighbor_window_hours,
            redact_paths=not no_redact,
            include_assertions=include_assertions,
        )
        run_query_set_read_view(
            env,
            request,
            view=primary_view,
            output_format=output_format,
            fields=fields,
            destination=destination,
            out_path=out_path,
            projection_spec=projection_spec,
        )
        return

    # General context-image path: multi-view composition, token-bounded
    # accumulation, assertion inclusion, and the context-image lens all collapse
    # onto the shared compile_context engine rather than parallel assemblers.
    needs_context_image = (
        len(view_tokens) > 1
        or primary_view == "context-image"
        or (max_tokens is not None and primary_view != "dialogue")
        or include_assertions
    )
    if needs_context_image and destination != "browser":
        projection_spec = _build_read_projection_spec(
            request,
            views=tuple(view_tokens),
            output_format=output_format,
            destination=destination,
            out_path=out_path,
            max_tokens=max_tokens,
            selection_limit=spec_selection_limit if uses_context_image_selector else limit,
            render_layout=_read_render_layout(tuple(view_tokens), override=render_layout),
            selection_query=spec_selection_query,
            selection_origin=spec_selection_origin,
            selection_since=spec_selection_since,
            selection_until=spec_selection_until,
            selection_project_path=spec_selection_project_path,
            selection_project_repo=spec_selection_project_repo,
            redact_paths=not no_redact,
            include_assertions=include_assertions,
        )
        run_read_context_image(
            env,
            request,
            views=tuple(view_tokens),
            max_tokens=max_tokens,
            include_assertions=include_assertions,
            max_sessions=context_image_max_sessions,
            project_path=project_path,
            project_repo=project_repo,
            since=since,
            until=until,
            context_origin=context_origin,
            context_query=context_query,
            no_redact=no_redact,
            output_format=output_format,
            destination=destination,
            out_path=out_path,
            first_only=first_only,
            projection_spec=projection_spec,
        )
        return

    handler_metadata = READ_VIEW_HANDLER_METADATA[primary_view]
    session_id = None
    exact_session_ref = _spec_is_exact_session_ref(request.query_spec())
    if destination == "browser" or first_only or exact_session_ref or not handler_metadata.accepts_query_set:
        session_id = _resolve_query_action_session_id(env, request, operation="read", first_only=first_only)
    effective_format = _effective_read_output_format(request, view=primary_view, output_format=output_format)
    projection_spec = _build_read_projection_spec(
        request,
        views=tuple(view_tokens),
        output_format=output_format,
        destination=destination,
        out_path=out_path,
        max_tokens=max_tokens,
        selection_limit=selection_limit,
        render_layout=_read_render_layout(tuple(view_tokens), override=render_layout),
        edge_limit=projection_edge_limit,
        body_limit=projection_body_limit,
        body_offset=projection_body_offset,
        neighbor_limit=projection_neighbor_limit,
        neighbor_window_hours=projection_neighbor_window_hours,
    )
    explicit_options = _explicit_read_view_options(ctx)
    if primary_view == "dialogue" and session_id is None:
        if limit is not None:
            request = request.with_param_updates(limit=limit)
        run_query_set_read_view(
            env,
            request,
            view=primary_view,
            output_format=effective_format,
            fields=fields,
            destination=destination,
            out_path=out_path,
            projection_spec=projection_spec,
        )
        return
    if destination == "browser":
        run_read_view(
            env,
            request,
            _read_view_invocation(
                view="summary",
                session_id=session_id,
                output_format=effective_format,
                destination=destination,
                out_path=out_path,
                explicit_options=explicit_options,
                projection_spec=projection_spec,
            ),
        )
        return

    run_read_view(
        env,
        request,
        _read_view_invocation(
            view=primary_view,
            session_id=session_id,
            output_format=effective_format,
            destination=destination,
            out_path=out_path,
            options=read_view_options_for_view(
                primary_view,
                _read_view_option_values(
                    limit=limit,
                    offset=offset,
                    related_limit=related_limit,
                    project_path=project_path,
                    project_repo=project_repo,
                    since=since,
                    until=until,
                    context_origin=context_origin,
                    context_query=context_query,
                    max_sessions=max_sessions,
                    no_redact=no_redact,
                    window_hours=window_hours,
                    repo_path=repo_path,
                    since_hours=since_hours,
                    confidence_threshold=confidence_threshold,
                    github_api=github_api,
                    otlp=otlp,
                ),
            ),
            explicit_options=explicit_options,
            projection_spec=projection_spec,
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
    "--json",
    "output_format",
    flag_value="json",
    default=None,
    help="Shortcut for --format json.",
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
    session_id = _resolve_query_action_session_id(env, request, operation="continue")
    if session_id is None:
        raise click.UsageError("continue requires one matched session (use --id, --latest, or a narrowing query).")
    session = run_coroutine_sync(env.polylogue.get_session(session_id))
    if session is None:
        raise click.UsageError(f"Session not found: {session_id}")
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
                    read_views=("messages",),
                    unit_queries=_successor_context_unit_queries(session_id),
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
    if effective_format not in (None, "markdown"):
        raise click.UsageError("continue supports markdown output by default or --format json.")
    from polylogue.context.compiler import ContextSpec

    image = run_coroutine_sync(
        env.polylogue.compile_context(
            ContextSpec(
                purpose="continue",
                seed_refs=(f"session:{session_id}",),
                read_views=("messages",),
                unit_queries=_successor_context_unit_queries(session_id),
            )
        )
    )
    _deliver_content(env, _render_context_image_markdown(image), destination=destination, out_path=out_path)


@click.command("delete")
@click.option("--dry-run", is_flag=True, help="Preview what would be deleted without deleting")
@click.option("--yes", "yes_flag", is_flag=True, help="Confirm the deletion (required for actual deletion)")
@click.option("--all", "all_flag", is_flag=True, help="Delete all matched sessions (required when multiple match)")
@click.option(
    "--json",
    "output_format",
    flag_value="json",
    default=None,
    help="Shortcut for --format json.",
)
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["json"]),
    default=None,
    help="Output format. JSON emits a MutationResultPayload.",
)
@click.pass_context
def delete_verb(
    ctx: click.Context,
    dry_run: bool,
    yes_flag: bool,
    all_flag: bool,
    output_format: str | None,
) -> None:
    """Delete matched sessions.

    \b
    Cardinality rules:
      --dry-run       Preview what would be deleted (no confirmation needed).
      --yes           Confirm deletion for a single matched session.
      --yes --all     Required when the query matches more than one session.

    \b
    Examples:
        polylogue find id:abc then delete --dry-run
        polylogue find id:abc then delete --dry-run --format json
        polylogue find id:abc then delete --yes
        polylogue find 'repo:polylogue since:7d' then delete --dry-run --all
        polylogue find 'repo:polylogue since:7d' then delete --yes --all
    """
    from polylogue.cli.verb_cardinality import (
        CardinalityError,
        check_cardinality,
        probe_session_ids_for_verb,
        resolve_session_ids_for_verb,
    )

    env: AppEnv = ctx.obj
    request = _parent_request(ctx)
    if _explain_terminal_action(
        request,
        action="delete",
        dry_run=dry_run,
        yes=yes_flag,
        all=all_flag,
        format=output_format or request.params.get("output_format") or "default",
    ):
        return

    from polylogue.cli.archive_query import execute_delete_by_session_ids

    # dry-run: require explicit multi-target scope before materializing a broad
    # preview. Once --all is supplied, resolve the SAME full ID set the real
    # delete uses rather than re-running the query through _execute_query_verb,
    # which caps at the default limit of 20 and would preview fewer sessions
    # than --yes --all actually deletes (#1873).
    if dry_run:
        probe_ids = probe_session_ids_for_verb(env, request, limit=2)
        if len(probe_ids) > 1 and not all_flag:
            raise click.UsageError(
                "'delete dry-run' matched multiple sessions. "
                "Use --all to preview every matched session, or narrow the query."
            )
        session_ids = resolve_session_ids_for_verb(env, request)
        execute_delete_by_session_ids(env, session_ids, force=True, dry_run=True)
        return
    if not yes_flag:
        raise click.UsageError("delete requires --yes for actual deletion; use --dry-run to preview.")

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
@click.option(
    "--json",
    "output_format",
    flag_value="json",
    default=None,
    help="Shortcut for --format json.",
)
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["json"]),
    default=None,
    help="Output format. JSON emits a MutationResultPayload.",
)
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
    output_format: str | None,
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
        format=output_format or "default",
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
    if output_format == "json":
        from polylogue.surfaces.payloads import MutationResultPayload

        click.echo(
            MutationResultPayload(
                status="ok",
                operation="mutate",
                session_count=count,
                affected_count=count if ops else 0,
                session_ids=tuple(target_ids),
            ).to_json(exclude_none=True)
        )
    elif ops:
        click.echo(f"Marked {count} session(s): {'; '.join(ops)}")
    else:
        click.echo("No mark operations specified.")


@mark_verb.group("candidates")
def mark_candidates_group() -> None:
    """Review assertion candidates for a selected session or target ref.

    Candidate commands own review state and claim judgment: review, list,
    accept, reject, defer, or supersede candidate assertions. They do not apply ordinary session tags,
    stars, pins, archive marks, or notes.
    """


@mark_candidates_group.command("list")
@click.option("--target-ref", default=None, help="Limit candidates to one target object ref.")
@click.option("--limit", "-l", type=int, default=50, show_default=True)
@click.option("--json", "output_format", flag_value="json", default=None, help="Shortcut for --format json.")
@click.option("--format", "-f", "output_format", type=click.Choice(["json"]), default=None)
@click.pass_obj
def list_mark_candidates_command(env: AppEnv, target_ref: str | None, limit: int, output_format: str | None) -> None:
    """List candidate assertion claims."""
    items = run_coroutine_sync(env.polylogue.list_assertion_candidates(target_ref=target_ref, limit=limit))
    if output_format == "json":
        from polylogue.surfaces.payloads import AssertionClaimListPayload

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


@mark_candidates_group.command("review")
@click.option("--target-ref", default=None, help="Limit candidate review rows to one target object ref.")
@click.option("--limit", "-l", type=int, default=50, show_default=True)
@click.option("--json", "output_format", flag_value="json", default=None, help="Shortcut for --format json.")
@click.option("--format", "-f", "output_format", type=click.Choice(["json"]), default=None)
@click.pass_obj
def review_mark_candidates_command(env: AppEnv, target_ref: str | None, limit: int, output_format: str | None) -> None:
    """List candidate assertion review state and disabled actions."""
    from polylogue.surfaces.payloads import AssertionCandidateReviewListPayload

    payload = run_coroutine_sync(env.polylogue.list_assertion_candidate_reviews(target_ref=target_ref, limit=limit))
    if not isinstance(payload, AssertionCandidateReviewListPayload):
        payload = AssertionCandidateReviewListPayload.from_envelopes(
            payload,
            limit=limit,
            target_ref=target_ref,
        )
    if output_format == "json":
        click.echo(serialize_surface_payload(payload, exclude_none=True))
        return
    if not payload.items:
        click.echo("No assertion candidate review rows found.")
        return
    for item in payload.items:
        disabled = sorted(
            {
                action.availability.disabled_reason
                for action in item.action_affordances
                if action.availability.disabled_reason
            }
        )
        detail = item.candidate.body_text or (
            json.dumps(item.candidate.value, sort_keys=True) if item.candidate.value is not None else ""
        )
        suffix = f" disabled={','.join(disabled)}" if disabled else ""
        click.echo(
            f"{item.candidate.assertion_id:<32} {item.review_status:<10} "
            f"{item.candidate.kind:<20} {item.candidate.target_ref} {detail}{suffix}"
        )


@mark_candidates_group.command("accept")
@click.argument("candidate_ref")
@click.option("--reason", default=None)
@click.option("--actor-ref", default="user:local", show_default=True)
@click.option("--json", "output_format", flag_value="json", default=None, help="Shortcut for --format json.")
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
@click.option("--json", "output_format", flag_value="json", default=None, help="Shortcut for --format json.")
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
@click.option("--json", "output_format", flag_value="json", default=None, help="Shortcut for --format json.")
@click.option("--format", "-f", "output_format", type=click.Choice(["json"]), default=None)
@click.pass_obj
def defer_mark_candidate_command(
    env: AppEnv,
    candidate_ref: str,
    reason: str | None,
    actor_ref: str,
    output_format: str | None,
) -> None:
    """Record a durable candidate assertion deferral."""
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
@click.option("--json", "output_format", flag_value="json", default=None, help="Shortcut for --format json.")
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
@click.option(
    "--postmortem",
    "show_postmortem",
    is_flag=True,
    help="Distilled postmortem bundle over the matched session scope (#2380).",
)
@click.option(
    "--portfolio",
    "show_portfolio",
    is_flag=True,
    help="Corpus-wide sanitized portfolio report over the matched scope (#2437).",
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
    "--json",
    "output_format",
    flag_value="json",
    default=None,
    help="Shortcut for --format json.",
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
    show_postmortem: bool = False,
    show_portfolio: bool = False,
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
                show_postmortem,
                show_portfolio,
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

    selected_modes = sum(
        bool(flag)
        for flag in (count_only, show_facets, cost_outlook, show_postmortem, show_portfolio, stats_by is not None)
    )
    if selected_modes > 1:
        raise click.UsageError(
            "Choose only one analyze mode: --count, --facets, --cost-outlook, "
            "--postmortem, --portfolio, --by, or default stats."
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

    if show_postmortem:
        if output_format not in (None, "json", "markdown", "plaintext"):
            raise click.UsageError("`analyze --postmortem` only supports terminal text, --format json, or markdown")
        from polylogue.cli.shared.machine_errors import success
        from polylogue.insights.postmortem import render_postmortem_markdown, render_postmortem_plain
        from polylogue.surfaces.payloads import model_json_document

        spec = request.query_spec()
        bundle = run_coroutine_sync(env.polylogue.postmortem_bundle(spec, limit=limit))
        if output_format == "json":
            click.echo(success({"postmortem": model_json_document(bundle, exclude_none=True)}).to_json())
        elif output_format == "markdown":
            click.echo(render_postmortem_markdown(bundle))
        else:
            click.echo(render_postmortem_plain(bundle))
        return

    if show_portfolio:
        if output_format not in (None, "json", "markdown", "plaintext"):
            raise click.UsageError("`analyze --portfolio` only supports terminal text, --format json, or markdown")
        from polylogue.cli.shared.machine_errors import success
        from polylogue.insights.portfolio import render_portfolio_markdown, render_portfolio_plain
        from polylogue.surfaces.payloads import model_json_document

        spec = request.query_spec()
        portfolio = run_coroutine_sync(env.polylogue.portfolio_bundle(spec, limit=limit))
        if output_format == "json":
            payload = model_json_document(portfolio, exclude_none=True)
            rendered_json = success({"portfolio": payload}).to_json()
            click.echo(rendered_json)
            return
        rendered = (
            render_portfolio_markdown(portfolio) if output_format == "markdown" else render_portfolio_plain(portfolio)
        )
        click.echo(rendered)
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
        emit_facets_response(response, output_format=output_format)
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
    from polylogue.cli.click_command_registration import _LazyCommand, _LazyGroup

    analyze_verb.add_command(
        _LazyGroup(
            "insights",
            "polylogue.cli.commands.insights",
            "analyze_insights_command",
            short_help="Check and export derived insight materialization.",
        )
    )
    for name, attr, help_text in (
        ("pace", "pace_command", "Analyze session pacing, gaps, and burstiness."),
        ("tools", "tools_command", "Analyze tool usage across sessions."),
        ("turns", "turns_command", "Analyze turn structure for one session."),
        ("usage", "usage_command", "Analyze provider usage events."),
    ):
        analyze_verb.add_command(_LazyCommand(name, "polylogue.cli.commands.diagnostics", attr, short_help=help_text))


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
    return output_format


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


def _resolve_query_action_session_id(
    env: AppEnv,
    request: RootModeRequest,
    *,
    operation: str,
    first_only: bool = False,
) -> str | None:
    """Resolve one query-action session with explicit ranked-result cardinality."""
    if request.query_terms:
        from dataclasses import replace

        from polylogue.cli.verb_cardinality import CardinalityError, check_cardinality

        explicit = request.params.get("conv_id")
        if isinstance(explicit, str) and explicit:
            return explicit
        spec = request.query_spec()
        if _spec_is_exact_session_ref(spec):
            return cast("str", spec.session_id)
        if not spec.latest and not spec.has_filters():
            return None

        resolve_limit = 1 if first_only else 2
        bounded_spec = replace(spec, limit=resolve_limit)

        async def _resolve() -> list[str]:
            summaries = await bounded_spec.list_summaries(env.config)
            return [str(summary.id) for summary in summaries]

        session_ids = run_coroutine_sync(_resolve())
        multi_match_hint = "Narrow the query to one session or run select first." if operation == "continue" else None
        try:
            if multi_match_hint is not None:
                check_cardinality(
                    len(session_ids),
                    allow_all=False,
                    first_only=first_only,
                    operation=operation,
                    multi_match_hint=multi_match_hint,
                )
            else:
                check_cardinality(
                    len(session_ids),
                    allow_all=False,
                    first_only=first_only,
                    operation=operation,
                )
        except CardinalityError as exc:
            raise click.UsageError(str(exc)) from exc
        return session_ids[0] if session_ids else None

    return _resolve_target_session_id(request)


def _resolve_query_action_session_ids(
    env: AppEnv,
    request: RootModeRequest,
    *,
    limit: int,
    first_only: bool = False,
) -> list[str]:
    """Resolve up to ``limit`` matched sessions for a multi-session read action.

    Honors the find selection (query terms, filters, ``--id``, ``--latest``)
    rather than discarding it. This is the multi-session counterpart to
    :func:`_resolve_query_action_session_id`.
    """
    if request.query_terms:
        from dataclasses import replace

        explicit = request.params.get("conv_id")
        if isinstance(explicit, str) and explicit:
            return [explicit]
        spec = request.query_spec()
        if _spec_is_exact_session_ref(spec):
            return [cast("str", spec.session_id)]
        if not spec.latest and not spec.has_filters():
            return []
        bounded_spec = replace(spec, limit=1 if first_only else limit)

        async def _resolve() -> list[str]:
            summaries = await bounded_spec.list_summaries(env.config)
            return [str(summary.id) for summary in summaries]

        return run_coroutine_sync(_resolve())

    single = _resolve_target_session_id(request)
    return [single] if single else []


def _render_context_image_markdown(image: object) -> str:
    """Render a compiled ContextImage to markdown for terminal/file delivery."""
    spec = getattr(image, "spec", None)
    segments = tuple(getattr(image, "segments", ()))
    omitted = tuple(getattr(image, "omitted", ()))
    caveats = tuple(getattr(image, "caveats", ()))
    purpose = getattr(spec, "purpose", None) or "context"
    read_views = tuple(getattr(spec, "read_views", ()) or ())
    projection_spec = getattr(image, "projection_spec", None)
    token_estimate = getattr(image, "token_estimate", None)
    lines: list[str] = [
        "# Context Image",
        "",
        f"- Purpose: {purpose}",
        f"- Views: {', '.join(str(view) for view in read_views) or 'none'}",
        f"- Segments: {len(segments)}",
        f"- Omissions: {len(omitted)}",
    ]
    if token_estimate is not None:
        lines.append(f"- Token estimate: {token_estimate}")
    if projection_spec is not None:
        selection = getattr(projection_spec, "selection", None)
        projection = getattr(projection_spec, "projection", None)
        render = getattr(projection_spec, "render", None)
        query = getattr(selection, "query", None)
        refs = tuple(getattr(selection, "refs", ()) or ())
        selection_origin = getattr(selection, "origin", None)
        selection_since = getattr(selection, "since", None)
        selection_until = getattr(selection, "until", None)
        selection_project_path = getattr(selection, "project_path", None)
        selection_project_repo = getattr(selection, "project_repo", None)
        selection_limit = getattr(selection, "limit", None)
        families = tuple(getattr(projection, "families", ()) or ())
        body_policy = getattr(projection, "body_policy", None)
        max_tokens = getattr(projection, "max_tokens", None)
        edge_limit = getattr(projection, "edge_limit", None)
        body_limit = getattr(projection, "body_limit", None)
        body_offset = getattr(projection, "body_offset", None)
        neighbor_limit = getattr(projection, "neighbor_limit", None)
        neighbor_window_hours = getattr(projection, "neighbor_window_hours", None)
        redact_paths = getattr(projection, "redact_paths", None)
        render_format = getattr(render, "format", None)
        render_destination = getattr(render, "destination", None)
        render_layout = getattr(render, "layout", None)
        render_timestamps = getattr(render, "timestamps", None)
        if query:
            lines.append(f"- Selection query: {query}")
        if selection_origin:
            lines.append(f"- Selection origin: {selection_origin}")
        if selection_since:
            lines.append(f"- Selection since: {selection_since}")
        if selection_until:
            lines.append(f"- Selection until: {selection_until}")
        if selection_project_path:
            lines.append(f"- Selection project path: {selection_project_path}")
        if selection_project_repo:
            lines.append(f"- Selection project repo: {selection_project_repo}")
        if selection_limit is not None:
            lines.append(f"- Selection limit: {selection_limit}")
        if refs:
            shown_refs = refs[:3]
            refs_text = ", ".join(str(ref) for ref in shown_refs)
            remaining = len(refs) - len(shown_refs)
            if remaining:
                refs_text = f"{refs_text}, ... {remaining} more"
            lines.append(f"- Selection refs: {refs_text}")
        if families:
            lines.append(f"- Projection families: {', '.join(str(family.value) for family in families)}")
        if body_policy is not None:
            lines.append(f"- Body policy: {body_policy.value}")
        if max_tokens is not None:
            lines.append(f"- Projection max tokens: {max_tokens}")
        if edge_limit is not None:
            lines.append(f"- Projection edge limit: {edge_limit}")
        if body_limit is not None:
            lines.append(f"- Projection body limit: {body_limit}")
        if body_offset is not None:
            lines.append(f"- Projection body offset: {body_offset}")
        if neighbor_limit is not None:
            lines.append(f"- Projection neighbor limit: {neighbor_limit}")
        if neighbor_window_hours is not None:
            lines.append(f"- Projection neighbor window hours: {neighbor_window_hours}")
        if redact_paths is not None:
            lines.append(f"- Projection redact paths: {str(bool(redact_paths)).lower()}")
        if render_format is not None and render_destination is not None:
            lines.append(f"- Render: {render_format.value} to {render_destination.value}")
        if render_layout:
            lines.append(f"- Render layout: {render_layout}")
        if render_timestamps is not None:
            lines.append(f"- Render timestamps: {render_timestamps.value}")
    if caveats:
        lines.append(f"- Caveats: {', '.join(str(caveat) for caveat in caveats)}")
    parts: list[str] = ["\n".join(lines)]
    for index, segment in enumerate(segments, start=1):
        markdown = getattr(segment, "markdown", None)
        if markdown:
            title = getattr(segment, "title", None) or getattr(segment, "payload_kind", None) or f"Segment {index}"
            payload_kind = getattr(segment, "payload_kind", None)
            token_count = getattr(segment, "token_estimate", None)
            segment_lines = [f"## {index}. {title}", ""]
            meta: list[str] = []
            if payload_kind:
                meta.append(f"kind={payload_kind}")
            if token_count is not None:
                meta.append(f"tokens={token_count}")
            if meta:
                segment_lines.extend([f"_({'; '.join(meta)})_", ""])
            segment_lines.append(markdown.rstrip())
            parts.append("\n".join(segment_lines))
    if omitted:
        parts.append("## Omitted")
        for omission in omitted:
            target = omission.ref or omission.query or omission.view or "?"
            parts.append(f"- {target} [{omission.reason}]: {omission.detail}")
    return "\n\n".join(parts).rstrip() + "\n"


def run_read_context_image(
    env: AppEnv,
    request: RootModeRequest,
    *,
    views: tuple[str, ...],
    max_tokens: int | None,
    include_assertions: bool,
    max_sessions: int,
    project_path: str | None,
    project_repo: str | None,
    since: str | None,
    until: str | None,
    context_origin: str | None,
    context_query: str | None,
    no_redact: bool,
    output_format: str | None,
    destination: str,
    out_path: str | None,
    first_only: bool,
    projection_spec: QueryProjectionSpec | None = None,
) -> None:
    """Compile and emit a bounded context image over the matched selection.

    This is the single general read path for multi-view composition, token
    budgeting, assertion inclusion, and the context-image lens. It delegates to
    ``compile_context`` (seed refs from the resolved selection) or, when only
    context-image filters narrow the set, to ``context_image_payload``.
    """
    from polylogue.context.compiler import (
        DEFAULT_CONTEXT_IMAGE_MAX_CHARS_PER_MESSAGE,
        DEFAULT_CONTEXT_IMAGE_MAX_MESSAGES_PER_SESSION,
        ContextSpec,
    )

    poly = env.polylogue
    redact = not no_redact
    limit = max(1, min(max_sessions, 20))
    session_ids = _resolve_query_action_session_ids(env, request, limit=limit, first_only=first_only)
    projection_spec = _projection_spec_with_resolved_session_refs(projection_spec, tuple(session_ids))

    # compile_context handles read views and query-unit context; the context-image
    # token maps to the message transcript view. Other tokens become honest
    # "unsupported" omissions inside compile_context rather than silent drops.
    compile_views = tuple("messages" if token == "context-image" else token for token in views)
    uses_context_image_defaults = "context-image" in views

    if session_ids:
        spec = ContextSpec(
            purpose="continue",
            seed_refs=tuple(f"session:{session_id}" for session_id in session_ids),
            read_views=compile_views,
            max_tokens=max_tokens,
            max_messages_per_session=(
                DEFAULT_CONTEXT_IMAGE_MAX_MESSAGES_PER_SESSION if uses_context_image_defaults else None
            ),
            max_chars_per_message=(
                DEFAULT_CONTEXT_IMAGE_MAX_CHARS_PER_MESSAGE if uses_context_image_defaults else None
            ),
            include_assertions=include_assertions,
            redaction_policy="raw-opt-in" if not redact else "default",
        )
        image = run_coroutine_sync(poly.compile_context(spec))
    elif "context-image" in views:
        image = run_coroutine_sync(
            poly.context_image_payload(
                project_path=project_path,
                project_repo=project_repo,
                since=since,
                until=until,
                origin=context_origin,
                query=context_query,
                max_sessions=limit,
                max_tokens=max_tokens,
                include_messages="messages" in compile_views,
                include_assertions=include_assertions,
                redact_paths=redact,
            )
        )
    else:
        raise click.UsageError(
            "read with --max-tokens, --include-assertions, or multiple --view values "
            "requires a seed (use --id, --latest, id:prefix, or a query)."
        )

    if projection_spec is not None:
        image = image.model_copy(update={"projection_spec": projection_spec})

    if output_format == "json":
        content = serialize_surface_payload(image, exclude_none=True) + "\n"
    else:
        content = _render_context_image_markdown(image)
    _deliver_content(env, content, destination=destination, out_path=out_path)


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
