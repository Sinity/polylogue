"""Run command and pipeline stage subcommands."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import click

from polylogue.cli.helpers import (
    complete_run_source_names,
    fail,
    maybe_prompt_sources,
    resolve_sources,
)
from polylogue.cli.run_workflow import (
    WatchDisplayObserver as _WatchDisplayObserver,
)
from polylogue.cli.run_workflow import (
    WatchStatusObserver as _WatchStatusObserver,
)
from polylogue.cli.run_workflow import (
    display_result as _display_result,
)
from polylogue.cli.run_workflow import (
    handle_drive_error,
    render_preview_summary,
)
from polylogue.cli.run_workflow import (
    run_sync_once as _run_sync_once,
)
from polylogue.cli.run_workflow import (
    run_with_progress as _run_with_progress,
)
from polylogue.cli.types import AppEnv
from polylogue.pipeline.observers import (
    CompositeObserver,
    ExecObserver,
    NotificationObserver,
    RunObserver,
    WebhookObserver,
)
from polylogue.pipeline.run_support import RUN_STAGE_CHOICES, expand_requested_stage, normalize_stage_sequence
from polylogue.pipeline.runner import plan_sources
from polylogue.sources import DriveError
from polylogue.storage.state_views import RunResult
from polylogue.sync_bridge import run_coroutine_sync

INTERACTIVE_RUN_STAGE_CHOICES: tuple[str, ...] = (
    "all",
    "reprocess",
    "acquire",
    "schema",
    "parse",
    "materialize",
    "render",
    "site",
    "index",
)


@dataclass(frozen=True, slots=True)
class EmbedOptions:
    """Options specific to the embed stage."""

    conversation: str | None = None
    model: str = "voyage-4"
    rebuild: bool = False
    stats: bool = False
    json_output: bool = False
    limit: int | None = None


@dataclass(frozen=True, slots=True)
class RunStageRequest:
    name: str
    stage_sequence: tuple[str, ...]
    render_format: str | None = None
    site_options: dict[str, Any] | None = None
    embed_options: EmbedOptions | None = None


def maybe_prompt_run_stage(
    env: AppEnv,
    *,
    stage: str,
    prompt: bool,
) -> str:
    """Prompt for a workflow stage when the user did not choose one explicitly."""
    if not prompt or env.ui.plain:
        return stage
    choice = env.ui.choose("Select workflow for run", list(INTERACTIVE_RUN_STAGE_CHOICES))
    if not choice:
        fail("run", "No workflow selected")
    return choice


def _make_stage_request(
    name: str,
    *,
    render_format: str | None = None,
    site_options: dict[str, Any] | None = None,
) -> RunStageRequest:
    return RunStageRequest(
        name=name,
        stage_sequence=expand_requested_stage(name),
        render_format=render_format,
        site_options=site_options,
    )


def _flatten_stage_requests(stage_requests: list[RunStageRequest]) -> tuple[str, ...]:
    return tuple(stage_name for request in stage_requests for stage_name in request.stage_sequence)


def _resolve_canonical_stage(stage_requests: list[RunStageRequest]) -> str:
    if not stage_requests:
        return "all"
    flattened = _flatten_stage_requests(stage_requests)
    if len(stage_requests) == 1 and flattened == stage_requests[0].stage_sequence:
        return stage_requests[0].name
    for stage_name in RUN_STAGE_CHOICES:
        if flattened == expand_requested_stage(stage_name):
            return stage_name
    return "all"


def _display_stage_label(stage_requests: list[RunStageRequest], canonical_stage: str) -> str:
    if not stage_requests:
        return canonical_stage
    if len(stage_requests) == 1:
        return stage_requests[0].name
    return " -> ".join(request.name for request in stage_requests)


def _resolve_render_format(stage_requests: list[RunStageRequest]) -> str:
    render_formats = {request.render_format for request in stage_requests if request.render_format is not None}
    if not render_formats:
        return "html"
    if len(render_formats) > 1:
        fail("run", f"Conflicting render formats requested: {', '.join(sorted(render_formats))}")
    return next(iter(render_formats))


def _resolve_embed_options(stage_requests: list[RunStageRequest]) -> EmbedOptions | None:
    options = [request.embed_options for request in stage_requests if request.embed_options is not None]
    if not options:
        return None
    if len(options) > 1:
        fail("run", "Multiple embed stage requests with different options")
    return options[0]


def _resolve_site_options(stage_requests: list[RunStageRequest]) -> dict[str, Any] | None:
    options = [request.site_options for request in stage_requests if request.site_options is not None]
    if not options:
        return None
    if len(options) > 1:
        fail("run", "Multiple site stage requests with different options")
    return options[0]


@click.group("run", chain=True, invoke_without_command=True)
@click.option("--preview", is_flag=True, help="Preview work without writing")
@click.option(
    "--source",
    "sources",
    multiple=True,
    shell_complete=complete_run_source_names,
    help="Configured source name (repeatable). Accepts 'last' for the previously synced source.",
)
@click.option("--watch", is_flag=True, help="Watch sources for changes and run continuously")
@click.option("--notify", is_flag=True, help="Desktop notification on conversation changes (requires --watch)")
@click.option("--exec", "exec_cmd", help="Execute command on conversation changes (requires --watch)")
@click.option("--webhook", help="Call webhook URL on conversation changes (requires --watch)")
@click.option("--reparse", is_flag=True, help="Force re-parsing of all raw conversations (clears parse tracking)")
@click.pass_context
def run_command(
    ctx: click.Context,
    preview: bool,
    sources: tuple[str, ...],
    watch: bool,
    notify: bool,
    exec_cmd: str | None,
    webhook: str | None,
    reparse: bool,
) -> None:
    """Run pipeline stages on configured sources."""


@run_command.result_callback()
@click.pass_context
def _run_result_callback(
    ctx: click.Context,
    stage_requests: list[RunStageRequest],
    preview: bool,
    sources: tuple[str, ...],
    watch: bool,
    notify: bool,
    exec_cmd: str | None,
    webhook: str | None,
    reparse: bool,
) -> None:
    env: AppEnv = ctx.obj
    if (notify or exec_cmd or webhook) and not watch:
        fail("run", "--notify, --exec, and --webhook require --watch mode")

    if not stage_requests:
        prompted_stage = maybe_prompt_run_stage(
            env,
            stage="all",
            prompt=not env.ui.plain,
        )
        stage_requests = [_make_stage_request(prompted_stage)]

    display_stage = _display_stage_label(stage_requests, _resolve_canonical_stage(stage_requests))
    canonical_stage = _resolve_canonical_stage(stage_requests)
    try:
        stage_sequence = normalize_stage_sequence(
            stage=canonical_stage,
            stage_sequence=_flatten_stage_requests(stage_requests),
        )
    except ValueError as exc:
        fail("run", str(exc))
    render_format = _resolve_render_format(stage_requests)
    site_options = _resolve_site_options(stage_requests)
    embed_options = _resolve_embed_options(stage_requests)

    # Embed is a standalone stage — handle it directly, outside the
    # normal pipeline flow.  When embed-only, return immediately.
    # When chained with other stages, run embed first, then strip it
    # from the sequence so the pipeline sees only real pipeline stages.
    if embed_options is not None:
        _run_embed_standalone(ctx.obj, embed_options)
        remaining = tuple(s for s in stage_sequence if s != "embed")
        if not remaining:
            return
        stage_sequence = remaining

    cfg = env.config
    selected_sources = resolve_sources(cfg, sources, "run")
    selected_sources = maybe_prompt_sources(env, cfg, selected_sources, "run")

    if reparse:
        reset_count = run_coroutine_sync(env.repository.reset_parse_status())
        click.echo(f"Reset parse status for {reset_count:,} raw records.", err=False)

    plan_snapshot = None
    if preview:
        try:
            from polylogue.cli.run_observers import progress_observer

            with progress_observer(
                env,
                initial_desc="Planning preview...",
                plain_banner="Planning preview...",
            ) as progress_observer_handle:
                plan_snapshot = plan_sources(
                    cfg,
                    stage=canonical_stage,
                    stage_sequence=stage_sequence,
                    ui=env.ui,
                    source_names=selected_sources,
                    backend=env.backend,
                    progress_callback=progress_observer_handle.on_progress,
                )
        except DriveError as exc:
            handle_drive_error(exc)
        render_preview_summary(
            env,
            selected_sources=selected_sources,
            plan_snapshot=plan_snapshot,
        )
        if env.ui.plain or not env.ui.confirm("Sync now using this snapshot?", default=False):
            return

    if not watch and not plan_snapshot and not env.ui.plain and not env.ui.confirm("Proceed?", default=True):
        env.ui.console.print("Cancelled.")
        return

    if watch:
        from polylogue.pipeline.watch import WatchRunner

        observers: list[RunObserver] = [
            _WatchDisplayObserver(
                env,
                cfg,
                canonical_stage,
                selected_sources,
                display_stage=display_stage,
                stage_sequence=stage_sequence,
            ),
            _WatchStatusObserver(),
        ]

        if notify:
            observers.append(NotificationObserver())
        if exec_cmd:
            observers.append(ExecObserver(exec_cmd))
        if webhook:
            observers.append(WebhookObserver(webhook))

        composite = CompositeObserver(observers)

        def _sync_once() -> RunResult:
            return _run_with_progress(
                cfg,
                env,
                canonical_stage,
                stage_sequence,
                selected_sources,
                render_format,
                plan_snapshot=plan_snapshot,
                observer=composite,
                site_options=site_options,
            )

        env.ui.console.print("Watch mode: syncing every 60 seconds. Press Ctrl+C to stop.")
        runner = WatchRunner(
            sync_fn=_sync_once,
            observer=composite,
            interval=60,
        )
        runner.run()
        env.ui.console.print("\nWatch mode stopped.")
        return

    try:
        result = _run_sync_once(
            cfg,
            env,
            canonical_stage,
            stage_sequence,
            selected_sources,
            render_format,
            plan_snapshot,
            site_options=site_options,
        )
    except DriveError as exc:
        fail("run", str(exc))
    _display_result(
        env,
        cfg,
        result,
        canonical_stage,
        selected_sources,
        display_stage=display_stage,
        stage_sequence=stage_sequence,
    )


@run_command.command("acquire")
def run_acquire_stage() -> RunStageRequest:
    """Capture raw payloads only."""
    return _make_stage_request("acquire")


@run_command.command("schema")
def run_schema_stage() -> RunStageRequest:
    """Infer schemas from acquired raw payloads."""
    return _make_stage_request("schema")


@run_command.command("parse")
def run_parse_stage() -> RunStageRequest:
    """Parse persisted raw backlog into normalized conversations."""
    return _make_stage_request("parse")


@run_command.command("materialize")
def run_materialize_stage() -> RunStageRequest:
    """Refresh derived read models."""
    return _make_stage_request("materialize")


@run_command.command("render")
@click.option(
    "--format",
    "render_format",
    type=click.Choice(["markdown", "html"]),
    default="html",
    show_default=True,
    help="Output format for rendering (markdown or html)",
)
def run_render_stage(render_format: str) -> RunStageRequest:
    """Render human-facing publication artifacts."""
    return _make_stage_request("render", render_format=render_format)


@run_command.command("index")
def run_index_stage() -> RunStageRequest:
    """Build retrieval and search indexes."""
    return _make_stage_request("index")


@run_command.command("reprocess")
def run_reprocess_stage() -> RunStageRequest:
    """Skip acquisition and rerun downstream stages."""
    return _make_stage_request("reprocess")


@run_command.command("all")
def run_all_stage() -> RunStageRequest:
    """Run the full pipeline."""
    return _make_stage_request("all")


@run_command.command("site")
@click.option(
    "--output",
    "-o",
    type=click.Path(path_type=Path),
    default=None,
    help="Output directory for generated site (default: ~/.local/share/polylogue/site)",
)
@click.option(
    "--title",
    default="Polylogue Archive",
    help="Site title",
)
@click.option(
    "--search/--no-search",
    default=True,
    help="Enable client-side search (default: enabled)",
)
@click.option(
    "--search-provider",
    type=click.Choice(["pagefind", "lunr"]),
    default="pagefind",
    help="Search index provider (default: pagefind)",
)
@click.option(
    "--dashboard/--no-dashboard",
    default=True,
    help="Generate dashboard page (default: enabled)",
)
def run_site_stage(
    output: Path | None,
    title: str,
    search: bool,
    search_provider: str,
    dashboard: bool,
) -> RunStageRequest:
    """Generate a static HTML site from the archive."""
    return _make_stage_request(
        "site",
        site_options={
            "output": output,
            "title": title,
            "search": search,
            "search_provider": search_provider,
            "dashboard": dashboard,
        },
    )


def _run_embed_standalone(env: AppEnv, opts: EmbedOptions) -> None:
    """Execute the embed stage directly, outside the normal pipeline flow."""
    import os

    from polylogue.cli.embed_runtime import embed_batch, embed_single
    from polylogue.cli.embed_stats import show_embedding_stats

    if opts.json_output and not opts.stats:
        click.echo("Error: --json requires --stats", err=True)
        raise click.Abort()

    voyage_key = os.environ.get("POLYLOGUE_VOYAGE_API_KEY") or os.environ.get("VOYAGE_API_KEY")
    if not voyage_key and not opts.stats:
        click.echo("Error: VOYAGE_API_KEY environment variable not set", err=True)
        click.echo("Set it with: export VOYAGE_API_KEY=your-api-key  (or POLYLOGUE_VOYAGE_API_KEY)", err=True)
        raise click.Abort()

    if opts.stats:
        show_embedding_stats(env, json_output=opts.json_output)
        return

    from polylogue.storage.search_providers import create_vector_provider

    vec_provider = create_vector_provider(voyage_api_key=voyage_key)
    if vec_provider is None:
        click.echo("Error: sqlite-vec not available", err=True)
        click.echo("sqlite-vec is not available (ensure it is in your Nix flake or virtualenv)", err=True)
        raise click.Abort()

    if opts.model != "voyage-4":
        vec_provider.model = opts.model

    repo = env.repository

    if opts.conversation:
        embed_single(env, repo, vec_provider, opts.conversation)
        return

    embed_batch(env, repo, vec_provider, rebuild=opts.rebuild, limit=opts.limit)


@run_command.command("embed")
@click.option(
    "--conversation",
    "-c",
    type=str,
    default=None,
    help="Embed a specific conversation by ID",
)
@click.option(
    "--model",
    type=click.Choice(["voyage-4", "voyage-4-large", "voyage-4-lite"]),
    default="voyage-4",
    help="Voyage AI model: voyage-4 (default), voyage-4-large, voyage-4-lite",
)
@click.option(
    "--rebuild",
    "-r",
    is_flag=True,
    help="Re-embed all conversations (ignore existing embeddings)",
)
@click.option(
    "--stats",
    "-s",
    is_flag=True,
    help="Show embedding statistics only",
)
@click.option(
    "--json",
    "json_output",
    is_flag=True,
    help="Emit embedding statistics as JSON (requires --stats)",
)
@click.option(
    "--limit",
    "-n",
    type=int,
    default=None,
    help="Maximum number of conversations to embed",
)
def run_embed_stage(
    conversation: str | None,
    model: str,
    rebuild: bool,
    stats: bool,
    json_output: bool,
    limit: int | None,
) -> RunStageRequest:
    """Generate semantic embeddings for conversations."""
    return RunStageRequest(
        name="embed",
        stage_sequence=expand_requested_stage("embed"),
        embed_options=EmbedOptions(
            conversation=conversation,
            model=model,
            rebuild=rebuild,
            stats=stats,
            json_output=json_output,
            limit=limit,
        ),
    )


__all__ = ["run_command"]
