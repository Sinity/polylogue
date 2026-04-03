"""Sync and sources commands."""

from __future__ import annotations

from dataclasses import dataclass

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
    render_sources,
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
    "index",
)


@dataclass(frozen=True, slots=True)
class RunStageRequest:
    name: str
    stage_sequence: tuple[str, ...]
    render_format: str | None = None


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


def _make_stage_request(name: str, *, render_format: str | None = None) -> RunStageRequest:
    return RunStageRequest(
        name=name,
        stage_sequence=expand_requested_stage(name),
        render_format=render_format,
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
    render_formats = {
        request.render_format
        for request in stage_requests
        if request.render_format is not None
    }
    if not render_formats:
        return "html"
    if len(render_formats) > 1:
        fail("run", f"Conflicting render formats requested: {', '.join(sorted(render_formats))}")
    return next(iter(render_formats))


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


@click.command("sources")
@click.option("--json", "json_output", is_flag=True, help="Output JSON")
@click.pass_obj
def sources_command(env: AppEnv, json_output: bool) -> None:
    """List configured sources."""
    render_sources(env, json_output=json_output)


__all__ = ["run_command", "sources_command"]
