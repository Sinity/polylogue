"""Sync and sources commands."""

from __future__ import annotations

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
from polylogue.pipeline.runner import RUN_STAGE_CHOICES, plan_sources
from polylogue.sources import DriveError
from polylogue.storage.state_views import RunResult
from polylogue.sync_bridge import run_coroutine_sync


@click.command("run")
@click.option("--preview", is_flag=True, help="Preview work without writing")
@click.option(
    "--stage",
    type=click.Choice(list(RUN_STAGE_CHOICES)),
    default="all",
    show_default=True,
    help=(
        "Pipeline stage: acquire (store raw), "
        "parse (extract conversations), materialize (derived products), "
        "render (output), index (search), generate-schemas, or all"
    ),
)
@click.option(
    "--source",
    "sources",
    multiple=True,
    shell_complete=complete_run_source_names,
    help="Configured source name (repeatable). Accepts 'last' for the previously synced source.",
)
@click.option(
    "--format",
    "render_format",
    type=click.Choice(["markdown", "html"]),
    default="html",
    show_default=True,
    help="Output format for rendering (markdown or html)",
)
@click.option("--watch", is_flag=True, help="Watch sources for changes and run continuously")
@click.option("--notify", is_flag=True, help="Desktop notification on conversation changes (requires --watch)")
@click.option("--exec", "exec_cmd", help="Execute command on conversation changes (requires --watch)")
@click.option("--webhook", help="Call webhook URL on conversation changes (requires --watch)")
@click.option("--reparse", is_flag=True, help="Force re-parsing of all raw conversations (clears parse tracking)")
@click.pass_obj
def run_command(
    env: AppEnv,
    preview: bool,
    stage: str,
    sources: tuple[str, ...],
    render_format: str,
    watch: bool,
    notify: bool,
    exec_cmd: str | None,
    webhook: str | None,
    reparse: bool,
) -> None:
    """Run pipeline stages on configured sources."""
    # Validate watch-related flags
    if (notify or exec_cmd or webhook) and not watch:
        fail("run", "--notify, --exec, and --webhook require --watch mode")

    cfg = env.config

    selected_sources = resolve_sources(cfg, sources, "run")
    selected_sources = maybe_prompt_sources(env, cfg, selected_sources, "run")

    # Reset parse tracking if --reparse was requested
    if reparse:
        reset_count = run_coroutine_sync(env.repository.reset_parse_status())
        click.echo(f"Reset parse status for {reset_count:,} raw records.", err=False)

    # Preview mode
    plan_snapshot = None
    if preview:
        try:
            from polylogue.cli.run_observers import progress_observer

            with progress_observer(
                env,
                initial_desc="Planning preview...",
                plain_banner="Planning preview...",
            ) as progress_observer:
                plan_snapshot = plan_sources(
                    cfg,
                    stage=stage,
                    ui=env.ui,
                    source_names=selected_sources,
                    backend=env.backend,
                    progress_callback=progress_observer.on_progress,
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

    # Interactive confirmation (non-preview, non-watch)
    if not watch and not plan_snapshot and not env.ui.plain and not env.ui.confirm("Proceed?", default=True):
        env.ui.console.print("Cancelled.")
        return

    # Watch mode
    if watch:
        from polylogue.pipeline.watch import WatchRunner

        observers: list[RunObserver] = [
            _WatchDisplayObserver(env, cfg, stage, selected_sources),
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
                stage,
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
    else:
        # Single sync run
        try:
            result = _run_sync_once(cfg, env, stage, selected_sources, render_format, plan_snapshot)
        except DriveError as exc:
            fail("run", str(exc))
        _display_result(env, cfg, result, stage, selected_sources)


@click.command("sources")
@click.option("--json", "json_output", is_flag=True, help="Output JSON")
@click.pass_obj
def sources_command(env: AppEnv, json_output: bool) -> None:
    """List configured sources."""
    render_sources(env, json_output=json_output)


__all__ = ["run_command", "sources_command"]
