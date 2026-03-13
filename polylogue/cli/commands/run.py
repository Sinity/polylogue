"""Sync and sources commands."""

from __future__ import annotations

import asyncio
import json
import time

import click

from polylogue.cli.formatting import (
    format_counts,
    format_cursors,
    format_index_status,
    format_plan_counts,
    format_plan_details,
    format_run_details,
)
from polylogue.cli.helpers import (
    fail,
    maybe_prompt_sources,
    resolve_sources,
)
from polylogue.cli.run_observers import (
    _format_elapsed,
    progress_observer as _progress_observer,
)
from polylogue.cli.types import AppEnv
from polylogue.config import Config
from polylogue.lib.timestamps import format_timestamp
from polylogue.pipeline.observers import (
    CompositeObserver,
    ExecObserver,
    NotificationObserver,
    RunObserver,
    WebhookObserver,
)
from polylogue.pipeline.runner import RUN_STAGE_CHOICES, plan_sources, run_sources
from polylogue.protocols import ProgressCallback
from polylogue.sources import DriveError
from polylogue.storage.store import PlanResult, RunResult


def _execute_sync_once(
    cfg: Config,
    env: AppEnv,
    stage: str,
    selected_sources: list[str] | None,
    render_format: str,
    plan_snapshot: PlanResult | None = None,
    progress_callback: ProgressCallback | None = None,
) -> RunResult:
    """Execute a single sync run with the provided progress callback."""
    return asyncio.run(run_sources(
        config=cfg,
        stage=stage,
        plan=plan_snapshot,
        ui=env.ui,
        source_names=selected_sources,
        progress_callback=progress_callback,
        render_format=render_format,
        backend=env.backend,
        repository=env.repository,
    ))


def _run_with_progress(
    cfg: Config,
    env: AppEnv,
    stage: str,
    selected_sources: list[str] | None,
    render_format: str,
    plan_snapshot: PlanResult | None = None,
    observer: RunObserver | None = None,
) -> RunResult:
    """Execute a sync run while bridging progress updates through a run observer."""
    with _progress_observer(env) as progress_observer:
        progress_bridge = (
            CompositeObserver([progress_observer, observer])
            if observer is not None
            else progress_observer
        )
        result = _execute_sync_once(
            cfg,
            env,
            stage,
            selected_sources,
            render_format,
            plan_snapshot=plan_snapshot,
            progress_callback=progress_bridge.on_progress,
        )
        progress_observer.on_completed(result)
        return result


def _run_sync_once(
    cfg: Config,
    env: AppEnv,
    stage: str,
    selected_sources: list[str] | None,
    render_format: str,
    plan_snapshot: PlanResult | None = None,
    observer: RunObserver | None = None,
) -> RunResult:
    """Execute a single sync run and emit lifecycle notifications to an observer."""
    try:
        result = _run_with_progress(
            cfg,
            env,
            stage,
            selected_sources,
            render_format,
            plan_snapshot=plan_snapshot,
            observer=observer,
        )
    except Exception as exc:
        if observer is not None:
            observer.on_error(exc)
        raise

    if observer is not None:
        observer.on_completed(result)
    return result


def _display_result(
    env: AppEnv,
    cfg: Config,
    result: RunResult,
    stage: str,
    selected_sources: list[str] | None,
) -> None:
    """Display sync result summary."""
    run_lines = [
        f"Counts: {format_counts(result.counts)}",
        *format_run_details(result.counts),
        f"Duration: {result.duration_ms}ms",
    ]
    title_parts: list[str] = []
    if stage != "all":
        title_parts.append(stage)
    if selected_sources:
        title_parts.append(", ".join(selected_sources))
    title = f"Sync ({'; '.join(title_parts)})" if title_parts else "Sync"

    if stage == "index":
        run_lines = [
            format_index_status(stage, result.indexed, result.index_error),
            f"Duration: {result.duration_ms}ms",
        ]
    elif result.index_error:
        run_lines.insert(1, format_index_status(stage, result.indexed, result.index_error))

    env.ui.summary(title, run_lines)

    if stage in {"render", "all"}:
        from polylogue.cli.helpers import latest_render_path

        latest = latest_render_path(cfg.render_root)
        if latest:
            env.ui.console.print(f"Latest render: {latest}")

    # Surface render failures so users know which conversations couldn't be rendered
    if result.render_failures:
        n = len(result.render_failures)
        click.echo(f"\nRender failures ({n}):", err=True)
        show_limit = 10
        for failure in result.render_failures[:show_limit]:
            conv_id = failure.get("conversation_id", "?")
            error = failure.get("error", "unknown error")
            click.echo(f"  {conv_id}: {error}", err=True)
        if n > show_limit:
            click.echo(f"  ... and {n - show_limit} more", err=True)
        click.echo(
            "Hint: re-run with `polylogue run --stage render` to retry rendering.",
            err=True,
        )

    if result.index_error:
        error_line = f"Index error: {result.index_error}"
        hint_line = "Hint: run `polylogue run --stage index` to rebuild the index."
        click.echo(error_line, err=True)
        click.echo(hint_line, err=True)


class _WatchDisplayObserver(RunObserver):
    """Prints result summary when new conversations arrive in watch mode."""

    def __init__(self, env: AppEnv, cfg: Config, stage: str, selected_sources: list[str] | None) -> None:
        self._env = env
        self._cfg = cfg
        self._stage = stage
        self._selected_sources = selected_sources

    def on_completed(self, result: RunResult) -> None:
        if result.counts.get("conversations", 0) > 0:
            _display_result(self._env, self._cfg, result, self._stage, self._selected_sources)


class _WatchStatusObserver(RunObserver):
    """Prints idle and error status in watch mode."""

    def on_idle(self, result: RunResult) -> None:
        click.echo(f"No new conversations at {time.strftime('%H:%M:%S')}")

    def on_error(self, exc: Exception) -> None:
        if isinstance(exc, DriveError):
            click.echo(f"Sync error: {exc}", err=True)
        else:
            click.echo(f"Unexpected error during sync: {exc}", err=True)


@click.command("run")
@click.option("--preview", is_flag=True, help="Preview work without writing")
@click.option(
    "--stage",
    type=click.Choice(list(RUN_STAGE_CHOICES)),
    default="all",
    show_default=True,
    help=(
        "Pipeline stage: acquire (store raw), validate (schema check raw payloads), "
        "parse (extract conversations), render (output), index (search), "
        "generate-schemas, or all"
    ),
)
@click.option(
    "--source",
    "sources",
    multiple=True,
    help="Limit to source name (repeatable). 'last' = previously synced source. List with: polylogue sources",
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
@click.option("--notify", is_flag=True, help="Desktop notification on new conversations (requires --watch)")
@click.option("--exec", "exec_cmd", help="Execute command on new conversations (requires --watch)")
@click.option("--webhook", help="Call webhook URL on new conversations (requires --watch)")
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
        reset_count = asyncio.run(env.repository.reset_parse_status())
        click.echo(f"Reset parse status for {reset_count:,} raw records.", err=False)

    # Preview mode
    plan_snapshot = None
    if preview:
        try:
            with _progress_observer(
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
            fail("run", str(exc))
        plan_lines = []
        if selected_sources:
            plan_lines.append(f"Sources: {', '.join(selected_sources)}")
        if plan_snapshot is not None:
            plan_lines.append(f"Work: {format_plan_counts(plan_snapshot.counts)}")
            detail_line = format_plan_details(plan_snapshot.details)
            if detail_line:
                plan_lines.append(f"State: {detail_line}")
            cursor_line = format_cursors(plan_snapshot.cursors)
            if cursor_line:
                plan_lines.append(f"Cursors: {cursor_line}")
            plan_lines.append(f"Snapshot: {format_timestamp(plan_snapshot.timestamp)}")
        env.ui.summary("Preview", plan_lines)
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
    cfg = env.config
    if json_output:
        payload = [
            {
                "name": source.name,
                "path": str(source.path) if source.path else None,
                "folder": source.folder,
                "kind": "drive" if source.folder else "path",
            }
            for source in cfg.sources
        ]
        click.echo(json.dumps(payload, indent=2))
        return
    lines = []
    for source in cfg.sources:
        if source.folder:
            lines.append(f"{source.name}: drive folder '{source.folder}'")
        elif source.path:
            lines.append(f"{source.name}: {source.path}")
        else:
            lines.append(f"{source.name}: (missing path)")
    env.ui.summary("Sources", lines)


__all__ = ["run_command", "sources_command"]
