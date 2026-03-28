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
)
from polylogue.cli.helpers import (
    fail,
    maybe_prompt_sources,
    resolve_sources,
)
from polylogue.cli.types import AppEnv
from polylogue.config import Config
from polylogue.lib.timestamps import format_timestamp
from polylogue.pipeline.runner import run_sources, plan_sources
from polylogue.sources import DriveError
from polylogue.storage.store import PlanResult, RunResult


def _run_sync_once(
    cfg: Config,
    env: AppEnv,
    stage: str,
    selected_sources: list[str] | None,
    render_format: str,
    plan_snapshot: PlanResult | None = None,
) -> RunResult:
    """Execute a single sync run."""
    if env.ui.plain:
        print("Syncing...", flush=True)
        last_update = [time.time()]
        processed = [0]

        def plain_progress(amount: int, desc: str | None = None) -> None:
            processed[0] += amount
            now = time.time()
            if now - last_update[0] >= 1:
                print(f"  {desc or 'Processing'}: {processed[0]:,} items...", flush=True)
                last_update[0] = now

        return asyncio.run(run_sources(
            config=cfg,
            stage=stage,
            plan=plan_snapshot,
            ui=env.ui,
            source_names=selected_sources,
            progress_callback=plain_progress,
            render_format=render_format,
        ))
    else:
        from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeRemainingColumn

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeRemainingColumn(),
            console=env.ui.console,  # type: ignore[arg-type]
            transient=True,
        ) as progress:
            task_id = progress.add_task("Syncing sources...", total=None)

            def progress_callback(amount: int, desc: str | None = None) -> None:
                if desc:
                    progress.update(task_id, description=desc)
                progress.update(task_id, advance=amount)

            return asyncio.run(run_sources(
                config=cfg,
                stage=stage,
                plan=plan_snapshot,
                ui=env.ui,
                source_names=selected_sources,
                progress_callback=progress_callback,
                render_format=render_format,
            ))


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


@click.command("run")
@click.option("--preview", is_flag=True, help="Preview work without writing")
@click.option(
    "--stage",
    type=click.Choice(["acquire", "parse", "render", "index", "generate-schemas", "all"]),
    default="all",
    show_default=True,
    help="Pipeline stage: acquire (store raw), parse (raw→conversations), render, index, or all",
)
@click.option(
    "--source",
    "sources",
    multiple=True,
    help="Limit to source name (repeatable, or 'last'). Use `polylogue sources` to list.",
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
) -> None:
    """Run pipeline stages on configured sources."""
    # Validate watch-related flags
    if (notify or exec_cmd or webhook) and not watch:
        fail("run", "--notify, --exec, and --webhook require --watch mode")

    from polylogue.services import get_service_config

    cfg = get_service_config()

    selected_sources = resolve_sources(cfg, sources, "run")
    selected_sources = maybe_prompt_sources(env, cfg, selected_sources, "run")

    # Preview mode
    plan_snapshot = None
    if preview:
        try:
            plan_snapshot = plan_sources(cfg, ui=env.ui, source_names=selected_sources)
        except DriveError as exc:
            fail("run", str(exc))
        plan_lines = []
        if selected_sources:
            plan_lines.append(f"Sources: {', '.join(selected_sources)}")
        if plan_snapshot is not None:
            plan_lines.append(f"Counts: {format_counts(plan_snapshot.counts)}")
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
        from polylogue.pipeline.events import (
            CompositeSyncHandler,
            ExecHandler,
            NotificationHandler,
            SyncEvent,
            WebhookHandler,
        )
        from polylogue.pipeline.watch import WatchRunner

        # Build event handlers from CLI flags
        handlers: list[object] = []

        # Always display results when new conversations arrive
        class _DisplayHandler:
            def on_sync(self, event: SyncEvent) -> None:
                if event.new_conversations > 0 and event.run_result is not None:
                    _display_result(env, cfg, event.run_result, stage, selected_sources)

        handlers.append(_DisplayHandler())

        if notify:
            handlers.append(NotificationHandler())
        if exec_cmd:
            handlers.append(ExecHandler(exec_cmd))
        if webhook:
            handlers.append(WebhookHandler(webhook))

        composite = CompositeSyncHandler(handlers)  # type: ignore[arg-type]

        def _sync_once() -> RunResult:
            return _run_sync_once(cfg, env, stage, selected_sources, render_format)

        def _on_idle(result: RunResult) -> None:
            click.echo(f"No new conversations at {time.strftime('%H:%M:%S')}")

        def _on_error(exc: Exception) -> None:
            if isinstance(exc, DriveError):
                click.echo(f"Sync error: {exc}", err=True)
            else:
                click.echo(f"Unexpected error during sync: {exc}", err=True)

        env.ui.console.print("Watch mode: syncing every 60 seconds. Press Ctrl+C to stop.")
        runner = WatchRunner(
            sync_fn=_sync_once,
            handler=composite,
            interval=60,
            on_idle=_on_idle,
            on_error=_on_error,
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
    from polylogue.services import get_service_config

    cfg = get_service_config()
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
