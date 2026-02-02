"""Sync and sources commands."""

from __future__ import annotations

import json
import time

import click

from polylogue.cli.container import create_config
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
from polylogue.core.timestamps import format_timestamp
from polylogue.ingestion import DriveError
from polylogue.storage.store import PlanResult, RunResult
from polylogue.pipeline.runner import plan_sources, run_sources


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
            if now - last_update[0] >= 5:
                print(f"  {desc or 'Processing'}: {processed[0]:,} items...", flush=True)
                last_update[0] = now

        return run_sources(
            config=cfg,
            stage=stage,
            plan=plan_snapshot,
            ui=env.ui,
            source_names=selected_sources,
            progress_callback=plain_progress,
            render_format=render_format,
        )
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

            return run_sources(
                config=cfg,
                stage=stage,
                plan=plan_snapshot,
                ui=env.ui,
                source_names=selected_sources,
                progress_callback=progress_callback,
                render_format=render_format,
            )


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

    if result.index_error:
        error_line = f"Index error: {result.index_error}"
        hint_line = "Hint: run `polylogue run --stage index` to rebuild the index."
        if env.ui.plain:
            env.ui.console.print(error_line)
            env.ui.console.print(hint_line)
        else:
            env.ui.console.print(f"[yellow]{error_line}[/yellow]")
            env.ui.console.print(f"[yellow]{hint_line}[/yellow]")


def _notify_new_conversations(count: int) -> None:
    """Send desktop notification for new conversations."""
    try:
        import subprocess

        subprocess.run(
            ["notify-send", "Polylogue", f"Synced {count} new conversation(s)"],
            capture_output=True,
            check=False,
        )
    except FileNotFoundError:
        pass  # notify-send not available


def _exec_on_new(exec_cmd: str, count: int) -> None:
    """Execute command when new conversations are synced."""
    import os
    import subprocess

    env = os.environ.copy()
    env["POLYLOGUE_NEW_COUNT"] = str(count)
    subprocess.run(exec_cmd, shell=True, env=env, check=False)


def _webhook_on_new(webhook_url: str, count: int) -> None:
    """Call webhook URL when new conversations are synced."""
    try:
        import json as json_lib
        import urllib.request

        data = json_lib.dumps({"event": "sync", "new_conversations": count}).encode()
        req = urllib.request.Request(
            webhook_url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        urllib.request.urlopen(req, timeout=10)
    except Exception:
        pass  # Silently fail on webhook errors


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
@click.option("-p", "--provider", help="Limit to specific provider")
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
    provider: str | None,
) -> None:
    """Run pipeline stages on configured sources."""
    # Validate watch-related flags
    if (notify or exec_cmd or webhook) and not watch:
        fail("run", "--notify, --exec, and --webhook require --watch mode")

    cfg = create_config()

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
    if not watch and not plan_snapshot and not env.ui.plain:
        if not env.ui.confirm("Proceed?", default=True):
            env.ui.console.print("Cancelled.")
            return

    # Watch mode
    if watch:
        env.ui.console.print("Watch mode: syncing every 60 seconds. Press Ctrl+C to stop.")
        poll_interval = 60
        try:
            while True:
                try:
                    result = _run_sync_once(cfg, env, stage, selected_sources, render_format)
                    new_count = result.counts.get("conversations", 0)
                    if new_count > 0:
                        _display_result(env, cfg, result, stage, selected_sources)
                        if notify:
                            _notify_new_conversations(new_count)
                        if exec_cmd:
                            _exec_on_new(exec_cmd, new_count)
                        if webhook:
                            _webhook_on_new(webhook, new_count)
                    else:
                        env.ui.console.print(f"[dim]No new conversations at {time.strftime('%H:%M:%S')}[/dim]")
                except DriveError as exc:
                    env.ui.console.print(f"[red]Sync error: {exc}[/red]")
                time.sleep(poll_interval)
        except KeyboardInterrupt:
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
    cfg = create_config()
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
        env.ui.console.print(json.dumps(payload, indent=2))
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


# Deprecated alias for backwards compatibility
@click.command("sync", deprecated=True, hidden=True)
@click.option("--preview", is_flag=True)
@click.option("--stage", type=click.Choice(["ingest", "render", "index", "all"]), default="all")
@click.option("--source", "sources", multiple=True)
@click.option("--format", "render_format", type=click.Choice(["markdown", "html"]), default="html")
@click.option("--watch", is_flag=True)
@click.option("--notify", is_flag=True)
@click.option("--exec", "exec_cmd")
@click.option("--webhook")
@click.option("-p", "--provider")
@click.pass_context
def sync_command(
    ctx: click.Context,
    preview: bool,
    stage: str,
    sources: tuple[str, ...],
    render_format: str,
    watch: bool,
    notify: bool,
    exec_cmd: str | None,
    webhook: str | None,
    provider: str | None,
) -> None:
    """[DEPRECATED] Use 'polylogue run' instead."""
    env: AppEnv = ctx.obj

    # Map old stage names to new
    stage_map = {"ingest": "parse"}  # "ingest" → "parse" (acquire+parse)
    mapped_stage = stage_map.get(stage, stage)

    env.ui.console.print("[yellow]Warning: 'sync' is deprecated. Use 'polylogue run' instead.[/yellow]")

    # Forward to run_command using ctx.invoke
    ctx.invoke(
        run_command,
        preview=preview,
        stage=mapped_stage,
        sources=sources,
        render_format=render_format,
        watch=watch,
        notify=notify,
        exec_cmd=exec_cmd,
        webhook=webhook,
        provider=provider,
    )
