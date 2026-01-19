"""Run and sources commands."""

from __future__ import annotations

import json

import click

from polylogue.cli.formatting import (
    format_counts,
    format_cursors,
    format_index_status,
    format_timestamp,
)
from polylogue.cli.helpers import (
    fail,
    load_effective_config,
    maybe_prompt_sources,
    resolve_sources,
)
from polylogue.cli.types import AppEnv
from polylogue.config import ConfigError
from polylogue.drive_client import DriveError
from polylogue.pipeline.runner import plan_sources, run_sources


@click.command("run")
@click.option("--preview", is_flag=True, help="Preview work without writing")
@click.option("--stage", type=click.Choice(["ingest", "render", "index", "all"]), default="all", show_default=True)
@click.option(
    "--source",
    "sources",
    multiple=True,
    help="Limit to source name (repeatable, or 'last'). Use `polylogue sources` to list.",
)
@click.pass_obj
def run_command(
    env: AppEnv,
    preview: bool,
    stage: str,
    sources: tuple[str, ...],
) -> None:
    try:
        config = load_effective_config(env)
    except ConfigError as exc:
        fail("run", str(exc))
    selected_sources = resolve_sources(config, sources, "run")
    selected_sources = maybe_prompt_sources(env, config, selected_sources, "run")
    if preview:
        try:
            plan_result = plan_sources(config, ui=env.ui, source_names=selected_sources)
        except DriveError as exc:
            fail("run", str(exc))
        plan_lines = []
        if selected_sources:
            plan_lines.append(f"Sources: {', '.join(selected_sources)}")
        plan_lines.append(f"Counts: {format_counts(plan_result.counts)}")
        cursor_line = format_cursors(plan_result.cursors)
        if cursor_line:
            plan_lines.append(f"Cursors: {cursor_line}")
        plan_lines.append(f"Snapshot: {format_timestamp(plan_result.timestamp)}")
        env.ui.summary("Preview", plan_lines)
        return
    if not env.ui.plain and not env.ui.confirm("Proceed with run?", default=True):
        env.ui.console.print("Run cancelled.")
        return
    try:
        import time as _time

        # In plain mode, show periodic progress updates
        if env.ui.plain:
            print("Running...", flush=True)
            last_update = [_time.time()]
            processed = [0]

            def plain_progress(amount: int, desc: str | None = None) -> None:
                processed[0] += amount
                now = _time.time()
                # Print progress every 5 seconds
                if now - last_update[0] >= 5:
                    print(f"  {desc or 'Processing'}: {processed[0]:,} items...", flush=True)
                    last_update[0] = now

            result = run_sources(
                config=config,
                stage=stage,
                plan=None,
                ui=env.ui,
                source_names=selected_sources,
                progress_callback=plain_progress,
            )
        else:
            from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeRemainingColumn

            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TimeRemainingColumn(),
                console=env.ui.console,
                transient=True,
            ) as progress:
                task_id = progress.add_task("Running sources...", total=None)

                def progress_callback(amount: int, desc: str | None = None) -> None:
                    if desc:
                        progress.update(task_id, description=desc)
                    progress.update(task_id, advance=amount)

                result = run_sources(
                    config=config,
                    stage=stage,
                    plan=None,
                    ui=env.ui,
                    source_names=selected_sources,
                    progress_callback=progress_callback,
                )
    except DriveError as exc:
        fail("run", str(exc))
    run_lines = [
        f"Counts: {format_counts(result.counts)}",
        f"Duration: {result.duration_ms}ms",
    ]
    title_parts: list[str] = []
    if stage != "all":
        title_parts.append(stage)
    if selected_sources:
        title_parts.append(", ".join(selected_sources))
    title = f"Run ({'; '.join(title_parts)})" if title_parts else "Run"
    if stage == "index":
        run_lines = [
            format_index_status(stage, result.indexed, result.index_error),
            f"Duration: {result.duration_ms}ms",
        ]
    elif result.index_error:
        run_lines.insert(1, format_index_status(stage, result.indexed, result.index_error))
    env.ui.summary(
        title,
        run_lines,
    )
    if stage in {"render", "all"}:
        from polylogue.cli.helpers import latest_render_path

        latest = latest_render_path(config.render_root)
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


@click.command("sources")
@click.option("--json", "json_output", is_flag=True, help="Output JSON")
@click.pass_obj
def sources_command(env: AppEnv, json_output: bool) -> None:
    try:
        config = load_effective_config(env)
    except ConfigError as exc:
        fail("sources", str(exc))
    if json_output:
        payload = [
            {
                "name": source.name,
                "path": str(source.path) if source.path else None,
                "folder": source.folder,
                "kind": "drive" if source.folder else "path",
            }
            for source in config.sources
        ]
        env.ui.console.print(json.dumps(payload, indent=2))
        return
    lines = []
    for source in config.sources:
        if source.folder:
            lines.append(f"{source.name}: drive folder '{source.folder}'")
        elif source.path:
            lines.append(f"{source.name}: {source.path}")
        else:
            lines.append(f"{source.name}: (missing path)")
    env.ui.summary("Sources", lines)
