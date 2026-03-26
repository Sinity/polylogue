"""Workflow helpers for run and sources commands."""

from __future__ import annotations

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
from polylogue.cli.helpers import fail
from polylogue.cli.run_observers import progress_observer as _progress_observer
from polylogue.cli.types import AppEnv
from polylogue.config import Config
from polylogue.lib.run_activity import conversation_activity_counts
from polylogue.lib.timestamps import format_timestamp
from polylogue.pipeline.observers import CompositeObserver, RunObserver
from polylogue.pipeline.runner import run_sources
from polylogue.protocols import ProgressCallback
from polylogue.sources import DriveError
from polylogue.storage.state_views import PlanResult, RunResult
from polylogue.sync_bridge import run_coroutine_sync


def execute_sync_once(
    cfg: Config,
    env: AppEnv,
    stage: str,
    selected_sources: list[str] | None,
    render_format: str,
    plan_snapshot: PlanResult | None = None,
    progress_callback: ProgressCallback | None = None,
) -> RunResult:
    return run_coroutine_sync(run_sources(
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


def run_with_progress(
    cfg: Config,
    env: AppEnv,
    stage: str,
    selected_sources: list[str] | None,
    render_format: str,
    plan_snapshot: PlanResult | None = None,
    observer: RunObserver | None = None,
) -> RunResult:
    with _progress_observer(env) as progress_observer:
        progress_bridge = (
            CompositeObserver([progress_observer, observer])
            if observer is not None
            else progress_observer
        )
        result = execute_sync_once(
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


def run_sync_once(
    cfg: Config,
    env: AppEnv,
    stage: str,
    selected_sources: list[str] | None,
    render_format: str,
    plan_snapshot: PlanResult | None = None,
    observer: RunObserver | None = None,
) -> RunResult:
    try:
        result = run_with_progress(
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


def display_result(
    env: AppEnv,
    cfg: Config,
    result: RunResult,
    stage: str,
    selected_sources: list[str] | None,
) -> None:
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
        click.echo(f"Index error: {result.index_error}", err=True)
        click.echo(
            "Hint: run `polylogue run --stage index` to rebuild the index.",
            err=True,
        )


class WatchDisplayObserver(RunObserver):
    """Print result summary when conversation changes arrive in watch mode."""

    def __init__(self, env: AppEnv, cfg: Config, stage: str, selected_sources: list[str] | None) -> None:
        self._env = env
        self._cfg = cfg
        self._stage = stage
        self._selected_sources = selected_sources

    def on_completed(self, result: RunResult) -> None:
        activity_count, _, _ = conversation_activity_counts(result.counts, result.drift)
        if activity_count > 0:
            display_result(self._env, self._cfg, result, self._stage, self._selected_sources)


class WatchStatusObserver(RunObserver):
    """Print idle and error status in watch mode."""

    def on_idle(self, result: RunResult) -> None:
        click.echo(f"No conversation changes at {time.strftime('%H:%M:%S')}")

    def on_error(self, exc: Exception) -> None:
        if isinstance(exc, DriveError):
            click.echo(f"Sync error: {exc}", err=True)
        else:
            click.echo(f"Unexpected error during sync: {exc}", err=True)


def render_sources(env: AppEnv, *, json_output: bool) -> None:
    from polylogue.cli.machine_errors import emit_success

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
        emit_success({"sources": payload})
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


def handle_drive_error(exc: DriveError) -> None:
    fail("run", str(exc))


def render_preview_summary(
    env: AppEnv,
    *,
    selected_sources: list[str] | None,
    plan_snapshot: PlanResult | None,
) -> None:
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


__all__ = [
    "WatchDisplayObserver",
    "WatchStatusObserver",
    "display_result",
    "execute_sync_once",
    "handle_drive_error",
    "render_preview_summary",
    "render_sources",
    "run_sync_once",
    "run_with_progress",
]
