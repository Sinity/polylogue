"""Workflow helpers for run and sources commands."""

from __future__ import annotations

from collections.abc import Sequence

import click

from polylogue.api.sync.bridge import run_coroutine_sync
from polylogue.cli.formatting import (
    format_counts,
    format_cursors,
    format_index_status,
    format_plan_counts,
    format_plan_details,
    format_run_details,
)
from polylogue.cli.helpers import fail
from polylogue.cli.run_display_workflow import render_sources
from polylogue.cli.run_observers import progress_observer
from polylogue.cli.run_watch_workflow import WatchDisplayObserver, WatchStatusObserver
from polylogue.cli.types import AppEnv
from polylogue.config import Config
from polylogue.lib.timestamps import format_timestamp
from polylogue.pipeline.observers import CompositeObserver, RunObserver
from polylogue.pipeline.payload_types import SiteBuildOptions
from polylogue.pipeline.run_support import expand_requested_stage
from polylogue.pipeline.runner import run_sources
from polylogue.protocols import ProgressCallback
from polylogue.sources import DriveError
from polylogue.storage.run_state import PlanResult, RunResult


def execute_sync_once(
    cfg: Config,
    env: AppEnv,
    stage: str,
    stage_sequence: Sequence[str] | None,
    selected_sources: list[str] | None,
    render_format: str,
    plan_snapshot: PlanResult | None = None,
    progress_callback: ProgressCallback | None = None,
    site_options: SiteBuildOptions | None = None,
) -> RunResult:
    return run_coroutine_sync(
        run_sources(
            config=cfg,
            stage=stage,
            stage_sequence=stage_sequence,
            plan=plan_snapshot,
            ui=env.ui,
            source_names=selected_sources,
            progress_callback=progress_callback,
            render_format=render_format,
            site_options=site_options,
            backend=env.backend,
            repository=env.repository,
        )
    )


def run_with_progress(
    cfg: Config,
    env: AppEnv,
    stage: str,
    stage_sequence: Sequence[str] | None,
    selected_sources: list[str] | None,
    render_format: str,
    plan_snapshot: PlanResult | None = None,
    observer: RunObserver | None = None,
    site_options: SiteBuildOptions | None = None,
) -> RunResult:
    with progress_observer(env) as progress_observer_handle:
        progress_bridge = (
            CompositeObserver([progress_observer_handle, observer])
            if observer is not None
            else progress_observer_handle
        )
        result = execute_sync_once(
            cfg,
            env,
            stage,
            stage_sequence,
            selected_sources,
            render_format,
            plan_snapshot=plan_snapshot,
            progress_callback=progress_bridge.on_progress,
            site_options=site_options,
        )
        progress_observer_handle.on_completed(result)
        return result


def run_sync_once(
    cfg: Config,
    env: AppEnv,
    stage: str,
    stage_sequence: Sequence[str] | None,
    selected_sources: list[str] | None,
    render_format: str,
    plan_snapshot: PlanResult | None = None,
    observer: RunObserver | None = None,
    site_options: SiteBuildOptions | None = None,
) -> RunResult:
    try:
        result = run_with_progress(
            cfg,
            env,
            stage,
            stage_sequence,
            selected_sources,
            render_format,
            plan_snapshot=plan_snapshot,
            observer=observer,
            site_options=site_options,
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
    *,
    display_stage: str | None = None,
    stage_sequence: Sequence[str] | None = None,
) -> None:
    normalized_stage_sequence = tuple(stage_sequence or expand_requested_stage(stage))
    title_stage = display_stage or stage
    run_lines = [
        f"Counts: {format_counts(result.counts)}",
        *format_run_details(result.counts),
        f"Duration: {result.duration_ms}ms",
    ]
    title_parts: list[str] = []
    if title_stage != "all":
        title_parts.append(title_stage)
    if selected_sources:
        title_parts.append(", ".join(selected_sources))
    title = f"Sync ({'; '.join(title_parts)})" if title_parts else "Sync"

    if normalized_stage_sequence == ("index",):
        run_lines = [
            format_index_status(stage, result.indexed, result.index_error),
            f"Duration: {result.duration_ms}ms",
        ]
    elif result.index_error:
        run_lines.insert(1, format_index_status(stage, result.indexed, result.index_error))

    env.ui.summary(title, run_lines)

    if "render" in normalized_stage_sequence:
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
            "Hint: re-run with `polylogue run render` to retry rendering.",
            err=True,
        )

    if result.index_error:
        click.echo(f"Index error: {result.index_error}", err=True)
        click.echo(
            "Hint: run `polylogue run index` to rebuild the index.",
            err=True,
        )


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
    "format_counts",
    "format_cursors",
    "format_index_status",
    "format_plan_counts",
    "format_plan_details",
    "format_run_details",
    "handle_drive_error",
    "render_preview_summary",
    "render_sources",
    "run_coroutine_sync",
    "run_sources",
    "run_sync_once",
    "run_with_progress",
]
