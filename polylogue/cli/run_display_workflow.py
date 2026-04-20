"""Display helpers for run and sources commands."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

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
from polylogue.lib.timestamps import format_timestamp
from polylogue.pipeline.run_support import expand_requested_stage
from polylogue.sources import DriveError
from polylogue.storage.state_views import PlanResult, RunResult


def display_result(
    env: Any,
    cfg: Any,
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


def render_sources(env: Any, *, json_output: bool) -> None:
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
    env: Any,
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


__all__ = ["display_result", "handle_drive_error", "render_preview_summary", "render_sources"]
