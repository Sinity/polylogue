"""Watch-mode observers for run workflows."""

from __future__ import annotations

import time

import click

from polylogue.lib.run_activity import conversation_activity_counts
from polylogue.pipeline.observers import RunObserver
from polylogue.sources import DriveError

from .run_display_workflow import display_result


class WatchDisplayObserver(RunObserver):
    """Print result summary when conversation changes arrive in watch mode."""

    def __init__(self, env, cfg, stage: str, selected_sources: list[str] | None) -> None:
        self._env = env
        self._cfg = cfg
        self._stage = stage
        self._selected_sources = selected_sources

    def on_completed(self, result) -> None:
        activity_count, _, _ = conversation_activity_counts(result.counts, result.drift)
        if activity_count > 0:
            display_result(self._env, self._cfg, result, self._stage, self._selected_sources)


class WatchStatusObserver(RunObserver):
    """Print idle and error status in watch mode."""

    def on_idle(self, result) -> None:
        click.echo(f"No conversation changes at {time.strftime('%H:%M:%S')}")

    def on_error(self, exc: Exception) -> None:
        if isinstance(exc, DriveError):
            click.echo(f"Sync error: {exc}", err=True)
        else:
            click.echo(f"Unexpected error during sync: {exc}", err=True)


__all__ = ["WatchDisplayObserver", "WatchStatusObserver"]
