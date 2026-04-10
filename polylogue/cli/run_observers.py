"""Progress observers for CLI run commands."""

from __future__ import annotations

import re
import time
from collections.abc import Iterator
from contextlib import contextmanager

from polylogue.cli.formatting import format_counts
from polylogue.cli.types import AppEnv
from polylogue.pipeline.observers import RunObserver
from polylogue.storage.state_views import RunResult

_PROGRESS_FRACTION_RE = re.compile(r"(?P<completed>\d[\d,]*)/(?P<total>\d[\d,]*)")


def _format_elapsed(seconds: float) -> str:
    """Format elapsed seconds as a compact duration string."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    if minutes < 60:
        return f"{minutes}m{secs:02d}s"
    hours = minutes // 60
    mins = minutes % 60
    return f"{hours}h{mins:02d}m{secs:02d}s"


class PlainProgressObserver(RunObserver):
    """Plain-text progress output for non-TTY runs."""

    def __init__(self, *, banner: str = "Syncing...") -> None:
        self.pipeline_start = time.time()
        self.stage_start = self.pipeline_start
        self.last_update = self.pipeline_start
        self.stage_processed = 0
        self.last_desc = ""
        self.last_stage = ""
        print(banner, flush=True)

    def _stage_key(self, desc: str) -> str:
        return desc.split(":")[0].split("[")[0].strip()

    def on_progress(self, amount: int, desc: str | None = None) -> None:
        self.stage_processed += amount
        now = time.time()
        current_stage = self._stage_key(desc) if desc else self.last_stage
        is_stage_change = current_stage != self.last_stage and current_stage
        if is_stage_change:
            if self.last_stage:
                prev_elapsed = now - self.stage_start
                print(
                    f"  {self.last_stage}: done ({self.stage_processed - amount:,} in {_format_elapsed(prev_elapsed)})",
                    flush=True,
                )
            self.last_stage = current_stage
            self.stage_start = now
            self.stage_processed = amount
        if desc:
            self.last_desc = desc
        if is_stage_change or now - self.last_update >= 1:
            elapsed = now - self.stage_start
            total_elapsed = now - self.pipeline_start
            rate = self.stage_processed / elapsed if elapsed > 0.5 else 0
            rate_str = f" ({rate:,.0f}/s)" if rate > 0 else ""
            print(
                f"  {self.last_desc or 'Processing'}: {self.stage_processed:,}{rate_str}"
                f" [{_format_elapsed(total_elapsed)} total]...",
                flush=True,
            )
            self.last_update = now

    def on_completed(self, result: RunResult) -> None:
        total_elapsed = time.time() - self.pipeline_start
        count_summary = format_counts(result.counts)
        if count_summary:
            print(f"  Counts: {count_summary}", flush=True)
        print(f"  Pipeline complete in {_format_elapsed(total_elapsed)}", flush=True)


class RichProgressObserver(RunObserver):
    """Rich progress bridge for TTY runs."""

    __slots__ = ("_progress", "_task_id")

    def __init__(self, progress: object, task_id: object) -> None:
        self._progress = progress
        self._task_id = task_id

    @staticmethod
    def _progress_bounds(desc: str | None) -> tuple[int, int] | None:
        if not desc:
            return None
        matches = list(_PROGRESS_FRACTION_RE.finditer(desc))
        if not matches:
            return None
        match = matches[-1]
        completed = int(match.group("completed").replace(",", ""))
        total = int(match.group("total").replace(",", ""))
        if total <= 0 or completed < 0 or completed > total:
            return None
        return completed, total

    def on_progress(self, amount: int, desc: str | None = None) -> None:
        parsed_bounds = self._progress_bounds(desc)
        if parsed_bounds is not None:
            completed, total = parsed_bounds
            if desc:
                self._progress.update(
                    self._task_id,
                    description=desc,
                    total=total,
                    completed=completed,
                )
            else:
                self._progress.update(
                    self._task_id,
                    total=total,
                    completed=completed,
                )
            return

        if desc:
            self._progress.update(self._task_id, description=desc)
        if amount:
            self._progress.update(self._task_id, advance=amount)


@contextmanager
def progress_observer(
    env: AppEnv,
    *,
    initial_desc: str = "Syncing sources...",
    plain_banner: str = "Syncing...",
) -> Iterator[RunObserver]:
    """Yield a progress observer appropriate for the active UI."""
    if env.ui.plain:
        yield PlainProgressObserver(banner=plain_banner)
        return

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
        task_id = progress.add_task(initial_desc, total=None)
        yield RichProgressObserver(progress, task_id)


__all__ = [
    "PlainProgressObserver",
    "RichProgressObserver",
    "_format_elapsed",
    "progress_observer",
]
