"""Long-running ``polylogue watch`` command.

Starts the live filesystem watcher against ``~/.claude/projects/`` and
``~/.codex/sessions/`` (or user-specified roots). On every modify-and-quiet
event, the affected JSONL is re-parsed through the regular ingest pipeline
(idempotent via content-hash).
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import click

from polylogue.api import Polylogue
from polylogue.cli.shared.types import AppEnv
from polylogue.sources.live import LiveWatcher, WatchSource
from polylogue.sources.live.watcher import default_sources


async def _run(
    *,
    sources: tuple[WatchSource, ...],
    debounce_s: float,
) -> None:
    async with Polylogue() as polylogue:
        watcher = LiveWatcher(polylogue, sources, debounce_s=debounce_s)
        try:
            await watcher.run()
        except KeyboardInterrupt:
            watcher.stop()


@click.command("watch", help="Watch source directories and ingest new sessions live.")
@click.option(
    "--root",
    "roots",
    multiple=True,
    type=click.Path(exists=False, path_type=Path),
    help="Override watch root (repeatable). Defaults to ~/.claude/projects and ~/.codex/sessions.",
)
@click.option(
    "--debounce-s",
    type=float,
    default=2.0,
    show_default=True,
    help="Quiet-period (seconds) before parsing a modified file.",
)
@click.pass_obj
def watch_command(env: AppEnv, roots: tuple[Path, ...], debounce_s: float) -> None:
    sources = tuple(WatchSource(name=p.name, root=p) for p in roots) if roots else default_sources()

    click.echo(
        f"Watching {len(sources)} source(s); debounce={debounce_s}s. Ctrl-C to stop.",
        err=True,
    )
    asyncio.run(_run(sources=sources, debounce_s=debounce_s))


__all__ = ["watch_command"]
