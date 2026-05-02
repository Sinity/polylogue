"""Command line entrypoint for long-running Polylogue services."""

from __future__ import annotations

import asyncio
from pathlib import Path

import click

from polylogue.api import Polylogue
from polylogue.core.json import dumps
from polylogue.daemon.browser_capture import browser_capture_command
from polylogue.daemon.status import daemon_status_payload
from polylogue.sources.live import LiveWatcher, WatchSource
from polylogue.sources.live.watcher import default_sources


async def run_live_watcher(
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


@click.group(help="Run long-lived Polylogue local services.")
def main() -> None:
    pass


main.add_command(browser_capture_command)


@main.command("status", help="Show configured daemon component status.")
@click.option("--spool", "spool_path", type=click.Path(path_type=Path), default=None)
@click.option("--format", "output_format", type=click.Choice(["json"]), default=None, help="Output format.")
def status_command(spool_path: Path | None, output_format: str | None) -> None:
    payload = daemon_status_payload(browser_capture_spool_path=spool_path)
    if output_format == "json":
        click.echo(dumps(payload))
        return
    click.echo("Polylogue daemon")
    live = payload["live"]
    browser_capture = payload["browser_capture"]
    if isinstance(live, dict):
        click.echo(f"Live sources: {live.get('existing_source_count', 0)}/{live.get('source_count', 0)} available")
        sources = live.get("sources", [])
        if isinstance(sources, list):
            for source in sources:
                if isinstance(source, dict):
                    state = "available" if source.get("exists") else "missing"
                    click.echo(f"  {source.get('name')}: {source.get('root')} ({state})")
    if isinstance(browser_capture, dict):
        click.echo(f"Browser capture spool: {browser_capture.get('spool_path')}")
        origins = browser_capture.get("allowed_origins", [])
        origin_text = ", ".join(str(item) for item in origins) if isinstance(origins, list) else str(origins)
        click.echo(f"Browser capture origins: {origin_text}")


@main.command("watch", help="Watch source directories and ingest new sessions live.")
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
def watch_command(roots: tuple[Path, ...], debounce_s: float) -> None:
    sources = tuple(WatchSource(name=p.name, root=p) for p in roots) if roots else default_sources()

    click.echo(
        f"Watching {len(sources)} source(s); debounce={debounce_s}s. Ctrl-C to stop.",
        err=True,
    )
    asyncio.run(run_live_watcher(sources=sources, debounce_s=debounce_s))


__all__ = ["main", "run_live_watcher", "status_command", "watch_command"]
