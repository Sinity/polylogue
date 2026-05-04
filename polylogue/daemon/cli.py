"""Command line entrypoint for long-running Polylogue services."""

from __future__ import annotations

import asyncio
import os
from http.server import ThreadingHTTPServer
from pathlib import Path

import click

from polylogue.api import Polylogue
from polylogue.browser_capture.server import BrowserCaptureHTTPServer, make_server
from polylogue.core.json import dumps
from polylogue.daemon.browser_capture import browser_capture_command
from polylogue.daemon.status import daemon_status_payload, format_daemon_status_lines
from polylogue.sources.live import LiveWatcher, WatchSource
from polylogue.sources.live.watcher import default_sources


async def _ensure_fts_startup_readiness() -> None:
    """Check FTS coverage on daemon startup, rebuild if incomplete.

    A gap can only exist from pre-daemon historical data. Once caught up,
    the write path maintains FTS atomically via commit_archive_write_effects.
    This check runs once at startup and never again.
    """
    from polylogue.paths import archive_root, db_path
    from polylogue.storage.sqlite.connection_profile import open_connection

    db = db_path() or Path(archive_root()) / "polylogue.db"
    if not db.exists():
        return

    conn = None
    try:
        conn = open_connection(db, timeout=10.0)
        total = conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
        if total == 0:
            conn.close()
            return
        fts_count = conn.execute("SELECT COUNT(*) FROM messages_fts").fetchone()[0]
        gap = total - fts_count
        if gap <= 0:
            conn.close()
            return

        import logging

        logger = logging.getLogger(__name__)
        logger.warning(
            "daemon: FTS index incomplete (%d/%d messages, %.1f%% gap). Rebuilding once.",
            fts_count,
            total,
            100 * gap / total,
        )
        from polylogue.storage.fts.fts_lifecycle import rebuild_fts_index_sync

        rebuild_fts_index_sync(conn)
        conn.commit()
        new_count = conn.execute("SELECT COUNT(*) FROM messages_fts").fetchone()[0]
        logger.info("daemon: FTS rebuild complete — %d messages indexed.", new_count)
    except Exception:
        import logging

        logging.getLogger(__name__).warning("daemon: FTS startup check failed", exc_info=True)
    finally:
        if conn is not None:
            with __import__("contextlib").suppress(Exception):
                conn.close()


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


async def run_daemon_services(
    *,
    sources: tuple[WatchSource, ...],
    debounce_s: float,
    enable_watch: bool,
    enable_browser_capture: bool,
    browser_capture_host: str,
    browser_capture_port: int,
    browser_capture_spool_path: Path | None,
    browser_capture_allow_remote: bool = False,
    browser_capture_auth_token: str | None = None,
    browser_capture_extra_origins: tuple[str, ...] = (),
    enable_api: bool = False,
    api_host: str = "127.0.0.1",
    api_port: int = 8766,
) -> None:
    """Run configured daemon components until interrupted."""
    from polylogue.paths import archive_root

    # Ensure FTS is consistent on startup. A gap can only exist from
    # pre-daemon historical data; once caught up, the write path maintains
    # FTS atomically via commit_archive_write_effects().
    await _ensure_fts_startup_readiness()

    pidfile = Path(archive_root()) / "daemon.pid"

    # Prevent concurrent daemon instances
    if pidfile.exists():
        try:
            old_pid = int(pidfile.read_text().strip())
            os.kill(old_pid, 0)  # signal 0 = check if process exists
            raise RuntimeError(f"Daemon already running (PID {old_pid}). Stop it first or remove {pidfile}")
        except (OSError, ValueError):
            pidfile.unlink(missing_ok=True)

    pidfile.write_text(str(os.getpid()))

    api_server: ThreadingHTTPServer | None = None
    api_server_task: asyncio.Task[None] | None = None
    server: BrowserCaptureHTTPServer | None = None
    server_task: asyncio.Task[None] | None = None
    watcher: LiveWatcher | None = None
    watcher_task: asyncio.Task[None] | None = None
    tasks: list[asyncio.Task[None]] = []

    try:
        if enable_browser_capture:
            server = make_server(
                browser_capture_host,
                browser_capture_port,
                spool_path=browser_capture_spool_path,
                allow_remote=browser_capture_allow_remote,
                auth_token=browser_capture_auth_token,
                extra_origins=browser_capture_extra_origins,
            )
            server_task = asyncio.create_task(asyncio.to_thread(server.serve_forever, 0.5))
            tasks.append(server_task)

        if enable_api:
            from polylogue.daemon.http import DaemonAPIHandler, DaemonAPIHTTPServer

            api_server = DaemonAPIHTTPServer((api_host, api_port), DaemonAPIHandler)
            api_server_task = asyncio.create_task(asyncio.to_thread(api_server.serve_forever, 0.5))
            tasks.append(api_server_task)

        if enable_watch:
            async with Polylogue() as polylogue:
                watcher = LiveWatcher(polylogue, sources, debounce_s=debounce_s)
                watcher_task = asyncio.create_task(watcher.run())
                tasks.append(watcher_task)
                await asyncio.gather(*tasks)
        elif tasks:
            await asyncio.gather(*tasks)
    finally:
        if watcher is not None:
            watcher.stop()
        if server is not None:
            server.shutdown()
        if api_server is not None:
            api_server.shutdown()
        for task in tasks:
            if not task.done():
                task.cancel()
        await _drain_tasks(tasks)
        if server is not None:
            server.server_close()
        if api_server is not None:
            api_server.server_close()


async def _drain_tasks(tasks: list[asyncio.Task[None]]) -> None:
    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)


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
    for line in format_daemon_status_lines(payload):
        click.echo(line)


@main.command("run", help="Run configured long-lived daemon components.")
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
@click.option("--host", default="127.0.0.1", show_default=True, help="Browser-capture receiver host.")
@click.option("--port", default=8765, show_default=True, type=int, help="Browser-capture receiver port.")
@click.option("--spool", "spool_path", type=click.Path(path_type=Path), default=None)
@click.option("--no-watch", is_flag=True, help="Do not run the live source watcher.")
@click.option("--no-browser-capture", is_flag=True, help="Do not run the browser-capture receiver.")
@click.option(
    "--insecure-allow-remote",
    is_flag=True,
    default=False,
    help="Allow binding non-loopback addresses for browser-capture (requires --browser-capture-auth-token).",
)
@click.option(
    "--browser-capture-auth-token",
    default=None,
    help="Auth token for non-loopback browser-capture requests.",
)
@click.option(
    "--browser-capture-origin",
    "browser_capture_origins",
    multiple=True,
    default=(),
    help="Additional allowed browser-capture origin (repeatable).",
)
@click.option(
    "--enable-api",
    is_flag=True,
    default=False,
    help="Run the daemon HTTP API server.",
)
@click.option(
    "--api-host",
    default="127.0.0.1",
    show_default=True,
    help="Daemon API server host.",
)
@click.option(
    "--api-port",
    default=8766,
    show_default=True,
    type=int,
    help="Daemon API server port.",
)
def run_command(
    roots: tuple[Path, ...],
    debounce_s: float,
    host: str,
    port: int,
    spool_path: Path | None,
    no_watch: bool,
    no_browser_capture: bool,
    insecure_allow_remote: bool,
    browser_capture_auth_token: str | None,
    browser_capture_origins: tuple[str, ...],
    enable_api: bool,
    api_host: str,
    api_port: int,
) -> None:
    enable_watch = not no_watch
    enable_browser_capture = not no_browser_capture
    if not enable_watch and not enable_browser_capture and not enable_api:
        raise click.UsageError("at least one daemon component must be enabled")

    sources = tuple(WatchSource(name=p.name, root=p) for p in roots) if roots else default_sources()
    components = []
    if enable_watch:
        components.append(f"watch={len(sources)} source(s)")
    if enable_browser_capture:
        components.append(f"browser-capture=http://{host}:{port}")
    if enable_api:
        components.append(f"api=http://{api_host}:{api_port}")
    click.echo(f"Starting polylogued ({', '.join(components)}). Ctrl-C to stop.", err=True)

    try:
        asyncio.run(
            run_daemon_services(
                sources=sources,
                debounce_s=debounce_s,
                enable_watch=enable_watch,
                enable_browser_capture=enable_browser_capture,
                browser_capture_host=host,
                browser_capture_port=port,
                browser_capture_spool_path=spool_path,
                browser_capture_allow_remote=insecure_allow_remote,
                browser_capture_auth_token=browser_capture_auth_token,
                browser_capture_extra_origins=browser_capture_origins,
                enable_api=enable_api,
                api_host=api_host,
                api_port=api_port,
            )
        )
    except KeyboardInterrupt:
        click.echo("Stopping polylogued.", err=True)


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


__all__ = ["main", "run_command", "run_daemon_services", "run_live_watcher", "status_command", "watch_command"]
