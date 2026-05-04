"""Command line entrypoint for long-running Polylogue services."""

from __future__ import annotations

import asyncio
import atexit
import contextlib
import faulthandler
import fcntl
import os
from http.server import ThreadingHTTPServer
from pathlib import Path

import click

from polylogue.api import Polylogue
from polylogue.browser_capture.server import BrowserCaptureHTTPServer, make_server
from polylogue.core.json import dumps
from polylogue.daemon.browser_capture import browser_capture_command
from polylogue.daemon.status import daemon_status_payload, format_daemon_status_lines
from polylogue.logging import configure_logging, get_logger
from polylogue.sources.live import LiveWatcher, WatchSource
from polylogue.sources.live.watcher import default_sources

logger = get_logger(__name__)

# Track the pidfile path for atexit cleanup.
_pidfile_path: Path | None = None


def _cleanup_pidfile() -> None:
    """Remove the daemon pidfile on exit."""
    global _pidfile_path
    if _pidfile_path is not None and _pidfile_path.exists():
        with contextlib.suppress(OSError):
            _pidfile_path.unlink(missing_ok=True)


def _verify_pidfile(pidfile: Path) -> bool:
    """Verify that the pidfile refers to a running polylogued process.

    Reads /proc/<PID>/cmdline and checks it contains "polylogued".
    Returns True if the pidfile is valid (process is alive and is polylogued).
    """
    try:
        old_pid = int(pidfile.read_text().strip())
    except (ValueError, OSError):
        return False

    try:
        os.kill(old_pid, 0)  # signal 0 = check if process exists
    except OSError:
        return False

    # Verify the PID actually belongs to a polylogued process.
    try:
        cmdline = Path(f"/proc/{old_pid}/cmdline").read_bytes()
        return b"polylogued" in cmdline
    except OSError:
        return False


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
        logger.warning("daemon: FTS startup check failed", exc_info=True)
    finally:
        if conn is not None:
            with __import__("contextlib").suppress(Exception):
                conn.close()


async def _periodic_wal_checkpoint() -> None:
    """Run WAL checkpoint every 5 minutes to keep the WAL file bounded."""
    from polylogue.paths import archive_root, db_path
    from polylogue.storage.sqlite.connection_profile import open_connection

    db = db_path() or Path(archive_root()) / "polylogue.db"
    while True:
        await asyncio.sleep(300)  # 5 minutes
        if not db.exists():
            continue
        try:
            conn = open_connection(db, timeout=5.0)
            try:
                conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
                logger.debug("daemon: WAL checkpoint completed")
            finally:
                conn.close()
        except Exception:
            logger.warning("daemon: WAL checkpoint failed", exc_info=True)


async def _periodic_heartbeat() -> None:
    """Log daemon heartbeat with archive stats every 15 minutes."""
    from polylogue.paths import archive_root, db_path
    from polylogue.storage.sqlite.connection_profile import open_connection

    db = db_path() or Path(archive_root()) / "polylogue.db"
    while True:
        await asyncio.sleep(900)  # 15 minutes
        if not db.exists():
            continue
        try:
            conn = open_connection(db, timeout=5.0)
            try:
                n_conv = conn.execute("SELECT COUNT(*) FROM conversations").fetchone()[0]
                n_msg = conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
                logger.info(
                    "daemon heartbeat: %d conversations, %d messages indexed",
                    n_conv,
                    n_msg,
                )
            finally:
                conn.close()
        except Exception:
            logger.warning("daemon: heartbeat query failed", exc_info=True)


def _acquire_pidfile(pidfile: Path) -> int:
    """Acquire an advisory lock on the pidfile via fcntl.flock.

    Returns the open fd. The lock is held until process exit or explicit close.
    """
    pidfile.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(pidfile, os.O_RDWR | os.O_CREAT | os.O_TRUNC, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as err:
        os.close(fd)
        raise RuntimeError(f"Could not acquire lock on {pidfile} — another daemon may be running") from err
    os.write(fd, str(os.getpid()).encode())
    os.fsync(fd)
    return fd


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

    global _pidfile_path

    logger.info("daemon started")

    # Ensure FTS is consistent on startup. A gap can only exist from
    # pre-daemon historical data; once caught up, the write path maintains
    # FTS atomically via commit_archive_write_effects().
    await _ensure_fts_startup_readiness()

    pidfile = Path(archive_root()) / "daemon.pid"
    _pidfile_path = pidfile
    pidfile_fd: int | None = None

    # Prevent concurrent daemon instances: verify existing pidfile, then
    # acquire an advisory flock.
    if pidfile.exists():
        if _verify_pidfile(pidfile):
            old_pid = pidfile.read_text().strip()
            raise RuntimeError(f"Daemon already running (PID {old_pid}). Stop it first or remove {pidfile}")
        pidfile.unlink(missing_ok=True)

    pidfile_fd = _acquire_pidfile(pidfile)

    # Periodic maintenance tasks.
    wal_task = asyncio.create_task(_periodic_wal_checkpoint())
    heartbeat_task = asyncio.create_task(_periodic_heartbeat())
    maintenance_tasks = [wal_task, heartbeat_task]

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
            from polylogue.daemon.http import (
                DaemonAPIHandler,
                DaemonAPIHTTPServer,
            )

            api_server = DaemonAPIHTTPServer((api_host, api_port), DaemonAPIHandler)
            api_server_task = asyncio.create_task(asyncio.to_thread(api_server.serve_forever, 0.5))
            tasks.append(api_server_task)

        if enable_watch:
            async with Polylogue() as polylogue:
                watcher = LiveWatcher(polylogue, sources, debounce_s=debounce_s)
                watcher_task = asyncio.create_task(watcher.run())
                tasks.append(watcher_task)
                all_tasks = tasks + maintenance_tasks
                await asyncio.gather(*all_tasks)
        elif tasks:
            all_tasks = tasks + maintenance_tasks
            await asyncio.gather(*all_tasks)
        else:
            await asyncio.gather(*maintenance_tasks)
    finally:
        if watcher is not None:
            watcher.stop()
        if server is not None:
            server.shutdown()
        if api_server is not None:
            api_server.shutdown()

        # Cancel all component tasks.
        for task in tasks:
            if not task.done():
                task.cancel()

        # Cancel orphaned debounced watcher child tasks.
        if watcher is not None:
            watcher.cancel_pending()

        # Drain component tasks with a timeout.
        drained_results = await _drain_tasks(tasks, timeout=5.0)
        _report_drain_exceptions(drained_results)

        # Cancel and drain maintenance tasks.
        for mt in maintenance_tasks:
            if not mt.done():
                mt.cancel()
        await _drain_tasks(maintenance_tasks, timeout=5.0)

        if server is not None:
            server.server_close()
        if api_server is not None:
            api_server.server_close()

        # Clean up pidfile and release advisory lock.
        if pidfile_fd is not None:
            with contextlib.suppress(OSError):
                os.close(pidfile_fd)
        _cleanup_pidfile()

    logger.info("daemon stopped")


async def _drain_tasks(tasks: list[asyncio.Task[None]], *, timeout: float = 5.0) -> list[BaseException | None]:
    """Drain tasks, returning collected exceptions.

    Returns a list parallel to *tasks*, where each element is the
    exception from that task (or None if it completed cleanly).
    """
    if not tasks:
        return []
    results = await asyncio.wait_for(
        asyncio.gather(*tasks, return_exceptions=True),
        timeout=timeout,
    )
    return [r if isinstance(r, BaseException) else None for r in results]


def _report_drain_exceptions(results: list[BaseException | None]) -> None:
    """Log any exceptions found during task draining."""
    for exc in results:
        if exc is not None:
            logger.warning("daemon: task raised during shutdown: %s", exc)


@click.group(help="Run long-lived Polylogue local services.")
def main() -> None:
    pass


main.add_command(browser_capture_command)


@main.command("status", help="Show configured daemon component status.")
@click.option(
    "--spool",
    "spool_path",
    type=click.Path(path_type=Path),
    default=None,
)
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["json"]),
    default=None,
    help="Output format.",
)
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
    help="Override watch root (repeatable).",
)
@click.option(
    "--debounce-s",
    type=float,
    default=2.0,
    show_default=True,
    help="Quiet-period (seconds) before parsing a modified file.",
)
@click.option(
    "--host",
    default="127.0.0.1",
    show_default=True,
    help="Browser-capture receiver host.",
)
@click.option(
    "--port",
    default=8765,
    show_default=True,
    type=int,
    help="Browser-capture receiver port.",
)
@click.option(
    "--spool",
    "spool_path",
    type=click.Path(path_type=Path),
    default=None,
)
@click.option(
    "--no-watch",
    is_flag=True,
    help="Do not run the live source watcher.",
)
@click.option(
    "--no-browser-capture",
    is_flag=True,
    help="Do not run the browser-capture receiver.",
)
@click.option(
    "--insecure-allow-remote",
    is_flag=True,
    default=False,
    help="Allow non-loopback browser-capture addresses.",
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
    """Run configured daemon components.

    This is the entry point for the polylogued systemd service.
    """
    faulthandler.enable()
    configure_logging()

    enable_watch = not no_watch
    enable_browser_capture = not no_browser_capture
    if not enable_watch and not enable_browser_capture and not enable_api:
        raise click.UsageError("at least one daemon component must be enabled")

    atexit.register(_cleanup_pidfile)

    sources = tuple(WatchSource(name=p.name, root=p) for p in roots) if roots else default_sources()
    components = []
    if enable_watch:
        components.append(f"watch={len(sources)} source(s)")
    if enable_browser_capture:
        components.append(f"browser-capture=http://{host}:{port}")
    if enable_api:
        components.append(f"api=http://{api_host}:{api_port}")
    click.echo(
        f"Starting polylogued ({', '.join(components)}). Ctrl-C to stop.",
        err=True,
    )

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
    help="Override watch root (repeatable).",
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


__all__ = [
    "main",
    "run_command",
    "run_daemon_services",
    "run_live_watcher",
    "status_command",
    "watch_command",
]
