"""Command line entrypoint for long-running Polylogue services."""

from __future__ import annotations

import asyncio
import atexit
import contextlib
import faulthandler
import fcntl
import os
import sqlite3
import sys
from contextlib import redirect_stdout
from datetime import UTC, datetime
from http.server import ThreadingHTTPServer
from pathlib import Path

import click

from polylogue.api import Polylogue
from polylogue.browser_capture.server import BrowserCaptureHTTPServer, make_server
from polylogue.core.degraded import DegradedReason, set_degraded
from polylogue.core.json import dumps
from polylogue.core.loopback import is_loopback_host
from polylogue.daemon.browser_capture import browser_capture_command
from polylogue.daemon.health import (
    HealthSeverity,
    HealthTier,
    _check_schema_version_fast,
    check_health,
    format_health_lines,
)
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


_EXPECTED_FTS_TRIGGERS: tuple[str, ...] = (
    "messages_fts_ai",
    "messages_fts_ad",
    "messages_fts_au",
    "action_events_fts_ai",
    "action_events_fts_ad",
    "action_events_fts_au",
)
"""Canonical FTS sync triggers (#1242).

Mirrors ``polylogue.daemon.health._EXPECTED_FTS_TRIGGERS``. Defined
here to keep the daemon startup path independent of the health-check
module's import surface.
"""


def _missing_fts_triggers_sync(conn: sqlite3.Connection) -> list[str]:
    placeholders = ",".join("?" for _ in _EXPECTED_FTS_TRIGGERS)
    rows = conn.execute(
        f"SELECT name FROM sqlite_master WHERE type='trigger' AND name IN ({placeholders})",
        _EXPECTED_FTS_TRIGGERS,
    ).fetchall()
    present = {row[0] for row in rows}
    return [name for name in _EXPECTED_FTS_TRIGGERS if name not in present]


async def _ensure_fts_startup_readiness() -> None:
    """Ensure FTS triggers and index are healthy on daemon startup.

    Three failure modes are recovered here:

    1. FTS table missing entirely (historical rows pre-date FTS) →
       rebuild from messages/action_events.
    2. FTS table exists but is empty while messages exist → rebuild.
    3. FTS triggers missing (the SIGKILL-during-bulk-suspend signature
       called out in ``docs/internals.md`` "FTS5 Model → Risk") →
       restore triggers and rebuild the FTS index. The bulk-ingest
       path drops these triggers as DDL (auto-committed); a SIGKILL
       between the drop and the matching restore leaves them gone
       across process death, and subsequent writes silently bypass the
       FTS index. See #1242.
    """
    from polylogue.paths import db_path
    from polylogue.storage.sqlite.connection_profile import open_connection

    db = db_path()
    if not db.exists():
        return

    conn: sqlite3.Connection | None = None
    try:
        conn = open_connection(db, timeout=10.0)
        row = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='messages_fts'").fetchone()
        has_fts_table = row is not None
        has_indexable_messages = (
            conn.execute("SELECT 1 FROM messages WHERE text IS NOT NULL LIMIT 1").fetchone() is not None
        )

        from polylogue.storage.fts.fts_lifecycle import (
            ensure_fts_index_sync,
            rebuild_fts_index_sync,
            restore_fts_triggers_sync,
        )

        if not has_fts_table:
            logger.warning("daemon: message FTS table missing. Rebuilding once.")
            rebuild_fts_index_sync(conn)
            conn.commit()
            logger.info("daemon: FTS rebuild complete.")
            return

        missing_triggers = _missing_fts_triggers_sync(conn)
        if missing_triggers:
            logger.warning(
                "daemon: FTS triggers missing on startup (%s). "
                "SIGKILL-during-bulk-suspend signature; restoring and rebuilding FTS index.",
                ", ".join(missing_triggers),
            )
            restore_fts_triggers_sync(conn)
            rebuild_fts_index_sync(conn)
            conn.commit()
            logger.info("daemon: FTS trigger restore + rebuild complete.")
            return

        ensure_fts_index_sync(conn)
        has_indexed_messages = conn.execute("SELECT 1 FROM messages_fts_docsize LIMIT 1").fetchone() is not None
        if has_indexable_messages and not has_indexed_messages:
            logger.warning("daemon: message FTS is empty while archive has messages. Rebuilding once.")
            rebuild_fts_index_sync(conn)
            conn.commit()
            logger.info("daemon: FTS rebuild complete.")
            return

        conn.commit()
    except Exception:
        logger.warning("daemon: FTS startup check failed", exc_info=True)
    finally:
        if conn is not None:
            with contextlib.suppress(Exception):
                conn.close()


async def _periodic_wal_checkpoint() -> None:
    """Run WAL checkpoint every 5 minutes to keep the WAL file bounded."""
    from polylogue.paths import db_path
    from polylogue.storage.sqlite.connection_profile import open_connection

    db = db_path()
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
    from polylogue.paths import db_path
    from polylogue.storage.sqlite.connection_profile import open_connection

    db = db_path()
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


async def _periodic_db_optimize() -> None:
    """Run SQLite PRAGMA optimize once daily to keep query plans current.

    On a 60 GB archive with millions of rows, the query planner's
    internal statistics drift as the table sizes change.  PRAGMA optimize
    is a lightweight maintenance pass — it only runs ANALYZE on tables
    that need it and never blocks reads.  It is NOT a VACUUM and does
    not rewrite the file.
    """
    from polylogue.paths import db_path
    from polylogue.storage.sqlite.connection_profile import open_connection

    db = db_path()
    while True:
        await asyncio.sleep(86_400)  # 24 hours
        if not db.exists():
            continue
        try:
            conn = open_connection(db, timeout=30.0)
            try:
                conn.execute("PRAGMA optimize")
                logger.info("daemon: DB optimize completed")
            finally:
                conn.close()
        except Exception:
            logger.warning("daemon: DB optimize failed", exc_info=True)


async def _periodic_convergence_check(
    sources: tuple[WatchSource, ...],
) -> None:
    """Periodically verify FTS coverage, insight freshness, and repair gaps.

    This is a safety net that runs every 10 minutes. The write path
    maintains these atomically via commit_archive_write_effects(),
    so gaps should be rare — this catches edge cases like crash recovery.
    """
    from polylogue.paths import db_path
    from polylogue.storage.sqlite.connection_profile import open_connection

    db = db_path()
    while True:
        await asyncio.sleep(600)  # 10 minutes
        if not db.exists():
            continue
        try:
            repaired = await asyncio.to_thread(_drain_convergence_debt_once, db)
            if repaired:
                logger.info("convergence: retried %d derived debt item(s)", repaired)
            conn = open_connection(db, timeout=5.0)
            try:
                total_msgs = conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
                if total_msgs == 0:
                    continue
                fts_count = (
                    conn.execute("SELECT COUNT(*) FROM messages_fts_docsize").fetchone()[0]
                    if conn.execute(
                        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='messages_fts_docsize'"
                    ).fetchone()
                    else 0
                )
                gap = total_msgs - fts_count
                if gap > 0:
                    logger.warning(
                        "convergence: FTS gap detected (%d/%d messages, %.1f%%). Rebuilding.",
                        fts_count,
                        total_msgs,
                        100 * gap / total_msgs,
                    )
                    from polylogue.storage.fts.fts_lifecycle import rebuild_fts_index_sync

                    rebuild_fts_index_sync(conn)
                    conn.commit()
                    logger.info("convergence: FTS rebuild complete — %d messages indexed.", total_msgs)
            finally:
                conn.close()
        except Exception:
            logger.warning("convergence: check failed", exc_info=True)


def _drain_convergence_debt_once(db: Path, *, limit: int = 100) -> int:
    """Retry due derived convergence debt without rereading source payloads."""
    from polylogue.daemon.convergence import DaemonConverger
    from polylogue.daemon.convergence_stages import make_default_convergence_stages
    from polylogue.sources.live.cursor import CursorStore

    cursor = CursorStore(db)
    now = datetime.now(UTC)
    due_debt = [
        debt
        for debt in cursor.list_convergence_debt(limit=limit)
        if debt.subject_type in {"source_path", "conversation_id"} and _debt_retry_due(debt, now=now)
    ]
    if not due_debt:
        return 0
    conversation_ids = tuple(
        dict.fromkeys(debt.subject_id for debt in due_debt if debt.subject_type == "conversation_id")
    )
    paths = tuple(dict.fromkeys([Path(debt.subject_id) for debt in due_debt if debt.subject_type == "source_path"]))
    if not paths and not conversation_ids:
        return 0
    converger = DaemonConverger(stages=make_default_convergence_stages(db), max_workers=2)
    path_states, _path_timings = converger.converge_batch(paths)
    conversation_states, _conversation_timings = converger.converge_conversations(conversation_ids)
    retried = 0
    for debt in due_debt:
        subject_states: list[object | None]
        if debt.subject_type == "conversation_id":
            subject_states = [conversation_states.get(debt.subject_id)]
        else:
            subject_states = [path_states.get(Path(debt.subject_id))]
        converged = all(state is not None and bool(getattr(state, "converged", False)) for state in subject_states)
        if converged:
            cursor.clear_convergence_debt(subject_type=debt.subject_type, subject_id=debt.subject_id)
            retried += 1
            continue
        failed_stages: tuple[str, ...] = ()
        last_error: object = None
        for state in subject_states:
            stages = getattr(state, "stages", {}) if state is not None else {}
            last_error = getattr(state, "last_error", None) if state is not None else None
            failed_stages = _failed_convergence_stage_names(stages)
            if failed_stages:
                break
        if not failed_stages:
            failed_stages = ("convergence",)
        cursor.clear_convergence_debt(subject_type=debt.subject_type, subject_id=debt.subject_id)
        for stage in failed_stages:
            cursor.record_convergence_debt(
                stage=stage,
                subject_type=debt.subject_type,
                subject_id=debt.subject_id,
                error=last_error if isinstance(last_error, str) and last_error else "retry did not converge",
            )
        retried += 1
    return retried


def _debt_retry_due(debt: object, *, now: datetime) -> bool:
    next_retry_at = getattr(debt, "next_retry_at", None)
    if not isinstance(next_retry_at, str) or not next_retry_at:
        return True
    try:
        retry_at = datetime.fromisoformat(next_retry_at)
    except ValueError:
        return True
    if retry_at.tzinfo is None:
        retry_at = retry_at.replace(tzinfo=UTC)
    return retry_at <= now


def _failed_convergence_stage_names(stages: object) -> tuple[str, ...]:
    if not isinstance(stages, dict):
        return ()
    failed: list[str] = []
    for stage_name, stage_state in stages.items():
        state_value = getattr(stage_state, "value", stage_state)
        if state_value not in {"done", "skipped"}:
            failed.append(str(stage_name))
    return tuple(failed)


async def _periodic_health_check() -> None:
    """Run periodic health checks with config-driven notification backend.

    Health check tiers and interval are read from PolylogueConfig.
    Notifications are sent through the configured notification backend.
    """
    while True:
        from polylogue.config import load_polylogue_config

        cfg = load_polylogue_config()
        interval = cfg.health_check_interval_s
        tier_str = cfg.health_check_tiers

        # Resolve health tiers from config.
        tier_map = {
            "fast": HealthTier.FAST,
            "medium": HealthTier.MEDIUM,
            "expensive": HealthTier.EXPENSIVE,
        }
        tiers: set[HealthTier] = set()
        for t in tier_str.split(","):
            t = t.strip()
            if t in tier_map:
                tiers.add(tier_map[t])
        if not tiers:
            tiers = {HealthTier.FAST, HealthTier.MEDIUM}

        await asyncio.sleep(interval)
        try:
            from polylogue.daemon.health import check_health
            from polylogue.daemon.notifications import send_notifications

            health = check_health(tiers=tiers)
            if health.overall_status != "ok":
                send_notifications(health.alerts, config=cfg.raw)
        except Exception:
            logger.warning("health: periodic check failed", exc_info=True)


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


def _emit_live_batch_event(kind: str, payload: dict[str, object]) -> None:
    """Persist a live-ingest batch event and fan out granular #1204 topics.

    The legacy ``ingestion_batch`` kind is preserved verbatim for existing
    consumers (status views, polling fallback). When the batch payload
    carries succeeded counts we additionally emit per-topic events
    (``conversation.appended`` / ``message.appended``) so the reader can
    subscribe selectively and animate just-touched rows.
    """
    from polylogue.daemon.events import (
        emit_conversation_appended,
        emit_daemon_event,
        emit_message_appended,
    )

    emit_daemon_event(kind, payload=payload)

    if kind != "ingestion_batch":
        return

    succeeded_raw = payload.get("succeeded_file_count", 0)
    failed_raw = payload.get("failed_file_count", 0)
    succeeded = int(succeeded_raw) if isinstance(succeeded_raw, int | float) else 0
    failed = int(failed_raw) if isinstance(failed_raw, int | float) else 0
    if succeeded <= 0:
        return

    # ``source_group_count`` is the only per-source breakdown the batch
    # payload carries today. Emit one aggregate granular event per batch;
    # source-level fan-out can be added once the metrics payload exposes
    # per-source success counts (deferred — tracked under #1204 follow-ups).
    emit_conversation_appended(
        source_name=None,
        succeeded_file_count=succeeded,
        failed_file_count=failed,
    )
    emit_message_appended(
        conversation_id=None,
        source_name=None,
        appended_count=succeeded,
    )


async def run_live_watcher(
    *,
    sources: tuple[WatchSource, ...],
    debounce_s: float,
) -> None:
    async with Polylogue() as polylogue:
        watcher = LiveWatcher(polylogue, sources, debounce_s=debounce_s, event_emitter=_emit_live_batch_event)
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
    api_auth_token: str | None = None,
) -> None:
    """Run configured daemon components until interrupted."""
    from polylogue.paths import archive_root

    global _pidfile_path

    # Non-localhost API binding requires explicit opt-in AND an auth token.
    if enable_api and not is_loopback_host(api_host):
        if not browser_capture_allow_remote:
            raise click.UsageError(
                f"--api-host={api_host} is not a loopback address. "
                f"Add --insecure-allow-remote to accept the risk of exposing the daemon API."
            )
        if not api_auth_token:
            raise click.UsageError(
                f"--api-host={api_host} with --insecure-allow-remote requires --api-auth-token. "
                f"Remote binding without authentication is not supported."
            )

    logger.info("daemon started")

    # Schema preflight runs FIRST, before any DB-touching startup task. A
    # mismatched runtime/db combination must not even open the DB for FTS or
    # heartbeat queries — that is the IO cost #1003 is meant to avoid.
    schema_alert = _check_schema_version_fast()
    watcher_blocked = enable_watch and schema_alert.severity == HealthSeverity.CRITICAL
    if watcher_blocked:
        logger.error(
            "daemon: schema preflight CRITICAL — %s. Refusing to start the live watcher; "
            "HTTP/health surfaces remain available so this state is observable.",
            schema_alert.message,
        )
        set_degraded(
            DegradedReason(
                code="schema_incompatible",
                message=schema_alert.message,
                detail={"check_name": schema_alert.check_name},
            )
        )

    # Ensure FTS is consistent on startup. A gap can only exist from
    # pre-daemon historical data; once caught up, the write path maintains
    # FTS atomically via commit_archive_write_effects(). Skipped when the
    # schema is structurally unusable for this runtime — opening the DB at
    # all is what the preflight is preventing.
    if not watcher_blocked:
        await _ensure_fts_startup_readiness()

    pidfile = Path(archive_root()) / "daemon.pid"
    pidfile_fd: int | None = None

    # Prevent concurrent daemon instances: verify existing pidfile, then
    # acquire an advisory flock.
    if pidfile.exists():
        if _verify_pidfile(pidfile):
            old_pid = pidfile.read_text().strip()
            raise RuntimeError(f"Daemon already running (PID {old_pid}). Stop it first or remove {pidfile}")
        pidfile.unlink(missing_ok=True)

    pidfile_fd = _acquire_pidfile(pidfile)
    # Only register the pidfile for atexit cleanup AFTER lock acquisition.
    # Setting _pidfile_path before _acquire_pidfile() means a failed lock
    # attempt by an ephemeral instance would still atexit-unlink the live
    # daemon's pidfile.
    _pidfile_path = pidfile

    # Periodic maintenance tasks.
    wal_task = asyncio.create_task(_periodic_wal_checkpoint())
    heartbeat_task = asyncio.create_task(_periodic_heartbeat())
    convergence_task = asyncio.create_task(_periodic_convergence_check(sources))
    health_task = asyncio.create_task(_periodic_health_check())
    db_optimize_task = asyncio.create_task(_periodic_db_optimize())
    maintenance_tasks = [wal_task, heartbeat_task, convergence_task, health_task, db_optimize_task]

    api_server: ThreadingHTTPServer | None = None
    api_server_task: asyncio.Task[None] | None = None
    server: BrowserCaptureHTTPServer | None = None
    server_task: asyncio.Task[None] | None = None
    watcher: LiveWatcher | None = None
    watcher_task: asyncio.Task[None] | None = None
    converger: DaemonConverger | None = None
    tasks: list[asyncio.Task[None]] = []

    try:
        # Start the post-ingest convergence engine. The live watcher owns
        # batched source ingestion; these stages repair archive indexes and
        # refresh derived state after successful writes.
        from polylogue.daemon.convergence import DaemonConverger
        from polylogue.daemon.convergence_stages import make_default_convergence_stages
        from polylogue.paths import db_path

        _db = db_path()

        converger = DaemonConverger(
            stages=make_default_convergence_stages(_db),
            max_workers=2,
        )
        await converger.start()

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

            api_server = DaemonAPIHTTPServer(
                (api_host, api_port),
                DaemonAPIHandler,
                auth_token=api_auth_token,
                api_host=api_host,
            )
            api_server_task = asyncio.create_task(asyncio.to_thread(api_server.serve_forever, 0.5))
            tasks.append(api_server_task)

        # Preflight already ran at the top of run_daemon_services (see
        # ``watcher_blocked`` above); reuse that result.
        if enable_watch and not watcher_blocked:
            async with Polylogue() as polylogue:
                watcher = LiveWatcher(
                    polylogue,
                    sources,
                    debounce_s=debounce_s,
                    converger=converger,
                    event_emitter=_emit_live_batch_event,
                )
                watcher_task = asyncio.create_task(watcher.run())
                tasks.append(watcher_task)
                all_tasks = tasks + maintenance_tasks
                await asyncio.gather(*all_tasks)
        elif tasks:
            # Watcher disabled or preflight-blocked: keep HTTP/health and
            # other components serving so operators see the degraded state.
            all_tasks = tasks + maintenance_tasks
            await asyncio.gather(*all_tasks)
        else:
            await asyncio.gather(*maintenance_tasks)
    finally:
        if watcher is not None:
            watcher.stop()
        if converger is not None:
            await converger.stop()
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
            cancel_pending = getattr(watcher, "cancel_pending", None)
            if callable(cancel_pending):
                cancel_pending()

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
    configure_logging()
    if output_format == "json":
        with redirect_stdout(sys.stderr):
            payload = daemon_status_payload(browser_capture_spool_path=spool_path)
        click.echo(dumps(payload))
        return
    payload = daemon_status_payload(browser_capture_spool_path=spool_path)
    for line in format_daemon_status_lines(payload):
        click.echo(line)


@main.command("health", help="Run tiered daemon health checks.")
@click.option(
    "--tier",
    "tiers",
    type=click.Choice(["fast", "medium", "expensive"]),
    multiple=True,
    default=None,
    help="Run specific health check tiers (repeatable). Default: fast + medium.",
)
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["json"]),
    default=None,
    help="Output format.",
)
@click.option(
    "--expensive",
    "include_expensive",
    is_flag=True,
    default=False,
    help="Include expensive checks (DB integrity).",
)
def health_command(
    tiers: tuple[str, ...],
    output_format: str | None,
    include_expensive: bool,
) -> None:
    """Run tiered daemon health checks.

    By default runs FAST + MEDIUM checks. Use --tier to select specific
    tiers or --expensive to add the EXPENSIVE tier.
    """
    configure_logging()

    if tiers:
        health_tiers: set[HealthTier] = {HealthTier(t) for t in tiers}
    else:
        health_tiers = {HealthTier.FAST, HealthTier.MEDIUM}
        if include_expensive:
            health_tiers.add(HealthTier.EXPENSIVE)

    health = check_health(tiers=health_tiers)

    if output_format == "json":
        click.echo(health.model_dump_json(indent=2))
        if health.overall_status.value in ("error", "critical"):
            raise SystemExit(1)
        return

    for line in format_health_lines(health):
        click.echo(line)
    if health.overall_status.value in ("error", "critical"):
        raise SystemExit(1)


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
    "--no-api",
    is_flag=True,
    default=False,
    help="Disable the daemon HTTP API server (web reader + /api/*).",
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
@click.option(
    "--api-auth-token",
    default=None,
    help="Daemon API auth token (generate one if not provided; write to archive root).",
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
    no_api: bool,
    api_host: str,
    api_port: int,
    api_auth_token: str | None,
) -> None:
    """Run configured daemon components.

    This is the entry point for the polylogued systemd service.
    """
    faulthandler.enable()
    configure_logging()

    enable_watch = not no_watch
    enable_browser_capture = not no_browser_capture
    enable_api = not no_api
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
                api_auth_token=api_auth_token,
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
    "health_command",
    "main",
    "run_command",
    "run_daemon_services",
    "run_live_watcher",
    "status_command",
    "watch_command",
]
