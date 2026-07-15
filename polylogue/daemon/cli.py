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
import threading
from contextlib import redirect_stdout
from datetime import UTC, datetime
from http.server import ThreadingHTTPServer
from pathlib import Path
from typing import TYPE_CHECKING, Any

import click

from polylogue.api import Polylogue
from polylogue.browser_capture.receiver import resolve_receiver_auth_token
from polylogue.browser_capture.server import BrowserCaptureHTTPServer, make_server
from polylogue.core.degraded import DegradedReason, set_degraded
from polylogue.core.json import dumps
from polylogue.core.loopback import bind_hosts_overlap, is_loopback_host
from polylogue.daemon.browser_capture import browser_capture_command

# FTS startup readiness extracted to ``polylogue.daemon.fts_startup`` (#1614).
# ``_ensure_fts_startup_readiness`` is re-exported here because the daemon's
# coroutine wiring and the ``test_daemon_cli_remote_bind`` monkeypatch site
# both reach for it via the historical ``polylogue.daemon.cli`` path. The
# other helpers (active/missing/table_exists/record_freshness) are called
# only inside ``fts_startup.ensure_fts_startup_readiness_sync`` and their
# tests patch the new module path directly.
from polylogue.daemon.fts_startup import (
    ensure_fts_startup_readiness as _ensure_fts_startup_readiness,
)
from polylogue.daemon.fts_startup import (
    ensure_fts_startup_readiness_sync as _ensure_fts_startup_readiness_sync,
)
from polylogue.daemon.health import (
    HealthSeverity,
    HealthTier,
    _check_schema_version_fast,
    check_health,
    format_health_lines,
    resolve_health_tiers,
)
from polylogue.daemon.lineage_startup import (
    ensure_lineage_startup_readiness_sync as _ensure_lineage_startup_readiness_sync,
)
from polylogue.daemon.status import daemon_status_payload, format_daemon_status_lines
from polylogue.daemon.write_coordinator import (
    DaemonWriteCoordinator,
    DaemonWriteThreadBridge,
    daemon_write_coordinator,
)
from polylogue.logging import configure_logging, get_logger
from polylogue.sources.live import LiveWatcher, WatchSource
from polylogue.sources.live.sqlite_locking import is_transient_sqlite_lock
from polylogue.sources.live.watcher import INBOX_SOURCE_SUFFIXES, default_sources
from polylogue.version import POLYLOGUE_VERSION

if TYPE_CHECKING:
    from polylogue.daemon.lifecycle import DaemonLifecycle

logger = get_logger(__name__)
_CONVERGENCE_DEBT_RETRY_INTERVAL_SECONDS = 60
_RAW_MATERIALIZATION_CONVERGENCE_INTERVAL_SECONDS = 30
_RAW_MATERIALIZATION_CONVERGENCE_BATCH_LIMIT = 25


async def _run_startup_fts_readiness(coordinator: DaemonWriteCoordinator) -> None:
    """Run the real startup FTS writer on an exit-safe coordinator thread."""
    await coordinator.run_sync("startup.fts_readiness", _ensure_fts_startup_readiness_sync)


async def _run_startup_lineage_readiness(coordinator: DaemonWriteCoordinator) -> int:
    """Run the real startup lineage writer on an exit-safe coordinator thread."""
    return await coordinator.run_sync("startup.lineage_readiness", _ensure_lineage_startup_readiness_sync)


_SESSION_INSIGHT_CONVERGENCE_INTERVAL_SECONDS = 60
_SESSION_INSIGHT_CONVERGENCE_BATCH_LIMIT = 100
_SESSION_INSIGHT_CONVERGENCE_BURST_LIMIT = 10
_SESSION_INSIGHT_CONVERGENCE_BURST_PAUSE_SECONDS = 1
_DRIVE_SOURCE_CATCHUP_INTERVAL_SECONDS = 3600
_BLOB_REFERENCE_RESTORE_CONVERGENCE_BATCH_LIMIT = 25

# Track the pidfile path for atexit cleanup.
_pidfile_path: Path | None = None
_daemon_lifecycle: DaemonLifecycle | None = None


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


def _enable_faulthandler_if_supported() -> None:
    """Enable faulthandler when stderr exposes a real file descriptor."""
    with contextlib.suppress(Exception):
        faulthandler.enable()


def _watch_sources_from_roots(
    roots: tuple[Path, ...],
    *,
    browser_capture_spool_path: Path | None = None,
) -> tuple[WatchSource, ...]:
    """Build watch sources for explicit daemon roots.

    An explicit ``--root`` normally means a JSONL session tree. The archive
    inbox is different: ``polylogue import`` stages approved exports there,
    including ChatGPT ``.json`` files and zipped takeouts, so an isolated
    daemon pointed at that inbox must keep the same suffix contract as the
    default inbox source.
    """
    if not roots:
        sources = list(default_sources())
        if browser_capture_spool_path is not None:
            spool = browser_capture_spool_path.expanduser()
            sources = [source for source in sources if source.name != "browser-capture"]
            sources.append(WatchSource(name="browser-capture", root=spool, suffixes=(".json",)))
        return tuple(sources)

    from polylogue.paths import archive_root, browser_capture_spool_root

    inbox_root = (archive_root() / "inbox").resolve(strict=False)
    browser_root = (
        browser_capture_spool_path.expanduser()
        if browser_capture_spool_path is not None
        else browser_capture_spool_root()
    ).resolve(strict=False)
    explicit_sources: list[WatchSource] = []
    for root in roots:
        resolved = root.resolve(strict=False)
        if resolved == inbox_root:
            explicit_sources.append(WatchSource(name="inbox", root=root, suffixes=INBOX_SOURCE_SUFFIXES))
        elif resolved == browser_root:
            explicit_sources.append(WatchSource(name="browser-capture", root=root, suffixes=(".json",)))
        else:
            explicit_sources.append(WatchSource(name=root.name, root=root, suffixes=(".jsonl",)))
    return tuple(explicit_sources)


def _active_index_db_path() -> Path:
    """Return the currently active archive database for daemon maintenance."""
    from polylogue.paths import active_index_db_path

    return active_index_db_path()


def _heartbeat_counts(db: Path) -> tuple[int, int, str]:
    """Return session and message counts for the current archive."""
    from polylogue.storage.sqlite.connection_profile import open_readonly_connection

    conn = open_readonly_connection(db, timeout=5.0)
    try:
        tables = {
            str(row[0])
            for row in conn.execute(
                """
                SELECT name
                FROM sqlite_master
                WHERE type = 'table'
                  AND name IN ('sessions', 'sessions', 'messages')
                """
            ).fetchall()
        }
        if "sessions" in tables:
            n_sessions = int(conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0] or 0)
            n_messages = int(conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0] or 0)
            return n_sessions, n_messages, "sessions"
        if "sessions" in tables:
            n_sessions = int(conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0] or 0)
            n_messages = int(conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0] or 0)
            return n_sessions, n_messages, "sessions"
        return 0, 0, "sessions" if db.name == "index.db" else "sessions"
    finally:
        conn.close()


def _mark_interrupted_live_ingest_attempts_on_shutdown() -> None:
    """Close current-process running ingest attempts during graceful shutdown."""
    from polylogue.sources.live.cursor import CursorStore

    # CursorStore construction runs the same interrupted-attempt recovery used
    # on daemon startup. Calling it at shutdown keeps Ctrl-C dogfood runs from
    # leaving false in-flight work until the next daemon start.
    CursorStore(_active_index_db_path())


async def _configure_fts_automerge() -> None:
    """Persist FTS5 automerge=0 for all surfaces at daemon startup (#1851).

    FTS5's default automerge=8 merges existing (large) segments whenever a
    write accumulates ≥8 level-0 segments.  On a mature archive this causes
    ~8–12 MiB of WAL writes per small ingest batch.  Setting automerge=0
    disables per-write merging; the periodic ``_periodic_fts_merge`` loop
    amortises merge cost over time instead.
    """
    db = _active_index_db_path()
    if not db.exists():
        return
    try:
        await daemon_write_coordinator().run_sync("startup.fts_automerge", _configure_fts_automerge_sync, db)
    except Exception:
        logger.warning("daemon: FTS automerge configuration failed", exc_info=True)


def _configure_fts_automerge_sync(db: Path) -> None:
    from polylogue.daemon.fts_automerge import configure_fts_automerge_sync
    from polylogue.storage.sqlite.connection_profile import open_connection

    conn = open_connection(db, timeout=30.0)
    try:
        configure_fts_automerge_sync(conn)
    finally:
        conn.close()


async def _periodic_fts_merge() -> None:
    """Run a bounded FTS5 merge every 5 minutes to amortise segment cost (#1851).

    With automerge=0, level-0 FTS5 segments accumulate over time.  This
    periodic pass merges them in bounded 500-work-unit chunks so query
    performance stays good without ever paying the full merge cost in a
    single write transaction.  The 5-minute interval matches the WAL
    checkpoint loop so the two share the same maintenance cadence.
    """
    from polylogue.daemon.fts_automerge import run_periodic_fts_merge_sync

    while True:
        await asyncio.sleep(300)
        db = _active_index_db_path()
        if not db.exists():
            continue
        try:
            await daemon_write_coordinator().run_sync("maintenance.fts_merge", run_periodic_fts_merge_sync, db)
        except Exception:
            logger.warning("daemon: FTS periodic merge failed", exc_info=True)


async def _periodic_wal_checkpoint() -> None:
    """Run WAL checkpoints every 5 minutes to keep tier WAL files bounded."""
    from polylogue.paths import archive_root
    from polylogue.storage.sqlite.wal_checkpoint import maybe_checkpoint_archive_wals

    while True:
        await asyncio.sleep(300)
        root = archive_root()
        if not root.exists():
            continue
        try:
            observations = await daemon_write_coordinator().run_sync(
                "maintenance.wal_checkpoint",
                maybe_checkpoint_archive_wals,
                root,
                reason="periodic",
            )
            for observation in observations:
                if not observation.ran:
                    continue
                logger.info(
                    "daemon: WAL checkpoint %s before=%d after=%d busy=%d checkpointed=%d error=%s blockers=%s",
                    observation.mode,
                    observation.wal_bytes_before,
                    observation.wal_bytes_after,
                    observation.busy_pages,
                    observation.checkpointed_pages,
                    observation.error,
                    ",".join(observation.blocking_processes[:5]),
                )
        except Exception:
            logger.warning("daemon: WAL checkpoint failed", exc_info=True)


async def _periodic_status_snapshot_refresh() -> None:
    """Refresh the rich daemon status snapshot outside request handlers."""
    from polylogue.daemon.status_snapshot import refresh_status_snapshot

    while True:
        try:
            await asyncio.to_thread(refresh_status_snapshot)
        except Exception:
            logger.warning("daemon: status snapshot refresh failed", exc_info=True)
        await asyncio.sleep(10)


async def _run_drive_source_catchup_once() -> int:
    """Acquire and parse configured Drive sources once.

    The live watcher only observes filesystem roots. Google Drive sources are
    remote acquisition roots, so the daemon has to run their catch-up through
    the staged acquisition pipeline explicitly.
    """
    from polylogue.config import get_config
    from polylogue.pipeline.services.ingest_batch import refresh_session_insights_bulk
    from polylogue.pipeline.services.parsing import ParsingService
    from polylogue.services import build_runtime_services

    config = get_config()
    sources = [source for source in config.sources if source.is_drive]
    if not sources:
        return 0

    services = build_runtime_services(config=config, db_path=config.db_path)
    try:
        repository = services.get_repository()
        backend = services.get_backend()
        parser = ParsingService(
            repository=repository,
            archive_root=config.archive_root,
            config=config,
        )
        result = await parser.ingest_sources(
            sources=sources,
            stage="all",
            parse_records=True,
        )
        session_ids = sorted(result.parse_result.processed_ids)
        if session_ids:
            await refresh_session_insights_bulk(backend, session_ids)
        logger.info(
            "daemon: Drive source catch-up complete — sources=%d raw=%d sessions=%d changed=%d errors=%d",
            len(sources),
            len(result.acquire_result.raw_ids),
            result.parse_result.counts["sessions"],
            len(session_ids),
            result.acquire_result.errors,
        )
        return len(session_ids)
    finally:
        await services.close()


async def _run_drive_source_catchup_safely() -> int:
    """Run Drive catch-up without letting remote-source failures kill daemon."""
    try:
        return await _run_drive_source_catchup_once()
    except asyncio.CancelledError:
        raise
    except Exception:
        logger.warning("daemon: Drive source catch-up failed", exc_info=True)
        return 0


async def _periodic_drive_source_catchup() -> None:
    """Periodically converge remote Drive sources such as AiStudio exports."""
    while True:
        await asyncio.sleep(_DRIVE_SOURCE_CATCHUP_INTERVAL_SECONDS)
        coordinator = daemon_write_coordinator()
        changed = await coordinator.run("maintenance.drive_catchup", _run_drive_source_catchup_safely)
        if changed:
            logger.info("daemon: Drive catch-up refreshed %d session(s)", changed)


async def _periodic_heartbeat() -> None:
    """Log daemon heartbeat with archive stats every 15 minutes."""
    while True:
        await asyncio.sleep(900)  # 15 minutes
        db = _active_index_db_path()
        if not db.exists():
            continue
        try:
            n_sessions, n_messages, noun = await asyncio.to_thread(_heartbeat_counts, db)
            logger.info(
                "daemon heartbeat: %d %s, %d messages indexed",
                n_sessions,
                noun,
                n_messages,
            )
        except Exception:
            logger.warning("daemon: heartbeat query failed", exc_info=True)


async def _periodic_lifecycle_heartbeat(*, interval_s: float | None = None) -> None:
    """Advance the ops-only heartbeat even when archive work is blocked.

    This must remain independent of index stats and convergence: a daemon that
    deliberately keeps only API/health surfaces available after schema
    preflight failure is still alive and must not age into a false vanished
    state.
    """
    from polylogue.daemon.lifecycle import DAEMON_HEARTBEAT_INTERVAL_SECONDS

    interval = DAEMON_HEARTBEAT_INTERVAL_SECONDS if interval_s is None else interval_s
    while True:
        await asyncio.sleep(interval)
        lifecycle = _daemon_lifecycle
        if lifecycle is None:
            continue
        try:
            await daemon_write_coordinator().run_sync("daemon.lifecycle.heartbeat", lifecycle.heartbeat)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.warning("daemon: lifecycle heartbeat write failed", exc_info=True)


async def _periodic_db_optimize() -> None:
    """Run SQLite PRAGMA optimize once daily to keep query plans current.

    On a 60 GB archive with millions of rows, the query planner's
    internal statistics drift as the table sizes change.  PRAGMA optimize
    is an explicit background maintenance pass, not startup readiness.
    The daemon must bind, catch up, and converge changed files before it
    considers planner-stat maintenance; otherwise a large archive can pay
    broad read IO at the exact moment live catch-up already needs the disk.
    """
    from polylogue.paths import archive_root
    from polylogue.storage.sqlite.maintenance import maybe_optimize_archive_tiers

    while True:
        await asyncio.sleep(86_400)  # 24 hours; no startup optimize.
        root = archive_root()
        if not root.exists():
            continue
        try:
            observations = await daemon_write_coordinator().run_sync(
                "maintenance.db_optimize",
                maybe_optimize_archive_tiers,
                root,
                reason="periodic",
            )
            ran = sum(1 for observation in observations if observation.ran)
            errors = [observation.error for observation in observations if observation.error]
            logger.info("daemon: DB optimize completed tiers=%d errors=%d", ran, len(errors))
        except Exception:
            logger.warning("daemon: DB optimize failed", exc_info=True)


async def _periodic_convergence_check(
    sources: tuple[WatchSource, ...],
    *,
    catch_up_complete: asyncio.Event | None = None,
) -> None:
    """Periodically retry recorded derived convergence debt."""
    db = _active_index_db_path()
    if catch_up_complete is not None:
        await catch_up_complete.wait()
    while True:
        await _retry_convergence_debt_once(db)
        await asyncio.sleep(_CONVERGENCE_DEBT_RETRY_INTERVAL_SECONDS)


async def _retry_convergence_debt_once(db: Path) -> None:
    """Run one logged derived-debt retry pass when the archive exists."""
    if not db.exists():
        return
    try:
        repaired = await daemon_write_coordinator().run_sync(
            "maintenance.convergence_debt",
            _drain_convergence_debt_once,
            db,
        )
        if repaired:
            logger.info("convergence: retried %d derived debt item(s)", repaired)
    except sqlite3.OperationalError as exc:
        if is_transient_sqlite_lock(exc):
            logger.info("convergence: archive busy; retrying derived debt on next tick: %s", exc)
            return
        logger.warning("convergence: check failed", exc_info=True)
    except Exception:
        logger.warning("convergence: check failed", exc_info=True)


async def _periodic_raw_materialization_convergence() -> None:
    """Continuously converge durable raw source rows into the index tier."""
    await _periodic_raw_materialization_convergence_after()


async def _periodic_raw_materialization_convergence_after(
    catch_up_complete: asyncio.Event | None = None,
) -> None:
    """Continuously converge durable raw rows after initial source catch-up."""
    if catch_up_complete is not None:
        await catch_up_complete.wait()
    while True:
        try:
            materialized = await daemon_write_coordinator().run_sync(
                "maintenance.raw_materialization",
                _drain_raw_materialization_once,
            )
            if materialized:
                logger.info("raw materialization: converged %d session(s)", materialized)
        except sqlite3.OperationalError as exc:
            if is_transient_sqlite_lock(exc):
                logger.info("raw materialization: archive busy; retrying on next tick: %s", exc)
            else:
                logger.warning("raw materialization: convergence check failed", exc_info=True)
        except Exception:
            logger.warning("raw materialization: convergence check failed", exc_info=True)
        await asyncio.sleep(_RAW_MATERIALIZATION_CONVERGENCE_INTERVAL_SECONDS)


async def _periodic_session_insight_convergence_after(
    catch_up_complete: asyncio.Event | None = None,
) -> None:
    """Continuously converge missing/stale session insights after catch-up."""
    if catch_up_complete is not None:
        await catch_up_complete.wait()
    while True:
        burst_count = 0
        try:
            while burst_count < _SESSION_INSIGHT_CONVERGENCE_BURST_LIMIT:
                refreshed = await daemon_write_coordinator().run_sync(
                    "maintenance.session_insights",
                    _drain_session_insights_once,
                )
                if not refreshed:
                    break
                burst_count += 1
                logger.info("insights: converged %d session profile(s)", refreshed)
                await asyncio.sleep(_SESSION_INSIGHT_CONVERGENCE_BURST_PAUSE_SECONDS)
        except sqlite3.OperationalError as exc:
            if is_transient_sqlite_lock(exc):
                logger.info("insights: archive busy; retrying profile backlog on next tick: %s", exc)
            else:
                logger.warning("insights: profile backlog convergence failed", exc_info=True)
        except Exception:
            logger.warning("insights: profile backlog convergence failed", exc_info=True)
        await asyncio.sleep(_SESSION_INSIGHT_CONVERGENCE_INTERVAL_SECONDS)


async def _bridge_catch_up_complete(
    source: asyncio.Event,
    target: asyncio.Event,
) -> None:
    """Forward watcher catch-up completion to daemon maintenance loops."""
    await source.wait()
    target.set()


async def _reconcile_blob_publications() -> None:
    """Classify crash-left publication reservations before source catch-up."""
    from polylogue.paths import archive_root
    from polylogue.storage.blob_publication import reconcile_blob_publication_reservations

    root = archive_root()
    if not (root / "source.db").exists():
        return
    outcome = await daemon_write_coordinator().run_sync(
        "startup.blob_publications",
        reconcile_blob_publication_reservations,
        root / "source.db",
        root / "blob",
        index_db_path=root / "index.db",
    )
    if (
        outcome.cleared_referenced
        or outcome.cleared_missing
        or outcome.retained_referenced
        or outcome.retained_missing
        or outcome.unresolved
    ):
        logger.info(
            "blob publications: classified cleared_ref=%d cleared_missing=%d "
            "retained_ref=%d retained_missing=%d unresolved=%d",
            outcome.cleared_referenced,
            outcome.cleared_missing,
            outcome.retained_referenced,
            outcome.retained_missing,
            outcome.unresolved,
        )
    retained = outcome.retained_referenced + outcome.retained_missing + outcome.unresolved
    if retained:
        logger.warning(
            "blob publications: retained %d receipt(s) for inspection or explicit abandonment",
            retained,
        )


def _drain_raw_materialization_once(*, limit: int = _RAW_MATERIALIZATION_CONVERGENCE_BATCH_LIMIT) -> int:
    """Run one bounded raw source→index convergence pass."""
    from polylogue.config import Config
    from polylogue.paths import archive_root, render_root
    from polylogue.storage.blob_integrity import restore_direct_blob_reference_debt
    from polylogue.storage.repair import repair_raw_materialization

    archive = archive_root()
    restored = restore_direct_blob_reference_debt(
        archive / "source.db",
        dry_run=False,
        max_count=_BLOB_REFERENCE_RESTORE_CONVERGENCE_BATCH_LIMIT,
        sample_size=0,
    )
    if restored.restored_count:
        logger.info(
            "blob references: restored %d direct source blob(s) before raw materialization",
            restored.restored_count,
        )

    config = Config(
        archive_root=archive,
        render_root=render_root(),
        sources=[],
    )
    try:
        result = repair_raw_materialization(config, dry_run=False, raw_artifact_limit=limit)
    finally:
        _close_raw_materialization_fts(config.archive_root / "index.db")
    _emit_raw_materialization_pass(result)
    if not result.success:
        logger.warning("raw materialization: bounded convergence incomplete: %s", result.detail)
    return result.repaired_count


def _emit_raw_materialization_pass(result: Any) -> None:
    """Persist the conserved plan outcomes for one bounded daemon pass."""
    outcomes = tuple(getattr(result, "plan_outcomes", ()))
    from polylogue.daemon.events import emit_daemon_event

    metrics = dict(getattr(result, "metrics", {}))
    emit_daemon_event(
        "raw_materialization_pass",
        payload={
            "pass_id": f"raw-materialization:{os.urandom(16).hex()}",
            "success": bool(result.success),
            "repaired_count": int(result.repaired_count),
            "detail": str(result.detail),
            "metrics": metrics,
            "plan_outcomes": [outcome.to_dict() for outcome in outcomes],
        },
    )


def _close_raw_materialization_fts(index_db: Path) -> None:
    """Return message search to ready or leave explicit retryable debt.

    Large raw replay batches deliberately suspend FTS triggers and may skip
    inline repair.  This closure runs under the same daemon write lease as the
    replay, including replay exception/cancellation cleanup.
    """
    if not index_db.exists():
        return
    from polylogue.daemon.convergence_stages import repair_fts_surface

    try:
        needs_repair = _raw_materialization_fts_needs_repair(index_db)
    except Exception as exc:
        _record_raw_materialization_fts_debt(index_db, f"FTS readiness probe failed after raw materialization: {exc}")
        return
    if not needs_repair:
        return
    try:
        repaired = repair_fts_surface(index_db, "messages_fts")
    except Exception as exc:
        # Preserve the original raw-materialization outcome. A stale
        # freshness row plus explicit debt keeps readiness negative and makes
        # this closure retryable instead of masking the initiating failure.
        _record_raw_materialization_fts_debt(
            index_db,
            f"FTS repair failed after raw materialization: {type(exc).__name__}: {exc}",
        )
        return
    if repaired:
        try:
            from polylogue.sources.live.cursor import CursorStore

            CursorStore(index_db).clear_convergence_debt(
                subject_type="fts_surface",
                subject_id="messages_fts",
                stage="fts",
            )
        except Exception:
            logger.warning("raw materialization: failed to clear repaired message FTS debt", exc_info=True)
        return
    _record_raw_materialization_fts_debt(
        index_db,
        "raw materialization exited without restoring message FTS readiness",
    )


def _record_raw_materialization_fts_debt(index_db: Path, error: str) -> None:
    from polylogue.sources.live.cursor import CursorStore

    try:
        CursorStore(index_db).record_convergence_debt(
            stage="fts",
            subject_type="fts_surface",
            subject_id="messages_fts",
            error=error,
        )
    except Exception:
        # The stale FTS freshness row remains a durable negative readiness
        # verdict even if the richer retry queue cannot be updated.
        logger.warning("raw materialization: failed to record message FTS convergence debt", exc_info=True)


def _raw_materialization_fts_needs_repair(index_db: Path) -> bool:
    from polylogue.storage.fts.freshness import message_fts_recorded_readiness_sync
    from polylogue.storage.fts.fts_lifecycle import message_fts_readiness_sync
    from polylogue.storage.sqlite.connection_profile import open_daemon_connection

    with open_daemon_connection(index_db, timeout=5.0) as conn:
        recorded = message_fts_recorded_readiness_sync(conn)
        if recorded is not None:
            return not bool(recorded["ready"])
        readiness = message_fts_readiness_sync(conn, verify_total_rows=False)
        return bool(readiness["exists"]) and not bool(readiness["ready"])


def _drain_session_insights_once(*, limit: int = _SESSION_INSIGHT_CONVERGENCE_BATCH_LIMIT) -> int:
    """Run one bounded missing/stale session-profile convergence pass."""
    from polylogue.daemon.convergence_stages import (
        _archive_insights_execute_ids,
        _schema_archive_session_ids_missing_profiles,
    )
    from polylogue.paths import active_index_db_path
    from polylogue.storage.sqlite.connection_profile import open_daemon_connection

    db = active_index_db_path()
    if not db.exists():
        return 0
    conn = open_daemon_connection(db, timeout=30.0)
    try:
        ids = _schema_archive_session_ids_missing_profiles(conn, limit=limit)
        if not ids:
            return 0
        result = _archive_insights_execute_ids(conn, ids)
        return len(ids) if bool(getattr(result, "success", result)) else 0
    finally:
        conn.close()


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
        if debt.subject_type in {"source_path", "session_id", "fts_surface"} and _debt_retry_due(debt, now=now)
    ]
    if not due_debt:
        return 0
    session_ids = tuple(dict.fromkeys(debt.subject_id for debt in due_debt if debt.subject_type == "session_id"))
    paths = tuple(dict.fromkeys([Path(debt.subject_id) for debt in due_debt if debt.subject_type == "source_path"]))
    fts_surfaces = tuple(dict.fromkeys(debt.subject_id for debt in due_debt if debt.subject_type == "fts_surface"))
    if not paths and not session_ids and not fts_surfaces:
        return 0
    fts_surface_results = _drain_fts_surface_debt(db, fts_surfaces)
    converger = DaemonConverger(stages=make_default_convergence_stages(db), max_workers=2)
    path_states, _path_timings = converger.converge_batch(paths)
    session_states, _session_timings = converger.converge_sessions(session_ids)
    retried = 0
    for debt in due_debt:
        subject_states: list[object | None]
        if debt.subject_type == "session_id":
            subject_states = [session_states.get(debt.subject_id)]
        elif debt.subject_type == "fts_surface":
            if fts_surface_results.get(debt.subject_id) is True:
                cursor.clear_convergence_debt(subject_type=debt.subject_type, subject_id=debt.subject_id)
                retried += 1
                continue
            cursor.clear_convergence_debt(subject_type=debt.subject_type, subject_id=debt.subject_id)
            cursor.record_convergence_debt(
                stage=debt.stage,
                subject_type=debt.subject_type,
                subject_id=debt.subject_id,
                error="FTS freshness convergence did not converge",
            )
            retried += 1
            continue
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


def _drain_fts_surface_debt(db: Path, surfaces: tuple[str, ...]) -> dict[str, bool]:
    if not surfaces:
        return {}
    from polylogue.daemon.convergence_stages import repair_fts_surface

    results: dict[str, bool] = {}
    for surface in surfaces:
        results[surface] = repair_fts_surface(db, surface)
    return results


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
        tiers = resolve_health_tiers(cfg.health_check_tiers)

        await asyncio.sleep(interval)
        try:
            from polylogue.daemon.health import check_health
            from polylogue.daemon.notifications import send_notifications

            health = await daemon_write_coordinator().run_sync(
                "maintenance.health_check",
                check_health,
                tiers=tiers,
            )
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


def _release_pidfile_after_writer_drain(pidfile_fd: int | None, *, writer_drained: bool) -> int | None:
    """Release daemon ownership only after every admitted writer is idle."""
    if not writer_drained:
        logger.error("daemon: writer coordinator remains active; retaining pidfile ownership until process exit")
        return pidfile_fd
    if pidfile_fd is not None:
        with contextlib.suppress(OSError):
            os.close(pidfile_fd)
    _cleanup_pidfile()
    return None


def _emit_live_batch_event(kind: str, payload: dict[str, object]) -> None:
    """Persist a live-ingest batch event and fan out granular #1204 topics.

    The legacy ``ingestion_batch`` kind is preserved verbatim for existing
    consumers (status views, polling fallback). When the batch payload
    carries succeeded counts we additionally emit per-topic events
    (``session.appended`` / ``message.appended``) so the reader can
    subscribe selectively and animate just-touched rows.
    """
    from polylogue.daemon.events import (
        emit_daemon_event,
        emit_message_appended,
        emit_session_appended,
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
    emit_session_appended(
        source_name=None,
        succeeded_file_count=succeeded,
        failed_file_count=failed,
    )
    emit_message_appended(
        session_id=None,
        source_name=None,
        appended_count=succeeded,
    )


async def _emit_daemon_lifecycle_event(
    phase: str,
    *,
    archive_root_path: Path,
    status: str = "ok",
    component: str | None = None,
    payload: dict[str, object] | None = None,
) -> None:
    """Persist a daemon lifecycle event without making observability fatal."""
    from polylogue.daemon.events import emit_daemon_event

    run_id = os.environ.get("POLYLOGUE_DEV_LOOP_RUN_ID")
    event_payload: dict[str, object] = {
        "phase": phase,
        "status": status,
        "pid": os.getpid(),
        "cwd": os.getcwd(),
        "archive_root": str(archive_root_path),
    }
    if component is not None:
        event_payload["component"] = component
    log_dir = os.environ.get("POLYLOGUE_DEV_LOOP_LOG_DIR")
    if run_id:
        event_payload["dev_loop_run_id"] = run_id
    if log_dir:
        event_payload["dev_loop_log_dir"] = log_dir
    if payload:
        event_payload.update(payload)
    try:
        async with asyncio.timeout(0.5):
            await daemon_write_coordinator().run_sync(
                f"daemon.lifecycle.{phase}",
                emit_daemon_event,
                "daemon.lifecycle",
                operation_id=run_id,
                payload=event_payload,
            )
    except TimeoutError:
        logger.warning("daemon: timed out emitting lifecycle event %s", phase)
    except Exception:
        logger.warning("daemon: failed to emit lifecycle event %s", phase, exc_info=True)


async def run_live_watcher(
    *,
    sources: tuple[WatchSource, ...],
    debounce_s: float,
) -> None:
    async with Polylogue() as polylogue:
        watcher = LiveWatcher(
            polylogue,
            sources,
            debounce_s=debounce_s,
            event_emitter=_emit_live_batch_event,
            write_coordinator=daemon_write_coordinator(),
        )
        try:
            await watcher.run()
        except KeyboardInterrupt:
            watcher.stop()


async def run_daemon_services(
    *,
    sources: tuple[WatchSource, ...],
    debounce_s: float,
    enable_watch: bool,
    enable_source_catchup: bool = True,
    enable_browser_capture: bool,
    browser_capture_host: str,
    browser_capture_port: int,
    browser_capture_spool_path: Path | None,
    browser_capture_allow_remote: bool = False,
    browser_capture_auth_token: str | None = None,
    browser_capture_allow_no_auth: bool = False,
    browser_capture_extra_origins: tuple[str, ...] = (),
    enable_api: bool = False,
    api_host: str = "127.0.0.1",
    api_port: int = 8766,
    api_auth_token: str | None = None,
) -> None:
    """Run configured daemon components until interrupted."""
    from polylogue.daemon import process_start as _process_start
    from polylogue.daemon.status_snapshot import configure_runtime_components
    from polylogue.paths import archive_root

    global _daemon_lifecycle, _pidfile_path
    _process_start.started_at_wall()
    archive_root_path = Path(archive_root())
    from polylogue.paths import active_index_db_path
    from polylogue.storage.archive_identity import assert_writable_archive_identity

    # Identity precedes schema checks, pidfiles, HTTP startup, and every other
    # component: a split-root daemon must not become partially observable as a
    # healthy writer before its first ArchiveStore happens to open.
    active_root = active_index_db_path().parent
    assert_writable_archive_identity(configured_root=archive_root_path, active_root=active_root)

    if (
        enable_api
        and enable_browser_capture
        and api_port == browser_capture_port
        and bind_hosts_overlap(api_host, browser_capture_host)
    ):
        raise click.UsageError(
            f"Daemon API {api_host}:{api_port} conflicts with browser-capture "
            f"receiver {browser_capture_host}:{browser_capture_port}. "
            f"Set distinct --api-port/--port values or bind one component to a non-overlapping host."
        )

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
    configure_runtime_components(
        api_enabled=enable_api,
        watcher_enabled=enable_watch,
        watcher_roots=tuple(str(source.root) for source in sources),
        browser_capture_enabled=enable_browser_capture,
        browser_capture_spool_path=browser_capture_spool_path,
    )

    logger.info("daemon started")

    # Schema preflight runs FIRST, before any DB-touching startup task. A
    # mismatched runtime/db combination must not even open the DB for FTS or
    # heartbeat queries — that is the IO cost #1003 is meant to avoid.
    schema_alert = _check_schema_version_fast()
    watcher_blocked = enable_watch and schema_alert.severity == HealthSeverity.CRITICAL
    lifecycle_events_enabled = not watcher_blocked
    if watcher_blocked:
        logger.error(
            "daemon: schema preflight CRITICAL — %s. Refusing to start the live watcher; "
            "HTTP/health surfaces remain available so this state is observable.",
            schema_alert.message,
        )
        set_degraded(
            DegradedReason(
                code="schema_version_mismatch",
                message=schema_alert.message,
                detail={"check_name": schema_alert.check_name},
            )
        )

    pidfile = archive_root_path / "daemon.pid"
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
    from polylogue.daemon.lifecycle import DaemonLifecycle, install_signal_handlers, restore_signal_handlers

    write_coordinator: DaemonWriteCoordinator = daemon_write_coordinator()
    try:
        _daemon_lifecycle = await write_coordinator.run_sync(
            "daemon.lifecycle.start",
            DaemonLifecycle.start,
            details={"archive_root": str(archive_root_path)},
        )
        previous_signal_handlers = install_signal_handlers(_daemon_lifecycle)
    except BaseException:
        lifecycle = _daemon_lifecycle
        if lifecycle is not None:
            with contextlib.suppress(Exception):
                await write_coordinator.run_sync("daemon.lifecycle.stop", lifecycle.stop, exit_kind="error")
        writer_drained = await write_coordinator.shutdown(timeout=5.0)
        _release_pidfile_after_writer_drain(pidfile_fd, writer_drained=writer_drained)
        _daemon_lifecycle = None
        raise

    # Ensure all configured source roots exist so health checks don't flag
    # never-yet-used sources (e.g. hooks sidecar dir) as missing.
    for src in sources:
        src.root.mkdir(parents=True, exist_ok=True)

    if lifecycle_events_enabled:
        await _emit_daemon_lifecycle_event(
            "startup",
            archive_root_path=archive_root_path,
            status="starting",
            payload={
                "api_enabled": enable_api,
                "api_host": api_host,
                "api_port": api_port,
                "browser_capture_enabled": enable_browser_capture,
                "browser_capture_host": browser_capture_host,
                "browser_capture_port": browser_capture_port,
                "watch_enabled": enable_watch,
                "source_catchup_enabled": enable_source_catchup,
                "source_roots": [str(src.root) for src in sources],
            },
        )

    # Periodic maintenance tasks. If schema preflight blocks the watcher, do
    # not start any background loop that opens the archive: a mismatched
    # runtime/database pair must remain observable without doing catch-up,
    # FTS freshness recovery, status snapshots, WAL checkpointing, or convergence work.
    #
    # The task list is populated only after startup FTS readiness completes.
    # Several maintenance loops can write the archive, especially convergence
    # debt retry; starting them before FTS startup freshness recovery self-contends on
    # SQLite during daemon bootstrap.
    # The lifecycle tick is deliberately scheduled before the schema-block
    # guard. It writes only the disposable ops tier and proves that the
    # surviving API/health process is still alive while archive work is
    # intentionally withheld.
    maintenance_tasks: list[asyncio.Task[None]] = [asyncio.create_task(_periodic_lifecycle_heartbeat())]

    api_server: ThreadingHTTPServer | None = None
    api_server_task: asyncio.Task[None] | None = None
    uds_server: Any | None = None
    uds_server_task: asyncio.Task[None] | None = None
    server: BrowserCaptureHTTPServer | None = None
    server_task: asyncio.Task[None] | None = None
    watcher: LiveWatcher | None = None
    watcher_task: asyncio.Task[None] | None = None
    converger: DaemonConverger | None = None
    catch_up_complete_gate: asyncio.Event | None = None
    tasks: list[asyncio.Task[None]] = []
    cleanup_task: asyncio.Task[object] | None = None
    cleanup_cancel_requests = 0
    termination: BaseException | None = None
    try:
        if enable_browser_capture:
            resolved_browser_capture_auth_token = resolve_receiver_auth_token(
                browser_capture_auth_token, allow_no_auth=browser_capture_allow_no_auth
            )
            server = make_server(
                browser_capture_host,
                browser_capture_port,
                spool_path=browser_capture_spool_path,
                allow_remote=browser_capture_allow_remote,
                auth_token=resolved_browser_capture_auth_token,
                extra_origins=browser_capture_extra_origins,
            )
            server_task = asyncio.create_task(asyncio.to_thread(server.serve_forever, 0.5))
            tasks.append(server_task)
            if lifecycle_events_enabled:
                await _emit_daemon_lifecycle_event(
                    "component_started",
                    archive_root_path=archive_root_path,
                    component="browser_capture",
                    payload={
                        "host": browser_capture_host,
                        "port": browser_capture_port,
                        "spool_path": str(browser_capture_spool_path)
                        if browser_capture_spool_path is not None
                        else None,
                        "auth_enabled": resolved_browser_capture_auth_token is not None,
                    },
                )

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
                write_bridge=DaemonWriteThreadBridge(write_coordinator, asyncio.get_running_loop()),
            )
            api_server_task = asyncio.create_task(asyncio.to_thread(api_server.serve_forever, 0.5))
            tasks.append(api_server_task)
            from polylogue.daemon.uds import DaemonAPIUnixHTTPServer, daemon_socket_path

            uds_server = DaemonAPIUnixHTTPServer(
                daemon_socket_path(),
                DaemonAPIHandler,
                auth_token=api_auth_token,
                write_bridge=DaemonWriteThreadBridge(write_coordinator, asyncio.get_running_loop()),
            )
            uds_server_task = asyncio.create_task(asyncio.to_thread(uds_server.serve_forever, 0.5))
            tasks.append(uds_server_task)
            if lifecycle_events_enabled:
                await _emit_daemon_lifecycle_event(
                    "component_started",
                    archive_root_path=archive_root_path,
                    component="api",
                    payload={"host": api_host, "port": api_port, "auth_enabled": bool(api_auth_token)},
                )

        # Ensure FTS structure after HTTP surfaces are bound and before live
        # catch-up starts. Startup FTS maintenance and catch-up ingestion are
        # both write-heavy; running them concurrently makes SQLite maintenance
        # time out behind the daemon's own writer.
        if not watcher_blocked:
            from polylogue.daemon.convergence import DaemonConverger
            from polylogue.daemon.convergence_stages import make_default_convergence_stages
            from polylogue.daemon.embedding_backlog import (
                periodic_embedding_backlog_check,
                periodic_embedding_orphan_reconcile_check,
            )

            await _run_startup_fts_readiness(write_coordinator)
            if lifecycle_events_enabled:
                await _emit_daemon_lifecycle_event(
                    "component_ready",
                    archive_root_path=archive_root_path,
                    component="fts_startup",
                )
            await _run_startup_lineage_readiness(write_coordinator)
            if lifecycle_events_enabled:
                await _emit_daemon_lifecycle_event(
                    "component_ready",
                    archive_root_path=archive_root_path,
                    component="lineage_startup",
                )
            await _reconcile_blob_publications()
            # Disable per-write FTS5 automerge so each small ingest batch does
            # not trigger a merge of the full (hundreds-of-MB) existing
            # segments (#1851).  A periodic merge pass amortises the cost.
            await _configure_fts_automerge()
            if enable_source_catchup:
                changed_drive_sessions = await write_coordinator.run(
                    "startup.drive_catchup",
                    _run_drive_source_catchup_safely,
                )
                if changed_drive_sessions:
                    logger.info("daemon: startup Drive catch-up refreshed %d session(s)", changed_drive_sessions)
            else:
                logger.info("daemon: configured source catch-up disabled for this run")
            catch_up_complete_gate = asyncio.Event() if enable_watch else None
            periodic_loops = [
                _periodic_raw_materialization_convergence_after(catch_up_complete_gate),
                _periodic_session_insight_convergence_after(catch_up_complete_gate),
                _periodic_convergence_check(sources, catch_up_complete=catch_up_complete_gate),
                _periodic_wal_checkpoint(),
                _periodic_fts_merge(),
                _periodic_heartbeat(),
                periodic_embedding_backlog_check(catch_up_complete=catch_up_complete_gate),
                periodic_embedding_orphan_reconcile_check(catch_up_complete=catch_up_complete_gate),
                _periodic_health_check(),
                _periodic_db_optimize(),
                _periodic_status_snapshot_refresh(),
            ]
            if enable_source_catchup:
                periodic_loops.append(_periodic_drive_source_catchup())
            maintenance_tasks.extend(asyncio.create_task(loop) for loop in periodic_loops)
            _db = _active_index_db_path()
            converger = DaemonConverger(
                stages=make_default_convergence_stages(_db),
                max_workers=2,
            )
            await converger.start()
            if lifecycle_events_enabled:
                await _emit_daemon_lifecycle_event(
                    "component_started",
                    archive_root_path=archive_root_path,
                    component="converger",
                )

        # Preflight already ran at the top of run_daemon_services (see
        # ``watcher_blocked`` above); reuse that result.
        try:
            if enable_watch and not watcher_blocked:
                async with Polylogue() as polylogue:
                    watcher = LiveWatcher(
                        polylogue,
                        sources,
                        debounce_s=debounce_s,
                        converger=converger,
                        event_emitter=_emit_live_batch_event,
                        write_coordinator=write_coordinator,
                    )
                    watcher_catch_up_complete = getattr(watcher, "catch_up_complete", None)
                    if catch_up_complete_gate is not None and watcher_catch_up_complete is not None:
                        maintenance_tasks.append(
                            asyncio.create_task(
                                _bridge_catch_up_complete(
                                    watcher_catch_up_complete,
                                    catch_up_complete_gate,
                                )
                            )
                        )
                    watcher_task = asyncio.create_task(watcher.run())
                    tasks.append(watcher_task)
                    if lifecycle_events_enabled:
                        await _emit_daemon_lifecycle_event(
                            "component_started",
                            archive_root_path=archive_root_path,
                            component="watcher",
                            payload={"source_count": len(sources), "debounce_s": debounce_s},
                        )
                    all_tasks = tasks + maintenance_tasks
                    await asyncio.gather(*all_tasks)
            elif tasks:
                # Watcher disabled or preflight-blocked: keep HTTP/health and
                # other components serving so operators see the degraded state.
                if lifecycle_events_enabled:
                    await _emit_daemon_lifecycle_event(
                        "component_skipped",
                        archive_root_path=archive_root_path,
                        component="watcher",
                        payload={
                            "reason": "schema_blocked" if watcher_blocked else "disabled",
                            "watch_enabled": enable_watch,
                        },
                    )
                all_tasks = tasks + maintenance_tasks
                await asyncio.gather(*all_tasks)
            else:
                if lifecycle_events_enabled:
                    await _emit_daemon_lifecycle_event(
                        "component_skipped",
                        archive_root_path=archive_root_path,
                        component="watcher",
                        payload={
                            "reason": "schema_blocked" if watcher_blocked else "disabled",
                            "watch_enabled": enable_watch,
                        },
                    )
                await asyncio.gather(*maintenance_tasks)
        except BaseException as exc:
            termination = exc
            if not isinstance(exc, (asyncio.CancelledError, KeyboardInterrupt)):
                _log_completed_daemon_tasks(tasks + maintenance_tasks)
            raise
    finally:
        cleanup_task = asyncio.current_task()
        if cleanup_task is not None:
            cleanup_cancel_requests = cleanup_task.cancelling()
            for _ in range(cleanup_cancel_requests):
                cleanup_task.uncancel()
        try:
            lifecycle = _daemon_lifecycle
            signal_termination = lifecycle.received_signal_name is not None or isinstance(termination, SystemExit)
            if lifecycle_events_enabled and not signal_termination:
                await _emit_daemon_lifecycle_event(
                    "shutdown_started",
                    archive_root_path=archive_root_path,
                    status="stopping",
                )
            if watcher is not None:
                watcher.stop()
            if converger is not None:
                try:
                    async with asyncio.timeout(5.0):
                        await converger.stop()
                except TimeoutError:
                    logger.warning("daemon: timed out stopping convergence executor")
            if server is not None:
                await _shutdown_server_if_serving(server, server_task, label="browser-capture")
            if api_server is not None:
                await _shutdown_server_if_serving(api_server, api_server_task, label="api")
            if uds_server is not None:
                await _shutdown_server_if_serving(uds_server, uds_server_task, label="uds")

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

            try:
                async with asyncio.timeout(5.0):
                    await write_coordinator.run_sync(
                        "shutdown.live_ingest_attempts",
                        _mark_interrupted_live_ingest_attempts_on_shutdown,
                    )
            except TimeoutError:
                logger.warning("daemon: timed out recording interrupted ingest attempts during shutdown")

            if lifecycle is not None:
                exit_kind = "clean"
                if signal_termination:
                    exit_kind = "signal"
                elif termination is not None and not isinstance(
                    termination, (KeyboardInterrupt, asyncio.CancelledError)
                ):
                    exit_kind = "error"
                try:
                    await write_coordinator.run_sync(
                        "daemon.lifecycle.stop",
                        lifecycle.stop,
                        exit_kind=exit_kind,
                        bounded=signal_termination,
                    )
                except Exception:
                    logger.warning("daemon: could not persist final lifecycle stop", exc_info=True)

            writer_drained = await write_coordinator.shutdown(timeout=5.0)
            pidfile_fd = _release_pidfile_after_writer_drain(pidfile_fd, writer_drained=writer_drained)
        finally:
            if server is not None:
                with contextlib.suppress(Exception):
                    server.server_close()
            if api_server is not None:
                with contextlib.suppress(Exception):
                    api_server.server_close()
            if uds_server is not None:
                with contextlib.suppress(Exception):
                    uds_server.server_close()
            if cleanup_task is not None:
                for _ in range(cleanup_cancel_requests):
                    cleanup_task.cancel()
            restore_signal_handlers(previous_signal_handlers)
            _daemon_lifecycle = None

    logger.info("daemon stopped")


async def _drain_tasks(tasks: list[asyncio.Task[None]], *, timeout: float = 5.0) -> list[BaseException | None]:
    """Drain tasks, returning collected exceptions.

    Returns a list parallel to *tasks*, where each element is the
    exception from that task (or None if it completed cleanly).
    """
    if not tasks:
        return []
    try:
        results = await asyncio.wait_for(
            asyncio.gather(*tasks, return_exceptions=True),
            timeout=timeout,
        )
    except TimeoutError as exc:
        logger.warning("daemon: timed out draining %d task(s) during shutdown", len(tasks))
        return [exc]
    return [r if isinstance(r, BaseException) else None for r in results]


def _report_drain_exceptions(results: list[BaseException | None]) -> None:
    """Log any exceptions found during task draining."""
    for exc in results:
        if exc is not None and not isinstance(exc, asyncio.CancelledError):
            logger.warning("daemon: task raised during shutdown: %s", exc)


def _log_completed_daemon_tasks(tasks: list[asyncio.Task[None]]) -> None:
    for task in tasks:
        if not task.done():
            continue
        try:
            exc = task.exception()
        except asyncio.CancelledError:
            logger.warning("daemon: component task cancelled unexpectedly")
            continue
        if exc is None:
            logger.warning("daemon: component task exited unexpectedly")
        else:
            logger.warning("daemon: component task failed unexpectedly: %s", exc)


async def _shutdown_server_if_serving(
    server: BrowserCaptureHTTPServer | ThreadingHTTPServer,
    task: asyncio.Task[None] | None,
    *,
    label: str,
) -> None:
    if task is None:
        return
    if task.done():
        try:
            exc = task.exception()
        except asyncio.CancelledError:
            # Cancelling the asyncio Future returned by to_thread() does not
            # stop the underlying socketserver.serve_forever thread. Continue
            # into server.shutdown() so Ctrl-C can actually drain the executor.
            logger.debug("daemon: %s server task cancelled; shutting down server anyway", label)
            exc = None
        if exc is None and not task.cancelled():
            logger.warning("daemon: %s server task exited before shutdown", label)
            return
        if exc is not None:
            logger.warning("daemon: %s server task failed before shutdown: %s", label, exc)
            return
    # ``socketserver.BaseServer.shutdown()`` blocks until ``serve_forever()`` sets
    # its internal ``__is_shut_down`` Event. ``serve_forever`` runs off the loop in
    # the default executor (see ``server_task`` creation), so calling
    # ``server.shutdown()`` directly on the loop thread would block it and deadlock
    # when startup fails before the worker entered ``serve_forever``. Run shutdown
    # off the loop in a DEDICATED DAEMON thread — NOT ``asyncio.to_thread`` / a
    # ThreadPoolExecutor:
    #   * The default executor's workers can already be occupied (both
    #     ``serve_forever`` calls, an embedder configured ``max_workers=1``, or
    #     maintenance ``to_thread`` jobs); a queued ``shutdown()`` would then wait
    #     behind the very ``serve_forever`` it must stop, and the 5s ``wait_for``
    #     would cancel the still-queued task without ever setting socketserver's
    #     flag — re-deadlocking ``asyncio.run`` teardown (#1877 Codex review).
    #   * A ThreadPoolExecutor would not help: ``shutdown(wait=False)`` does not
    #     stop a running task and its worker threads are still joined at interpreter
    #     exit, so a genuinely wedged ``server.shutdown()`` would still hang exit
    #     (#1877 CodeRabbit review). A daemon thread is abandoned at exit instead.
    # Completion is signalled back to the loop via ``call_soon_threadsafe``; the 5s
    # timeout is a last-resort guard and the caller's ``server_close()`` closes the
    # socket regardless.
    loop = asyncio.get_running_loop()
    shutdown_done = asyncio.Event()

    def _run_shutdown() -> None:
        try:
            server.shutdown()
        finally:
            loop.call_soon_threadsafe(shutdown_done.set)

    threading.Thread(target=_run_shutdown, name=f"{label}-shutdown", daemon=True).start()
    try:
        await asyncio.wait_for(shutdown_done.wait(), timeout=5.0)
    except TimeoutError:
        logger.warning(
            "daemon: %s server shutdown did not complete within 5s; closing socket directly",
            label,
        )


@click.group(help="Run long-lived Polylogue local services.")
@click.version_option(version=POLYLOGUE_VERSION, prog_name="polylogued")
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
            payload = daemon_status_payload(
                browser_capture_spool_path=spool_path,
                include_browser_capture_spool_path=spool_path is not None,
            )
        click.echo(dumps(payload))
        return
    payload = daemon_status_payload(
        browser_capture_spool_path=spool_path,
        include_browser_capture_spool_path=spool_path is not None,
    )
    for line in format_daemon_status_lines(payload):
        click.echo(line)


@main.command("health", help="Run tiered daemon health checks.")
@click.option(
    "--tier",
    "tiers",
    type=click.Choice(["fast", "medium", "expensive"]),
    multiple=True,
    default=None,
    help="Run specific health check tiers (repeatable). Default: fast.",
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

    By default runs FAST checks. Use --tier to select specific
    tiers or --expensive to add the EXPENSIVE tier.
    """
    configure_logging()

    if tiers:
        health_tiers: set[HealthTier] = {HealthTier(t) for t in tiers}
    else:
        health_tiers = {HealthTier.FAST}
        if include_expensive:
            health_tiers.update({HealthTier.MEDIUM, HealthTier.EXPENSIVE})

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
    "--no-source-catchup",
    is_flag=True,
    help="Do not run configured non-watch source catch-up during this daemon run.",
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
    help="Browser-capture bearer token; auto-minted/loaded from a 0600 file if not given.",
)
@click.option(
    "--browser-capture-allow-no-auth",
    is_flag=True,
    default=False,
    help=(
        "Run the browser-capture receiver with no bearer token at all. Any local process can "
        "then read/post to it -- default OFF; an explicit opt-out for the auto-minted-token default."
    ),
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
@click.pass_context
def run_command(
    ctx: click.Context,
    roots: tuple[Path, ...],
    debounce_s: float,
    host: str,
    port: int,
    spool_path: Path | None,
    no_watch: bool,
    no_source_catchup: bool,
    no_browser_capture: bool,
    insecure_allow_remote: bool,
    browser_capture_auth_token: str | None,
    browser_capture_allow_no_auth: bool,
    browser_capture_origins: tuple[str, ...],
    no_api: bool,
    api_host: str,
    api_port: int,
    api_auth_token: str | None,
) -> None:
    """Run configured daemon components.

    This is the entry point for the polylogued systemd service.
    """
    _enable_faulthandler_if_supported()
    configure_logging()

    from polylogue.config import load_polylogue_config

    cfg = load_polylogue_config()

    def parameter_is_default(name: str) -> bool:
        source = ctx.get_parameter_source(name)
        return source is None or source is click.core.ParameterSource.DEFAULT

    if not roots and cfg.source_roots:
        roots = tuple(Path(root).expanduser() for root in cfg.source_roots)
    if parameter_is_default("debounce_s"):
        debounce_s = cfg.watch_debounce_s
    if parameter_is_default("host"):
        if cfg.layer_of("browser_capture_host") != "default":
            host = cfg.browser_capture_host
        elif cfg.layer_of("daemon_host") != "default":
            host = cfg.daemon_host
    if parameter_is_default("port") and cfg.layer_of("browser_capture_port") != "default":
        port = cfg.browser_capture_port
    if parameter_is_default("spool_path") and cfg.browser_capture_spool_path:
        spool_path = Path(cfg.browser_capture_spool_path).expanduser()
    if parameter_is_default("insecure_allow_remote"):
        insecure_allow_remote = cfg.browser_capture_allow_remote
    if parameter_is_default("browser_capture_auth_token") and cfg.browser_capture_auth_token:
        browser_capture_auth_token = cfg.browser_capture_auth_token
    if parameter_is_default("browser_capture_allow_no_auth"):
        browser_capture_allow_no_auth = cfg.browser_capture_allow_no_auth
    if not browser_capture_origins and cfg.layer_of("browser_capture_allowed_origins") != "default":
        browser_capture_origins = tuple(
            origin.strip() for origin in cfg.browser_capture_allowed_origins.split(",") if origin.strip()
        )
    if parameter_is_default("api_host"):
        if cfg.layer_of("api_host") != "default":
            api_host = cfg.api_host
        elif cfg.layer_of("daemon_host") != "default":
            api_host = cfg.daemon_host
    if parameter_is_default("api_port"):
        if cfg.layer_of("api_port") != "default":
            api_port = cfg.api_port
        elif cfg.layer_of("daemon_port") != "default":
            api_port = cfg.daemon_port
    if parameter_is_default("api_auth_token") and cfg.api_auth_token:
        api_auth_token = cfg.api_auth_token

    enable_watch = not no_watch
    enable_source_catchup = not no_source_catchup
    enable_browser_capture = not no_browser_capture
    enable_api = not no_api
    if not enable_watch and not enable_browser_capture and not enable_api:
        raise click.UsageError("at least one daemon component must be enabled")

    atexit.register(_cleanup_pidfile)

    sources = _watch_sources_from_roots(roots, browser_capture_spool_path=spool_path)
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
                enable_source_catchup=enable_source_catchup,
                enable_browser_capture=enable_browser_capture,
                browser_capture_host=host,
                browser_capture_port=port,
                browser_capture_spool_path=spool_path,
                browser_capture_allow_remote=insecure_allow_remote,
                browser_capture_auth_token=browser_capture_auth_token,
                browser_capture_allow_no_auth=browser_capture_allow_no_auth,
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
    sources = _watch_sources_from_roots(roots)

    click.echo(
        f"Watching {len(sources)} source(s); debounce={debounce_s}s. Ctrl-C to stop.",
        err=True,
    )
    asyncio.run(run_live_watcher(sources=sources, debounce_s=debounce_s))


__all__ = [
    "_ensure_fts_startup_readiness",
    "health_command",
    "main",
    "run_command",
    "run_daemon_services",
    "run_live_watcher",
    "status_command",
    "watch_command",
]
