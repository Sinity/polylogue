"""Command line entrypoint for long-running Polylogue services."""

from __future__ import annotations

import asyncio
import atexit
import contextlib
import faulthandler
import fcntl
import functools
import os
import sqlite3
import sys
import threading
import time
import uuid
from collections.abc import Awaitable, Callable, Coroutine, Iterator, Mapping, Sequence
from datetime import UTC, datetime
from functools import partial
from http.server import ThreadingHTTPServer
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, TypeVar, cast

import click

from polylogue.api import Polylogue
from polylogue.browser_capture.receiver import resolve_receiver_auth_token
from polylogue.browser_capture.server import BrowserCaptureHTTPServer, make_server
from polylogue.core.compute import BoundedComputeAdapter, publish_compute_adapter, reset_compute_adapter
from polylogue.core.compute_cancel import check_compute_cancelled
from polylogue.core.degraded import DegradedReason, set_degraded
from polylogue.core.durable_fs import atomic_create
from polylogue.core.enums import ValidationMode
from polylogue.core.json import dumps
from polylogue.core.loopback import bind_hosts_overlap, is_loopback_host
from polylogue.core.stage_admission import (
    StageWriteAdmission,
    admit_stage_write,
    stage_write_admission,
)
from polylogue.daemon.api_auth import API_ALLOW_NO_AUTH_ENV
from polylogue.daemon.api_auth import resolve_api_auth_token as resolve_api_auth_token
from polylogue.daemon.event_bus import IngestCommitted, daemon_event_bus
from polylogue.daemon.health import (
    HealthSeverity,
    HealthTier,
    _check_schema_version_fast,
    check_health,
    durable_tier_schema_mismatch,
    format_health_lines,
    resolve_health_tiers,
)
from polylogue.daemon.intake import AdmissionOutcome, AdmissionResult, FairIntakeDispatcher, IntakeClassSpec
from polylogue.daemon.lineage_startup import LineageStartupCensus
from polylogue.daemon.lineage_startup import census_lineage_startup_sync as _census_lineage_startup_sync
from polylogue.daemon.notification_backends import NotificationBackend
from polylogue.daemon.periodic import daemon_periodic_runner, watcher_registered_gate
from polylogue.daemon.service_halt import HaltReason, HaltRegistry, UnitKind, unit_id
from polylogue.daemon.services import (
    PRODUCTION_PROFILE,
    DaemonServiceSpec,
    ServiceCapability,
    ServiceProfile,
    ServiceState,
)
from polylogue.daemon.supervisor import DaemonSupervisor
from polylogue.daemon.write_coordinator import (
    DaemonWriteCoordinator,
    DaemonWriteThreadBridge,
    daemon_write_coordinator,
)
from polylogue.logging import (
    DEBUG,
    ERROR,
    INFO,
    WARNING,
    configure_events,
    configure_logging,
    current_context,
    emit,
    propagate,
    set_run_context,
    shutdown_events,
    span,
)
from polylogue.operations.drive_readiness import DriveCatchupReport, DriveCatchupState
from polylogue.operations.embedding_lifecycle import (
    ensure_embedding_lifecycle_startup as _ensure_embedding_lifecycle_startup_sync,
)
from polylogue.sources.live import LiveWatcher, WatchSource
from polylogue.sources.live.sqlite_locking import is_transient_sqlite_lock
from polylogue.sources.live.watcher import POLYLOGUE_OWNED_SOURCE_NAMES, daemon_watch_sources, default_sources

# The daemon ring's seam onto the storage checkpoint and one-tier writer
# factories: daemon modules take them from here rather than each reaching
# into storage on its own.
from polylogue.storage.sqlite.connection_profile import (
    open_isolated_write_connection as open_isolated_write_connection,
)
from polylogue.storage.sqlite.wal_checkpoint import (
    checkpoint_connection as checkpoint_connection,
)


def validate_api_bind_policy(*, enabled: bool, host: str, allow_remote: bool, auth_token: str | None) -> None:
    """Enforce the daemon API's remote-bind policy before service startup."""
    if enabled and not is_loopback_host(host):
        if not allow_remote:
            raise click.UsageError(
                f"--api-host={host} is not a loopback address. "
                f"Add --insecure-allow-remote to accept the risk of exposing the daemon API."
            )
        if not auth_token:
            raise click.UsageError(
                f"--api-host={host} with --insecure-allow-remote requires --api-auth-token "
                f"(or an auto-minted token; drop --api-allow-no-auth). "
                f"Remote binding without authentication is not supported."
            )


if TYPE_CHECKING:
    from polylogue.daemon.derivation import DerivationReport
    from polylogue.daemon.events import DaemonEventRecord
    from polylogue.daemon.fts_convergence import FtsConvergenceOwner
    from polylogue.daemon.http import DaemonAPIHTTPServer
    from polylogue.daemon.intake_adapters import ColdBuildGeneration
    from polylogue.daemon.lifecycle import DaemonLifecycle
    from polylogue.daemon.operation_runtime import DaemonOperationRuntime
    from polylogue.daemon.raw_observation_owner import RawObservationConvergenceOwner
    from polylogue.daemon.session_profile_composition import ComposedSessionProfiles, SessionProfileCallback
    from polylogue.maintenance.raw_authority import ArchiveWriterRebuildExclusion
    from polylogue.operations.raw_observation_owner import RawObservationReplacement
    from polylogue.sources.live.cursor import CursorStore
    from polylogue.sources.live.watcher import EmbeddingConvergenceOwner
    from polylogue.storage.blob_publication import BlobPublicationReconciliation

_CONVERGENCE_DEBT_RETRY_INTERVAL_SECONDS = 60
#: Wall budget for back-to-back bounded session-derivation passes in one
#: periodic convergence tick; the tick interval leaves the rest for the others.
_SESSION_PROFILE_BACKLOG_SECONDS = 45.0
_SESSION_PROFILE_AUDIT_INTERVAL_SECONDS = 60
#: Debt rows one retry tick inspects, shared by the admitted pass and the
#: lease-free embedding pass that precedes it so both see the same window.
_CONVERGENCE_DEBT_RETRY_LIMIT = 100
#: Wall budget for draining due debt page after page within one retry tick.
#: One page per tick made the drain cadence-bound: a cold build records a few
#: deferred rows per admitted file, and 100 rows a minute is days of backlog
#: for a full archive. Pages continue while they make progress, inside a
#: budget shorter than the tick so ticks never pile up.
_CONVERGENCE_DEBT_DRAIN_BUDGET_SECONDS = 45.0
#: Convergence-debt stages whose backlog has its own recurring domain owner.
#: The generic drain neither retries nor reports on these: the owner does.
#: ``raw_retention`` is drained by ``LiveBatchProcessor`` on live-ingest passes
#: and by the convergence tick's watcher callback when no source changed
#: (``polylogue.sources.live.batch.RAW_RETENTION_STAGE``). Admission
#: refusals are also retried by that pass, which rechecks the source path and
#: clears its debt after successful convergence.
_OWNED_DEBT_STAGES = frozenset(
    {"derived", "fts", "fts_readiness", "raw_parse_recovery", "embed", "raw_retention", "live_ingest_admission"}
)

T = TypeVar("T")
_RAW_MATERIALIZATION_CONVERGENCE_INTERVAL_SECONDS = 30

# Parse passes checkpoint between raw batches. Acquisition has no pass-time
# limit, but its downloads and preparation hold no archive writer lease.
_DRIVE_CATCHUP_MAX_PASS_SECONDS = 20.0
# A spool file younger than this is in the live route's normal debounce/
# batch flow, not stalled; only older cursor-less files park the conveyor.
_SPOOL_PENDING_GRACE_SECONDS = 300


async def _run_startup_embedding_lifecycle(coordinator: DaemonWriteCoordinator, archive_root_path: Path) -> Path:
    """Run embedding lifecycle recovery before any embedding maintenance starts."""
    return await coordinator.run_sync(
        "startup.embedding_lifecycle",
        _ensure_embedding_lifecycle_startup_sync,
        archive_root_path,
    )


async def _run_startup_lineage_census() -> LineageStartupCensus:
    """Measure dangling lineage branch points at startup.

    Read-only by construction: the census reports what the writer's own scoped
    correction left behind, and a component that could repair it would hide
    the producer defect instead (polylogue-6kur AC4). It therefore takes no
    write lease and never enters the write coordinator.
    """
    return await asyncio.to_thread(_census_lineage_startup_sync)


def _lineage_startup_lifecycle_phase(census: LineageStartupCensus) -> str:
    """A converged census is a ready component; a dangling edge is a degraded one.

    Named rather than inlined so the mapping itself is assertable: the whole
    point of the census is that a non-zero count reaches an operator as a
    condition, and a component that reported ``ready`` regardless would put it
    back where the deleted startup sweep left it.
    """
    return "component_ready" if census.converged else "component_degraded"


_BLOB_REFERENCE_RESTORE_CONVERGENCE_BATCH_LIMIT = 25
_SCHEMA_PREFLIGHT_RECHECK_INTERVAL_SECONDS = 60
#: How long shutdown waits for compute workers to exit before naming them.
#: It bounds reporting only: a worker still running is not interrupted.
_COMPUTE_JOIN_TIMEOUT_S = 5.0
#: Cadences that used to be bare literals inside their own ``while True``.
#: They live here so the runner, the service registry and a reader of this
#: module see one value per loop (polylogue-74wvj).
_WAL_CHECKPOINT_INTERVAL_SECONDS = 300
_STATUS_SNAPSHOT_REFRESH_INTERVAL_SECONDS = 10
_HEARTBEAT_INTERVAL_SECONDS = 900
_DB_OPTIMIZE_INTERVAL_SECONDS = 86_400

# polylogue-5xxmc: watcher readiness sequences a maintenance loop behind
# watch registration (see ``_bridge_watcher_registered`` below) so
# archive-wide convergence work never races a starting watcher for the single
# writer. Acquisition itself is the dispatcher's, so the event is now set as
# soon as the watch is registered. That sequencing is a startup-ordering aid,
# not a permanent kill switch -- if the watcher never registers
# (crash, hang, or any other path that leaves the bridge task without a source
# event to forward) every gated loop must still eventually run instead of
# parking on ``Event.wait()`` forever. Verified live 2026-08-03: gated
# maintenance loops frozen, convergence_debt retries stalled since 2026-07-31
# with zero journal signal beyond the periodic schema_version health line.
_WATCHER_REGISTRATION_TIMEOUT_SECONDS = 1800.0  # 30 minutes


def _archive_root_exists() -> bool:
    from polylogue.paths import archive_root

    return archive_root().exists()


def _health_check_interval_s() -> float:
    from polylogue.config import load_polylogue_config

    return float(load_polylogue_config().health_check_interval_s)


async def _await_watcher_registration(
    watcher_registered: asyncio.Event | None,
    *,
    loop_name: str,
    timeout_s: float = _WATCHER_REGISTRATION_TIMEOUT_SECONDS,
) -> None:
    """Wait for watcher registration, bounded by ``timeout_s``.

    Preserves the exact prior behavior when the gate is released promptly
    (or never supplied): the wait returns as soon as ``watcher_registered``
    is set. Only a gate that stays unset for the full timeout gets a single
    WARNING and the loop proceeds without having observed registration.
    """
    if watcher_registered is None or watcher_registered.is_set():
        return
    try:
        await asyncio.wait_for(watcher_registered.wait(), timeout=timeout_s)
    except TimeoutError:
        emit(
            "daemon.watcher_registered.timeout",
            level=WARNING,
            outcome="unmeasured",
            reason="watcher_registration_not_observed",
            loop=loop_name,
            timeout_ms=round(timeout_s * 1000, 3),
        )


# The full set of daemon-owned periodic maintenance loops withheld while the
# watcher is schema-blocked (see ``watcher_blocked`` in
# ``run_daemon_services``) -- the whole ``if not watcher_blocked:`` block
# below is skipped in that state, so none of these loops are even created,
# let alone run, until an operator resolves the schema mismatch and the
# daemon is restarted (or ``_periodic_schema_preflight_recheck`` detects
# recovery and restarts it). Named here so the startup alert can say exactly
# what is frozen instead of leaving it to be inferred from source.
_SCHEMA_BLOCKED_MAINTENANCE_LOOP_NAMES: tuple[str, ...] = (
    "raw materialization convergence",
    "session insight convergence",
    "convergence debt retry",
    "wal checkpoint",
    "fts merge",
    "heartbeat",
    "embedding backlog catch-up",
    "embedding orphan reconcile",
    "db optimize",
    "status snapshot refresh",
    "judgment automation sweep",
    "fts identity drift recompute",
    "fts orphan audit",
    "blob gc check",
    "blob publication reconciliation",
    "secret scan sweep",
)
_SCHEMA_BLOCKED_OPTIONAL_DRIVE_CATCHUP_LOOP_NAME = "drive source catch-up"


async def _periodic_schema_preflight_recheck() -> None:
    """Poll for tier-layout recovery while the watcher is schema-blocked.

    The blocked decision is made once at startup, but tiers heal out of
    band: an operator migrates a durable tier, a derived tier is rebuilt,
    or a disposable tier bootstraps. Without this loop the daemon stayed
    watcher-less forever after recovery (2026-07-18: an hour of dead
    watcher until a manual restart). When the layout becomes ready, raise
    so the supervisor restarts the daemon into a healthy boot — the
    startup sequence is the only sanctioned way to construct the watcher.
    """
    await daemon_periodic_runner().run(
        "schema_preflight_recheck",
        _schema_preflight_recheck_once,
        interval_s=_SCHEMA_PREFLIGHT_RECHECK_INTERVAL_SECONDS,
        # The recovery signal *is* an exception: it must reach the supervisor
        # so the daemon restarts into a healthy boot.
        on_error="propagate",
    )


async def _schema_preflight_recheck_once() -> None:
    alert = _check_schema_version_fast()
    if alert.severity != HealthSeverity.CRITICAL:
        emit(
            "daemon.schema_preflight.recovered",
            outcome="ok",
            reason="restart_required",
            error_detail=alert.message,
        )
        raise RuntimeError("schema preflight recovered; restart required to start the live watcher")


# Track the pidfile path for atexit cleanup.
_pidfile_path: Path | None = None
_daemon_lifecycle: DaemonLifecycle | None = None
_daemon_supervisor: DaemonSupervisor | None = None


def _set_active_supervisor(supervisor: DaemonSupervisor | None) -> None:
    global _daemon_supervisor
    _daemon_supervisor = supervisor


def active_supervisor() -> DaemonSupervisor | None:
    """Return this process' service supervisor, if a daemon is composed here.

    Status reads service state from the supervisor rather than inferring it
    from whichever loops happen to be alive.
    """
    return _daemon_supervisor


def _degrade_for_failed_service(spec: DaemonServiceSpec, exc: BaseException) -> None:
    """Mark the daemon degraded when a ``DEGRADE`` service fails."""
    set_degraded(
        DegradedReason(
            code="service_failed",
            message=f"{spec.name}: {type(exc).__name__}: {exc}",
            detail={"service": spec.name, "owner": spec.owner},
        )
    )


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


def _watch_sources(*, hermes_root: Path | None = None) -> tuple[WatchSource, ...]:
    """The daemon's watch set (see :func:`daemon_watch_sources`)."""
    return daemon_watch_sources(hermes_root=hermes_root)


def _is_polylogue_owned_source(source: WatchSource) -> bool:
    """Whether Polylogue owns *source*'s directory, decided by role.

    The browser-capture spool and the archive inbox are Polylogue's, and so
    is the primary writable hook spool its installed hooks write. A
    read-only legacy spool and every provider directory belong to someone
    else and are never created here.
    """
    return source.name in POLYLOGUE_OWNED_SOURCE_NAMES or source.role == "primary-writable"


def _active_index_db_path() -> Path:
    """Return the archive-rooted ``index.db`` path for daemon maintenance.

    Deliberately the plain ``archive_root() / "index.db"`` construction, not
    a pointer-resolved active-generation path: this value is threaded as the
    ``db_path`` anchor into :func:`~polylogue.daemon.convergence_stages.make_default_convergence_stages`
    and friends, whose entire stage family relies on ``db_path.parent`` being
    the durable-tier archive root (never a ``.index-generations/<gen>``
    subdirectory) to derive ``ops.db``/``embeddings.db``/``source.db``/
    ``user.db`` siblings correctly. SQLite transparently follows the
    ``index.db`` symlink when connecting, so this is equally correct for
    opening the active generation's content.
    """
    from polylogue.paths import archive_root

    return archive_root() / "index.db"


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

    This is a best-effort fast path only, and deliberately not the guarantee:
    a wiped archive has no ``index.db`` at startup (it is created later and
    lazily by the ingest write path), so this returns having configured
    nothing. ``_periodic_fts_merge`` re-checks the index on every iteration
    and ensures ``automerge=0`` there, which is what actually holds for the
    first post-wipe rebuild.
    """
    db = _active_index_db_path()
    if not db.exists():
        return
    try:
        await daemon_write_coordinator().run_sync("startup.fts_automerge", _configure_fts_automerge_sync, db)
    except Exception as exc:
        emit(
            "daemon.fts_automerge.configure_failed",
            level=WARNING,
            outcome="error",
            path=db,
            error_type=type(exc).__name__,
            error_detail=str(exc),
        )


def _configure_fts_automerge_sync(db: Path) -> None:
    from polylogue.daemon.fts_automerge import configure_fts_automerge_sync
    from polylogue.storage.sqlite.connection_profile import open_connection

    conn = open_connection(db, timeout=30.0, archive_root=db.parent)
    try:
        configure_fts_automerge_sync(conn)
    finally:
        conn.close()


#: Live incident 2026-07-27: a long-running ingest/append pass can hold the
#: sole-writer lock for 14+ minutes straight (observed: 860s for one live
#: watcher catch-up chunk). At the previous 300s interval this starved the
#: merge task outright during exactly the backlog conditions it exists to
#: help with -- messages_fts_data grew to 700K+ rows (steady state is far
#: lower) before a merge got a turn, and unmerged segment bloat itself makes
#: every subsequent per-block FTS5 insert slower, which lengthens the next
#: writer hold and starves the merge task further: a genuine, self-
#: reinforcing degradation spiral, not merely a slow one-off pass. Shortening
#: the interval does not touch the per-call work bound (still the same small,
#: bounded 500-work-unit / 2-4 MiB chunk -- the merge pass itself must never
#: become a long writer hold, see fts_automerge.py) -- it only asks for the
#: writer's turn more often, so once contention eases the backlog clears
#: roughly 5x faster than before. This does not fix worst-case starvation
#: (a single 14-minute hold still blocks every queued actor including this
#: one) -- see polylogue-de2a for the harder writer-fairness question.
_FTS_MERGE_INTERVAL_SECONDS = 60

#: The recurring checkpoint coordinator's writer actor. Named here so its hold
#: budget in ``write_coordinator.py`` binds to the one caller that uses it.
WAL_CHECKPOINT_ACTOR = "maintenance.wal_checkpoint"


async def _periodic_fts_merge() -> None:
    """Run a bounded FTS5 merge every 60s to amortise segment cost (#1851).

    With automerge=0, level-0 FTS5 segments accumulate over time.  This
    periodic pass merges them in bounded 500-work-unit chunks so query
    performance stays good without ever paying the full merge cost in a
    single write transaction.
    """
    await daemon_periodic_runner().run(
        "fts_merge",
        _fts_merge_once,
        interval_s=_FTS_MERGE_INTERVAL_SECONDS,
        precondition=lambda: _active_index_db_path().exists(),
        error_event="daemon.fts_merge.failed",
    )


async def _fts_merge_once() -> None:
    from polylogue.daemon.fts_automerge import run_periodic_fts_merge_sync

    await daemon_write_coordinator().run_sync(
        "maintenance.fts_merge", run_periodic_fts_merge_sync, _active_index_db_path()
    )


async def _periodic_wal_checkpoint() -> None:
    """Run the process' only ordinary WAL checkpoints, every 5 minutes.

    Recurring escalation is PASSIVE and nothing further: RESTART needs a
    declared quiescent boundary and TRUNCATE belongs to seal, shutdown and
    offline generation lifecycle. A busy result therefore retains the WAL and
    reports its blockers instead of retrying, because the reader holding those
    frames is doing legitimate work.

    It runs under the writer gate with its own actor, so its wait and hold are
    attributed to checkpointing and measured against
    ``CHECKPOINT_HOLD_BUDGET_S`` rather than absorbed into a publication hold.
    """
    await daemon_periodic_runner().run(
        "wal_checkpoint",
        _wal_checkpoint_once,
        interval_s=_WAL_CHECKPOINT_INTERVAL_SECONDS,
        precondition=_archive_root_exists,
        error_event="daemon.wal_checkpoint.failed",
    )


async def _wal_checkpoint_once() -> None:
    from polylogue.paths import archive_root
    from polylogue.storage.sqlite.connection_profile import CHECKPOINT_HOLD_BUDGET_S
    from polylogue.storage.sqlite.wal_checkpoint import checkpoint_archive_wals

    root = archive_root()
    observations = await daemon_write_coordinator().run_sync(
        WAL_CHECKPOINT_ACTOR,
        checkpoint_archive_wals,
        root,
        reason="periodic",
        escalation="recurring",
        collect_blockers=True,
    )
    for observation in observations:
        if not observation.ran and observation.error is None:
            continue
        failed = observation.error is not None
        blockers = ",".join((*observation.blocking_read_frames[:5], *observation.blocking_processes[:5]))
        emit(
            "daemon.wal_checkpoint.observed",
            level=WARNING if failed else INFO,
            outcome="error" if failed else ("degraded" if observation.busy_pages else "ok"),
            reason="checkpoint_error" if failed else ("reader_held_frames" if observation.busy_pages else "clean"),
            loop="wal checkpoint",
            mode=str(observation.mode),
            bytes_before=observation.wal_bytes_before,
            bytes_after=observation.wal_bytes_after,
            busy_pages=observation.busy_pages,
            checkpointed_pages=observation.checkpointed_pages,
            duration_ms=round(observation.elapsed_s * 1000, 3),
            budget_ms=round(CHECKPOINT_HOLD_BUDGET_S * 1000, 3),
            error_detail=f"{observation.error or ''} blockers={blockers}".strip(),
        )


async def _periodic_status_snapshot_refresh() -> None:
    """Refresh the rich daemon status snapshot outside request handlers."""
    from polylogue.daemon.discovery_progress import log_discovery_progress_if_due
    from polylogue.daemon.status_snapshot import refresh_status_snapshot

    async def once() -> None:
        await asyncio.to_thread(log_discovery_progress_if_due)
        await asyncio.to_thread(refresh_status_snapshot)

    await daemon_periodic_runner().run(
        "status_snapshot_refresh",
        once,
        interval_s=_STATUS_SNAPSHOT_REFRESH_INTERVAL_SECONDS,
        # Every surface reads this snapshot: publish one before the first sleep
        # so a fresh daemon is never serving an absent snapshot for a cadence.
        run_first=True,
        error_event="daemon.status_snapshot.refresh_failed",
    )


async def _run_drive_source_catchup_once(
    session_profile_callback: SessionProfileCallback | None,
    *,
    raw_owner: RawObservationConvergenceOwner | None,
    compute_owner: BoundedComputeAdapter,
) -> DriveCatchupReport:
    """Acquire and parse configured Drive sources once.

    The live watcher only observes filesystem roots. Google Drive sources are
    remote acquisition roots, so the daemon has to run their catch-up through
    the staged acquisition pipeline explicitly.
    """
    from polylogue.config import get_config
    from polylogue.daemon.drive_catchup import DriveCatchupExecution
    from polylogue.pipeline.services.parsing import ParsingService
    from polylogue.services import build_runtime_services

    config = get_config()
    sources = [source for source in config.sources if source.is_drive]
    if not sources:
        emit(
            "daemon.drive_catchup.pass.empty",
            level=DEBUG,
            outcome="empty",
            reason="no_drive_sources_configured",
            loop="drive source catch-up",
        )
        return DriveCatchupReport(
            DriveCatchupState.COMPLETE, enumerated_count=0, acquired_count=0, materialization_pending=0
        )

    from polylogue.operations.drive_readiness import (
        drive_readiness_observation,
        inspect_current_drive_readiness,
        reobserve_drive_readiness,
    )

    observation = drive_readiness_observation(config.archive_root)
    resume = observation.resume_materialization
    if not resume:
        observation.close()
    services = build_runtime_services(config=config, db_path=config.db_path)
    try:
        with span("daemon.drive_catchup.pass", loop="drive source catch-up") as pass_span:
            repository = services.get_repository()
            execution = DriveCatchupExecution(daemon_write_coordinator(), compute_adapter=compute_owner)
            parser = ParsingService(
                repository=repository,
                archive_root=config.archive_root,
                config=config,
                execution=execution,
                retained_runner=raw_owner.ingest_retained_raw_ids if raw_owner is not None else None,
            )
            result = await parser.ingest_sources(
                sources=sources,
                stage="all",
                parse_records=raw_owner is not None,
                max_pass_seconds=_DRIVE_CATCHUP_MAX_PASS_SECONDS,
                skip_acquire=resume,
            )
            if not resume:
                observation.witnesses.update(result.acquire_result.drive_witnesses)
            else:
                # Recheck currency after a parse checkpoint without reacquiring
                # unchanged bytes or throwing away the retained membership.
                def reobserve() -> None:
                    reobserve_drive_readiness(observation, config.drive_config)

                await execution.prepare(reobserve)
            session_ids = tuple(sorted(result.parse_result.processed_ids))
            if session_ids and session_profile_callback is not None:
                try:
                    await session_profile_callback(session_ids)
                except Exception as exc:
                    emit(
                        "daemon.drive_catchup.session_profile_failed",
                        level=WARNING,
                        outcome="degraded",
                        reason="session_profile_convergence_failed",
                        sessions=len(session_ids),
                        error_type=type(exc).__name__,
                        error_detail=str(exc),
                    )
            errors = int(result.acquire_result.errors)
            counts: dict[str, object] = {
                "sources": len(sources),
                "raws": len(result.acquire_result.raw_ids),
                "sessions": int(result.parse_result.counts["sessions"]),
                "changed": len(session_ids),
                "errors": errors,
                "budget_ms": round(_DRIVE_CATCHUP_MAX_PASS_SECONDS * 1000, 3),
            }

            report = await execution.prepare(
                lambda: inspect_current_drive_readiness(
                    config.archive_root,
                    sources,
                    observation.witnesses,
                    changed_count=len(session_ids),
                    raw_owner_available=raw_owner is not None,
                    validation_mode=ValidationMode.from_string(config.schema_validation),
                )
            )
            if report.state is DriveCatchupState.COMPLETE:
                pass_span.ok(**counts)
            else:
                pass_span.degraded("drive_" + report.state.value, **counts)
            observation.resume_materialization = (
                report.state is DriveCatchupState.PENDING
                and "drive_listing_changed" not in report.gaps
                and "drive_raw_not_retained" not in report.gaps
            )
            return report
    finally:
        await services.close()


async def _run_drive_source_catchup_safely(
    session_profile_callback: SessionProfileCallback | None,
    *,
    raw_owner: RawObservationConvergenceOwner | None,
    compute_owner: BoundedComputeAdapter,
) -> DriveCatchupReport:
    """Run Drive catch-up without letting remote-source failures kill daemon."""
    from polylogue.core.compute import DaemonOperationCancelled

    try:
        return await _run_drive_source_catchup_once(
            session_profile_callback, raw_owner=raw_owner, compute_owner=compute_owner
        )
    except (asyncio.CancelledError, DaemonOperationCancelled):
        raise
    except Exception as exc:
        emit(
            "daemon.drive_catchup.failed",
            level=WARNING,
            outcome="error",
            loop="drive source catch-up",
            error_type=type(exc).__name__,
            error_detail=str(exc),
        )
        return DriveCatchupReport(DriveCatchupState.RETRYABLE, gaps=(f"drive_catchup_failed:{type(exc).__name__}",))


async def _periodic_heartbeat(*, sources: tuple[WatchSource, ...] = ()) -> None:
    """Log daemon heartbeat with archive stats every 15 minutes."""
    if not sources:
        sources = default_sources()

    async def once() -> None:
        db = _active_index_db_path()
        n_sessions, n_messages, _noun = await asyncio.to_thread(_heartbeat_counts, db)
        emit(
            "daemon.heartbeat.indexed",
            outcome="ok",
            loop="heartbeat",
            sessions=n_sessions,
            messages=n_messages,
        )
        await asyncio.to_thread(_log_spool_depth_if_notable)

    await daemon_periodic_runner().run(
        "heartbeat",
        once,
        interval_s=_HEARTBEAT_INTERVAL_SECONDS,
        precondition=lambda: _active_index_db_path().exists(),
        error_event="daemon.heartbeat.query_failed",
    )


_BROWSER_CAPTURE_SPOOL_DEPTH_ALERT_CAP = 2000


def _log_spool_depth_if_notable() -> None:
    """Log pending-queue depth once per heartbeat when it looks abnormal.

    Bounded/capped counts only -- this must never itself become an O(n) scan
    of an unboundedly large backlog. Hook carriers are deliberately absent:
    they are ordinary watched files whose backlog is the dispatcher's intake
    backlog, so a second hook-specific depth probe would be a parallel ledger
    of the same fact (polylogue-k3ahm).
    """
    from polylogue.hooks import hook_install_sidecar_drift

    for harness in ("claude-code", "codex"):
        with _alert_probe("hook_install_sidecar_drift", component=harness):
            drift = hook_install_sidecar_drift(harness)
            if drift:
                emit(
                    "daemon.hook_install.sidecar_drift",
                    level=WARNING,
                    outcome="degraded",
                    reason="stale_sidecar_dir",
                    loop="heartbeat",
                    component=harness,
                    files=len(drift),
                    error_detail=", ".join(str(path) for path in drift),
                )
    with _alert_probe("browser_capture_spool_depth", component="browser-capture"):
        browser_capture_depth = _browser_capture_spool_pending_file_count(cap=_BROWSER_CAPTURE_SPOOL_DEPTH_ALERT_CAP)
        if browser_capture_depth >= _BROWSER_CAPTURE_SPOOL_DEPTH_ALERT_CAP:
            emit(
                "daemon.browser_capture_spool.backlog",
                level=WARNING,
                outcome="degraded",
                reason="spool_not_draining",
                loop="heartbeat",
                depth=browser_capture_depth,
                limit=_BROWSER_CAPTURE_SPOOL_DEPTH_ALERT_CAP,
            )


def _browser_capture_spool_pending_file_count(*, cap: int) -> int:
    """Bounded count of ``*.json`` files under the browser-capture spool.

    Purely for observability logging: capped so a large spool cannot turn
    this into the same O(n) hazard the hook spool had. Distinct from
    ``_browser_capture_spool_has_pending_files`` (which answers a
    content-vs-cursor question, not a raw file count).
    """
    from itertools import islice

    from polylogue.paths import browser_capture_spool_root

    spool = browser_capture_spool_root()
    if not spool.exists():
        return 0
    return sum(1 for _ in islice(spool.rglob("*.json"), cap))


async def _periodic_lifecycle_heartbeat(*, interval_s: float | None = None) -> None:
    """Advance the ops-only heartbeat even when archive work is blocked.

    This must remain independent of index stats and convergence: a daemon that
    deliberately keeps only API/health surfaces available after schema
    preflight failure is still alive and must not age into a false vanished
    state.
    """
    from polylogue.daemon.lifecycle import DAEMON_HEARTBEAT_INTERVAL_SECONDS

    interval = DAEMON_HEARTBEAT_INTERVAL_SECONDS if interval_s is None else interval_s

    async def once() -> None:
        lifecycle = _daemon_lifecycle
        assert lifecycle is not None  # guarded by the precondition below
        await daemon_write_coordinator().run_sync("daemon.lifecycle.heartbeat", lifecycle.heartbeat)

    await daemon_periodic_runner().run(
        "lifecycle_heartbeat",
        once,
        interval_s=interval,
        precondition=lambda: _daemon_lifecycle is not None,
        error_event="daemon.lifecycle_heartbeat.write_failed",
    )


async def _reconcile_ended_daemon_runs() -> None:
    """Classify how every earlier run of this archive ended (polylogue-peo).

    One pass per daemon start, as its own service rather than a startup step:
    reading the journal and cgroup evidence happens off the writer at its own
    pace, and only the receipt write takes the writer. A run that cannot be
    reconciled now stays pending in status and is reconciled by the next start.
    """
    from polylogue.operations.daemon_termination import HostTerminationEvidence

    lifecycle = _daemon_lifecycle
    if lifecycle is None:
        return
    # ``to_thread`` carries this context, so the read keeps the run's correlation.
    receipts = await asyncio.to_thread(lifecycle.reconcile_ended_runs, HostTerminationEvidence())
    if not receipts:
        return
    await daemon_write_coordinator().run_sync(
        "daemon.lifecycle.reconcile", lifecycle.record_termination_receipts, receipts
    )
    for receipt in receipts:
        emit(
            "daemon.lifecycle.run_reconciled",
            level=WARNING if receipt.classification.value not in ("clean", "handled_signal") else INFO,
            outcome="ok",
            run_id=receipt.run_id,
            state=receipt.classification.value,
        )


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

    async def once() -> None:
        root = archive_root()
        observations = await daemon_write_coordinator().run_sync(
            "maintenance.db_optimize",
            maybe_optimize_archive_tiers,
            root,
            reason="periodic",
        )
        ran = sum(1 for observation in observations if observation.ran)
        errors = [observation.error for observation in observations if observation.error]
        emit(
            "daemon.db_optimize.completed",
            level=WARNING if errors else INFO,
            outcome="degraded" if errors else "ok",
            reason="tier_errors" if errors else "complete",
            loop="db optimize",
            tiers=ran,
            errors=len(errors),
            error_detail="; ".join(str(error) for error in errors) if errors else "",
        )

    await daemon_periodic_runner().run(
        "db_optimize",
        once,
        # Deliberately no startup optimize: a large archive must not pay broad
        # read IO at the moment live catch-up needs the disk.
        interval_s=_DB_OPTIMIZE_INTERVAL_SECONDS,
        precondition=_archive_root_exists,
        error_event="daemon.db_optimize.failed",
    )


async def _periodic_convergence_check(
    sources: tuple[WatchSource, ...],
    *,
    fts_owner: FtsConvergenceOwner,
    watcher_registered: asyncio.Event | None = None,
    raw_retention_callback: Callable[[], Awaitable[None]] | None = None,
) -> None:
    """Periodically retry recorded convergence debt."""
    db = _active_index_db_path()

    async def once() -> None:
        debt_error = await _retry_convergence_debt_once(db)
        if raw_retention_callback is not None:
            try:
                await raw_retention_callback()
            except Exception as exc:
                emit(
                    "daemon.raw_retention.retry_failed",
                    level=WARNING,
                    outcome="degraded",
                    reason="retention_retry_failed",
                    error_type=type(exc).__name__,
                    error_detail=str(exc),
                )
        await fts_owner.converge()
        if debt_error is not None:
            raise debt_error

    await daemon_periodic_runner().run(
        "convergence_check",
        once,
        interval_s=_CONVERGENCE_DEBT_RETRY_INTERVAL_SECONDS,
        gate=watcher_registered_gate(watcher_registered),
        run_first=True,
    )


async def _periodic_session_profile_audit(
    session_profile_callback: Callable[[tuple[str, ...] | None], Awaitable[object]],
    *,
    watcher_registered: asyncio.Event | None = None,
) -> None:
    """Converge profile demand, then resume the promoted session audit in bounded passes."""
    from polylogue.daemon.session_profile_composition import ComposedSessionProfiles

    async def once() -> None:
        if isinstance(session_profile_callback, ComposedSessionProfiles):
            await session_profile_callback.converge_backlog(_SESSION_PROFILE_BACKLOG_SECONDS)
        else:
            await session_profile_callback(None)

    await daemon_periodic_runner().run(
        "session_profile_audit",
        once,
        interval_s=_SESSION_PROFILE_AUDIT_INTERVAL_SECONDS,
        gate=watcher_registered_gate(watcher_registered),
        run_first=True,
    )


def _drive_sources_configured() -> bool:
    """Whether any configured source is a Drive source.

    Without a readable config the Drive intake class stays unregistered for
    this daemon's life, so the failure is reported rather than read as "no
    Drive sources" (polylogue-hu24g).
    """
    try:
        from polylogue.config import get_config

        return any(source.is_drive for source in get_config().sources)
    except Exception as exc:
        emit(
            "daemon.intake.drive_config_unreadable",
            level=WARNING,
            outcome="degraded",
            reason="config_unreadable",
            error_type=type(exc).__name__,
            error_detail=str(exc),
        )
        return False


@contextlib.contextmanager
def _alert_probe(probe: str, *, component: str) -> Iterator[None]:
    """Run one alert probe; a probe that raises is itself an alert.

    An alert probe swallowed by ``suppress`` can never fire, so its failure is
    reported instead of hidden (polylogue-hu24g).
    """
    try:
        yield
    except Exception as exc:
        emit(
            "daemon.alert_probe.failed",
            level=WARNING,
            outcome="degraded",
            reason="alert_probe_failed",
            loop="heartbeat",
            component=component,
            operation=probe,
            error_type=type(exc).__name__,
            error_detail=str(exc),
        )


async def _retry_convergence_debt_once(db: Path) -> BaseException | None:
    """Run one logged derived-debt retry pass when the archive exists.

    A failed pass is returned rather than swallowed: the caller finishes its
    other stages and then re-raises it, so the periodic loop records it as
    its ``last_error`` (polylogue-hu24g).
    """
    from polylogue.daemon.intake_adapters import active_cold_build_generation

    if active_cold_build_generation() is not None:
        # Debt recorded during a cold build describes the unpromoted candidate,
        # while every stage here reads the active generation. Running them now
        # spends writer admission on the wrong archive, and an archive-wide
        # stage would answer "converged" for the empty active index and clear
        # rows the candidate still owes. Promotion ends the build; the next
        # tick drains against the promoted generation.
        emit(
            "daemon.convergence_debt.pass.skipped",
            level=DEBUG,
            outcome="skipped",
            reason="cold_build_in_progress",
            loop="convergence debt retry",
            path=db,
        )
        return None
    if not db.exists():
        emit(
            "daemon.convergence_debt.pass.skipped",
            level=DEBUG,
            outcome="skipped",
            reason="index_db_absent",
            loop="convergence debt retry",
            path=db,
        )
        return None
    # The span emits its terminal ``.error`` event from ``__exit__``, before
    # this handler runs; the failure is also returned to the caller.
    try:
        await _run_convergence_debt_pass(db)
    except Exception as exc:
        return exc
    return None


async def _run_convergence_debt_pass(db: Path) -> None:
    with span("daemon.convergence_debt.pass", loop="convergence debt retry", path=db) as pass_span:
        try:
            # The drain builds the stage set (which resolves the configured
            # Sinex transport) and walks it. Neither may happen under the
            # writer lease (polylogue-ssplv): each stage bridges its own short
            # write back through the admission bound here.
            from polylogue.core.compute import compute_adapter

            adapter = compute_adapter()
            submitted = adapter.submit(
                propagate(
                    partial(
                        _run_with_stage_admission,
                        _daemon_stage_write_admission(),
                        partial(_drain_convergence_debt_and_frontier, db, compute_adapter=adapter),
                    )
                ),
                admission_class="incremental-background",
                exclusive_bytes=True,
                estimated_bytes=0,
            )
            repaired = await submitted.wait()
        except sqlite3.OperationalError as exc:
            if is_transient_sqlite_lock(exc):
                pass_span.degraded(
                    "archive_busy",
                    error_type=type(exc).__name__,
                    error_detail=str(exc),
                )
                return
            raise
        if repaired:
            pass_span.ok(retried=repaired)
        else:
            pass_span.empty(retried=0)


def _drain_convergence_debt_and_frontier(db: Path, *, compute_adapter: BoundedComputeAdapter) -> int:
    """Derive frontier work even when an unadmitted pass recorded no debt.

    This runs on the periodic pass's original admitted compute worker, with
    its existing stage writer bridge. Completed current coverage skips the
    census; a changed source/cursor watermark or an absent mark requires it.
    """
    from polylogue.core.compute import DaemonOperationCancelled
    from polylogue.operations.raw_frontier_inspection import make_raw_frontier_inspection_stage
    from polylogue.sources.live.cursor import CursorStore

    # Ops is disposable: restore it through the original admitted bootstrap
    # before querying debt, while retaining the pre-drain ownership snapshot.
    cursor = admit_stage_write("maintenance.convergence_debt.initialize", partial(CursorStore, db))
    frontier_debt = cursor.list_convergence_debt(stage="raw_frontier_inspection", limit=1)
    retried = _drain_convergence_debt_backlog(db, compute_adapter=compute_adapter)
    # Recorded debt owns both due execution and backoff. Check before draining
    # as well: a successful due retry may clear its row during this pass.
    if frontier_debt or cursor.list_convergence_debt(stage="raw_frontier_inspection", limit=1):
        return retried
    stage = make_raw_frontier_inspection_stage(db, compute_adapter=compute_adapter)
    if stage.check(db):
        try:
            healthy = stage.execute(db)
        except DaemonOperationCancelled:
            raise
        except Exception as exc:
            check_compute_cancelled()
            admit_stage_write(
                "maintenance.convergence_debt.frontier",
                partial(
                    cursor.record_convergence_debt,
                    stage=stage.name,
                    subject_type="source_path",
                    subject_id=str(db),
                    error=str(exc),
                ),
            )
            raise
        if not healthy:
            check_compute_cancelled()
            admit_stage_write(
                "maintenance.convergence_debt.frontier",
                partial(
                    cursor.record_convergence_debt,
                    stage=stage.name,
                    subject_type="source_path",
                    subject_id=str(db),
                    error="stage state: pending",
                    deferred=True,
                ),
            )
        emit(
            "daemon.raw_frontier_inspection.pass.completed",
            outcome="ok" if healthy else "degraded",
            reason="complete" if healthy else "frontier_inspection_blocked",
            path=db,
        )
    return retried


async def _periodic_raw_materialization_convergence(
    *,
    watcher_registered: asyncio.Event | None = None,
    raw_intake_wakeup: asyncio.Event,
) -> None:
    """Wake fair raw intake after watcher registration."""

    async def once() -> None:
        raw_intake_wakeup.set()

    await daemon_periodic_runner().run(
        "raw_observation_convergence",
        once,
        interval_s=_RAW_MATERIALIZATION_CONVERGENCE_INTERVAL_SECONDS,
        gate=watcher_registered_gate(watcher_registered),
        run_first=True,
        error_event="daemon.raw_materialization.wakeup_failed",
    )


async def _bridge_watcher_registered(
    source: asyncio.Event,
    target: asyncio.Event,
) -> None:
    """Forward watcher readiness to daemon maintenance loops."""
    await source.wait()
    target.set()


async def _reconcile_blob_publications(
    *,
    actor: str = "startup.blob_publications",
    max_count: int | None = None,
    after_publication_id: str | None = None,
) -> BlobPublicationReconciliation | None:
    """Classify crash-left publication reservations against the active archive."""
    from polylogue.paths import archive_root
    from polylogue.storage.archive_identity import resolve_active_index_path
    from polylogue.storage.blob_publication import reconcile_blob_publication_reservations_under_exclusion

    root = archive_root()
    if not (root / "source.db").exists():
        return None
    # Reconciliation only clears rows under a live ArchiveWriterExclusion; the
    # `_under_exclusion` entry point acquires it itself so this startup call
    # cannot silently regress into a no-op reconciliation (polylogue-qs0a).
    outcome = await daemon_write_coordinator().run_sync(
        actor,
        reconcile_blob_publication_reservations_under_exclusion,
        root / "source.db",
        root / "blob",
        index_db_path=resolve_active_index_path(root),
        max_count=max_count,
        after_publication_id=after_publication_id,
    )
    if (
        outcome.cleared_referenced
        or outcome.cleared_missing
        or outcome.retained_referenced
        or outcome.retained_missing
        or outcome.unresolved
    ):
        emit(
            "daemon.blob_publications.classified",
            outcome="ok",
            actor=actor,
            cleared=outcome.cleared_referenced + outcome.cleared_missing,
            retained=outcome.retained_referenced + outcome.retained_missing,
            unresolved=outcome.unresolved,
        )
    retained = outcome.retained_referenced + outcome.retained_missing + outcome.unresolved
    if retained:
        emit(
            "daemon.blob_publications.retained",
            level=WARNING,
            outcome="degraded",
            reason="awaiting_inspection_or_abandonment",
            actor=actor,
            retained=retained,
        )
    return outcome


def _emit_mapped_bytes_budget_check(check: Any) -> None:
    """Report the SQLite mapped-bytes budget against the detected cgroup limit.

    The storage helper this replaces took a prose logger and passed structured
    keywords that the stdlib path discarded outright, so the numbers never
    reached an operator. Emitting here keeps the same three states -- no limit
    detected, at risk, within budget -- as distinguishable events.
    """
    if check.memory_max_bytes is None and check.memory_high_bytes is None:
        emit(
            "daemon.mmap_budget.no_cgroup_limit",
            level=DEBUG,
            outcome="skipped",
            reason="no_cgroup_limit_detected",
            bytes=check.budget_bytes,
            limit=check.effective_memory_budget_bytes,
        )
        return
    at_risk = check.at_risk_limits
    emit(
        "daemon.mmap_budget.checked",
        level=WARNING if at_risk else INFO,
        outcome="degraded" if at_risk else "ok",
        reason=",".join(at_risk) if at_risk else "within_budget",
        bytes=check.budget_bytes,
        limit=check.effective_memory_budget_bytes,
        active=check.concurrent_read_connections,
    )


def _raw_materialized_session_ids(archive: Path, raw_id: str) -> tuple[str, ...]:
    """Return every current session output in one admitted raw component.

    The query deliberately follows the active index generation after raw
    publication. It is not a raw-planning hint: this is the domain output
    that tells the session-profile derivation which partitions the completed
    replay may have changed. A raw may legitimately produce zero sessions or
    split into several; canonical replay can also expand it into a membership
    component, so callers must preserve every affected output partition.
    """
    from polylogue.operations.raw_observation_derivation import raw_observation_output_session_ids

    return raw_observation_output_session_ids(archive, raw_id)


def _derivation_admission(report: DerivationReport, key: str, *, subject: str) -> AdmissionResult:
    """Map one exact-key derivation pass onto a fair-intake admission verdict.

    A discovery or bulk-inspection failure is recorded under the domain-wide
    key ``"*"``, not under ``key``; it still means this pass did not decide
    ``key`` and must be retried, never acknowledged as a duplicate. The
    retained outcome sample may be truncated, so the authoritative failure
    count also counts. No ``actual_cost`` is returned: discovery costed the
    item in payload bytes, and that estimate is what the pass spent.
    """
    from polylogue.daemon.derivation import Outcome

    outcomes = tuple(outcome for outcome in report.outcomes if outcome.key.key in (key, "*"))
    failed = next((outcome for outcome in outcomes if outcome.outcome is Outcome.FAILED), None)
    if failed is not None and failed.key.key == key and failed.terminal_refusal is not None:
        return AdmissionResult(AdmissionOutcome.EXCLUDED, reason=failed.error or failed.terminal_refusal.value)
    if failed is not None or (report.done == 0 and report.failed > 0):
        return AdmissionResult(
            AdmissionOutcome.RETRYABLE,
            reason=(failed.error if failed is not None else None) or f"{subject} derivation failed",
            transient=failed.transient if failed is not None else True,
        )
    pending = next((outcome for outcome in outcomes if outcome.outcome is Outcome.PENDING), None)
    if pending is not None or (report.done == 0 and report.pending > 0):
        return AdmissionResult(
            AdmissionOutcome.RETRYABLE,
            reason=pending.reason.value if pending is not None and pending.reason is not None else f"{subject} pending",
        )
    if report.done:
        return AdmissionResult(AdmissionOutcome.ADMITTED)
    # A concurrent publisher may have made the inspected key valid between
    # discovery and this exact pass. Acknowledgement is then warranted, but
    # only because the canonical output relation said so.
    return AdmissionResult(AdmissionOutcome.DUPLICATE)


async def _converge_raw_materialized_session_profiles(
    archive: Path,
    raw_id: str,
    callback: Callable[[Sequence[str] | None], Awaitable[object]] | None,
) -> None:
    """Hand a completed raw replay to the canonical lease-free derivation.

    ``admit_raw_intake`` calls this only after ``run_sync`` returns, so profile
    compute and its short publications cannot extend the raw replay's writer
    hold. The periodic no-hint sweep still owns retry after a callback failure;
    source admission is already durable at this boundary.
    """
    if callback is None:
        return
    session_ids = _raw_materialized_session_ids(archive, raw_id)
    if not session_ids:
        return
    try:
        await callback(session_ids)
    except Exception as exc:
        emit(
            "daemon.raw_materialization.session_profile_incomplete",
            level=WARNING,
            outcome="degraded",
            reason="lease_free_convergence_did_not_complete",
            raw_id=raw_id,
            sessions=len(session_ids),
            error_type=type(exc).__name__,
            error_detail=str(exc),
        )


def _browser_capture_spool_has_pending_files() -> bool:
    """Whether live browser evidence is stalled short of its ingest route.

    Only files genuinely waiting on the live route's writer count. Terminal
    and failed files remain owned by exclusion/retry policy, while fresh
    files remain with the live watcher's normal debounce and batch route.
    """
    from polylogue.paths import browser_capture_spool_root
    from polylogue.sources.live.batch import fingerprint_file
    from polylogue.sources.live.cursor import CursorStore
    from polylogue.sources.live.watcher import _PARSER_FINGERPRINT

    spool = browser_capture_spool_root()
    if not spool.exists():
        return False
    now = time.time()
    # Read-only probe: the writer initialized the ops tier at startup, and
    # re-initializing here would write against the writer lease.
    cursor_store = CursorStore(_active_index_db_path(), initialize=False)
    for path in spool.rglob("*.json"):
        try:
            stat = path.stat()
        except FileNotFoundError:
            continue
        if now - stat.st_mtime < _SPOOL_PENDING_GRACE_SECONDS:
            continue
        cursor = cursor_store.get_record(path)
        if cursor is None:
            return True
        if cursor.excluded or cursor.failure_count:
            continue
        if (
            cursor.parser_fingerprint != _PARSER_FINGERPRINT
            or cursor.byte_size != stat.st_size
            or cursor.st_dev != stat.st_dev
            or cursor.st_ino != stat.st_ino
            or cursor.mtime_ns != stat.st_mtime_ns
            or cursor.content_fingerprint is None
        ):
            try:
                fingerprint, _last_newline = fingerprint_file(path)
            except OSError:
                return True
            if fingerprint != cursor.content_fingerprint:
                return True
    return False


def _daemon_stage_write_admission() -> StageWriteAdmission:
    """Admission that holds the daemon writer for one stage's write section.

    The section runs on the stage's own thread under a delegated lease, as raw
    observation publication does: a write section may use SQLite connections
    the stage opened for its observation, and a connection never crosses
    threads.
    """
    from polylogue.core.write_lease import adopt_write_lease

    bridge = DaemonWriteThreadBridge(daemon_write_coordinator(), asyncio.get_running_loop())

    def admission(actor: str, work: Callable[[], Any]) -> Any:
        with bridge.hold(actor) as delegation, adopt_write_lease(delegation):
            return work()

    return admission


def _run_with_stage_admission(admission: StageWriteAdmission, work: Callable[[], T]) -> T:
    """Run ``work`` on this thread with the stage writer admission bound."""
    with stage_write_admission(admission):
        return work()


def _drain_convergence_debt_backlog(
    db: Path,
    *,
    compute_adapter: BoundedComputeAdapter,
    limit: int = _CONVERGENCE_DEBT_RETRY_LIMIT,
    budget_s: float = _CONVERGENCE_DEBT_DRAIN_BUDGET_SECONDS,
) -> int:
    """Drain due debt page by page until it is exhausted, stalls, or the budget ends.

    A page that retried nothing, or found fewer due rows than a full page, ends
    the drain: the rest is either absent or waiting for its retry time. Each
    page's ledger write is separately admitted, so the writer interleaves other
    work between pages.
    """
    deadline = time.monotonic() + budget_s
    total = 0
    while True:
        check_compute_cancelled()
        retried, candidates = _drain_convergence_debt_page(db, limit=limit, compute_adapter=compute_adapter)
        total += retried
        if retried == 0 or candidates < limit or time.monotonic() >= deadline:
            return total


def _drain_convergence_debt_once(
    db: Path, *, compute_adapter: BoundedComputeAdapter, limit: int = _CONVERGENCE_DEBT_RETRY_LIMIT
) -> int:
    """Retry one page of due derived convergence debt; return the rows retried."""
    return _drain_convergence_debt_page(db, limit=limit, compute_adapter=compute_adapter)[0]


def _drain_convergence_debt_page(
    db: Path, *, compute_adapter: BoundedComputeAdapter, limit: int = _CONVERGENCE_DEBT_RETRY_LIMIT
) -> tuple[int, int]:
    """Retry due derived convergence debt without rereading source payloads.

    Debt identity is stage-scoped. A retry therefore runs only the recorded
    stage for the recorded subject and updates that same ops-ledger row on a
    further deferral/failure. The legacy ``convergence`` stage remains an
    all-stage fallback for older generic rows.

    ``_OWNED_DEBT_STAGES`` names the stages whose backlog belongs to a domain
    owner rather than to this generic drain. Retrying one here would run the
    wrong executor, and reporting it as ``stage_unimplemented`` would claim the
    row was never retried when its owner retries it every pass.
    """
    from polylogue.daemon.convergence import DaemonConverger
    from polylogue.daemon.convergence_stages import (
        make_default_convergence_stages,
        make_hook_paste_enrichment_stage,
    )
    from polylogue.sources.live.cursor import CursorStore

    # Constructing the store is a write section, not a read: it bootstraps the
    # ops tier, migrates retired convergence-debt stage names and rewinds
    # interrupted attempts. This drain runs on a maintenance worker whose
    # writes are admitted one section at a time, so doing it inline took the
    # daemon's own ops tier through an unleased ``sqlite3.connect`` and every
    # debt pass died on ``UnleasedWriteError`` before reading a row. Admitting
    # it keeps that startup-equivalent recovery, now under the writer.
    cursor = admit_stage_write("maintenance.convergence_debt.initialize", partial(CursorStore, db))
    now = datetime.now(UTC)
    default_stages = make_default_convergence_stages(db, compute_adapter=compute_adapter)
    stages_by_name = {stage.name: stage for stage in default_stages}
    stages_by_name["hook_paste_enrichment"] = make_hook_paste_enrichment_stage(db)
    implemented_stages = set(stages_by_name) | {"convergence"}
    # Intake-owned and unavailable stages keep their original visible debt,
    # but cannot occupy this owner's bounded retry page.
    page = cursor.list_convergence_debt(
        limit=limit,
        retry_due_only=True,
        include_stages=implemented_stages,
        subject_types=("source_path", "session_id"),
        exclude_stages=_OWNED_DEBT_STAGES,
    )
    candidate_debt = [debt for debt in page if _debt_retry_due(debt, now=now)]
    if not candidate_debt:
        return 0, len(page)

    due_debt = candidate_debt

    subject_states: dict[tuple[str, str, str], object] = {}
    converged_whole_archive: dict[str, int] = {}
    run_started_by_stage: dict[str, int] = {}
    retryable_debt = tuple(due_debt)
    if retryable_debt:
        for stage_name in dict.fromkeys(debt.stage for debt in retryable_debt):
            check_compute_cancelled()
            selected_stages = default_stages if stage_name == "convergence" else (stages_by_name[stage_name],)
            stage_debt = tuple(debt for debt in retryable_debt if debt.stage == stage_name)
            paths = tuple(
                dict.fromkeys(Path(debt.subject_id) for debt in stage_debt if debt.subject_type == "source_path")
            )
            session_ids = tuple(
                dict.fromkeys(debt.subject_id for debt in stage_debt if debt.subject_type == "session_id")
            )
            subject_independent = stage_name != "convergence" and stages_by_name[stage_name].subject_independent
            run_started_ms = int(time.time() * 1000)
            run_started_by_stage[stage_name] = run_started_ms
            if subject_independent and paths:
                # The stage's check and work ignore the subject, so one run
                # answers for every subject it owes. Run it once, on one
                # representative subject, instead of once per page of rows.
                paths = paths[:1]
            converger = DaemonConverger(stages=selected_stages)
            path_states, _path_timings = converger.converge_batch(paths)
            session_states, _session_timings = converger.converge_sessions(session_ids)
            if subject_independent and paths:
                representative = path_states.get(paths[0])
                from polylogue.sources.live.convergence_debt import stage_state_value

                measured_stages = getattr(representative, "stages", None)
                if isinstance(measured_stages, dict) and stage_state_value(measured_stages.get(stage_name)) == "done":
                    converged_whole_archive[stage_name] = run_started_ms
                path_states = (
                    {Path(debt.subject_id): representative for debt in stage_debt if debt.subject_type == "source_path"}
                    if representative is not None
                    else {}
                )
            subject_states.update(
                ((stage_name, "source_path", str(path)), state) for path, state in path_states.items()
            )
            subject_states.update(
                ((stage_name, "session_id", session_id), state) for session_id, state in session_states.items()
            )

    check_compute_cancelled()
    retried = admit_stage_write(
        "maintenance.convergence_debt.ledger",
        partial(
            _record_convergence_debt_retries,
            cursor,
            due_debt,
            subject_states,
            converged_whole_archive,
            run_started_by_stage=run_started_by_stage,
        ),
    )
    return retried, len(page)


def _record_convergence_debt_retries(
    cursor: CursorStore,
    due_debt: Sequence[Any],
    subject_states: dict[tuple[str, str, str], object],
    converged_whole_archive: Mapping[str, int] | None = None,
    *,
    run_started_by_stage: Mapping[str, int],
) -> int:
    """Update the ops debt ledger for one drained pass. The only write here."""
    from polylogue.sources.live.convergence_debt import is_deferred_stage_state, stage_state_value

    retried = 0
    # Stages settled here by one archive-wide clear. Their rows are owned by
    # that clear's cutoff: clearing them again per row would delete a row
    # re-recorded after the converging run started.
    settled_stages: set[str] = set()
    cleared_stages: set[str] = set()
    for stage_name, started_ms in (converged_whole_archive or {}).items():
        try:
            cleared = cursor.clear_stage_convergence_debt(stage=stage_name, recorded_before_ms=started_ms)
        except RuntimeError as exc:
            emit(
                "daemon.convergence_debt.stage_clear_failed",
                level=WARNING,
                outcome="degraded",
                reason="stage_clear_not_written",
                stage=stage_name,
                error_detail=str(exc),
            )
        else:
            emit(
                "daemon.convergence_debt.stage_cleared",
                outcome="ok",
                stage=stage_name,
                rows=cleared,
            )
            cleared_stages.add(stage_name)
        settled_stages.add(stage_name)
    for debt in due_debt:
        if debt.stage in settled_stages:
            # Cleared above, or left for the next pass when the clear could
            # not be written; either way not this row's per-subject outcome.
            retried += debt.stage in cleared_stages
            continue
        state = subject_states.get((debt.stage, debt.subject_type, debt.subject_id))
        if state is None:
            # The stage ran but returned no state for this subject: the row's
            # outcome is unmeasured, so its recorded error stands untouched.
            emit(
                "daemon.convergence_debt.retry_unmeasured",
                level=WARNING,
                outcome="degraded",
                reason="retry_state_missing",
                stage=debt.stage,
                subject_type=debt.subject_type,
            )
            continue
        retried += 1
        # The stage released its publication admission before this ledger
        # admission. Live ingest may have recorded a newer failure for this
        # same subject in between. Its row owns the next retry, regardless of
        # whether this older pass succeeded or failed.
        if not cursor.convergence_debt_recorded_before(
            stage=debt.stage,
            subject_type=debt.subject_type,
            subject_id=debt.subject_id,
            recorded_before_ms=run_started_by_stage[debt.stage],
        ):
            continue
        stages_map = getattr(state, "stages", None)
        stages_map = stages_map if isinstance(stages_map, dict) else {}
        settled = (
            bool(stages_map) and all(stage_state_value(value) == "done" for value in stages_map.values())
            if debt.stage == "convergence"
            else stage_state_value(stages_map.get(debt.stage)) == "done"
        )
        if settled:
            cursor.clear_convergence_debt(
                stage=debt.stage,
                subject_type=debt.subject_type,
                subject_id=debt.subject_id,
            )
            continue

        last_error = getattr(state, "last_error", None)
        retry_error = last_error if isinstance(last_error, str) and last_error else "retry did not converge"
        if debt.stage != "convergence":
            cursor.record_convergence_debt(
                stage=debt.stage,
                subject_type=debt.subject_type,
                subject_id=debt.subject_id,
                error=retry_error,
                materializer_version=debt.materializer_version,
                deferred=is_deferred_stage_state(stages_map.get(debt.stage)),
            )
            continue

        # Generic rows predate stage-scoped retry identity. Preserve the old
        # migration behavior by replacing only that generic row with the exact
        # stages that remain pending; other stage rows for the subject survive.
        pending_stages = _pending_convergence_stage_names(stages_map) or ("convergence",)
        cursor.clear_convergence_debt(
            stage=debt.stage,
            subject_type=debt.subject_type,
            subject_id=debt.subject_id,
        )
        for stage in pending_stages:
            cursor.record_convergence_debt(
                stage=stage,
                subject_type=debt.subject_type,
                subject_id=debt.subject_id,
                error=retry_error,
                materializer_version=debt.materializer_version,
                deferred=is_deferred_stage_state(stages_map.get(stage)),
            )
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


def _pending_convergence_stage_names(stages: object) -> tuple[str, ...]:
    if not isinstance(stages, dict):
        return ()
    pending: list[str] = []
    for stage_name, stage_state in stages.items():
        state_value = getattr(stage_state, "value", stage_state)
        if state_value != "done":
            pending.append(str(stage_name))
    return tuple(pending)


async def _periodic_health_check(
    *, backend: NotificationBackend, sources: tuple[WatchSource, ...] | None = None
) -> None:
    """Run periodic health checks with config-driven notification backend.

    Health check tiers and interval are read from PolylogueConfig.
    Notifications are sent through the configured notification backend.
    """

    from polylogue.daemon.notifications import send_notifications

    async def once() -> None:
        from polylogue.config import load_polylogue_config
        from polylogue.daemon.health import check_health

        cfg = load_polylogue_config()
        # The checks are reads on their own read-only connections, so they run
        # off the writer; the cursor-lag sample is the one durable write, and
        # it bridges to the writer through the admission bound here.
        health = await asyncio.to_thread(
            _run_with_stage_admission,
            _daemon_stage_write_admission(),
            partial(check_health, tiers=resolve_health_tiers(cfg.health_check_tiers), sources=sources),
        )
        if health.overall_status != "ok":
            send_notifications(health.alerts, backend=backend, config=cfg.raw)

    await daemon_periodic_runner().run(
        "health_check",
        once,
        # The cadence is operator config and may change while the daemon runs,
        # so it is re-read each tick rather than captured at registration.
        interval_s=_health_check_interval_s,
        error_event="daemon.health_check.failed",
    )


def _acquire_pidfile(pidfile: Path) -> int:
    """Acquire an advisory lock on the pidfile via fcntl.flock.

    Returns the open fd. The lock is held until process exit or explicit close.
    """
    pidfile.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(pidfile, os.O_RDWR | os.O_CREAT, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as err:
        os.close(fd)
        raise RuntimeError(
            f"Could not acquire lock on {pidfile}; another daemon or offline archive writer may own this archive"
        ) from err
    os.ftruncate(fd, 0)
    os.write(fd, str(os.getpid()).encode())
    os.fsync(fd)
    return fd


def _release_pidfile_after_writer_drain(
    pidfile_fd: int | None,
    *,
    writer_drained: bool,
    reason: str = "writer_coordinator_not_drained",
) -> int | None:
    """Release daemon ownership only after every admitted writer is idle."""
    if not writer_drained:
        reason_code, _, _detail = reason.partition(":")
        emit(
            "daemon.pidfile.retained",
            level=ERROR,
            outcome="degraded",
            reason=reason_code,
            path=_pidfile_path,
            error_detail=reason,
        )
        return pidfile_fd
    if pidfile_fd is not None:
        with contextlib.suppress(OSError):
            os.close(pidfile_fd)
    _cleanup_pidfile()
    return None


def _session_id_touches(payload: dict[str, object], key: str) -> list[tuple[str | None, str]]:
    """Extract ``(source_name, session_id)`` pairs from a batch payload list.

    ``key`` is ``"new_sessions"`` or ``"updated_sessions"`` -- the identity
    threaded from :class:`polylogue.sources.live.metrics.LiveBatchMetrics`
    (polylogue-20d.13). Malformed/legacy payloads (missing key, non-list,
    entries without a usable ``session_id``) yield no touches rather than
    raising -- this is an observability fan-out, not a load-bearing write.
    """
    raw = payload.get(key)
    if not isinstance(raw, list):
        return []
    touches: list[tuple[str | None, str]] = []
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        session_id = entry.get("session_id")
        if not isinstance(session_id, str) or not session_id:
            continue
        source_name = entry.get("source_name")
        touches.append((source_name if isinstance(source_name, str) else None, session_id))
    return touches


def _emit_live_batch_event(kind: str, payload: dict[str, object], *, archive_root_path: Path) -> None:
    """Persist a live-ingest batch event and fan out granular #1204 topics.

    The legacy ``ingestion_batch`` kind is preserved verbatim for existing
    consumers (status views, polling fallback). When the batch payload
    carries real per-session identity (``new_sessions`` / ``updated_sessions``,
    threaded from the live-ingest full/append routes) each touched session
    gets its own identity-scoped ``session.appended`` / ``session.updated`` /
    ``message.appended`` event, so a reader with session A open is never
    refreshed by an event that only ever touched session B
    (polylogue-20d.13 -- the defect the description names: "an unscoped
    message event currently refreshes whichever session a browser has
    open"). The summary and every per-session event land in one ledger
    transaction.
    """
    from polylogue.daemon.events import DaemonEventRecord, emit_daemon_events

    records = [DaemonEventRecord(kind, payload)]
    if kind == "ingestion_batch":
        records.extend(_live_batch_session_events(payload))
    emit_daemon_events(records, archive_root_path=archive_root_path)


def _live_batch_session_events(payload: dict[str, object]) -> list[DaemonEventRecord]:
    """The identity-scoped session and message events one live batch implies."""
    from collections import Counter

    from polylogue.daemon.events import message_appended_event, session_appended_event, session_updated_event

    succeeded_raw = payload.get("succeeded_file_count", 0)
    failed_raw = payload.get("failed_file_count", 0)
    succeeded = int(succeeded_raw) if isinstance(succeeded_raw, int | float) else 0
    failed = int(failed_raw) if isinstance(failed_raw, int | float) else 0
    if succeeded <= 0:
        return []

    new_touches = _session_id_touches(payload, "new_sessions")
    updated_touches = _session_id_touches(payload, "updated_sessions")

    if not new_touches and not updated_touches:
        # No real per-session identity was resolved for this batch (a source
        # family or code path this bead's threading does not yet cover).
        # Preserve the pre-#20d.13 unscoped aggregate rather than silently
        # dropping the notification -- a coarse "something changed" signal
        # is still better than none, and existing consumers already treat
        # an absent session_id as "refresh regardless".
        return [
            session_appended_event(source_name=None, succeeded_file_count=succeeded, failed_file_count=failed),
            message_appended_event(session_id=None, source_name=None, appended_count=succeeded),
        ]

    records: list[DaemonEventRecord] = []
    for (source_name, session_id), count in Counter(new_touches).items():
        records.append(
            session_appended_event(source_name=source_name, succeeded_file_count=count, session_id=session_id)
        )
        records.append(message_appended_event(session_id=session_id, source_name=source_name, appended_count=count))
    for (source_name, session_id), count in Counter(updated_touches).items():
        records.append(session_updated_event(session_id=session_id, source_name=source_name, appended_count=count))
        records.append(message_appended_event(session_id=session_id, source_name=source_name, appended_count=count))
    return records


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

    event_payload: dict[str, object] = {
        "phase": phase,
        "status": status,
        "pid": os.getpid(),
        "cwd": os.getcwd(),
        "archive_root": str(archive_root_path),
    }
    if component is not None:
        event_payload["component"] = component
    if payload:
        event_payload.update(payload)
    try:
        async with asyncio.timeout(0.5):
            await daemon_write_coordinator().run_sync(
                f"daemon.lifecycle.{phase}",
                emit_daemon_event,
                "daemon.lifecycle",
                operation_id=None,
                payload=event_payload,
                archive_root_path=archive_root_path,
            )
    except TimeoutError:
        emit(
            "daemon.lifecycle_event.timeout",
            level=WARNING,
            outcome="unmeasured",
            reason="lifecycle_event_write_timed_out",
            phase=phase,
            state=status,
            timeout_ms=500,
        )
    except Exception as exc:
        emit(
            "daemon.lifecycle_event.failed",
            level=WARNING,
            outcome="error",
            phase=phase,
            state=status,
            error_type=type(exc).__name__,
            error_detail=str(exc),
        )


def _retain_rebuild_exclusion_for_undrained_writer(
    rebuild_exclusion: ArchiveWriterRebuildExclusion,
    *,
    writer_drained: bool,
) -> None:
    """Transfer rebuild exclusion to process lifetime after a drain timeout."""
    if not writer_drained:
        rebuild_exclusion.retain_until_process_exit()


def _ownership_retention_reason(*, writer_drained: bool, orphaned_services: Sequence[str]) -> str | None:
    """Why this process must keep archive ownership, or ``None`` to release it.

    Two independent facts retain it and they mean the same thing: this
    process still contains something that can commit.

    An undrained coordinator has admitted work in flight. An *orphaned*
    service is a cancelled child that outlived its declared deadline and is
    still running -- the supervisor reported incomplete shutdown precisely
    because it could not prove the child had stopped. A ``failed`` service is
    deliberately not here: it already terminated with an exception, so it
    cannot write.

    Releasing on either would drop the pidfile, the rebuild exclusion and the
    durable archive lease beside live code, letting a successor writer -- an
    offline rebuild, a second daemon, an interactive CLI mutation -- start
    while the orphan can still commit (polylogue-avmq AC2).
    """
    if not writer_drained:
        return "writer_coordinator_not_drained"
    if orphaned_services:
        return "services_outlived_shutdown_deadline: " + ", ".join(orphaned_services)
    return None


async def _shutdown_writer_coordinator_with_rebuild_exclusion(
    coordinator: DaemonWriteCoordinator,
    rebuild_exclusion: ArchiveWriterRebuildExclusion,
    *,
    timeout: float,
) -> bool:
    """Drain writers or retain rebuild exclusion when drain cannot be proven."""
    try:
        writer_drained = await coordinator.shutdown(timeout=timeout)
    except BaseException:
        rebuild_exclusion.retain_until_process_exit()
        raise
    _retain_rebuild_exclusion_for_undrained_writer(
        rebuild_exclusion,
        writer_drained=writer_drained,
    )
    return writer_drained


def compose_ingest_owner(
    archive_root: Path,
    write_bridge: DaemonWriteThreadBridge,
    *,
    compute_adapter: BoundedComputeAdapter,
    validation_mode: ValidationMode,
) -> tuple[DaemonOperationRuntime, ComposedSessionProfiles]:
    """The ingest owner of a daemon that serves no API: its operation runtime.

    It shares the daemon's compute capacity and session-profile maintenance
    and re-drives accepted ingests exactly as the API server's runtime does;
    the caller starts the re-drive and shuts the runtime down.
    """
    from polylogue.daemon.operation_runtime import DaemonOperationRuntime
    from polylogue.daemon.raw_observation_owner import RawObservationConvergenceOwner
    from polylogue.daemon.session_profile_composition import compose_session_profile_callback

    profiles = compose_session_profile_callback(
        archive_root,
        compute_adapter=compute_adapter,
        write_bridge=write_bridge,
        now=time.time,
    )
    runtime = DaemonOperationRuntime(
        archive_root,
        write_bridge=write_bridge,
        execution_kernel=compute_adapter,
        raw_observation_owner=RawObservationConvergenceOwner(
            archive_root,
            compute_adapter=compute_adapter,
            write_bridge=write_bridge,
            write_coordinator=write_bridge.coordinator,
            validation_mode=validation_mode,
        ),
        owner_loop=write_bridge.owner_loop,
        session_maintenance=profiles.maintenance,
    )
    return runtime, profiles


def daemon_run_id() -> str:
    """Return the one run id of this daemon run, binding it on first use.

    Every record of a run -- each event (the logging run context crosses every
    task, thread and writer-lease hop), the ``daemon_lifecycle`` row, its
    heartbeats and the status projection of that row -- carries this id, so
    evidence about a run that died joins to everything else it left behind
    (polylogue-peo AC1). ``polylogued run`` binds it at process start;
    ``polylogued watch`` and embedded callers of :func:`run_daemon_services`
    are bound here on entry.
    """
    bound = current_context().get("run_id")
    if isinstance(bound, str) and bound:
        return bound
    run_id = uuid.uuid4().hex
    set_run_context(**{**current_context(), "run_id": run_id, "component": "daemon"})
    return run_id


async def run_daemon_services(
    *,
    sources: tuple[WatchSource, ...],
    enable_watch: bool,
    enable_source_catchup: bool = True,
    enable_browser_capture: bool,
    browser_capture_host: str,
    browser_capture_port: int,
    browser_capture_allow_remote: bool = False,
    browser_capture_auth_token: str | None = None,
    browser_capture_allow_no_auth: bool = False,
    browser_capture_extra_origins: tuple[str, ...] = (),
    enable_api: bool = False,
    api_host: str = "127.0.0.1",
    api_port: int = 8766,
    browser_port: int | None = None,
    api_auth_token: str | None = None,
    api_allow_no_auth: bool = False,
    startup_message: str | None = None,
    service_profile: ServiceProfile = PRODUCTION_PROFILE,
    cold_build_index: bool = False,
    listener_info_path: Path | None = None,
) -> None:
    """Run the daemon while excluding every offline index rebuild.

    The lease is intentionally process-lifetime authority rather than a
    per-maintenance-call guard.  Startup readiness, reservation recovery,
    live acquisition, and periodic convergence all mutate source or index
    state; an offline rebuild must therefore refuse the daemon before any of
    those routes can run, and the daemon must prevent a rebuild from starting
    until its writer coordinator has drained.
    """
    from polylogue.core.write_lease import arm_write_lease_enforcement
    from polylogue.daemon.events import EVENT_SUBSCRIBERS
    from polylogue.maintenance.raw_authority import archive_writer_rebuild_exclusion
    from polylogue.paths import archive_root
    from polylogue.storage.sqlite.connection_profile import arm_recurring_checkpoint_owner

    daemon_run_id()
    archive_root_path = Path(archive_root())
    archive_root_path.mkdir(mode=0o700, parents=True, exist_ok=True)
    with (
        archive_writer_rebuild_exclusion(archive_root_path) as rebuild_exclusion,
        arm_write_lease_enforcement(process_wide=True),
        arm_recurring_checkpoint_owner(),
        # This process serves every SSE subscriber, so it alone knows every
        # live cursor and is the only one that prunes the event ledger.
        EVENT_SUBSCRIBERS.owning(),
    ):
        await _run_daemon_services_under_active_writer_lease(
            rebuild_exclusion=rebuild_exclusion,
            sources=sources,
            enable_watch=enable_watch,
            enable_source_catchup=enable_source_catchup,
            enable_browser_capture=enable_browser_capture,
            browser_capture_host=browser_capture_host,
            browser_capture_port=browser_capture_port,
            browser_capture_allow_remote=browser_capture_allow_remote,
            browser_capture_auth_token=browser_capture_auth_token,
            browser_capture_allow_no_auth=browser_capture_allow_no_auth,
            browser_capture_extra_origins=browser_capture_extra_origins,
            enable_api=enable_api,
            api_host=api_host,
            api_port=api_port,
            browser_port=browser_port,
            api_auth_token=api_auth_token,
            api_allow_no_auth=api_allow_no_auth,
            startup_message=startup_message,
            service_profile=service_profile,
            cold_build_index=cold_build_index,
            listener_info_path=listener_info_path,
        )


async def _observe_faulted_baseline_cancellable(
    generation: ColdBuildGeneration, sources: tuple[WatchSource, ...]
) -> Any:
    """Keep a source scan off the writer without joining it at loop shutdown."""
    from polylogue.daemon.discovery_progress import run_source_observation

    return await run_source_observation(lambda cancel: generation.observe_faulted_baseline(sources, cancel=cancel))


async def _run_daemon_services_under_active_writer_lease(
    *,
    rebuild_exclusion: ArchiveWriterRebuildExclusion,
    sources: tuple[WatchSource, ...],
    enable_watch: bool,
    enable_source_catchup: bool = True,
    enable_browser_capture: bool,
    browser_capture_host: str,
    browser_capture_port: int,
    browser_capture_allow_remote: bool = False,
    browser_capture_auth_token: str | None = None,
    browser_capture_allow_no_auth: bool = False,
    browser_capture_extra_origins: tuple[str, ...] = (),
    enable_api: bool = False,
    api_host: str = "127.0.0.1",
    api_port: int = 8766,
    browser_port: int | None = None,
    api_auth_token: str | None = None,
    api_allow_no_auth: bool = False,
    startup_message: str | None = None,
    service_profile: ServiceProfile = PRODUCTION_PROFILE,
    cold_build_index: bool = False,
    listener_info_path: Path | None = None,
) -> None:
    """Run configured daemon components until interrupted.

    *service_profile* selects which declared services run. A focused test
    narrows it; nothing else may, and no caller can start a service the
    registry does not declare.
    """
    from polylogue.daemon import process_start as _process_start
    from polylogue.daemon.intake_adapters import ColdBuildGeneration
    from polylogue.daemon.status_snapshot import configure_browser_capture_status, configure_runtime_components
    from polylogue.paths import archive_root

    global _daemon_lifecycle, _pidfile_path
    _process_start.started_at_wall()
    archive_root_path = Path(archive_root())
    from polylogue.operations.drive_readiness import reset_drive_readiness_observation

    reset_drive_readiness_observation(archive_root_path)
    # The ownership proof is descriptor-backed and therefore requires an
    # existing root. A daemon is also the production first-run entry point, so
    # create an otherwise absent configured root before identity/ownership
    # validation rather than making fresh service startup depend on a separate
    # bootstrap invocation.
    archive_root_path.mkdir(mode=0o700, parents=True, exist_ok=True)
    from polylogue.storage.archive_identity import assert_writable_archive_identity

    # Identity precedes schema checks, pidfiles, HTTP startup, and every other
    # component: a split-root daemon must not become partially observable as a
    # healthy writer before its first ArchiveStore happens to open.
    assert_writable_archive_identity(configured_root=archive_root_path, active_root=archive_root_path)

    if listener_info_path is not None:
        listener_info_path = listener_info_path.expanduser().resolve()
        if listener_info_path.is_relative_to(archive_root_path.resolve()):
            raise click.UsageError("--listener-info-path must be outside the archive root")

    if (
        enable_api
        and enable_browser_capture
        and api_port != 0
        and api_port == browser_capture_port
        and bind_hosts_overlap(api_host, browser_capture_host)
    ):
        raise click.UsageError(
            f"Daemon API {api_host}:{api_port} conflicts with browser-capture "
            f"receiver {browser_capture_host}:{browser_capture_port}. "
            f"Set distinct --api-port/--port values or bind one component to a non-overlapping host."
        )
    if browser_port is not None:
        if not enable_api:
            raise click.UsageError("--browser-port requires the daemon API; remove --no-api")
        if not is_loopback_host(api_host) and api_host not in {"0.0.0.0", "::"}:
            raise click.UsageError(
                "--browser-port requires an API bind reachable on loopback (--api-host 127.0.0.1, ::1, 0.0.0.0, or ::)"
            )
        if browser_port == api_port:
            raise click.UsageError("--browser-port must differ from --api-port")
        if (
            enable_browser_capture
            and browser_port == browser_capture_port
            and bind_hosts_overlap(api_host, browser_capture_host)
        ):
            raise click.UsageError("--browser-port conflicts with the browser-capture receiver")

    # The daemon API must never start in an ambiguous unauthenticated state
    # (polylogue-rzve): an explicit --api-auth-token always wins, otherwise a
    # persisted 0600 token is auto-minted/loaded (mirroring the
    # browser-capture receiver's resolve_receiver_auth_token contract), and
    # --api-allow-no-auth is the loud, explicit opt-out for a deployment that
    # wants the API fully open.
    resolved_api_auth_token = (
        resolve_api_auth_token(api_auth_token, allow_no_auth=api_allow_no_auth) if enable_api else None
    )

    # Non-localhost API binding requires explicit opt-in AND an auth token --
    # allow_no_auth cannot be combined with a remote bind.
    validate_api_bind_policy(
        enabled=enable_api,
        host=api_host,
        allow_remote=browser_capture_allow_remote,
        auth_token=resolved_api_auth_token,
    )
    configure_runtime_components(
        api_enabled=enable_api,
        watcher_enabled=enable_watch,
        watch_sources=sources if enable_watch else (),
        browser_capture_enabled=enable_browser_capture,
    )

    emit("daemon.started", outcome="ok", pid=os.getpid(), root=archive_root_path)

    # polylogue-e98k: log the computed SQLite mmap/cache budget against this
    # process' cgroup memory limits before anything else runs. This is the
    # observability half of the 2026-07-31 memory-throttling incident (see
    # connection_profile.py's module comment) -- a mismatch used to be
    # discoverable only by symptom (slow_write, throttling, a stalled ingest)
    # hours later; now it is a one-line grep of the startup log.
    from polylogue.storage.sqlite.connection_profile import (
        check_mapped_bytes_budget_against_cgroup_limit,
    )

    _emit_mapped_bytes_budget_check(check_mapped_bytes_budget_against_cgroup_limit())

    # One stable archive ownership lock is shared with offline maintenance.
    # The pidfile below remains process metadata only and is never the
    # authority used to exclude a concurrent migration or startup.
    from polylogue.core.write_lease import write_lease
    from polylogue.operations.durable_change_train import (
        DurableChangeTrainError,
        acquire_durable_archive_ownership,
        reconcile_durable_change_trains_on_startup,
    )

    archive_owner = acquire_durable_archive_ownership(archive_root_path, owner_id=f"daemon:{os.getpid()}")
    from polylogue.core.compute import compute_adapter
    from polylogue.core.write_lease import adopt_write_lease
    from polylogue.storage.sqlite.connection_profile import retained_native_settlement_owners_on_current_thread

    write_coordinator = daemon_write_coordinator()
    startup_kernel = compute_adapter()
    startup_bridge = DaemonWriteThreadBridge(write_coordinator, asyncio.get_running_loop())

    def prepare_startup_tiers() -> None:
        startup_kernel.require_current_creator()
        with startup_bridge.hold("daemon.archive_admission.startup") as delegation, adopt_write_lease(delegation):
            # Startup durable-change-train reconciliation opens the durable tiers
            # ``mode=rw`` and migrates them. Exclusive archive ownership already
            # excludes another *process*; the lease is what puts this writer inside
            # the daemon's own single-writer boundary rather than beside it, so it
            # is one authority and not a declared bypass (polylogue-8qm4k).
            with write_lease(
                "daemon.durable_change_train.startup",
                archive_root=archive_root_path,
            ):
                try:
                    recovered_train_paths = reconcile_durable_change_trains_on_startup(archive_root_path)
                except DurableChangeTrainError as exc:
                    # A typed refusal, not a crash: the archive cannot be admitted
                    # by this runtime, so nothing else may start (polylogue-w6nrl).
                    emit(
                        "daemon.startup.refused",
                        level=ERROR,
                        outcome="refused",
                        reason="durable_change_train_admission",
                        error_detail=str(exc),
                    )
                    raise
            if recovered_train_paths:
                emit(
                    "daemon.change_train.reconciled",
                    level=WARNING,
                    outcome="degraded",
                    reason="interrupted_change_trains_recovered_at_startup",
                    files=len(recovered_train_paths),
                    error_detail=", ".join(str(path) for path in recovered_train_paths),
                )
            # Declared durable migrations are ordinary lifecycle: apply them now,
            # under the same exclusive ownership, before anything serves.
            from polylogue.daemon.durable_migrations import apply_declared_durable_migrations
            from polylogue.operations.durable_change_train import initialize_fresh_archive_on_startup

            initialize_fresh_archive_on_startup(
                archive_root_path,
                archive_owner=archive_owner,
                write_lease=lambda actor: write_lease(actor, archive_root=archive_root_path),
            )

            applied_migrations = apply_declared_durable_migrations(
                archive_root_path,
                archive_owner=archive_owner,
                write_lease=lambda actor: write_lease(actor, archive_root=archive_root_path),
            )
            for migration in applied_migrations:
                emit(
                    "daemon.durable_migration.applied",
                    level=WARNING,
                    outcome="ok",
                    reason="declared_durable_migration_applied_at_open",
                    tier=migration.tier.value,
                    error_detail=(
                        f"v{migration.current_version} -> v{migration.target_version}"
                        f"{' behind a verified backup' if migration.requires_backup else ''}"
                    ),
                )
            # A reset of index.db/ops.db staged by the previous daemon lands here:
            # this process holds exclusive ownership and has opened no tier yet,
            # so no connection can be left on an unlinked file. Bootstrap then
            # recreates the deleted tiers and the empty index cold-builds.
            from polylogue.operations.mutation_replay import apply_staged_archive_resets

            if not durable_tier_schema_mismatch():
                with write_lease("daemon.archive_reset.startup", archive_root=archive_root_path):
                    applied_resets = apply_staged_archive_resets(archive_root_path)
                from polylogue.operations.mutation_replay import reconverge_disposable_ops_on_startup
                from polylogue.operations.reset_safety import archive_tiers_closed

                with (
                    write_lease("daemon.ops_reconvergence.startup", archive_root=archive_root_path),
                    archive_tiers_closed(archive_root_path),
                ):
                    ops_reconverged = reconverge_disposable_ops_on_startup(
                        archive_root_path, archive_owner=archive_owner
                    )
                if ops_reconverged:
                    emit(
                        "daemon.ops_reconverged", level=WARNING, outcome="ok", reason="derived_schema_skew", tier="ops"
                    )
                if applied_resets:
                    emit(
                        "daemon.archive_reset.applied",
                        level=WARNING,
                        outcome="ok",
                        reason="staged_reset_applied_before_tiers_open",
                        rows=len(applied_resets),
                        error_detail=", ".join(applied_resets),
                    )

    try:
        await write_coordinator.run_prepared_sync(
            "daemon.archive_admission.startup",
            prepare_startup_tiers,
            submit_worker=lambda worker: (
                startup_kernel.submit(
                    propagate(worker), admission_class="control", estimated_bytes=0, exclusive_bytes=True
                ).future
            ),
            settlement_owners=retained_native_settlement_owners_on_current_thread,
        )
    except BaseException:
        archive_owner.release()
        raise

    def prepare_empty_index_transition() -> str | None:
        from contextlib import contextmanager

        from polylogue.operations.empty_index_startup import (
            EmptyIndexTransitionRefusedError,
            replace_empty_managed_index_on_startup,
        )
        from polylogue.operations.reset_safety import archive_tiers_closed

        startup_kernel.require_current_creator()

        @contextmanager
        def admit_writer(actor: str) -> Iterator[object]:
            with (
                startup_bridge.hold(actor) as delegation,
                adopt_write_lease(delegation),
                write_lease(actor, archive_root=archive_root_path),
            ):
                yield None

        with archive_tiers_closed(archive_root_path):
            try:
                generation_id = replace_empty_managed_index_on_startup(
                    archive_root_path, archive_owner=archive_owner, write_admission=admit_writer
                )
            except EmptyIndexTransitionRefusedError as exc:
                emit("daemon.empty_index.transition", level=WARNING, outcome="refused", reason=exc.reason)
                return None
        if generation_id is not None:
            emit(
                "daemon.empty_index.transition",
                outcome="ok",
                reason="empty_managed_schema_replaced",
                generation_id=generation_id,
            )
        return generation_id

    try:
        empty_generation_id = await write_coordinator.run_prepared_sync(
            "daemon.empty_index.startup",
            prepare_empty_index_transition,
            submit_worker=lambda worker: (
                startup_kernel.submit(
                    propagate(worker), admission_class="control", estimated_bytes=0, exclusive_bytes=True
                ).future
            ),
            settlement_owners=retained_native_settlement_owners_on_current_thread,
        )
    except BaseException:
        archive_owner.release()
        raise

    retained_reconvergence: list[RawObservationReplacement] = []

    def prepare_index_reconvergence() -> None:
        if empty_generation_id is not None:
            return
        from contextlib import contextmanager

        from polylogue.operations.index_reconvergence_startup import (
            IndexReconvergenceRefusedError,
            reconverge_managed_index_on_startup,
        )
        from polylogue.operations.reset_safety import archive_tiers_closed

        startup_kernel.require_current_creator()

        @contextmanager
        def admit_writer(actor: str) -> Iterator[object]:
            with (
                startup_bridge.hold(actor) as delegation,
                adopt_write_lease(delegation),
                write_lease(actor, archive_root=archive_root_path),
            ):
                yield None

        with archive_tiers_closed(archive_root_path):
            try:
                reconverge_managed_index_on_startup(
                    archive_root_path,
                    archive_owner=archive_owner,
                    compute_adapter=startup_kernel,
                    write_admission=admit_writer,
                    retained=retained_reconvergence,
                )
            except IndexReconvergenceRefusedError as exc:
                emit("daemon.index_reconvergence.startup", level=WARNING, outcome="refused", reason=exc.reason)

    try:
        from polylogue.operations.raw_observation_owner import retained_settlement_owners

        await write_coordinator.run_prepared_sync(
            "daemon.index_reconvergence.startup",
            prepare_index_reconvergence,
            submit_worker=lambda worker: (
                startup_kernel.submit(
                    propagate(worker), admission_class="control", estimated_bytes=0, exclusive_bytes=True
                ).future
            ),
            settlement_owners=lambda: (
                *retained_native_settlement_owners_on_current_thread(),
                *retained_settlement_owners(retained_reconvergence),
            ),
        )
    except BaseException:
        archive_owner.release()
        raise

    # Announce only after the daemon has acquired the authoritative archive
    # lease, so a competing owner receives an error without a false startup
    # message.
    if startup_message is not None:
        click.echo(startup_message, err=True)

    # Schema preflight runs FIRST, before any DB-touching startup task. A
    # mismatched runtime/db combination must not even open the DB for FTS or
    # heartbeat queries — that is the IO cost #1003 is meant to avoid.
    try:
        schema_alert = _check_schema_version_fast()
    except BaseException:
        archive_owner.release()
        raise
    # polylogue-gbs02: "watcher_blocked" here means "the derived tier (or
    # worse) is stale, so materialize/index-writing maintenance loops must
    # stay parked" -- unchanged from before, still gates the loops/converger/
    # FTS-readiness block below. A NARROWER, separate check
    # (``durable_mismatch``) decides whether the live watcher itself (raw
    # acquisition) may still start: acquisition only ever writes source.db,
    # so a derived-only mismatch (index.db/embeddings.db) must not stop it.
    schema_blocked = schema_alert.severity == HealthSeverity.CRITICAL
    if listener_info_path is not None and enable_api and schema_blocked:
        archive_owner.release()
        raise click.UsageError("--listener-info-path requires every enabled listener to bind")
    watcher_blocked = enable_watch and schema_blocked
    # Unconditional (not gated on ``enable_watch``): operation recovery below
    # touches audit.db on every startup regardless of whether the watcher is
    # even enabled, so it needs its own answer to "is a durable tier missing
    # or version-mismatched" rather than borrowing the watch-gated one.
    durable_schema_mismatch = durable_tier_schema_mismatch()
    durable_mismatch = enable_watch and durable_schema_mismatch
    watcher_creation_blocked = durable_mismatch
    lifecycle_events_enabled = not watcher_blocked
    if watcher_blocked:
        emit(
            "daemon.schema_preflight.critical",
            level=ERROR,
            outcome="refused",
            reason="durable_tier_mismatch" if durable_mismatch else "derived_tier_mismatch",
            state="watcher_refused" if durable_mismatch else "acquire_only",
            error_detail=schema_alert.message,
        )
        set_degraded(
            DegradedReason(
                code="schema_version_mismatch",
                message=schema_alert.message,
                detail={"check_name": schema_alert.check_name},
                derived_only=not durable_mismatch,
            )
        )
        parked_loop_names = _SCHEMA_BLOCKED_MAINTENANCE_LOOP_NAMES
        if enable_source_catchup:
            parked_loop_names = (*parked_loop_names, _SCHEMA_BLOCKED_OPTIONAL_DRIVE_CATCHUP_LOOP_NAME)
        emit(
            "daemon.maintenance_loops.parked",
            level=ERROR,
            outcome="refused",
            reason="schema_version_mismatch",
            loops=len(parked_loop_names),
        )
        # One event per loop, not one joined string: ``error_detail`` truncates
        # at 300 characters, which silently drops the tail of this list -- and
        # naming exactly what is frozen is the whole point of this alert.
        for parked_loop_name in parked_loop_names:
            emit(
                "daemon.maintenance_loop.parked",
                level=ERROR,
                outcome="refused",
                reason="schema_version_mismatch",
                loop=parked_loop_name,
            )
        try:
            from polylogue.daemon.events import emit_daemon_event

            emit_daemon_event(
                "maintenance_loops_parked",
                archive_root_path=archive_root_path,
                payload={
                    "reason": "schema_version_mismatch",
                    "loop_count": len(parked_loop_names),
                    "loop_names": list(parked_loop_names),
                    "schema_alert_message": schema_alert.message,
                    "derived_only": not durable_mismatch,
                },
            )
        except Exception as exc:
            # daemon_events lives in the disposable ops.db tier, independent
            # of whatever tier tripped the schema mismatch above -- but this
            # is best-effort observability, not load-bearing startup work, so
            # a failure here must never block the (already-decided) degraded
            # startup path.
            emit(
                "daemon.maintenance_loops.park_event_failed",
                level=WARNING,
                outcome="degraded",
                reason="ops_tier_event_write_failed",
                loops=len(parked_loop_names),
                error_type=type(exc).__name__,
                error_detail=str(exc),
            )

    pidfile = archive_root_path / "daemon.pid"
    pidfile_fd: int | None = None

    # Prevent concurrent daemon instances: verify existing pidfile, then
    # acquire an advisory flock.
    try:
        if pidfile.exists() and _verify_pidfile(pidfile):
            old_pid = pidfile.read_text().strip()
            raise RuntimeError(f"Daemon already running (PID {old_pid}). Stop it first or remove {pidfile}")
        # A stale pidfile is overwritten in place after the stable archive
        # lock is held. It is never unlinked as a lock-reclamation strategy.
        pidfile_fd = _acquire_pidfile(pidfile)
    except BaseException:
        archive_owner.release()
        raise
    # Only register the pidfile for atexit cleanup AFTER lock acquisition.
    # Setting _pidfile_path before _acquire_pidfile() means a failed lock
    # attempt by an ephemeral instance would still atexit-unlink the live
    # daemon's pidfile.
    _pidfile_path = pidfile
    from polylogue.daemon.lifecycle import DaemonLifecycle, install_signal_handlers, restore_signal_handlers

    try:
        _daemon_lifecycle = await write_coordinator.run_sync(
            "daemon.lifecycle.start",
            DaemonLifecycle.start,
            run_id=daemon_run_id(),
            archive_root_path=archive_root_path,
            details={"archive_root": str(archive_root_path)},
        )
        # Interrupted effects are classified once under the real daemon writer
        # lease. Executor construction is request-local and must never recover.
        # polylogue-39pdi: a missing/version-mismatched audit.db (or another
        # ingestion-blocking durable tier) makes this touch a schema the
        # running build cannot read, which used to surface as an uncaught
        # sqlite3.Error escaping into the BaseException shutdown path below
        # and killing the daemon instead of leaving it degraded. Route it
        # through the same parked-startup-task decision the schema preflight
        # already makes for other archive work.
        if durable_schema_mismatch:
            emit(
                "daemon.operation_recovery.parked",
                level=ERROR,
                outcome="refused",
                reason="durable_tier_schema_mismatch",
                error_detail=schema_alert.message,
            )
            set_degraded(
                DegradedReason(
                    code="operation_recovery_parked",
                    message=schema_alert.message,
                    detail={"check_name": schema_alert.check_name},
                )
            )
        else:
            from polylogue.core.compute import compute_adapter
            from polylogue.daemon.http import _recover_startup_with_compute

            await _recover_startup_with_compute(
                DaemonWriteThreadBridge(write_coordinator, asyncio.get_running_loop()),
                compute_adapter(),
                archive_root_path,
            )
        previous_signal_handlers = install_signal_handlers(_daemon_lifecycle)
    except BaseException:
        lifecycle = _daemon_lifecycle
        if lifecycle is not None:
            with contextlib.suppress(Exception):
                await write_coordinator.run_sync("daemon.lifecycle.stop", lifecycle.stop, exit_kind="error")
        writer_drained = await _shutdown_writer_coordinator_with_rebuild_exclusion(
            write_coordinator,
            rebuild_exclusion,
            timeout=5.0,
        )
        _release_pidfile_after_writer_drain(pidfile_fd, writer_drained=writer_drained)
        if writer_drained:
            archive_owner.release()
        _daemon_lifecycle = None
        raise

    try:
        # Create the source roots Polylogue owns (hook carriers, the
        # browser-capture spool, the inbox) so health checks don't flag a
        # never-yet-used one as missing. A provider's own directory belongs
        # to that tool: it is never created here, so an uninstalled tool or a
        # relocated directory whose symlink is currently dangling reads as an
        # unavailable source instead of being fabricated or failing startup.
        # Ownership is the source's role, never where its path resolves: a
        # provider directory relocated by a symlink into the archive tree
        # still belongs to its tool, and a dangling one is a retryable gap.
        for src in sources:
            if _is_polylogue_owned_source(src):
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
                    "watch_roots": [str(src.root) for src in sources],
                },
            )
    except BaseException:
        lifecycle = _daemon_lifecycle
        if lifecycle is not None:
            with contextlib.suppress(Exception):
                await write_coordinator.run_sync("daemon.lifecycle.stop", lifecycle.stop, exit_kind="error")
        writer_drained = await _shutdown_writer_coordinator_with_rebuild_exclusion(
            write_coordinator,
            rebuild_exclusion,
            timeout=5.0,
        )
        _release_pidfile_after_writer_drain(pidfile_fd, writer_drained=writer_drained)
        if writer_drained:
            archive_owner.release()
        _daemon_lifecycle = None
        raise

    # Periodic maintenance tasks. If schema preflight blocks the watcher, do
    # not start any background loop that opens the archive: a mismatched
    # runtime/database pair must remain observable without doing catch-up,
    # FTS convergence, status snapshots, WAL checkpointing, or convergence work.
    #
    # The task list is populated only after startup FTS readiness completes.
    # Several maintenance loops can write the archive, especially convergence
    # debt retry; starting them before the first FTS pass self-contends on
    # SQLite during daemon bootstrap.
    # The lifecycle tick is deliberately scheduled before the schema-block
    # guard. It writes only the disposable ops tier and proves that the
    # surviving API/health process is still alive while archive work is
    # intentionally withheld.
    # Fast-tier health checks are read-only (schema_version, disk/WAL space,
    # hook flow, source availability, heartbeat staleness) and stay
    # meaningful precisely when the watcher is schema-blocked -- that is
    # exactly when an operator most needs "archive tier layout is not ready"
    # to keep surfacing rather than going silent (polylogue-7eo7 #4: the
    # daemon used to be blind for the entire blocked duration because this
    # loop only started inside the `if not watcher_blocked:` branch below).
    capabilities: set[ServiceCapability] = set()
    if enable_watch and not watcher_creation_blocked:
        capabilities.add(ServiceCapability.WATCH)
    if enable_source_catchup:
        capabilities.add(ServiceCapability.SOURCE_CATCHUP)
    if enable_browser_capture:
        capabilities.add(ServiceCapability.BROWSER_CAPTURE)
    if enable_api:
        capabilities.add(ServiceCapability.API)
    if browser_port is not None:
        capabilities.add(ServiceCapability.BROWSER_HOST)
    from polylogue.config import load_polylogue_config

    runtime_config = load_polylogue_config()
    validation_mode = ValidationMode.from_string(runtime_config.schema_validation)
    embedding_config = None
    if schema_blocked:
        capabilities.add(ServiceCapability.SCHEMA_BLOCKED)
    else:
        capabilities.add(ServiceCapability.DERIVED_WRITES)
        from polylogue.daemon.embedding_backlog import embedding_convergence_unavailable_reason

        embedding_config = runtime_config
        if embedding_convergence_unavailable_reason(embedding_config) is None:
            capabilities.add(ServiceCapability.EMBEDDINGS)

    halts = HaltRegistry(archive_root_path)
    supervisor = DaemonSupervisor(
        profile=service_profile,
        capabilities=capabilities,
        halts=halts,
        frame=f"daemon:{os.getpid()}",
        on_degraded=_degrade_for_failed_service,
    )
    _set_active_supervisor(supervisor)
    # A partially initialized archive can have an index but no durable raw
    # tier. Fresh archives are different: their first acquisition creates the
    # tier, so only an existing index makes this a failed prerequisite.
    if (archive_root_path / "index.db").exists() and not (archive_root_path / "source.db").exists():
        if service_profile is ServiceProfile.PRODUCTION:
            supervisor.mark_unavailable("raw_observation_convergence", reason="source.db is absent")
        if supervisor.is_schedulable("fair_intake"):
            supervisor.mark_unavailable("fair_intake", reason="source.db is absent")
    for halted in supervisor.halted_records():
        emit(
            "daemon.service.halted",
            level=ERROR,
            outcome="refused",
            component=halted.unit,
            reason=halted.reason.value,
            error_detail=halted.message,
        )

    supervisor.start("lifecycle_heartbeat", _periodic_lifecycle_heartbeat)
    supervisor.start("termination_reconciliation", _reconcile_ended_daemon_runs)
    from polylogue.daemon.notifications import ConfiguredNotificationBackend

    notifications = ConfiguredNotificationBackend()
    supervisor.start(
        "health_check",
        lambda: _periodic_health_check(backend=notifications, sources=sources if enable_watch else ()),
    )
    supervisor.start("schema_preflight_recheck", _periodic_schema_preflight_recheck)

    api_server: DaemonAPIHTTPServer | None = None
    api_server_task: asyncio.Task[None] | None = None
    # The ingest owner when no API server carries one (``--no-api``).
    ingest_owner_runtime: DaemonOperationRuntime | None = None
    owner_session_profiles: ComposedSessionProfiles | None = None
    uds_server: Any | None = None
    uds_server_task: asyncio.Task[None] | None = None
    server: BrowserCaptureHTTPServer | None = None
    server_task: asyncio.Task[None] | None = None
    watcher: LiveWatcher | None = None
    converger: DaemonConverger | None = None
    session_profile_callback: SessionProfileCallback | None = None
    embedding_callback: EmbeddingConvergenceOwner | None = None
    raw_observation_owner: RawObservationConvergenceOwner | None = None
    watcher_registered_gate_event: asyncio.Event | None = None
    raw_intake_wakeup = asyncio.Event()
    cleanup_task: asyncio.Task[object] | None = None
    cleanup_cancel_requests = 0
    termination: BaseException | None = None
    writer_drained = False
    ownership_retained_reason: str | None = "shutdown_not_reached"
    cold_build: ColdBuildGeneration | None = None
    try:
        if enable_browser_capture:
            from polylogue.paths import browser_capture_receiver_token_path, browser_capture_spool_root

            resolved_browser_capture_auth_token = resolve_receiver_auth_token(
                browser_capture_auth_token, allow_no_auth=browser_capture_allow_no_auth
            )
            server = make_server(
                browser_capture_host,
                browser_capture_port,
                allow_remote=browser_capture_allow_remote,
                auth_token=resolved_browser_capture_auth_token,
                auth_token_path=(
                    browser_capture_receiver_token_path() if resolved_browser_capture_auth_token is not None else None
                ),
                extra_origins=browser_capture_extra_origins,
                archive_root=archive_root_path,
                api_auth_token=api_auth_token,
                api_allow_no_auth=api_allow_no_auth,
            )
            server_task = supervisor.start(
                "browser_capture_server",
                lambda: _serve_until_complete(server, label="browser-capture"),
            )
            configure_browser_capture_status(server.config if server_task is not None else None)
            if lifecycle_events_enabled and server_task is not None:
                await _emit_daemon_lifecycle_event(
                    "component_started",
                    archive_root_path=archive_root_path,
                    component="browser_capture",
                    payload={
                        "host": browser_capture_host,
                        "port": int(server.server_address[1]) if browser_capture_port == 0 else browser_capture_port,
                        "spool_path": str(browser_capture_spool_root()),
                        "auth_enabled": resolved_browser_capture_auth_token is not None,
                    },
                )

        # Embedding reads are exposed by the API, so recover the active
        # generation before publishing either HTTP socket.  Otherwise an
        # immediate similarity request can recreate legacy WAL sidecars after
        # the lifecycle checkpoint and turn a clean restart into a failure.
        # Filled once the watcher exists; maintenance loops consult it for
        # catch-up activity without holding the watcher before creation.
        watcher_holder: list[LiveWatcher] = []
        archive_work_scheduled = any(
            supervisor.is_schedulable(name)
            for name in (
                "fair_intake",
                "watcher",
                "convergence_check",
                "session_profile_audit",
                "raw_observation_convergence",
            )
        )
        if not schema_blocked and (archive_work_scheduled or enable_api):
            await _run_startup_embedding_lifecycle(write_coordinator, archive_root_path)
            if lifecycle_events_enabled:
                await _emit_daemon_lifecycle_event(
                    "component_ready",
                    archive_root_path=archive_root_path,
                    component="embedding_lifecycle_startup",
                )

        if enable_api:
            if not durable_schema_mismatch:
                from polylogue.operations.operation_context import prepare_operation_journals

                await write_coordinator.run_sync(
                    "daemon.operation_journals.startup", prepare_operation_journals, archive_root_path
                )
            from polylogue.core.compute import compute_adapter
            from polylogue.daemon.http import (
                DaemonAPIHandler,
                DaemonAPIHTTPServer,
            )

            api_server = DaemonAPIHTTPServer(
                (api_host, api_port),
                DaemonAPIHandler,
                auth_token=resolved_api_auth_token,
                api_host=api_host,
                write_bridge=DaemonWriteThreadBridge(write_coordinator, asyncio.get_running_loop()),
                archive_root=archive_root_path,
                watch_sources=sources if enable_watch else (),
                execution_kernel=startup_kernel,
            )
            # Daemon-internal lease-free work shares the capacity the API
            # server already owns rather than standing up a second pool
            # (polylogue-c0l7n).
            publish_compute_adapter(api_server.execution_kernel)
            # The re-drive's claim phase runs on this loop; the listeners
            # serve only after it claimed every interrupted accepted ingest.
            await api_server.operation_runtime.accepted_ingest_redrive_claimed()
            api_server_task = supervisor.start(
                "api_server",
                lambda: _serve_until_complete(api_server, label="api"),
            )
            from polylogue.daemon.uds import DaemonAPIUnixHTTPServer, daemon_socket_path

            uds_server = DaemonAPIUnixHTTPServer(
                daemon_socket_path(archive_root_path),
                archive_root=archive_root_path,
                auth_token=resolved_api_auth_token,
                write_bridge=DaemonWriteThreadBridge(write_coordinator, asyncio.get_running_loop()),
                execution_kernel=api_server.execution_kernel,
                operation_runtime=api_server.operation_runtime,
            )
            uds_server_task = supervisor.start(
                "uds_server",
                lambda: _serve_until_complete(uds_server, label="uds"),
            )
            if browser_port is not None:
                # The browser child needs the actual bound port (including
                # port 0), while ordinary API tests and alternate server
                # implementations need no server_address introspection.
                bound_api_port = int(api_server.server_address[1])
                upstream_host = "127.0.0.1" if api_host == "0.0.0.0" else "::1" if api_host == "::" else api_host
                if ":" in upstream_host:
                    upstream_host = f"[{upstream_host}]"
                supervisor.start(
                    "browser_host",
                    lambda: _run_browser_host(
                        host=api_host,
                        port=browser_port,
                        daemon_origin=f"http://{upstream_host}:{bound_api_port}",
                    ),
                )
            if lifecycle_events_enabled:
                await _emit_daemon_lifecycle_event(
                    "component_started",
                    archive_root_path=archive_root_path,
                    component="api",
                    payload={
                        "host": api_host,
                        "port": int(api_server.server_address[1]) if api_port == 0 else api_port,
                        "auth_enabled": resolved_api_auth_token is not None,
                    },
                )

        if listener_info_path is not None:
            if (enable_api and api_server is None) or (enable_browser_capture and server is None):
                raise click.UsageError("--listener-info-path requires every enabled listener to bind")
            # Publish only after every enabled TCP listener owns its socket.
            # This is bind readback, not archive or service readiness.
            listeners = {
                "api": {"host": str(api_server.server_address[0]), "port": int(api_server.server_address[1])}
                if api_server is not None
                else None,
                "browser_capture": {"host": str(server.server_address[0]), "port": int(server.server_address[1])}
                if server is not None
                else None,
            }
            atomic_create(
                listener_info_path,
                (dumps({"pid": os.getpid(), "listeners": listeners}) + "\n").encode(),
                mode=0o600,
            )

        if api_server is None and not schema_blocked:
            # The ingest owner does not depend on the HTTP surface: accepted
            # ingests that startup recovery left for it are re-driven by a
            # watcher-only daemon too.
            ingest_owner_runtime, owner_session_profiles = compose_ingest_owner(
                archive_root_path,
                DaemonWriteThreadBridge(write_coordinator, asyncio.get_running_loop()),
                compute_adapter=startup_kernel,
                validation_mode=validation_mode,
            )
            ingest_owner_runtime.start_accepted_ingest_redrive()
            await ingest_owner_runtime.accepted_ingest_redrive_claimed()

        # Ensure FTS structure after HTTP surfaces are bound and before live
        # catch-up starts. Startup FTS maintenance and catch-up ingestion are
        # both write-heavy; running them concurrently makes SQLite maintenance
        # time out behind the daemon's own writer.
        if not schema_blocked and archive_work_scheduled:
            from polylogue.daemon.blob_gc_periodic import (
                periodic_blob_gc_check,
                periodic_blob_publication_reconciliation_check,
            )
            from polylogue.daemon.convergence import DaemonConverger
            from polylogue.daemon.convergence_stages import make_default_convergence_stages
            from polylogue.daemon.embedding_backlog import (
                periodic_embedding_backlog_check,
                periodic_embedding_orphan_reconcile_check,
            )
            from polylogue.daemon.judgment_automation import periodic_judgment_automation_sweep
            from polylogue.daemon.secret_scan_sweep import periodic_secret_scan_sweep
            from polylogue.daemon.session_profile_composition import (
                ComposedSessionProfiles,
                compose_session_profile_callback,
            )

            if api_server is not None:
                daemon_compute = api_server.execution_kernel
                session_profile_callback = api_server.session_profile_callback
            else:
                from polylogue.core.compute import compute_adapter

                daemon_compute = startup_kernel
                session_profile_callback = owner_session_profiles or compose_session_profile_callback(
                    archive_root_path,
                    compute_adapter=daemon_compute,
                    write_bridge=DaemonWriteThreadBridge(write_coordinator, asyncio.get_running_loop()),
                    now=time.time,
                )
            from polylogue.daemon.embedding_owner import ComposedEmbeddingConvergence, compose_embedding_convergence
            from polylogue.operations.embedding_derivation import embedding_session_ids_for_paths

            embedding_convergence: ComposedEmbeddingConvergence = (
                cast("ComposedEmbeddingConvergence", api_server.operation_runtime.embedding_convergence)
                if api_server is not None and api_server.operation_runtime.embedding_convergence is not None
                else compose_embedding_convergence(
                    archive_root_path / "index.db",
                    compute_adapter=daemon_compute,
                    write_bridge=DaemonWriteThreadBridge(write_coordinator, asyncio.get_running_loop()),
                    config=embedding_config,
                )
            )
            # Every operation runtime this daemon composes shares its one
            # embedding owner: the API's, or the ingest owner under --no-api.
            for operation_runtime in (
                api_server.operation_runtime if api_server is not None else None,
                ingest_owner_runtime,
            ):
                if operation_runtime is not None:
                    operation_runtime.embedding_convergence = embedding_convergence

            async def converge_ingest_embeddings(index_db: Path, paths: Sequence[Path]) -> bool:
                ids = embedding_session_ids_for_paths(index_db, archive_root=archive_root_path, paths=paths)
                if not ids:
                    return True
                return (await embedding_convergence(ids)).converged

            embedding_callback = converge_ingest_embeddings
            resident_runtime = api_server.operation_runtime if api_server is not None else ingest_owner_runtime
            if resident_runtime is None:
                raise RuntimeError("live ingestion requires its supplied resident operation owner")
            raw_observation_owner = resident_runtime.raw_observation_owner
            from polylogue.daemon.intake_adapters import RawMaterializationDiscovery

            raw_intake_discovery = RawMaterializationDiscovery(
                archive_root_path,
            )

            from polylogue.daemon.convergence import DerivationConvergenceOwner
            from polylogue.daemon.fts_convergence import FtsConvergenceOwner
            from polylogue.operations.fts_derivation import make_fts_derivation, make_fts_frame
            from polylogue.operations.sinex_convergence import configured_derivation_barrier

            fts_index = archive_root_path / "index.db"
            fts_owner = FtsConvergenceOwner(
                DerivationConvergenceOwner(
                    DaemonConverger(
                        (),
                        derivations=(make_fts_derivation(fts_index, archive_root=archive_root_path),),
                        derivation_barrier=configured_derivation_barrier(archive_root_path),
                    ),
                    compute_adapter=daemon_compute,
                    write_bridge=DaemonWriteThreadBridge(write_coordinator, asyncio.get_running_loop()),
                ),
                lambda scope: make_fts_frame(fts_index, archive_root=archive_root_path, scope=scope),
            )
            fts_startup = await fts_owner.converge()
            if lifecycle_events_enabled:
                await _emit_daemon_lifecycle_event(
                    "component_degraded" if fts_startup.failed else "component_started",
                    archive_root_path=archive_root_path,
                    component="fts",
                )
            lineage_census = await _run_startup_lineage_census()
            if lifecycle_events_enabled:
                await _emit_daemon_lifecycle_event(
                    _lineage_startup_lifecycle_phase(lineage_census),
                    archive_root_path=archive_root_path,
                    component="lineage_startup",
                )
            await _reconcile_blob_publications()
            # Disable per-write FTS5 automerge so each small ingest batch does
            # not trigger a merge of the full (hundreds-of-MB) existing
            # segments (#1851).  A periodic merge pass amortises the cost.
            await _configure_fts_automerge()
            if not enable_source_catchup:
                emit(
                    "daemon.drive_catchup.disabled",
                    level=DEBUG,
                    outcome="skipped",
                    reason="disabled_for_this_run",
                    loop="drive source catch-up",
                )
            watcher_registered_gate_event = asyncio.Event() if enable_watch else None
            gate = watcher_registered_gate_event
            # One real producer/consumer pair on the in-process bus
            # (polylogue-14t7): the post-commit write effect announces a
            # committed ingest, and the embedding backlog loop wakes on it
            # instead of sleeping out its whole poll interval. The publish
            # happens on the writer's worker thread, so the hand-off to the
            # loop must go through the event loop rather than touch the
            # asyncio.Event directly.
            ingest_wakeup = asyncio.Event()
            _loop = asyncio.get_running_loop()

            def _wake_on_ingest(_event: IngestCommitted) -> None:
                _loop.call_soon_threadsafe(ingest_wakeup.set)

            daemon_event_bus().subscribe(IngestCommitted, _wake_on_ingest)

            async def retry_raw_retention() -> None:
                if watcher_holder:
                    await watcher_holder[0].retry_raw_retention_backlog()

            periodic_services: tuple[tuple[str, Callable[[], Coroutine[Any, Any, None]]], ...] = (
                (
                    "convergence_check",
                    lambda: _periodic_convergence_check(
                        sources,
                        fts_owner=fts_owner,
                        watcher_registered=gate,
                        raw_retention_callback=retry_raw_retention,
                    ),
                ),
                (
                    "session_profile_audit",
                    functools.partial(
                        _periodic_session_profile_audit, session_profile_callback, watcher_registered=gate
                    ),
                ),
                (
                    "raw_observation_convergence",
                    lambda: _periodic_raw_materialization_convergence(
                        watcher_registered=gate,
                        raw_intake_wakeup=raw_intake_wakeup,
                    ),
                ),
                ("wal_checkpoint", _periodic_wal_checkpoint),
                ("fts_merge", _periodic_fts_merge),
                ("heartbeat", lambda: _periodic_heartbeat(sources=sources)),
                (
                    "embedding_backlog",
                    lambda: periodic_embedding_backlog_check(
                        watcher_registered=gate,
                        converge=embedding_convergence.callback,
                        wakeup=ingest_wakeup,
                    ),
                ),
                (
                    "embedding_orphan_reconcile",
                    lambda: periodic_embedding_orphan_reconcile_check(watcher_registered=gate),
                ),
                ("db_optimize", _periodic_db_optimize),
                ("status_snapshot_refresh", _periodic_status_snapshot_refresh),
                (
                    "judgment_automation",
                    lambda: periodic_judgment_automation_sweep(
                        watcher_registered=gate,
                        archive_root_path=archive_root_path,
                    ),
                ),
                ("blob_gc", lambda: periodic_blob_gc_check(watcher_registered=gate)),
                (
                    "blob_publication_reconciliation",
                    lambda: periodic_blob_publication_reconciliation_check(watcher_registered=gate),
                ),
                ("secret_scan_sweep", lambda: periodic_secret_scan_sweep(watcher_registered=gate)),
            )
            for service_name, service_factory in periodic_services:
                if supervisor.state(service_name) is ServiceState.UNAVAILABLE:
                    continue
                supervisor.start(service_name, service_factory)
            _db = _active_index_db_path()
            converger = DaemonConverger(stages=make_default_convergence_stages(_db, compute_adapter=daemon_compute))
            if lifecycle_events_enabled:
                await _emit_daemon_lifecycle_event(
                    "component_started",
                    archive_root_path=archive_root_path,
                    component="converger",
                )

        # Preflight already ran at the top of run_daemon_services (see
        # ``watcher_creation_blocked``/``watcher_blocked`` above); reuse that
        # result. The watcher itself gates only on ``watcher_creation_blocked``
        # (durable-tier mismatch) -- a derived-only mismatch leaves
        # ``converger``/``watcher_registered_gate_event`` at their None defaults
        # (the ``if not watcher_blocked:`` block above was skipped), so the
        # watcher runs acquire-only: raw acquisition proceeds, no
        # convergence coupling (polylogue-gbs02).
        # Ask the supervisor what it will actually run rather than building
        # the intake stack and discarding it. A profile that declares no
        # intake service -- or one whose intake is durably halted -- must not
        # open an archive generation, register adapters or begin a cold build
        # on their behalf: constructing work nothing will schedule is how a
        # focused daemon test ends up doing real materialization.
        intake_scheduled = supervisor.is_schedulable("fair_intake") or supervisor.is_schedulable("watcher")
        try:
            if not watcher_creation_blocked and intake_scheduled:
                async with Polylogue() as polylogue:
                    from polylogue.core.compute import compute_adapter
                    from polylogue.daemon.drive_catchup import DriveCatchupExecution
                    from polylogue.daemon.intake_adapters import (
                        ColdBuildGeneration,
                        ColdBuildSettlement,
                        DaemonIntakeContext,
                        DaemonIntakeService,
                        active_index_generation_is_empty,
                        build_intake_adapters,
                        classify_cold_build_settlement_failure,
                        clear_cold_build_generation,
                        promote_cold_build_covering_active_index,
                        register_cold_build_generation,
                    )

                    promotion_compute = api_server.execution_kernel if api_server is not None else compute_adapter()

                    cold_build_promotion_execution = DriveCatchupExecution(
                        write_coordinator,
                        compute_adapter=promotion_compute,
                    )

                    from polylogue.operations.canonical_archive_ingest import live_sqlite_capture_stage

                    watcher = LiveWatcher(
                        polylogue,
                        sources,
                        converger=converger,
                        event_emitter=functools.partial(_emit_live_batch_event, archive_root_path=archive_root_path),
                        write_coordinator=write_coordinator,
                        sqlite_capture_stage=live_sqlite_capture_stage(promotion_compute),
                        embedding_owner=embedding_callback,
                        session_profile_callback=session_profile_callback,
                        append_runner=raw_observation_owner.ingest_append_plans
                        if raw_observation_owner is not None
                        else None,
                        convergence_runner=(
                            raw_observation_owner.run_convergence_sync if raw_observation_owner is not None else None
                        ),
                        retained_runner=(
                            raw_observation_owner.ingest_retained_raw_ids if raw_observation_owner is not None else None
                        ),
                        intake_wakeup=raw_intake_wakeup,
                    )
                    watcher_holder.append(watcher)

                    async def run_intake_write(
                        actor: str,
                        function: Callable[..., Any],
                        /,
                        *args: Any,
                        **kwargs: Any,
                    ) -> Any:
                        return await write_coordinator.run_sync(actor, function, *args, **kwargs)

                    async def run_remote_intake() -> DriveCatchupReport:
                        if session_profile_callback is None:
                            # Derived schema skew blocks profile publication,
                            # but source acquisition remains durable and safe.
                            return await _run_drive_source_catchup_safely(
                                None, raw_owner=raw_observation_owner, compute_owner=daemon_compute
                            )
                        return await _run_drive_source_catchup_safely(
                            session_profile_callback, raw_owner=raw_observation_owner, compute_owner=daemon_compute
                        )

                    async def discover_raw_intake(limit: int) -> tuple[tuple[str, int], ...]:
                        submitted = daemon_compute.submit(
                            # The compute kernel's thread pool predates every
                            # bind, so contextvars do not reach its workers.
                            cast(
                                "Callable[[], tuple[tuple[str, int], ...]]",
                                propagate(functools.partial(raw_intake_discovery.discover_pending_raw_ids, limit)),
                            ),
                            admission_class="incremental-background",
                        )
                        return await submitted.wait()

                    async def admit_raw_intake(raw_id: str) -> AdmissionResult:
                        if raw_observation_owner is None:
                            raise RuntimeError("raw intake requires the supplied observation owner")
                        report = await raw_observation_owner.converge_raw_id(raw_id)
                        result = _derivation_admission(report, raw_id, subject="raw observation")
                        if result.outcome is AdmissionOutcome.ADMITTED:
                            await _converge_raw_materialized_session_profiles(
                                archive_root_path,
                                raw_id,
                                session_profile_callback,
                            )
                        return result

                    async def discover_hook_events(limit: int) -> Sequence[tuple[str, int]]:
                        from polylogue.operations.hook_event_derivation import discover_pending_hook_carriers

                        submitted = daemon_compute.submit(
                            propagate(functools.partial(discover_pending_hook_carriers, archive_root_path, limit)),
                            admission_class="incremental-background",
                        )
                        return await submitted.wait()

                    async def admit_hook_events(raw_id: str) -> AdmissionResult:
                        """Materialize exactly one acquired carrier's events.

                        Hook carrier discovery and derivation stay on the
                        compute pool; its short publication uses the daemon's
                        ordinary writer bridge. The kernel's verdicts decide
                        the outcome: a refusal that a
                        later pass could resolve is retryable, a carrier that
                        is already materialized is a duplicate.
                        """

                        from polylogue.core.stage_admission import stage_write_admission
                        from polylogue.daemon.convergence import _DerivationAdmission
                        from polylogue.operations.hook_event_derivation import converge_hook_carriers

                        loop = asyncio.get_running_loop()
                        admission = _DerivationAdmission(
                            DaemonWriteThreadBridge(write_coordinator, loop),
                            loop_thread_id=threading.get_ident(),
                        )
                        with stage_write_admission(admission.stage_write):
                            submitted = daemon_compute.submit(
                                propagate(
                                    functools.partial(
                                        converge_hook_carriers, archive_root_path, raw_ids=(raw_id,), limit=1
                                    )
                                ),
                                admission_class="incremental-background",
                            )
                        report = await submitted.wait()
                        return _derivation_admission(report, raw_id, subject="hook event")

                    drive_sources_configured = _drive_sources_configured()
                    # polylogue-f7pdm: availability is re-evaluated per
                    # discovery pass, not latched at startup. On a fresh root
                    # ``source.db`` does not exist yet, and a one-shot
                    # existence check left the raw class unregistered for the
                    # whole daemon lifetime. ``RawMaterializationDiscovery``
                    # returns an empty page while the tier is absent, so
                    # registering here costs nothing and the class starts
                    # admitting as soon as the first acquisition commits.
                    raw_materialization_available = not schema_blocked
                    from polylogue.daemon.intake_adapters import SubUnitHaltPolicy

                    # A configured source that terminally refuses halts
                    # *itself*, not the whole ``configured_local`` class it
                    # shares with its siblings. The registry is the same
                    # durable one the supervisor and the dispatcher read, so
                    # status names the source and the next process still
                    # refuses to plan it (polylogue-kqrbw).
                    def _source_is_halted(name: str) -> bool:
                        return halts.is_halted(unit_id(UnitKind.SOURCE, name))

                    def _halt_source(name: str, message: str) -> None:
                        halts.halt(
                            unit_id(UnitKind.SOURCE, name),
                            reason=HaltReason.TERMINAL_REFUSAL,
                            message=message,
                            frame=f"daemon:{os.getpid()}",
                        )

                    adapter_pairs = build_intake_adapters(
                        DaemonIntakeContext(
                            archive_root=archive_root_path,
                            watcher=watcher,
                            sources=sources if enable_watch else (),
                            write_runner=run_intake_write,
                        ),
                        source_halts=SubUnitHaltPolicy(
                            is_halted=_source_is_halted,
                            halt=_halt_source,
                        ),
                        remote_callback=run_remote_intake
                        if not schema_blocked and enable_source_catchup and drive_sources_configured
                        else None,
                        raw_callback=admit_raw_intake if raw_materialization_available else None,
                        raw_discover=discover_raw_intake if raw_materialization_available else None,
                        raw_discovery_pending=lambda: raw_intake_discovery.discovery_pending,
                        raw_suspended=lambda: cold_build is not None and not cold_build.settled,
                        hook_events_callback=admit_hook_events if raw_materialization_available else None,
                        hook_events_discover=discover_hook_events if raw_materialization_available else None,
                    )
                    dispatcher = FairIntakeDispatcher(
                        tuple(IntakeClassSpec(name=name, adapter=adapter) for name, adapter in adapter_pairs),
                        halts=halts,
                        board=supervisor.board,
                        frame=f"daemon:{os.getpid()}",
                    )
                    # polylogue-b7dkb: a fresh root builds its index as an
                    # owned inactive generation. The pass that fills it is
                    # this same dispatcher route -- nothing here changes what
                    # ingest does, only which index.db its rows land in --
                    # and readers keep resolving the previous active
                    # generation until the readiness pass promotes this one.
                    # Generation bootstrap opens a writable index directly;
                    # keep its probe inside the same coordinator as every
                    # other daemon archive writer.
                    await write_coordinator.run_sync(
                        "daemon.cold_build.reconcile_interrupted",
                        ColdBuildGeneration.reconcile_interrupted_promotions,
                        archive_root_path,
                    )
                    if schema_blocked:
                        # The probe is a writable open, which admits the
                        # active Index's derived identity first: a stale one
                        # raises SchemaSkew. Raw materialization is parked in
                        # this mode, so a cold build could not fill anyway;
                        # the daemon stays up and status reports the skew
                        # with its rebuild route.
                        emit(
                            "daemon.cold_build.withheld",
                            level=WARNING,
                            outcome="refused",
                            reason="schema_blocked",
                            requested=cold_build_index,
                        )
                        cold_build_requested = False
                    else:
                        active_generation_empty = await write_coordinator.run_sync(
                            "daemon.cold_build.probe",
                            active_index_generation_is_empty,
                            archive_root_path,
                        )
                        cold_build_requested = cold_build_index or active_generation_empty
                    if cold_build_requested:
                        from polylogue.daemon.discovery_progress import run_cold_build_preparation

                        # Baseline enumeration, classification and hashing
                        # read sources off the writer; only binding the
                        # observation into a generation is a writer call.
                        # Status reports the phase and counts meanwhile.
                        cold_build = await run_cold_build_preparation(
                            write_coordinator,
                            "daemon.cold_build.begin",
                            partial(ColdBuildGeneration.observe_source_baseline, sources),
                            ColdBuildGeneration.begin,
                            archive_root_path,
                            reason="explicit_cold_build" if cold_build_index else "empty_active_index_generation",
                        )
                        register_cold_build_generation(cold_build)
                        from polylogue.daemon.catchup_status import set_cold_build_progress_provider

                        set_cold_build_progress_provider(lambda: cold_build.accepted_progress)
                        from polylogue.daemon.catchup_status import set_cold_build_settlement_provider

                        set_cold_build_settlement_provider(
                            lambda: (
                                cold_build.generation_id,
                                cold_build.settlement_state,
                                cold_build.settlement_reason,
                                cold_build.settlement_last_error,
                                cold_build.settlement_attempts,
                                cold_build.settlement_next_retry_at,
                            )
                        )

                    async def settle_cold_build() -> ColdBuildSettlement:
                        """Promote or discard the candidate once intake drains."""
                        generation = cold_build
                        if generation is None or generation.publication_complete or generation.discarded:
                            return ColdBuildSettlement("complete")
                        promoted = False
                        try:
                            if generation.promoted:
                                await write_coordinator.run_sync(
                                    "daemon.cold_build.reconcile_promoted", generation.reconcile_promoted
                                )
                                promoted = True
                            else:
                                observed_baseline = await _observe_faulted_baseline_cancellable(generation, sources)
                                if observed_baseline is not None:
                                    await write_coordinator.run_sync(
                                        "daemon.cold_build.refresh_faulted_baseline",
                                        generation.refresh_faulted_baseline,
                                        observed_baseline,
                                    )
                                session_count = await write_coordinator.run_sync(
                                    "daemon.cold_build.session_count",
                                    generation.session_count,
                                )
                                if session_count > 0:
                                    # A candidate missing a session the active
                                    # generation serves from a retained raw is
                                    # refused, not promoted (polylogue-5hcbg).
                                    await write_coordinator.run_sync(
                                        "daemon.cold_build.prepare_candidate",
                                        generation.prepare_promotion_candidate,
                                    )
                                    await cold_build_promotion_execution.publish_prepared_sync(
                                        "daemon.cold_build.promote",
                                        generation.prepare_promotion_proof,
                                        functools.partial(
                                            promote_cold_build_covering_active_index,
                                            generation,
                                        ),
                                    )
                                    promoted = True
                                else:
                                    # An empty candidate cannot replace an active index.
                                    await write_coordinator.run_sync(
                                        "daemon.cold_build.discard",
                                        generation.discard,
                                    )
                        except Exception as exc:
                            classification = classify_cold_build_settlement_failure(exc)
                            if classification is None:
                                raise
                            reason, retryable = classification
                            previous = (generation.settlement_state, generation.settlement_reason)
                            state: Literal["retryable", "blocked"] = "retryable" if retryable else "blocked"
                            retry_at = (
                                time.monotonic() + (60.0 if reason == "source_integrity" else 5.0)
                                if retryable
                                else None
                            )
                            generation.record_settlement(
                                state,
                                reason=reason,
                                last_error=f"{type(exc).__name__}: {exc}"[:256],
                                next_retry_at=retry_at,
                            )
                            if previous != (state, reason):
                                emit(
                                    "daemon.cold_build.settlement",
                                    level=WARNING,
                                    outcome="degraded",
                                    state=state,
                                    reason=reason,
                                    generation_id=generation.generation_id,
                                    error_type=type(exc).__name__,
                                )
                            return ColdBuildSettlement(
                                state, reason=reason, attempts=generation.settlement_attempts, next_retry_at=retry_at
                            )
                        else:
                            previous = (generation.settlement_state, generation.settlement_reason)
                            generation.record_settlement("complete")
                            clear_cold_build_generation()
                            set_cold_build_progress_provider(None)
                            if previous != ("complete", None):
                                emit(
                                    "daemon.cold_build.settlement",
                                    outcome="ok",
                                    state="complete",
                                    reason="promoted" if promoted else "discarded",
                                    generation_id=generation.generation_id,
                                )
                        if watcher is not None:
                            # Work deferred behind the candidate (stages the
                            # build could not run, retention it could not
                            # apply) accrued retry backoff while it waited,
                            # though nothing failed. Whether the candidate was
                            # promoted or discarded, the deferral has ended.
                            try:
                                released = await write_coordinator.run_sync(
                                    "daemon.cold_build.release_deferred_debt",
                                    watcher._cursor.release_deferred_convergence_debt,
                                )
                            except Exception as exc:
                                emit(
                                    "daemon.cold_build.release_deferred_debt_failed",
                                    level=WARNING,
                                    outcome="degraded",
                                    error_type=type(exc).__name__,
                                    error_detail=str(exc),
                                )
                            else:
                                emit("daemon.cold_build.deferred_debt_released", outcome="ok", rows=released)
                        if promoted and isinstance(session_profile_callback, ComposedSessionProfiles):
                            try:
                                await session_profile_callback.converge_promoted()
                            except Exception as exc:
                                # The periodic owner retries from the promoted
                                # output relations; intake publication is done.
                                emit(
                                    "daemon.cold_build.profile_convergence_failed",
                                    level=WARNING,
                                    outcome="degraded",
                                    error_type=type(exc).__name__,
                                    error_detail=str(exc),
                                )
                        return ColdBuildSettlement("complete", attempts=generation.settlement_attempts)

                    async def refresh_cold_build_progress(_result: object) -> None:
                        generation = cold_build
                        if generation is None or generation.settled:
                            return
                        if generation.settlement_state == "building" and any(
                            row.disposition == "fault" for row in generation.source_baseline.decisions
                        ):
                            generation.record_settlement(
                                "blocked",
                                reason="source_integrity",
                                last_error="production source baseline has unresolved discovery faults",
                                attempted=False,
                            )
                            emit(
                                "daemon.cold_build.settlement",
                                level=WARNING,
                                outcome="degraded",
                                state="blocked",
                                reason="source_integrity",
                                generation_id=generation.generation_id,
                                error_type="ProductionBaselineError",
                            )
                        try:
                            await write_coordinator.run_sync(
                                "daemon.cold_build.accepted_progress",
                                generation.refresh_accepted_progress,
                            )
                        except Exception as exc:
                            # This projection feeds status only. Intake and
                            # readiness must continue even when ETA cannot be
                            # measured from the candidate.
                            generation.invalidate_accepted_progress()
                            emit(
                                "daemon.cold_build.accepted_progress_failed",
                                level=WARNING,
                                outcome="unmeasured",
                                reason="candidate_progress_unavailable",
                                error_type=type(exc).__name__,
                                error_detail=str(exc),
                            )

                    intake_service = DaemonIntakeService(
                        dispatcher,
                        wakeup=raw_intake_wakeup,
                        on_backlog_drained=settle_cold_build if cold_build is not None else None,
                        settlement_revision=lambda: (
                            tuple(watcher.intake_revision(source) for source in sources)
                            + (cold_build.settlement_evidence_revision(sources) if cold_build is not None else ())
                        ),
                        settlement_external_revision=lambda: (
                            tuple(watcher.intake_revision(source) for source in sources)
                            + (cold_build.settlement_external_revision(sources) if cold_build is not None else ())
                        ),
                        # A scheduled retry can be absent from a bounded
                        # discovery pass until its deadline. The candidate
                        # cannot become active while that obligation remains.
                        has_pending_backlog=(
                            lambda: watcher._cursor.has_pending_retries(source.root for source in sources) is not False
                        )
                        if cold_build is not None
                        else None,
                        on_pass_complete=refresh_cold_build_progress if cold_build is not None else None,
                    )
                    supervisor.start("fair_intake", intake_service.run)
                    if enable_watch and not watcher.prepare_watch_roots():
                        # A watcher with nothing to watch is unavailable, not
                        # a watch that completed. Fair intake's idle pass
                        # still discovers files once a root appears.
                        supervisor.mark_unavailable("watcher", reason="no configured source root exists")
                        if supervisor.is_schedulable("watcher_registered_bridge"):
                            supervisor.mark_unavailable("watcher_registered_bridge", reason="watcher is unavailable")
                        if watcher_registered_gate_event is not None:
                            watcher_registered_gate_event.set()
                    elif enable_watch:
                        watcher_registered = getattr(watcher, "watcher_ready", None)
                        supervisor.start("watcher", watcher.run)
                        if watcher_registered_gate_event is not None and watcher_registered is not None:
                            supervisor.start(
                                "watcher_registered_bridge",
                                lambda: _bridge_watcher_registered(
                                    watcher_registered,
                                    watcher_registered_gate_event,
                                ),
                            )
                    if lifecycle_events_enabled:
                        await _emit_daemon_lifecycle_event(
                            "component_started",
                            archive_root_path=archive_root_path,
                            component="intake",
                            payload={"source_count": len(sources)},
                        )
                    await supervisor.wait()
            else:
                # Preflight-blocked, or no intake service is schedulable under
                # this profile: keep HTTP/health and other components serving
                # so operators see the degraded state.
                async def unresolved_intake_service() -> None:
                    raise AssertionError("an intake service was selected after its construction guard")

                for service_name in ("fair_intake", "watcher"):
                    if supervisor.state(service_name) is ServiceState.PENDING:
                        if watcher_creation_blocked:
                            supervisor.mark_unavailable(
                                service_name, reason="durable schema mismatch blocks intake construction"
                            )
                        else:
                            supervisor.start(service_name, unresolved_intake_service)
                if lifecycle_events_enabled:
                    await _emit_daemon_lifecycle_event(
                        "component_skipped",
                        archive_root_path=archive_root_path,
                        component="watcher",
                        payload={
                            "reason": "schema_blocked" if watcher_creation_blocked else "not_scheduled_by_profile",
                            "watch_enabled": enable_watch,
                            "profile": service_profile.value,
                        },
                    )
                await supervisor.wait()
        except BaseException as exc:
            termination = exc
            if not isinstance(exc, (asyncio.CancelledError, KeyboardInterrupt)):
                _log_completed_daemon_tasks(list(supervisor.tasks))
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
            if server is not None:
                await _shutdown_server_if_serving(server, server_task, label="browser-capture")
            if api_server is not None:
                await _shutdown_server_if_serving(api_server, api_server_task, label="api")
            if uds_server is not None:
                await _shutdown_server_if_serving(uds_server, uds_server_task, label="uds")
            if api_server is not None:
                await api_server.operation_runtime.shutdown()
            if ingest_owner_runtime is not None:
                await ingest_owner_runtime.shutdown()

            # One owner cancels and awaits every child inside its declared
            # deadline. Anything still running afterwards is named here
            # rather than abandoned unrecorded.
            shutdown_report = await supervisor.shutdown()
            if shutdown_report.orphaned:
                emit(
                    "daemon.shutdown.services_orphaned",
                    level=WARNING,
                    outcome="unmeasured",
                    reason="outlived_shutdown_deadline",
                    orphaned=len(shutdown_report.orphaned),
                    error_detail=", ".join(shutdown_report.orphaned),
                )
            if shutdown_report.failed:
                emit(
                    "daemon.shutdown.services_failed",
                    level=WARNING,
                    outcome="error",
                    services=len(shutdown_report.failed),
                    error_detail=", ".join(shutdown_report.failed),
                )

            # Fair intake owns candidate writes; the watcher can still hold
            # its live batch route. Other orphaned maintenance services do
            # not make the inactive candidate unsafe to discard. A candidate
            # writer orphan leaves it for the next build's reclamation.
            candidate_writer_orphaned = any(name in {"fair_intake", "watcher"} for name in shutdown_report.orphaned)
            if cold_build is not None and not cold_build.settled and not candidate_writer_orphaned:
                try:
                    async with asyncio.timeout(5.0):
                        await write_coordinator.run_sync("daemon.cold_build.shutdown_discard", cold_build.discard)
                except Exception as exc:
                    emit(
                        "daemon.cold_build.shutdown_discard_failed",
                        level=WARNING,
                        outcome="error",
                        generation_id=cold_build.generation_id,
                        error_type=type(exc).__name__,
                        error_detail=str(exc),
                    )

            if signal_termination:
                # CursorStore initialization re-applies OPS-tier DDL before it
                # marks running attempts interrupted.  That best-effort
                # recovery can wait behind an external OPS lock even after a
                # SIGTERM/SIGINT has made process exit the priority.  The same
                # CursorStore recovery runs on the next daemon startup, so
                # defer this nonessential shutdown write rather than stranding
                # the signal path behind its coordinator-owned worker.
                emit(
                    "daemon.shutdown.ingest_recovery_deferred",
                    outcome="skipped",
                    reason="signal_termination",
                    phase="shutdown",
                )
            else:
                try:
                    async with asyncio.timeout(5.0):
                        await write_coordinator.run_sync(
                            "shutdown.live_ingest_attempts",
                            _mark_interrupted_live_ingest_attempts_on_shutdown,
                        )
                except TimeoutError:
                    emit(
                        "daemon.shutdown.ingest_recovery_timeout",
                        level=WARNING,
                        outcome="unmeasured",
                        reason="interrupted_ingest_attempts_not_recorded",
                        phase="shutdown",
                        timeout_ms=5000,
                    )

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
                except Exception as exc:
                    emit(
                        "daemon.shutdown.lifecycle_stop_failed",
                        level=WARNING,
                        outcome="error",
                        phase="shutdown",
                        state=exit_kind,
                        error_type=type(exc).__name__,
                        error_detail=str(exc),
                    )

            ownership_retained_reason = _ownership_retention_reason(
                writer_drained=await _shutdown_writer_coordinator_with_rebuild_exclusion(
                    write_coordinator,
                    rebuild_exclusion,
                    timeout=5.0,
                ),
                orphaned_services=shutdown_report.orphaned,
            )
            # From here ``writer_drained`` is the ownership question, not just
            # the coordinator's: it gates the pidfile, the rebuild exclusion
            # and the durable archive lease below, and an orphaned child is
            # every bit as live a writer as an admitted operation.
            writer_drained = ownership_retained_reason is None
            pidfile_fd = _release_pidfile_after_writer_drain(
                pidfile_fd,
                writer_drained=writer_drained,
                reason=ownership_retained_reason or "",
            )
        finally:
            # Any exception or repeated cancellation before coordinator
            # shutdown leaves writer drain unproven.  The outer product
            # context must not interpret that control-flow escape as a safe
            # release: keep rebuild exclusion until process exit unless the
            # coordinator returned an affirmative drain result.
            _retain_rebuild_exclusion_for_undrained_writer(
                rebuild_exclusion,
                writer_drained=writer_drained,
            )
            # The process-local route must end even if best-effort discard
            # failed; leave any inactive generation on disk for recovery.
            if cold_build is not None:
                from polylogue.daemon.intake_adapters import clear_cold_build_generation

                clear_cold_build_generation()
            from polylogue.daemon.catchup_status import (
                set_cold_build_progress_provider,
                set_cold_build_settlement_provider,
            )

            set_cold_build_progress_provider(None)
            set_cold_build_settlement_provider(None)
            configure_browser_capture_status(None)
            if server is not None:
                with contextlib.suppress(Exception):
                    server.server_close()
            if api_server is not None:
                with contextlib.suppress(Exception):
                    api_server.server_close()
            if uds_server is not None:
                with contextlib.suppress(Exception):
                    uds_server.server_close()
            # The process compute capacity is the API server's kernel or the
            # shared fallback; either way this run published it, so this run
            # joins its workers. Every service and the writer have stopped, so
            # nothing new is admitted; a worker still running is named.
            surviving_compute = reset_compute_adapter(join_timeout_s=_COMPUTE_JOIN_TIMEOUT_S)
            if not surviving_compute:
                reset_drive_readiness_observation(archive_root_path)
            if surviving_compute:
                emit(
                    "daemon.shutdown.compute_threads_orphaned",
                    level=WARNING,
                    outcome="degraded",
                    reason="outlived_shutdown_deadline",
                    orphaned=len(surviving_compute),
                    error_detail=", ".join(surviving_compute),
                    timeout_ms=round(_COMPUTE_JOIN_TIMEOUT_S * 1000),
                )
                # A live compute worker is an incomplete shutdown: keep archive
                # ownership and report the stop as degraded, never clean.
                writer_drained = False
                ownership_retained_reason = "compute_threads_orphaned"
            if cleanup_task is not None:
                for _ in range(cleanup_cancel_requests):
                    cleanup_task.cancel()
            restore_signal_handlers(previous_signal_handlers)
            if writer_drained:
                archive_owner.release()
            _daemon_lifecycle = None
            _set_active_supervisor(None)

    emit(
        "daemon.stopped",
        level=WARNING if not writer_drained else INFO,
        outcome="ok" if writer_drained else "degraded",
        reason="clean" if writer_drained else (ownership_retained_reason or "ownership_retained"),
        pid=os.getpid(),
        held=not writer_drained,
    )


def _log_completed_daemon_tasks(tasks: list[asyncio.Task[None]]) -> None:
    for task in tasks:
        if not task.done():
            continue
        try:
            exc = task.exception()
        except asyncio.CancelledError:
            emit(
                "daemon.component_task.cancelled",
                level=WARNING,
                outcome="unmeasured",
                reason="cancelled_unexpectedly",
                component=task.get_name(),
            )
            continue
        if exc is None:
            emit(
                "daemon.component_task.exited",
                level=WARNING,
                outcome="degraded",
                reason="exited_unexpectedly",
                component=task.get_name(),
            )
        else:
            emit(
                "daemon.component_task.failed",
                level=WARNING,
                outcome="error",
                component=task.get_name(),
                error_type=type(exc).__name__,
                error_detail=str(exc),
            )


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
            emit(
                "daemon.server_task.cancelled",
                level=DEBUG,
                outcome="skipped",
                reason="shutting_down_server_anyway",
                component=label,
            )
            exc = None
        if exc is None and not task.cancelled():
            emit(
                "daemon.server_task.exited_early",
                level=WARNING,
                outcome="degraded",
                reason="exited_before_shutdown",
                component=label,
            )
            return
        if exc is not None:
            emit(
                "daemon.server_task.failed_early",
                level=WARNING,
                outcome="error",
                component=label,
                error_type=type(exc).__name__,
                error_detail=str(exc),
            )
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
        emit(
            "daemon.server_shutdown.timeout",
            level=WARNING,
            outcome="unmeasured",
            reason="closing_socket_directly",
            component=label,
            timeout_ms=5000,
        )


async def _run_browser_host(*, host: str, port: int, daemon_origin: str) -> None:
    """Own the optional browser process for one supervised service lifetime."""
    process = await asyncio.create_subprocess_exec(
        sys.executable,
        "-c",
        f"import sys, runpy; sys.path[:] = {sys.path!r}; "
        "runpy.run_module('polylogue.daemon.browser_host', run_name='__main__', alter_sys=True)",
        "--host",
        host,
        "--port",
        str(port),
        "--daemon-origin",
        daemon_origin,
    )
    try:
        returncode = await process.wait()
        raise RuntimeError(f"browser host exited unexpectedly with status {returncode}")
    finally:
        if process.returncode is None:
            with contextlib.suppress(ProcessLookupError):
                process.terminate()
            try:
                await asyncio.wait_for(process.wait(), timeout=5.0)
            except TimeoutError:
                with contextlib.suppress(ProcessLookupError):
                    process.kill()
                await asyncio.wait_for(process.wait(), timeout=1.0)


async def _serve_until_complete(
    server: Any,
    *,
    label: str,
) -> None:
    """Serve a long-lived socket server without occupying asyncio's executor.

    ``asyncio.run()`` always joins every default-executor worker during loop
    teardown. A cancelled ``to_thread(serve_forever)`` future does not stop its
    worker, so one missed or raced shutdown can freeze the entire daemon exit.
    Start the server in an explicitly daemonized thread and expose only its
    completion to the task graph. The normal shutdown path still calls
    ``server.shutdown()`` and drains this task; a genuinely wedged server can no
    longer strand interpreter teardown.
    """
    loop = asyncio.get_running_loop()
    completed = asyncio.Event()
    failure: list[BaseException] = []

    def _serve() -> None:
        try:
            server.serve_forever(0.5)
        except BaseException as exc:
            failure.append(exc)
        finally:
            with contextlib.suppress(RuntimeError):
                loop.call_soon_threadsafe(completed.set)
                # The daemon thread is intentionally non-owning. If a wedged
                # server outlives loop teardown, there is no loop consumer left.

    threading.Thread(target=_serve, name=f"{label}-server", daemon=True).start()

    await completed.wait()
    if failure:
        raise failure[0]


@click.command("health", help="Run tiered daemon health checks.")
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


@click.command("run", help="Run configured long-lived daemon components.")
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
    "--cold-build-index",
    is_flag=True,
    help=(
        "Build the index into a new inactive generation and promote it when intake drains. "
        "Implied on a root whose active index generation is empty; readers keep the current "
        "generation until promotion, and an interrupted build is discarded."
    ),
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
    "--browser-port",
    default=None,
    type=click.IntRange(1, 65535),
    help="Run the optional browser host on this port (for example 8767). Requires the daemon API.",
)
@click.option(
    "--api-auth-token",
    default=None,
    help="Daemon API auth token; auto-minted/loaded from a 0600 file if not given.",
)
@click.option(
    "--api-allow-no-auth",
    is_flag=True,
    default=False,
    envvar=API_ALLOW_NO_AUTH_ENV,
    help=(
        "Run the daemon API with no bearer token at all. Any local process can then read/write "
        "through it -- default OFF; an explicit opt-out for the auto-minted-token default."
    ),
)
@click.option(
    "--listener-info-path",
    type=click.Path(path_type=Path),
    default=None,
    help="Atomically write actual bound API and capture addresses as private JSON after listener startup.",
)
@click.pass_context
def run_command(
    ctx: click.Context,
    host: str,
    port: int,
    no_watch: bool,
    cold_build_index: bool,
    no_source_catchup: bool,
    no_browser_capture: bool,
    insecure_allow_remote: bool,
    browser_capture_auth_token: str | None,
    browser_capture_allow_no_auth: bool,
    browser_capture_origins: tuple[str, ...],
    no_api: bool,
    api_host: str,
    api_port: int,
    browser_port: int | None,
    api_auth_token: str | None,
    api_allow_no_auth: bool,
    listener_info_path: Path | None,
) -> None:
    """Run configured daemon components.

    This is the entry point for the polylogued systemd service.
    """
    _enable_faulthandler_if_supported()
    configure_logging()
    configure_events()

    # Bound before anything else emits, so the earliest events of this run
    # carry the id its lifecycle row, heartbeats and status report.
    daemon_run_id()
    emit("daemon.run.start", pid=os.getpid())

    from polylogue.config import resolve_runtime_config
    from polylogue.operations.durable_change_train import DurableChangeTrainError

    runtime = resolve_runtime_config()
    cfg = runtime.settings

    def parameter_is_default(name: str) -> bool:
        source = ctx.get_parameter_source(name)
        return source is None or source is click.core.ParameterSource.DEFAULT

    if parameter_is_default("host") and cfg.layer_of("browser_capture_host") != "default":
        host = cfg.browser_capture_host
    if parameter_is_default("port") and cfg.layer_of("browser_capture_port") != "default":
        port = cfg.browser_capture_port
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
    if parameter_is_default("api_host") and cfg.layer_of("api_host") != "default":
        api_host = cfg.api_host
    if parameter_is_default("api_port") and cfg.layer_of("api_port") != "default":
        api_port = cfg.api_port
    if parameter_is_default("api_auth_token") and cfg.api_auth_token:
        api_auth_token = cfg.api_auth_token
    if parameter_is_default("api_allow_no_auth"):
        api_allow_no_auth = cfg.api_allow_no_auth

    enable_watch = not no_watch
    enable_source_catchup = not no_source_catchup
    enable_browser_capture = not no_browser_capture
    enable_api = not no_api
    if not enable_watch and not enable_browser_capture and not enable_api:
        raise click.UsageError("at least one daemon component must be enabled")

    atexit.register(_cleanup_pidfile)

    sources = _watch_sources(hermes_root=runtime.source_paths.hermes)
    components = []
    if enable_watch:
        components.append(f"watch={len(sources)} source(s)")
    if enable_browser_capture:
        components.append(f"browser-capture=http://{host}:{port}")
    if enable_api:
        components.append(f"api=http://{api_host}:{api_port}")
    if browser_port is not None:
        components.append(f"browser=http://{api_host}:{browser_port}")
    click.echo(
        f"Starting polylogued ({', '.join(components)}). Ctrl-C to stop.",
        err=True,
    )

    try:
        asyncio.run(
            run_daemon_services(
                sources=sources,
                enable_watch=enable_watch,
                enable_source_catchup=enable_source_catchup,
                enable_browser_capture=enable_browser_capture,
                browser_capture_host=host,
                browser_capture_port=port,
                browser_capture_allow_remote=insecure_allow_remote,
                browser_capture_auth_token=browser_capture_auth_token,
                browser_capture_allow_no_auth=browser_capture_allow_no_auth,
                browser_capture_extra_origins=browser_capture_origins,
                enable_api=enable_api,
                api_host=api_host,
                api_port=api_port,
                browser_port=browser_port,
                api_auth_token=api_auth_token,
                api_allow_no_auth=api_allow_no_auth,
                cold_build_index=cold_build_index,
                listener_info_path=listener_info_path,
            )
        )
    except KeyboardInterrupt:
        click.echo("Stopping polylogued.", err=True)
    except DurableChangeTrainError as exc:
        raise click.ClickException(f"polylogued refused to open the archive: {exc}") from exc
    finally:
        # The CLI configured the process-global queued event sink. Drain it
        # after the daemon's final stop event; embedded callers of
        # run_daemon_services do not own that global sink.
        shutdown_events(timeout_s=0.25)


@click.command("watch", help="Watch source directories and ingest new sessions live.")
def watch_command() -> None:
    from polylogue.config import resolve_runtime_config
    from polylogue.operations.durable_change_train import ArchiveOwnershipError, DurableChangeTrainError
    from polylogue.paths import archive_root

    runtime_source_paths = resolve_runtime_config().source_paths
    sources = _watch_sources(hermes_root=runtime_source_paths.hermes)

    archive_root_path = Path(archive_root())
    archive_root_path.mkdir(mode=0o700, parents=True, exist_ok=True)
    # Keep the standalone command on the same supervised composition as
    # ``polylogued run``.  In particular, LiveWatcher is only a filesystem
    # hint producer here; FairIntakeDispatcher owns source discovery and
    # admission just as it does for the ordinary daemon.
    try:
        asyncio.run(
            run_daemon_services(
                sources=sources,
                enable_watch=True,
                enable_source_catchup=True,
                enable_browser_capture=False,
                browser_capture_host="127.0.0.1",
                browser_capture_port=8765,
                enable_api=False,
                startup_message=f"Watching {len(sources)} source(s). Ctrl-C to stop.",
            )
        )
    except ArchiveOwnershipError as exc:
        raise click.ClickException(f"watch could not acquire exclusive archive ownership: {exc}") from exc
    except DurableChangeTrainError as exc:
        raise click.ClickException(f"watch refused to open the archive: {exc}") from exc


__all__ = [
    "default_sources",
    "health_command",
    "run_command",
    "run_daemon_services",
    "watch_command",
]
