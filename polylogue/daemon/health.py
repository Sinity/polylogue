"""Tiered daemon health checks with typed alerts and severity escalation.

Health checks are grouped into three tiers by cost:
- FAST: sub-1s checks (liveness, disk space, WAL size, source availability)
- MEDIUM: sub-10s queries (FTS readiness, insight freshness, stale attempts,
  repeated stage failures, raw failures)
- EXPENSIVE: longer-running checks (DB integrity, blob integrity, embedding coverage)

Each check produces a ``HealthAlert`` with severity, message, and checked_at.
Alerts accrue a ``consecutive_failures`` counter that carries forward across
check cycles so operators can detect persistent conditions.
"""

from __future__ import annotations

import os
import sqlite3
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path

from pydantic import BaseModel, Field

from polylogue.config import PolylogueConfig
from polylogue.daemon.embedding_readiness import embedding_readiness_info
from polylogue.logging import get_logger
from polylogue.paths import archive_root, db_path, index_db_path, resolve_active_index_db_path
from polylogue.storage.fts.fts_lifecycle import FTS_TRIGGER_NAMES as _EXPECTED_FTS_TRIGGERS

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Typed enums
# ---------------------------------------------------------------------------


class HealthSeverity(str, Enum):
    """Alert severity with implied escalation.

    ``INFO`` is reserved for known-transient states (e.g., FTS triggers
    dropped inside an in-flight bulk catch-up writer; see #1613) where
    the underlying check would otherwise fire CRITICAL but the
    operator must be told it is expected, not paged. INFO does not
    escalate ``overall_status``.
    """

    OK = "ok"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class HealthTier(str, Enum):
    """Check cost tier. Determines how often checks run and when they block."""

    FAST = "fast"
    MEDIUM = "medium"
    EXPENSIVE = "expensive"


# ---------------------------------------------------------------------------
# Typed models
# ---------------------------------------------------------------------------


class HealthAlert(BaseModel):
    """A single health check result with severity and retry tracking."""

    check_name: str
    tier: HealthTier
    severity: HealthSeverity
    message: str
    checked_at: str
    consecutive_failures: int = 0


class DaemonHealth(BaseModel):
    """Aggregated health state across all tiers.

    ``overall_status`` reflects the worst severity among all alerts.
    ``tier_summary`` maps tier name to counts grouped by severity.
    """

    overall_status: HealthSeverity = HealthSeverity.OK
    checked_at: str = ""
    alerts: list[HealthAlert] = Field(default_factory=list)
    tier_summary: dict[str, dict[str, int]] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------

_DISK_FREE_WARN_BYTES = 500 * 1024 * 1024  # 500 MB
_DISK_FREE_CRIT_BYTES = 100 * 1024 * 1024  # 100 MB
_WAL_WARN_BYTES = 50 * 1024 * 1024  # 50 MB
_WAL_CRIT_BYTES = 200 * 1024 * 1024  # 200 MB
_RAW_FAILURE_WARN_COUNT = 10
_RAW_FAILURE_ERROR_COUNT = 50


def _active_health_db_path() -> Path:
    return resolve_active_index_db_path(db_anchor=db_path(), index_db=index_db_path())


# ---------------------------------------------------------------------------
# Check state tracking (in-memory; resets on daemon restart)
# ---------------------------------------------------------------------------

_failure_counts: dict[str, int] = {}


def _record_failure(check_name: str, is_ok: bool) -> int:
    """Update the consecutive-failure counter for a named check.

    Returns the new count (0 on success, incremented on failure).
    """
    if is_ok:
        _failure_counts[check_name] = 0
        return 0
    count = _failure_counts.get(check_name, 0) + 1
    _failure_counts[check_name] = count
    return count


# ---------------------------------------------------------------------------
# Fast checks (< 1s)
# ---------------------------------------------------------------------------


def _check_daemon_liveness_fast() -> HealthAlert:
    """Check whether the daemon process is running via pidfile."""
    from polylogue.daemon.status import _check_daemon_liveness

    now = datetime.now(UTC).isoformat()
    try:
        alive = _check_daemon_liveness()
        is_ok = alive
        severity = HealthSeverity.OK if is_ok else HealthSeverity.WARNING
        message = "daemon is running" if is_ok else "daemon process not detected"
        return HealthAlert(
            check_name="daemon_liveness",
            tier=HealthTier.FAST,
            severity=severity,
            message=message,
            checked_at=now,
            consecutive_failures=_record_failure("daemon_liveness", is_ok),
        )
    except Exception as exc:
        return HealthAlert(
            check_name="daemon_liveness",
            tier=HealthTier.FAST,
            severity=HealthSeverity.ERROR,
            message=f"liveness check failed: {exc}",
            checked_at=now,
            consecutive_failures=_record_failure("daemon_liveness", False),
        )


def _check_disk_space_fast() -> HealthAlert:
    """Check free disk space on the archive volume."""
    now = datetime.now(UTC).isoformat()
    try:
        root = archive_root()
        st = os.statvfs(str(root))
        free = st.f_frsize * st.f_bavail
        if free >= _DISK_FREE_WARN_BYTES:
            severity = HealthSeverity.OK
            message = f"disk free: {free / (1024**3):.1f} GB"
        elif free >= _DISK_FREE_CRIT_BYTES:
            severity = HealthSeverity.WARNING
            message = f"disk space low: {free / (1024**2):.0f} MB free"
        else:
            severity = HealthSeverity.CRITICAL
            message = f"disk space critically low: {free / (1024**2):.0f} MB free"
        return HealthAlert(
            check_name="disk_space",
            tier=HealthTier.FAST,
            severity=severity,
            message=message,
            checked_at=now,
            consecutive_failures=_record_failure("disk_space", severity == HealthSeverity.OK),
        )
    except Exception as exc:
        return HealthAlert(
            check_name="disk_space",
            tier=HealthTier.FAST,
            severity=HealthSeverity.ERROR,
            message=f"disk check failed: {exc}",
            checked_at=now,
            consecutive_failures=_record_failure("disk_space", False),
        )


def _check_wal_size_fast() -> HealthAlert:
    """Check WAL file size is within bounds."""
    now = datetime.now(UTC).isoformat()
    dbf = _active_health_db_path()
    wal = dbf.with_suffix(".db-wal")
    try:
        if not wal.exists():
            return HealthAlert(
                check_name="wal_size",
                tier=HealthTier.FAST,
                severity=HealthSeverity.OK,
                message="WAL file not present",
                checked_at=now,
                consecutive_failures=_record_failure("wal_size", True),
            )
        size = wal.stat().st_size
        if size < _WAL_WARN_BYTES:
            severity = HealthSeverity.OK
            message = f"WAL size: {size / (1024**2):.1f} MB"
        elif size < _WAL_CRIT_BYTES:
            severity = HealthSeverity.WARNING
            message = f"WAL size elevated: {size / (1024**2):.1f} MB"
        else:
            severity = HealthSeverity.ERROR
            message = f"WAL size too large: {size / (1024**2):.1f} MB — checkpoint overdue"
        return HealthAlert(
            check_name="wal_size",
            tier=HealthTier.FAST,
            severity=severity,
            message=message,
            checked_at=now,
            consecutive_failures=_record_failure("wal_size", severity == HealthSeverity.OK),
        )
    except Exception as exc:
        return HealthAlert(
            check_name="wal_size",
            tier=HealthTier.FAST,
            severity=HealthSeverity.ERROR,
            message=f"WAL check failed: {exc}",
            checked_at=now,
            consecutive_failures=_record_failure("wal_size", False),
        )


def _check_source_availability_fast() -> HealthAlert:
    """Check that configured watch source roots exist and are readable."""
    from polylogue.sources.live.watcher import default_sources

    now = datetime.now(UTC).isoformat()
    try:
        sources = default_sources()
        missing: list[str] = []
        unreadable: list[str] = []
        available = 0
        for src in sources:
            if not src.exists():
                missing.append(src.name)
            elif not os.access(str(src.root), os.R_OK):
                unreadable.append(src.name)
            else:
                available += 1

        if not missing and not unreadable:
            severity = HealthSeverity.OK
            message = f"{available} source(s) available"
        elif missing and not unreadable:
            severity = HealthSeverity.WARNING
            message = f"{available}/{len(sources)} source(s) available, missing: {', '.join(missing)}"
        elif unreadable:
            severity = HealthSeverity.ERROR
            message = f"{available}/{len(sources)} source(s) available, unreadable: {', '.join(unreadable)}"
        else:
            severity = HealthSeverity.WARNING
            message = f"{available}/{len(sources)} source(s) available"

        return HealthAlert(
            check_name="source_availability",
            tier=HealthTier.FAST,
            severity=severity,
            message=message,
            checked_at=now,
            consecutive_failures=_record_failure("source_availability", severity == HealthSeverity.OK),
        )
    except Exception as exc:
        return HealthAlert(
            check_name="source_availability",
            tier=HealthTier.FAST,
            severity=HealthSeverity.ERROR,
            message=f"source availability check failed: {exc}",
            checked_at=now,
            consecutive_failures=_record_failure("source_availability", False),
        )


"""Canonical FTS sync triggers for archive and insight search surfaces."""


def _find_missing_fts_triggers(conn: sqlite3.Connection) -> list[str]:
    """Return the names of expected FTS triggers that do not exist."""
    expected = _active_fts_triggers(conn)
    if not expected:
        return []
    placeholders = ",".join("?" for _ in expected)
    rows = conn.execute(
        f"SELECT name FROM sqlite_schema WHERE type='trigger' AND name IN ({placeholders})",
        expected,
    ).fetchall()
    present = {row[0] for row in rows}
    return [name for name in expected if name not in present]


def _table_exists_for_fts_trigger(conn: sqlite3.Connection, table_name: str) -> bool:
    return (
        conn.execute(
            "SELECT 1 FROM sqlite_schema WHERE type='table' AND name = ? LIMIT 1",
            (table_name,),
        ).fetchone()
        is not None
    )


def _active_fts_triggers(conn: sqlite3.Connection) -> tuple[str, ...]:
    from polylogue.daemon.fts_startup import active_fts_triggers_sync

    return active_fts_triggers_sync(conn)


def _auto_restore_fts_triggers(dbf: object) -> tuple[list[str], bool, str]:
    """Restore missing FTS triggers and rebuild the FTS index.

    Used by :func:`_check_fts_trigger_drift_fast` when
    ``health.fts_auto_restore = true``. Restore is metadata-only (drops
    and re-creates triggers from the canonical DDL in
    ``polylogue.storage.fts.fts_lifecycle``) and the subsequent rebuild
    reconstructs the FTS index from the persisted ``messages`` and
    ``action_events`` rows. Both operations are read-only in the sense
    that they do not modify user-visible archive data.

    Returns:
        ``(restored, success, detail)`` — list of trigger names restored,
        whether the operation succeeded, and an operator-facing detail.
    """
    from polylogue.storage.fts.fts_lifecycle import (
        rebuild_fts_index_sync,
        restore_fts_triggers_sync,
    )

    conn = sqlite3.connect(str(dbf), timeout=5.0)
    try:
        missing_before = _find_missing_fts_triggers(conn)
        if not missing_before:
            return [], True, "no missing triggers to restore"
        restore_fts_triggers_sync(conn)
        rebuild_fts_index_sync(conn)
        conn.commit()
        missing_after = _find_missing_fts_triggers(conn)
        if missing_after:
            return (
                missing_before,
                False,
                (f"restore attempted but {len(missing_after)} trigger(s) still missing: {', '.join(missing_after)}"),
            )
        return (
            missing_before,
            True,
            (f"restored {len(missing_before)} trigger(s) and rebuilt FTS index: {', '.join(missing_before)}"),
        )
    finally:
        conn.close()


# Window inside which a ``live_ingest_attempt`` row with
# ``status='running'`` is considered "actively bulk-writing right now"
# rather than a stuck/SIGKILLed row. Sized to comfortably exceed the
# canonical heartbeat cadence (#1198 ~10s) plus the slow-vs-stuck classifier
# tolerance (#1246) so a normal long chunk does not flip back to CRITICAL.
_BULK_ATTEMPT_FRESHNESS_S = 120


def _active_bulk_ingest_attempt(
    conn: sqlite3.Connection,
    *,
    now_iso: str,
    ops_db: Path | None = None,
) -> tuple[str, str, str] | None:
    """Return ``(attempt_id, phase, updated_at)`` for an in-flight bulk attempt.

    Returns ``None`` when:

    - the ``live_ingest_attempt`` table doesn't exist yet (fresh archive),
    - no row has ``status='running'`` in a bulk-suspending phase, or
    - the most recent qualifying row's ``updated_at`` is stale (no
      heartbeat for ``_BULK_ATTEMPT_FRESHNESS_S`` seconds — the daemon
      is gone, the row is orphaned, and the dropped triggers are no
      longer "in flight" but "leaked").

    The bulk-suspending phases match the live watcher's full-ingest
    progress phases that wrap the ``suspend_fts_triggers=True`` window
    in ``pipeline/services/ingest_batch/_core.py`` (``full_parse``,
    ``full_worker_wait``). When either phase is the latest progress
    record on a live attempt, FTS triggers may legitimately be dropped
    inside the writer's open transaction.
    """
    ops_attempt = _archive_active_bulk_ingest_attempt(ops_db, now_iso=now_iso)
    if ops_attempt is not None:
        return ops_attempt
    try:
        row = conn.execute(
            "SELECT 1 FROM sqlite_schema WHERE type='table' AND name='live_ingest_attempt' LIMIT 1"
        ).fetchone()
    except sqlite3.Error:
        return None
    if row is None:
        return None
    try:
        record = conn.execute(
            """
            SELECT attempt_id, phase, updated_at
            FROM live_ingest_attempt
            WHERE status = 'running' AND phase IN ('full_parse', 'full_worker_wait')
            ORDER BY updated_at DESC
            LIMIT 1
            """
        ).fetchone()
    except sqlite3.Error:
        return None
    if record is None:
        return None
    attempt_id, phase, updated_at = str(record[0]), str(record[1]), str(record[2])
    try:
        attempt_dt = datetime.fromisoformat(updated_at.replace("Z", "+00:00"))
        now_dt = datetime.fromisoformat(now_iso.replace("Z", "+00:00"))
        if (now_dt - attempt_dt).total_seconds() > _BULK_ATTEMPT_FRESHNESS_S:
            return None
    except (TypeError, ValueError):
        # Unparseable timestamp — refuse to downgrade. Better to over-alert
        # than to silence a real drift.
        return None
    return attempt_id, phase, updated_at


def _archive_active_bulk_ingest_attempt(ops_db: Path | None, *, now_iso: str) -> tuple[str, str, str] | None:
    if ops_db is None or not ops_db.exists():
        return None
    try:
        conn = sqlite3.connect(str(ops_db))
        try:
            has_table = bool(
                conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='ingest_attempts'").fetchone()
            )
            if not has_table:
                return None
            record = conn.execute(
                """
                SELECT attempt_id, phase, heartbeat_at_ms
                FROM ingest_attempts
                WHERE status = 'running' AND phase IN ('full_parse', 'full_worker_wait')
                ORDER BY heartbeat_at_ms DESC, started_at_ms DESC
                LIMIT 1
                """
            ).fetchone()
        finally:
            conn.close()
    except sqlite3.Error:
        return None
    if record is None or record[2] is None:
        return None
    attempt_id, phase = str(record[0]), str(record[1])
    try:
        updated_at = datetime.fromtimestamp(int(record[2]) / 1000, tz=UTC).isoformat()
        attempt_dt = datetime.fromisoformat(updated_at)
        now_dt = datetime.fromisoformat(now_iso.replace("Z", "+00:00"))
        if (now_dt - attempt_dt).total_seconds() > _BULK_ATTEMPT_FRESHNESS_S:
            return None
    except (TypeError, ValueError, OSError):
        return None
    return attempt_id, phase, updated_at


def _check_fts_trigger_drift_fast() -> HealthAlert:
    """Detect missing FTS sync triggers; auto-restore when configured.

    The active FTS triggers keep each schema-present search surface in sync
    with its source tables. A SIGKILL during
    the bulk-write trigger-suspension window
    (see ``docs/internals.md`` "FTS5 Model") leaves them dropped, which
    silently corrupts search results until the next bulk operation
    restores them. The archive checks ``blocks_fts`` triggers.

    Behavior:

    - Missing triggers raise CRITICAL — silent search corruption is a
      semantic regression, not a perf wart.
    - When ``health.fts_auto_restore = true`` (env
      ``POLYLOGUE_HEALTH_FTS_AUTO_RESTORE``) and triggers are missing,
      the check restores them and rebuilds the FTS index in place. The
      alert then carries severity WARNING (still surfaced so the
      operator knows recovery happened) with a ``recovery=`` detail.
    - Restore failures fall back to CRITICAL with the restore-error
      detail appended so the operator can intervene.
    """
    from polylogue.config import load_polylogue_config

    now = datetime.now(UTC).isoformat()
    dbf = _active_health_db_path()
    if not dbf.exists():
        return HealthAlert(
            check_name="fts_trigger_drift",
            tier=HealthTier.FAST,
            severity=HealthSeverity.OK,
            message="no database yet (FTS triggers will be installed at fresh-init)",
            checked_at=now,
            consecutive_failures=_record_failure("fts_trigger_drift", True),
        )

    try:
        # Read-only inspection first so the cheap healthy path never
        # opens a writable connection.
        conn = sqlite3.connect(f"file:{dbf}?mode=ro", uri=True, timeout=2.0)
        try:
            active_trigger_count = len(_active_fts_triggers(conn))
            missing = _find_missing_fts_triggers(conn)
            bulk_attempt = (
                _active_bulk_ingest_attempt(conn, now_iso=now, ops_db=dbf.with_name("ops.db")) if missing else None
            )
        finally:
            conn.close()

        if not missing:
            return HealthAlert(
                check_name="fts_trigger_drift",
                tier=HealthTier.FAST,
                severity=HealthSeverity.OK,
                message=f"all {active_trigger_count} active FTS triggers present",
                checked_at=now,
                consecutive_failures=_record_failure("fts_trigger_drift", True),
            )

        # #1613: missing triggers during an actively running bulk catch-up
        # are normal — the writer dropped them inside its own transaction
        # and the commit_archive_write_effects path will restore them
        # before the commit lands. Downgrade the alert to INFO so the
        # operator dashboard does not cry-wolf during a healthy bootstrap
        # rebuild. SIGKILLed leaks remain CRITICAL because the bulk-
        # attempt freshness window (_BULK_ATTEMPT_FRESHNESS_S) excludes
        # rows whose heartbeat stopped.
        if bulk_attempt is not None:
            attempt_id, phase, updated_at = bulk_attempt
            return HealthAlert(
                check_name="fts_trigger_drift",
                tier=HealthTier.FAST,
                severity=HealthSeverity.INFO,
                message=(
                    f"FTS triggers suspended for bulk catch-up "
                    f"(attempt={attempt_id} phase={phase}; fresh as of {updated_at}); "
                    f"{len(missing)}/{active_trigger_count} missing inside the writer's transaction"
                ),
                checked_at=now,
                consecutive_failures=_record_failure("fts_trigger_drift", True),
            )

        restore_cmd = "polylogue check --repair-fts"
        missing_text = ", ".join(missing)
        try:
            cfg = load_polylogue_config()
            auto_restore = bool(cfg.health_fts_auto_restore)
        except Exception:
            auto_restore = False

        if not auto_restore:
            message = (
                f"FTS trigger drift: {len(missing)}/{active_trigger_count} missing "
                f"({missing_text}); restore with `{restore_cmd}` "
                "(rebuild ~O(messages))"
            )
            return HealthAlert(
                check_name="fts_trigger_drift",
                tier=HealthTier.FAST,
                severity=HealthSeverity.CRITICAL,
                message=message,
                checked_at=now,
                consecutive_failures=_record_failure("fts_trigger_drift", False),
            )

        try:
            restored, success, detail = _auto_restore_fts_triggers(dbf)
        except Exception as exc:
            return HealthAlert(
                check_name="fts_trigger_drift",
                tier=HealthTier.FAST,
                severity=HealthSeverity.CRITICAL,
                message=(
                    f"FTS trigger drift: {len(missing)} missing ({missing_text}); "
                    f"auto-restore failed: {exc}; restore manually with `{restore_cmd}`"
                ),
                checked_at=now,
                consecutive_failures=_record_failure("fts_trigger_drift", False),
            )

        if success:
            logger.info(
                "daemon.health.fts_trigger_drift: auto-restored %d trigger(s): %s",
                len(restored),
                ", ".join(restored),
            )
            # Surface a WARNING-level recovery event so the notification
            # backend forwards it to the operator — silent self-heal
            # would hide the precipitating SIGKILL/suspension-leak.
            return HealthAlert(
                check_name="fts_trigger_drift",
                tier=HealthTier.FAST,
                severity=HealthSeverity.WARNING,
                message=f"FTS trigger drift auto-recovered: {detail}",
                checked_at=now,
                consecutive_failures=_record_failure("fts_trigger_drift", True),
            )

        return HealthAlert(
            check_name="fts_trigger_drift",
            tier=HealthTier.FAST,
            severity=HealthSeverity.CRITICAL,
            message=(f"FTS trigger drift: {detail}; restore manually with `{restore_cmd}`"),
            checked_at=now,
            consecutive_failures=_record_failure("fts_trigger_drift", False),
        )
    except Exception as exc:
        return HealthAlert(
            check_name="fts_trigger_drift",
            tier=HealthTier.FAST,
            severity=HealthSeverity.ERROR,
            message=f"FTS trigger drift check failed: {exc}",
            checked_at=now,
            consecutive_failures=_record_failure("fts_trigger_drift", False),
        )


def _check_schema_version_fast() -> HealthAlert:
    """Compare the on-disk ``PRAGMA user_version`` to the runtime ``SCHEMA_VERSION``.

    A mismatch means this binary cannot read or write the database safely —
    every ingest attempt would raise :class:`SchemaVersionMismatchError`, and
    retrying produces only IO load with no progress. The check therefore
    raises CRITICAL so the daemon's preflight can refuse to start the watcher
    and an operator/dashboard immediately sees the structural cause.

    The check uses ``PRAGMA user_version`` directly rather than the schema
    snapshot machinery so it is tolerant of structural drift and stays cheap
    (a single read on a read-only connection).
    """
    from polylogue.storage.sqlite.schema_bootstrap import (
        SCHEMA_VERSION,
        capture_schema_snapshot,
        decide_schema_bootstrap,
    )

    now = datetime.now(UTC).isoformat()
    dbf = _active_health_db_path()
    if not dbf.exists():
        # Fresh install — fresh-init will create the DB at the current version.
        return HealthAlert(
            check_name="schema_version",
            tier=HealthTier.FAST,
            severity=HealthSeverity.OK,
            message=f"no database yet (will create v{SCHEMA_VERSION})",
            checked_at=now,
            consecutive_failures=_record_failure("schema_version", True),
        )
    try:
        # Open read-only via URI to avoid creating/locking on a misconfigured DB.
        conn = sqlite3.connect(f"file:{dbf}?mode=ro", uri=True, timeout=2.0)
        try:
            snapshot = capture_schema_snapshot(conn)
        finally:
            conn.close()

        current = snapshot.current_version
        # Polylogue has no in-place upgrade chain. Either the file is empty, it
        # is the archive shape this runtime knows, or the watcher refuses to
        # operate against it.
        decision = decide_schema_bootstrap(snapshot)

        if current == SCHEMA_VERSION:
            severity = HealthSeverity.OK
            message = f"schema v{current} matches runtime"
        elif current == 0:
            # Empty DB file (created but never bootstrapped) — fresh-init will
            # handle this, not a structural mismatch.
            severity = HealthSeverity.OK
            message = f"empty database (will bootstrap v{SCHEMA_VERSION})"
        else:
            assert decision.action == "version_mismatch"
            severity = HealthSeverity.CRITICAL
            if current > SCHEMA_VERSION:
                hint = "rebuild or redeploy polylogue to a build that supports this schema"
            else:
                hint = "no in-place upgrade exists; rebuild the archive from source or move it aside"
            message = f"schema v{current} is not runtime v{SCHEMA_VERSION} — {hint}"

        is_ok = severity == HealthSeverity.OK
        return HealthAlert(
            check_name="schema_version",
            tier=HealthTier.FAST,
            severity=severity,
            message=message,
            checked_at=now,
            consecutive_failures=_record_failure("schema_version", is_ok),
        )
    except Exception as exc:
        return HealthAlert(
            check_name="schema_version",
            tier=HealthTier.FAST,
            severity=HealthSeverity.ERROR,
            message=f"schema version check failed: {exc}",
            checked_at=now,
            consecutive_failures=_record_failure("schema_version", False),
        )


def _run_fast_checks() -> list[HealthAlert]:
    return [
        _check_daemon_liveness_fast(),
        _check_schema_version_fast(),
        _check_disk_space_fast(),
        _check_wal_size_fast(),
        _check_source_availability_fast(),
        _check_fts_trigger_drift_fast(),
    ]


# ---------------------------------------------------------------------------
# Medium checks (< 10s)
# ---------------------------------------------------------------------------


def _check_fts_readiness_medium() -> HealthAlert:
    """Check every active FTS-backed search surface is exactly fresh."""
    now = datetime.now(UTC).isoformat()
    dbf = _active_health_db_path()
    if not dbf.exists():
        return HealthAlert(
            check_name="fts_readiness",
            tier=HealthTier.MEDIUM,
            severity=HealthSeverity.ERROR,
            message="database not found",
            checked_at=now,
            consecutive_failures=_record_failure("fts_readiness", False),
        )
    try:
        conn = sqlite3.connect(str(dbf))
        try:
            from polylogue.storage.fts.fts_lifecycle import fts_invariant_snapshot_sync

            snapshot = fts_invariant_snapshot_sync(conn)
            broken = [surface for surface in snapshot.surfaces if not surface.ready]
            if not broken:
                severity = HealthSeverity.OK
                message = "FTS up to date"
            else:
                severity = HealthSeverity.ERROR
                details = []
                for surface in broken:
                    if not surface.source_exists:
                        details.append(f"{surface.name}: unexpected table without source")
                    elif not surface.exists:
                        details.append(f"{surface.name}: missing table")
                    elif not surface.triggers_present:
                        details.append(f"{surface.name}: missing triggers")
                    elif surface.missing_rows:
                        details.append(f"{surface.name}: {surface.missing_rows} missing row(s)")
                    elif surface.excess_rows:
                        details.append(f"{surface.name}: {surface.excess_rows} stale row(s)")
                    elif surface.duplicate_rows:
                        details.append(f"{surface.name}: {surface.duplicate_rows} duplicate row(s)")
                    else:
                        details.append(f"{surface.name}: not fresh")
                message = "FTS invariant failed: " + "; ".join(details)
            is_ok = severity == HealthSeverity.OK
            return HealthAlert(
                check_name="fts_readiness",
                tier=HealthTier.MEDIUM,
                severity=severity,
                message=message,
                checked_at=now,
                consecutive_failures=_record_failure("fts_readiness", is_ok),
            )
        finally:
            conn.close()
    except Exception as exc:
        return HealthAlert(
            check_name="fts_readiness",
            tier=HealthTier.MEDIUM,
            severity=HealthSeverity.ERROR,
            message=f"FTS check failed: {exc}",
            checked_at=now,
            consecutive_failures=_record_failure("fts_readiness", False),
        )


def _check_raw_failures_medium() -> HealthAlert:
    """Check raw session parse/validation/maintenance failure counts.

    Maintenance failures routed via
    :func:`polylogue.maintenance.failure_routing.route_failure_sample`
    (#1198) participate in the same alert ladder as ingest failures.
    When the maintenance bucket dominates, the message names a
    representative ``operation_id`` so the operator can pull the
    originating replay state file directly.
    """
    now = datetime.now(UTC).isoformat()
    try:
        from polylogue.daemon.status import _raw_failure_info

        info = _raw_failure_info()
        raw_parse = info.get("parse_failures", 0)
        parse = int(raw_parse) if isinstance(raw_parse, (int, float)) else 0
        raw_val = info.get("validation_failures", 0)
        validation = int(raw_val) if isinstance(raw_val, (int, float)) else 0
        quarantined = info.get("quarantined", 0) if isinstance(info.get("quarantined"), int) else 0
        raw_maint = info.get("maintenance_failures", 0)
        maintenance = int(raw_maint) if isinstance(raw_maint, (int, float)) else 0
        total_failures = parse + validation + maintenance

        op_hint = ""
        if maintenance > 0:
            samples = info.get("samples", [])
            if isinstance(samples, list):
                for sample in samples:
                    op_id = getattr(sample, "operation_id", None)
                    src = getattr(sample, "source", None)
                    if src == "maintenance" and op_id:
                        op_hint = f" (op={str(op_id)[:8]})"
                        break

        if total_failures == 0:
            severity = HealthSeverity.OK
            message = "no raw failures"
        elif total_failures <= _RAW_FAILURE_WARN_COUNT:
            severity = HealthSeverity.WARNING
            message = (
                f"{total_failures} raw failures ({quarantined} quarantined, {maintenance} maintenance){op_hint}"
                if maintenance
                else f"{total_failures} raw failures ({quarantined} quarantined)"
            )
        elif total_failures <= _RAW_FAILURE_ERROR_COUNT:
            severity = HealthSeverity.ERROR
            message = (
                f"{total_failures} raw failures ({quarantined} quarantined, {maintenance} maintenance){op_hint}"
                if maintenance
                else f"{total_failures} raw failures ({quarantined} quarantined)"
            )
        else:
            severity = HealthSeverity.CRITICAL
            base = (
                f"{total_failures} raw failures ({quarantined} quarantined, {maintenance} maintenance){op_hint}"
                if maintenance
                else f"{total_failures} raw failures ({quarantined} quarantined)"
            )
            message = f"{base} — investigation needed"
        return HealthAlert(
            check_name="raw_failures",
            tier=HealthTier.MEDIUM,
            severity=severity,
            message=message,
            checked_at=now,
            consecutive_failures=_record_failure("raw_failures", severity == HealthSeverity.OK),
        )
    except Exception as exc:
        return HealthAlert(
            check_name="raw_failures",
            tier=HealthTier.MEDIUM,
            severity=HealthSeverity.ERROR,
            message=f"raw failure check failed: {exc}",
            checked_at=now,
            consecutive_failures=_record_failure("raw_failures", False),
        )


def _check_stale_ingest_attempts_medium() -> HealthAlert:
    """Check for stale live ingest attempts."""
    from polylogue.daemon.status import _live_ingest_attempt_summary_info

    now = datetime.now(UTC).isoformat()
    try:
        summary = _live_ingest_attempt_summary_info()
        if summary.running_count == 0:
            severity = HealthSeverity.OK
            message = "no running ingest attempts"
        elif summary.stale_running_count > 0:
            severity = HealthSeverity.WARNING
            message = f"{summary.stale_running_count} of {summary.running_count} running attempts are stale"
        else:
            severity = HealthSeverity.OK
            message = f"{summary.running_count} running attempts, none stale"
        return HealthAlert(
            check_name="stale_ingest_attempts",
            tier=HealthTier.MEDIUM,
            severity=severity,
            message=message,
            checked_at=now,
            consecutive_failures=_record_failure("stale_ingest_attempts", severity == HealthSeverity.OK),
        )
    except Exception as exc:
        return HealthAlert(
            check_name="stale_ingest_attempts",
            tier=HealthTier.MEDIUM,
            severity=HealthSeverity.ERROR,
            message=f"ingest attempt check failed: {exc}",
            checked_at=now,
            consecutive_failures=_record_failure("stale_ingest_attempts", False),
        )


def _check_insight_freshness_medium() -> HealthAlert:
    """Check session profile coverage against total sessions.

    A gap indicates insight materialization is incomplete — profiles,
    work events, phases, and threads may be stale or missing.
    """
    now = datetime.now(UTC).isoformat()
    try:
        from polylogue.daemon.status import _insight_freshness_info

        info = _insight_freshness_info()
        total_sessions = info.get("total_sessions", 0)
        total = total_sessions if isinstance(total_sessions, int) else 0
        profiled = info.get("sessions_with_profiles", 0)
        with_profiles = profiled if isinstance(profiled, int) else 0
        gap = total - with_profiles

        if total == 0:
            severity = HealthSeverity.OK
            message = "no sessions to profile"
        elif gap == 0:
            severity = HealthSeverity.OK
            message = f"{with_profiles} sessions profiled"
        elif gap <= total * 0.1:
            severity = HealthSeverity.WARNING
            gap_pct = 100 * gap / total
            message = f"{gap} of {total} sessions missing profiles ({gap_pct:.1f}%)"
        else:
            severity = HealthSeverity.ERROR
            gap_pct = 100 * gap / total
            message = f"{gap} of {total} sessions missing profiles ({gap_pct:.1f}%) — convergence may be stalled"

        return HealthAlert(
            check_name="insight_freshness",
            tier=HealthTier.MEDIUM,
            severity=severity,
            message=message,
            checked_at=now,
            consecutive_failures=_record_failure("insight_freshness", severity == HealthSeverity.OK),
        )
    except Exception as exc:
        return HealthAlert(
            check_name="insight_freshness",
            tier=HealthTier.MEDIUM,
            severity=HealthSeverity.ERROR,
            message=f"insight freshness check failed: {exc}",
            checked_at=now,
            consecutive_failures=_record_failure("insight_freshness", False),
        )


def _check_repeated_stage_failures_medium() -> HealthAlert:
    """Check for repeated stage failures in live ingest attempts.

    Looks at recent ingest attempts and flags when a non-trivial portion
    have failed — indicating a persistent stage-level problem rather than
    a transient source-file issue.
    """
    now = datetime.now(UTC).isoformat()
    dbf = _active_health_db_path()
    ops_info = _archive_repeated_stage_failure_info(dbf.with_name("ops.db"))
    if ops_info is not None and (ops_info[0] > 0 or not dbf.exists()):
        total_recent, failed_recent, error_row = ops_info
        return _repeated_stage_failure_alert(now, total_recent, failed_recent, error_row)
    if not dbf.exists():
        return HealthAlert(
            check_name="repeated_stage_failures",
            tier=HealthTier.MEDIUM,
            severity=HealthSeverity.ERROR,
            message="database not found",
            checked_at=now,
            consecutive_failures=_record_failure("repeated_stage_failures", False),
        )
    try:
        conn = sqlite3.connect(str(dbf))
        try:
            has_table = bool(
                conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='live_ingest_attempt'").fetchone()
            )
            if not has_table:
                return HealthAlert(
                    check_name="repeated_stage_failures",
                    tier=HealthTier.MEDIUM,
                    severity=HealthSeverity.OK,
                    message="no ingest attempt history",
                    checked_at=now,
                    consecutive_failures=_record_failure("repeated_stage_failures", True),
                )

            total_recent = conn.execute(
                "SELECT COUNT(*) FROM (SELECT 1 FROM live_ingest_attempt ORDER BY started_at DESC LIMIT 20)"
            ).fetchone()[0]
            failed_recent = conn.execute(
                "SELECT COUNT(*) FROM ("
                "SELECT 1 FROM live_ingest_attempt "
                "WHERE status = 'failed' "
                "ORDER BY started_at DESC LIMIT 20"
                ")"
            ).fetchone()[0]

            error_row = conn.execute(
                "SELECT phase, error FROM live_ingest_attempt "
                "WHERE status = 'failed' AND error IS NOT NULL "
                "ORDER BY started_at DESC LIMIT 1"
            ).fetchone()

            return _repeated_stage_failure_alert(now, total_recent, failed_recent, error_row)
        finally:
            conn.close()
    except Exception as exc:
        return HealthAlert(
            check_name="repeated_stage_failures",
            tier=HealthTier.MEDIUM,
            severity=HealthSeverity.ERROR,
            message=f"stage failure check failed: {exc}",
            checked_at=now,
            consecutive_failures=_record_failure("repeated_stage_failures", False),
        )


def _archive_repeated_stage_failure_info(
    ops_db: Path,
) -> tuple[int, int, sqlite3.Row | tuple[object, ...] | None] | None:
    if not ops_db.exists():
        return None
    try:
        conn = sqlite3.connect(str(ops_db))
        try:
            has_table = bool(
                conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='ingest_attempts'").fetchone()
            )
            if not has_table:
                return None
            total_recent = int(
                conn.execute(
                    "SELECT COUNT(*) FROM (SELECT 1 FROM ingest_attempts ORDER BY started_at_ms DESC LIMIT 20)"
                ).fetchone()[0]
                or 0
            )
            failed_recent = int(
                conn.execute(
                    "SELECT COUNT(*) FROM ("
                    "SELECT 1 FROM ingest_attempts "
                    "WHERE status = 'failed' "
                    "ORDER BY started_at_ms DESC LIMIT 20"
                    ")"
                ).fetchone()[0]
                or 0
            )
            error_row = conn.execute(
                "SELECT phase, error_message FROM ingest_attempts "
                "WHERE status = 'failed' AND error_message IS NOT NULL "
                "ORDER BY started_at_ms DESC LIMIT 1"
            ).fetchone()
            return total_recent, failed_recent, error_row
        finally:
            conn.close()
    except sqlite3.Error:
        return None


def _repeated_stage_failure_alert(
    now: str,
    total_recent: int,
    failed_recent: int,
    error_row: sqlite3.Row | tuple[object, ...] | None,
) -> HealthAlert:
    if total_recent == 0:
        severity = HealthSeverity.OK
        message = "no recent ingest attempts"
    elif failed_recent == 0:
        severity = HealthSeverity.OK
        message = f"no failures in last {total_recent} attempts"
    elif failed_recent <= 2:
        severity = HealthSeverity.WARNING
        message = f"{failed_recent}/{total_recent} recent attempts failed"
    else:
        error_hint = ""
        if error_row:
            phase = error_row[0] or "unknown"
            error_text = error_row[1] or ""
            error_hint = f" (phase={phase}: {str(error_text)[:80]})"
        severity = HealthSeverity.ERROR
        message = f"{failed_recent}/{total_recent} recent attempts failed{error_hint}"

    return HealthAlert(
        check_name="repeated_stage_failures",
        tier=HealthTier.MEDIUM,
        severity=severity,
        message=message,
        checked_at=now,
        consecutive_failures=_record_failure("repeated_stage_failures", severity == HealthSeverity.OK),
    )


def _check_convergence_debt_medium() -> list[HealthAlert]:
    """Per-source-family convergence-debt threshold alerts (#1226).

    Aggregates ``live_convergence_debt`` by inferred source family and
    raises one alert per family that crosses its configured warning or
    error threshold. Returns an empty list when there is no debt and no
    pending dedup state — keeps the periodic health loop quiet on a
    healthy archive.

    Returns a list rather than a single :class:`HealthAlert` because
    multiple families may breach at once; the calling tier aggregator
    flattens the result into the overall alert stream.
    """
    now = datetime.now(UTC).isoformat()
    try:
        from polylogue.config import load_polylogue_config
        from polylogue.daemon.convergence_debt_alert import (
            evaluate_convergence_debt,
            get_default_dedup_state,
            load_thresholds_from_config,
        )
        from polylogue.daemon.convergence_debt_status import convergence_debt_summary_info

        cfg = load_polylogue_config()
        thresholds = load_thresholds_from_config(cfg)
        summary = convergence_debt_summary_info(_active_health_db_path())
        alerts = evaluate_convergence_debt(
            summary,
            thresholds=thresholds,
            state=get_default_dedup_state(),
        )
        # Record one umbrella failure counter so the health tier summary
        # carries a stable per-cycle signal even when the per-family
        # alerts are deduped away.
        is_ok = not any(a.severity != HealthSeverity.OK for a in alerts)
        _record_failure("convergence_debt", is_ok)
        return alerts
    except Exception as exc:
        return [
            HealthAlert(
                check_name="convergence_debt",
                tier=HealthTier.MEDIUM,
                severity=HealthSeverity.ERROR,
                message=f"convergence debt check failed: {exc}",
                checked_at=now,
                consecutive_failures=_record_failure("convergence_debt", False),
            )
        ]


def _check_cursor_lag_medium() -> list[HealthAlert]:
    """Per-source-family cursor-lag SLO + anomaly-band alerts (#1232, #1349).

    Two layered checks against one shared lag projection:

    1. **Static ladder** (#1232) — per-family hardcoded
       warning/error/critical thresholds from
       :mod:`polylogue.daemon.cursor_lag_alert`. This is the hard
       page-me-now floor.
    2. **Anomaly band** (#1349) — per-family "current lag is N× rolling
       p95 baseline" thresholds from
       :mod:`polylogue.daemon.cursor_lag_anomaly`. Additive: alerts use a
       distinct ``check_name`` and a separate dedup state, so this layer
       cannot suppress static escalations. Has no CRITICAL tier on
       purpose — only the static ladder can page critical.

    Side effects of the anomaly layer:

    - Records one ``live_cursor_lag_sample`` row per stuck family per
      tick (the periodic-loop cadence is determined by
      ``health_check_interval_s``).
    - GCs samples older than ``retention_days``.

    Both side effects are no-ops when no family is stuck or when the
    samples table cannot be opened. The static-ladder layer keeps working
    in either case.

    Returns a list rather than a single :class:`HealthAlert` because
    multiple families may breach simultaneously; the calling tier
    aggregator flattens the result into the overall alert stream.
    """
    now = datetime.now(UTC).isoformat()
    try:
        from polylogue.config import load_polylogue_config
        from polylogue.daemon.cursor_lag_alert import (
            evaluate_cursor_lag,
            get_default_dedup_state,
            load_thresholds_from_config,
        )
        from polylogue.daemon.cursor_lag_status import cursor_lag_summary_info

        cfg = load_polylogue_config()
        thresholds = load_thresholds_from_config(cfg)
        summary = cursor_lag_summary_info(_active_health_db_path())
        static_alerts = evaluate_cursor_lag(
            summary,
            thresholds=thresholds,
            state=get_default_dedup_state(),
        )
        is_ok = not any(a.severity != HealthSeverity.OK for a in static_alerts)
        _record_failure("cursor_lag", is_ok)

        anomaly_alerts = _check_cursor_lag_anomaly_layer(summary, cfg)

        return [*static_alerts, *anomaly_alerts]
    except Exception as exc:
        return [
            HealthAlert(
                check_name="cursor_lag",
                tier=HealthTier.MEDIUM,
                severity=HealthSeverity.ERROR,
                message=f"cursor lag check failed: {exc}",
                checked_at=now,
                consecutive_failures=_record_failure("cursor_lag", False),
            )
        ]


def _check_cursor_lag_anomaly_layer(
    summary: object,
    cfg: PolylogueConfig | None,
) -> list[HealthAlert]:
    """Run the anomaly-band layer on the shared lag summary.

    Failure-isolated from the static-ladder layer: any exception here is
    converted into a single ``cursor_lag_anomaly`` ERROR alert so the
    static-ladder alerts still fire on the same tick.
    """
    now = datetime.now(UTC).isoformat()
    try:
        from polylogue.daemon.cursor_lag_anomaly import (
            evaluate_cursor_lag_anomaly,
            load_anomaly_thresholds_from_config,
        )
        from polylogue.daemon.cursor_lag_anomaly import (
            get_default_dedup_state as get_anomaly_dedup_state,
        )
        from polylogue.daemon.cursor_lag_baseline import (
            gc_cursor_lag_samples,
            load_family_baselines,
            record_cursor_lag_sample,
        )
        from polylogue.daemon.cursor_lag_status import CursorLagSummary

        if not isinstance(summary, CursorLagSummary):
            return []

        anomaly_thresholds = load_anomaly_thresholds_from_config(cfg)
        dbf = _active_health_db_path()

        # Load the baseline BEFORE recording the current moment's sample.
        # This is load-bearing: if we wrote first, the current spike would
        # be folded into its own p95 and the multiplier would collapse
        # toward 1.0. Reading first means the alert evaluates against the
        # pre-spike rolling baseline, which is the operator-visible
        # "this lag is N× normal" semantics the SLO promises.
        families_to_check = sorted({fs.family for fs in summary.family_summaries})
        baselines = load_family_baselines(
            dbf,
            families_to_check,
            window_days=anomaly_thresholds.baseline_window_days,
            min_samples=anomaly_thresholds.baseline_min_samples,
        )

        # Record the current moment's sample after baseline read (no-op if
        # no stuck files). Failure here is non-fatal: anomaly evaluation
        # has already produced its baseline view.
        try:
            record_cursor_lag_sample(dbf, summary)
        except Exception:
            logger.warning("cursor_lag_anomaly: sample record failed", exc_info=True)

        # GC old samples. Retention covers at least 2x the baseline window
        # so a paused daemon does not starve its own baseline on restart.
        try:
            retention = max(
                anomaly_thresholds.retention_days,
                anomaly_thresholds.baseline_window_days * 2,
            )
            gc_cursor_lag_samples(dbf, retention_days=retention)
        except Exception:
            logger.warning("cursor_lag_anomaly: sample GC failed", exc_info=True)

        if not anomaly_thresholds.enabled:
            _record_failure("cursor_lag_anomaly", True)
            return []

        alerts = evaluate_cursor_lag_anomaly(
            summary,
            baselines,
            thresholds=anomaly_thresholds,
            state=get_anomaly_dedup_state(),
        )
        is_ok = not any(a.severity != HealthSeverity.OK for a in alerts)
        _record_failure("cursor_lag_anomaly", is_ok)
        return alerts
    except Exception as exc:
        return [
            HealthAlert(
                check_name="cursor_lag_anomaly",
                tier=HealthTier.MEDIUM,
                severity=HealthSeverity.ERROR,
                message=f"cursor lag anomaly check failed: {exc}",
                checked_at=now,
                consecutive_failures=_record_failure("cursor_lag_anomaly", False),
            )
        ]


def _run_medium_checks() -> list[HealthAlert]:
    return [
        _check_fts_readiness_medium(),
        _check_raw_failures_medium(),
        _check_stale_ingest_attempts_medium(),
        _check_insight_freshness_medium(),
        _check_repeated_stage_failures_medium(),
        *_check_convergence_debt_medium(),
        *_check_cursor_lag_medium(),
    ]


# ---------------------------------------------------------------------------
# Expensive checks (potentially > 10s)
# ---------------------------------------------------------------------------


def _check_db_integrity_expensive() -> HealthAlert:
    """Run PRAGMA integrity_check on the main database."""
    now = datetime.now(UTC).isoformat()
    dbf = _active_health_db_path()
    if not dbf.exists():
        return HealthAlert(
            check_name="db_integrity",
            tier=HealthTier.EXPENSIVE,
            severity=HealthSeverity.ERROR,
            message="database not found",
            checked_at=now,
            consecutive_failures=_record_failure("db_integrity", False),
        )
    try:
        conn = sqlite3.connect(str(dbf))
        try:
            results = conn.execute("PRAGMA integrity_check").fetchall()
            ok = len(results) == 1 and results[0][0] == "ok"
            if ok:
                severity = HealthSeverity.OK
                message = "database integrity ok"
            else:
                severity = HealthSeverity.CRITICAL
                message = f"database integrity errors: {'; '.join(str(r[0]) for r in results[:5])}"
            return HealthAlert(
                check_name="db_integrity",
                tier=HealthTier.EXPENSIVE,
                severity=severity,
                message=message,
                checked_at=now,
                consecutive_failures=_record_failure("db_integrity", ok),
            )
        finally:
            conn.close()
    except Exception as exc:
        return HealthAlert(
            check_name="db_integrity",
            tier=HealthTier.EXPENSIVE,
            severity=HealthSeverity.ERROR,
            message=f"integrity check failed: {exc}",
            checked_at=now,
            consecutive_failures=_record_failure("db_integrity", False),
        )


def _check_blob_integrity_expensive() -> list[HealthAlert]:
    """Sample-check blob store integrity through the shared read-only scanner."""
    from polylogue.config import load_polylogue_config
    from polylogue.daemon.blob_integrity_alerts import blob_integrity_alerts_from_report
    from polylogue.storage.blob_integrity import scan_blob_integrity

    now = datetime.now(UTC).isoformat()
    try:
        cfg = load_polylogue_config()
        report = scan_blob_integrity(
            _active_health_db_path(),
            full=False,
            sample_size=max(1, cfg.health_blob_integrity_sample_size),
        )
        return blob_integrity_alerts_from_report(report, now, _record_failure)
    except Exception as exc:
        return [
            HealthAlert(
                check_name="blob_integrity",
                tier=HealthTier.EXPENSIVE,
                severity=HealthSeverity.ERROR,
                message=f"blob integrity check failed: {exc}",
                checked_at=now,
                consecutive_failures=_record_failure("blob_integrity", False),
            )
        ]


def _check_embedding_coverage_expensive() -> HealthAlert:
    """Check embedding coverage when embedding is enabled.

    Only raises non-OK when embedding is configured but coverage is low
    or there are embedding failures. Disabled embedding is OK.
    """
    from polylogue.config import load_polylogue_config

    now = datetime.now(UTC).isoformat()
    try:
        cfg = load_polylogue_config()
        enabled = bool(cfg.embedding_enabled) and cfg.voyage_api_key is not None

        if not enabled:
            return HealthAlert(
                check_name="embedding_coverage",
                tier=HealthTier.EXPENSIVE,
                severity=HealthSeverity.OK,
                message="embedding disabled",
                checked_at=now,
                consecutive_failures=_record_failure("embedding_coverage", True),
            )

        info = embedding_readiness_info(_active_health_db_path())
        coverage = info.get("embedding_coverage_percent", 0.0)
        cov_pct = float(coverage) if isinstance(coverage, (int, float)) and not isinstance(coverage, bool) else 0.0
        failure_count = info.get("embedding_failure_count", 0)
        failures = failure_count if isinstance(failure_count, int) else 0

        if failures > 0 and cov_pct < 50:
            severity = HealthSeverity.ERROR
            message = f"embedding coverage {cov_pct:.1f}% with {failures} failures"
        elif failures > 0:
            severity = HealthSeverity.WARNING
            message = f"embedding coverage {cov_pct:.1f}%, {failures} failures"
        elif cov_pct < 10:
            severity = HealthSeverity.WARNING
            message = f"embedding coverage low: {cov_pct:.1f}%"
        else:
            severity = HealthSeverity.OK
            message = f"embedding coverage {cov_pct:.1f}%"

        return HealthAlert(
            check_name="embedding_coverage",
            tier=HealthTier.EXPENSIVE,
            severity=severity,
            message=message,
            checked_at=now,
            consecutive_failures=_record_failure("embedding_coverage", severity == HealthSeverity.OK),
        )
    except Exception as exc:
        return HealthAlert(
            check_name="embedding_coverage",
            tier=HealthTier.EXPENSIVE,
            severity=HealthSeverity.ERROR,
            message=f"embedding coverage check failed: {exc}",
            checked_at=now,
            consecutive_failures=_record_failure("embedding_coverage", False),
        )


def _run_expensive_checks() -> list[HealthAlert]:
    alerts = [_check_db_integrity_expensive()]
    alerts.extend(_check_blob_integrity_expensive())
    alerts.append(_check_embedding_coverage_expensive())
    return alerts


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def check_health(*, tiers: set[HealthTier] | None = None) -> DaemonHealth:
    """Run tiered health checks and return aggregated ``DaemonHealth``.

    Args:
        tiers: Which tiers to run. If None, runs FAST only.
               Pass ``{HealthTier.FAST, HealthTier.MEDIUM, HealthTier.EXPENSIVE}``
               for a full check.

    Returns:
        ``DaemonHealth`` with alerts and tier summary.
    """
    if tiers is None:
        tiers = {HealthTier.FAST}

    alerts: list[HealthAlert] = []

    if HealthTier.FAST in tiers:
        alerts.extend(_run_fast_checks())
    if HealthTier.MEDIUM in tiers:
        alerts.extend(_run_medium_checks())
    if HealthTier.EXPENSIVE in tiers:
        alerts.extend(_run_expensive_checks())

    overall = _compute_overall_health(alerts)

    # Build tier summary.
    tier_summary: dict[str, dict[str, int]] = {}
    for alert in alerts:
        tier_key = alert.tier.value
        if tier_key not in tier_summary:
            tier_summary[tier_key] = {"ok": 0, "info": 0, "warning": 0, "error": 0, "critical": 0}
        tier_summary[tier_key][alert.severity.value] += 1

    return DaemonHealth(
        overall_status=overall,
        checked_at=datetime.now(UTC).isoformat(),
        alerts=alerts,
        tier_summary=tier_summary,
    )


def format_health_lines(health: DaemonHealth) -> list[str]:
    """Render health status as plain-text lines."""
    lines: list[str] = []
    status_icon = _severity_icon(health.overall_status)
    lines.append(f"Daemon health: {health.overall_status.value} {status_icon}")

    for tier in ("fast", "medium", "expensive"):
        bucket = [a for a in health.alerts if a.tier.value == tier]
        if not bucket:
            continue
        lines.append(f"  [{tier}]")
        for alert in bucket:
            icon = _severity_icon(alert.severity)
            extra = f" (x{alert.consecutive_failures})" if alert.consecutive_failures > 1 else ""
            lines.append(f"    {alert.check_name}: {alert.severity.value} {icon}{extra} — {alert.message}")

    return lines


# ``INFO`` ranks alongside ``OK`` so a known-transient bulk-stage signal
# does not escalate the daemon-level overall status (#1613).
_OVERALL_SEVERITY_RANK = {
    HealthSeverity.OK: 0,
    HealthSeverity.INFO: 0,
    HealthSeverity.WARNING: 1,
    HealthSeverity.ERROR: 2,
    HealthSeverity.CRITICAL: 3,
}


def _compute_overall_health(alerts: list[HealthAlert]) -> HealthSeverity:
    """Return the worst severity among ``alerts`` (INFO and OK tie at the bottom)."""
    overall = HealthSeverity.OK
    for alert in alerts:
        if _OVERALL_SEVERITY_RANK[alert.severity] > _OVERALL_SEVERITY_RANK[overall]:
            overall = alert.severity
    return overall


def _severity_icon(severity: HealthSeverity) -> str:
    if severity == HealthSeverity.OK:
        return "[OK]"
    if severity == HealthSeverity.INFO:
        return "[INFO]"
    if severity == HealthSeverity.WARNING:
        return "[WARN]"
    if severity == HealthSeverity.ERROR:
        return "[ERROR]"
    if severity == HealthSeverity.CRITICAL:
        return "[CRIT]"
    return ""


__all__ = [
    "DaemonHealth",
    "HealthAlert",
    "HealthSeverity",
    "HealthTier",
    "_EXPECTED_FTS_TRIGGERS",
    "_check_fts_trigger_drift_fast",
    "_check_schema_version_fast",
    "_find_missing_fts_triggers",
    "check_health",
    "format_health_lines",
]
