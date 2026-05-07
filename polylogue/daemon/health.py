"""Tiered daemon health checks with typed alerts and severity escalation.

Health checks are grouped into three tiers by cost:
- FAST: sub-1s checks (liveness, disk space, WAL size)
- MEDIUM: sub-10s queries (FTS readiness, insight freshness, stale attempts, failures)
- EXPENSIVE: longer-running checks (DB integrity, blob integrity)

Each check produces a ``HealthAlert`` with severity, message, and checked_at.
Alerts accrue a ``consecutive_failures`` counter that carries forward across
check cycles so operators can detect persistent conditions.
"""

from __future__ import annotations

import os
import sqlite3
from datetime import UTC, datetime
from enum import Enum

from pydantic import BaseModel, Field

from polylogue.daemon.status import _raw_failure_info
from polylogue.logging import get_logger
from polylogue.paths import archive_root, db_path

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Typed enums
# ---------------------------------------------------------------------------


class HealthSeverity(str, Enum):
    """Alert severity with implied escalation."""

    OK = "ok"
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
    dbf = db_path()
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


def _run_fast_checks() -> list[HealthAlert]:
    return [
        _check_daemon_liveness_fast(),
        _check_disk_space_fast(),
        _check_wal_size_fast(),
    ]


# ---------------------------------------------------------------------------
# Medium checks (< 10s)
# ---------------------------------------------------------------------------


def _check_fts_readiness_medium() -> HealthAlert:
    """Check FTS tables exist and are populated."""
    now = datetime.now(UTC).isoformat()
    dbf = db_path()
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
            has_messages_fts = bool(
                conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='messages_fts'").fetchone()
            )
            has_action_fts = bool(
                conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='action_events_fts'").fetchone()
            )
            total = conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
            fts_count = conn.execute("SELECT COUNT(*) FROM messages_fts").fetchone()[0] if has_messages_fts else 0
            gap = total - fts_count

            if not has_messages_fts and not has_action_fts:
                severity = HealthSeverity.WARNING
                message = "FTS tables missing"
            elif gap > 0:
                gap_pct = 100 * gap / total if total else 0
                severity = HealthSeverity.WARNING if gap_pct < 10 else HealthSeverity.ERROR
                message = f"FTS gap: {gap} of {total} messages unindexed ({gap_pct:.1f}%)"
            else:
                severity = HealthSeverity.OK
                message = "FTS up to date"
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
    """Check raw conversation parse/validation failure counts."""
    now = datetime.now(UTC).isoformat()
    try:
        info = _raw_failure_info()
        parse = info.get("parse_failures", 0) if isinstance(info.get("parse_failures"), int) else 0
        validation = info.get("validation_failures", 0) if isinstance(info.get("validation_failures"), int) else 0
        quarantined = info.get("quarantined", 0) if isinstance(info.get("quarantined"), int) else 0
        total_failures = parse + validation

        if total_failures == 0:
            severity = HealthSeverity.OK
            message = "no raw failures"
        elif total_failures <= _RAW_FAILURE_WARN_COUNT:
            severity = HealthSeverity.WARNING
            message = f"{total_failures} raw failures ({quarantined} quarantined)"
        elif total_failures <= _RAW_FAILURE_ERROR_COUNT:
            severity = HealthSeverity.ERROR
            message = f"{total_failures} raw failures ({quarantined} quarantined)"
        else:
            severity = HealthSeverity.CRITICAL
            message = f"{total_failures} raw failures ({quarantined} quarantined) — investigation needed"
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


def _run_medium_checks() -> list[HealthAlert]:
    return [
        _check_fts_readiness_medium(),
        _check_raw_failures_medium(),
        _check_stale_ingest_attempts_medium(),
    ]


# ---------------------------------------------------------------------------
# Expensive checks (potentially > 10s)
# ---------------------------------------------------------------------------


def _check_db_integrity_expensive() -> HealthAlert:
    """Run PRAGMA integrity_check on the main database."""
    now = datetime.now(UTC).isoformat()
    dbf = db_path()
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


def _run_expensive_checks() -> list[HealthAlert]:
    return [
        _check_db_integrity_expensive(),
    ]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def check_health(*, tiers: set[HealthTier] | None = None) -> DaemonHealth:
    """Run tiered health checks and return aggregated ``DaemonHealth``.

    Args:
        tiers: Which tiers to run. If None, runs FAST + MEDIUM only.
               Pass ``{HealthTier.FAST, HealthTier.MEDIUM, HealthTier.EXPENSIVE}``
               for a full check.

    Returns:
        ``DaemonHealth`` with alerts and tier summary.
    """
    if tiers is None:
        tiers = {HealthTier.FAST, HealthTier.MEDIUM}

    alerts: list[HealthAlert] = []

    if HealthTier.FAST in tiers:
        alerts.extend(_run_fast_checks())
    if HealthTier.MEDIUM in tiers:
        alerts.extend(_run_medium_checks())
    if HealthTier.EXPENSIVE in tiers:
        alerts.extend(_run_expensive_checks())

    # Compute overall severity (worst among all alerts).
    _severity_rank = {
        HealthSeverity.OK: 0,
        HealthSeverity.WARNING: 1,
        HealthSeverity.ERROR: 2,
        HealthSeverity.CRITICAL: 3,
    }
    overall = HealthSeverity.OK
    for alert in alerts:
        if _severity_rank[alert.severity] > _severity_rank[overall]:
            overall = alert.severity

    # Build tier summary.
    tier_summary: dict[str, dict[str, int]] = {}
    for alert in alerts:
        tier_key = alert.tier.value
        if tier_key not in tier_summary:
            tier_summary[tier_key] = {"ok": 0, "warning": 0, "error": 0, "critical": 0}
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


def _severity_icon(severity: HealthSeverity) -> str:
    if severity == HealthSeverity.OK:
        return "[OK]"
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
    "check_health",
    "format_health_lines",
]
