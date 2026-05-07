"""Shared daemon status payloads."""

from __future__ import annotations

import contextlib
import os
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, Field

from polylogue.browser_capture.receiver import BrowserCaptureReceiverConfig, receiver_status_payload
from polylogue.core.json import JSONDocument, json_document
from polylogue.daemon.health import DaemonHealth, check_health
from polylogue.paths import db_path
from polylogue.sources.live import WatchSource
from polylogue.sources.live.watcher import default_sources
from polylogue.storage.sqlite.connection_profile import open_readonly_connection

_LIVE_INGEST_ATTEMPT_STALE_AFTER_S = 180.0
_LIVE_CURSOR_FAILURE_SAMPLE_LIMIT = 50

# ---------------------------------------------------------------------------
# Typed sub-models
# ---------------------------------------------------------------------------


class ComponentState(BaseModel):
    watcher: str = "stopped"
    api: str = "stopped"
    browser_capture: str = "stopped"


class SourceLagItem(BaseModel):
    name: str
    root: str
    exists: bool
    file_count: int = 0


class IngestionThroughput(BaseModel):
    messages_per_second: float = 0.0
    files_per_second: float = 0.0


class FTSReadiness(BaseModel):
    messages_ready: bool = False
    action_events_ready: bool = False


class InsightFreshness(BaseModel):
    sessions_with_profiles: int = 0
    total_sessions: int = 0


class EmbeddingReadiness(BaseModel):
    embedding_enabled: bool = False
    embedding_model: str = ""
    embedding_dimension: int = 0
    embedding_pending_count: int = 0
    embedding_stale_count: int = 0
    embedding_coverage_percent: float = 0.0
    embedding_failure_count: int = 0
    embedding_estimated_cost_usd: float = 0.0


class LiveCursorFileState(BaseModel):
    source_path: str
    failure_count: int = 0
    next_retry_at: str | None = None
    excluded: bool = False
    retry_due: bool = False


class LiveCursorSummary(BaseModel):
    tracked_file_count: int = 0
    failed_file_count: int = 0
    excluded_file_count: int = 0
    retry_due_file_count: int = 0
    in_backoff_file_count: int = 0
    sampled_file_count: int = 0
    omitted_file_count: int = 0
    failing_files: list[LiveCursorFileState] = Field(default_factory=list)


class LiveIngestAttemptState(BaseModel):
    attempt_id: str
    started_at: str
    updated_at: str
    status: str
    phase: str
    queued_file_count: int = 0
    needed_file_count: int = 0
    succeeded_file_count: int = 0
    failed_file_count: int = 0
    input_bytes: int = 0
    source_payload_read_bytes: int = 0
    cursor_fingerprint_read_bytes: int = 0
    parse_time_s: float = 0.0
    convergence_time_s: float = 0.0
    current_source: str | None = None
    current_path: str | None = None
    error: str | None = None
    rss_current_mb: float | None = None
    rss_peak_self_mb: float | None = None
    rss_peak_children_mb: float | None = None
    cgroup_path: str | None = None
    cgroup_memory_current_mb: float | None = None
    cgroup_memory_peak_mb: float | None = None
    cgroup_memory_swap_current_mb: float | None = None
    worker_in_flight_count: int | None = None
    worker_completed_count: int | None = None
    worker_total_count: int | None = None
    updated_age_s: float | None = None
    stale: bool = False
    completed_at: str | None = None


class LiveIngestAttemptSummary(BaseModel):
    running_count: int = 0
    stale_running_count: int = 0
    recent: list[LiveIngestAttemptState] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# DaemonStatus — typed model consumed by all surfaces
# ---------------------------------------------------------------------------


class DaemonStatus(BaseModel):
    """Typed daemon status consumed by CLI, TUI, web, browser extension, MCP."""

    daemon_liveness: bool = False
    component_state: ComponentState = Field(default_factory=ComponentState)
    source_lag: list[SourceLagItem] = Field(default_factory=list)
    failing_files: list[str] = Field(default_factory=list)
    current_operations: list[dict[str, object]] = Field(default_factory=list)
    reset_queue: list[dict[str, object]] = Field(default_factory=list)
    ingestion_throughput: IngestionThroughput = Field(default_factory=IngestionThroughput)
    live_cursor: LiveCursorSummary = Field(default_factory=LiveCursorSummary)
    live_ingest_attempts: LiveIngestAttemptSummary = Field(default_factory=LiveIngestAttemptSummary)
    db_size_bytes: int = 0
    wal_size_bytes: int = 0
    blob_dir_size_bytes: int = 0
    disk_free_bytes: int = 0
    fts_readiness: FTSReadiness = Field(default_factory=FTSReadiness)
    insight_freshness: InsightFreshness = Field(default_factory=InsightFreshness)
    embedding_readiness: EmbeddingReadiness = Field(default_factory=EmbeddingReadiness)
    browser_capture_active: bool = False
    raw_parse_failures: int = 0
    raw_validation_failures: int = 0
    raw_quarantined: int = 0
    raw_failure_samples: list[dict[str, object]] = Field(default_factory=list)
    raw_detection_warnings: int = 0
    health: DaemonHealth = Field(default_factory=DaemonHealth)
    checked_at: str = ""


# ---------------------------------------------------------------------------
# Status helpers
# ---------------------------------------------------------------------------


def live_source_status_payload(sources: tuple[WatchSource, ...]) -> JSONDocument:
    """Return status for configured live-ingest roots."""
    items = [
        {
            "name": source.name,
            "root": str(source.root),
            "exists": source.exists(),
        }
        for source in sources
    ]
    existing = sum(1 for item in items if item["exists"])
    return json_document(
        {
            "ok": True,
            "source_count": len(items),
            "existing_source_count": existing,
            "sources": items,
        }
    )


def browser_capture_status_payload(spool_path: Path | None = None) -> JSONDocument:
    """Return status for the browser-capture receiver component."""
    cfg_default = BrowserCaptureReceiverConfig.default()
    if spool_path is not None:
        config = BrowserCaptureReceiverConfig(
            spool_path=spool_path,
            allowed_origins=cfg_default.allowed_origins,
            allow_remote=cfg_default.allow_remote,
            auth_token=cfg_default.auth_token,
        )
    else:
        config = cfg_default
    return json_document(receiver_status_payload(config))


def _db_size_info() -> dict[str, object]:
    dbf = db_path()
    info: dict[str, object] = {"db_path": str(dbf)}
    if dbf.exists():
        info["db_size_bytes"] = dbf.stat().st_size
        wal = dbf.with_suffix(".db-wal")
        if wal.exists():
            info["wal_size_bytes"] = wal.stat().st_size
        try:
            st = os.statvfs(str(dbf.parent))
            info["disk_free_bytes"] = st.f_frsize * st.f_bavail
        except OSError:
            pass
    return info


def _blob_size_info() -> int:
    # Status must remain responsive while convergence is reading or writing the
    # archive. A recursive blob-store walk is proportional to archive size and
    # can make `polylogued status` look hung on production archives.
    return 0


def _fts_readiness_info() -> dict[str, bool]:
    """Check FTS table presence through a bounded read-only probe."""
    dbf = db_path()
    if not dbf.exists():
        return {"messages_ready": False, "action_events_ready": False}
    try:
        conn = open_readonly_connection(dbf)
        try:
            tables = {
                row[0]
                for row in conn.execute(
                    """
                    SELECT name
                    FROM sqlite_master
                    WHERE type = 'table'
                      AND name IN ('messages_fts', 'action_events_fts')
                    """
                ).fetchall()
            }
        finally:
            conn.close()
        return {
            "messages_ready": "messages_fts" in tables,
            "action_events_ready": "action_events_fts" in tables,
        }
    except sqlite3.Error:
        return {"messages_ready": False, "action_events_ready": False}


def _insight_freshness_info() -> dict[str, object]:
    """Check insight materialization status through bounded SQL counts."""
    dbf = db_path()
    if not dbf.exists():
        return {"sessions_with_profiles": 0, "total_sessions": 0}
    try:
        conn = open_readonly_connection(dbf)
        try:
            tables = {
                row[0]
                for row in conn.execute(
                    """
                    SELECT name
                    FROM sqlite_master
                    WHERE type = 'table'
                      AND name IN ('conversations', 'session_profiles')
                    """
                ).fetchall()
            }
            total_sessions = 0
            sessions_with_profiles = 0
            if "conversations" in tables:
                total_sessions = int(conn.execute("SELECT COUNT(*) FROM conversations").fetchone()[0] or 0)
            if "session_profiles" in tables:
                sessions_with_profiles = int(conn.execute("SELECT COUNT(*) FROM session_profiles").fetchone()[0] or 0)
        finally:
            conn.close()
        return {
            "sessions_with_profiles": sessions_with_profiles,
            "total_sessions": total_sessions,
        }
    except sqlite3.Error:
        return {"sessions_with_profiles": 0, "total_sessions": 0}


def _raw_failure_info() -> dict[str, object]:
    """Query raw_conversations for parse/validation failure counts and bounded samples."""
    dbf = db_path()
    if not dbf.exists():
        return {"parse_failures": 0, "validation_failures": 0, "quarantined": 0, "samples": []}

    try:
        conn = open_readonly_connection(dbf)
        try:
            tables = {
                row[0]
                for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name='raw_conversations'"
                ).fetchall()
            }
            if "raw_conversations" not in tables:
                return {
                    "parse_failures": 0,
                    "validation_failures": 0,
                    "quarantined": 0,
                    "detection_warnings": 0,
                    "samples": [],
                }

            parse_fail = int(
                conn.execute("SELECT COUNT(*) FROM raw_conversations WHERE parse_error IS NOT NULL").fetchone()[0] or 0
            )
            validation_fail = int(
                conn.execute("SELECT COUNT(*) FROM raw_conversations WHERE validation_status = 'FAILED'").fetchone()[0]
                or 0
            )
            quarantined = int(
                conn.execute(
                    "SELECT COUNT(*) FROM raw_conversations WHERE parsed_at IS NULL AND (parse_error IS NOT NULL OR validation_status = 'FAILED')"
                ).fetchone()[0]
                or 0
            )
            detection_warnings_count = 0
            try:
                with contextlib.suppress(sqlite3.OperationalError):
                    detection_warnings_count = int(
                        conn.execute(
                            "SELECT COUNT(*) FROM raw_conversations WHERE detection_warnings IS NOT NULL"
                        ).fetchone()[0]
                        or 0
                    )
            except Exception:
                pass
            # Bounded failure samples (most recent 50)
            samples: list[dict[str, object]] = []
            for row in conn.execute(
                "SELECT raw_id, source_path, parse_error, validation_status "
                "FROM raw_conversations "
                "WHERE parse_error IS NOT NULL OR validation_status = 'FAILED' "
                "ORDER BY acquired_at DESC LIMIT 50"
            ).fetchall():
                error_text = row["parse_error"] or ""
                error_class = error_text.split("\n")[0].strip()[:120] if error_text else None
                samples.append(
                    {
                        "raw_id": row["raw_id"],
                        "source_path": row["source_path"] or "",
                        "error_class": error_class,
                        "validation_status": row["validation_status"] or "",
                    }
                )
        finally:
            conn.close()
        return {
            "parse_failures": parse_fail,
            "validation_failures": validation_fail,
            "quarantined": quarantined,
            "detection_warnings": detection_warnings_count,
            "samples": samples,
        }
    except sqlite3.Error:
        return {"parse_failures": 0, "validation_failures": 0, "quarantined": 0, "detection_warnings": 0, "samples": []}


def _safe_list_of_dicts(value: object) -> list[dict[str, object]]:
    if isinstance(value, list):
        return [item for item in value if isinstance(item, dict)]
    return []


def _safe_int(value: object) -> int:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return int(value)
    return 0


def _required_str(value: object) -> str:
    return value if isinstance(value, str) else str(value)


def _optional_str(value: object) -> str | None:
    return value if isinstance(value, str) else None


def _row_int(value: object) -> int:
    if isinstance(value, bool):
        return 0
    if isinstance(value, int | float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return 0
    return 0


def _row_float(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def _safe_float(value: object, *, default: float = 0.0) -> float:
    """Coerce value to float, returning default on failure."""
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return default
    return default


def _failing_files_info() -> list[str]:
    """Return live-source files currently marked failed or excluded."""
    return [item.source_path for item in _live_cursor_summary_info().failing_files]


def _live_cursor_summary_info() -> LiveCursorSummary:
    """Return live cursor backlog/failure state without source-tree scans."""
    dbf = db_path()
    if not dbf.exists():
        return LiveCursorSummary()
    try:
        conn = open_readonly_connection(dbf)
        try:
            has_table = conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'live_cursor'"
            ).fetchone()
            if has_table is None:
                return LiveCursorSummary()
            tracked_file_count = int(conn.execute("SELECT COUNT(*) FROM live_cursor").fetchone()[0])
            failed_file_count = int(
                conn.execute("SELECT COUNT(*) FROM live_cursor WHERE failure_count > 0").fetchone()[0]
            )
            excluded_file_count = int(conn.execute("SELECT COUNT(*) FROM live_cursor WHERE excluded = 1").fetchone()[0])
            attention_file_count = int(
                conn.execute("SELECT COUNT(*) FROM live_cursor WHERE failure_count > 0 OR excluded = 1").fetchone()[0]
            )
            rows = conn.execute(
                """
                SELECT source_path, failure_count, next_retry_at, excluded
                FROM live_cursor
                WHERE failure_count > 0 OR excluded = 1
                ORDER BY source_path
                LIMIT ?
                """,
                (_LIVE_CURSOR_FAILURE_SAMPLE_LIMIT,),
            ).fetchall()
            retry_rows = conn.execute(
                """
                SELECT next_retry_at
                FROM live_cursor
                WHERE failure_count > 0
                """
            ).fetchall()
        finally:
            conn.close()
    except sqlite3.Error:
        return LiveCursorSummary()

    now = datetime.now(UTC)
    failing_files: list[LiveCursorFileState] = []
    retry_due_file_count = 0
    for row in retry_rows:
        if _retry_due(row[0], now=now):
            retry_due_file_count += 1
    in_backoff_file_count = max(0, failed_file_count - retry_due_file_count)
    for row in rows:
        failure_count = _row_int(row[1])
        excluded = bool(row[3])
        retry_due = failure_count > 0 and _retry_due(row[2], now=now)
        failing_files.append(
            LiveCursorFileState(
                source_path=str(row[0]),
                failure_count=failure_count,
                next_retry_at=row[2],
                excluded=excluded,
                retry_due=retry_due,
            )
        )

    return LiveCursorSummary(
        tracked_file_count=tracked_file_count,
        failed_file_count=failed_file_count,
        excluded_file_count=excluded_file_count,
        retry_due_file_count=retry_due_file_count,
        in_backoff_file_count=in_backoff_file_count,
        sampled_file_count=len(failing_files),
        omitted_file_count=max(0, attention_file_count - len(failing_files)),
        failing_files=failing_files,
    )


def _live_ingest_attempt_summary_info() -> LiveIngestAttemptSummary:
    """Return recent durable live-ingest attempt snapshots."""
    dbf = db_path()
    if not dbf.exists():
        return LiveIngestAttemptSummary()
    try:
        conn = open_readonly_connection(dbf)
        try:
            has_table = conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'live_ingest_attempt'"
            ).fetchone()
            if has_table is None:
                return LiveIngestAttemptSummary()
            columns = {row[1] for row in conn.execute("PRAGMA table_info(live_ingest_attempt)")}
            cgroup_path_expr = "cgroup_path" if "cgroup_path" in columns else "NULL"
            cgroup_current_expr = "cgroup_memory_current_mb" if "cgroup_memory_current_mb" in columns else "NULL"
            cgroup_peak_expr = "cgroup_memory_peak_mb" if "cgroup_memory_peak_mb" in columns else "NULL"
            cgroup_swap_expr = "cgroup_memory_swap_current_mb" if "cgroup_memory_swap_current_mb" in columns else "NULL"
            worker_in_flight_expr = "worker_in_flight_count" if "worker_in_flight_count" in columns else "NULL"
            worker_completed_expr = "worker_completed_count" if "worker_completed_count" in columns else "NULL"
            worker_total_expr = "worker_total_count" if "worker_total_count" in columns else "NULL"
            rows = conn.execute(
                f"""
                SELECT
                    attempt_id,
                    started_at,
                    updated_at,
                    completed_at,
                    status,
                    phase,
                    queued_file_count,
                    needed_file_count,
                    succeeded_file_count,
                    failed_file_count,
                    input_bytes,
                    source_payload_read_bytes,
                    cursor_fingerprint_read_bytes,
                    parse_time_s,
                    convergence_time_s,
                    current_source,
                    current_path,
                    error,
                    rss_current_mb,
                    rss_peak_self_mb,
                    rss_peak_children_mb,
                    {cgroup_path_expr},
                    {cgroup_current_expr},
                    {cgroup_peak_expr},
                    {cgroup_swap_expr},
                    {worker_in_flight_expr},
                    {worker_completed_expr},
                    {worker_total_expr}
                FROM live_ingest_attempt
                ORDER BY updated_at DESC, started_at DESC
                LIMIT 5
                """
            ).fetchall()
            running_rows = conn.execute(
                "SELECT updated_at FROM live_ingest_attempt WHERE status = 'running'"
            ).fetchall()
        finally:
            conn.close()
    except sqlite3.Error:
        return LiveIngestAttemptSummary()

    now = datetime.now(UTC)
    recent_attempts = [_live_ingest_attempt_state_from_row(row, now=now) for row in rows]
    stale_running_count = sum(
        1
        for row in running_rows
        if (age_s := _attempt_updated_age_s(_required_str(row[0]), now=now)) is not None
        and age_s >= _LIVE_INGEST_ATTEMPT_STALE_AFTER_S
    )
    return LiveIngestAttemptSummary(
        running_count=len(running_rows),
        stale_running_count=stale_running_count,
        recent=recent_attempts,
    )


def _live_ingest_attempt_state_from_row(
    row: sqlite3.Row | tuple[object, ...], *, now: datetime
) -> LiveIngestAttemptState:
    updated_at = _required_str(row[2])
    updated_age_s = _attempt_updated_age_s(updated_at, now=now)
    stale = (
        _required_str(row[4]) == "running"
        and updated_age_s is not None
        and updated_age_s >= _LIVE_INGEST_ATTEMPT_STALE_AFTER_S
    )
    return LiveIngestAttemptState(
        attempt_id=_required_str(row[0]),
        started_at=_required_str(row[1]),
        updated_at=updated_at,
        completed_at=_optional_str(row[3]),
        status=_required_str(row[4]),
        phase=_required_str(row[5]),
        queued_file_count=_row_int(row[6]),
        needed_file_count=_row_int(row[7]),
        succeeded_file_count=_row_int(row[8]),
        failed_file_count=_row_int(row[9]),
        input_bytes=_row_int(row[10]),
        source_payload_read_bytes=_row_int(row[11]),
        cursor_fingerprint_read_bytes=_row_int(row[12]),
        parse_time_s=_row_float(row[13]) or 0.0,
        convergence_time_s=_row_float(row[14]) or 0.0,
        current_source=_optional_str(row[15]),
        current_path=_optional_str(row[16]),
        error=_optional_str(row[17]),
        rss_current_mb=_row_float(row[18]),
        rss_peak_self_mb=_row_float(row[19]),
        rss_peak_children_mb=_row_float(row[20]),
        cgroup_path=_optional_str(row[21]),
        cgroup_memory_current_mb=_row_float(row[22]),
        cgroup_memory_peak_mb=_row_float(row[23]),
        cgroup_memory_swap_current_mb=_row_float(row[24]),
        worker_in_flight_count=_row_int(row[25]) if len(row) > 25 else None,
        worker_completed_count=_row_int(row[26]) if len(row) > 26 else None,
        worker_total_count=_row_int(row[27]) if len(row) > 27 else None,
        updated_age_s=updated_age_s,
        stale=stale,
    )


def _attempt_updated_age_s(updated_at: str, *, now: datetime) -> float | None:
    try:
        updated = datetime.fromisoformat(updated_at)
    except ValueError:
        return None
    if updated.tzinfo is None:
        updated = updated.replace(tzinfo=UTC)
    return max(0.0, round((now - updated.astimezone(UTC)).total_seconds(), 3))


def _retry_due(next_retry_at: str | None, *, now: datetime) -> bool:
    if not next_retry_at:
        return True
    try:
        retry_at = datetime.fromisoformat(next_retry_at)
    except ValueError:
        return True
    if retry_at.tzinfo is None:
        retry_at = retry_at.replace(tzinfo=UTC)
    return retry_at <= now


def _embedding_readiness_info() -> dict[str, object]:
    """Query embedding tables for daemon status visibility.

    Returns empty/inactive defaults when tables don't exist or embedding is
    not configured. This is intentionally bounded — expensive retrieval-band
    computation only happens through the dedicated stats path.
    """
    from polylogue.config import load_polylogue_config
    from polylogue.storage.embeddings.support import optional_count_sync

    cfg = load_polylogue_config()
    enabled = bool(cfg.embedding_enabled) and cfg.voyage_api_key is not None
    model = cfg.embedding_model
    dimension = cfg.embedding_dimension

    dbf = db_path()
    if not dbf.exists() or not enabled:
        return {
            "embedding_enabled": enabled,
            "embedding_model": model,
            "embedding_dimension": dimension,
            "embedding_pending_count": 0,
            "embedding_stale_count": 0,
            "embedding_coverage_percent": 0.0,
            "embedding_failure_count": 0,
            "embedding_estimated_cost_usd": 0.0,
        }

    pending = 0
    stale = 0
    failure = 0
    total = 0
    cost = 0.0

    try:
        conn = open_readonly_connection(dbf)
        try:
            pending = optional_count_sync(conn, "SELECT COUNT(*) FROM embedding_status WHERE needs_reindex = 1")
            stale = optional_count_sync(
                conn,
                """
                SELECT COUNT(*)
                FROM message_embeddings me
                JOIN messages m ON m.message_id = me.message_id
                LEFT JOIN embeddings_meta em
                  ON em.target_id = me.message_id
                 AND em.target_type = 'message'
                WHERE em.target_id IS NULL
                   OR (em.content_hash IS NOT NULL AND em.content_hash != m.content_hash)
                """,
            )
            embedded_msg = optional_count_sync(conn, "SELECT COUNT(*) FROM message_embeddings")
            failure = optional_count_sync(conn, "SELECT COUNT(*) FROM embedding_status WHERE error_message IS NOT NULL")

            total_conv = int(
                conn.execute("SELECT COUNT(*) FROM conversations").fetchone()[0] or 0
            ) if _table_exists_readonly(conn, "conversations") else 0

            if total_conv > 0:
                total = embedded_msg

            # Rough cost: (embedded + pending) * 500 tokens/msg * $0.10/1M tokens
            from polylogue.storage.search_providers.sqlite_vec_support import (
                ESTIMATED_TOKENS_PER_MESSAGE,
                VOYAGE_4_COST_PER_1M_TOKENS,
            )

            estimated_tokens = (total + pending) * ESTIMATED_TOKENS_PER_MESSAGE
            cost = round(estimated_tokens * VOYAGE_4_COST_PER_1M_TOKENS / 1_000_000, 2)
        finally:
            conn.close()
    except (sqlite3.Error, OSError):
        pass

    coverage = 0.0
    if total_conv > 0:
        embedded_conv = max(0, total_conv - pending)
        coverage = embedded_conv / total_conv * 100

    return {
        "embedding_enabled": enabled,
        "embedding_model": model,
        "embedding_dimension": dimension,
        "embedding_pending_count": pending,
        "embedding_stale_count": stale,
        "embedding_coverage_percent": round(coverage, 1),
        "embedding_failure_count": failure,
        "embedding_estimated_cost_usd": cost,
    }


def _table_exists_readonly(conn: sqlite3.Connection, table: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=? LIMIT 1",
        (table,),
    ).fetchone()
    return row is not None


def _check_daemon_liveness() -> bool:
    """Check whether the daemon process is running via pidfile."""
    try:
        from polylogue.paths import archive_root

        pidfile = Path(archive_root()) / "daemon.pid"
        if not pidfile.exists():
            return False
        pid = int(pidfile.read_text().strip())
        os.kill(pid, 0)
        return True
    except (OSError, ValueError):
        return False


def build_daemon_status(
    *,
    sources: tuple[WatchSource, ...] | None = None,
    browser_capture_spool_path: Path | None = None,
    include_expensive_health: bool = False,
) -> DaemonStatus:
    """Build a typed DaemonStatus from durable component state."""
    watch_sources = sources if sources is not None else default_sources()
    db_info = _db_size_info()
    fts = _fts_readiness_info()
    freshness = _insight_freshness_info()
    live_cursor = _live_cursor_summary_info()
    live_ingest_attempts = _live_ingest_attempt_summary_info()
    raw_failures = _raw_failure_info()
    embedding_info = _embedding_readiness_info()

    # Build health status (FAST + MEDIUM by default; EXPENSIVE opt-in).
    from polylogue.daemon.health import HealthTier

    health_tiers: set[HealthTier] = {HealthTier.FAST, HealthTier.MEDIUM}
    if include_expensive_health:
        health_tiers.add(HealthTier.EXPENSIVE)
    try:
        health = check_health(tiers=health_tiers)
    except Exception:
        health = DaemonHealth()

    return DaemonStatus(
        raw_parse_failures=_safe_int(raw_failures.get("parse_failures", 0)),
        raw_validation_failures=_safe_int(raw_failures.get("validation_failures", 0)),
        raw_quarantined=_safe_int(raw_failures.get("quarantined", 0)),
        raw_failure_samples=_safe_list_of_dicts(raw_failures.get("samples")),
        raw_detection_warnings=_safe_int(raw_failures.get("detection_warnings", 0)),
        daemon_liveness=_check_daemon_liveness(),
        component_state=ComponentState(
            watcher="running" if watch_sources else "stopped",
            api="running",
            browser_capture="running" if browser_capture_spool_path else "stopped",
        ),
        source_lag=[SourceLagItem(name=s.name, root=str(s.root), exists=s.exists()) for s in watch_sources],
        failing_files=[item.source_path for item in live_cursor.failing_files],
        live_cursor=live_cursor,
        live_ingest_attempts=live_ingest_attempts,
        db_size_bytes=_safe_int(db_info.get("db_size_bytes", 0)),
        wal_size_bytes=_safe_int(db_info.get("wal_size_bytes", 0)),
        blob_dir_size_bytes=_blob_size_info(),
        disk_free_bytes=_safe_int(db_info.get("disk_free_bytes", 0)),
        fts_readiness=FTSReadiness(
            messages_ready=fts.get("messages_ready", False),
            action_events_ready=fts.get("action_events_ready", False),
        ),
        insight_freshness=InsightFreshness(
            sessions_with_profiles=_safe_int(freshness.get("sessions_with_profiles", 0)),
            total_sessions=_safe_int(freshness.get("total_sessions", 0)),
        ),
        embedding_readiness=EmbeddingReadiness(
            embedding_enabled=bool(embedding_info.get("embedding_enabled", False)),
            embedding_model=str(embedding_info.get("embedding_model", "")),
            embedding_dimension=_safe_int(embedding_info.get("embedding_dimension", 0)),
            embedding_pending_count=_safe_int(embedding_info.get("embedding_pending_count", 0)),
            embedding_stale_count=_safe_int(embedding_info.get("embedding_stale_count", 0)),
            embedding_coverage_percent=_safe_float(embedding_info.get("embedding_coverage_percent")),
            embedding_failure_count=_safe_int(embedding_info.get("embedding_failure_count", 0)),
            embedding_estimated_cost_usd=_safe_float(embedding_info.get("embedding_estimated_cost_usd")),
        ),
        health=health,
        browser_capture_active=browser_capture_spool_path is not None,
        checked_at=datetime.now(UTC).isoformat(),
    )


def daemon_status_payload(
    *,
    sources: tuple[WatchSource, ...] | None = None,
    browser_capture_spool_path: Path | None = None,
) -> JSONDocument:
    """Return the local daemon component status payload (backward-compat dict)."""
    watch_sources = sources if sources is not None else default_sources()

    last_ingestion = None
    try:
        from polylogue.daemon.events import get_last_ingestion_batch

        last = get_last_ingestion_batch()
        if last:
            last_ingestion = {
                "ts": last.get("ts"),
                "payload": last.get("payload"),
            }
    except Exception:
        pass

    status = build_daemon_status(
        sources=sources,
        browser_capture_spool_path=browser_capture_spool_path,
    )

    return json_document(
        {
            "ok": True,
            "daemon": "polylogued",
            "daemon_liveness": status.daemon_liveness,
            "checked_at": status.checked_at,
            "component_state": status.component_state.model_dump(),
            "live": live_source_status_payload(watch_sources),
            "browser_capture": browser_capture_status_payload(browser_capture_spool_path),
            "db_path": str(db_path()),
            "db_size_bytes": status.db_size_bytes,
            "wal_size_bytes": status.wal_size_bytes,
            "blob_dir_size_bytes": status.blob_dir_size_bytes,
            "disk_free_bytes": status.disk_free_bytes,
            "quick_check_result": "unknown",
            "quick_check_age_s": None,
            "watcher_roots": [str(s.root) for s in watch_sources],
            "browser_capture_active": status.browser_capture_active,
            "failing_files": status.failing_files,
            "live_cursor": status.live_cursor.model_dump(),
            "live_ingest_attempts": status.live_ingest_attempts.model_dump(),
            "operations": status.current_operations,
            "last_ingestion_batch": last_ingestion,
            "fts_readiness": status.fts_readiness.model_dump(),
            "embedding_readiness": status.embedding_readiness.model_dump(),
            "health": {
                "overall_status": status.health.overall_status.value,
                "checked_at": status.health.checked_at,
                "alert_count": len(status.health.alerts),
                "tier_summary": status.health.tier_summary,
            },
        }
    )


def format_daemon_status_lines(payload: JSONDocument) -> list[str]:
    """Render daemon component status as plain text lines."""
    lines = ["Polylogue daemon"]
    if payload.get("daemon_liveness"):
        lines.append("  Status: running")
    live = payload.get("live")
    if isinstance(live, dict):
        lines.append(f"Live sources: {live.get('existing_source_count', 0)}/{live.get('source_count', 0)} available")
        sources = live.get("sources", [])
        if isinstance(sources, list):
            for source in sources:
                if isinstance(source, dict):
                    state = "available" if source.get("exists") else "missing"
                    lines.append(f"  {source.get('name')}: {source.get('root')} ({state})")
    browser_capture = payload.get("browser_capture")
    if isinstance(browser_capture, dict):
        lines.append(f"Browser capture spool: {browser_capture.get('spool_path')}")
        origins = browser_capture.get("allowed_origins", [])
        origin_text = ", ".join(str(item) for item in origins) if isinstance(origins, list) else str(origins)
        lines.append(f"Browser capture origins: {origin_text}")
    failing_files = payload.get("failing_files")
    live_cursor = payload.get("live_cursor")
    if isinstance(live_cursor, dict):
        lines.append(
            "Live cursor: "
            f"{live_cursor.get('tracked_file_count', 0)} tracked, "
            f"{live_cursor.get('failed_file_count', 0)} failed, "
            f"{live_cursor.get('excluded_file_count', 0)} excluded, "
            f"{live_cursor.get('retry_due_file_count', 0)} retry due, "
            f"{live_cursor.get('in_backoff_file_count', 0)} in backoff"
        )
    if isinstance(failing_files, list) and failing_files:
        omitted = _row_int(live_cursor.get("omitted_file_count")) if isinstance(live_cursor, dict) else 0
        if omitted:
            lines.append(f"Failing files: {len(failing_files)} shown, {omitted} omitted")
        else:
            lines.append(f"Failing files: {len(failing_files)}")
        for path in failing_files:
            lines.append(f"  {path}")
    attempts = payload.get("live_ingest_attempts")
    if isinstance(attempts, dict):
        recent = attempts.get("recent", [])
        running_count = attempts.get("running_count", 0)
        stale_count = attempts.get("stale_running_count", 0)
        if stale_count:
            lines.append(f"Live ingest attempts: {running_count} running, {stale_count} stale")
        else:
            lines.append(f"Live ingest attempts: {running_count} running")
        if isinstance(recent, list) and recent:
            latest = recent[0]
            if isinstance(latest, dict):
                stale_marker = " stale" if latest.get("stale") else ""
                lines.append(
                    "  latest: "
                    f"{latest.get('status')}{stale_marker} {latest.get('phase')} "
                    f"{latest.get('succeeded_file_count', 0)}/{latest.get('needed_file_count', 0)} files"
                )
                cgroup_current = latest.get("cgroup_memory_current_mb")
                if cgroup_current is not None:
                    cgroup_peak = latest.get("cgroup_memory_peak_mb")
                    cgroup_text = f"  memory: cgroup {cgroup_current} MiB"
                    if cgroup_peak is not None:
                        cgroup_text += f" peak {cgroup_peak} MiB"
                    lines.append(cgroup_text)
    # Health summary
    health = payload.get("health")
    if isinstance(health, dict):
        overall = health.get("overall_status", "unknown")
        lines.append(f"Health: {overall} ({health.get('alert_count', 0)} alerts)")
    # Embedding readiness
    embedding = payload.get("embedding_readiness")
    if isinstance(embedding, dict):
        if embedding.get("embedding_enabled"):
            coverage = _safe_float(embedding.get("embedding_coverage_percent"))
            pending = _safe_int(embedding.get("embedding_pending_count"))
            stale = _safe_int(embedding.get("embedding_stale_count"))
            lines.append(
                f"Embeddings: {coverage:.1f}% coverage, {pending} pending, {stale} stale"
            )
            if _safe_int(embedding.get("embedding_failure_count")) > 0:
                lines.append(f"  failures: {_safe_int(embedding.get('embedding_failure_count'))}")
            cost = _safe_float(embedding.get("embedding_estimated_cost_usd"))
            if cost > 0:
                model = str(embedding.get("embedding_model", ""))
                dimension = _safe_int(embedding.get("embedding_dimension"))
                lines.append(f"  model: {model} ({dimension}d), est. cost: ${cost:.2f}")
        else:
            lines.append("Embeddings: disabled")
    return lines


__all__ = [
    "DaemonStatus",
    "EmbeddingReadiness",
    "build_daemon_status",
    "browser_capture_status_payload",
    "daemon_status_payload",
    "format_daemon_status_lines",
    "live_source_status_payload",
]
