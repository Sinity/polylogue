"""Shared daemon status payloads."""

from __future__ import annotations

import contextlib
import json
import os
import re
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, cast

from pydantic import BaseModel, Field, field_validator

from polylogue.browser_capture.receiver import BrowserCaptureReceiverConfig, receiver_status_payload
from polylogue.core.json import JSONDocument, json_document
from polylogue.core.payload_coercion import optional_str as _optional_str
from polylogue.core.payload_coercion import required_str as _required_str
from polylogue.core.payload_coercion import row_float as _row_float
from polylogue.core.payload_coercion import row_int as _row_int
from polylogue.core.stats import percentile
from polylogue.daemon.catchup_status import (
    CatchupStatus as CatchupStatus,
)
from polylogue.daemon.catchup_status import (
    catchup_status_info,
    format_catchup_status_lines,
)
from polylogue.daemon.convergence_debt_status import (
    ConvergenceDebtSummary as ConvergenceDebtSummary,
)
from polylogue.daemon.convergence_debt_status import (
    convergence_debt_summary_info,
)
from polylogue.daemon.cursor_lag_status import CursorLagSummary as CursorLagSummary
from polylogue.daemon.cursor_lag_status import cursor_lag_summary_info
from polylogue.daemon.embedding_readiness import embedding_readiness_info
from polylogue.daemon.fts_status import FTSReadiness, fts_readiness_info
from polylogue.daemon.health import DaemonHealth, HealthAlert, HealthSeverity, check_health
from polylogue.daemon.live_ingest_attempt_models import (
    LiveIngestAttemptState,
)
from polylogue.daemon.live_ingest_attempt_models import (
    LiveIngestAttemptSummary as LiveIngestAttemptSummary,
)
from polylogue.daemon.live_ingest_attempt_progress import (
    SLOW_MIN_SAMPLES,
    SLOW_P95_QUANTILE,
    STUCK_AFTER_S,
    classify_attempt_progress,
    compute_slow_threshold_s,
)
from polylogue.daemon.live_ingest_attempt_workload import (
    LiveIngestStageEventInfo,
    latest_stage_events,
    workload_fields,
)
from polylogue.logging import get_logger
from polylogue.paths import archive_root, db_path, index_db_path, resolve_active_index_db_path
from polylogue.readiness.capability import CapabilityReadinessState, ComponentReadiness
from polylogue.readiness.claim_guard import derive_claim_guard
from polylogue.sources.live import WatchSource
from polylogue.sources.live.watcher import default_sources
from polylogue.storage.archive_readiness import (
    active_rebuild_index_attempts,
    raw_materialization_readiness_snapshot,
    raw_materialization_ready,
)
from polylogue.storage.raw_retention import raw_frontier_integrity_projection, raw_frontier_integrity_summary
from polylogue.storage.repair import raw_materialization_replay_backlog
from polylogue.storage.sqlite.archive_tiers import ARCHIVE_VERSION_BY_TIER
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from polylogue.storage.sqlite.connection_profile import open_readonly_connection

logger = get_logger(__name__)

# Backwards-compatible alias for the stuck threshold (#1246). The "stale"
# rollup field stays in the typed status payload to avoid breaking
# downstream consumers, but the threshold itself is owned by
# ``live_ingest_attempt_progress``.
_LIVE_INGEST_ATTEMPT_STALE_AFTER_S = STUCK_AFTER_S
_LIVE_CURSOR_FAILURE_SAMPLE_LIMIT = 50


def _active_status_db_path() -> Path:
    return resolve_active_index_db_path(db_anchor=db_path(), index_db=index_db_path())


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


class InsightFreshness(BaseModel):
    sessions_with_profiles: int = 0
    total_sessions: int = 0


class EmbeddingReadiness(BaseModel):
    embedding_enabled: bool = False
    embedding_config_enabled: bool = False
    embedding_has_voyage_key: bool = False
    embedding_model: str = ""
    embedding_dimension: int = 0
    embedding_status: str = "empty"
    embedding_freshness_status: str = "empty"
    embedding_retrieval_ready: bool = False
    embedding_pending_count: int = 0
    embedding_pending_message_count: int | None = None
    embedding_pending_message_count_exact: bool = False
    embedding_stale_count: int = 0
    embedding_coverage_percent: float = 0.0
    embedding_failure_count: int = 0
    embedding_estimated_cost_usd: float = 0.0
    embedding_latest_catchup_run: dict[str, object] | None = None
    embedding_latest_material_catchup_run: dict[str, object] | None = None


class RawMaterializationReadiness(BaseModel):
    available: bool = True
    classification: str | None = None
    precision: str | None = None
    raw_artifact_count: int = 0
    materialized_raw_artifact_count: int = 0
    archive_session_count: int = 0
    join_gap_count: int = 0
    total: int = 0
    critical: int = 0
    warning: int = 0
    actionable: int = 0
    blocked: int = 0
    classified: int = 0
    unchecked: int = 0
    affected_total: int = 0
    affected_actionable: int = 0
    affected_blocked: int = 0
    affected_open: int = 0
    affected_classified: int = 0
    affected_unchecked: int = 0
    lost_source_evidence_count: int = 0
    lost_source_evidence_samples: list[dict[str, object]] = Field(default_factory=list)
    category_counts: dict[str, int] = Field(default_factory=dict)
    source_family_counts: dict[str, int] = Field(default_factory=dict)
    sampled_rows: list[dict[str, object]] = Field(default_factory=list)
    raw_authority_census: dict[str, object] | None = None
    raw_authority_frontier: dict[str, object] | None = None
    raw_authority_frontier_blocking_count: int = 0
    raw_authority_frontier_remediation_refs: list[dict[str, object]] = Field(default_factory=list)
    raw_authority_blocker_count: int = 0
    raw_authority_pending_census_count: int = 0
    raw_authority_ledger_counts: dict[str, int] = Field(default_factory=dict)


class RawFrontierIntegrity(BaseModel):
    """Standing readiness projection of authoritative raw-frontier gaps (polylogue-yla8.7).

    Process health and raw-materialization candidate counts can both be
    green while an accepted append head references a deleted predecessor or
    an ingest cursor sits ahead of accepted material — yla8.6 found this only
    through manual SQL after ordinary use broke. This model composes three
    independently-degrading facts so that gap can never render silently green:

    * ``broken_head_*`` — a distinct current raw seed from either
      ``sessions.raw_id`` or ``raw_revision_heads`` whose transitive
      predecessor chain (the same
      chain validator ``active_raw_retention_authority`` uses to protect
      retention) is missing or invalid.
    * ``missing_source_raw_*`` — re-projects
      ``RawMaterializationReadiness.lost_source_evidence_count`` /
      ``lost_source_evidence_samples`` (an index ``sessions.raw_id`` absent
      from ``source.raw_sessions``); computed once there, not requeried here.
    * ``cursor_ahead_*`` — an ``ops.ingest_cursor`` committed byte frontier
      past the byte frontier actually accepted into the index for that
      logical source — the exact symptom yla8.6 found only via manual SQL.

    ``overall_status`` is ``"violated"`` when any check proves a real gap,
    even if a sibling remains unknown; otherwise unknown authority degrades
    the result and only fully-proven zeroes render ``"healthy"``.
    """

    available: bool = False
    overall_status: Literal["healthy", "unknown", "violated"] = "unknown"

    broken_head_status: Literal["healthy", "unknown", "violated"] = "unknown"
    broken_head_count: int = 0
    broken_head_checked_count: int = 0
    broken_head_samples: list[dict[str, object]] = Field(default_factory=list)
    broken_head_reason: str = ""

    missing_source_raw_status: Literal["healthy", "unknown", "violated"] = "unknown"
    missing_source_raw_count: int = 0
    missing_source_raw_samples: list[dict[str, object]] = Field(default_factory=list)
    missing_source_raw_reason: str = ""

    cursor_ahead_status: Literal["healthy", "unknown", "violated"] = "unknown"
    cursor_ahead_count: int = 0
    cursor_ahead_checked_count: int = 0
    cursor_head_comparison_count: int = 0
    cursor_ahead_comparison_count: int = 0
    cursor_ahead_samples: list[dict[str, object]] = Field(default_factory=list)
    cursor_authority_gap_count: int = 0
    cursor_authority_gap_samples: list[dict[str, object]] = Field(default_factory=list)
    cursor_ahead_reason: str = ""


class ArchiveTierStatus(BaseModel):
    name: Literal["source", "index", "embeddings", "user", "ops"]
    path: str
    resolved_path: str = ""
    device: int | None = None
    inode: int | None = None
    stable_id: str = ""
    exists: bool = False
    size_bytes: int = 0
    wal_size_bytes: int = 0
    user_version: int | None = None
    expected_user_version: int | None = None
    version_status: Literal["ok", "missing", "mismatch", "invalid"] = "missing"
    table_count: int = 0


class ArchiveStorageStatus(BaseModel):
    active_store: Literal["archive_file_set", "empty"] = "empty"
    active_db_path: str = ""
    archive_root: str = ""
    configured_archive_root: str = ""
    archive_root_matches_configured: bool = True
    archive_ready: bool = False
    archive_materialization_ready: bool = False
    active_rebuild_index_attempts: list[dict[str, object]] = Field(default_factory=list)
    final_shape_ready: bool = False
    archive_schema_ready: bool = False
    schema_mismatches: list[str] = Field(default_factory=list)
    present_tiers: list[str] = Field(default_factory=list)
    missing_tiers: list[str] = Field(default_factory=list)
    tiers: list[ArchiveTierStatus] = Field(default_factory=list)
    identity: dict[str, object] = Field(default_factory=dict)
    identity_conflicts: list[dict[str, object]] = Field(default_factory=list)


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


class RawFailureSample(BaseModel):
    """One bounded raw-ingest failure record surfaced in daemon status.

    Constructed by ``_raw_failure_info()`` from the ``raw_sessions``
    table and consumed by the typed ``DaemonStatus`` model. Every surface
    (CLI, MCP, daemon HTTP) receives the same structured taxonomy.

    Two failure surfaces share this envelope:

    * ``source == "ingest"`` (default) — failures observed on
      :mod:`polylogue.pipeline` parse/validate paths; the ``raw_id``
      and provider-specific signals come from ``raw_sessions``.
    * ``source == "maintenance"`` — failures observed on
      :mod:`polylogue.maintenance.replay` per-record paths, routed via
      :func:`polylogue.maintenance.failure_routing.route_failure_sample`
      and reattached to status here. They carry the originating
      :attr:`operation_id` and the typed planner :attr:`locator`.
    """

    failure_kind: Literal["decode_error", "parse_error", "schema_violation", "maintenance", "unknown"]
    provider_hint: str | None = None
    redacted_error: str = ""
    source: Literal["ingest", "maintenance"] = "ingest"
    operation_id: str | None = None
    locator: str | None = None

    @field_validator("redacted_error", mode="before")
    @classmethod
    def _redact_file_paths(cls, v: object) -> str:
        """Strip absolute file paths from error strings at construction time.

        Replaces absolute Unix paths (starting with ``/``) with
        ``[redacted]`` so that local filesystem layout is never
        exposed in status payloads.  URL path segments (e.g.
        ``https://host/v1/data``) and relative paths are not redacted.
        """
        if not isinstance(v, str):
            return ""

        def _replace(m: re.Match[str]) -> str:
            start = m.start()
            # Absolute path at start of string is always redacted.
            if start == 0:
                return "[redacted]"
            prev = v[start - 1]
            # Keep when preceded by a letter, digit, dot, or colon
            # (these indicate a URL host segment like "example.com/path").
            if prev.isalnum() or prev in (".", ":"):
                return m.group(0)
            # Keep when the surrounding context contains a URL protocol.
            # Look back up to 16 chars for "://".
            prefix = v[max(0, start - 16) : start + 1]
            if "://" in prefix:
                return m.group(0)
            return "[redacted]"

        return _PATH_REDACTION_RE.sub(_replace, v)


# Matches candidate absolute Unix paths: a ``/`` followed by one or more
# path segments (alphanumeric, dots, dashes, underscores).  The
# ``_redact_file_paths`` validator refines matches with context checks.
_PATH_REDACTION_RE = re.compile(r"/(?:[a-zA-Z0-9._\-]+/)*[a-zA-Z0-9._\-]+")


# ---------------------------------------------------------------------------
# DaemonStatus — typed model consumed by all surfaces
# ---------------------------------------------------------------------------


class DaemonStatus(BaseModel):
    """Typed daemon status consumed by CLI, TUI, web, browser extension, MCP."""

    daemon_liveness: bool = False
    daemon_lifecycle: dict[str, object] = Field(default_factory=dict)
    component_state: ComponentState = Field(default_factory=ComponentState)
    source_lag: list[SourceLagItem] = Field(default_factory=list)
    failing_files: list[str] = Field(default_factory=list)
    current_operations: list[dict[str, object]] = Field(default_factory=list)
    reset_queue: list[dict[str, object]] = Field(default_factory=list)
    ingestion_throughput: IngestionThroughput = Field(default_factory=IngestionThroughput)
    live_cursor: LiveCursorSummary = Field(default_factory=LiveCursorSummary)
    live_ingest_attempts: LiveIngestAttemptSummary = Field(default_factory=LiveIngestAttemptSummary)
    catchup: CatchupStatus = Field(default_factory=CatchupStatus)
    convergence: ConvergenceDebtSummary = Field(default_factory=ConvergenceDebtSummary)
    cursor_lag: CursorLagSummary = Field(default_factory=CursorLagSummary)
    db_size_bytes: int = 0
    wal_size_bytes: int = 0
    blob_dir_size_bytes: int = 0
    disk_free_bytes: int = 0
    fts_readiness: FTSReadiness = Field(default_factory=FTSReadiness)
    insight_freshness: InsightFreshness = Field(default_factory=InsightFreshness)
    embedding_readiness: EmbeddingReadiness = Field(default_factory=EmbeddingReadiness)
    raw_materialization_readiness: RawMaterializationReadiness = Field(default_factory=RawMaterializationReadiness)
    raw_frontier_integrity: RawFrontierIntegrity = Field(default_factory=RawFrontierIntegrity)
    raw_replay_backlog: dict[str, object] = Field(default_factory=dict)
    archive_storage: ArchiveStorageStatus = Field(default_factory=ArchiveStorageStatus)
    component_readiness: dict[str, object] = Field(default_factory=dict)
    claim_guard: dict[str, object] = Field(default_factory=dict)
    browser_capture_active: bool = False
    raw_parse_failures: int = 0
    raw_validation_failures: int = 0
    raw_quarantined: int = 0
    raw_maintenance_failures: int = 0
    raw_failure_samples: list[RawFailureSample] = Field(default_factory=list)
    raw_detection_warnings: int = 0
    health: DaemonHealth = Field(default_factory=DaemonHealth)
    checked_at: str = ""
    # Memory pressure — surfaced from the most recent running live ingest attempt
    rss_current_mb: float | None = None
    rss_peak_mb: float | None = None
    cgroup_memory_current_mb: float | None = None


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


def browser_capture_status_payload(
    spool_path: Path | None = None,
    *,
    include_spool_path: bool = False,
) -> JSONDocument:
    """Return safe status for the browser-capture receiver component."""
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
    payload = receiver_status_payload(config)
    if not include_spool_path:
        payload.pop("spool_path", None)
        payload.pop("artifact_path", None)
    return json_document(payload)


def browser_capture_status_public_payload(spool_path: Path | None = None) -> JSONDocument:
    """Return browser-capture status safe for the daemon web/status API."""
    return browser_capture_status_payload(spool_path, include_spool_path=False)


def _db_size_info() -> dict[str, object]:
    dbf = _active_status_db_path()
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


def _archive_storage_info() -> ArchiveStorageStatus:
    from polylogue.storage.archive_identity import ArchiveIdentity, archive_identity_conflicts

    active_db = _active_status_db_path()
    configured_root = archive_root()
    root = _archive_storage_root(configured_root=configured_root, active_db=active_db)
    tier_paths: dict[Literal["source", "index", "embeddings", "user", "ops"], Path] = {
        "source": root / "source.db",
        "index": root / "index.db",
        "embeddings": root / "embeddings.db",
        "user": root / "user.db",
        "ops": root / "ops.db",
    }
    identity = ArchiveIdentity.resolve(root)
    conflicts = archive_identity_conflicts(configured_root=configured_root, active_root=root)
    tiers = [_archive_tier_status(name, path) for name, path in tier_paths.items()]
    identity_by_name = {tier.name: tier for tier in identity.tiers}
    tiers = [
        tier.model_copy(
            update={
                "resolved_path": str(identity_by_name[tier.name].resolved_path),
                "device": identity_by_name[tier.name].device,
                "inode": identity_by_name[tier.name].inode,
                "stable_id": identity_by_name[tier.name].stable_id,
            }
        )
        for tier in tiers
    ]
    present_tiers = [str(tier.name) for tier in tiers if tier.exists]
    missing_tiers = [str(tier.name) for tier in tiers if not tier.exists]
    index_exists = "index" in present_tiers
    source_exists = "source" in present_tiers
    active_rebuild_attempts = active_rebuild_index_attempts(tier_paths["ops"])
    final_shape_ready = not missing_tiers
    schema_mismatches = [str(tier.name) for tier in tiers if tier.exists and tier.version_status != "ok"]
    archive_schema_ready = final_shape_ready and not schema_mismatches
    archive_ready = (
        index_exists and source_exists and archive_schema_ready and not active_rebuild_attempts and not conflicts
    )
    if index_exists and source_exists:
        active_store: Literal["archive_file_set", "empty"] = "archive_file_set"
    else:
        active_store = "empty"
    return ArchiveStorageStatus(
        active_store=active_store,
        active_db_path=str(active_db),
        archive_root=str(root),
        configured_archive_root=str(configured_root),
        archive_root_matches_configured=root == configured_root,
        archive_ready=archive_ready,
        archive_materialization_ready=archive_ready,
        active_rebuild_index_attempts=active_rebuild_attempts,
        final_shape_ready=final_shape_ready,
        archive_schema_ready=archive_schema_ready,
        schema_mismatches=schema_mismatches,
        present_tiers=present_tiers,
        missing_tiers=missing_tiers,
        tiers=tiers,
        identity=identity.as_dict(unit="polylogued.service"),
        identity_conflicts=[conflict.as_dict(unit="polylogued.service") for conflict in conflicts],
    )


def _archive_storage_root(*, configured_root: Path, active_db: Path) -> Path:
    if active_db.parent != configured_root and active_db.name == "index.db":
        return active_db.parent
    return configured_root


def _archive_tier_status(
    name: Literal["source", "index", "embeddings", "user", "ops"],
    path: Path,
) -> ArchiveTierStatus:
    wal_path = Path(f"{path}-wal")
    tier = ArchiveTier(name)
    expected_user_version = ARCHIVE_VERSION_BY_TIER[tier]
    if not path.exists():
        return ArchiveTierStatus(name=name, path=str(path), expected_user_version=expected_user_version)
    user_version: int | None = None
    table_count = 0
    version_status: Literal["ok", "missing", "mismatch", "invalid"] = "invalid"
    try:
        conn = open_readonly_connection(path)
        try:
            user_version = _row_int(conn.execute("PRAGMA user_version").fetchone()[0])
            version_status = "ok" if user_version == expected_user_version else "mismatch"
            table_count = _row_int(
                conn.execute("SELECT COUNT(*) FROM sqlite_master WHERE type = 'table'").fetchone()[0]
            )
        finally:
            conn.close()
    except sqlite3.Error as exc:
        # version_status already defaults to "invalid" above and stays that
        # way here, so this branch does carry a signal — but the exception
        # itself was previously discarded. Log it so operators can tell a
        # read failure from a genuinely corrupt/unversioned tier.
        logger.warning("archive tier status query failed for %s (%s): %s", path, name, exc, exc_info=True)
    return ArchiveTierStatus(
        name=name,
        path=str(path),
        exists=True,
        size_bytes=path.stat().st_size,
        wal_size_bytes=wal_path.stat().st_size if wal_path.exists() else 0,
        user_version=user_version,
        expected_user_version=expected_user_version,
        version_status=version_status,
        table_count=table_count,
    )


def _fts_readiness_info() -> dict[str, object]:
    return fts_readiness_info(_active_status_db_path())


def _insight_freshness_info() -> dict[str, object]:
    """Check insight materialization status through bounded SQL counts."""
    from polylogue.paths import sibling_index_db

    dbf = _active_status_db_path()
    if not dbf.exists():
        index_db = sibling_index_db(dbf, require_exists=False)
        if index_db is not None:
            archive_info = _archive_insight_freshness_info(index_db)
            if archive_info is not None:
                return archive_info
        return {"sessions_with_profiles": 0, "total_sessions": 0}
    index_db = sibling_index_db(dbf, require_exists=False)
    if index_db is not None:
        archive_info = _archive_insight_freshness_info(index_db)
        if archive_info is not None:
            return archive_info
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
                      AND name IN ('sessions', 'session_profiles')
                    """
                ).fetchall()
            }
            total_sessions = 0
            sessions_with_profiles = 0
            if "sessions" in tables:
                total_sessions = int(conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0] or 0)
            if "session_profiles" in tables:
                sessions_with_profiles = int(conn.execute("SELECT COUNT(*) FROM session_profiles").fetchone()[0] or 0)
        finally:
            conn.close()
        return {
            "sessions_with_profiles": sessions_with_profiles,
            "total_sessions": total_sessions,
        }
    except sqlite3.Error as exc:
        logger.warning("status: insight-freshness query failed for %s: %s", dbf, exc, exc_info=True)
        return {"sessions_with_profiles": 0, "total_sessions": 0}


def _archive_insight_freshness_info(archive_db: Path) -> dict[str, object] | None:
    if not archive_db.exists():
        return None
    try:
        conn = open_readonly_connection(archive_db)
        try:
            tables = {
                row[0]
                for row in conn.execute(
                    """
                    SELECT name
                    FROM sqlite_master
                    WHERE type = 'table'
                      AND name IN ('sessions', 'session_profiles')
                    """
                ).fetchall()
            }
            if "sessions" not in tables:
                return None
            total_sessions = int(conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0] or 0)
            sessions_with_profiles = 0
            if "session_profiles" in tables:
                sessions_with_profiles = int(conn.execute("SELECT COUNT(*) FROM session_profiles").fetchone()[0] or 0)
        finally:
            conn.close()
        return {
            "sessions_with_profiles": sessions_with_profiles,
            "total_sessions": total_sessions,
        }
    except sqlite3.Error:
        return None


def _raw_failure_info() -> dict[str, object]:
    """Query raw_sessions + maintenance routing for failure counts and samples.

    The payload merges two failure surfaces (#1198):

    * ``parse_failures`` / ``validation_failures`` / ``quarantined`` /
      ``detection_warnings`` come from the ``raw_sessions`` table
      (ingest path);
    * ``maintenance_failures`` comes from the JSONL file written by
      :func:`polylogue.maintenance.failure_routing.route_failure_sample`
      (replay path).

    ``samples`` interleaves both surfaces — newest first — capped at
    50 entries total. Each sample carries ``source`` so consumers can
    distinguish ``"ingest"`` rows from ``"maintenance"`` rows without
    re-querying.
    """
    dbf = _active_status_db_path()
    maintenance_samples, maintenance_count = _maintenance_failure_info()
    if not dbf.exists():
        archive_info = _archive_raw_failure_info(
            dbf.with_name("source.db"),
            maintenance_samples=maintenance_samples,
            maintenance_count=maintenance_count,
        )
        if archive_info is not None:
            return archive_info
        return {
            "parse_failures": 0,
            "validation_failures": 0,
            "quarantined": 0,
            "maintenance_failures": maintenance_count,
            "samples": maintenance_samples,
        }
    archive_info = _archive_raw_failure_info(
        dbf.with_name("source.db"),
        maintenance_samples=maintenance_samples,
        maintenance_count=maintenance_count,
    )
    if dbf.with_name("source.db").exists() and archive_info is not None:
        return archive_info

    try:
        conn = open_readonly_connection(dbf)
        try:
            tables = {
                row[0]
                for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name='raw_sessions'"
                ).fetchall()
            }
            if "raw_sessions" not in tables:
                return {
                    "parse_failures": 0,
                    "validation_failures": 0,
                    "quarantined": 0,
                    "detection_warnings": 0,
                    "maintenance_failures": maintenance_count,
                    "samples": maintenance_samples,
                }

            parse_fail = int(
                conn.execute("SELECT COUNT(*) FROM raw_sessions WHERE parse_error IS NOT NULL").fetchone()[0] or 0
            )
            validation_fail = int(
                conn.execute("SELECT COUNT(*) FROM raw_sessions WHERE validation_status = 'FAILED'").fetchone()[0] or 0
            )
            quarantined = int(
                conn.execute(
                    "SELECT COUNT(*) FROM raw_sessions WHERE parsed_at IS NULL AND (parse_error IS NOT NULL OR validation_status = 'FAILED')"
                ).fetchone()[0]
                or 0
            )
            detection_warnings_count = 0
            try:
                with contextlib.suppress(sqlite3.OperationalError):
                    detection_warnings_count = int(
                        conn.execute(
                            "SELECT COUNT(*) FROM raw_sessions WHERE detection_warnings IS NOT NULL"
                        ).fetchone()[0]
                        or 0
                    )
            except Exception as exc:
                # contextlib.suppress above already covers the expected
                # "detection_warnings column not on this schema yet" case;
                # anything reaching here is unexpected and previously
                # vanished silently behind a 0 count.
                logger.warning("detection-warnings count query failed: %s", exc, exc_info=True)
            # Bounded failure samples (most recent 50), typed.
            samples: list[RawFailureSample] = []
            raw_columns = {str(row[1]) for row in conn.execute("PRAGMA table_info(raw_sessions)").fetchall()}
            acquired_order_column = "acquired_at_ms" if "acquired_at_ms" in raw_columns else "acquired_at"
            for row in conn.execute(
                "SELECT raw_id, source_name, parse_error, validation_status, validation_error "
                "FROM raw_sessions "
                "WHERE parse_error IS NOT NULL OR validation_status = 'FAILED' "
                f"ORDER BY {acquired_order_column} DESC LIMIT 50"
            ).fetchall():
                parse_err = str(row[2] or "") if row[2] else ""
                val_status = str(row[3] or "") if row[3] else ""
                val_err = str(row[4] or "") if row[4] else ""
                provider = str(row[1]) if row[1] else None

                # Classify failure kind from the available error signals.
                if "JSONDecodeError" in parse_err or "decode error" in parse_err.lower():
                    kind: Literal["decode_error", "parse_error", "schema_violation", "unknown"] = "decode_error"
                elif val_status == "FAILED":
                    kind = "schema_violation"
                elif parse_err:
                    kind = "parse_error"
                else:
                    kind = "unknown"

                # Redacted error text: prefer parse_error, fall back to validation_error.
                error_text = parse_err or val_err

                samples.append(
                    RawFailureSample(
                        failure_kind=kind,
                        provider_hint=provider,
                        redacted_error=error_text,
                    )
                )
        finally:
            conn.close()
        combined: list[RawFailureSample] = list(samples)
        combined.extend(maintenance_samples)
        if len(combined) > 50:
            combined = combined[:50]
        return {
            "parse_failures": parse_fail,
            "validation_failures": validation_fail,
            "quarantined": quarantined,
            "detection_warnings": detection_warnings_count,
            "maintenance_failures": maintenance_count,
            "samples": combined,
        }
    except sqlite3.Error as exc:
        logger.warning("status: raw-failure query failed for %s: %s", dbf, exc, exc_info=True)
        return {
            "parse_failures": 0,
            "validation_failures": 0,
            "quarantined": 0,
            "detection_warnings": 0,
            "maintenance_failures": maintenance_count,
            "samples": maintenance_samples,
        }


def _archive_raw_failure_info(
    archive_db: Path,
    *,
    maintenance_samples: list[RawFailureSample],
    maintenance_count: int,
) -> dict[str, object] | None:
    if not archive_db.exists():
        return None
    try:
        conn = open_readonly_connection(archive_db)
        try:
            if not conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='raw_sessions'").fetchone():
                return None
            parse_fail = int(
                conn.execute("SELECT COUNT(*) FROM raw_sessions WHERE parse_error IS NOT NULL").fetchone()[0] or 0
            )
            validation_fail = int(
                conn.execute("SELECT COUNT(*) FROM raw_sessions WHERE validation_status = 'failed'").fetchone()[0] or 0
            )
            quarantined = int(
                conn.execute(
                    """
                    SELECT COUNT(*)
                    FROM raw_sessions
                    WHERE parsed_at_ms IS NULL
                      AND (parse_error IS NOT NULL OR validation_status = 'failed')
                    """
                ).fetchone()[0]
                or 0
            )
            detection_warnings_count = int(
                conn.execute(
                    """
                    SELECT COUNT(*)
                    FROM raw_sessions
                    WHERE detection_warnings_json IS NOT NULL
                      AND detection_warnings_json != '[]'
                    """
                ).fetchone()[0]
                or 0
            )
            samples: list[RawFailureSample] = []
            for row in conn.execute(
                """
                SELECT raw_id, origin, parse_error, validation_status, validation_error
                FROM raw_sessions
                WHERE parse_error IS NOT NULL OR validation_status = 'failed'
                ORDER BY acquired_at_ms DESC
                LIMIT 50
                """
            ).fetchall():
                parse_err = str(row[2] or "") if row[2] else ""
                val_status = str(row[3] or "") if row[3] else ""
                val_err = str(row[4] or "") if row[4] else ""
                origin = str(row[1]) if row[1] else None
                if "JSONDecodeError" in parse_err or "decode error" in parse_err.lower():
                    kind: Literal["decode_error", "parse_error", "schema_violation", "unknown"] = "decode_error"
                elif val_status == "failed":
                    kind = "schema_violation"
                elif parse_err:
                    kind = "parse_error"
                else:
                    kind = "unknown"
                samples.append(
                    RawFailureSample(
                        failure_kind=kind,
                        provider_hint=origin,
                        redacted_error=parse_err or val_err,
                    )
                )
        finally:
            conn.close()
        combined: list[RawFailureSample] = list(samples)
        combined.extend(maintenance_samples)
        if len(combined) > 50:
            combined = combined[:50]
        return {
            "parse_failures": parse_fail,
            "validation_failures": validation_fail,
            "quarantined": quarantined,
            "detection_warnings": detection_warnings_count,
            "maintenance_failures": maintenance_count,
            "samples": combined,
        }
    except sqlite3.Error:
        return None


def _maintenance_failure_info() -> tuple[list[RawFailureSample], int]:
    """Read routed maintenance failures into typed daemon samples (#1198).

    Returns a ``(samples, total_count)`` pair so the caller can both
    surface bounded samples and report the absolute count to the
    raw-failures health check.
    """
    from polylogue.maintenance.failure_routing import (
        count_maintenance_failures,
        read_maintenance_failures,
    )

    try:
        root = archive_root()
        records = read_maintenance_failures(root)
        total = count_maintenance_failures(root)
    except Exception as exc:
        logger.warning("status: maintenance-failure read failed: %s", exc, exc_info=True)
        return [], 0

    samples: list[RawFailureSample] = []
    for record in records:
        samples.append(
            RawFailureSample(
                failure_kind="maintenance",
                provider_hint=record.target or None,
                redacted_error=f"{record.kind}: {record.message}" if record.kind else record.message,
                source="maintenance",
                operation_id=record.operation_id or None,
                locator=record.locator or None,
            )
        )
    return samples, total


def _typed_failure_samples(value: object) -> list[RawFailureSample]:
    """Safely extract typed failure samples from a potentially heterogeneous source."""
    if isinstance(value, list):
        return [item for item in value if isinstance(item, RawFailureSample)]
    return []


def _safe_int(value: object) -> int:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return int(value)
    return 0


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


def _fmt_bytes(value: int) -> str:
    if value >= 1_000_000_000:
        return f"{value / 1_000_000_000:.1f} GB"
    if value >= 1_000_000:
        return f"{value / 1_000_000:.1f} MB"
    if value > 0:
        return f"{value / 1_000:.0f} KB"
    return "0 KB"


def _failing_files_info() -> list[str]:
    """Return live-source files currently marked failed or excluded."""
    return [item.source_path for item in _live_cursor_summary_info().failing_files]


def _live_cursor_summary_info() -> LiveCursorSummary:
    """Return live cursor backlog/failure state without source-tree scans."""
    dbf = _active_status_db_path()
    ops_summary = _archive_live_cursor_summary_info(dbf.with_name("ops.db"))
    if ops_summary is not None:
        return ops_summary
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
    except sqlite3.Error as exc:
        logger.warning("status: live-cursor summary query failed for %s: %s", dbf, exc, exc_info=True)
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


def _archive_live_cursor_summary_info(ops_db: Path) -> LiveCursorSummary | None:
    """Return cursor backlog/failure state from archive OPS when populated."""
    if not ops_db.exists():
        return None
    try:
        conn = open_readonly_connection(ops_db)
        try:
            has_table = conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'ingest_cursor'"
            ).fetchone()
            if has_table is None:
                return None
            columns = {str(row[1]) for row in conn.execute("PRAGMA table_info(ingest_cursor)")}
            if not {"failure_count", "next_retry_at", "excluded"}.issubset(columns):
                return None
            tracked_file_count = int(conn.execute("SELECT COUNT(*) FROM ingest_cursor").fetchone()[0])
            if tracked_file_count == 0:
                return None
            failed_file_count = int(
                conn.execute("SELECT COUNT(*) FROM ingest_cursor WHERE failure_count > 0").fetchone()[0]
            )
            excluded_file_count = int(
                conn.execute("SELECT COUNT(*) FROM ingest_cursor WHERE excluded = 1").fetchone()[0]
            )
            attention_file_count = int(
                conn.execute("SELECT COUNT(*) FROM ingest_cursor WHERE failure_count > 0 OR excluded = 1").fetchone()[0]
            )
            rows = conn.execute(
                """
                SELECT source_path, failure_count, next_retry_at, excluded
                FROM ingest_cursor
                WHERE failure_count > 0 OR excluded = 1
                ORDER BY source_path
                LIMIT ?
                """,
                (_LIVE_CURSOR_FAILURE_SAMPLE_LIMIT,),
            ).fetchall()
            retry_rows = conn.execute(
                """
                SELECT next_retry_at
                FROM ingest_cursor
                WHERE failure_count > 0
                """
            ).fetchall()
        finally:
            conn.close()
    except sqlite3.Error:
        return None

    now = datetime.now(UTC)
    retry_due_file_count = sum(1 for row in retry_rows if _retry_due(_optional_str(row[0]), now=now))
    failing_files = [
        LiveCursorFileState(
            source_path=str(row[0]),
            failure_count=_row_int(row[1]),
            next_retry_at=_optional_str(row[2]),
            excluded=bool(row[3]),
            retry_due=_row_int(row[1]) > 0 and _retry_due(_optional_str(row[2]), now=now),
        )
        for row in rows
    ]
    return LiveCursorSummary(
        tracked_file_count=tracked_file_count,
        failed_file_count=failed_file_count,
        excluded_file_count=excluded_file_count,
        retry_due_file_count=retry_due_file_count,
        in_backoff_file_count=max(0, failed_file_count - retry_due_file_count),
        sampled_file_count=len(failing_files),
        omitted_file_count=max(0, attention_file_count - len(failing_files)),
        failing_files=failing_files,
    )


def _live_ingest_attempt_summary_info() -> LiveIngestAttemptSummary:
    """Return recent durable live-ingest attempt snapshots."""
    from polylogue.paths import sibling_index_db

    dbf = _active_status_db_path()
    ops_summary = _archive_live_ingest_attempt_summary_info(dbf.with_name("ops.db"))
    index_db = sibling_index_db(dbf, require_exists=False)
    if ((index_db is not None and index_db.exists()) or not dbf.exists()) and ops_summary is not None:
        return ops_summary
    if not dbf.exists():
        return ops_summary if ops_summary is not None else LiveIngestAttemptSummary()
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
            cgroup_anon_expr = "cgroup_memory_anon_mb" if "cgroup_memory_anon_mb" in columns else "NULL"
            cgroup_file_expr = "cgroup_memory_file_mb" if "cgroup_memory_file_mb" in columns else "NULL"
            cgroup_inactive_expr = (
                "cgroup_memory_inactive_file_mb" if "cgroup_memory_inactive_file_mb" in columns else "NULL"
            )
            worker_in_flight_expr = "worker_in_flight_count" if "worker_in_flight_count" in columns else "NULL"
            worker_completed_expr = "worker_completed_count" if "worker_completed_count" in columns else "NULL"
            worker_total_expr = "worker_total_count" if "worker_total_count" in columns else "NULL"
            stale_cursor_write_expr = "stale_cursor_write_count" if "stale_cursor_write_count" in columns else "0"
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
                    {cgroup_anon_expr},
                    {cgroup_file_expr},
                    {cgroup_inactive_expr},
                    {worker_in_flight_expr},
                    {worker_completed_expr},
                    {worker_total_expr},
                    {stale_cursor_write_expr}
                FROM live_ingest_attempt
                ORDER BY updated_at DESC, started_at DESC
                LIMIT 5
                """
            ).fetchall()
            stage_events = latest_stage_events(
                conn,
                [_required_str(row[0]) for row in rows],
            )
            running_rows = conn.execute(
                """
                SELECT updated_at, started_at, completed_at
                FROM live_ingest_attempt
                WHERE status = 'running'
                """
            ).fetchall()
            slow_threshold_s = compute_slow_threshold_s(conn)
        finally:
            conn.close()
    except sqlite3.Error as exc:
        logger.warning("status: live-ingest-attempt summary query failed for %s: %s", dbf, exc, exc_info=True)
        return LiveIngestAttemptSummary()

    now = datetime.now(UTC)
    recent_attempts = [
        _live_ingest_attempt_state_from_row(
            row,
            now=now,
            stage_event=stage_events.get(_required_str(row[0])),
            slow_threshold_s=slow_threshold_s,
        )
        for row in rows
    ]
    stale_running_count = 0
    slow_running_count = 0
    stuck_running_count = 0
    for running_row in running_rows:
        updated_at = _required_str(running_row[0])
        age_s = _attempt_updated_age_s(updated_at, now=now)
        if age_s is not None and age_s >= _LIVE_INGEST_ATTEMPT_STALE_AFTER_S:
            stale_running_count += 1
            stuck_running_count += 1
            continue
        # ``total_time_s`` for a running attempt approximates as wall-clock
        # elapsed since ``started_at`` (the per-attempt rollup that the
        # operator already sees in ``recent``). The rollup only needs
        # slow/stuck counts; the full workload bundle stays on the
        # per-attempt path.
        started_at = _required_str(running_row[1])
        ended_at = _optional_str(running_row[2]) or updated_at
        try:
            started = datetime.fromisoformat(started_at)
            ended = datetime.fromisoformat(ended_at)
        except ValueError:
            continue
        if started.tzinfo is None:
            started = started.replace(tzinfo=UTC)
        if ended.tzinfo is None:
            ended = ended.replace(tzinfo=UTC)
        total_time_s = max(0.0, (ended.astimezone(UTC) - started.astimezone(UTC)).total_seconds())
        classification = classify_attempt_progress(
            status="running",
            updated_age_s=age_s,
            total_time_s=total_time_s,
            slow_threshold_s=slow_threshold_s,
        )
        if classification == "slow":
            slow_running_count += 1
        elif classification == "stuck":
            stuck_running_count += 1
    return LiveIngestAttemptSummary(
        running_count=len(running_rows),
        stale_running_count=stale_running_count,
        slow_running_count=slow_running_count,
        stuck_running_count=stuck_running_count,
        slow_threshold_s=slow_threshold_s,
        recent=recent_attempts,
    )


def _archive_live_ingest_attempt_summary_info(ops_db: Path) -> LiveIngestAttemptSummary | None:
    """Return live ingest-attempt status from archive OPS when populated."""
    if not ops_db.exists():
        return None
    try:
        conn = open_readonly_connection(ops_db)
        try:
            has_table = conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'ingest_attempts'"
            ).fetchone()
            if has_table is None:
                return None
            attempt_count = int(conn.execute("SELECT COUNT(*) FROM ingest_attempts").fetchone()[0])
            if attempt_count == 0:
                return None
            rows = conn.execute(
                """
                SELECT
                    attempt_id,
                    source_path,
                    origin,
                    status,
                    phase,
                    started_at_ms,
                    heartbeat_at_ms,
                    finished_at_ms,
                    parsed_raw_count,
                    materialized_count,
                    error_message
                FROM ingest_attempts
                ORDER BY COALESCE(heartbeat_at_ms, finished_at_ms, started_at_ms) DESC, started_at_ms DESC
                LIMIT 5
                """
            ).fetchall()
            stage_payloads = _archive_latest_stage_payloads(conn, [_required_str(row[0]) for row in rows])
            running_rows = conn.execute(
                """
                SELECT started_at_ms, heartbeat_at_ms, finished_at_ms
                FROM ingest_attempts
                WHERE status = 'running'
                """
            ).fetchall()
            slow_threshold_s = _archive_compute_slow_threshold_s(conn)
        finally:
            conn.close()
    except sqlite3.Error:
        return None

    now = datetime.now(UTC)
    recent_attempts = [
        _archive_live_ingest_attempt_state_from_row(
            row,
            now=now,
            stage_payload=stage_payloads.get(_required_str(row[0])),
            slow_threshold_s=slow_threshold_s,
        )
        for row in rows
    ]
    stale_running_count = 0
    slow_running_count = 0
    stuck_running_count = 0
    for row in running_rows:
        started_ms = _row_int(row[0])
        updated_at = _epoch_ms_to_iso(_row_int(row[1]) or _row_int(row[2]) or _row_int(row[0]))
        age_s = _attempt_updated_age_s(updated_at, now=now)
        updated_ms = _row_int(row[1]) or _row_int(row[2]) or started_ms
        total_time_s = max(0.0, (updated_ms - started_ms) / 1000.0)
        if age_s is not None and age_s >= _LIVE_INGEST_ATTEMPT_STALE_AFTER_S:
            stale_running_count += 1
            stuck_running_count += 1
            continue
        if (
            classify_attempt_progress(
                status="running",
                updated_age_s=age_s,
                total_time_s=total_time_s,
                slow_threshold_s=slow_threshold_s,
            )
            == "slow"
        ):
            slow_running_count += 1
    return LiveIngestAttemptSummary(
        running_count=len(running_rows),
        stale_running_count=stale_running_count,
        slow_running_count=slow_running_count,
        stuck_running_count=stuck_running_count,
        slow_threshold_s=slow_threshold_s,
        recent=recent_attempts,
    )


def _archive_compute_slow_threshold_s(conn: sqlite3.Connection) -> float | None:
    rows = conn.execute(
        """
        SELECT started_at_ms, finished_at_ms
        FROM ingest_attempts
        WHERE status = 'completed'
          AND started_at_ms IS NOT NULL
          AND finished_at_ms IS NOT NULL
        """
    ).fetchall()
    samples: list[float] = []
    for row in rows:
        started_ms = _row_int(row[0])
        finished_ms = _row_int(row[1])
        duration = max(0.0, (finished_ms - started_ms) / 1000.0)
        if duration > 0.0:
            samples.append(duration)
    if len(samples) < SLOW_MIN_SAMPLES:
        return None
    samples.sort()
    return percentile(samples, SLOW_P95_QUANTILE)


def _archive_latest_stage_payloads(
    conn: sqlite3.Connection,
    attempt_ids: list[str],
) -> dict[str, dict[str, object]]:
    if not attempt_ids:
        return {}
    has_table = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'daemon_stage_events'"
    ).fetchone()
    if has_table is None:
        return {}
    placeholders = ", ".join("?" for _ in attempt_ids)
    rows = conn.execute(
        f"""
        SELECT attempt_id, payload_json
        FROM daemon_stage_events
        WHERE attempt_id IN ({placeholders})
        ORDER BY observed_at_ms DESC, rowid DESC
        """,
        tuple(attempt_ids),
    ).fetchall()
    payloads: dict[str, dict[str, object]] = {}
    for row in rows:
        attempt_id = _optional_str(row[0])
        if attempt_id is None:
            continue
        merged = payloads.setdefault(attempt_id, {})
        for key, value in _json_payload(row[1]).items():
            merged.setdefault(key, value)
    return payloads


def _archive_live_ingest_attempt_state_from_row(
    row: sqlite3.Row | tuple[object, ...],
    *,
    now: datetime,
    stage_payload: dict[str, object] | None = None,
    slow_threshold_s: float | None = None,
) -> LiveIngestAttemptState:
    payload = stage_payload or {}
    started_ms = _row_int(row[5])
    heartbeat_ms = _row_int(row[6])
    finished_ms = _row_int(row[7])
    updated_ms = heartbeat_ms or finished_ms or started_ms
    started_at = _epoch_ms_to_iso(started_ms)
    updated_at = _epoch_ms_to_iso(updated_ms)
    completed_at = _epoch_ms_to_iso(finished_ms) if finished_ms else None
    updated_age_s = _attempt_updated_age_s(updated_at, now=now)
    status_value = _required_str(row[3])
    stale = (
        status_value == "running" and updated_age_s is not None and updated_age_s >= _LIVE_INGEST_ATTEMPT_STALE_AFTER_S
    )
    total_time_s = _payload_float(payload, "total_time_s", default=max(0.0, (updated_ms - started_ms) / 1000.0))
    input_bytes = _payload_int(payload, "input_bytes")
    source_payload_read_bytes = _payload_int(payload, "source_payload_read_bytes")
    cursor_fingerprint_read_bytes = _payload_int(payload, "cursor_fingerprint_read_bytes")
    total_read_bytes = source_payload_read_bytes + cursor_fingerprint_read_bytes
    return LiveIngestAttemptState(
        attempt_id=_required_str(row[0]),
        started_at=started_at,
        updated_at=updated_at,
        completed_at=completed_at,
        status=status_value,
        phase=_optional_str(row[4]) or _payload_str(payload, "phase", default="") or "",
        queued_file_count=_row_int(row[8]) or _payload_int(payload, "queued_file_count"),
        needed_file_count=_row_int(row[8]) or _payload_int(payload, "needed_file_count"),
        succeeded_file_count=_row_int(row[9]) or _payload_int(payload, "succeeded_file_count"),
        failed_file_count=_payload_int(payload, "failed_file_count", default=1 if status_value == "failed" else 0),
        input_bytes=input_bytes,
        source_payload_read_bytes=source_payload_read_bytes,
        cursor_fingerprint_read_bytes=cursor_fingerprint_read_bytes,
        total_read_bytes=total_read_bytes,
        read_amplification=round(total_read_bytes / input_bytes, 3) if input_bytes > 0 else 0.0,
        files_per_second=(
            round((_row_int(row[9]) or _payload_int(payload, "succeeded_file_count")) / total_time_s, 3)
            if total_time_s > 0
            else 0.0
        ),
        source_mb_per_second=(
            round((source_payload_read_bytes / (1024 * 1024)) / total_time_s, 3) if total_time_s > 0 else 0.0
        ),
        archive_write_bytes_delta=_payload_int(payload, "archive_write_bytes_delta"),
        parse_time_s=_payload_float(payload, "parse_time_s"),
        convergence_time_s=_payload_float(payload, "convergence_time_s"),
        total_time_s=total_time_s,
        stage_timings_s=_stage_timings_from_payload(payload.get("stage_timings_json")),
        current_source=_optional_str(row[2]) or _payload_str(payload, "current_source"),
        current_path=_optional_str(row[1]) or _payload_str(payload, "current_path"),
        storage_route=_payload_str(payload, "storage_route"),
        storage_tiers=_payload_str(payload, "storage_tiers"),
        payload_available_file_count=_payload_optional_int(payload, "payload_available_file_count"),
        payload_unavailable_file_count=_payload_optional_int(payload, "payload_unavailable_file_count"),
        payload_replayed_from_blob_file_count=_payload_optional_int(payload, "payload_replayed_from_blob_file_count"),
        written_raw_count=_payload_optional_int(payload, "written_raw_count"),
        error=_optional_str(row[10]) or _payload_str(payload, "error"),
        rss_current_mb=_payload_optional_float(payload, "rss_current_mb"),
        rss_peak_self_mb=_payload_optional_float(payload, "rss_peak_self_mb"),
        rss_peak_children_mb=_payload_optional_float(payload, "rss_peak_children_mb"),
        cgroup_path=_payload_str(payload, "cgroup_path"),
        cgroup_memory_current_mb=_payload_optional_float(payload, "cgroup_memory_current_mb"),
        cgroup_memory_peak_mb=_payload_optional_float(payload, "cgroup_memory_peak_mb"),
        cgroup_memory_swap_current_mb=_payload_optional_float(payload, "cgroup_memory_swap_current_mb"),
        cgroup_memory_anon_mb=_payload_optional_float(payload, "cgroup_memory_anon_mb"),
        cgroup_memory_file_mb=_payload_optional_float(payload, "cgroup_memory_file_mb"),
        cgroup_memory_inactive_file_mb=_payload_optional_float(payload, "cgroup_memory_inactive_file_mb"),
        worker_in_flight_count=_payload_optional_int(payload, "worker_in_flight_count"),
        worker_completed_count=_payload_optional_int(payload, "worker_completed_count"),
        worker_total_count=_payload_optional_int(payload, "worker_total_count"),
        updated_age_s=updated_age_s,
        stale=stale,
        progress_classification=classify_attempt_progress(
            status=status_value,
            updated_age_s=updated_age_s,
            total_time_s=total_time_s,
            slow_threshold_s=slow_threshold_s,
        ),
    )


def _json_payload(value: object) -> dict[str, object]:
    if not isinstance(value, str):
        return {}
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return {}
    if not isinstance(parsed, dict):
        return {}
    return {str(key): item for key, item in parsed.items() if _is_json_scalar_or_container(item)}


def _is_json_scalar_or_container(value: object) -> bool:
    return value is None or isinstance(value, str | int | float | bool | list | dict)


def _payload_int(payload: dict[str, object], key: str, *, default: int = 0) -> int:
    return _row_int(payload.get(key, default))


def _payload_optional_int(payload: dict[str, object], key: str) -> int | None:
    return None if payload.get(key) is None else _row_int(payload[key])


def _payload_float(payload: dict[str, object], key: str, *, default: float = 0.0) -> float:
    value = payload.get(key, default)
    coerced = _row_float(value)
    return default if coerced is None else coerced


def _payload_optional_float(payload: dict[str, object], key: str) -> float | None:
    return _row_float(payload[key]) if key in payload and payload[key] is not None else None


def _payload_str(payload: dict[str, object], key: str, *, default: str | None = None) -> str | None:
    value = payload.get(key)
    return value if isinstance(value, str) else default


def _stage_timings_from_payload(value: object) -> dict[str, float]:
    if not isinstance(value, str):
        return {}
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return {}
    if not isinstance(parsed, dict):
        return {}
    return {str(key): float(item) for key, item in parsed.items() if isinstance(item, int | float)}


def _live_ingest_attempt_state_from_row(
    row: sqlite3.Row | tuple[object, ...],
    *,
    now: datetime,
    stage_event: LiveIngestStageEventInfo | None = None,
    slow_threshold_s: float | None = None,
) -> LiveIngestAttemptState:
    updated_at = _required_str(row[2])
    updated_age_s = _attempt_updated_age_s(updated_at, now=now)
    status_value = _required_str(row[4])
    stale = (
        status_value == "running" and updated_age_s is not None and updated_age_s >= _LIVE_INGEST_ATTEMPT_STALE_AFTER_S
    )
    workload = workload_fields(row, stage_event=stage_event)
    progress_classification = classify_attempt_progress(
        status=status_value,
        updated_age_s=updated_age_s,
        total_time_s=_safe_float(workload["total_time_s"]),
        slow_threshold_s=slow_threshold_s,
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
        total_read_bytes=_safe_int(workload["total_read_bytes"]),
        read_amplification=_safe_float(workload["read_amplification"]),
        files_per_second=_safe_float(workload["files_per_second"]),
        source_mb_per_second=_safe_float(workload["source_mb_per_second"]),
        archive_write_bytes_delta=_safe_int(workload["archive_write_bytes_delta"]),
        parse_time_s=_row_float(row[13]) or 0.0,
        convergence_time_s=_row_float(row[14]) or 0.0,
        total_time_s=_safe_float(workload["total_time_s"]),
        stage_timings_s=workload["stage_timings_s"] if isinstance(workload["stage_timings_s"], dict) else {},
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
        cgroup_memory_anon_mb=_row_float(row[25]) if len(row) > 25 else None,
        cgroup_memory_file_mb=_row_float(row[26]) if len(row) > 26 else None,
        cgroup_memory_inactive_file_mb=_row_float(row[27]) if len(row) > 27 else None,
        worker_in_flight_count=_row_int(row[28]) if len(row) > 28 else None,
        worker_completed_count=_row_int(row[29]) if len(row) > 29 else None,
        worker_total_count=_row_int(row[30]) if len(row) > 30 else None,
        stale_cursor_write_count=_row_int(row[31]) if len(row) > 31 else 0,
        updated_age_s=updated_age_s,
        stale=stale,
        progress_classification=progress_classification,
        slow_threshold_s=slow_threshold_s,
    )


def _attempt_updated_age_s(updated_at: str, *, now: datetime) -> float | None:
    try:
        updated = datetime.fromisoformat(updated_at)
    except ValueError:
        return None
    if updated.tzinfo is None:
        updated = updated.replace(tzinfo=UTC)
    return max(0.0, round((now - updated.astimezone(UTC)).total_seconds(), 3))


def _epoch_ms_to_iso(value: int) -> str:
    return datetime.fromtimestamp(value / 1000.0, UTC).isoformat()


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


def _check_daemon_liveness(lifecycle: dict[str, object] | None = None) -> bool:
    """Return whether one durable daemon lifecycle snapshot is fresh."""
    if lifecycle is None:
        from polylogue.daemon.lifecycle import lifecycle_status

        lifecycle = lifecycle_status()
    return bool(lifecycle.get("running", False))


def _daemon_component_readiness(
    *,
    component_state: ComponentState,
    fts_readiness: FTSReadiness,
    insight_freshness: InsightFreshness,
    embedding_readiness: EmbeddingReadiness,
    raw_materialization_readiness: RawMaterializationReadiness,
    raw_frontier_integrity: RawFrontierIntegrity,
    archive_storage: ArchiveStorageStatus,
    live_ingest_attempts: LiveIngestAttemptSummary,
) -> dict[str, object]:
    components: dict[str, object] = {
        "daemon_api": _component_from_daemon_state("daemon_api", component_state.api, scope="daemon").to_dict(),
        "daemon_watcher": _component_from_daemon_state(
            "daemon_watcher",
            component_state.watcher,
            scope="daemon",
        ).to_dict(),
        "browser_capture": _component_from_daemon_state(
            "browser_capture",
            component_state.browser_capture,
            scope="daemon",
        ).to_dict(),
        "search": _component_from_fts_readiness(fts_readiness).to_dict(),
        "raw_materialization": _component_from_raw_materialization_readiness(raw_materialization_readiness).to_dict(),
        "raw_frontier_integrity": _component_from_raw_frontier_integrity(raw_frontier_integrity).to_dict(),
        "session_profiles": _component_from_insight_freshness(insight_freshness).to_dict(),
        "embeddings": _component_from_daemon_embedding_readiness(embedding_readiness).to_dict(),
        "archive_storage": _component_from_archive_storage(archive_storage).to_dict(),
        "daemon_ingest": _component_from_live_ingest(live_ingest_attempts).to_dict(),
    }
    return components


def _daemon_claim_guard(
    *,
    archive_storage: ArchiveStorageStatus,
    raw_materialization_readiness: RawMaterializationReadiness,
    raw_frontier_integrity: RawFrontierIntegrity,
    fts_readiness: FTSReadiness,
    live_ingest_attempts: LiveIngestAttemptSummary,
) -> dict[str, object]:
    """Derive the claim-guard block for the daemon-serving status path."""
    raw_component = _component_from_raw_materialization_readiness(raw_materialization_readiness)
    fts_component = _component_from_fts_readiness(fts_readiness)
    rebuild_attempts = len(archive_storage.active_rebuild_index_attempts)
    active_writer = bool(live_ingest_attempts.running_count) or bool(rebuild_attempts)
    writer_parts: list[str] = []
    if live_ingest_attempts.running_count:
        writer_parts.append(f"{live_ingest_attempts.running_count} live ingest attempt(s) running")
    if rebuild_attempts:
        writer_parts.append(f"{rebuild_attempts} index rebuild attempt(s) running")
    guard = derive_claim_guard(
        archive_schema_ready=archive_storage.archive_schema_ready,
        schema_mismatches=archive_storage.schema_mismatches,
        missing_tiers=archive_storage.missing_tiers,
        raw_materialization_ready=raw_materialization_ready(raw_materialization_readiness),
        raw_materialization_summary=raw_component.summary,
        raw_frontier_integrity_ready=raw_frontier_integrity.overall_status == "healthy",
        raw_frontier_integrity_summary=raw_frontier_integrity_summary(raw_frontier_integrity.model_dump()),
        search_ready=fts_readiness.messages_ready,
        search_summary=fts_component.summary,
        active_writer=active_writer,
        active_writer_summary="; ".join(writer_parts),
    )
    return cast(dict[str, object], guard.to_dict())


def _component_from_daemon_state(component: str, state: str, *, scope: str) -> ComponentReadiness:
    readiness_state = {
        "running": CapabilityReadinessState.READY,
        "degraded": CapabilityReadinessState.DEGRADED,
        "stopped": CapabilityReadinessState.MISSING,
        "disabled": CapabilityReadinessState.MISSING,
    }.get(state, CapabilityReadinessState.UNKNOWN)
    return ComponentReadiness(
        component=component,
        scope=scope,
        state=readiness_state,
        summary=state,
    )


def _component_from_fts_readiness(readiness: FTSReadiness) -> ComponentReadiness:
    if readiness.messages_ready:
        state = CapabilityReadinessState.READY
    elif readiness.message_indexable_count == 0 or readiness.message_indexed_count == 0:
        state = CapabilityReadinessState.MISSING
    else:
        state = CapabilityReadinessState.STALE
    return ComponentReadiness(
        component="search",
        scope="lexical",
        state=state,
        summary="ready" if readiness.messages_ready else "fts index incomplete",
        counts={
            "message_indexed_count": readiness.message_indexed_count,
            "message_indexable_count": readiness.message_indexable_count,
            "coverage_pct": readiness.coverage_pct,
            "coverage_exact": readiness.coverage_exact,
        },
        repair_hint=None if readiness.messages_ready else "polylogued run",
    )


def _component_from_raw_materialization_readiness(
    readiness: RawMaterializationReadiness,
) -> ComponentReadiness:
    from polylogue.readiness.capability import component_from_raw_materialization_readiness

    return component_from_raw_materialization_readiness(readiness.model_dump())


def _component_from_raw_frontier_integrity(integrity: RawFrontierIntegrity) -> ComponentReadiness:
    from polylogue.readiness.capability import component_from_raw_frontier_integrity

    return component_from_raw_frontier_integrity(integrity.model_dump())


def _component_from_insight_freshness(freshness: InsightFreshness) -> ComponentReadiness:
    total = freshness.total_sessions
    with_profiles = freshness.sessions_with_profiles
    if total <= 0:
        state = CapabilityReadinessState.MISSING
        summary = "no sessions"
    elif with_profiles >= total:
        state = CapabilityReadinessState.READY
        summary = "ready"
    elif with_profiles > 0:
        state = CapabilityReadinessState.DEGRADED
        summary = "partial"
    else:
        state = CapabilityReadinessState.STALE
        summary = "profiles missing"
    return ComponentReadiness(
        component="session_profiles",
        scope="insights",
        state=state,
        summary=summary,
        counts={
            "sessions_with_profiles": with_profiles,
            "total_sessions": total,
            "missing_profiles": max(0, total - with_profiles),
        },
        repair_hint=None if state is CapabilityReadinessState.READY else "polylogued run",
    )


def _component_from_daemon_embedding_readiness(readiness: EmbeddingReadiness) -> ComponentReadiness:
    if not readiness.embedding_config_enabled:
        state = CapabilityReadinessState.MISSING
    elif not readiness.embedding_has_voyage_key:
        state = CapabilityReadinessState.BLOCKED
    elif readiness.embedding_failure_count:
        state = CapabilityReadinessState.DEGRADED
    elif readiness.embedding_freshness_status == "stale" or readiness.embedding_stale_count:
        state = CapabilityReadinessState.STALE
    elif readiness.embedding_retrieval_ready:
        state = CapabilityReadinessState.READY
    elif readiness.embedding_pending_count or (readiness.embedding_pending_message_count or 0):
        state = CapabilityReadinessState.REBUILDING
    else:
        state = CapabilityReadinessState.MISSING

    return ComponentReadiness(
        component="embeddings",
        scope="semantic",
        state=state,
        summary=readiness.embedding_status,
        counts={
            "pending_sessions": readiness.embedding_pending_count,
            "pending_messages": readiness.embedding_pending_message_count,
            "pending_messages_exact": readiness.embedding_pending_message_count_exact,
            "stale_messages": readiness.embedding_stale_count,
            "failure_count": readiness.embedding_failure_count,
            "coverage_pct": readiness.embedding_coverage_percent,
            "retrieval_ready": readiness.embedding_retrieval_ready,
        },
        repair_hint=_daemon_embedding_repair_hint(readiness, state),
    )


def _daemon_embedding_repair_hint(
    readiness: EmbeddingReadiness,
    state: CapabilityReadinessState,
) -> str | None:
    if state == CapabilityReadinessState.MISSING and not readiness.embedding_config_enabled:
        return "polylogue ops embed enable"
    if state == CapabilityReadinessState.BLOCKED:
        return "configure Voyage API key"
    if state in {
        CapabilityReadinessState.REBUILDING,
        CapabilityReadinessState.STALE,
        CapabilityReadinessState.DEGRADED,
    }:
        return "polylogue ops embed backfill"
    return None


def _component_from_archive_storage(storage: ArchiveStorageStatus) -> ComponentReadiness:
    if storage.archive_ready:
        state = CapabilityReadinessState.READY
    elif storage.active_rebuild_index_attempts:
        state = CapabilityReadinessState.REBUILDING
    elif storage.final_shape_ready and storage.archive_schema_ready and not storage.archive_materialization_ready:
        state = CapabilityReadinessState.STALE
    elif storage.final_shape_ready or storage.schema_mismatches:
        state = CapabilityReadinessState.BLOCKED
    elif "source" in storage.present_tiers and "index" in storage.present_tiers:
        state = CapabilityReadinessState.DEGRADED
    elif storage.present_tiers:
        state = CapabilityReadinessState.BLOCKED
    else:
        state = CapabilityReadinessState.MISSING
    caveats: tuple[str, ...] = ()
    if storage.missing_tiers:
        caveats += (f"missing_tiers:{','.join(storage.missing_tiers)}",)
    if storage.schema_mismatches:
        caveats += (f"schema_mismatch:{','.join(storage.schema_mismatches)}",)
    if storage.final_shape_ready and storage.archive_schema_ready and not storage.archive_materialization_ready:
        caveats += ("materialization_pending",)
    repair_hint = None
    if state is not CapabilityReadinessState.READY:
        if storage.schema_mismatches == ["index"]:
            repair_hint = "polylogue ops reset --index && polylogued run"
        elif storage.missing_tiers:
            repair_hint = "polylogue ops maintenance archive-init --yes"
        else:
            repair_hint = "polylogued run"
    return ComponentReadiness(
        component="archive_storage",
        scope="archive",
        state=state,
        summary=storage.active_store,
        counts={
            "present_tier_count": len(storage.present_tiers),
            "missing_tier_count": len(storage.missing_tiers),
            "archive_ready": storage.archive_ready,
            "final_shape_ready": storage.final_shape_ready,
            "archive_schema_ready": storage.archive_schema_ready,
            "schema_mismatch_count": len(storage.schema_mismatches),
            "active_rebuild_index_attempt_count": len(storage.active_rebuild_index_attempts),
        },
        caveats=caveats,
        repair_hint=repair_hint,
    )


def _component_from_live_ingest(summary: LiveIngestAttemptSummary) -> ComponentReadiness:
    if summary.stuck_running_count or summary.stale_running_count:
        state = CapabilityReadinessState.DEGRADED
    elif summary.running_count:
        state = CapabilityReadinessState.REBUILDING
    else:
        state = CapabilityReadinessState.READY
    caveats: list[str] = []
    if summary.stuck_running_count:
        caveats.append("stuck_running_attempts")
    if summary.stale_running_count:
        caveats.append("stale_running_attempts")
    return ComponentReadiness(
        component="daemon_ingest",
        scope="daemon",
        state=state,
        summary="running" if summary.running_count else "idle",
        counts={
            "running_count": summary.running_count,
            "slow_running_count": summary.slow_running_count,
            "stale_running_count": summary.stale_running_count,
            "stuck_running_count": summary.stuck_running_count,
        },
        caveats=tuple(caveats),
    )


def _raw_materialization_readiness_info() -> RawMaterializationReadiness:
    """Summarize acquired-but-not-materialized raw evidence for readiness.

    This is intentionally cheaper than the full archive-debt endpoint: daemon
    status needs the invariant that raw acquisition and index materialization do
    not silently diverge, but normal status reads must not open raw blobs or run
    the exact classifier over large archives.
    """
    payload = raw_materialization_readiness_snapshot(archive_root())
    return RawMaterializationReadiness.model_validate(payload)


def _raw_frontier_integrity_info(
    raw_materialization_readiness: RawMaterializationReadiness,
) -> RawFrontierIntegrity:
    """Read the one canonical split-tier projection used by every surface."""

    active_root = _active_status_db_path().parent
    projection = raw_frontier_integrity_projection(
        active_root,
        raw_materialization_readiness.model_dump(),
    )
    return RawFrontierIntegrity.model_validate(projection.to_dict())


def _raw_replay_backlog_info(*, include: bool = True) -> dict[str, object]:
    """Return weighted raw replay backlog only for an explicitly rich read."""
    if not include:
        return {
            "available": False,
            "reason": "excluded_from_bounded_status_snapshot",
            "candidate_count": 0,
            "total_blob_bytes": 0,
            "top_raw_rows": [],
            "origin_summary": [],
            "source_path_summary": [],
        }
    try:
        from polylogue.config import Config
        from polylogue.paths import render_root

        return raw_materialization_replay_backlog(
            Config(archive_root=archive_root(), render_root=render_root(), sources=[]),
            limit=5,
        )
    except Exception as exc:
        return {
            "available": False,
            "reason": str(exc),
            "candidate_count": 0,
            "total_blob_bytes": 0,
            "top_raw_rows": [],
            "origin_summary": [],
            "source_path_summary": [],
        }


def build_daemon_status(
    *,
    sources: tuple[WatchSource, ...] | None = None,
    browser_capture_enabled: bool | None = None,
    browser_capture_spool_path: Path | None = None,
    include_expensive_health: bool = False,
    include_raw_replay_backlog: bool = True,
) -> DaemonStatus:
    """Build a typed DaemonStatus from durable component state."""
    watch_sources = sources if sources is not None else default_sources()
    effective_browser_capture_spool_path = (
        browser_capture_spool_path
        if browser_capture_spool_path is not None
        else BrowserCaptureReceiverConfig.default().spool_path
    )
    browser_capture_active = (
        browser_capture_enabled
        if browser_capture_enabled is not None
        else effective_browser_capture_spool_path is not None
    )
    db_info = _db_size_info()
    storage_info = _archive_storage_info()
    fts = _fts_readiness_info()
    freshness = _insight_freshness_info()
    raw_materialization_readiness = _raw_materialization_readiness_info()
    raw_frontier_integrity = _raw_frontier_integrity_info(raw_materialization_readiness)
    raw_replay_backlog = _raw_replay_backlog_info(include=include_raw_replay_backlog)
    materialization_ready = storage_info.archive_materialization_ready and raw_materialization_ready(
        raw_materialization_readiness
    )
    storage_info = storage_info.model_copy(
        update={
            "archive_materialization_ready": materialization_ready,
            "archive_ready": storage_info.archive_ready and materialization_ready,
        }
    )
    live_cursor = _live_cursor_summary_info()
    live_ingest_attempts = _live_ingest_attempt_summary_info()
    active_db = _active_status_db_path()
    convergence = convergence_debt_summary_info(active_db)
    cursor_lag = cursor_lag_summary_info(active_db)
    catchup = catchup_status_info(
        active_db,
        latest_attempt=live_ingest_attempts.recent[0] if live_ingest_attempts.recent else None,
        convergence=convergence,
    )
    raw_failures = _raw_failure_info()
    embedding_info = embedding_readiness_info(active_db)

    # Build health status. Keep the default bounded; medium includes exact
    # FTS invariant scans and must be requested explicitly by operator paths.
    from polylogue.daemon.health import HealthTier

    health_tiers: set[HealthTier] = {HealthTier.FAST}
    if include_expensive_health:
        health_tiers.update({HealthTier.MEDIUM, HealthTier.EXPENSIVE})
    try:
        health = check_health(tiers=health_tiers)
    except Exception as exc:
        # DaemonHealth() alone defaults to overall_status=OK with zero
        # alerts — the single most misleading fallback possible for a
        # health check: a failed check would present as "everything is
        # fine" instead of "the check itself broke" (polylogue-cpf.4).
        logger.warning("status: check_health() failed: %s", exc, exc_info=True)
        health = DaemonHealth(
            overall_status=HealthSeverity.ERROR,
            checked_at=datetime.now(UTC).isoformat(),
            alerts=[
                HealthAlert(
                    check_name="check_health",
                    tier=HealthTier.FAST,
                    severity=HealthSeverity.ERROR,
                    message=f"health check itself failed: {exc}",
                    checked_at=datetime.now(UTC).isoformat(),
                )
            ],
        )

    # Surface memory pressure from the most recent running attempt
    rss_current_mb: float | None = None
    rss_peak_mb: float | None = None
    cgroup_memory_current_mb: float | None = None
    for attempt in live_ingest_attempts.recent:
        if attempt.status == "running":
            rss_current_mb = attempt.rss_current_mb
            cgroup_memory_current_mb = attempt.cgroup_memory_current_mb
            if attempt.rss_peak_self_mb is not None:
                rss_peak_mb = attempt.rss_peak_self_mb
                if attempt.rss_peak_children_mb is not None:
                    rss_peak_mb += attempt.rss_peak_children_mb
                elif rss_peak_mb is None:
                    rss_peak_mb = attempt.rss_peak_children_mb
            elif attempt.rss_peak_children_mb is not None:
                rss_peak_mb = attempt.rss_peak_children_mb
            break

    component_state = ComponentState(
        watcher="running" if watch_sources else "stopped",
        api="running",
        browser_capture="running" if browser_capture_active else "stopped",
    )
    fts_readiness = FTSReadiness(
        indexed_surface=str(fts.get("indexed_surface", "messages_fts")),
        messages_ready=bool(fts.get("messages_ready", False)),
        session_work_events_ready=bool(fts.get("session_work_events_ready", False)),
        threads_ready=bool(fts.get("threads_ready", False)),
        invariant_ready=bool(fts.get("invariant_ready", False)),
        message_indexed_count=None
        if fts.get("message_indexed_count") is None
        else _safe_int(fts.get("message_indexed_count", 0)),
        message_indexable_count=None
        if fts.get("message_indexable_count") is None
        else _safe_int(fts.get("message_indexable_count", 0)),
        coverage_pct=None if fts.get("coverage_pct") is None else _safe_float(fts.get("coverage_pct")),
        coverage_exact=bool(fts.get("coverage_exact", True)),
        surfaces=cast(dict[str, dict[str, int | bool | str | None]], fts.get("surfaces", {})),
    )
    embedding_readiness = EmbeddingReadiness(
        embedding_enabled=bool(embedding_info.get("embedding_enabled", False)),
        embedding_config_enabled=bool(embedding_info.get("embedding_config_enabled", False)),
        embedding_has_voyage_key=bool(embedding_info.get("embedding_has_voyage_key", False)),
        embedding_model=str(embedding_info.get("embedding_model", "")),
        embedding_dimension=_safe_int(embedding_info.get("embedding_dimension", 0)),
        embedding_status=str(embedding_info.get("embedding_status", "empty")),
        embedding_freshness_status=str(embedding_info.get("embedding_freshness_status", "empty")),
        embedding_retrieval_ready=bool(embedding_info.get("embedding_retrieval_ready", False)),
        embedding_pending_count=_safe_int(embedding_info.get("embedding_pending_count", 0)),
        embedding_pending_message_count_exact=bool(embedding_info.get("embedding_pending_message_count_exact", False)),
        embedding_pending_message_count=_safe_int(embedding_info.get("embedding_pending_message_count", 0))
        if bool(embedding_info.get("embedding_pending_message_count_exact", False))
        else None,
        embedding_stale_count=_safe_int(embedding_info.get("embedding_stale_count", 0)),
        embedding_coverage_percent=_safe_float(embedding_info.get("embedding_coverage_percent")),
        embedding_failure_count=_safe_int(embedding_info.get("embedding_failure_count", 0)),
        embedding_estimated_cost_usd=_safe_float(embedding_info.get("embedding_estimated_cost_usd")),
        embedding_latest_catchup_run=cast(
            dict[str, object] | None,
            embedding_info.get("embedding_latest_catchup_run"),
        ),
        embedding_latest_material_catchup_run=cast(
            dict[str, object] | None,
            embedding_info.get("embedding_latest_material_catchup_run"),
        ),
    )

    insight_freshness = InsightFreshness(
        sessions_with_profiles=_safe_int(freshness.get("sessions_with_profiles", 0)),
        total_sessions=_safe_int(freshness.get("total_sessions", 0)),
    )

    from polylogue.daemon.lifecycle import lifecycle_status

    daemon_lifecycle = lifecycle_status()
    return DaemonStatus(
        raw_parse_failures=_safe_int(raw_failures.get("parse_failures", 0)),
        raw_validation_failures=_safe_int(raw_failures.get("validation_failures", 0)),
        raw_quarantined=_safe_int(raw_failures.get("quarantined", 0)),
        raw_maintenance_failures=_safe_int(raw_failures.get("maintenance_failures", 0)),
        raw_failure_samples=_typed_failure_samples(raw_failures.get("samples")),
        raw_detection_warnings=_safe_int(raw_failures.get("detection_warnings", 0)),
        daemon_liveness=_check_daemon_liveness(daemon_lifecycle),
        daemon_lifecycle=daemon_lifecycle,
        component_state=component_state,
        source_lag=[SourceLagItem(name=s.name, root=str(s.root), exists=s.exists()) for s in watch_sources],
        failing_files=[item.source_path for item in live_cursor.failing_files],
        live_cursor=live_cursor,
        live_ingest_attempts=live_ingest_attempts,
        catchup=catchup,
        convergence=convergence,
        cursor_lag=cursor_lag,
        db_size_bytes=_safe_int(db_info.get("db_size_bytes", 0)),
        wal_size_bytes=_safe_int(db_info.get("wal_size_bytes", 0)),
        blob_dir_size_bytes=_blob_size_info(),
        disk_free_bytes=_safe_int(db_info.get("disk_free_bytes", 0)),
        fts_readiness=fts_readiness,
        insight_freshness=insight_freshness,
        embedding_readiness=embedding_readiness,
        raw_materialization_readiness=raw_materialization_readiness,
        raw_frontier_integrity=raw_frontier_integrity,
        raw_replay_backlog=raw_replay_backlog,
        archive_storage=storage_info,
        component_readiness=_daemon_component_readiness(
            component_state=component_state,
            fts_readiness=fts_readiness,
            insight_freshness=insight_freshness,
            embedding_readiness=embedding_readiness,
            raw_materialization_readiness=raw_materialization_readiness,
            raw_frontier_integrity=raw_frontier_integrity,
            archive_storage=storage_info,
            live_ingest_attempts=live_ingest_attempts,
        ),
        claim_guard=_daemon_claim_guard(
            archive_storage=storage_info,
            raw_materialization_readiness=raw_materialization_readiness,
            raw_frontier_integrity=raw_frontier_integrity,
            fts_readiness=fts_readiness,
            live_ingest_attempts=live_ingest_attempts,
        ),
        health=health,
        browser_capture_active=browser_capture_active,
        rss_current_mb=rss_current_mb,
        rss_peak_mb=rss_peak_mb,
        cgroup_memory_current_mb=cgroup_memory_current_mb,
        checked_at=datetime.now(UTC).isoformat(),
    )


def daemon_status_payload(
    *,
    sources: tuple[WatchSource, ...] | None = None,
    browser_capture_enabled: bool | None = None,
    browser_capture_spool_path: Path | None = None,
    include_browser_capture_spool_path: bool = False,
    include_raw_replay_backlog: bool = True,
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
    except Exception as exc:
        # last_ingestion stays None, identical to "daemon hasn't ingested
        # anything yet" — log so a get_last_ingestion_batch() failure isn't
        # mistaken for a cold-start daemon.
        logger.warning("daemon status last-ingestion lookup failed: %s", exc, exc_info=True)

    status = build_daemon_status(
        sources=sources,
        browser_capture_enabled=browser_capture_enabled,
        browser_capture_spool_path=browser_capture_spool_path,
        include_raw_replay_backlog=include_raw_replay_backlog,
    )
    archive_debt = _archive_debt_status_summary()

    return json_document(
        {
            "ok": status.raw_frontier_integrity.overall_status == "healthy",
            "daemon": "polylogued",
            "daemon_liveness": status.daemon_liveness,
            "daemon_lifecycle": status.daemon_lifecycle,
            "checked_at": status.checked_at,
            "component_state": status.component_state.model_dump(),
            "component_readiness": status.component_readiness,
            "claim_guard": status.claim_guard,
            "live": live_source_status_payload(watch_sources),
            "browser_capture": browser_capture_status_payload(
                browser_capture_spool_path,
                include_spool_path=include_browser_capture_spool_path,
            ),
            "db_path": str(_active_status_db_path()),
            "db_size_bytes": status.db_size_bytes,
            "wal_size_bytes": status.wal_size_bytes,
            "blob_dir_size_bytes": status.blob_dir_size_bytes,
            "disk_free_bytes": status.disk_free_bytes,
            "archive_storage": status.archive_storage.model_dump(),
            "archive_debt": archive_debt,
            "quick_check_result": "unknown",
            "quick_check_age_s": None,
            "watcher_roots": [str(s.root) for s in watch_sources],
            "browser_capture_active": status.browser_capture_active,
            "failing_files": status.failing_files,
            "live_cursor": status.live_cursor.model_dump(),
            "live_ingest_attempts": status.live_ingest_attempts.model_dump(),
            "catchup": status.catchup.model_dump(),
            "convergence": status.convergence.model_dump(),
            "operations": status.current_operations,
            "last_ingestion_batch": last_ingestion,
            "fts_readiness": status.fts_readiness.model_dump(),
            "raw_materialization_readiness": status.raw_materialization_readiness.model_dump(),
            "raw_frontier_integrity": status.raw_frontier_integrity.model_dump(),
            "raw_replay_backlog": status.raw_replay_backlog,
            "embedding_readiness": status.embedding_readiness.model_dump(),
            "memory": {
                "rss_current_mb": status.rss_current_mb,
                "rss_peak_mb": status.rss_peak_mb,
                "cgroup_memory_current_mb": status.cgroup_memory_current_mb,
            },
            "health": {
                "overall_status": status.health.overall_status.value,
                "checked_at": status.health.checked_at,
                "alert_count": len(status.health.alerts),
                "tier_summary": status.health.tier_summary,
            },
            "raw_parse_failures": status.raw_parse_failures,
            "raw_validation_failures": status.raw_validation_failures,
            "raw_quarantined": status.raw_quarantined,
            "raw_maintenance_failures": status.raw_maintenance_failures,
            "raw_detection_warnings": status.raw_detection_warnings,
            "raw_failure_samples": [s.model_dump() for s in status.raw_failure_samples],
        }
    )


def _archive_debt_status_summary() -> dict[str, object]:
    """Return a bounded status link to the unified archive-debt surface."""
    try:
        from polylogue.operations.archive_debt import archive_debt_list

        payload = archive_debt_list(archive_root=archive_root(), limit=5, exact_fts=False)
    except Exception:
        return {
            "endpoint": "/api/archive-debt",
            "available": False,
            "rows": [],
            "totals": {},
        }
    return {
        "endpoint": "/api/archive-debt",
        "available": True,
        "rows": [row.model_dump(mode="json", exclude_none=True) for row in payload.rows],
        "totals": payload.totals.model_dump(mode="json"),
    }


def format_daemon_status_lines(payload: JSONDocument) -> list[str]:
    """Render daemon component status as plain text lines."""
    lines = ["Polylogue daemon"]
    lifecycle = payload.get("daemon_lifecycle")
    if payload.get("daemon_liveness"):
        age = lifecycle.get("heartbeat_age_s") if isinstance(lifecycle, dict) else None
        suffix = f" (heartbeat {age}s ago)" if isinstance(age, int | float) else ""
        lines.append(f"  Status: running{suffix}")
    elif isinstance(lifecycle, dict):
        lines.append(f"  Status: {lifecycle.get('state', 'absent')} heartbeat")
    storage = payload.get("archive_storage")
    if isinstance(storage, dict):
        present = storage.get("present_tiers", [])
        missing = storage.get("missing_tiers", [])
        present_text = ", ".join(str(item) for item in present) if isinstance(present, list) else str(present)
        missing_text = ", ".join(str(item) for item in missing) if isinstance(missing, list) else str(missing)
        line = f"Storage: {storage.get('active_store', 'unknown')}"
        if present_text:
            line += f" ({present_text})"
        if missing_text:
            line += f"; missing {missing_text}"
        elif storage.get("final_shape_ready"):
            line += "; final split complete"
        schema_mismatches = storage.get("schema_mismatches", [])
        if isinstance(schema_mismatches, list) and schema_mismatches:
            line += f"; schema mismatch {', '.join(str(item) for item in schema_mismatches)}"
        if storage.get("archive_root_matches_configured") is False:
            line += f"; active root {storage.get('archive_root', 'unknown')}"
        lines.append(line)
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
        spool_state = "ready" if browser_capture.get("spool_ready") else "unavailable"
        lines.append(f"Browser capture spool: {spool_state}")
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
        slow_count = attempts.get("slow_running_count", 0)
        stuck_count = attempts.get("stuck_running_count", 0)
        # The slow/stuck split (#1246) distinguishes the operator-actionable
        # case (stuck = no progress at all) from the
        # informational-but-not-urgent case (slow = still ticking but
        # exceeding p95 historical duration). The existing ``stale`` label is
        # kept as a synonym when a writer has not populated the typed
        # ``stuck_running_count``.
        summary_parts = [f"{running_count} running"]
        if stuck_count:
            summary_parts.append(f"{stuck_count} stuck")
        elif stale_count:
            summary_parts.append(f"{stale_count} stale")
        if slow_count:
            summary_parts.append(f"{slow_count} slow")
        lines.append("Live ingest attempts: " + ", ".join(summary_parts))
        if isinstance(recent, list) and recent:
            latest = recent[0]
            if isinstance(latest, dict):
                classification = str(latest.get("progress_classification", "healthy"))
                if classification == "stuck":
                    progress_marker = " stuck"
                elif classification == "slow":
                    progress_marker = " slow"
                elif latest.get("stale"):
                    progress_marker = " stale"
                else:
                    progress_marker = ""
                lines.append(
                    "  latest: "
                    f"{latest.get('status')}{progress_marker} {latest.get('phase')} "
                    f"{latest.get('succeeded_file_count', 0)}/{latest.get('needed_file_count', 0)} files"
                )
                storage_route = latest.get("storage_route")
                if storage_route:
                    storage_line = f"  storage route: {storage_route}"
                    storage_tiers = latest.get("storage_tiers")
                    if storage_tiers:
                        storage_line += f" ({storage_tiers})"
                    payload_unavailable = latest.get("payload_unavailable_file_count")
                    if payload_unavailable is not None:
                        storage_line += f", {payload_unavailable} payload-unavailable"
                    payload_replayed = latest.get("payload_replayed_from_blob_file_count")
                    if payload_replayed is not None:
                        storage_line += f", {payload_replayed} blob-replayed"
                    lines.append(storage_line)
                read_amp = _safe_float(latest.get("read_amplification"))
                source_rate = _safe_float(latest.get("source_mb_per_second"))
                file_rate = _safe_float(latest.get("files_per_second"))
                lines.append(
                    f"  workload: read amp {read_amp:.2f}x, {source_rate:.2f} MiB/s source, {file_rate:.2f} files/s"
                )
                cgroup_current = latest.get("cgroup_memory_current_mb")
                if cgroup_current is not None:
                    cgroup_peak = latest.get("cgroup_memory_peak_mb")
                    cgroup_text = f"  memory: cgroup {cgroup_current} MiB"
                    if cgroup_peak is not None:
                        cgroup_text += f" peak {cgroup_peak} MiB"
                    cgroup_file = latest.get("cgroup_memory_file_mb")
                    cgroup_inactive = latest.get("cgroup_memory_inactive_file_mb")
                    if cgroup_file is not None:
                        cgroup_text += f" file {cgroup_file} MiB"
                    if cgroup_inactive is not None:
                        cgroup_text += f" inactive_file {cgroup_inactive} MiB"
                    lines.append(cgroup_text)
                rss_current = latest.get("rss_current_mb")
                if rss_current is not None:
                    rss_text = f"  memory: RSS {rss_current} MiB"
                    rss_peak_self = _safe_float(latest.get("rss_peak_self_mb"), default=-1.0)
                    rss_peak_children = _safe_float(latest.get("rss_peak_children_mb"), default=-1.0)
                    if rss_peak_self >= 0.0 or rss_peak_children >= 0.0:
                        peak_total = max(0.0, rss_peak_self) + max(0.0, rss_peak_children)
                        rss_text += f" peak {peak_total} MiB"
                    lines.append(rss_text)
    lines.extend(format_catchup_status_lines(payload.get("catchup")))
    convergence = payload.get("convergence")
    if isinstance(convergence, dict):
        failed_count = _safe_int(convergence.get("failed_count"))
        retry_due_count = _safe_int(convergence.get("retry_due_count"))
        lines.append(f"Convergence debt: {failed_count} failed, {retry_due_count} retry due")
        stages = convergence.get("stage_summaries")
        if isinstance(stages, list):
            for stage in stages[:5]:
                if isinstance(stage, dict):
                    lines.append(
                        "  "
                        f"{stage.get('stage')}: {stage.get('failed_count', 0)} failed, "
                        f"{stage.get('retry_due_count', 0)} retry due"
                    )
        families = convergence.get("family_summaries")
        if isinstance(families, list) and families:
            lines.append("  by source family:")
            for family in families[:5]:
                if isinstance(family, dict):
                    lines.append(f"    {family.get('family')}: {family.get('failed_count', 0)} failed")
    # Cursor lag summary (#1232) — degraded cursors are outside the lag SLO,
    # but must be surfaced even when no ordinary cursor is stuck.
    cursor_lag = payload.get("cursor_lag")
    if isinstance(cursor_lag, dict):
        stuck = _safe_int(cursor_lag.get("stuck_file_count"))
        degraded = _safe_int(cursor_lag.get("degraded_file_count"))
        if stuck > 0 or degraded > 0:
            idle = _safe_int(cursor_lag.get("idle_file_count"))
            max_lag = _safe_float(cursor_lag.get("max_lag_s"))
            max_degraded_lag = _safe_float(cursor_lag.get("max_degraded_lag_s"))
            lines.append(
                f"Cursor lag: {stuck} stuck, {degraded} degraded, {idle} idle "
                f"(worst lag {max_lag:.0f}s; degraded age {max_degraded_lag:.0f}s)"
            )
            cursor_families = cursor_lag.get("family_summaries")
            if isinstance(cursor_families, list):
                for family in cursor_families[:5]:
                    if isinstance(family, dict) and (
                        _safe_int(family.get("stuck_file_count")) > 0
                        or _safe_int(family.get("degraded_file_count")) > 0
                    ):
                        lines.append(
                            "  "
                            f"{family.get('family')}: {_safe_int(family.get('stuck_file_count'))} stuck, "
                            f"{_safe_int(family.get('degraded_file_count'))} degraded, "
                            f"worst lag {_safe_float(family.get('max_lag_s')):.0f}s"
                        )
                        # Auto-calibration baseline state (#1349). Render only
                        # when a baseline has been recorded; warm-up periods
                        # stay quiet to avoid confusing operators with "0
                        # samples" lines on a fresh archive.
                        baseline = family.get("baseline")
                        if isinstance(baseline, dict) and _safe_int(baseline.get("sample_count")) > 0:
                            p95 = _safe_float(baseline.get("rolling_p95_lag_s"))
                            baseline_sample_count = _safe_int(baseline.get("sample_count"))
                            multiplier = _safe_float(baseline.get("current_multiplier"))
                            severity = str(baseline.get("anomaly_severity", "ok"))
                            confident = bool(baseline.get("confident", False))
                            confidence_tag = "" if confident else " (baseline accruing)"
                            anomaly_tag = ""
                            if severity == "warning":
                                anomaly_tag = " [ANOMALY warn]"
                            elif severity == "error":
                                anomaly_tag = " [ANOMALY err]"
                            lines.append(
                                "    "
                                f"baseline p95 {p95:.0f}s over {baseline_sample_count} samples; "
                                f"current {multiplier:.1f}x{confidence_tag}{anomaly_tag}"
                            )
    materialization = payload.get("raw_materialization_readiness")
    if isinstance(materialization, dict):
        materialization_total = _safe_int(materialization.get("total"))
        if materialization_total > 0:
            if _safe_int(materialization.get("affected_unchecked")) or _safe_int(materialization.get("unchecked")):
                raw_count = _safe_int(materialization.get("raw_artifact_count"))
                materialized_count = _safe_int(materialization.get("materialized_raw_artifact_count"))
                progress = f"{materialized_count:,}/{raw_count:,} materialized; " if raw_count else ""
                lines.append(
                    f"Raw materialization: {progress}{materialization_total} raw/index join gap(s) need classification"
                )
            else:
                lines.append(
                    "Raw materialization: "
                    f"{materialization_total} debt row(s), "
                    f"{_safe_int(materialization.get('critical'))} critical, "
                    f"{_safe_int(materialization.get('warning'))} warning, "
                    f"{_safe_int(materialization.get('blocked'))} blocked"
                )
    frontier = payload.get("raw_frontier_integrity")
    if isinstance(frontier, dict):
        overall = str(frontier.get("overall_status") or "unknown")
        summary = raw_frontier_integrity_summary(frontier)
        detail = "" if summary == "ready" else f": {summary}"
        lines.append(f"Raw frontier integrity: {overall}{detail}")
    backlog = payload.get("raw_replay_backlog")
    if isinstance(backlog, dict) and backlog.get("available"):
        candidates = _safe_int(backlog.get("candidate_count"))
        missing = _safe_int(backlog.get("missing_blob_count"))
        if candidates > 0 or missing > 0:
            total_bytes = _safe_int(backlog.get("total_blob_bytes"))
            max_bytes = _safe_int(backlog.get("max_blob_bytes"))
            oversized = _safe_int(backlog.get("oversized_count"))
            line = f"Raw replay backlog: {candidates:,} raw row(s), {_fmt_bytes(total_bytes)} pending"
            if max_bytes:
                line += f"; largest {_fmt_bytes(max_bytes)}"
            if missing:
                line += f"; {missing:,} missing blob(s)"
            if oversized:
                line += f"; {oversized:,} oversized"
            lines.append(line)
            block_reason = backlog.get("execution_block_reason")
            if backlog.get("execution_blocked") and isinstance(block_reason, str) and block_reason:
                lines.append(f"  {block_reason}")

            origins = backlog.get("origin_summary")
            if isinstance(origins, list) and origins:
                parts: list[str] = []
                for item in origins[:3]:
                    if not isinstance(item, dict):
                        continue
                    origin = item.get("origin") or "unknown"
                    raw_count = _safe_int(item.get("raw_count"))
                    blob_bytes = _safe_int(item.get("total_blob_bytes"))
                    parts.append(f"{origin}={raw_count:,}/{_fmt_bytes(blob_bytes)}")
                if parts:
                    lines.append(f"  weighted by origin: {', '.join(parts)}")
    # Health summary
    health = payload.get("health")
    if isinstance(health, dict):
        health_overall = str(health.get("overall_status", "unknown"))
        lines.append(f"Health: {health_overall} ({health.get('alert_count', 0)} alerts)")
    # Raw failure summary
    raw_parse = _safe_int(payload.get("raw_parse_failures"))
    raw_val = _safe_int(payload.get("raw_validation_failures"))
    raw_quarantined = _safe_int(payload.get("raw_quarantined"))
    raw_maintenance = _safe_int(payload.get("raw_maintenance_failures"))
    total_raw = raw_parse + raw_val + raw_maintenance
    if total_raw > 0:
        breakdown = f"{raw_parse} parse + {raw_val} validation"
        if raw_maintenance > 0:
            breakdown += f" + {raw_maintenance} maintenance"
        lines.append(f"Raw failures: {total_raw} total ({raw_quarantined} quarantined), {breakdown}")
        samples = payload.get("raw_failure_samples")
        if isinstance(samples, list) and samples:
            for s in samples[:5]:
                if isinstance(s, dict):
                    kind = s.get("failure_kind", "unknown")
                    hint = s.get("provider_hint") or "?"
                    error_text = str(s.get("redacted_error", ""))
                    source = s.get("source", "ingest")
                    op_id = s.get("operation_id")
                    suffix = ""
                    if source == "maintenance" and op_id:
                        suffix = f" (op={str(op_id)[:8]})"
                    lines.append(f"  [{kind}] {hint}: {error_text[:120]}{suffix}")
    # Embedding readiness
    embedding = payload.get("embedding_readiness")
    if isinstance(embedding, dict):
        status = str(embedding.get("embedding_status", "unknown"))
        freshness = str(embedding.get("embedding_freshness_status", status))
        retrieval_ready = "ready" if embedding.get("embedding_retrieval_ready") else "not ready"
        pending_message_count = _safe_int(embedding.get("embedding_pending_message_count"))
        pending_message_text = (
            f"{pending_message_count:,} pending msgs"
            if embedding.get("embedding_pending_message_count_exact")
            else "pending msgs not calculated"
        )
        if embedding.get("embedding_enabled"):
            coverage = _safe_float(embedding.get("embedding_coverage_percent"))
            pending = _safe_int(embedding.get("embedding_pending_count"))
            stale = _safe_int(embedding.get("embedding_stale_count"))
            lines.append(
                f"Embeddings: {status}/{freshness}, {retrieval_ready}; "
                f"{coverage:.1f}% coverage, {pending} pending convs, "
                f"{pending_message_text}, {stale} stale"
            )
            if _safe_int(embedding.get("embedding_failure_count")) > 0:
                lines.append(f"  failures: {_safe_int(embedding.get('embedding_failure_count'))}")
            cost = _safe_float(embedding.get("embedding_estimated_cost_usd"))
            if cost > 0:
                model = str(embedding.get("embedding_model", ""))
                dimension = _safe_int(embedding.get("embedding_dimension"))
                lines.append(f"  model: {model} ({dimension}d), est. cost: ${cost:.2f}")
            latest = embedding.get("embedding_latest_catchup_run")
            if isinstance(latest, dict):
                lines.append(
                    "  latest catch-up: "
                    f"{latest.get('status', 'unknown')}, "
                    f"{_safe_int(latest.get('processed_sessions'))}/"
                    f"{_safe_int(latest.get('planned_sessions'))} convs, "
                    f"{_safe_int(latest.get('embedded_messages')):,} msgs embedded"
                )
            material = embedding.get("embedding_latest_material_catchup_run")
            if isinstance(material, dict) and (
                not isinstance(latest, dict) or material.get("run_id") != latest.get("run_id")
            ):
                lines.append(
                    "  latest material catch-up: "
                    f"{material.get('status', 'unknown')}, "
                    f"{_safe_int(material.get('processed_sessions'))}/"
                    f"{_safe_int(material.get('planned_sessions'))} convs, "
                    f"{_safe_int(material.get('embedded_messages')):,} msgs embedded"
                )
        else:
            pending = _safe_int(embedding.get("embedding_pending_count"))
            key_state = "key present" if embedding.get("embedding_has_voyage_key") else "key missing"
            lines.append(
                f"Embeddings: disabled ({key_state}; {status}/{freshness}, {retrieval_ready}; "
                f"{pending} pending convs, {pending_message_text})"
            )
            latest = embedding.get("embedding_latest_catchup_run")
            if isinstance(latest, dict):
                lines.append(
                    "  latest catch-up: "
                    f"{latest.get('status', 'unknown')}, "
                    f"{_safe_int(latest.get('processed_sessions'))}/"
                    f"{_safe_int(latest.get('planned_sessions'))} convs"
                )
            material = embedding.get("embedding_latest_material_catchup_run")
            if isinstance(material, dict) and (
                not isinstance(latest, dict) or material.get("run_id") != latest.get("run_id")
            ):
                lines.append(
                    "  latest material catch-up: "
                    f"{material.get('status', 'unknown')}, "
                    f"{_safe_int(material.get('processed_sessions'))}/"
                    f"{_safe_int(material.get('planned_sessions'))} convs"
                )
    return lines
