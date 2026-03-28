<<<<<<< HEAD
"""Health checks, verification, and repair operations."""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

from .config import Config
from .lib.outcomes import OutcomeCheck as HealthCheck
from .lib.outcomes import OutcomeReport
from .lib.outcomes import OutcomeStatus as VerifyStatus
from polylogue.logging import get_logger
from .lib.provider_identity import CORE_SCHEMA_PROVIDERS
from .sources.drive_client import default_credentials_path, default_token_path
from .storage.backends.connection import connection_context, open_connection
from .storage.index import index_status

logger = get_logger(__name__)
HEALTH_TTL_SECONDS = 600


@dataclass
class HealthReport(OutcomeReport):
    """Comprehensive health and verification report."""

    timestamp: int = field(default_factory=lambda: int(time.time()))
    cached: bool = False
    age_seconds: int = 0

    @property
    def summary(self) -> dict[str, int]:
        return self.summary_counts()

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "cached": self.cached,
            "age_seconds": self.age_seconds,
            "checks": [
                {
                    "name": check.name,
                    "status": check.status.value,
                    "count": check.count,
                    "detail": check.summary,
                    "breakdown": check.breakdown,
                }
                for check in self.checks
            ],
            "summary": self.summary,
        }


def _cache_path(archive_root: Path) -> Path:
    return archive_root / "health.json"


def _load_cached(archive_root: Path) -> dict[str, Any] | None:
    path = _cache_path(archive_root)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            return payload
    except Exception as exc:
        logger.warning("Failed to load health cache: %s", exc)
    return None


def _write_cache(archive_root: Path, report: HealthReport | dict[str, Any]) -> None:
    path = _cache_path(archive_root)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        data = report.to_dict() if isinstance(report, HealthReport) else report
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    except Exception as exc:
        logger.warning("Failed to write health cache: %s", exc)


def run_health(config: Config, *, deep: bool = False) -> HealthReport:
    """Run comprehensive system health and data verification checks.

    Args:
        config: Application configuration
        deep: If True, run PRAGMA integrity_check (slow on large databases).
              Skipped by default since it reads every page in the DB file.
    """
    checks: list[HealthCheck] = []

    # 1. Environment & Paths
    checks.append(HealthCheck("config", VerifyStatus.OK, summary="Zero-config (XDG paths)"))

    for path_name in ("archive_root", "render_root"):
        path = getattr(config, path_name)
        if path.exists():
            checks.append(HealthCheck(path_name, VerifyStatus.OK, summary=str(path)))
        else:
            checks.append(HealthCheck(path_name, VerifyStatus.WARNING, summary=f"Missing {path}"))

    # 2. Database Reachability (& optional Integrity)
    db_error: str | None = None
    try:
        with open_connection(None) as conn:
            checks.append(HealthCheck("database", VerifyStatus.OK, summary="DB reachable"))
            if deep:
                integrity = conn.execute("PRAGMA integrity_check").fetchone()[0]
                checks.append(
                    HealthCheck(
                        "sqlite_integrity",
                        VerifyStatus.OK if integrity == "ok" else VerifyStatus.ERROR,
                        summary=integrity,
                    )
                )
    except Exception as exc:
        db_error = str(exc)
        checks.append(HealthCheck("database", VerifyStatus.ERROR, summary=f"DB error: {db_error}"))

    # 3. Search Index Status
    if db_error is None:
        idx = index_status()
        if idx["exists"]:
            checks.append(
                HealthCheck(
                    "index", VerifyStatus.OK, count=int(cast(Any, idx["count"])), summary=f"messages indexed: {idx['count']}"
                )
            )
        else:
            checks.append(HealthCheck("index", VerifyStatus.WARNING, summary="index not built"))
    else:
        checks.append(HealthCheck("index", VerifyStatus.WARNING, summary=f"Skipped: database unavailable ({db_error})"))

    # 4. Data Quality (from former verify.py)
    if db_error is None:
        with connection_context(None) as conn:
        # Orphaned messages — two-step approach for performance.
        # Step 1: find distinct orphan conversation_ids (fast via index on small conversations table).
        # Step 2: count messages only for those IDs (skipped entirely when no orphans exist).
            orphan_cids = conn.execute(
                """
                SELECT DISTINCT m.conversation_id FROM messages m
                WHERE NOT EXISTS (SELECT 1 FROM conversations c WHERE c.conversation_id = m.conversation_id)
            """
            ).fetchall()
            if orphan_cids:
                placeholders = ",".join("?" for _ in orphan_cids)
                orphan_count = conn.execute(
                    f"SELECT COUNT(*) FROM messages WHERE conversation_id IN ({placeholders})",
                    [row[0] for row in orphan_cids],
                ).fetchone()[0]
            else:
                orphan_count = 0
            checks.append(
                HealthCheck(
                    "orphaned_messages",
                    VerifyStatus.OK if orphan_count == 0 else VerifyStatus.ERROR,
                    count=orphan_count,
                    summary="No orphaned messages" if orphan_count == 0 else f"{orphan_count:,} orphaned messages",
                )
            )

            # Duplicate conversations
            dup_conv = conn.execute(
                """
                SELECT COUNT(*) FROM (
                    SELECT conversation_id FROM conversations GROUP BY conversation_id HAVING COUNT(*) > 1
                )
            """
            ).fetchone()[0]
            checks.append(
                HealthCheck(
                    "duplicate_conversations",
                    VerifyStatus.OK if dup_conv == 0 else VerifyStatus.ERROR,
                    count=dup_conv,
                    summary="No duplicates" if dup_conv == 0 else f"{dup_conv} duplicate conversation IDs",
                )
            )

            # Empty conversations (conversations with no messages)
            empty_conv = conn.execute(
                """
                SELECT COUNT(*) FROM conversations c
                WHERE NOT EXISTS (SELECT 1 FROM messages m WHERE m.conversation_id = c.conversation_id)
            """
            ).fetchone()[0]
            checks.append(
                HealthCheck(
                    "empty_conversations",
                    VerifyStatus.OK if empty_conv == 0 else VerifyStatus.WARNING,
                    count=empty_conv,
                    summary="No empty conversations" if empty_conv == 0 else f"{empty_conv} conversation(s) with no messages",
                )
            )

            provider_rows = conn.execute(
                """
                SELECT provider_name, COUNT(*) AS count
                FROM conversations
                GROUP BY provider_name
                ORDER BY count DESC, provider_name ASC
                """
            ).fetchall()
            provider_breakdown = {
                str(row["provider_name"]): int(row["count"])
                for row in provider_rows
            }
            checks.append(
                HealthCheck(
                    "provider_distribution",
                    VerifyStatus.OK,
                    count=sum(provider_breakdown.values()),
                    summary=f"{len(provider_breakdown)} provider(s) represented",
                    breakdown=provider_breakdown,
                )
            )

            # FTS sync check (verify messages_fts table exists and is in sync)
            try:
                # Check if messages_fts table exists
                fts_exists = conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name='messages_fts'"
                ).fetchone()

                if fts_exists:
                    # Count messages in both tables
                    # Note: COUNT(*) on the FTS virtual table is extremely slow (15s+ on large DBs).
                    # Use the backing docsize table instead, which has one row per indexed document.
                    msg_count = conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
                    fts_count = conn.execute("SELECT COUNT(*) FROM messages_fts_docsize").fetchone()[0]

                    if msg_count == fts_count:
                        checks.append(
                            HealthCheck(
                                "fts_sync",
                                VerifyStatus.OK,
                                count=fts_count,
                                summary=f"FTS index in sync ({fts_count:,} messages indexed)",
                            )
                        )
                    else:
                        checks.append(
                            HealthCheck(
                                "fts_sync",
                                VerifyStatus.WARNING,
                                count=abs(msg_count - fts_count),
                                summary=f"FTS out of sync: {msg_count:,} messages vs {fts_count:,} indexed",
                            )
                        )
                else:
                    # FTS table doesn't exist
                    msg_count = conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
                    checks.append(
                        HealthCheck(
                            "fts_sync",
                            VerifyStatus.WARNING,
                            count=msg_count,
                            summary=f"FTS index not built ({msg_count:,} messages not indexed)",
                        )
                    )
            except Exception as exc:
                checks.append(
                    HealthCheck(
                        "fts_sync",
                        VerifyStatus.ERROR,
                        summary=f"FTS check failed: {exc}",
                    )
                )

    # 5. Source Accessibility
    for source in config.sources:
        if source.folder:
            cred_path = default_credentials_path(config.drive_config)
            token_path = default_token_path(config.drive_config)
            cred_status = VerifyStatus.OK if cred_path.exists() else VerifyStatus.WARNING
            token_status = VerifyStatus.OK if token_path.exists() else VerifyStatus.WARNING
            checks.append(
                HealthCheck(
                    f"source:{source.name}",
                    cred_status,
                    summary=f"drive folder '{source.folder}' credentials: {cred_path}",
                )
            )
            checks.append(
                HealthCheck(
                    f"source:{source.name}:token",
                    token_status,
                    summary=f"drive token: {token_path}",
                )
            )
        else:
            if source.path and source.path.exists():
                checks.append(HealthCheck(f"source:{source.name}", VerifyStatus.OK, summary=str(source.path)))
            else:
                checks.append(
                    HealthCheck(f"source:{source.name}", VerifyStatus.WARNING, summary=f"missing path: {source.path}")
                )

    # 6. Schema Health
    try:
        from .schemas.registry import SchemaRegistry

        registry = SchemaRegistry()
        known_providers = list(CORE_SCHEMA_PROVIDERS)
        available = registry.list_providers()
        missing = [p for p in known_providers if p not in available]

        if missing:
            checks.append(
                HealthCheck(
                    "schemas_missing",
                    VerifyStatus.WARNING,
                    count=len(missing),
                    summary=f"Missing schemas for: {', '.join(missing)}",
                )
            )
        else:
            checks.append(
                HealthCheck(
                    "schemas_coverage",
                    VerifyStatus.OK,
                    count=len(available),
                    summary=f"All {len(available)} provider schemas present",
                )
            )

        # Check schema age (warn if >30 days)
        stale_providers = []
        for provider in available:
            age = registry.get_schema_age_days(provider)
            if age is not None and age > 30:
                stale_providers.append(f"{provider} ({age}d)")
        if stale_providers:
            checks.append(
                HealthCheck(
                    "schemas_freshness",
                    VerifyStatus.WARNING,
                    count=len(stale_providers),
                    summary=f"Stale schemas (>30d): {', '.join(stale_providers)}",
                )
            )
        else:
            checks.append(
                    HealthCheck("schemas_freshness", VerifyStatus.OK, summary="All schemas current")
            )
    except Exception as exc:
        checks.append(
            HealthCheck("schemas", VerifyStatus.WARNING, summary=f"Schema check failed: {exc}")
        )

    report = HealthReport(checks=checks)
    _write_cache(config.archive_root, report)
    return report


def get_health(config: Config, *, deep: bool = False) -> HealthReport:
    """Get health report, using cache if valid.

    Args:
        config: Application configuration
        deep: If True, skip cache and run PRAGMA integrity_check (slow).
    """
    if not deep:
        cached_data = _load_cached(config.archive_root)
        now = int(time.time())
        if cached_data:
            ts = cached_data.get("timestamp", 0)
            if (now - ts) < HEALTH_TTL_SECONDS:
                try:
                    checks = [
                        HealthCheck(
                            name=c["name"],
                            status=VerifyStatus(c["status"]),
                            count=c.get("count", 0),
                            summary=c.get("detail", c.get("summary", "")),
                            breakdown=c.get("breakdown", {}),
                        )
                        for c in cached_data.get("checks", [])
                    ]
                except (KeyError, ValueError):
                    # Corrupted cache — fall through to fresh health check
                    pass
                else:
                    return HealthReport(
                        checks=checks,
                        timestamp=ts,
                        cached=True,
                        age_seconds=now - ts,
                    )

    report = run_health(config, deep=deep)
    report.cached = False
    report.age_seconds = 0
    return report


def run_runtime_health(config: Config) -> HealthReport:
    """Run runtime environment health checks.

    These checks verify the runtime environment is correctly configured
    and operational, complementing the data-quality checks in run_health().
    """
    checks: list[HealthCheck] = []

    # 1. Database path writability
    db = config.db_path
    if db.exists():
        try:
            # Test writable by opening in append mode
            with open(db, "a"):
                pass
            checks.append(HealthCheck("db_writable", VerifyStatus.OK, summary=f"Writable: {db}"))
        except OSError as exc:
            checks.append(HealthCheck("db_writable", VerifyStatus.ERROR, summary=f"Not writable: {exc}"))
    else:
        # Check parent directory writability
        parent = db.parent
        if parent.exists():
            writable = os.access(parent, os.W_OK)
            if writable:
                checks.append(HealthCheck("db_writable", VerifyStatus.OK, summary=f"Parent writable, DB will be created: {db}"))
            else:
                checks.append(HealthCheck("db_writable", VerifyStatus.ERROR, summary=f"Parent not writable: {parent}"))
        else:
            checks.append(HealthCheck("db_writable", VerifyStatus.WARNING, summary=f"Parent missing: {parent}"))

    # 2. Schema version
    try:
        from polylogue.storage.backends.schema import SCHEMA_VERSION

        with open_connection(None) as conn:
            current = conn.execute("PRAGMA user_version").fetchone()[0]
            if current == SCHEMA_VERSION:
                checks.append(HealthCheck("schema_version", VerifyStatus.OK, summary=f"v{current} (current)"))
            elif current == 0:
                checks.append(HealthCheck("schema_version", VerifyStatus.WARNING, summary="Uninitialized (v0)"))
            else:
                checks.append(HealthCheck(
                    "schema_version",
                    VerifyStatus.ERROR,
                    summary=f"v{current} (expected v{SCHEMA_VERSION})",
                ))
    except Exception as exc:
        checks.append(HealthCheck("schema_version", VerifyStatus.ERROR, summary=f"Cannot check: {exc}"))

    # 3. FTS tables health
    try:
        with connection_context(None) as conn:
            fts = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='messages_fts'"
            ).fetchone()
            if fts:
                # Verify FTS integrity via simple query
                conn.execute("SELECT * FROM messages_fts LIMIT 0")
                checks.append(HealthCheck("fts_tables", VerifyStatus.OK, summary="FTS5 table present and queryable"))
            else:
                checks.append(HealthCheck("fts_tables", VerifyStatus.WARNING, summary="FTS5 table not found"))
    except Exception as exc:
        checks.append(HealthCheck("fts_tables", VerifyStatus.ERROR, summary=f"FTS check failed: {exc}"))

    # 4. sqlite-vec availability
    try:
        import sqlite_vec  # noqa: F401
        checks.append(HealthCheck("sqlite_vec", VerifyStatus.OK, summary="sqlite-vec extension available"))
    except ImportError:
        checks.append(HealthCheck("sqlite_vec", VerifyStatus.WARNING, summary="sqlite-vec not installed (vector search unavailable)"))

    # 5. Archive and render root writability
    for label, path in [("archive_root", config.archive_root), ("render_root", config.render_root)]:
        if path.exists():
            writable = os.access(path, os.W_OK)
            status = VerifyStatus.OK if writable else VerifyStatus.ERROR
            detail = f"Writable: {path}" if writable else f"Not writable: {path}"
        else:
            # Can parent be created?
            parent = path.parent
            if parent.exists() and os.access(parent, os.W_OK):
                status = VerifyStatus.OK
                detail = f"Will be created: {path}"
            else:
                status = VerifyStatus.WARNING
                detail = f"Missing and parent not writable: {path}"
        checks.append(HealthCheck(f"{label}_writable", status, summary=detail))

    # 6. Config paths
    from polylogue.paths import config_home

    cfg_home = config_home()
    if cfg_home.exists():
        checks.append(HealthCheck("config_path", VerifyStatus.OK, summary=str(cfg_home)))
    else:
        checks.append(HealthCheck("config_path", VerifyStatus.OK, summary=f"Not yet created: {cfg_home}"))

    # 7. Google credentials (if Drive sources configured)
    drive_sources = [s for s in config.sources if s.is_drive]
    if drive_sources and config.drive_config:
        from polylogue.sources.drive_client import default_credentials_path, default_token_path

        cred = default_credentials_path(config.drive_config)
        token = default_token_path(config.drive_config)
        if cred.exists():
            checks.append(HealthCheck("drive_credentials", VerifyStatus.OK, summary=str(cred)))
        else:
            checks.append(HealthCheck("drive_credentials", VerifyStatus.WARNING, summary=f"Missing: {cred}"))
        if token.exists():
            checks.append(HealthCheck("drive_token", VerifyStatus.OK, summary=str(token)))
        else:
            checks.append(HealthCheck("drive_token", VerifyStatus.WARNING, summary=f"Missing (auth required): {token}"))

    # 8. Terminal capabilities
    import shutil
    import sys

    term = os.environ.get("TERM", "unknown")
    cols, rows = shutil.get_terminal_size()
    is_tty = sys.stdout.isatty()
    force_plain = os.environ.get("POLYLOGUE_FORCE_PLAIN", "")

    term_detail = f"TERM={term}, {cols}x{rows}, tty={is_tty}"
    if force_plain:
        term_detail += ", POLYLOGUE_FORCE_PLAIN=1"
    checks.append(HealthCheck("terminal", VerifyStatus.OK, summary=term_detail))

    # Rich/Textual availability
    try:
        import rich  # noqa: F401
        rich_ok = True
    except ImportError:
        rich_ok = False
    try:
        import textual  # noqa: F401
        textual_ok = True
    except ImportError:
        textual_ok = False
    checks.append(HealthCheck(
        "ui_libraries",
        VerifyStatus.OK if rich_ok else VerifyStatus.WARNING,
        summary=f"Rich={'yes' if rich_ok else 'no'}, Textual={'yes' if textual_ok else 'no'}",
    ))

    # 9. VHS availability
    vhs_available = shutil.which("vhs") is not None
    checks.append(HealthCheck(
        "vhs",
        VerifyStatus.OK if vhs_available else VerifyStatus.WARNING,
        summary="VHS available" if vhs_available else "VHS not found (showcase capture unavailable)",
    ))

    return HealthReport(checks=checks)


def cached_health_summary(archive_root: Path) -> str:
    """Get a concise summary of the cached health state."""
    cached_data = _load_cached(archive_root)
    if not cached_data:
        return "not run"
    try:
        ts = int(cached_data.get("timestamp", 0))
    except (TypeError, ValueError):
        return "unknown"
    age = int(time.time()) - ts
    summary = cached_data.get("summary", {})
    if not summary:
        return f"cached {age}s ago"

    parts = []
    for s in (VerifyStatus.OK, VerifyStatus.WARNING, VerifyStatus.ERROR):
        if count := summary.get(s.value):
            parts.append(f"{s.value}={count}")
    return f"cached {age}s ago ({', '.join(parts)})"


__all__ = [
    "get_health",
    "run_health",
    "run_runtime_health",
    "HealthCheck",
    "HealthReport",
    "VerifyStatus",
    "cached_health_summary",
||||||| parent of c5d6c6a9 (refactor: narrow governance/health/repair (27 files deleted))
=======
"""Consolidated health checking: archive, runtime, and source checks."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any, cast

from polylogue.lib.outcomes import OutcomeCheck, OutcomeReport, OutcomeStatus
from polylogue.maintenance_models import DerivedModelStatus

# Re-export canonical types for downstream consumers.
HealthCheck = OutcomeCheck
VerifyStatus = OutcomeStatus

HEALTH_TTL_SECONDS = 600


@dataclass
class HealthReport(OutcomeReport):
    """Comprehensive health and verification report."""

    timestamp: int = field(default_factory=lambda: int(time.time()))
    derived_models: dict[str, DerivedModelStatus] = field(default_factory=dict)
    archive_debt: dict[str, Any] = field(default_factory=dict)

    @property
    def summary(self) -> dict[str, int]:
        return self.summary_counts()

    @property
    def provenance(self) -> _ReportProvenance:
        return _ReportProvenance()

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "provenance": self.provenance.to_dict(),
            "checks": [
                {
                    "name": check.name,
                    "status": check.status.value,
                    "count": check.count,
                    "detail": check.summary,
                    "breakdown": check.breakdown,
                }
                for check in self.checks
            ],
            "derived_models": {
                name: status.to_dict()
                for name, status in sorted(self.derived_models.items())
            },
            "archive_debt": {
                name: (status.to_dict() if hasattr(status, "to_dict") else status)
                for name, status in sorted(self.archive_debt.items())
            },
            "summary": self.summary,
        }


@dataclass(frozen=True)
class _ReportProvenance:
    """Minimal provenance marker — always live (caching layer removed)."""

    source: str = "live"

    def to_dict(self) -> dict[str, Any]:
        return {"source": self.source}


# ---------------------------------------------------------------------------
# Archive health (database, derived models, orphans, providers)
# ---------------------------------------------------------------------------


def run_archive_health(config: Any, *, deep: bool = False) -> HealthReport:
    from polylogue.storage.backends.connection import connection_context, open_connection
    from polylogue.storage.derived_status import collect_derived_model_statuses_sync
    from polylogue.storage.index import index_status
    from polylogue.storage.repair import collect_archive_debt_statuses_sync

    checks: list[HealthCheck] = []
    checks.append(HealthCheck("config", VerifyStatus.OK, summary="Zero-config (XDG paths)"))

    for path_name in ("archive_root", "render_root"):
        path = getattr(config, path_name)
        if path.exists():
            checks.append(HealthCheck(path_name, VerifyStatus.OK, summary=str(path)))
        else:
            checks.append(HealthCheck(path_name, VerifyStatus.WARNING, summary=f"Missing {path}"))

    # --- database reachability ---
    db_error: str | None = None
    try:
        with open_connection(None) as conn:
            checks.append(HealthCheck("database", VerifyStatus.OK, summary="DB reachable"))
            if deep:
                integrity = conn.execute("PRAGMA integrity_check").fetchone()[0]
                checks.append(
                    HealthCheck(
                        "sqlite_integrity",
                        VerifyStatus.OK if integrity == "ok" else VerifyStatus.ERROR,
                        summary=integrity,
                    )
                )
    except Exception as exc:
        db_error = str(exc)
        checks.append(HealthCheck("database", VerifyStatus.ERROR, summary=f"DB error: {db_error}"))

    # --- index ---
    if db_error is None:
        idx = index_status()
        if idx["exists"]:
            checks.append(
                HealthCheck(
                    "index",
                    VerifyStatus.OK,
                    count=int(cast(Any, idx["count"])),
                    summary=f"messages indexed: {idx['count']}",
                )
            )
        else:
            checks.append(HealthCheck("index", VerifyStatus.WARNING, summary="index not built"))
    else:
        checks.append(
            HealthCheck(
                "index",
                VerifyStatus.WARNING,
                summary=f"Skipped: database unavailable ({db_error})",
            )
        )

    if db_error is not None:
        return HealthReport(checks=checks)

    # --- derived models, debt, duplicates, providers ---
    derived_statuses: dict[str, DerivedModelStatus] = {}
    archive_debt: dict[str, Any] = {}
    with connection_context(config.db_path) as conn:
        derived_statuses = collect_derived_model_statuses_sync(conn)
        archive_debt = collect_archive_debt_statuses_sync(conn, derived_statuses=derived_statuses)
        for debt_name in ("orphaned_messages", "orphaned_content_blocks", "orphaned_attachments"):
            debt = archive_debt[debt_name]
            checks.append(
                HealthCheck(
                    debt.name,
                    VerifyStatus.OK if debt.healthy else VerifyStatus.ERROR,
                    count=debt.issue_count,
                    summary=debt.detail,
                )
            )

        dup_conv = conn.execute(
            """
            SELECT COUNT(*) FROM (
                SELECT conversation_id FROM conversations GROUP BY conversation_id HAVING COUNT(*) > 1
            )
            """
        ).fetchone()[0]
        checks.append(
            HealthCheck(
                "duplicate_conversations",
                VerifyStatus.OK if dup_conv == 0 else VerifyStatus.ERROR,
                count=dup_conv,
                summary="No duplicates" if dup_conv == 0 else f"{dup_conv} duplicate conversation IDs",
            )
        )

        empty_debt = archive_debt["empty_conversations"]
        checks.append(
            HealthCheck(
                "empty_conversations",
                VerifyStatus.OK if empty_debt.healthy else VerifyStatus.WARNING,
                count=empty_debt.issue_count,
                summary=empty_debt.detail,
            )
        )

        provider_rows = conn.execute(
            """
            SELECT provider_name, COUNT(*) AS count
            FROM conversations
            GROUP BY provider_name
            ORDER BY count DESC, provider_name ASC
            """
        ).fetchall()
        provider_breakdown = {str(row["provider_name"]): int(row["count"]) for row in provider_rows}
        checks.append(
            HealthCheck(
                "provider_distribution",
                VerifyStatus.OK,
                count=sum(provider_breakdown.values()),
                summary=f"{len(provider_breakdown)} provider(s) represented",
                breakdown=provider_breakdown,
            )
        )
        for name, status_key in (
            ("action_event_read_model", "action_events"),
            ("action_event_fts", "action_events_fts"),
            ("fts_sync", "messages_fts"),
            ("retrieval_evidence", "retrieval_evidence"),
            ("retrieval_inference", "retrieval_inference"),
            ("retrieval_enrichment", "retrieval_enrichment"),
            ("session_profile_rows", "session_profile_rows"),
            ("session_profile_evidence_fts", "session_profile_evidence_fts"),
            ("session_profile_inference_fts", "session_profile_inference_fts"),
            ("session_profile_enrichment_fts", "session_profile_enrichment_fts"),
            ("session_work_event_inference", "session_work_event_inference"),
            ("session_work_event_inference_fts", "session_work_event_inference_fts"),
            ("session_tag_rollups", "session_tag_rollups"),
            ("session_phase_inference", "session_phase_inference"),
            ("day_session_summaries", "day_session_summaries"),
            ("week_session_summaries", "week_session_summaries"),
        ):
            status = derived_statuses.get(status_key)
            if status is None:
                continue
            checks.append(
                HealthCheck(
                    name,
                    VerifyStatus.OK if status.ready else VerifyStatus.WARNING,
                    count=status.materialized_documents or status.materialized_rows or status.pending_rows,
                    summary=status.detail,
                )
            )

        transcript_embeddings = derived_statuses.get("transcript_embeddings")
        if transcript_embeddings is not None:
            checks.append(
                HealthCheck(
                    "transcript_embeddings",
                    VerifyStatus.OK if transcript_embeddings.ready else VerifyStatus.WARNING,
                    count=transcript_embeddings.pending_documents,
                    summary=transcript_embeddings.detail,
                )
            )
            freshness_status = (
                VerifyStatus.OK
                if transcript_embeddings.materialized_rows == 0
                or (
                    transcript_embeddings.stale_rows == 0
                    and transcript_embeddings.missing_provenance_rows == 0
                )
                else VerifyStatus.WARNING
            )
            checks.append(
                HealthCheck(
                    "transcript_embedding_freshness",
                    freshness_status,
                    count=transcript_embeddings.stale_rows,
                    summary=(
                        "No embedded messages to assess freshness"
                        if transcript_embeddings.materialized_rows == 0
                        else (
                            f"Transcript embeddings fresh ({transcript_embeddings.materialized_rows:,} messages)"
                            if freshness_status is VerifyStatus.OK
                            else (
                                f"Transcript embeddings stale ({transcript_embeddings.stale_rows:,} stale, "
                                f"{transcript_embeddings.missing_provenance_rows:,} missing provenance)"
                            )
                        )
                    ),
                )
            )

    # --- source checks ---
    checks.extend(_build_source_health_checks(config))
    checks.extend(_build_schema_health_checks())

    return HealthReport(checks=checks, derived_models=derived_statuses, archive_debt=archive_debt)


def get_health(config: Any, *, deep: bool = False, use_cached: bool = False) -> HealthReport:
    """Get an archive health report. ``use_cached`` is accepted but ignored (caching layer removed)."""
    return run_archive_health(config, deep=deep)


# ---------------------------------------------------------------------------
# Runtime health (writability, schema version, FTS, vec, terminal)
# ---------------------------------------------------------------------------


def run_runtime_health(config: Any) -> HealthReport:
    from polylogue.storage.backends.connection import connection_context, open_connection

    checks: list[HealthCheck] = []

    db = config.db_path
    if db.exists():
        try:
            with open(db, "a"):
                pass
            checks.append(HealthCheck("db_writable", VerifyStatus.OK, summary=f"Writable: {db}"))
        except OSError as exc:
            checks.append(HealthCheck("db_writable", VerifyStatus.ERROR, summary=f"Not writable: {exc}"))
    else:
        parent = db.parent
        if parent.exists():
            writable = os.access(parent, os.W_OK)
            if writable:
                checks.append(
                    HealthCheck("db_writable", VerifyStatus.OK, summary=f"Parent writable, DB will be created: {db}")
                )
            else:
                checks.append(HealthCheck("db_writable", VerifyStatus.ERROR, summary=f"Parent not writable: {parent}"))
        else:
            checks.append(HealthCheck("db_writable", VerifyStatus.WARNING, summary=f"Parent missing: {parent}"))

    try:
        from polylogue.storage.backends.schema import SCHEMA_VERSION

        with open_connection(None) as conn:
            current = conn.execute("PRAGMA user_version").fetchone()[0]
            if current == SCHEMA_VERSION:
                checks.append(HealthCheck("schema_version", VerifyStatus.OK, summary=f"v{current} (current)"))
            elif current == 0:
                checks.append(HealthCheck("schema_version", VerifyStatus.WARNING, summary="Uninitialized (v0)"))
            else:
                checks.append(
                    HealthCheck(
                        "schema_version",
                        VerifyStatus.ERROR,
                        summary=f"v{current} (expected v{SCHEMA_VERSION})",
                    )
                )
    except Exception as exc:
        checks.append(HealthCheck("schema_version", VerifyStatus.ERROR, summary=f"Cannot check: {exc}"))

    try:
        with connection_context(None) as conn:
            fts = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='messages_fts'"
            ).fetchone()
            if fts:
                conn.execute("SELECT * FROM messages_fts LIMIT 0")
                checks.append(HealthCheck("fts_tables", VerifyStatus.OK, summary="FTS5 table present and queryable"))
            else:
                checks.append(HealthCheck("fts_tables", VerifyStatus.WARNING, summary="FTS5 table not found"))
    except Exception as exc:
        checks.append(HealthCheck("fts_tables", VerifyStatus.ERROR, summary=f"FTS check failed: {exc}"))

    try:
        import sqlite_vec  # noqa: F401

        checks.append(HealthCheck("sqlite_vec", VerifyStatus.OK, summary="sqlite-vec extension available"))
    except ImportError:
        checks.append(
            HealthCheck("sqlite_vec", VerifyStatus.WARNING, summary="sqlite-vec not installed (vector search unavailable)")
        )

    for label, path in [("archive_root", config.archive_root), ("render_root", config.render_root)]:
        if path.exists():
            writable = os.access(path, os.W_OK)
            status = VerifyStatus.OK if writable else VerifyStatus.ERROR
            detail = f"Writable: {path}" if writable else f"Not writable: {path}"
        else:
            parent = path.parent
            if parent.exists() and os.access(parent, os.W_OK):
                status = VerifyStatus.OK
                detail = f"Will be created: {path}"
            else:
                status = VerifyStatus.WARNING
                detail = f"Missing and parent not writable: {path}"
        checks.append(HealthCheck(f"{label}_writable", status, summary=detail))

    from polylogue.paths import config_home

    cfg_home = config_home()
    if cfg_home.exists():
        checks.append(HealthCheck("config_path", VerifyStatus.OK, summary=str(cfg_home)))
    else:
        checks.append(HealthCheck("config_path", VerifyStatus.OK, summary=f"Not yet created: {cfg_home}"))

    drive_sources = [source for source in config.sources if source.is_drive]
    if drive_sources and config.drive_config:
        from polylogue.sources.drive_auth import default_credentials_path, default_token_path

        cred = default_credentials_path(config.drive_config)
        token = default_token_path(config.drive_config)
        if cred.exists():
            checks.append(HealthCheck("drive_credentials", VerifyStatus.OK, summary=str(cred)))
        else:
            checks.append(HealthCheck("drive_credentials", VerifyStatus.WARNING, summary=f"Missing: {cred}"))
        if token.exists():
            checks.append(HealthCheck("drive_token", VerifyStatus.OK, summary=str(token)))
        else:
            checks.append(
                HealthCheck("drive_token", VerifyStatus.WARNING, summary=f"Missing (auth required): {token}")
            )

    import shutil
    import sys

    term = os.environ.get("TERM", "unknown")
    cols, rows = shutil.get_terminal_size()
    is_tty = sys.stdout.isatty()
    force_plain = os.environ.get("POLYLOGUE_FORCE_PLAIN", "")

    term_detail = f"TERM={term}, {cols}x{rows}, tty={is_tty}"
    if force_plain:
        term_detail += ", POLYLOGUE_FORCE_PLAIN=1"
    checks.append(HealthCheck("terminal", VerifyStatus.OK, summary=term_detail))

    try:
        import rich  # noqa: F401

        rich_ok = True
    except ImportError:
        rich_ok = False
    try:
        import textual  # noqa: F401

        textual_ok = True
    except ImportError:
        textual_ok = False
    checks.append(
        HealthCheck(
            "ui_libraries",
            VerifyStatus.OK if rich_ok else VerifyStatus.WARNING,
            summary=f"Rich={'yes' if rich_ok else 'no'}, Textual={'yes' if textual_ok else 'no'}",
        )
    )

    vhs_available = shutil.which("vhs") is not None
    checks.append(
        HealthCheck(
            "vhs",
            VerifyStatus.OK if vhs_available else VerifyStatus.WARNING,
            summary="VHS available" if vhs_available else "VHS not found (showcase capture unavailable)",
        )
    )

    return HealthReport(checks=checks)


# ---------------------------------------------------------------------------
# Source + schema health helpers (private)
# ---------------------------------------------------------------------------


def _build_source_health_checks(config: Any) -> list[HealthCheck]:
    from polylogue.sources.drive_auth import default_credentials_path, default_token_path

    checks: list[HealthCheck] = []
    for source in config.sources:
        if source.folder:
            cred_path = default_credentials_path(config.drive_config)
            token_path = default_token_path(config.drive_config)
            cred_status = VerifyStatus.OK if cred_path.exists() else VerifyStatus.WARNING
            token_status = VerifyStatus.OK if token_path.exists() else VerifyStatus.WARNING
            checks.append(
                HealthCheck(
                    f"source:{source.name}",
                    cred_status,
                    summary=f"drive folder '{source.folder}' credentials: {cred_path}",
                )
            )
            checks.append(
                HealthCheck(f"source:{source.name}:token", token_status, summary=f"drive token: {token_path}")
            )
        elif source.path and source.path.exists():
            checks.append(HealthCheck(f"source:{source.name}", VerifyStatus.OK, summary=str(source.path)))
        else:
            checks.append(
                HealthCheck(f"source:{source.name}", VerifyStatus.WARNING, summary=f"missing path: {source.path}")
            )
    return checks


def _build_schema_health_checks() -> list[HealthCheck]:
    checks: list[HealthCheck] = []
    try:
        from polylogue.lib.provider_identity import CORE_SCHEMA_PROVIDERS
        from polylogue.schemas.registry import SchemaRegistry

        registry = SchemaRegistry()
        known_providers = list(CORE_SCHEMA_PROVIDERS)
        available = registry.list_providers()
        missing = [provider for provider in known_providers if provider not in available]

        if missing:
            checks.append(
                HealthCheck(
                    "schemas_missing",
                    VerifyStatus.WARNING,
                    count=len(missing),
                    summary=f"Missing schemas for: {', '.join(missing)}",
                )
            )
        else:
            checks.append(
                HealthCheck(
                    "schemas_coverage",
                    VerifyStatus.OK,
                    count=len(available),
                    summary=f"All {len(available)} provider schemas present",
                )
            )

        stale_providers = []
        for provider in available:
            age = registry.get_schema_age_days(provider)
            if age is not None and age > 30:
                stale_providers.append(f"{provider} ({age}d)")
        if stale_providers:
            checks.append(
                HealthCheck(
                    "schemas_freshness",
                    VerifyStatus.WARNING,
                    count=len(stale_providers),
                    summary=f"Stale schemas (>30d): {', '.join(stale_providers)}",
                )
            )
        else:
            checks.append(HealthCheck("schemas_freshness", VerifyStatus.OK, summary="All schemas current"))
    except Exception as exc:
        checks.append(HealthCheck("schemas", VerifyStatus.WARNING, summary=f"Schema check failed: {exc}"))
    return checks


# ---------------------------------------------------------------------------
# Convenience: cached_health_summary (no-op cache, returns "not cached")
# ---------------------------------------------------------------------------


def cached_health_summary(archive_root: Any) -> str:
    """Stub — caching layer removed. Always returns 'not cached'."""
    return "not cached"


__all__ = [
    "HEALTH_TTL_SECONDS",
    "HealthCheck",
    "HealthReport",
    "VerifyStatus",
    "cached_health_summary",
    "get_health",
    "run_archive_health",
    "run_runtime_health",
>>>>>>> c5d6c6a9 (refactor: narrow governance/health/repair (27 files deleted))
]
