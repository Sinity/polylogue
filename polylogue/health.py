"""Consolidated health checking: archive, runtime, and source checks."""

from __future__ import annotations

import os
import shutil
import sqlite3
import sys
import time
from collections.abc import Mapping
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from pathlib import Path
from typing import TypeAlias

from polylogue.config import Config
from polylogue.lib.outcomes import OutcomeCheck, OutcomeReport, OutcomeStatus
from polylogue.maintenance_models import DerivedModelStatus
from polylogue.maintenance_targets import build_maintenance_target_catalog
from polylogue.storage.repair import ArchiveDebtStatus

# Re-export canonical types for downstream consumers.
HealthCheck = OutcomeCheck
VerifyStatus = OutcomeStatus

HEALTH_TTL_SECONDS = 600
_MAINTENANCE_TARGET_CATALOG = build_maintenance_target_catalog()

JSONScalar: TypeAlias = str | int | float | bool | None
JSONValue: TypeAlias = JSONScalar | list["JSONValue"] | Mapping[str, "JSONValue"]


@dataclass
class HealthReport(OutcomeReport):
    """Comprehensive health and verification report."""

    timestamp: int = field(default_factory=lambda: int(time.time()))
    derived_models: dict[str, DerivedModelStatus] = field(default_factory=dict)
    archive_debt: dict[str, ArchiveDebtStatus] = field(default_factory=dict)

    @property
    def summary(self) -> dict[str, int]:
        return self.summary_counts()

    @property
    def provenance(self) -> _ReportProvenance:
        return _ReportProvenance()

    def to_dict(self) -> dict[str, object]:
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
            "derived_models": {name: status.to_dict() for name, status in sorted(self.derived_models.items())},
            "archive_debt": {name: status.to_dict() for name, status in sorted(self.archive_debt.items())},
            "summary": self.summary,
        }


@dataclass(frozen=True)
class _ReportProvenance:
    """Minimal provenance marker — always live (caching layer removed)."""

    source: str = "live"

    def to_dict(self) -> dict[str, object]:
        return {"source": self.source}


def _summarize_db_error(exc: Exception) -> str:
    detail = str(exc)
    if "database is locked" in detail.lower():
        return "database is locked (archive is busy; retry after the current run completes)"
    return detail


def _open_health_probe_connection(db_path: Path) -> AbstractContextManager[sqlite3.Connection]:
    from polylogue.storage.backends.connection import open_read_connection

    return open_read_connection(db_path)


# ---------------------------------------------------------------------------
# Archive health (database, derived models, orphans, providers)
# ---------------------------------------------------------------------------


def run_archive_health(config: Config, *, deep: bool = False, probe_only: bool = False) -> HealthReport:
    from polylogue.storage.backends.schema import assert_supported_archive_layout
    from polylogue.storage.derived_status import collect_derived_model_statuses_sync
    from polylogue.storage.fts_lifecycle import message_fts_readiness_sync
    from polylogue.storage.repair import collect_archive_debt_statuses_sync

    checks: list[HealthCheck] = []
    checks.append(HealthCheck("config", VerifyStatus.OK, summary="XDG defaults active"))

    for path_name in ("archive_root", "render_root"):
        path = getattr(config, path_name)
        if path.exists():
            checks.append(HealthCheck(path_name, VerifyStatus.OK, summary=str(path)))
        else:
            checks.append(HealthCheck(path_name, VerifyStatus.WARNING, summary=f"Missing {path}"))

    # --- database reachability ---
    db_error: str | None = None
    try:
        with _open_health_probe_connection(config.db_path) as conn:
            assert_supported_archive_layout(conn)
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
        db_error = _summarize_db_error(exc)
        checks.append(HealthCheck("database", VerifyStatus.ERROR, summary=f"DB error: {db_error}"))

    # --- index ---
    if db_error is not None:
        checks.append(
            HealthCheck(
                "index",
                VerifyStatus.WARNING,
                summary=f"Skipped: database unavailable ({db_error})",
            )
        )
        return HealthReport(checks=checks)

    # --- derived models, debt, duplicates, providers ---
    derived_statuses: dict[str, DerivedModelStatus] = {}
    archive_debt: dict[str, ArchiveDebtStatus] = {}
    with _open_health_probe_connection(config.db_path) as conn:
        exact_index_counts = deep or not probe_only
        index_readiness = message_fts_readiness_sync(conn, verify_total_rows=exact_index_counts)
        if not index_readiness["exists"]:
            checks.append(HealthCheck("index", VerifyStatus.WARNING, summary="index not built"))
        else:
            indexed_rows = int(index_readiness["indexed_rows"])
            total_rows = int(index_readiness["total_rows"])
            if bool(index_readiness["ready"]):
                checks.append(
                    HealthCheck(
                        "index",
                        VerifyStatus.OK,
                        count=indexed_rows if exact_index_counts else 0,
                        summary=(f"messages indexed: {indexed_rows}" if exact_index_counts else "messages FTS present"),
                    )
                )
            else:
                checks.append(
                    HealthCheck(
                        "index",
                        VerifyStatus.WARNING,
                        count=indexed_rows if exact_index_counts else 0,
                        summary=(
                            f"messages indexed: {indexed_rows:,}/{total_rows:,}"
                            if exact_index_counts
                            else "messages FTS missing or empty; use --deep to verify full coverage"
                        ),
                    )
                )

        derived_statuses = collect_derived_model_statuses_sync(conn, verify_full=deep)
        archive_debt = collect_archive_debt_statuses_sync(
            conn,
            derived_statuses=derived_statuses,
            include_expensive=deep,
            probe_only=probe_only,
        )
        for spec in _MAINTENANCE_TARGET_CATALOG.archive_health_specs(deep=deep):
            debt = archive_debt.get(spec.name)
            if debt is None:
                continue
            unhealthy_status = spec.archive_health_unhealthy_status or VerifyStatus.WARNING
            checks.append(
                HealthCheck(
                    debt.name,
                    VerifyStatus.OK if debt.healthy else unhealthy_status,
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
                or (transcript_embeddings.stale_rows == 0 and transcript_embeddings.missing_provenance_rows == 0)
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


def get_health(config: Config, *, deep: bool = False, probe_only: bool = False) -> HealthReport:
    """Get a live archive health report."""
    return run_archive_health(config, deep=deep, probe_only=probe_only)


# ---------------------------------------------------------------------------
# Runtime health (writability, schema version, FTS, vec, terminal)
# ---------------------------------------------------------------------------


def run_runtime_health(config: Config) -> HealthReport:
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
        from polylogue.storage.backends.schema import SCHEMA_VERSION, assert_supported_archive_layout

        with _open_health_probe_connection(config.db_path) as conn:
            assert_supported_archive_layout(conn)
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
        checks.append(
            HealthCheck("schema_version", VerifyStatus.ERROR, summary=f"Cannot check: {_summarize_db_error(exc)}")
        )

    try:
        with _open_health_probe_connection(config.db_path) as conn:
            fts = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='messages_fts'").fetchone()
            if fts:
                conn.execute("SELECT * FROM messages_fts LIMIT 0")
                checks.append(HealthCheck("fts_tables", VerifyStatus.OK, summary="FTS5 table present and queryable"))
            else:
                checks.append(HealthCheck("fts_tables", VerifyStatus.WARNING, summary="FTS5 table not found"))
    except Exception as exc:
        checks.append(
            HealthCheck("fts_tables", VerifyStatus.ERROR, summary=f"FTS check failed: {_summarize_db_error(exc)}")
        )

    try:
        import sqlite_vec  # noqa: F401

        checks.append(HealthCheck("sqlite_vec", VerifyStatus.OK, summary="sqlite-vec extension available"))
    except ImportError:
        checks.append(
            HealthCheck(
                "sqlite_vec", VerifyStatus.WARNING, summary="sqlite-vec not installed (vector search unavailable)"
            )
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
        checks.append(HealthCheck("config_home", VerifyStatus.OK, summary=str(cfg_home)))
    else:
        checks.append(HealthCheck("config_home", VerifyStatus.OK, summary=f"Not yet created: {cfg_home}"))

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
            checks.append(HealthCheck("drive_token", VerifyStatus.WARNING, summary=f"Missing (auth required): {token}"))

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


def _build_source_health_checks(config: Config) -> list[HealthCheck]:
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
# Convenience: quick_health_summary (lightweight, no full check)
# ---------------------------------------------------------------------------


def quick_health_summary(archive_root: Path) -> str:
    """Return a one-line health summary without running a full health check.

    Reads basic DB stats (conversation count, schema version) for a cheap
    summary suitable for the non-verbose CLI status line.
    """
    try:
        from polylogue.storage.backends.schema import SCHEMA_VERSION

        db_path = archive_root / "polylogue.db"
        with _open_health_probe_connection(db_path) as conn:
            version = conn.execute("PRAGMA user_version").fetchone()[0]
            if version != SCHEMA_VERSION:
                return f"schema v{version} (expected v{SCHEMA_VERSION})"
            row = conn.execute("SELECT COUNT(*) FROM conversations").fetchone()
            count = row[0] if row else 0
            return f"OK ({count:,} conversations)"
    except Exception as exc:
        return f"unavailable ({_summarize_db_error(exc)})"


__all__ = [
    "HEALTH_TTL_SECONDS",
    "HealthCheck",
    "HealthReport",
    "VerifyStatus",
    "quick_health_summary",
    "get_health",
    "run_archive_health",
    "run_runtime_health",
]
