"""Archive/data health checks."""

from __future__ import annotations

from typing import Any, cast

from polylogue.logging import get_logger

from .config import Config
from .health_cache import load_cached_report, write_cache
from .health_models import HealthCheck, HealthReport, VerifyStatus
from .lib.provider_identity import CORE_SCHEMA_PROVIDERS
from .sources.drive_auth import default_credentials_path, default_token_path
from .storage.archive_debt import collect_archive_debt_statuses_sync
from .storage.backends.connection import connection_context, open_connection
from .storage.derived_status import collect_derived_model_statuses_sync
from .storage.index import index_status

logger = get_logger(__name__)


def run_archive_health(config: Config, *, deep: bool = False) -> HealthReport:
    """Run comprehensive system health and data verification checks."""
    checks: list[HealthCheck] = []

    checks.append(HealthCheck("config", VerifyStatus.OK, summary="Zero-config (XDG paths)"))

    for path_name in ("archive_root", "render_root"):
        path = getattr(config, path_name)
        if path.exists():
            checks.append(HealthCheck(path_name, VerifyStatus.OK, summary=str(path)))
        else:
            checks.append(HealthCheck(path_name, VerifyStatus.WARNING, summary=f"Missing {path}"))

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

    if db_error is None:
        with connection_context(None) as conn:
            derived_statuses = collect_derived_model_statuses_sync(conn)
            archive_debt = collect_archive_debt_statuses_sync(conn, derived_statuses=derived_statuses)
            for debt_name in (
                "orphaned_messages",
                "orphaned_content_blocks",
                "orphaned_attachments",
            ):
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
                    summary=(
                        "No duplicates"
                        if dup_conv == 0
                        else f"{dup_conv} duplicate conversation IDs"
                    ),
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
            action_rows = derived_statuses["action_events"]
            action_fts = derived_statuses["action_events_fts"]
            message_fts = derived_statuses["messages_fts"]
            transcript_embeddings = derived_statuses["transcript_embeddings"]
            retrieval_evidence = derived_statuses["retrieval_evidence"]
            retrieval_inference = derived_statuses["retrieval_inference"]
            session_profile_rows = derived_statuses.get("session_profile_rows")
            session_profile_evidence_fts = derived_statuses.get("session_profile_evidence_fts")
            session_profile_inference_fts = derived_statuses.get("session_profile_inference_fts")
            session_work_event_inference = derived_statuses.get("session_work_event_inference")
            session_work_event_inference_fts = derived_statuses.get("session_work_event_inference_fts")
            session_phase_inference = derived_statuses.get("session_phase_inference")
            session_tag_rollups = derived_statuses.get("session_tag_rollups")
            day_session_summaries = derived_statuses.get("day_session_summaries")
            week_session_summaries = derived_statuses.get("week_session_summaries")
            checks.append(
                HealthCheck(
                    "action_event_read_model",
                    VerifyStatus.OK if action_rows.ready else VerifyStatus.WARNING,
                    count=action_rows.materialized_rows,
                    summary=action_rows.detail,
                )
            )
            checks.append(
                HealthCheck(
                    "action_event_fts",
                    VerifyStatus.OK if action_fts.ready else VerifyStatus.WARNING,
                    count=action_fts.materialized_rows,
                    summary=action_fts.detail,
                )
            )
            checks.append(
                HealthCheck(
                    "fts_sync",
                    VerifyStatus.OK if message_fts.ready else VerifyStatus.WARNING,
                    count=message_fts.materialized_rows if message_fts.ready else message_fts.pending_rows,
                    summary=message_fts.detail,
                )
            )
            embedding_status = VerifyStatus.OK if transcript_embeddings.ready else VerifyStatus.WARNING
            checks.append(
                HealthCheck(
                    "transcript_embeddings",
                    embedding_status,
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
            checks.append(
                HealthCheck(
                    "retrieval_evidence",
                    VerifyStatus.OK if retrieval_evidence.ready else VerifyStatus.WARNING,
                    count=retrieval_evidence.pending_rows,
                    summary=retrieval_evidence.detail,
                )
            )
            checks.append(
                HealthCheck(
                    "retrieval_inference",
                    VerifyStatus.OK if retrieval_inference.ready else VerifyStatus.WARNING,
                    count=retrieval_inference.pending_rows,
                    summary=retrieval_inference.detail,
                )
            )
            if session_profile_rows is not None:
                checks.append(
                    HealthCheck(
                        "session_profile_rows",
                        VerifyStatus.OK if session_profile_rows.ready else VerifyStatus.WARNING,
                        count=session_profile_rows.materialized_documents,
                        summary=session_profile_rows.detail,
                    )
                )
            if session_profile_evidence_fts is not None:
                checks.append(
                    HealthCheck(
                        "session_profile_evidence_fts",
                        VerifyStatus.OK if session_profile_evidence_fts.ready else VerifyStatus.WARNING,
                        count=session_profile_evidence_fts.materialized_rows,
                        summary=session_profile_evidence_fts.detail,
                    )
                )
            if session_profile_inference_fts is not None:
                checks.append(
                    HealthCheck(
                        "session_profile_inference_fts",
                        VerifyStatus.OK if session_profile_inference_fts.ready else VerifyStatus.WARNING,
                        count=session_profile_inference_fts.materialized_rows,
                        summary=session_profile_inference_fts.detail,
                    )
                )
            if session_work_event_inference is not None:
                checks.append(
                    HealthCheck(
                        "session_work_event_inference",
                        VerifyStatus.OK if session_work_event_inference.ready else VerifyStatus.WARNING,
                        count=session_work_event_inference.materialized_rows,
                        summary=session_work_event_inference.detail,
                    )
                )
            if session_work_event_inference_fts is not None:
                checks.append(
                    HealthCheck(
                        "session_work_event_inference_fts",
                        VerifyStatus.OK if session_work_event_inference_fts.ready else VerifyStatus.WARNING,
                        count=session_work_event_inference_fts.materialized_rows,
                        summary=session_work_event_inference_fts.detail,
                    )
                )
            if session_tag_rollups is not None:
                checks.append(
                    HealthCheck(
                        "session_tag_rollups",
                        VerifyStatus.OK if session_tag_rollups.ready else VerifyStatus.WARNING,
                        count=session_tag_rollups.materialized_rows,
                        summary=session_tag_rollups.detail,
                    )
                )
            if session_phase_inference is not None:
                checks.append(
                    HealthCheck(
                        "session_phase_inference",
                        VerifyStatus.OK if session_phase_inference.ready else VerifyStatus.WARNING,
                        count=session_phase_inference.materialized_rows,
                        summary=session_phase_inference.detail,
                    )
                )
            if day_session_summaries is not None:
                checks.append(
                    HealthCheck(
                        "day_session_summaries",
                        VerifyStatus.OK if day_session_summaries.ready else VerifyStatus.WARNING,
                        count=day_session_summaries.materialized_rows,
                        summary=day_session_summaries.detail,
                    )
                )
            if week_session_summaries is not None:
                checks.append(
                    HealthCheck(
                        "week_session_summaries",
                        VerifyStatus.OK if week_session_summaries.ready else VerifyStatus.WARNING,
                        count=week_session_summaries.materialized_rows,
                        summary=week_session_summaries.detail,
                    )
                )
    else:
        derived_statuses = {}
        archive_debt = {}

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
                checks.append(
                    HealthCheck(f"source:{source.name}", VerifyStatus.OK, summary=str(source.path))
                )
            else:
                checks.append(
                    HealthCheck(
                        f"source:{source.name}",
                        VerifyStatus.WARNING,
                        summary=f"missing path: {source.path}",
                    )
                )

    try:
        from .schemas.registry import SchemaRegistry

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
            checks.append(
                HealthCheck(
                    "schemas_freshness",
                    VerifyStatus.OK,
                    summary="All schemas current",
                )
            )
    except Exception as exc:
        checks.append(
            HealthCheck("schemas", VerifyStatus.WARNING, summary=f"Schema check failed: {exc}")
        )

    report = HealthReport(checks=checks, derived_models=derived_statuses, archive_debt=archive_debt)
    write_cache(config.archive_root, report)
    return report


def get_health(config: Config, *, deep: bool = False, use_cached: bool = False) -> HealthReport:
    """Get an archive health report, optionally using the cached report."""
    if use_cached and not deep:
        cached_report = load_cached_report(config.archive_root)
        if cached_report is not None:
            return cached_report

    report = run_archive_health(config, deep=deep)
    return report
