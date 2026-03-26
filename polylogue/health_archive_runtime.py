"""Runtime/archive health check families."""

from __future__ import annotations

from typing import Any, cast

from polylogue.config import Config
from polylogue.health_models import HealthCheck, VerifyStatus
from polylogue.storage.archive_debt import collect_archive_debt_statuses_sync
from polylogue.storage.backends.connection import connection_context, open_connection
from polylogue.storage.derived_status import collect_derived_model_statuses_sync
from polylogue.storage.index import index_status


def build_archive_runtime_checks(
    config: Config,
    *,
    deep: bool = False,
    collect_derived_statuses=collect_derived_model_statuses_sync,
) -> tuple[list[HealthCheck], dict[str, object], dict[str, object], str | None]:
    checks: list[HealthCheck] = []
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

    if db_error is not None:
        return checks, {}, {}, db_error

    with connection_context(config.db_path) as conn:
        derived_statuses = collect_derived_statuses(conn)
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

        transcript_embeddings = derived_statuses["transcript_embeddings"]
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
    return checks, derived_statuses, archive_debt, None


__all__ = ["build_archive_runtime_checks"]
