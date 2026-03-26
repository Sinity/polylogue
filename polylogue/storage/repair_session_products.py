"""Session-product derived repair flows."""

from __future__ import annotations

from polylogue.config import Config
from polylogue.maintenance_models import MaintenanceCategory

from .backends.connection import connection_context
from .repair_support import RepairResult
from .session_product_lifecycle import rebuild_session_products_sync, session_product_status_sync


def repair_session_products(config: Config, dry_run: bool = False) -> RepairResult:
    try:
        with connection_context(None) as conn:
            status = session_product_status_sync(conn)
            profile_merged_fts_pending = max(0, int(status["profile_row_count"]) - int(status["profile_merged_fts_count"]))
            profile_merged_fts_duplicates = max(0, int(status.get("profile_merged_fts_duplicate_count", 0)))
            profile_evidence_fts_pending = max(0, int(status["profile_row_count"]) - int(status["profile_evidence_fts_count"]))
            profile_evidence_fts_duplicates = max(0, int(status.get("profile_evidence_fts_duplicate_count", 0)))
            profile_inference_fts_pending = max(0, int(status["profile_row_count"]) - int(status["profile_inference_fts_count"]))
            profile_inference_fts_duplicates = max(0, int(status.get("profile_inference_fts_duplicate_count", 0)))
            profile_enrichment_fts_pending = max(0, int(status["profile_row_count"]) - int(status["profile_enrichment_fts_count"]))
            profile_enrichment_fts_duplicates = max(0, int(status.get("profile_enrichment_fts_duplicate_count", 0)))
            work_event_fts_pending = max(0, int(status["work_event_inference_count"]) - int(status["work_event_inference_fts_count"]))
            work_event_fts_duplicates = max(0, int(status.get("work_event_inference_fts_duplicate_count", 0)))
            thread_fts_pending = max(0, int(status["thread_count"]) - int(status["thread_fts_count"]))
            thread_fts_duplicates = max(0, int(status.get("thread_fts_duplicate_count", 0)))
            pending = (
                int(status["missing_profile_row_count"])
                + int(status["stale_profile_row_count"])
                + int(status["orphan_profile_row_count"])
                + int(status["stale_work_event_inference_count"])
                + int(status["orphan_work_event_inference_count"])
                + int(status["stale_phase_inference_count"])
                + int(status["orphan_phase_inference_count"])
                + int(status["stale_thread_count"])
                + int(status["orphan_thread_count"])
                + int(status["stale_tag_rollup_count"])
                + int(status["stale_day_summary_count"])
                + profile_merged_fts_pending
                + profile_merged_fts_duplicates
                + profile_evidence_fts_pending
                + profile_evidence_fts_duplicates
                + profile_inference_fts_pending
                + profile_inference_fts_duplicates
                + profile_enrichment_fts_pending
                + profile_enrichment_fts_duplicates
                + work_event_fts_pending
                + work_event_fts_duplicates
                + thread_fts_pending
                + thread_fts_duplicates
            )

            if dry_run:
                return RepairResult(
                    name="session_products",
                    category=MaintenanceCategory.DERIVED_REPAIR,
                    destructive=False,
                    repaired_count=pending,
                    success=True,
                    detail=(
                        "Would: session products already ready"
                        if pending == 0
                        and bool(status["profile_rows_ready"])
                        and bool(status["profile_merged_fts_ready"])
                        and bool(status["profile_evidence_fts_ready"])
                        and bool(status["profile_inference_fts_ready"])
                        and bool(status["profile_enrichment_fts_ready"])
                        and bool(status["work_event_inference_rows_ready"])
                        and bool(status["work_event_inference_fts_ready"])
                        and bool(status["phase_inference_rows_ready"])
                        and bool(status["threads_ready"])
                        and bool(status["threads_fts_ready"])
                        and bool(status["tag_rollups_ready"])
                        and bool(status["day_summaries_ready"])
                        and bool(status["week_summaries_ready"])
                        else (
                            "Would: rebuild session products "
                            f"(missing_profile_rows={int(status['missing_profile_row_count']):,}, "
                            f"stale_profile_rows={int(status['stale_profile_row_count']):,}, "
                            f"orphan_profile_rows={int(status['orphan_profile_row_count']):,}, "
                            f"stale_work_event_inference={int(status['stale_work_event_inference_count']):,}, "
                            f"orphan_work_event_inference={int(status['orphan_work_event_inference_count']):,}, "
                            f"stale_phase_inference={int(status['stale_phase_inference_count']):,}, "
                            f"orphan_phase_inference={int(status['orphan_phase_inference_count']):,}, "
                            f"stale_threads={int(status['stale_thread_count']):,}, "
                            f"orphan_threads={int(status['orphan_thread_count']):,}, "
                            f"stale_tag_rollups={int(status['stale_tag_rollup_count']):,}, "
                            f"stale_day_summaries={int(status['stale_day_summary_count']):,}, "
                            f"profile_merged_fts_pending={profile_merged_fts_pending:,}, "
                            f"profile_merged_fts_duplicates={profile_merged_fts_duplicates:,}, "
                            f"profile_evidence_fts_pending={profile_evidence_fts_pending:,}, "
                            f"profile_evidence_fts_duplicates={profile_evidence_fts_duplicates:,}, "
                            f"profile_inference_fts_pending={profile_inference_fts_pending:,}, "
                            f"profile_inference_fts_duplicates={profile_inference_fts_duplicates:,}, "
                            f"profile_enrichment_fts_pending={profile_enrichment_fts_pending:,}, "
                            f"profile_enrichment_fts_duplicates={profile_enrichment_fts_duplicates:,}, "
                            f"work_event_fts_pending={work_event_fts_pending:,}, "
                            f"work_event_fts_duplicates={work_event_fts_duplicates:,}, "
                            f"thread_fts_pending={thread_fts_pending:,}, "
                            f"thread_fts_duplicates={thread_fts_duplicates:,})"
                        )
                    ),
                )

            rebuilt = rebuild_session_products_sync(conn)
            conn.commit()
            refreshed = session_product_status_sync(conn)
            success = (
                bool(refreshed["profile_rows_ready"])
                and bool(refreshed["profile_merged_fts_ready"])
                and bool(refreshed["profile_evidence_fts_ready"])
                and bool(refreshed["profile_inference_fts_ready"])
                and bool(refreshed["profile_enrichment_fts_ready"])
                and bool(refreshed["work_event_inference_rows_ready"])
                and bool(refreshed["work_event_inference_fts_ready"])
                and bool(refreshed["phase_inference_rows_ready"])
                and bool(refreshed["threads_ready"])
                and bool(refreshed["threads_fts_ready"])
                and bool(refreshed["tag_rollups_ready"])
                and bool(refreshed["day_summaries_ready"])
                and bool(refreshed["week_summaries_ready"])
            )
            return RepairResult(
                name="session_products",
                category=MaintenanceCategory.DERIVED_REPAIR,
                destructive=False,
                repaired_count=(
                    int(rebuilt["profiles"])
                    + int(rebuilt["work_events"])
                    + int(rebuilt["phases"])
                    + int(rebuilt["threads"])
                    + int(rebuilt["tag_rollups"])
                    + int(rebuilt["day_summaries"])
                ),
                success=success,
                detail=(
                    "Session products ready"
                    if success
                    else (
                        "Session products still incomplete: "
                        f"profile_rows={int(refreshed['profile_row_count']):,}/{int(refreshed['total_conversations']):,}, "
                        f"profile_merged_fts={int(refreshed['profile_merged_fts_count']):,}/{int(refreshed['profile_row_count']):,}, "
                        f"profile_evidence_fts={int(refreshed['profile_evidence_fts_count']):,}/{int(refreshed['profile_row_count']):,}, "
                        f"profile_inference_fts={int(refreshed['profile_inference_fts_count']):,}/{int(refreshed['profile_row_count']):,}, "
                        f"profile_enrichment_fts={int(refreshed['profile_enrichment_fts_count']):,}/{int(refreshed['profile_row_count']):,}, "
                        f"work_event_inference={int(refreshed['work_event_inference_count']):,}/{int(refreshed['expected_work_event_inference_count']):,}, "
                        f"work_event_inference_fts={int(refreshed['work_event_inference_fts_count']):,}/{int(refreshed['work_event_inference_count']):,}, "
                        f"phase_inference={int(refreshed['phase_inference_count']):,}/{int(refreshed['expected_phase_inference_count']):,}, "
                        f"threads={int(refreshed['thread_count']):,}/{int(refreshed['root_threads']):,}, "
                        f"thread_fts={int(refreshed['thread_fts_count']):,}/{int(refreshed['thread_count']):,}, "
                        f"tag_rollups={int(refreshed['tag_rollup_count']):,}/{int(refreshed['expected_tag_rollup_count']):,}, "
                        f"day_summaries={int(refreshed['day_summary_count']):,}/{int(refreshed['expected_day_summary_count']):,}"
                    )
                ),
            )
    except Exception as exc:
        return RepairResult(
            name="session_products",
            category=MaintenanceCategory.DERIVED_REPAIR,
            destructive=False,
            repaired_count=0,
            success=False,
            detail=f"Failed to repair session products: {exc}",
        )


def preview_session_products(*, count: int) -> RepairResult:
    return RepairResult(
        name="session_products",
        category=MaintenanceCategory.DERIVED_REPAIR,
        destructive=False,
        repaired_count=count,
        success=True,
        detail="Would: session products already ready" if count == 0 else f"Would: rebuild session-product rows/fts for {count:,} pending items",
    )


__all__ = ["preview_session_products", "repair_session_products"]
