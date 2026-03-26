"""Derived-data and database-maintenance repair flows."""

from __future__ import annotations

from pathlib import Path

from polylogue.config import Config
from polylogue.maintenance_models import MaintenanceCategory

from .action_event_lifecycle import (
    action_event_read_model_status_sync,
    action_event_repair_candidates_sync,
    rebuild_action_event_read_model_sync,
    valid_action_event_source_ids_sync,
)
from .backends.connection import connection_context, default_db_path
from .fts_lifecycle import repair_fts_index_sync
from .repair_support import RepairResult
from .session_product_lifecycle import rebuild_session_products_sync, session_product_status_sync


def repair_dangling_fts(config: Config, dry_run: bool = False) -> RepairResult:
    try:
        with connection_context(None) as conn:
            fts_exists = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='messages_fts'"
            ).fetchone()

            if not fts_exists:
                return RepairResult(
                    name="dangling_fts",
                    category=MaintenanceCategory.DERIVED_REPAIR,
                    destructive=False,
                    repaired_count=0,
                    success=True,
                    detail="FTS table does not exist, skipping",
                )

            if dry_run:
                msg_count = conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
                fts_count = conn.execute("SELECT COUNT(*) FROM messages_fts_docsize").fetchone()[0]
                diff = abs(msg_count - fts_count)

                if diff == 0:
                    return RepairResult(
                        name="dangling_fts",
                        category=MaintenanceCategory.DERIVED_REPAIR,
                        destructive=False,
                        repaired_count=0,
                        success=True,
                        detail="FTS index in sync",
                    )

                return RepairResult(
                    name="dangling_fts",
                    category=MaintenanceCategory.DERIVED_REPAIR,
                    destructive=False,
                    repaired_count=diff,
                    success=True,
                    detail=f"Would: FTS sync: {msg_count:,} messages vs {fts_count:,} indexed ({diff:,} difference)",
                )

            deleted = conn.execute(
                """
                DELETE FROM messages_fts
                WHERE rowid IN (
                    SELECT f.rowid FROM messages_fts f
                    WHERE NOT EXISTS (SELECT 1 FROM messages m WHERE m.rowid = f.rowid)
                )
                """
            ).rowcount

            inserted = conn.execute(
                """
                INSERT INTO messages_fts (rowid, message_id, conversation_id, text)
                SELECT m.rowid, m.message_id, m.conversation_id, m.text FROM messages m
                WHERE NOT EXISTS (SELECT 1 FROM messages_fts f WHERE f.rowid = m.rowid)
                """
            ).rowcount

            conn.commit()

            total = deleted + inserted
            return RepairResult(
                name="dangling_fts",
                category=MaintenanceCategory.DERIVED_REPAIR,
                destructive=False,
                repaired_count=total,
                success=True,
                detail=f"FTS sync: deleted {deleted} orphaned, added {inserted} missing entries",
            )
    except Exception as exc:
        return RepairResult(
            name="dangling_fts",
            category=MaintenanceCategory.DERIVED_REPAIR,
            destructive=False,
            repaired_count=0,
            success=False,
            detail=f"Failed to repair FTS index: {exc}",
        )


def preview_dangling_fts(*, count: int) -> RepairResult:
    return RepairResult(
        name="dangling_fts",
        category=MaintenanceCategory.DERIVED_REPAIR,
        destructive=False,
        repaired_count=count,
        success=True,
        detail=f"Would: FTS sync pending {count:,} rows" if count else "FTS index in sync",
    )


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


def repair_action_event_read_model(config: Config, dry_run: bool = False) -> RepairResult:
    try:
        with connection_context(None) as conn:
            status = action_event_read_model_status_sync(conn)
            candidate_ids = action_event_repair_candidates_sync(conn)

            missing_conversations = max(
                0,
                int(status["valid_source_conversation_count"]) - int(status["materialized_conversation_count"]),
            )
            stale_conversations = int(status["stale_count"])
            action_fts_pending = max(0, int(status["count"]) - int(status["action_fts_count"]))
            pending = max(len(candidate_ids), missing_conversations + stale_conversations) + action_fts_pending

            if dry_run:
                return RepairResult(
                    name="action_event_read_model",
                    category=MaintenanceCategory.DERIVED_REPAIR,
                    destructive=False,
                    repaired_count=pending,
                    success=True,
                    detail=(
                        "Would: action-event read model already ready"
                        if pending == 0 and bool(status["rows_ready"]) and bool(status["action_fts_ready"])
                        else (
                            "Would: repair action-event rows for "
                            f"{len(candidate_ids):,} conversations; "
                            f"action FTS pending {action_fts_pending:,}"
                        )
                    ),
                )

            repaired = 0
            if candidate_ids:
                repaired = rebuild_action_event_read_model_sync(conn, conversation_ids=candidate_ids)

            if not bool(status["action_fts_ready"]):
                repair_targets = candidate_ids or valid_action_event_source_ids_sync(conn)
                if repair_targets:
                    repair_fts_index_sync(conn, repair_targets)

            conn.commit()
            refreshed = action_event_read_model_status_sync(conn)
            return RepairResult(
                name="action_event_read_model",
                category=MaintenanceCategory.DERIVED_REPAIR,
                destructive=False,
                repaired_count=repaired + action_fts_pending,
                success=bool(refreshed["ready"]),
                detail=(
                    "Action-event read model ready"
                    if refreshed["ready"]
                    else (
                        "Action-event read model still incomplete: "
                        f"{refreshed['materialized_conversation_count']:,}/"
                        f"{refreshed['valid_source_conversation_count']:,} conversations, "
                        f"action FTS {refreshed['action_fts_count']:,}/{refreshed['count']:,}"
                    )
                ),
            )
    except Exception as exc:
        return RepairResult(
            name="action_event_read_model",
            category=MaintenanceCategory.DERIVED_REPAIR,
            destructive=False,
            repaired_count=0,
            success=False,
            detail=f"Failed to repair action-event read model: {exc}",
        )


def preview_action_event_read_model(*, count: int) -> RepairResult:
    return RepairResult(
        name="action_event_read_model",
        category=MaintenanceCategory.DERIVED_REPAIR,
        destructive=False,
        repaired_count=count,
        success=True,
        detail="Would: action-event read model already ready" if count == 0 else f"Would: repair action-event rows/fts for {count:,} pending items",
    )


def repair_wal_checkpoint(config: Config, dry_run: bool = False) -> RepairResult:
    try:
        if dry_run:
            db_path = default_db_path()
            wal_path = Path(str(db_path) + "-wal")
            if wal_path.exists():
                wal_size = wal_path.stat().st_size
                pages_estimate = wal_size // 4096
                return RepairResult(
                    name="wal_checkpoint",
                    category=MaintenanceCategory.DATABASE_MAINTENANCE,
                    destructive=False,
                    repaired_count=pages_estimate,
                    success=True,
                    detail=f"Would: WAL checkpoint (~{pages_estimate} pages, {wal_size:,} bytes)",
                )
            return RepairResult(
                name="wal_checkpoint",
                category=MaintenanceCategory.DATABASE_MAINTENANCE,
                destructive=False,
                repaired_count=0,
                success=True,
                detail="Would: No WAL file present, nothing to checkpoint",
            )

        with connection_context(None) as conn:
            row = conn.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
            busy, log, checkpointed = row[0], row[1], row[2]
            if busy:
                return RepairResult(
                    name="wal_checkpoint",
                    category=MaintenanceCategory.DATABASE_MAINTENANCE,
                    destructive=False,
                    repaired_count=0,
                    success=False,
                    detail=f"WAL checkpoint had busy pages: {busy} busy, {log} log, {checkpointed} checkpointed",
                )
            return RepairResult(
                name="wal_checkpoint",
                category=MaintenanceCategory.DATABASE_MAINTENANCE,
                destructive=False,
                repaired_count=checkpointed if checkpointed > 0 else 0,
                success=True,
                detail=f"WAL checkpoint complete: {checkpointed} pages checkpointed",
            )
    except Exception as exc:
        return RepairResult(
            name="wal_checkpoint",
            category=MaintenanceCategory.DATABASE_MAINTENANCE,
            destructive=False,
            repaired_count=0,
            success=False,
            detail=f"WAL checkpoint failed: {exc}",
        )


__all__ = [
    "preview_action_event_read_model",
    "preview_dangling_fts",
    "preview_session_products",
    "repair_action_event_read_model",
    "repair_dangling_fts",
    "repair_session_products",
    "repair_wal_checkpoint",
]
