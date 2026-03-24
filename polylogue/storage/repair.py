"""Database repair operations for orphaned and dangling data."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from polylogue.config import Config
from polylogue.logging import get_logger
from polylogue.maintenance_models import MaintenanceCategory

from .action_event_lifecycle import (
    action_event_read_model_status_sync,
    action_event_repair_candidates_sync,
    rebuild_action_event_read_model_sync,
    valid_action_event_source_ids_sync,
)
from .backends.connection import connection_context, default_db_path
from .fts_lifecycle import repair_fts_index_sync
from .session_product_lifecycle import rebuild_session_products_sync, session_product_status_sync

logger = get_logger(__name__)

SAFE_REPAIR_TARGETS = (
    "session_products",
    "action_event_read_model",
    "dangling_fts",
    "wal_checkpoint",
)
CLEANUP_TARGETS = (
    "orphaned_messages",
    "orphaned_content_blocks",
    "empty_conversations",
    "orphaned_attachments",
)
MAINTENANCE_TARGET_NAMES = SAFE_REPAIR_TARGETS + CLEANUP_TARGETS


@dataclass
class RepairResult:
    """Result of a repair operation."""

    name: str
    category: MaintenanceCategory
    destructive: bool
    repaired_count: int
    success: bool
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "category": self.category.value,
            "destructive": self.destructive,
            "repaired_count": self.repaired_count,
            "success": self.success,
            "detail": self.detail,
        }


def _run_repair(
    name: str,
    *,
    category: MaintenanceCategory,
    destructive: bool,
    count_sql: str,
    action_sql: str | None,
    dry_run: bool,
    conn: sqlite3.Connection,
) -> RepairResult:
    """Generic repair framework for data cleanup operations.

    Args:
        name: Name of the repair (used in logs and results)
        count_sql: SQL query that returns COUNT(*) to identify affected rows
        action_sql: SQL query to execute the repair (optional for dry-run-only repairs)
        dry_run: If True, count only; if False, execute action_sql
        conn: Database connection

    Returns:
        RepairResult with count and status
    """
    try:
        # Get count of affected rows
        count = conn.execute(count_sql).fetchone()[0]

        if dry_run:
            # Dry-run: just report count
            return RepairResult(
                name=name,
                category=category,
                destructive=destructive,
                repaired_count=count,
                success=True,
                detail=f"Would: {count} rows affected" if count else "Would: No issues found",
            )

        # Execute repair
        if action_sql:
            result = conn.execute(action_sql)
            conn.commit()
            return RepairResult(
                name=name,
                category=category,
                destructive=destructive,
                repaired_count=result.rowcount,
                success=True,
                detail=f"Repaired {result.rowcount} rows" if result.rowcount else "No repairs needed",
            )

        return RepairResult(
            name=name,
            category=category,
            destructive=destructive,
            repaired_count=0,
            success=True,
            detail="No action SQL provided",
        )
    except Exception as exc:
        return RepairResult(
            name=name,
            category=category,
            destructive=destructive,
            repaired_count=0,
            success=False,
            detail=f"Repair failed: {exc}",
        )


def repair_orphaned_messages(config: Config, dry_run: bool = False) -> RepairResult:
    """Delete messages that reference non-existent conversations."""
    with connection_context(None) as conn:
        # Two-step count for performance (avoids full table scan on 1.5M+ rows)
        orphan_cids = conn.execute(
            """
            SELECT DISTINCT conversation_id FROM messages
            WHERE NOT EXISTS (SELECT 1 FROM conversations c WHERE c.conversation_id = messages.conversation_id)
            """
        ).fetchall()

        if not orphan_cids:
            return RepairResult(
                name="orphaned_messages",
                category=MaintenanceCategory.ARCHIVE_CLEANUP,
                destructive=True,
                repaired_count=0,
                success=True,
                detail="No orphaned messages found",
            )

        placeholders = ",".join("?" for _ in orphan_cids)
        count_sql = f"SELECT COUNT(*) FROM messages WHERE conversation_id IN ({placeholders})"

        # Manually execute the count query for this case
        try:
            count = conn.execute(count_sql, [row[0] for row in orphan_cids]).fetchone()[0]
            if dry_run:
                return RepairResult(
                    name="orphaned_messages",
                    category=MaintenanceCategory.ARCHIVE_CLEANUP,
                    destructive=True,
                    repaired_count=count,
                    success=True,
                    detail=f"Would: Delete {count} orphaned messages" if count else "Would: No orphaned messages found",
                )

            result = conn.execute(
                f"DELETE FROM messages WHERE conversation_id IN ({placeholders})",
                [row[0] for row in orphan_cids],
            )
            conn.commit()
            return RepairResult(
                name="orphaned_messages",
                category=MaintenanceCategory.ARCHIVE_CLEANUP,
                destructive=True,
                repaired_count=result.rowcount,
                success=True,
                detail=f"Deleted {result.rowcount} orphaned messages" if result.rowcount else "No orphaned messages found",
            )
        except Exception as exc:
            return RepairResult(
                name="orphaned_messages",
                category=MaintenanceCategory.ARCHIVE_CLEANUP,
                destructive=True,
                repaired_count=0,
                success=False,
                detail=f"Failed to delete orphaned messages: {exc}",
            )


def preview_orphaned_messages(*, count: int) -> RepairResult:
    """Build a dry-run orphaned-messages result from a known count."""
    return RepairResult(
        name="orphaned_messages",
        category=MaintenanceCategory.ARCHIVE_CLEANUP,
        destructive=True,
        repaired_count=count,
        success=True,
        detail=(
            f"Would: Delete {count} orphaned messages"
            if count
            else "Would: No orphaned messages found"
        ),
    )


def repair_empty_conversations(config: Config, dry_run: bool = False) -> RepairResult:
    """Delete conversations that have no messages."""
    with connection_context(None) as conn:
        return _run_repair(
            name="empty_conversations",
            category=MaintenanceCategory.ARCHIVE_CLEANUP,
            destructive=True,
            count_sql="SELECT COUNT(*) FROM conversations c WHERE NOT EXISTS (SELECT 1 FROM messages m WHERE m.conversation_id = c.conversation_id)",
            action_sql="DELETE FROM conversations WHERE NOT EXISTS (SELECT 1 FROM messages m WHERE m.conversation_id = conversations.conversation_id)",
            dry_run=dry_run,
            conn=conn,
        )


def preview_empty_conversations(*, count: int) -> RepairResult:
    """Build a dry-run empty-conversation result from a known count."""
    return RepairResult(
        name="empty_conversations",
        category=MaintenanceCategory.ARCHIVE_CLEANUP,
        destructive=True,
        repaired_count=count,
        success=True,
        detail=(
            f"Would: {count} rows affected"
            if count
            else "Would: No issues found"
        ),
    )


def repair_orphaned_content_blocks(config: Config, dry_run: bool = False) -> RepairResult:
    """Delete content blocks whose parent conversation or message no longer exists."""
    with connection_context(None) as conn:
        return _run_repair(
            name="orphaned_content_blocks",
            category=MaintenanceCategory.ARCHIVE_CLEANUP,
            destructive=True,
            count_sql="""
                SELECT COUNT(*)
                FROM content_blocks cb
                WHERE NOT EXISTS (SELECT 1 FROM conversations c WHERE c.conversation_id = cb.conversation_id)
                   OR NOT EXISTS (SELECT 1 FROM messages m WHERE m.message_id = cb.message_id)
            """,
            action_sql="""
                DELETE FROM content_blocks
                WHERE NOT EXISTS (SELECT 1 FROM conversations c WHERE c.conversation_id = content_blocks.conversation_id)
                   OR NOT EXISTS (SELECT 1 FROM messages m WHERE m.message_id = content_blocks.message_id)
            """,
            dry_run=dry_run,
            conn=conn,
        )


def preview_orphaned_content_blocks(*, count: int) -> RepairResult:
    """Build a dry-run orphaned-content-block result from a known count."""
    return RepairResult(
        name="orphaned_content_blocks",
        category=MaintenanceCategory.ARCHIVE_CLEANUP,
        destructive=True,
        repaired_count=count,
        success=True,
        detail=(
            f"Would: {count} rows affected"
            if count
            else "Would: No issues found"
        ),
    )


def repair_dangling_fts(config: Config, dry_run: bool = False) -> RepairResult:
    """Rebuild FTS index entries that are out of sync with messages table."""
    try:
        with connection_context(None) as conn:
            # Check if FTS table exists
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
                # Fast estimate: compare row counts
                # Use docsize backing table — COUNT(*) on FTS virtual table is 15s+ on large DBs
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

            # Delete FTS entries that don't have corresponding messages
            result = conn.execute(
                """
                DELETE FROM messages_fts
                WHERE rowid IN (
                    SELECT f.rowid FROM messages_fts f
                    WHERE NOT EXISTS (SELECT 1 FROM messages m WHERE m.rowid = f.rowid)
                )
                """
            )
            deleted = result.rowcount

            # Insert missing entries into FTS
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
    """Build a dry-run FTS repair result from a known pending-row count."""
    return RepairResult(
        name="dangling_fts",
        category=MaintenanceCategory.DERIVED_REPAIR,
        destructive=False,
        repaired_count=count,
        success=True,
        detail=(
            f"Would: FTS sync pending {count:,} rows"
            if count
            else "FTS index in sync"
        ),
    )


def repair_session_products(config: Config, dry_run: bool = False) -> RepairResult:
    """Repair durable session-profile, phase, work-event, and work-thread products."""
    try:
        with connection_context(None) as conn:
            status = session_product_status_sync(conn)
            profile_fts_pending = max(
                0,
                int(status["profile_count"]) - int(status["profile_fts_count"]),
            )
            profile_fts_duplicates = max(0, int(status.get("profile_fts_duplicate_count", 0)))
            work_event_fts_pending = max(
                0,
                int(status["work_event_count"]) - int(status["work_event_fts_count"]),
            )
            work_event_fts_duplicates = max(0, int(status.get("work_event_fts_duplicate_count", 0)))
            thread_fts_pending = max(
                0,
                int(status["thread_count"]) - int(status["thread_fts_count"]),
            )
            thread_fts_duplicates = max(0, int(status.get("thread_fts_duplicate_count", 0)))
            pending = (
                int(status["missing_profile_count"])
                + int(status["stale_profile_count"])
                + int(status["orphan_profile_count"])
                + int(status["stale_work_event_count"])
                + int(status["orphan_work_event_count"])
                + int(status["stale_phase_count"])
                + int(status["orphan_phase_count"])
                + int(status["stale_thread_count"])
                + int(status["orphan_thread_count"])
                + int(status["stale_tag_rollup_count"])
                + int(status["stale_day_summary_count"])
                + profile_fts_pending
                + profile_fts_duplicates
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
                        and bool(status["profiles_ready"])
                        and bool(status["profiles_fts_ready"])
                        and bool(status["work_events_ready"])
                        and bool(status["work_events_fts_ready"])
                        and bool(status["phases_ready"])
                        and bool(status["threads_ready"])
                        and bool(status["threads_fts_ready"])
                        and bool(status["tag_rollups_ready"])
                        and bool(status["day_summaries_ready"])
                        and bool(status["week_summaries_ready"])
                        else (
                            "Would: rebuild session products "
                            f"(missing_profiles={int(status['missing_profile_count']):,}, "
                            f"stale_profiles={int(status['stale_profile_count']):,}, "
                            f"orphan_profiles={int(status['orphan_profile_count']):,}, "
                            f"stale_work_events={int(status['stale_work_event_count']):,}, "
                            f"orphan_work_events={int(status['orphan_work_event_count']):,}, "
                            f"stale_phases={int(status['stale_phase_count']):,}, "
                            f"orphan_phases={int(status['orphan_phase_count']):,}, "
                            f"stale_threads={int(status['stale_thread_count']):,}, "
                            f"orphan_threads={int(status['orphan_thread_count']):,}, "
                            f"stale_tag_rollups={int(status['stale_tag_rollup_count']):,}, "
                            f"stale_day_summaries={int(status['stale_day_summary_count']):,}, "
                            f"profile_fts_pending={profile_fts_pending:,}, "
                            f"profile_fts_duplicates={profile_fts_duplicates:,}, "
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
                bool(refreshed["profiles_ready"])
                and bool(refreshed["profiles_fts_ready"])
                and bool(refreshed["work_events_ready"])
                and bool(refreshed["work_events_fts_ready"])
                and bool(refreshed["phases_ready"])
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
                        f"profiles={int(refreshed['profile_count']):,}/"
                        f"{int(refreshed['total_conversations']):,}, "
                        f"profile_fts={int(refreshed['profile_fts_count']):,}/"
                        f"{int(refreshed['profile_count']):,}, "
                        f"work_events={int(refreshed['work_event_count']):,}/"
                        f"{int(refreshed['expected_work_event_count']):,}, "
                        f"work_event_fts={int(refreshed['work_event_fts_count']):,}/"
                        f"{int(refreshed['work_event_count']):,}, "
                        f"phases={int(refreshed['phase_count']):,}/"
                        f"{int(refreshed['expected_phase_count']):,}, "
                        f"threads={int(refreshed['thread_count']):,}/"
                        f"{int(refreshed['root_threads']):,}, "
                        f"thread_fts={int(refreshed['thread_fts_count']):,}/"
                        f"{int(refreshed['thread_count']):,}, "
                        f"tag_rollups={int(refreshed['tag_rollup_count']):,}/"
                        f"{int(refreshed['expected_tag_rollup_count']):,}, "
                        f"day_summaries={int(refreshed['day_summary_count']):,}/"
                        f"{int(refreshed['expected_day_summary_count']):,}"
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
    """Build a dry-run session-product repair result from a known pending count."""
    return RepairResult(
        name="session_products",
        category=MaintenanceCategory.DERIVED_REPAIR,
        destructive=False,
        repaired_count=count,
        success=True,
        detail=(
            "Would: session products already ready"
            if count == 0
            else f"Would: rebuild session-product rows/fts for {count:,} pending items"
        ),
    )


def repair_action_event_read_model(config: Config, dry_run: bool = False) -> RepairResult:
    """Repair the durable action-event read model and its action FTS rows."""
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
                        f"action FTS {refreshed['action_fts_count']:,}/"
                        f"{refreshed['count']:,}"
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
    """Build a dry-run action-event repair result from a known pending-row count."""
    return RepairResult(
        name="action_event_read_model",
        category=MaintenanceCategory.DERIVED_REPAIR,
        destructive=False,
        repaired_count=count,
        success=True,
        detail=(
            "Would: action-event read model already ready"
            if count == 0
            else f"Would: repair action-event rows/fts for {count:,} pending items"
        ),
    )


def repair_orphaned_attachments(config: Config, dry_run: bool = False) -> RepairResult:
    """Delete attachments that are not referenced by any message or have orphaned refs."""
    try:
        with connection_context(None) as conn:
            if dry_run:
                # Count distinct orphaned refs (a single ref can be orphaned on both axes)
                orphaned_refs = conn.execute(
                    """
                    SELECT COUNT(*) FROM attachment_refs ar
                    WHERE (ar.message_id IS NOT NULL AND NOT EXISTS (SELECT 1 FROM messages m WHERE m.message_id = ar.message_id))
                       OR NOT EXISTS (SELECT 1 FROM conversations c WHERE c.conversation_id = ar.conversation_id)
                    """
                ).fetchone()[0]

                atts_deleted = conn.execute(
                    """
                    SELECT COUNT(*) FROM attachments a
                    WHERE NOT EXISTS (SELECT 1 FROM attachment_refs ar WHERE ar.attachment_id = a.attachment_id)
                    """
                ).fetchone()[0]

                total = orphaned_refs + atts_deleted
                return RepairResult(
                    name="orphaned_attachments",
                    category=MaintenanceCategory.ARCHIVE_CLEANUP,
                    destructive=True,
                    repaired_count=total,
                    success=True,
                    detail=f"Would: Clean {orphaned_refs} orphaned refs, {atts_deleted} unreferenced attachments",
                )

            # First, delete attachment_refs that point to non-existent messages
            ref_result = conn.execute(
                """
                DELETE FROM attachment_refs
                WHERE message_id IS NOT NULL AND NOT EXISTS (SELECT 1 FROM messages m WHERE m.message_id = attachment_refs.message_id)
                """
            )
            refs_deleted = ref_result.rowcount

            # Delete attachment_refs that point to non-existent conversations
            conv_ref_result = conn.execute(
                """
                DELETE FROM attachment_refs
                WHERE NOT EXISTS (SELECT 1 FROM conversations c WHERE c.conversation_id = attachment_refs.conversation_id)
                """
            )
            conv_refs_deleted = conv_ref_result.rowcount

            # Delete attachments that have no remaining refs
            att_result = conn.execute(
                """
                DELETE FROM attachments
                WHERE NOT EXISTS (SELECT 1 FROM attachment_refs ar WHERE ar.attachment_id = attachments.attachment_id)
                """
            )
            atts_deleted = att_result.rowcount

            conn.commit()

            total = refs_deleted + conv_refs_deleted + atts_deleted
            return RepairResult(
                name="orphaned_attachments",
                category=MaintenanceCategory.ARCHIVE_CLEANUP,
                destructive=True,
                repaired_count=total,
                success=True,
                detail=f"Cleaned {refs_deleted} orphaned refs, {conv_refs_deleted} conv refs, {atts_deleted} attachments",
            )
    except Exception as exc:
        return RepairResult(
            name="orphaned_attachments",
            category=MaintenanceCategory.ARCHIVE_CLEANUP,
            destructive=True,
            repaired_count=0,
            success=False,
            detail=f"Failed to clean orphaned attachments: {exc}",
        )


def repair_wal_checkpoint(config: Config, dry_run: bool = False) -> RepairResult:
    """Force WAL checkpoint to resolve busy pages and reclaim WAL space."""
    try:
        if dry_run:
            # All PRAGMA wal_checkpoint modes actually perform a checkpoint.
            # For true dry-run, inspect the WAL file on disk instead.
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
            result = conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            row = result.fetchone()
            # wal_checkpoint returns (busy, log, checkpointed)
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


def run_safe_repairs(
    config: Config,
    dry_run: bool = False,
    *,
    preview_counts: dict[str, int] | None = None,
    targets: tuple[str, ...] = (),
) -> list[RepairResult]:
    """Run non-destructive derived-data and database maintenance repairs."""
    preview_counts = preview_counts or {}
    selected = set(targets) if targets else set(SAFE_REPAIR_TARGETS)
    results: list[RepairResult] = []
    if "session_products" in selected:
        results.append(
            preview_session_products(count=preview_counts["session_products"])
            if dry_run and "session_products" in preview_counts
            else repair_session_products(config, dry_run=dry_run)
        )
    if "action_event_read_model" in selected:
        results.append(
            preview_action_event_read_model(count=preview_counts["action_event_read_model"])
            if dry_run and "action_event_read_model" in preview_counts
            else repair_action_event_read_model(config, dry_run=dry_run)
        )
    if "dangling_fts" in selected:
        results.append(
            preview_dangling_fts(count=preview_counts["dangling_fts"])
            if dry_run and "dangling_fts" in preview_counts
            else repair_dangling_fts(config, dry_run=dry_run)
        )
    if "wal_checkpoint" in selected:
        results.append(repair_wal_checkpoint(config, dry_run=dry_run))
    return results


def run_archive_cleanup(
    config: Config,
    dry_run: bool = False,
    *,
    preview_counts: dict[str, int] | None = None,
    targets: tuple[str, ...] = (),
) -> list[RepairResult]:
    """Run destructive archive cleanup operations."""
    preview_counts = preview_counts or {}
    selected = set(targets) if targets else set(CLEANUP_TARGETS)
    results: list[RepairResult] = []
    if "orphaned_messages" in selected:
        results.append(
            preview_orphaned_messages(count=preview_counts["orphaned_messages"])
            if dry_run and "orphaned_messages" in preview_counts
            else repair_orphaned_messages(config, dry_run=dry_run)
        )
    if "orphaned_content_blocks" in selected:
        results.append(
            preview_orphaned_content_blocks(count=preview_counts["orphaned_content_blocks"])
            if dry_run and "orphaned_content_blocks" in preview_counts
            else repair_orphaned_content_blocks(config, dry_run=dry_run)
        )
    if "empty_conversations" in selected:
        results.append(
            preview_empty_conversations(count=preview_counts["empty_conversations"])
            if dry_run and "empty_conversations" in preview_counts
            else repair_empty_conversations(config, dry_run=dry_run)
        )
    if "orphaned_attachments" in selected:
        results.append(repair_orphaned_attachments(config, dry_run=dry_run))
    return results


def run_selected_maintenance(
    config: Config,
    *,
    repair: bool,
    cleanup: bool,
    dry_run: bool = False,
    preview_counts: dict[str, int] | None = None,
    targets: tuple[str, ...] = (),
) -> list[RepairResult]:
    """Run the selected maintenance operations and return their results."""
    results: list[RepairResult] = []
    repair_targets = tuple(name for name in targets if name in SAFE_REPAIR_TARGETS)
    cleanup_targets = tuple(name for name in targets if name in CLEANUP_TARGETS)
    if repair:
        results.extend(
            run_safe_repairs(
                config,
                dry_run=dry_run,
                preview_counts=preview_counts,
                targets=repair_targets,
            )
        )
    if cleanup:
        results.extend(
            run_archive_cleanup(
                config,
                dry_run=dry_run,
                preview_counts=preview_counts,
                targets=cleanup_targets,
            )
        )
    return results


__all__ = [
    "RepairResult",
    "CLEANUP_TARGETS",
    "MAINTENANCE_TARGET_NAMES",
    "SAFE_REPAIR_TARGETS",
    "repair_orphaned_messages",
    "repair_orphaned_content_blocks",
    "repair_empty_conversations",
    "repair_session_products",
    "repair_action_event_read_model",
    "repair_dangling_fts",
    "repair_orphaned_attachments",
    "repair_wal_checkpoint",
    "run_archive_cleanup",
    "run_safe_repairs",
    "run_selected_maintenance",
]
