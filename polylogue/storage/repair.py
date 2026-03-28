"""Database repair operations for orphaned and dangling data."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from polylogue.config import Config
from polylogue.logging import get_logger
from .backends.connection import connection_context, default_db_path

logger = get_logger(__name__)


@dataclass
class RepairResult:
    """Result of a repair operation."""

    name: str
    repaired_count: int
    success: bool
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "repaired_count": self.repaired_count,
            "success": self.success,
            "detail": self.detail,
        }


def _run_repair(
    name: str,
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
                repaired_count=result.rowcount,
                success=True,
                detail=f"Repaired {result.rowcount} rows" if result.rowcount else "No repairs needed",
            )

        return RepairResult(
            name=name,
            repaired_count=0,
            success=True,
            detail="No action SQL provided",
        )
    except Exception as exc:
        return RepairResult(
            name=name,
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
                repaired_count=result.rowcount,
                success=True,
                detail=f"Deleted {result.rowcount} orphaned messages" if result.rowcount else "No orphaned messages found",
            )
        except Exception as exc:
            return RepairResult(
                name="orphaned_messages",
                repaired_count=0,
                success=False,
                detail=f"Failed to delete orphaned messages: {exc}",
            )


def repair_empty_conversations(config: Config, dry_run: bool = False) -> RepairResult:
    """Delete conversations that have no messages."""
    with connection_context(None) as conn:
        return _run_repair(
            name="empty_conversations",
            count_sql="SELECT COUNT(*) FROM conversations c WHERE NOT EXISTS (SELECT 1 FROM messages m WHERE m.conversation_id = c.conversation_id)",
            action_sql="DELETE FROM conversations WHERE NOT EXISTS (SELECT 1 FROM messages m WHERE m.conversation_id = conversations.conversation_id)",
            dry_run=dry_run,
            conn=conn,
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
                        repaired_count=0,
                        success=True,
                        detail="FTS index in sync",
                    )

                return RepairResult(
                    name="dangling_fts",
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
                repaired_count=total,
                success=True,
                detail=f"FTS sync: deleted {deleted} orphaned, added {inserted} missing entries",
            )
    except Exception as exc:
        return RepairResult(
            name="dangling_fts",
            repaired_count=0,
            success=False,
            detail=f"Failed to repair FTS index: {exc}",
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
                repaired_count=total,
                success=True,
                detail=f"Cleaned {refs_deleted} orphaned refs, {conv_refs_deleted} conv refs, {atts_deleted} attachments",
            )
    except Exception as exc:
        return RepairResult(
            name="orphaned_attachments",
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
                    repaired_count=pages_estimate,
                    success=True,
                    detail=f"Would: WAL checkpoint (~{pages_estimate} pages, {wal_size:,} bytes)",
                )
            return RepairResult(
                name="wal_checkpoint",
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
                    repaired_count=0,
                    success=False,
                    detail=f"WAL checkpoint had busy pages: {busy} busy, {log} log, {checkpointed} checkpointed",
                )
            return RepairResult(
                name="wal_checkpoint",
                repaired_count=checkpointed if checkpointed > 0 else 0,
                success=True,
                detail=f"WAL checkpoint complete: {checkpointed} pages checkpointed",
            )
    except Exception as exc:
        return RepairResult(
            name="wal_checkpoint",
            repaired_count=0,
            success=False,
            detail=f"WAL checkpoint failed: {exc}",
        )


def run_all_repairs(config: Config, dry_run: bool = False) -> list[RepairResult]:
    """Run all repair operations and return results.

    Args:
        config: Configuration object
        dry_run: If True, show what would be repaired without making changes
    """
    return [
        repair_orphaned_messages(config, dry_run=dry_run),
        repair_empty_conversations(config, dry_run=dry_run),
        repair_dangling_fts(config, dry_run=dry_run),
        repair_orphaned_attachments(config, dry_run=dry_run),
        repair_wal_checkpoint(config, dry_run=dry_run),
    ]


__all__ = [
    "RepairResult",
    "repair_orphaned_messages",
    "repair_empty_conversations",
    "repair_dangling_fts",
    "repair_orphaned_attachments",
    "repair_wal_checkpoint",
    "run_all_repairs",
]
