"""Destructive archive-cleanup maintenance operations."""

from __future__ import annotations

from polylogue.config import Config
from polylogue.maintenance_models import MaintenanceCategory

from .backends.connection import connection_context
from .repair_support import RepairResult, run_sql_repair


def repair_orphaned_messages(config: Config, dry_run: bool = False) -> RepairResult:
    """Delete messages that reference non-existent conversations."""
    with connection_context(None) as conn:
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
    return RepairResult(
        name="orphaned_messages",
        category=MaintenanceCategory.ARCHIVE_CLEANUP,
        destructive=True,
        repaired_count=count,
        success=True,
        detail=f"Would: Delete {count} orphaned messages" if count else "Would: No orphaned messages found",
    )


def repair_empty_conversations(config: Config, dry_run: bool = False) -> RepairResult:
    with connection_context(None) as conn:
        return run_sql_repair(
            name="empty_conversations",
            category=MaintenanceCategory.ARCHIVE_CLEANUP,
            destructive=True,
            count_sql="SELECT COUNT(*) FROM conversations c WHERE NOT EXISTS (SELECT 1 FROM messages m WHERE m.conversation_id = c.conversation_id)",
            action_sql="DELETE FROM conversations WHERE NOT EXISTS (SELECT 1 FROM messages m WHERE m.conversation_id = conversations.conversation_id)",
            dry_run=dry_run,
            conn=conn,
        )


def preview_empty_conversations(*, count: int) -> RepairResult:
    return RepairResult(
        name="empty_conversations",
        category=MaintenanceCategory.ARCHIVE_CLEANUP,
        destructive=True,
        repaired_count=count,
        success=True,
        detail=f"Would: {count} rows affected" if count else "Would: No issues found",
    )


def repair_orphaned_content_blocks(config: Config, dry_run: bool = False) -> RepairResult:
    with connection_context(None) as conn:
        return run_sql_repair(
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
    return RepairResult(
        name="orphaned_content_blocks",
        category=MaintenanceCategory.ARCHIVE_CLEANUP,
        destructive=True,
        repaired_count=count,
        success=True,
        detail=f"Would: {count} rows affected" if count else "Would: No issues found",
    )


def repair_orphaned_attachments(config: Config, dry_run: bool = False) -> RepairResult:
    try:
        with connection_context(None) as conn:
            if dry_run:
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

            ref_result = conn.execute(
                """
                DELETE FROM attachment_refs
                WHERE message_id IS NOT NULL AND NOT EXISTS (SELECT 1 FROM messages m WHERE m.message_id = attachment_refs.message_id)
                """
            )
            refs_deleted = ref_result.rowcount

            conv_ref_result = conn.execute(
                """
                DELETE FROM attachment_refs
                WHERE NOT EXISTS (SELECT 1 FROM conversations c WHERE c.conversation_id = attachment_refs.conversation_id)
                """
            )
            conv_refs_deleted = conv_ref_result.rowcount

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


__all__ = [
    "preview_empty_conversations",
    "preview_orphaned_content_blocks",
    "preview_orphaned_messages",
    "repair_empty_conversations",
    "repair_orphaned_attachments",
    "repair_orphaned_content_blocks",
    "repair_orphaned_messages",
]
