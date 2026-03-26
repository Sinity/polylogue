"""FTS drift repair flows."""

from __future__ import annotations

from polylogue.config import Config
from polylogue.maintenance_models import MaintenanceCategory

from .backends.connection import connection_context
from .repair_support import RepairResult


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


__all__ = ["preview_dangling_fts", "repair_dangling_fts"]
