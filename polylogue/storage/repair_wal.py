"""Database-maintenance WAL repair flows."""

from __future__ import annotations

from pathlib import Path

from polylogue.config import Config
from polylogue.maintenance_models import MaintenanceCategory

from .backends.connection import connection_context, default_db_path
from .repair_support import RepairResult


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


__all__ = ["repair_wal_checkpoint"]
