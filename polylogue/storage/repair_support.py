"""Shared maintenance repair result and helper definitions."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from typing import Any

from polylogue.logging import get_logger
from polylogue.maintenance_models import MaintenanceCategory

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


def run_sql_repair(
    name: str,
    *,
    category: MaintenanceCategory,
    destructive: bool,
    count_sql: str,
    action_sql: str | None,
    dry_run: bool,
    conn: sqlite3.Connection,
) -> RepairResult:
    """Generic repair framework for data cleanup operations."""
    try:
        count = conn.execute(count_sql).fetchone()[0]

        if dry_run:
            return RepairResult(
                name=name,
                category=category,
                destructive=destructive,
                repaired_count=count,
                success=True,
                detail=f"Would: {count} rows affected" if count else "Would: No issues found",
            )

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


__all__ = [
    "CLEANUP_TARGETS",
    "MAINTENANCE_TARGET_NAMES",
    "RepairResult",
    "SAFE_REPAIR_TARGETS",
    "run_sql_repair",
]
