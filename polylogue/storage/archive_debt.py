"""Canonical archive-debt status assembly."""

from __future__ import annotations

import sqlite3

from polylogue.maintenance_models import ArchiveDebtStatus, DerivedModelStatus, MaintenanceCategory
from polylogue.storage.archive_debt_counts import (
    count_empty_conversations_sync,
    count_orphaned_attachments_sync,
    count_orphaned_content_blocks_sync,
    count_orphaned_messages_sync,
)
from polylogue.storage.archive_debt_repairs import (
    action_event_repair_count,
    dangling_fts_repair_count,
    session_product_repair_count,
)
from polylogue.storage.derived_status import collect_derived_model_statuses_sync


def collect_archive_debt_statuses_sync(
    conn: sqlite3.Connection,
    *,
    derived_statuses: dict[str, DerivedModelStatus] | None = None,
) -> dict[str, ArchiveDebtStatus]:
    statuses = derived_statuses or collect_derived_model_statuses_sync(conn)

    orphaned_messages = count_orphaned_messages_sync(conn)
    orphaned_content_blocks = count_orphaned_content_blocks_sync(conn)
    empty_conversations = count_empty_conversations_sync(conn)
    orphaned_attachments = count_orphaned_attachments_sync(conn)
    session_products = session_product_repair_count(statuses)
    action_events = action_event_repair_count(statuses)
    dangling_fts = dangling_fts_repair_count(statuses)

    return {
        "orphaned_messages": ArchiveDebtStatus(
            name="orphaned_messages",
            category=MaintenanceCategory.ARCHIVE_CLEANUP,
            destructive=True,
            issue_count=orphaned_messages,
            detail="No orphaned messages" if orphaned_messages == 0 else f"{orphaned_messages:,} orphaned messages",
            maintenance_target="orphaned_messages",
        ),
        "orphaned_content_blocks": ArchiveDebtStatus(
            name="orphaned_content_blocks",
            category=MaintenanceCategory.ARCHIVE_CLEANUP,
            destructive=True,
            issue_count=orphaned_content_blocks,
            detail=(
                "No orphaned content blocks"
                if orphaned_content_blocks == 0
                else f"{orphaned_content_blocks:,} orphaned content blocks"
            ),
            maintenance_target="orphaned_content_blocks",
        ),
        "empty_conversations": ArchiveDebtStatus(
            name="empty_conversations",
            category=MaintenanceCategory.ARCHIVE_CLEANUP,
            destructive=True,
            issue_count=empty_conversations,
            detail=(
                "No empty conversations"
                if empty_conversations == 0
                else f"{empty_conversations:,} empty conversations"
            ),
            maintenance_target="empty_conversations",
        ),
        "orphaned_attachments": ArchiveDebtStatus(
            name="orphaned_attachments",
            category=MaintenanceCategory.ARCHIVE_CLEANUP,
            destructive=True,
            issue_count=orphaned_attachments,
            detail=(
                "No orphaned attachments"
                if orphaned_attachments == 0
                else f"{orphaned_attachments:,} orphaned attachment rows"
            ),
            maintenance_target="orphaned_attachments",
        ),
        "session_products": ArchiveDebtStatus(
            name="session_products",
            category=MaintenanceCategory.DERIVED_REPAIR,
            destructive=False,
            issue_count=session_products,
            detail=(
                "Session-product read models ready"
                if session_products == 0
                else f"{session_products:,} pending/stale/orphaned session-product rows"
            ),
            maintenance_target="session_products",
        ),
        "action_event_read_model": ArchiveDebtStatus(
            name="action_event_read_model",
            category=MaintenanceCategory.DERIVED_REPAIR,
            destructive=False,
            issue_count=action_events,
            detail=(
                "Action-event read model ready"
                if action_events == 0
                else f"{action_events:,} pending/stale action-event rows"
            ),
            maintenance_target="action_event_read_model",
        ),
        "dangling_fts": ArchiveDebtStatus(
            name="dangling_fts",
            category=MaintenanceCategory.DERIVED_REPAIR,
            destructive=False,
            issue_count=dangling_fts,
            detail="FTS synchronized" if dangling_fts == 0 else f"{dangling_fts:,} dangling FTS rows",
            maintenance_target="dangling_fts",
        ),
    }


def preview_counts_from_archive_debt(
    statuses: dict[str, ArchiveDebtStatus],
) -> dict[str, int]:
    return {
        status.maintenance_target: status.issue_count
        for status in statuses.values()
        if status.issue_count > 0 or status.maintenance_target in {"session_products", "action_event_read_model", "dangling_fts"}
    }


__all__ = [
    "action_event_repair_count",
    "collect_archive_debt_statuses_sync",
    "count_empty_conversations_sync",
    "count_orphaned_attachments_sync",
    "count_orphaned_content_blocks_sync",
    "count_orphaned_messages_sync",
    "dangling_fts_repair_count",
    "preview_counts_from_archive_debt",
    "session_product_repair_count",
]
