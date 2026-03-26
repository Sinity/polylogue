"""Canonical live archive-debt counting and status helpers."""

from __future__ import annotations

import sqlite3

from polylogue.maintenance_models import ArchiveDebtStatus, DerivedModelStatus, MaintenanceCategory
from polylogue.storage.derived_status import collect_derived_model_statuses_sync


def count_orphaned_messages_sync(conn: sqlite3.Connection) -> int:
    orphan_cids = conn.execute(
        """
        SELECT DISTINCT m.conversation_id FROM messages m
        WHERE NOT EXISTS (SELECT 1 FROM conversations c WHERE c.conversation_id = m.conversation_id)
        """
    ).fetchall()
    if not orphan_cids:
        return 0
    placeholders = ",".join("?" for _ in orphan_cids)
    return int(
        conn.execute(
            f"SELECT COUNT(*) FROM messages WHERE conversation_id IN ({placeholders})",
            [row[0] for row in orphan_cids],
        ).fetchone()[0]
    )


def count_orphaned_content_blocks_sync(conn: sqlite3.Connection) -> int:
    return int(
        conn.execute(
            """
            SELECT COUNT(*)
            FROM content_blocks cb
            WHERE NOT EXISTS (SELECT 1 FROM conversations c WHERE c.conversation_id = cb.conversation_id)
               OR NOT EXISTS (SELECT 1 FROM messages m WHERE m.message_id = cb.message_id)
            """
        ).fetchone()[0]
    )


def count_empty_conversations_sync(conn: sqlite3.Connection) -> int:
    return int(
        conn.execute(
            """
            SELECT COUNT(*) FROM conversations c
            WHERE NOT EXISTS (SELECT 1 FROM messages m WHERE m.conversation_id = c.conversation_id)
            """
        ).fetchone()[0]
    )


def count_orphaned_attachments_sync(conn: sqlite3.Connection) -> int:
    orphaned_refs = int(
        conn.execute(
            """
            SELECT COUNT(*) FROM attachment_refs ar
            WHERE (ar.message_id IS NOT NULL AND NOT EXISTS (SELECT 1 FROM messages m WHERE m.message_id = ar.message_id))
               OR NOT EXISTS (SELECT 1 FROM conversations c WHERE c.conversation_id = ar.conversation_id)
            """
        ).fetchone()[0]
    )
    unreferenced_attachments = int(
        conn.execute(
            """
            SELECT COUNT(*) FROM attachments a
            WHERE NOT EXISTS (SELECT 1 FROM attachment_refs ar WHERE ar.attachment_id = a.attachment_id)
            """
        ).fetchone()[0]
    )
    return orphaned_refs + unreferenced_attachments


def session_product_repair_count(
    derived_statuses: dict[str, DerivedModelStatus],
) -> int:
    session_profile_rows = derived_statuses.get("session_profile_rows")
    session_profile_merged_fts = derived_statuses.get("session_profile_merged_fts")
    session_profile_evidence_fts = derived_statuses.get("session_profile_evidence_fts")
    session_profile_inference_fts = derived_statuses.get("session_profile_inference_fts")
    session_profile_enrichment_fts = derived_statuses.get("session_profile_enrichment_fts")
    session_work_event_inference = derived_statuses.get("session_work_event_inference")
    session_work_event_inference_fts = derived_statuses.get("session_work_event_inference_fts")
    session_phase_inference = derived_statuses.get("session_phase_inference")
    work_threads = derived_statuses.get("work_threads")
    work_threads_fts = derived_statuses.get("work_threads_fts")
    session_tag_rollups = derived_statuses.get("session_tag_rollups")
    day_session_summaries = derived_statuses.get("day_session_summaries")
    week_session_summaries = derived_statuses.get("week_session_summaries")
    if not all(
        status is not None
        for status in (
            session_profile_rows,
            session_profile_merged_fts,
            session_profile_evidence_fts,
            session_profile_inference_fts,
            session_profile_enrichment_fts,
            session_work_event_inference,
            session_work_event_inference_fts,
            session_phase_inference,
            work_threads,
            work_threads_fts,
            session_tag_rollups,
            day_session_summaries,
            week_session_summaries,
        )
    ):
        return 0
    return (
        max(0, int(session_profile_rows.pending_documents or 0))
        + max(0, int(session_profile_rows.pending_rows or 0))
        + max(0, int(session_profile_rows.stale_rows or 0))
        + max(0, int(session_profile_rows.orphan_rows or 0))
        + max(0, int(session_profile_merged_fts.pending_rows or 0))
        + max(0, int(session_profile_merged_fts.stale_rows or 0))
        + max(0, int(session_profile_evidence_fts.pending_rows or 0))
        + max(0, int(session_profile_evidence_fts.stale_rows or 0))
        + max(0, int(session_profile_inference_fts.pending_rows or 0))
        + max(0, int(session_profile_inference_fts.stale_rows or 0))
        + max(0, int(session_profile_enrichment_fts.pending_rows or 0))
        + max(0, int(session_profile_enrichment_fts.stale_rows or 0))
        + max(0, int(session_work_event_inference.pending_rows or 0))
        + max(0, int(session_work_event_inference.stale_rows or 0))
        + max(0, int(session_work_event_inference.orphan_rows or 0))
        + max(0, int(session_work_event_inference_fts.pending_rows or 0))
        + max(0, int(session_work_event_inference_fts.stale_rows or 0))
        + max(0, int(session_phase_inference.pending_rows or 0))
        + max(0, int(session_phase_inference.stale_rows or 0))
        + max(0, int(session_phase_inference.orphan_rows or 0))
        + max(0, int(work_threads.pending_documents or 0))
        + max(0, int(work_threads.stale_rows or 0))
        + max(0, int(work_threads.orphan_rows or 0))
        + max(0, int(work_threads_fts.pending_rows or 0))
        + max(0, int(work_threads_fts.stale_rows or 0))
        + max(0, int(session_tag_rollups.pending_rows or 0))
        + max(0, int(session_tag_rollups.stale_rows or 0))
        + max(0, int(day_session_summaries.pending_rows or 0))
        + max(0, int(day_session_summaries.stale_rows or 0))
        + max(0, int(week_session_summaries.pending_rows or 0))
        + max(0, int(week_session_summaries.stale_rows or 0))
    )


def action_event_repair_count(
    derived_statuses: dict[str, DerivedModelStatus],
) -> int:
    action_events = derived_statuses.get("action_events")
    action_events_fts = derived_statuses.get("action_events_fts")
    if action_events is None or action_events_fts is None:
        return 0
    return (
        max(0, int(action_events.pending_documents or 0))
        + max(0, int(action_events.stale_rows or 0))
        + max(0, int(action_events_fts.pending_rows or 0))
    )


def dangling_fts_repair_count(
    derived_statuses: dict[str, DerivedModelStatus],
) -> int:
    messages_fts = derived_statuses.get("messages_fts")
    return max(0, int(messages_fts.pending_rows or 0)) if messages_fts is not None else 0


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
