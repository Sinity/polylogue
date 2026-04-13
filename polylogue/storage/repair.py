"""Consolidated archive repair: orphan detection, FTS repair, session products, WAL."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from polylogue.logging import get_logger
from polylogue.maintenance_models import DerivedModelStatus, MaintenanceCategory
from polylogue.storage.action_event_artifacts import ActionEventArtifactState

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# RepairResult
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Target constants
# ---------------------------------------------------------------------------

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

_CAT_DERIVED = MaintenanceCategory.DERIVED_REPAIR
_CAT_CLEANUP = MaintenanceCategory.ARCHIVE_CLEANUP
_CAT_DB = MaintenanceCategory.DATABASE_MAINTENANCE


# ---------------------------------------------------------------------------
# Orphan count queries (formerly archive_debt_counts)
# ---------------------------------------------------------------------------


def count_orphaned_messages_sync(conn: sqlite3.Connection) -> int:
    return int(
        conn.execute(
            """
            SELECT COUNT(*)
            FROM messages m
            LEFT JOIN conversations c ON c.conversation_id = m.conversation_id
            WHERE c.conversation_id IS NULL
            """
        ).fetchone()[0]
    )


def has_orphaned_messages_sync(conn: sqlite3.Connection) -> bool:
    return bool(
        conn.execute(
            """
            SELECT 1
            FROM messages m
            LEFT JOIN conversations c ON c.conversation_id = m.conversation_id
            WHERE c.conversation_id IS NULL
            LIMIT 1
            """
        ).fetchone()
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
            SELECT COUNT(*)
            FROM conversations c
            LEFT JOIN messages m ON m.conversation_id = c.conversation_id
            WHERE m.conversation_id IS NULL
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


# ---------------------------------------------------------------------------
# Derived repair count helpers (formerly archive_debt_repairs)
# ---------------------------------------------------------------------------


def session_product_repair_count(derived_statuses: dict[str, DerivedModelStatus]) -> int:
    keys = [
        "session_profile_rows",
        "session_profile_merged_fts",
        "session_profile_evidence_fts",
        "session_profile_inference_fts",
        "session_profile_enrichment_fts",
        "session_work_event_inference",
        "session_work_event_inference_fts",
        "session_phase_inference",
        "work_threads",
        "work_threads_fts",
        "session_tag_rollups",
        "day_session_summaries",
        "week_session_summaries",
    ]
    statuses = [derived_statuses.get(k) for k in keys]
    if not all(s is not None for s in statuses):
        return 0
    total = 0
    for s in statuses:
        total += max(0, int(s.pending_documents or 0))
        total += max(0, int(s.pending_rows or 0))
        total += max(0, int(s.stale_rows or 0))
        total += max(0, int(s.orphan_rows or 0))
    return total


def action_event_repair_count(derived_statuses: dict[str, DerivedModelStatus]) -> int:
    return _action_event_state_from_statuses(derived_statuses).repair_item_count


def _action_event_repair_components(
    derived_statuses: dict[str, DerivedModelStatus],
) -> tuple[int, int, int, int]:
    state = _action_event_state_from_statuses(derived_statuses)
    return (
        state.missing_conversations,
        state.stale_rows,
        state.pending_fts_rows,
        state.excess_fts_rows,
    )


def _action_event_state_from_statuses(
    derived_statuses: dict[str, DerivedModelStatus],
) -> ActionEventArtifactState:
    action_events = derived_statuses.get("action_events")
    action_events_fts = derived_statuses.get("action_events_fts")
    if action_events is None or action_events_fts is None:
        return ActionEventArtifactState(
            source_conversations=0,
            materialized_conversations=0,
            materialized_rows=0,
            fts_rows=0,
        )
    return ActionEventArtifactState(
        source_conversations=int(action_events.source_documents or 0),
        materialized_conversations=int(action_events.materialized_documents or 0),
        materialized_rows=int(action_events.materialized_rows or 0),
        fts_rows=int(action_events_fts.materialized_rows or 0),
        stale_rows=int(action_events.stale_rows or 0),
        orphan_rows=int(action_events.orphan_rows or 0),
        matches_version=bool(action_events.matches_version if action_events.matches_version is not None else True),
    )


def _action_event_repair_detail(derived_statuses: dict[str, DerivedModelStatus]) -> str:
    return _action_event_state_from_statuses(derived_statuses).repair_detail()


def dangling_fts_repair_count(derived_statuses: dict[str, DerivedModelStatus]) -> int:
    messages_fts = derived_statuses.get("messages_fts")
    return max(0, int(messages_fts.pending_rows or 0)) if messages_fts is not None else 0


# ---------------------------------------------------------------------------
# Archive debt collection (formerly archive_debt.py)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ArchiveDebtStatus:
    """Simple debt/orphan status for a single maintenance target."""

    name: str
    category: MaintenanceCategory
    destructive: bool
    issue_count: int
    detail: str
    maintenance_target: str

    @property
    def healthy(self) -> bool:
        return self.issue_count == 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "category": self.category.value,
            "destructive": self.destructive,
            "issue_count": self.issue_count,
            "detail": self.detail,
            "maintenance_target": self.maintenance_target,
            "healthy": self.healthy,
        }


def collect_archive_debt_statuses_sync(
    conn: sqlite3.Connection,
    *,
    derived_statuses: dict[str, DerivedModelStatus] | None = None,
    include_expensive: bool = True,
    probe_only: bool = False,
) -> dict[str, ArchiveDebtStatus]:
    from polylogue.storage.derived_status import collect_derived_model_statuses_sync

    statuses = derived_statuses or collect_derived_model_statuses_sync(conn, verify_full=include_expensive)

    orphaned_messages = (
        1
        if has_orphaned_messages_sync(conn)
        else 0
        if probe_only and not include_expensive
        else count_orphaned_messages_sync(conn)
    )
    empty_conversations = count_empty_conversations_sync(conn)
    orphaned_attachments = count_orphaned_attachments_sync(conn)
    session_products = session_product_repair_count(statuses)
    action_events = action_event_repair_count(statuses)
    dangling_fts = dangling_fts_repair_count(statuses)

    debt_statuses = {
        "orphaned_messages": ArchiveDebtStatus(
            name="orphaned_messages",
            category=_CAT_CLEANUP,
            destructive=True,
            issue_count=orphaned_messages,
            detail=(
                "No orphaned messages"
                if orphaned_messages == 0
                else (
                    "Orphaned messages present; use --deep for exact count"
                    if probe_only and not include_expensive
                    else f"{orphaned_messages:,} orphaned messages"
                )
            ),
            maintenance_target="orphaned_messages",
        ),
        "empty_conversations": ArchiveDebtStatus(
            name="empty_conversations",
            category=_CAT_CLEANUP,
            destructive=True,
            issue_count=empty_conversations,
            detail="No empty conversations"
            if empty_conversations == 0
            else f"{empty_conversations:,} empty conversations",
            maintenance_target="empty_conversations",
        ),
        "orphaned_attachments": ArchiveDebtStatus(
            name="orphaned_attachments",
            category=_CAT_CLEANUP,
            destructive=True,
            issue_count=orphaned_attachments,
            detail="No orphaned attachments"
            if orphaned_attachments == 0
            else f"{orphaned_attachments:,} orphaned attachment rows",
            maintenance_target="orphaned_attachments",
        ),
        "session_products": ArchiveDebtStatus(
            name="session_products",
            category=_CAT_DERIVED,
            destructive=False,
            issue_count=session_products,
            detail="Session-product read models ready"
            if session_products == 0
            else f"{session_products:,} pending/stale/orphaned session-product rows",
            maintenance_target="session_products",
        ),
        "action_event_read_model": ArchiveDebtStatus(
            name="action_event_read_model",
            category=_CAT_DERIVED,
            destructive=False,
            issue_count=action_events,
            detail=_action_event_repair_detail(statuses),
            maintenance_target="action_event_read_model",
        ),
        "dangling_fts": ArchiveDebtStatus(
            name="dangling_fts",
            category=_CAT_DERIVED,
            destructive=False,
            issue_count=dangling_fts,
            detail="FTS synchronized" if dangling_fts == 0 else f"{dangling_fts:,} dangling FTS rows",
            maintenance_target="dangling_fts",
        ),
    }
    if include_expensive:
        orphaned_content_blocks = count_orphaned_content_blocks_sync(conn)
        debt_statuses["orphaned_content_blocks"] = ArchiveDebtStatus(
            name="orphaned_content_blocks",
            category=_CAT_CLEANUP,
            destructive=True,
            issue_count=orphaned_content_blocks,
            detail="No orphaned content blocks"
            if orphaned_content_blocks == 0
            else f"{orphaned_content_blocks:,} orphaned content blocks",
            maintenance_target="orphaned_content_blocks",
        )
    return debt_statuses


def preview_counts_from_archive_debt(
    statuses: dict[str, ArchiveDebtStatus],
) -> dict[str, int]:
    return {
        status.maintenance_target: status.issue_count
        for status in statuses.values()
        if status.issue_count > 0
        or status.maintenance_target in {"session_products", "action_event_read_model", "dangling_fts"}
    }


# ---------------------------------------------------------------------------
# Generic SQL repair helper
# ---------------------------------------------------------------------------


def _run_sql_repair(
    name: str,
    *,
    category: MaintenanceCategory,
    destructive: bool,
    count_sql: str,
    action_sql: str | None,
    dry_run: bool,
    conn: sqlite3.Connection,
) -> RepairResult:
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


# ---------------------------------------------------------------------------
# Cleanup repairs (orphans, empty conversations, attachments)
# ---------------------------------------------------------------------------


def repair_orphaned_messages(config: Any, dry_run: bool = False) -> RepairResult:
    from polylogue.storage.backends.connection import connection_context

    with connection_context(None) as conn:
        count = count_orphaned_messages_sync(conn)
        if count == 0:
            return RepairResult(
                name="orphaned_messages",
                category=_CAT_CLEANUP,
                destructive=True,
                repaired_count=0,
                success=True,
                detail="No orphaned messages found",
            )
        try:
            if dry_run:
                return RepairResult(
                    name="orphaned_messages",
                    category=_CAT_CLEANUP,
                    destructive=True,
                    repaired_count=count,
                    success=True,
                    detail=f"Would: Delete {count} orphaned messages",
                )
            orphan_cids = conn.execute(
                "SELECT DISTINCT conversation_id FROM messages WHERE NOT EXISTS (SELECT 1 FROM conversations c WHERE c.conversation_id = messages.conversation_id)"
            ).fetchall()
            placeholders = ",".join("?" for _ in orphan_cids)
            result = conn.execute(
                f"DELETE FROM messages WHERE conversation_id IN ({placeholders})", [row[0] for row in orphan_cids]
            )
            conn.commit()
            return RepairResult(
                name="orphaned_messages",
                category=_CAT_CLEANUP,
                destructive=True,
                repaired_count=result.rowcount,
                success=True,
                detail=f"Deleted {result.rowcount} orphaned messages",
            )
        except Exception as exc:
            return RepairResult(
                name="orphaned_messages",
                category=_CAT_CLEANUP,
                destructive=True,
                repaired_count=0,
                success=False,
                detail=f"Failed to delete orphaned messages: {exc}",
            )


def preview_orphaned_messages(*, count: int) -> RepairResult:
    return RepairResult(
        name="orphaned_messages",
        category=_CAT_CLEANUP,
        destructive=True,
        repaired_count=count,
        success=True,
        detail=f"Would: Delete {count} orphaned messages" if count else "Would: No orphaned messages found",
    )


def repair_empty_conversations(config: Any, dry_run: bool = False) -> RepairResult:
    from polylogue.storage.backends.connection import connection_context

    with connection_context(None) as conn:
        return _run_sql_repair(
            name="empty_conversations",
            category=_CAT_CLEANUP,
            destructive=True,
            count_sql="SELECT COUNT(*) FROM conversations c WHERE NOT EXISTS (SELECT 1 FROM messages m WHERE m.conversation_id = c.conversation_id)",
            action_sql="DELETE FROM conversations WHERE NOT EXISTS (SELECT 1 FROM messages m WHERE m.conversation_id = conversations.conversation_id)",
            dry_run=dry_run,
            conn=conn,
        )


def preview_empty_conversations(*, count: int) -> RepairResult:
    return RepairResult(
        name="empty_conversations",
        category=_CAT_CLEANUP,
        destructive=True,
        repaired_count=count,
        success=True,
        detail=f"Would: {count} rows affected" if count else "Would: No issues found",
    )


def repair_orphaned_content_blocks(config: Any, dry_run: bool = False) -> RepairResult:
    from polylogue.storage.backends.connection import connection_context

    with connection_context(None) as conn:
        if dry_run:
            count = count_orphaned_content_blocks_sync(conn)
            return preview_orphaned_content_blocks(count=count)
        return _run_sql_repair(
            name="orphaned_content_blocks",
            category=_CAT_CLEANUP,
            destructive=True,
            count_sql="""
                SELECT COUNT(*) FROM content_blocks cb
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
        category=_CAT_CLEANUP,
        destructive=True,
        repaired_count=count,
        success=True,
        detail=f"Would: {count} rows affected" if count else "Would: No issues found",
    )


def repair_orphaned_attachments(config: Any, dry_run: bool = False) -> RepairResult:
    from polylogue.storage.backends.connection import connection_context

    try:
        with connection_context(None) as conn:
            if dry_run:
                return preview_orphaned_attachments(count=count_orphaned_attachments_sync(conn))

            ref_result = conn.execute(
                "DELETE FROM attachment_refs WHERE message_id IS NOT NULL AND NOT EXISTS (SELECT 1 FROM messages m WHERE m.message_id = attachment_refs.message_id)"
            )
            refs_deleted = ref_result.rowcount

            conv_ref_result = conn.execute(
                "DELETE FROM attachment_refs WHERE NOT EXISTS (SELECT 1 FROM conversations c WHERE c.conversation_id = attachment_refs.conversation_id)"
            )
            conv_refs_deleted = conv_ref_result.rowcount

            att_result = conn.execute(
                "DELETE FROM attachments WHERE NOT EXISTS (SELECT 1 FROM attachment_refs ar WHERE ar.attachment_id = attachments.attachment_id)"
            )
            atts_deleted = att_result.rowcount
            conn.commit()

            total = refs_deleted + conv_refs_deleted + atts_deleted
            return RepairResult(
                name="orphaned_attachments",
                category=_CAT_CLEANUP,
                destructive=True,
                repaired_count=total,
                success=True,
                detail=f"Cleaned {refs_deleted} orphaned refs, {conv_refs_deleted} conv refs, {atts_deleted} attachments",
            )
    except Exception as exc:
        return RepairResult(
            name="orphaned_attachments",
            category=_CAT_CLEANUP,
            destructive=True,
            repaired_count=0,
            success=False,
            detail=f"Failed to clean orphaned attachments: {exc}",
        )


def preview_orphaned_attachments(*, count: int) -> RepairResult:
    return RepairResult(
        name="orphaned_attachments",
        category=_CAT_CLEANUP,
        destructive=True,
        repaired_count=count,
        success=True,
        detail=f"Would: Clean {count} orphaned attachment rows" if count else "Would: No orphaned attachments found",
    )


# ---------------------------------------------------------------------------
# Derived repairs (session products, action events, FTS, WAL)
# ---------------------------------------------------------------------------


def repair_session_products(
    config: Any,
    dry_run: bool = False,
    *,
    progress_callback=None,
    progress_total: int | None = None,
) -> RepairResult:
    from polylogue.storage.backends.connection import connection_context
    from polylogue.storage.session_product_rebuild import rebuild_session_products_sync
    from polylogue.storage.session_product_status import session_product_status_sync

    try:
        with connection_context(None) as conn:
            status = session_product_status_sync(conn)
            profile_merged_fts_pending = max(
                0, int(status["profile_row_count"]) - int(status["profile_merged_fts_count"])
            )
            profile_merged_fts_duplicates = max(0, int(status.get("profile_merged_fts_duplicate_count", 0)))
            profile_evidence_fts_pending = max(
                0, int(status["profile_row_count"]) - int(status["profile_evidence_fts_count"])
            )
            profile_evidence_fts_duplicates = max(0, int(status.get("profile_evidence_fts_duplicate_count", 0)))
            profile_inference_fts_pending = max(
                0, int(status["profile_row_count"]) - int(status["profile_inference_fts_count"])
            )
            profile_inference_fts_duplicates = max(0, int(status.get("profile_inference_fts_duplicate_count", 0)))
            profile_enrichment_fts_pending = max(
                0, int(status["profile_row_count"]) - int(status["profile_enrichment_fts_count"])
            )
            profile_enrichment_fts_duplicates = max(0, int(status.get("profile_enrichment_fts_duplicate_count", 0)))
            work_event_fts_pending = max(
                0, int(status["work_event_inference_count"]) - int(status["work_event_inference_fts_count"])
            )
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
                    category=_CAT_DERIVED,
                    destructive=False,
                    repaired_count=pending,
                    success=True,
                    detail="Would: session products already ready"
                    if pending == 0
                    else f"Would: rebuild session products ({pending:,} pending items)",
                )

            rebuilt = rebuild_session_products_sync(
                conn,
                progress_callback=progress_callback,
                progress_total=progress_total,
            )
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
                category=_CAT_DERIVED,
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
                detail="Session products ready" if success else "Session products still incomplete",
            )
    except Exception as exc:
        return RepairResult(
            name="session_products",
            category=_CAT_DERIVED,
            destructive=False,
            repaired_count=0,
            success=False,
            detail=f"Failed to repair session products: {exc}",
        )


def preview_session_products(*, count: int) -> RepairResult:
    return RepairResult(
        name="session_products",
        category=_CAT_DERIVED,
        destructive=False,
        repaired_count=count,
        success=True,
        detail="Would: session products already ready"
        if count == 0
        else f"Would: rebuild session-product rows/fts for {count:,} pending items",
    )


def repair_action_event_read_model(config: Any, dry_run: bool = False) -> RepairResult:
    from polylogue.storage.action_event_rebuild_runtime import (
        action_event_repair_candidates_sync,
        rebuild_action_event_read_model_sync,
        valid_action_event_source_ids_sync,
    )
    from polylogue.storage.action_event_status import action_event_read_model_status_sync
    from polylogue.storage.backends.connection import connection_context
    from polylogue.storage.fts_lifecycle import repair_fts_index_sync

    try:
        with connection_context(None) as conn:
            status = action_event_read_model_status_sync(conn)
            state = ActionEventArtifactState.from_status_snapshot(status)
            candidate_ids = action_event_repair_candidates_sync(conn)
            pending = state.repair_item_count

            if dry_run:
                return RepairResult(
                    name="action_event_read_model",
                    category=_CAT_DERIVED,
                    destructive=False,
                    repaired_count=pending,
                    success=True,
                    detail="Would: action-event read model already ready"
                    if pending == 0
                    else f"Would: {state.repair_detail()[0].lower() + state.repair_detail()[1:]}",
                )

            repaired = 0
            if candidate_ids:
                repaired = rebuild_action_event_read_model_sync(conn, conversation_ids=candidate_ids)
            if not state.fts_ready:
                repair_targets = candidate_ids or valid_action_event_source_ids_sync(conn)
                if repair_targets:
                    repair_fts_index_sync(conn, repair_targets)
            conn.commit()
            refreshed = action_event_read_model_status_sync(conn)
            return RepairResult(
                name="action_event_read_model",
                category=_CAT_DERIVED,
                destructive=False,
                repaired_count=repaired + state.pending_fts_rows + state.excess_fts_rows,
                success=bool(refreshed["ready"]),
                detail="Action-event read model ready"
                if refreshed["ready"]
                else "Action-event read model still incomplete",
            )
    except Exception as exc:
        return RepairResult(
            name="action_event_read_model",
            category=_CAT_DERIVED,
            destructive=False,
            repaired_count=0,
            success=False,
            detail=f"Failed to repair action-event read model: {exc}",
        )


def preview_action_event_read_model(*, count: int) -> RepairResult:
    return RepairResult(
        name="action_event_read_model",
        category=_CAT_DERIVED,
        destructive=False,
        repaired_count=count,
        success=True,
        detail="Would: action-event read model already ready"
        if count == 0
        else f"Would: repair action-event rows/fts for {count:,} pending items",
    )


def repair_dangling_fts(config: Any, dry_run: bool = False) -> RepairResult:
    from polylogue.storage.backends.connection import connection_context
    from polylogue.storage.fts_lifecycle_sql import FTS_INDEXABLE_MESSAGE_COUNT_SQL

    try:
        with connection_context(None) as conn:
            fts_exists = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='messages_fts'"
            ).fetchone()
            if not fts_exists:
                return RepairResult(
                    name="dangling_fts",
                    category=_CAT_DERIVED,
                    destructive=False,
                    repaired_count=0,
                    success=True,
                    detail="FTS table does not exist, skipping",
                )

            if dry_run:
                msg_count = conn.execute(FTS_INDEXABLE_MESSAGE_COUNT_SQL).fetchone()[0]
                fts_count = conn.execute("SELECT COUNT(*) FROM messages_fts_docsize").fetchone()[0]
                diff = abs(msg_count - fts_count)
                if diff == 0:
                    return RepairResult(
                        name="dangling_fts",
                        category=_CAT_DERIVED,
                        destructive=False,
                        repaired_count=0,
                        success=True,
                        detail="FTS index in sync",
                    )
                return RepairResult(
                    name="dangling_fts",
                    category=_CAT_DERIVED,
                    destructive=False,
                    repaired_count=diff,
                    success=True,
                    detail=f"Would: FTS sync: {msg_count:,} messages vs {fts_count:,} indexed ({diff:,} difference)",
                )

            deleted = conn.execute(
                "DELETE FROM messages_fts WHERE rowid IN (SELECT f.rowid FROM messages_fts f WHERE NOT EXISTS (SELECT 1 FROM messages m WHERE m.rowid = f.rowid))"
            ).rowcount
            inserted = conn.execute(
                "INSERT INTO messages_fts (rowid, message_id, conversation_id, text) SELECT m.rowid, m.message_id, m.conversation_id, m.text FROM messages m WHERE m.text IS NOT NULL AND NOT EXISTS (SELECT 1 FROM messages_fts f WHERE f.rowid = m.rowid)"
            ).rowcount
            conn.commit()
            total = deleted + inserted
            return RepairResult(
                name="dangling_fts",
                category=_CAT_DERIVED,
                destructive=False,
                repaired_count=total,
                success=True,
                detail=f"FTS sync: deleted {deleted} orphaned, added {inserted} missing entries",
            )
    except Exception as exc:
        return RepairResult(
            name="dangling_fts",
            category=_CAT_DERIVED,
            destructive=False,
            repaired_count=0,
            success=False,
            detail=f"Failed to repair FTS index: {exc}",
        )


def preview_dangling_fts(*, count: int) -> RepairResult:
    return RepairResult(
        name="dangling_fts",
        category=_CAT_DERIVED,
        destructive=False,
        repaired_count=count,
        success=True,
        detail=f"Would: FTS sync pending {count:,} rows" if count else "FTS index in sync",
    )


def repair_wal_checkpoint(config: Any, dry_run: bool = False) -> RepairResult:
    from polylogue.storage.backends.connection import connection_context, default_db_path

    try:
        if dry_run:
            db_path = default_db_path()
            wal_path = Path(str(db_path) + "-wal")
            if wal_path.exists():
                wal_size = wal_path.stat().st_size
                pages_estimate = wal_size // 4096
                return RepairResult(
                    name="wal_checkpoint",
                    category=_CAT_DB,
                    destructive=False,
                    repaired_count=pages_estimate,
                    success=True,
                    detail=f"Would: WAL checkpoint (~{pages_estimate} pages, {wal_size:,} bytes)",
                )
            return RepairResult(
                name="wal_checkpoint",
                category=_CAT_DB,
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
                    category=_CAT_DB,
                    destructive=False,
                    repaired_count=0,
                    success=False,
                    detail=f"WAL checkpoint had busy pages: {busy} busy, {log} log, {checkpointed} checkpointed",
                )
            return RepairResult(
                name="wal_checkpoint",
                category=_CAT_DB,
                destructive=False,
                repaired_count=checkpointed if checkpointed > 0 else 0,
                success=True,
                detail=f"WAL checkpoint complete: {checkpointed} pages checkpointed",
            )
    except Exception as exc:
        return RepairResult(
            name="wal_checkpoint",
            category=_CAT_DB,
            destructive=False,
            repaired_count=0,
            success=False,
            detail=f"WAL checkpoint failed: {exc}",
        )


# ---------------------------------------------------------------------------
# Orchestration (run_safe_repairs, run_archive_cleanup, run_selected_maintenance)
# ---------------------------------------------------------------------------


def run_safe_repairs(
    config: Any,
    dry_run: bool = False,
    *,
    preview_counts: dict[str, int] | None = None,
    targets: tuple[str, ...] = (),
    session_product_progress_callback=None,
    session_product_progress_total: int | None = None,
) -> list[RepairResult]:
    preview_counts = preview_counts or {}
    selected = set(targets) if targets else set(SAFE_REPAIR_TARGETS)
    results: list[RepairResult] = []
    if "session_products" in selected:
        results.append(
            preview_session_products(count=preview_counts["session_products"])
            if dry_run and "session_products" in preview_counts
            else repair_session_products(
                config,
                dry_run=dry_run,
                progress_callback=session_product_progress_callback,
                progress_total=session_product_progress_total,
            )
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
    config: Any,
    dry_run: bool = False,
    *,
    preview_counts: dict[str, int] | None = None,
    targets: tuple[str, ...] = (),
) -> list[RepairResult]:
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
        results.append(
            preview_orphaned_attachments(count=preview_counts["orphaned_attachments"])
            if dry_run and "orphaned_attachments" in preview_counts
            else repair_orphaned_attachments(config, dry_run=dry_run)
        )
    return results


def run_selected_maintenance(
    config: Any,
    *,
    repair: bool,
    cleanup: bool,
    dry_run: bool = False,
    preview_counts: dict[str, int] | None = None,
    targets: tuple[str, ...] = (),
    session_product_progress_callback=None,
    session_product_progress_total: int | None = None,
) -> list[RepairResult]:
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
                session_product_progress_callback=session_product_progress_callback,
                session_product_progress_total=session_product_progress_total,
            )
        )
    if cleanup:
        results.extend(
            run_archive_cleanup(config, dry_run=dry_run, preview_counts=preview_counts, targets=cleanup_targets)
        )
    return results


__all__ = [
    "ArchiveDebtStatus",
    "CLEANUP_TARGETS",
    "MAINTENANCE_TARGET_NAMES",
    "RepairResult",
    "SAFE_REPAIR_TARGETS",
    "action_event_repair_count",
    "collect_archive_debt_statuses_sync",
    "count_empty_conversations_sync",
    "count_orphaned_attachments_sync",
    "count_orphaned_content_blocks_sync",
    "count_orphaned_messages_sync",
    "dangling_fts_repair_count",
    "preview_action_event_read_model",
    "preview_counts_from_archive_debt",
    "preview_dangling_fts",
    "preview_empty_conversations",
    "preview_orphaned_attachments",
    "preview_orphaned_content_blocks",
    "preview_orphaned_messages",
    "preview_session_products",
    "repair_action_event_read_model",
    "repair_dangling_fts",
    "repair_empty_conversations",
    "repair_orphaned_attachments",
    "repair_orphaned_content_blocks",
    "repair_orphaned_messages",
    "repair_session_products",
    "repair_wal_checkpoint",
    "run_archive_cleanup",
    "run_safe_repairs",
    "run_selected_maintenance",
    "session_product_repair_count",
]
