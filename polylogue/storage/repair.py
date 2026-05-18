"""Consolidated archive repair: orphan detection, FTS repair, session insights, WAL."""

from __future__ import annotations

import sqlite3
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from polylogue.config import Config
from polylogue.core.json import JSONDocument, json_document
from polylogue.logging import get_logger
from polylogue.maintenance.models import DerivedModelStatus, MaintenanceCategory
from polylogue.maintenance.targets import (
    CLEANUP_TARGETS,
    SAFE_REPAIR_TARGETS,
    MaintenanceTargetSpec,
    build_maintenance_target_catalog,
)
from polylogue.protocols import ProgressCallback
from polylogue.storage.action_events.artifacts import ActionEventArtifactState
from polylogue.storage.insights.session.runtime import SessionInsightReadyFlag, SessionInsightStatusSnapshot
from polylogue.storage.message_type_backfill import (
    BackfillResult,
    count_messages_by_type_sync,
    count_unclassified_message_type_sync,
)

logger = get_logger(__name__)
_MAINTENANCE_TARGET_CATALOG = build_maintenance_target_catalog()

_SESSION_INSIGHT_READY_FLAGS: tuple[SessionInsightReadyFlag, ...] = (
    "profile_rows_ready",
    "work_event_inference_rows_ready",
    "work_event_inference_fts_ready",
    "phase_inference_rows_ready",
    "threads_ready",
    "threads_fts_ready",
    "tag_rollups_ready",
    "day_summaries_ready",
    "week_summaries_ready",
)


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

    def to_dict(self) -> JSONDocument:
        return json_document(
            {
                "name": self.name,
                "category": self.category.value,
                "destructive": self.destructive,
                "repaired_count": self.repaired_count,
                "success": self.success,
                "detail": self.detail,
            }
        )


@dataclass(slots=True, frozen=True)
class _SessionInsightRepairAssessment:
    row_debt: int
    fts_debt: int

    @property
    def pending(self) -> int:
        return self.row_debt + self.fts_debt


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


def session_insight_repair_count(derived_statuses: dict[str, DerivedModelStatus]) -> int:
    keys = [
        "session_profile_rows",
        "session_work_event_inference",
        "session_work_event_inference_fts",
        "session_phase_inference",
        "work_threads",
        "work_threads_fts",
        "session_tag_rollups",
        "day_session_summaries",
        "week_session_summaries",
    ]
    maybe_statuses = [derived_statuses.get(k) for k in keys]
    if not all(status is not None for status in maybe_statuses):
        return 0
    statuses = [status for status in maybe_statuses if status is not None]
    total = 0
    for s in statuses:
        total += max(0, int(s.pending_documents or 0))
        total += max(0, int(s.pending_rows or 0))
        total += max(0, int(s.stale_rows or 0))
        total += max(0, int(s.orphan_rows or 0))
    return total


def _positive_count(value: int) -> int:
    return max(0, value)


def _fts_repair_count(*, source_rows: int, indexed_rows: int, duplicates: int) -> int:
    return _positive_count(source_rows - indexed_rows) + _positive_count(duplicates)


def _session_insight_row_repair_count(status: SessionInsightStatusSnapshot) -> int:
    return (
        status.missing_profile_row_count
        + status.stale_profile_row_count
        + status.orphan_profile_row_count
        + status.stale_work_event_inference_count
        + status.orphan_work_event_inference_count
        + status.stale_phase_inference_count
        + status.orphan_phase_inference_count
        + status.stale_thread_count
        + status.orphan_thread_count
        + status.stale_tag_rollup_count
        + status.stale_day_summary_count
    )


def _session_insight_fts_repair_count(status: SessionInsightStatusSnapshot) -> int:
    return sum(
        (
            _fts_repair_count(
                source_rows=status.work_event_inference_count,
                indexed_rows=status.work_event_inference_fts_count,
                duplicates=status.work_event_inference_fts_duplicate_count,
            ),
            _fts_repair_count(
                source_rows=status.thread_count,
                indexed_rows=status.thread_fts_count,
                duplicates=status.thread_fts_duplicate_count,
            ),
        )
    )


def _assess_session_insight_repairs(status: SessionInsightStatusSnapshot) -> _SessionInsightRepairAssessment:
    return _SessionInsightRepairAssessment(
        row_debt=_session_insight_row_repair_count(status),
        fts_debt=_session_insight_fts_repair_count(status),
    )


def _session_insight_status_ready(status: SessionInsightStatusSnapshot) -> bool:
    return all(status.ready_flag(flag) for flag in _SESSION_INSIGHT_READY_FLAGS)


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

    def to_dict(self) -> JSONDocument:
        return json_document(
            {
                "name": self.name,
                "category": self.category.value,
                "destructive": self.destructive,
                "issue_count": self.issue_count,
                "detail": self.detail,
                "maintenance_target": self.maintenance_target,
                "healthy": self.healthy,
            }
        )


def _maintenance_target_spec(name: str) -> MaintenanceTargetSpec:
    spec = _MAINTENANCE_TARGET_CATALOG.resolve_name(name)
    if spec is None:
        raise KeyError(f"Unknown maintenance target: {name}")
    return spec


def _repair_result(
    target_name: str,
    *,
    repaired_count: int,
    success: bool,
    detail: str,
) -> RepairResult:
    spec = _maintenance_target_spec(target_name)
    return RepairResult(
        name=spec.name,
        category=spec.category,
        destructive=spec.destructive,
        repaired_count=repaired_count,
        success=success,
        detail=detail,
    )


def _archive_debt_status(
    target_name: str,
    *,
    issue_count: int,
    detail: str,
) -> ArchiveDebtStatus:
    spec = _maintenance_target_spec(target_name)
    return ArchiveDebtStatus(
        name=spec.name,
        category=spec.category,
        destructive=spec.destructive,
        issue_count=issue_count,
        detail=detail,
        maintenance_target=spec.name,
    )


def collect_archive_debt_statuses_sync(
    conn: sqlite3.Connection,
    *,
    derived_statuses: dict[str, DerivedModelStatus] | None = None,
    include_expensive: bool = True,
    probe_only: bool = False,
) -> dict[str, ArchiveDebtStatus]:
    from polylogue.storage.derived.derived_status import collect_derived_model_statuses_sync

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
    session_insights = session_insight_repair_count(statuses)
    action_events = action_event_repair_count(statuses)
    dangling_fts = dangling_fts_repair_count(statuses)
    _unclassified = count_unclassified_message_type_sync(conn)

    debt_statuses = {
        "orphaned_messages": _archive_debt_status(
            "orphaned_messages",
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
        ),
        "empty_conversations": _archive_debt_status(
            "empty_conversations",
            issue_count=empty_conversations,
            detail="No empty conversations"
            if empty_conversations == 0
            else f"{empty_conversations:,} empty conversations",
        ),
        "orphaned_attachments": _archive_debt_status(
            "orphaned_attachments",
            issue_count=orphaned_attachments,
            detail="No orphaned attachments"
            if orphaned_attachments == 0
            else f"{orphaned_attachments:,} orphaned attachment rows",
        ),
        "session_insights": _archive_debt_status(
            "session_insights",
            issue_count=session_insights,
            detail="Session insight read models ready"
            if session_insights == 0
            else f"{session_insights:,} pending/stale/orphaned session-insight rows",
        ),
        "action_event_read_model": _archive_debt_status(
            "action_event_read_model",
            issue_count=action_events,
            detail=_action_event_repair_detail(statuses),
        ),
        "dangling_fts": _archive_debt_status(
            "dangling_fts",
            issue_count=dangling_fts,
            detail="FTS synchronized" if dangling_fts == 0 else f"{dangling_fts:,} dangling FTS rows",
        ),
        "message_type_backfill": _archive_debt_status(
            "message_type_backfill",
            issue_count=_unclassified,
            detail="No messages need context/protocol classification"
            if _unclassified == 0
            else f"{_unclassified:,} messages would be classified as context or protocol",
        ),
    }
    if include_expensive:
        orphaned_content_blocks = count_orphaned_content_blocks_sync(conn)
        debt_statuses["orphaned_content_blocks"] = _archive_debt_status(
            "orphaned_content_blocks",
            issue_count=orphaned_content_blocks,
            detail="No orphaned content blocks"
            if orphaned_content_blocks == 0
            else f"{orphaned_content_blocks:,} orphaned content blocks",
        )
        orphaned_blobs = count_orphaned_blobs_sync(conn)
        debt_statuses["orphaned_blobs"] = _archive_debt_status(
            "orphaned_blobs",
            issue_count=orphaned_blobs,
            detail="No orphaned blobs" if orphaned_blobs == 0 else f"{orphaned_blobs:,} orphaned blob files on disk",
        )
    return debt_statuses


def preview_counts_from_archive_debt(
    statuses: dict[str, ArchiveDebtStatus],
) -> dict[str, int]:
    preview_targets = set(_MAINTENANCE_TARGET_CATALOG.preview_target_names())
    return {
        status.maintenance_target: status.issue_count
        for status in statuses.values()
        if status.issue_count > 0 or status.maintenance_target in preview_targets
    }


# ---------------------------------------------------------------------------
# Generic SQL repair helper
# ---------------------------------------------------------------------------


def _run_sql_repair(
    target_name: str,
    *,
    count_sql: str,
    action_sql: str | None,
    dry_run: bool,
    conn: sqlite3.Connection,
) -> RepairResult:
    try:
        count = conn.execute(count_sql).fetchone()[0]
        if dry_run:
            return _repair_result(
                target_name,
                repaired_count=count,
                success=True,
                detail=f"Would: {count} rows affected" if count else "Would: No issues found",
            )
        if action_sql:
            result = conn.execute(action_sql)
            conn.commit()
            return _repair_result(
                target_name,
                repaired_count=result.rowcount,
                success=True,
                detail=f"Repaired {result.rowcount} rows" if result.rowcount else "No repairs needed",
            )
        return _repair_result(
            target_name,
            repaired_count=0,
            success=True,
            detail="No action SQL provided",
        )
    except Exception as exc:
        return _repair_result(
            target_name,
            repaired_count=0,
            success=False,
            detail=f"Repair failed: {exc}",
        )


# ---------------------------------------------------------------------------
# Cleanup repairs (orphans, empty conversations, attachments)
# ---------------------------------------------------------------------------


def repair_orphaned_messages(config: Config, dry_run: bool = False) -> RepairResult:
    from polylogue.storage.sqlite.connection import connection_context

    with connection_context(None) as conn:
        count = count_orphaned_messages_sync(conn)
        if count == 0:
            return _repair_result(
                "orphaned_messages",
                repaired_count=0,
                success=True,
                detail="No orphaned messages found",
            )
        try:
            if dry_run:
                return _repair_result(
                    "orphaned_messages",
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
            return _repair_result(
                "orphaned_messages",
                repaired_count=result.rowcount,
                success=True,
                detail=f"Deleted {result.rowcount} orphaned messages",
            )
        except Exception as exc:
            return _repair_result(
                "orphaned_messages",
                repaired_count=0,
                success=False,
                detail=f"Failed to delete orphaned messages: {exc}",
            )


def preview_orphaned_messages(*, count: int) -> RepairResult:
    return _repair_result(
        "orphaned_messages",
        repaired_count=count,
        success=True,
        detail=f"Would: Delete {count} orphaned messages" if count else "Would: No orphaned messages found",
    )


def repair_empty_conversations(config: Config, dry_run: bool = False) -> RepairResult:
    from polylogue.storage.sqlite.connection import connection_context

    with connection_context(None) as conn:
        return _run_sql_repair(
            "empty_conversations",
            count_sql="SELECT COUNT(*) FROM conversations c WHERE NOT EXISTS (SELECT 1 FROM messages m WHERE m.conversation_id = c.conversation_id)",
            action_sql="DELETE FROM conversations WHERE NOT EXISTS (SELECT 1 FROM messages m WHERE m.conversation_id = conversations.conversation_id)",
            dry_run=dry_run,
            conn=conn,
        )


def preview_empty_conversations(*, count: int) -> RepairResult:
    return _repair_result(
        "empty_conversations",
        repaired_count=count,
        success=True,
        detail=f"Would: {count} rows affected" if count else "Would: No issues found",
    )


def repair_orphaned_content_blocks(config: Config, dry_run: bool = False) -> RepairResult:
    from polylogue.storage.sqlite.connection import connection_context

    with connection_context(None) as conn:
        if dry_run:
            count = count_orphaned_content_blocks_sync(conn)
            return preview_orphaned_content_blocks(count=count)
        return _run_sql_repair(
            "orphaned_content_blocks",
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
    return _repair_result(
        "orphaned_content_blocks",
        repaired_count=count,
        success=True,
        detail=f"Would: {count} rows affected" if count else "Would: No issues found",
    )


# ---------------------------------------------------------------------------
# Blob cleanup
# ---------------------------------------------------------------------------


def count_orphaned_blobs_sync(conn: sqlite3.Connection) -> int:
    """Count blob files on disk that have no DB reference.

    This is a filesystem-intensive operation — it walks the blob store
    directory and compares against ``raw_conversations.raw_id``.  Call
    only when ``include_expensive`` or ``--deep`` is requested.
    """
    from polylogue.storage.blob_store import get_blob_store

    db_raw_ids: set[str] = set()
    for row in conn.execute("SELECT raw_id FROM raw_conversations"):
        db_raw_ids.add(row[0])
    blob_store = get_blob_store()
    result = blob_store.detect_orphans(db_raw_ids)
    return result.orphan_count


def repair_orphaned_blobs(config: Config, dry_run: bool = False) -> RepairResult:
    """Delete blob files that are no longer referenced in the archive.

    Dry-run (default via --preview) reports what would be deleted without
    touching the filesystem.  The live path deletes blobs one at a time
    and reports the aggregate result.
    """
    from polylogue.storage.blob_store import get_blob_store
    from polylogue.storage.sqlite.connection import connection_context

    blob_store = get_blob_store()
    db_raw_ids: set[str] = set()
    with connection_context(None) as conn:
        for row in conn.execute("SELECT raw_id FROM raw_conversations"):
            db_raw_ids.add(row[0])

    detect_result = blob_store.detect_orphans(db_raw_ids)
    if detect_result.orphan_count == 0:
        return _repair_result(
            "orphaned_blobs",
            repaired_count=0,
            success=True,
            detail="No orphaned blobs found",
        )

    orphan_hashes = set(detect_result.orphan_samples)
    # For the real cleanup we need all orphans, not just the sample.
    # Re-walk to build the full set.
    if not dry_run:
        orphan_hashes = {h for h in blob_store.iter_all() if h not in db_raw_ids}

    cleanup_result = blob_store.cleanup_orphans(orphan_hashes, dry_run=dry_run)

    if dry_run:
        return _repair_result(
            "orphaned_blobs",
            repaired_count=cleanup_result.would_delete_count,
            success=True,
            detail=(
                f"Would: delete {cleanup_result.would_delete_count} orphaned blobs "
                f"({cleanup_result.would_delete_bytes:,} bytes)"
            ),
        )
    return _repair_result(
        "orphaned_blobs",
        repaired_count=cleanup_result.deleted_count,
        success=cleanup_result.errors == 0,
        detail=(
            f"Deleted {cleanup_result.deleted_count} orphaned blobs "
            f"({cleanup_result.deleted_bytes:,} bytes)"
            + (f" with {cleanup_result.errors} errors" if cleanup_result.errors else "")
        ),
    )


def preview_orphaned_blobs(*, count: int) -> RepairResult:
    return _repair_result(
        "orphaned_blobs",
        repaired_count=count,
        success=True,
        detail=f"Would: delete {count} orphaned blobs" if count else "Would: No orphaned blobs found",
    )


def repair_orphaned_attachments(config: Config, dry_run: bool = False) -> RepairResult:
    from polylogue.storage.sqlite.connection import connection_context

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
            return _repair_result(
                "orphaned_attachments",
                repaired_count=total,
                success=True,
                detail=f"Cleaned {refs_deleted} orphaned refs, {conv_refs_deleted} conv refs, {atts_deleted} attachments",
            )
    except Exception as exc:
        return _repair_result(
            "orphaned_attachments",
            repaired_count=0,
            success=False,
            detail=f"Failed to clean orphaned attachments: {exc}",
        )


def preview_orphaned_attachments(*, count: int) -> RepairResult:
    return _repair_result(
        "orphaned_attachments",
        repaired_count=count,
        success=True,
        detail=f"Would: Clean {count} orphaned attachment rows" if count else "Would: No orphaned attachments found",
    )


# ---------------------------------------------------------------------------
# Derived repairs (session insights, action events, FTS, WAL)
# ---------------------------------------------------------------------------


def repair_session_insights(
    config: Config,
    dry_run: bool = False,
    *,
    progress_callback: ProgressCallback | None = None,
    progress_total: int | None = None,
    conversation_ids: tuple[str, ...] | None = None,
) -> RepairResult:
    """Repair / rebuild session insights.

    When ``conversation_ids`` is given, the rebuild is narrowed to that
    set instead of touching the full archive — used by the maintenance
    planner to honor :class:`MaintenanceScopeFilter.conversation_ids`.
    """
    from polylogue.storage.insights.session.rebuild import rebuild_session_insights_sync
    from polylogue.storage.insights.session.status import session_insight_status_sync
    from polylogue.storage.sqlite.connection import connection_context

    try:
        with connection_context(None) as conn:
            status = session_insight_status_sync(conn)
            assessment = _assess_session_insight_repairs(status)

            if dry_run:
                pending = (
                    min(assessment.pending, len(conversation_ids))
                    if conversation_ids is not None
                    else assessment.pending
                )
                return _repair_result(
                    "session_insights",
                    repaired_count=pending,
                    success=True,
                    detail="Would: session insights already ready"
                    if pending == 0
                    else f"Would: rebuild session insights ({pending:,} pending items)",
                )

            if conversation_ids is None and assessment.pending == 0 and _session_insight_status_ready(status):
                return _repair_result(
                    "session_insights",
                    repaired_count=0,
                    success=True,
                    detail="Session insights already ready",
                )

            rebuilt = rebuild_session_insights_sync(
                conn,
                conversation_ids=conversation_ids,
                progress_callback=progress_callback,
                progress_total=progress_total,
            )
            conn.commit()
            refreshed = session_insight_status_sync(conn)
            # A narrowed rebuild only attests its own slice; do not
            # demand global readiness for a scope-filtered call.
            success = True if conversation_ids is not None else _session_insight_status_ready(refreshed)
            return _repair_result(
                "session_insights",
                repaired_count=rebuilt.total(),
                success=success,
                detail="Session insights ready" if success else "Session insights still incomplete",
            )
    except Exception as exc:
        return _repair_result(
            "session_insights",
            repaired_count=0,
            success=False,
            detail=f"Failed to repair session insights: {exc}",
        )


def preview_session_insights(*, count: int) -> RepairResult:
    return _repair_result(
        "session_insights",
        repaired_count=count,
        success=True,
        detail="Would: session insights already ready"
        if count == 0
        else f"Would: rebuild session-insight rows/fts for {count:,} pending items",
    )


def repair_action_event_read_model(config: Config, dry_run: bool = False) -> RepairResult:
    from polylogue.storage.action_events.rebuild_runtime import (
        action_event_repair_candidates_sync,
        rebuild_action_event_read_model_sync,
        valid_action_event_source_ids_sync,
    )
    from polylogue.storage.action_events.status import action_event_read_model_status_sync
    from polylogue.storage.fts.fts_lifecycle import rebuild_fts_index_sync, repair_fts_index_sync
    from polylogue.storage.sqlite.connection import connection_context

    try:
        with connection_context(None) as conn:
            status = action_event_read_model_status_sync(conn)
            state = ActionEventArtifactState.from_status_snapshot(status)
            candidate_ids = action_event_repair_candidates_sync(conn)
            pending = state.repair_item_count

            if dry_run:
                return _repair_result(
                    "action_event_read_model",
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
                if state.excess_fts_rows:
                    rebuild_fts_index_sync(conn)
                else:
                    repair_targets = candidate_ids or valid_action_event_source_ids_sync(conn)
                    if repair_targets:
                        repair_fts_index_sync(conn, repair_targets)
            conn.commit()
            refreshed = action_event_read_model_status_sync(conn)
            return _repair_result(
                "action_event_read_model",
                repaired_count=repaired + state.pending_fts_rows + state.excess_fts_rows,
                success=bool(refreshed["ready"]),
                detail="Action-event read model ready"
                if refreshed["ready"]
                else "Action-event read model still incomplete",
            )
    except Exception as exc:
        return _repair_result(
            "action_event_read_model",
            repaired_count=0,
            success=False,
            detail=f"Failed to repair action-event read model: {exc}",
        )


def preview_action_event_read_model(*, count: int) -> RepairResult:
    return _repair_result(
        "action_event_read_model",
        repaired_count=count,
        success=True,
        detail="Would: action-event read model already ready"
        if count == 0
        else f"Would: repair action-event rows/fts for {count:,} pending items",
    )


def repair_dangling_fts(config: Config, dry_run: bool = False) -> RepairResult:
    from polylogue.storage.fts.sql import FTS_INDEXABLE_MESSAGE_COUNT_SQL
    from polylogue.storage.sqlite.connection import connection_context

    try:
        with connection_context(None) as conn:
            fts_exists = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='messages_fts'"
            ).fetchone()
            if not fts_exists:
                return _repair_result(
                    "dangling_fts",
                    repaired_count=0,
                    success=True,
                    detail="FTS table does not exist, skipping",
                )

            if dry_run:
                msg_count = conn.execute(FTS_INDEXABLE_MESSAGE_COUNT_SQL).fetchone()[0]
                fts_count = conn.execute("SELECT COUNT(*) FROM messages_fts_docsize").fetchone()[0]
                diff = abs(msg_count - fts_count)
                if diff == 0:
                    return _repair_result(
                        "dangling_fts",
                        repaired_count=0,
                        success=True,
                        detail="FTS index in sync",
                    )
                return _repair_result(
                    "dangling_fts",
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
            from polylogue.storage.fts.fts_lifecycle import restore_fts_triggers_sync

            restore_fts_triggers_sync(conn)
            conn.commit()
            total = deleted + inserted
            return _repair_result(
                "dangling_fts",
                repaired_count=total,
                success=True,
                detail=f"FTS sync: deleted {deleted} orphaned, added {inserted} missing entries",
            )
    except Exception as exc:
        return _repair_result(
            "dangling_fts",
            repaired_count=0,
            success=False,
            detail=f"Failed to repair FTS index: {exc}",
        )


def preview_dangling_fts(*, count: int) -> RepairResult:
    return _repair_result(
        "dangling_fts",
        repaired_count=count,
        success=True,
        detail=f"Would: FTS sync pending {count:,} rows" if count else "FTS index in sync",
    )


def repair_wal_checkpoint(config: Config, dry_run: bool = False) -> RepairResult:
    from polylogue.paths import db_path
    from polylogue.storage.sqlite.connection import connection_context

    try:
        if dry_run:
            wal_path = Path(str(db_path()) + "-wal")
            if wal_path.exists():
                wal_size = wal_path.stat().st_size
                pages_estimate = wal_size // 4096
                return _repair_result(
                    "wal_checkpoint",
                    repaired_count=pages_estimate,
                    success=True,
                    detail=f"Would: WAL checkpoint (~{pages_estimate} pages, {wal_size:,} bytes)",
                )
            return _repair_result(
                "wal_checkpoint",
                repaired_count=0,
                success=True,
                detail="Would: No WAL file present, nothing to checkpoint",
            )

        with connection_context(None) as conn:
            row = conn.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
            busy, log, checkpointed = row[0], row[1], row[2]
            if busy:
                return _repair_result(
                    "wal_checkpoint",
                    repaired_count=0,
                    success=False,
                    detail=f"WAL checkpoint had busy pages: {busy} busy, {log} log, {checkpointed} checkpointed",
                )
            return _repair_result(
                "wal_checkpoint",
                repaired_count=checkpointed if checkpointed > 0 else 0,
                success=True,
                detail=f"WAL checkpoint complete: {checkpointed} pages checkpointed",
            )
    except Exception as exc:
        return _repair_result(
            "wal_checkpoint",
            repaired_count=0,
            success=False,
            detail=f"WAL checkpoint failed: {exc}",
        )


def _to_repair_result(result: BackfillResult) -> RepairResult:
    """Adapt a ``BackfillResult`` to the shared ``RepairResult`` shape."""
    return RepairResult(
        name=result.name,
        category=result.category,
        destructive=result.destructive,
        repaired_count=result.repaired_count,
        success=result.success,
        detail=result.detail,
    )


def preview_message_type_backfill(*, count: int) -> RepairResult:
    """Preview handler for the #839 message_type backfill.

    Thin shim over ``message_type_backfill.preview_backfill`` so the
    repair orchestrator's preview dispatch keeps working.
    """
    from polylogue.storage.message_type_backfill import preview_backfill

    return _to_repair_result(preview_backfill(count=count))


def repair_message_type_backfill(config: Config, dry_run: bool = False) -> RepairResult:
    """Backfill ``message_type`` for pre-#839 rows.

    Delegates to ``storage.message_type_backfill.run_backfill``; the
    implementation lives there to keep this module under its file-size
    budget (see ``docs/plans/file-size-budgets.yaml``).
    """
    from polylogue.storage.message_type_backfill import run_backfill

    return _to_repair_result(run_backfill(config, dry_run=dry_run))


def repair_message_embeddings(config: Config, dry_run: bool = False) -> RepairResult:
    """No-op embedding rebuild stub.

    Embeddings are materialized exclusively by the daemon's embedding stage
    (see #828); there is no synchronous rebuild path. The maintenance target
    is registered so planners and surfaces can name the dormant work, but
    invoking it through ``doctor --repair`` is a no-op that records dormancy
    rather than failing the run.
    """
    verb = "Would skip" if dry_run else "Skipped"
    return _repair_result(
        "message_embeddings",
        repaired_count=0,
        success=True,
        detail=f"{verb}: embedding rebuild is daemon-owned and dormant (#828).",
    )


_PREVIEW_HANDLERS: dict[str, Callable[..., RepairResult]] = {
    "session_insights": preview_session_insights,
    "action_event_read_model": preview_action_event_read_model,
    "dangling_fts": preview_dangling_fts,
    "message_type_backfill": preview_message_type_backfill,
    "orphaned_messages": preview_orphaned_messages,
    "orphaned_content_blocks": preview_orphaned_content_blocks,
    "empty_conversations": preview_empty_conversations,
    "orphaned_attachments": preview_orphaned_attachments,
    "orphaned_blobs": preview_orphaned_blobs,
}

_REPAIR_HANDLERS: dict[str, Callable[..., RepairResult]] = {
    "session_insights": repair_session_insights,
    "action_event_read_model": repair_action_event_read_model,
    "dangling_fts": repair_dangling_fts,
    "message_type_backfill": repair_message_type_backfill,
    "message_embeddings": repair_message_embeddings,
    "wal_checkpoint": repair_wal_checkpoint,
    "orphaned_messages": repair_orphaned_messages,
    "orphaned_content_blocks": repair_orphaned_content_blocks,
    "empty_conversations": repair_empty_conversations,
    "orphaned_attachments": repair_orphaned_attachments,
    "orphaned_blobs": repair_orphaned_blobs,
}


# ---------------------------------------------------------------------------
# Orchestration (run_safe_repairs, run_archive_cleanup, run_selected_maintenance)
# ---------------------------------------------------------------------------


def run_safe_repairs(
    config: Config,
    dry_run: bool = False,
    *,
    preview_counts: dict[str, int] | None = None,
    targets: tuple[str, ...] = (),
    session_insight_progress_callback: ProgressCallback | None = None,
    session_insight_progress_total: int | None = None,
) -> list[RepairResult]:
    preview_counts = preview_counts or {}
    selected = set(targets) if targets else set(SAFE_REPAIR_TARGETS)
    results: list[RepairResult] = []
    for target_name in SAFE_REPAIR_TARGETS:
        if target_name not in selected:
            continue
        if dry_run and target_name in preview_counts:
            preview = _PREVIEW_HANDLERS.get(target_name)
            if preview is not None:
                results.append(preview(count=preview_counts[target_name]))
                continue
        repair = _REPAIR_HANDLERS[target_name]
        if target_name == "session_insights":
            results.append(
                repair(
                    config,
                    dry_run=dry_run,
                    progress_callback=session_insight_progress_callback,
                    progress_total=session_insight_progress_total,
                )
            )
            continue
        results.append(repair(config, dry_run=dry_run))
    return results


def run_archive_cleanup(
    config: Config,
    dry_run: bool = False,
    *,
    preview_counts: dict[str, int] | None = None,
    targets: tuple[str, ...] = (),
) -> list[RepairResult]:
    preview_counts = preview_counts or {}
    selected = set(targets) if targets else set(CLEANUP_TARGETS)
    results: list[RepairResult] = []
    for target_name in CLEANUP_TARGETS:
        if target_name not in selected:
            continue
        if dry_run and target_name in preview_counts:
            results.append(_PREVIEW_HANDLERS[target_name](count=preview_counts[target_name]))
            continue
        results.append(_REPAIR_HANDLERS[target_name](config, dry_run=dry_run))
    return results


def run_selected_maintenance(
    config: Config,
    *,
    repair: bool,
    cleanup: bool,
    dry_run: bool = False,
    preview_counts: dict[str, int] | None = None,
    targets: tuple[str, ...] = (),
    session_insight_progress_callback: ProgressCallback | None = None,
    session_insight_progress_total: int | None = None,
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
                session_insight_progress_callback=session_insight_progress_callback,
                session_insight_progress_total=session_insight_progress_total,
            )
        )
    if cleanup:
        results.extend(
            run_archive_cleanup(config, dry_run=dry_run, preview_counts=preview_counts, targets=cleanup_targets)
        )
    return results


__all__ = [
    "ArchiveDebtStatus",
    "RepairResult",
    "action_event_repair_count",
    "collect_archive_debt_statuses_sync",
    "count_empty_conversations_sync",
    "count_orphaned_attachments_sync",
    "count_orphaned_blobs_sync",
    "count_orphaned_content_blocks_sync",
    "count_orphaned_messages_sync",
    "count_messages_by_type_sync",
    "count_unclassified_message_type_sync",
    "dangling_fts_repair_count",
    "preview_action_event_read_model",
    "preview_counts_from_archive_debt",
    "preview_dangling_fts",
    "preview_empty_conversations",
    "preview_orphaned_attachments",
    "preview_orphaned_blobs",
    "preview_orphaned_content_blocks",
    "preview_orphaned_messages",
    "preview_message_type_backfill",
    "preview_session_insights",
    "repair_action_event_read_model",
    "repair_dangling_fts",
    "repair_empty_conversations",
    "repair_message_type_backfill",
    "repair_orphaned_attachments",
    "repair_orphaned_blobs",
    "repair_orphaned_content_blocks",
    "repair_orphaned_messages",
    "repair_session_insights",
    "repair_wal_checkpoint",
    "run_archive_cleanup",
    "run_safe_repairs",
    "run_selected_maintenance",
    "session_insight_repair_count",
]
