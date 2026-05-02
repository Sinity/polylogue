from __future__ import annotations

from polylogue.maintenance.models import DerivedModelStatus
from polylogue.storage import repair as repair_mod


def _status(
    *,
    source_documents: int = 0,
    materialized_documents: int = 0,
    materialized_rows: int = 0,
    pending_documents: int = 0,
    pending_rows: int = 0,
    stale_rows: int = 0,
    orphan_rows: int = 0,
) -> DerivedModelStatus:
    return DerivedModelStatus(
        name="test",
        ready=pending_documents == 0 and pending_rows == 0 and stale_rows == 0 and orphan_rows == 0,
        detail="",
        source_documents=source_documents,
        materialized_documents=materialized_documents,
        materialized_rows=materialized_rows,
        pending_documents=pending_documents,
        pending_rows=pending_rows,
        stale_rows=stale_rows,
        orphan_rows=orphan_rows,
    )


def test_action_event_repair_detail_reports_pending_fts_rows_only() -> None:
    statuses = {
        "action_events": _status(source_documents=0, materialized_documents=0, materialized_rows=323048),
        "action_events_fts": _status(materialized_rows=0, pending_rows=323048),
    }

    assert repair_mod.action_event_repair_count(statuses) == 323048
    assert repair_mod._action_event_repair_detail(statuses) == (
        "Action-event read model pending (323,048 pending action-event FTS rows)"
    )


def test_action_event_repair_detail_reports_missing_and_stale_rows() -> None:
    statuses = {
        "action_events": _status(
            source_documents=12,
            materialized_documents=0,
            materialized_rows=9,
            pending_documents=12,
            stale_rows=5,
        ),
        "action_events_fts": _status(materialized_rows=0, pending_rows=9),
    }

    assert repair_mod.action_event_repair_count(statuses) == 26
    assert repair_mod._action_event_repair_detail(statuses) == (
        "Action-event read model pending (12 missing conversations, 5 stale action-event rows, 9 pending action-event FTS rows)"
    )


def test_action_event_repair_detail_reports_orphan_rows() -> None:
    statuses = {
        "action_events": _status(
            source_documents=4,
            materialized_documents=4,
            materialized_rows=10,
            orphan_rows=2,
        ),
        "action_events_fts": _status(materialized_rows=10),
    }

    assert repair_mod.action_event_repair_count(statuses) == 2
    assert repair_mod._action_event_repair_detail(statuses) == (
        "Action-event read model pending (2 orphan action-event rows)"
    )


def test_action_event_repair_detail_reports_stale_extra_fts_rows() -> None:
    statuses = {
        "action_events": _status(
            source_documents=4,
            materialized_documents=4,
            materialized_rows=10,
        ),
        "action_events_fts": _status(
            materialized_rows=13,
            stale_rows=3,
        ),
    }

    assert repair_mod.action_event_repair_count(statuses) == 3
    assert repair_mod._action_event_repair_detail(statuses) == (
        "Action-event read model pending (3 stale extra action-event FTS rows)"
    )


def test_preview_counts_from_archive_debt_include_healthy_preview_targets_only() -> None:
    statuses = {
        "session_insights": repair_mod.ArchiveDebtStatus(
            name="session_insights",
            category=repair_mod._maintenance_target_spec("session_insights").category,
            destructive=False,
            issue_count=0,
            detail="ready",
            maintenance_target="session_insights",
        ),
        "dangling_fts": repair_mod.ArchiveDebtStatus(
            name="dangling_fts",
            category=repair_mod._maintenance_target_spec("dangling_fts").category,
            destructive=False,
            issue_count=0,
            detail="ready",
            maintenance_target="dangling_fts",
        ),
        "orphaned_messages": repair_mod.ArchiveDebtStatus(
            name="orphaned_messages",
            category=repair_mod._maintenance_target_spec("orphaned_messages").category,
            destructive=True,
            issue_count=0,
            detail="clean",
            maintenance_target="orphaned_messages",
        ),
        "empty_conversations": repair_mod.ArchiveDebtStatus(
            name="empty_conversations",
            category=repair_mod._maintenance_target_spec("empty_conversations").category,
            destructive=True,
            issue_count=4,
            detail="needs cleanup",
            maintenance_target="empty_conversations",
        ),
    }

    assert repair_mod.preview_counts_from_archive_debt(statuses) == {
        "session_insights": 0,
        "dangling_fts": 0,
        "empty_conversations": 4,
    }
