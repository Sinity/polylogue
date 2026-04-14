from __future__ import annotations

from polylogue.storage.action_event_artifacts import ActionEventArtifactState
from polylogue.storage.derived_status_products import build_action_statuses


def test_action_event_artifact_state_reports_pending_and_extra_fts_rows() -> None:
    state = ActionEventArtifactState(
        source_conversations=4,
        materialized_conversations=4,
        materialized_rows=10,
        fts_rows=13,
    )

    assert state.rows_ready is True
    assert state.fts_ready is False
    assert state.pending_fts_rows == 0
    assert state.excess_fts_rows == 3
    assert state.repair_item_count == 3
    assert state.repair_detail() == (
        "Action-event read model pending (3 stale extra action-event FTS rows)"
    )


def test_action_event_artifact_state_reports_missing_and_stale_rows() -> None:
    state = ActionEventArtifactState(
        source_conversations=12,
        materialized_conversations=9,
        materialized_rows=21,
        fts_rows=17,
        stale_rows=5,
        orphan_rows=2,
        matches_version=False,
    )

    row_status = state.row_status()
    fts_status = state.fts_status()

    assert row_status.ready is False
    assert row_status.pending_documents == 3
    assert row_status.stale_rows == 5
    assert row_status.orphan_rows == 2
    assert "stale rows 5" in row_status.detail
    assert "orphan rows 2" in row_status.detail

    assert fts_status.ready is False
    assert fts_status.pending_rows == 4
    assert fts_status.stale_rows == 0
    assert "4 pending rows" in fts_status.detail


def test_action_event_artifact_state_treats_orphan_rows_as_unready_and_repairable() -> None:
    state = ActionEventArtifactState(
        source_conversations=4,
        materialized_conversations=4,
        materialized_rows=10,
        fts_rows=10,
        orphan_rows=2,
    )

    row_status = state.row_status()

    assert state.rows_ready is False
    assert state.fts_ready is True
    assert state.repair_item_count == 2
    assert state.repair_detail() == (
        "Action-event read model pending (2 orphan action-event rows)"
    )
    assert row_status.ready is False
    assert row_status.orphan_rows == 2
    assert "orphan rows 2" in row_status.detail


def test_build_action_statuses_marks_extra_fts_rows_as_stale() -> None:
    statuses = build_action_statuses(
        {
            "message_fts_exact_counts": True,
            "message_source_rows": 0,
            "message_fts_rows": 0,
            "message_fts_ready": True,
            "action_source_documents": 4,
            "action_documents": 4,
            "action_rows": 10,
            "action_fts_rows": 13,
            "action_stale_rows": 0,
            "action_orphan_rows": 0,
            "action_matches_version": True,
        }
    )

    action_fts = statuses["action_events_fts"]
    assert action_fts.ready is False
    assert action_fts.pending_rows == 0
    assert action_fts.stale_rows == 3
    assert action_fts.detail == "Action-event FTS pending (13/10 rows; 3 stale extra rows)"
