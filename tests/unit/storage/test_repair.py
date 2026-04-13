from __future__ import annotations

from polylogue.maintenance_models import DerivedModelStatus
from polylogue.storage import repair as repair_mod


def _status(
    *,
    source_documents: int = 0,
    materialized_documents: int = 0,
    materialized_rows: int = 0,
    pending_documents: int = 0,
    pending_rows: int = 0,
    stale_rows: int = 0,
) -> DerivedModelStatus:
    return DerivedModelStatus(
        name="test",
        ready=pending_documents == 0 and pending_rows == 0 and stale_rows == 0,
        detail="",
        source_documents=source_documents,
        materialized_documents=materialized_documents,
        materialized_rows=materialized_rows,
        pending_documents=pending_documents,
        pending_rows=pending_rows,
        stale_rows=stale_rows,
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
