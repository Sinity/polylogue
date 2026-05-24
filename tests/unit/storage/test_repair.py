from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest

from polylogue.config import Config
from polylogue.maintenance.models import DerivedModelStatus
from polylogue.storage import repair as repair_mod
from polylogue.storage.insights.session.runtime import SessionInsightStatusSnapshot


def _config(tmp_path: Path) -> Config:
    return Config(archive_root=tmp_path, render_root=tmp_path, sources=[], db_path=tmp_path / "archive.db")


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


def _ready_session_insight_status() -> SessionInsightStatusSnapshot:
    return SessionInsightStatusSnapshot(
        profile_rows_ready=True,
        work_event_inference_rows_ready=True,
        work_event_inference_fts_ready=True,
        phase_inference_rows_ready=True,
        threads_ready=True,
        threads_fts_ready=True,
        tag_rollups_ready=True,
        day_summaries_ready=True,
        week_summaries_ready=True,
    )


def test_repair_session_insights_noops_when_ready(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    @contextmanager
    def fake_connection_context(_path: Path) -> Iterator[object]:
        yield object()

    def fail_rebuild(*args: object, **kwargs: object) -> int:
        raise AssertionError("ready session insights must not run a full rebuild")

    monkeypatch.setattr("polylogue.storage.sqlite.connection.connection_context", fake_connection_context)
    monkeypatch.setattr(
        "polylogue.storage.insights.session.status.session_insight_status_sync",
        lambda _conn: _ready_session_insight_status(),
    )
    monkeypatch.setattr(
        "polylogue.storage.insights.session.rebuild.rebuild_session_insights_sync",
        fail_rebuild,
    )

    result = repair_mod.repair_session_insights(_config(tmp_path), dry_run=False)

    assert result.success is True
    assert result.repaired_count == 0
    assert result.detail == "Session insights already ready"


def test_offline_maintenance_refuses_live_daemon(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr("polylogue.maintenance.offline_guard.running_daemon_pid", lambda _config: 1234)

    results = repair_mod.run_selected_maintenance(
        _config(tmp_path),
        repair=True,
        cleanup=False,
        targets=("session_insights",),
    )

    assert len(results) == 1
    assert results[0].name == "session_insights"
    assert results[0].success is False
    assert "polylogued PID 1234 is running" in results[0].detail


def test_offline_maintenance_preview_allowed_with_live_daemon(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr("polylogue.maintenance.offline_guard.running_daemon_pid", lambda _config: 1234)

    results = repair_mod.run_selected_maintenance(
        _config(tmp_path),
        repair=True,
        cleanup=False,
        dry_run=True,
        preview_counts={"session_insights": 2},
        targets=("session_insights",),
    )

    assert len(results) == 1
    assert results[0].success is True
    assert results[0].repaired_count == 2


def test_repair_action_event_read_model_rebuilds_fts_when_stale_extra_rows(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[str] = []

    class FakeConn:
        def commit(self) -> None:
            calls.append("commit")

    @contextmanager
    def fake_connection_context(_path: Path) -> Iterator[FakeConn]:
        yield FakeConn()

    def fake_status(_conn: FakeConn) -> dict[str, Any]:
        if "rebuild_fts" in calls:
            return {
                "ready": True,
                "valid_source_conversation_count": 1,
                "materialized_conversation_count": 1,
                "count": 1,
                "action_fts_count": 1,
                "stale_count": 0,
                "orphan_tool_block_count": 0,
                "matches_version": True,
            }
        return {
            "ready": False,
            "valid_source_conversation_count": 1,
            "materialized_conversation_count": 1,
            "count": 1,
            "action_fts_count": 2,
            "stale_count": 0,
            "orphan_tool_block_count": 0,
            "matches_version": True,
        }

    monkeypatch.setattr("polylogue.storage.sqlite.connection.connection_context", fake_connection_context)
    monkeypatch.setattr("polylogue.storage.action_events.status.action_event_read_model_status_sync", fake_status)
    monkeypatch.setattr(
        "polylogue.storage.action_events.rebuild_runtime.action_event_repair_candidates_sync", lambda _conn: []
    )
    monkeypatch.setattr(
        "polylogue.storage.action_events.rebuild_runtime.rebuild_action_event_read_model_sync", lambda *a, **k: 0
    )
    monkeypatch.setattr(
        "polylogue.storage.action_events.rebuild_runtime.valid_action_event_source_ids_sync",
        lambda _conn: (_ for _ in ()).throw(
            AssertionError("full FTS rebuild should not enumerate per-conversation targets")
        ),
    )
    monkeypatch.setattr(
        "polylogue.storage.fts.fts_lifecycle.repair_fts_index_sync",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("stale extra FTS rows require a full rebuild")),
    )
    monkeypatch.setattr(
        "polylogue.storage.fts.fts_lifecycle.rebuild_fts_index_sync", lambda _conn: calls.append("rebuild_fts")
    )

    result = repair_mod.repair_action_event_read_model(_config(tmp_path), dry_run=False)

    assert result.success is True
    assert "rebuild_fts" in calls
