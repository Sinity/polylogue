from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, cast

import pytest

from polylogue.config import Config
from polylogue.maintenance.models import DerivedModelStatus
from polylogue.storage import repair as repair_mod
from polylogue.storage.insights.session.runtime import SessionInsightCounts, SessionInsightStatusSnapshot


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
        "Action-event read model pending (12 missing sessions, 5 stale action-event rows, 9 pending action-event FTS rows)"
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
        "empty_sessions": repair_mod.ArchiveDebtStatus(
            name="empty_sessions",
            category=repair_mod._maintenance_target_spec("empty_sessions").category,
            destructive=True,
            issue_count=4,
            detail="needs cleanup",
            maintenance_target="empty_sessions",
        ),
    }

    assert repair_mod.preview_counts_from_archive_debt(statuses) == {
        "session_insights": 0,
        "dangling_fts": 0,
        "empty_sessions": 4,
    }


def test_probe_only_archive_debt_skips_large_message_scans(monkeypatch: pytest.MonkeyPatch) -> None:
    class Conn:
        def execute(self, *_args: object, **_kwargs: object) -> object:
            raise AssertionError("large probe mode should not run exact SQL scans")

    statuses = {
        "messages_fts": _status(),
        "action_events": _status(),
        "action_events_fts": _status(),
    }
    monkeypatch.setattr(repair_mod, "_table_has_more_than", lambda *_args: True)
    monkeypatch.setattr(repair_mod, "count_orphaned_messages_sync", lambda _conn: (_ for _ in ()).throw(AssertionError))
    monkeypatch.setattr(repair_mod, "count_empty_sessions_sync", lambda _conn: (_ for _ in ()).throw(AssertionError))
    monkeypatch.setattr(
        repair_mod, "count_unclassified_message_type_sync", lambda _conn: (_ for _ in ()).throw(AssertionError)
    )
    monkeypatch.setattr(repair_mod, "count_orphaned_attachments_sync", lambda _conn: 0)

    debt = repair_mod.collect_archive_debt_statuses_sync(
        cast(Any, Conn()), derived_statuses=statuses, include_expensive=False, probe_only=True
    )

    assert debt["orphaned_messages"].skipped is True
    assert debt["empty_sessions"].skipped is True
    assert debt["message_type_backfill"].skipped is True
    assert debt["orphaned_attachments"].skipped is False


def _ready_session_insight_status() -> SessionInsightStatusSnapshot:
    return SessionInsightStatusSnapshot(
        profile_rows_ready=True,
        latency_profile_rows_ready=True,
        work_event_inference_rows_ready=True,
        work_event_inference_fts_ready=True,
        phase_inference_rows_ready=True,
        threads_ready=True,
        threads_fts_ready=True,
        tag_rollups_ready=True,
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


def test_repair_session_insights_uses_stale_profile_candidates(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls: list[tuple[str, tuple[str, ...] | None]] = []

    class FakeArchive:
        def session_insight_status(self) -> SessionInsightStatusSnapshot:
            return next(statuses)

        def __enter__(self) -> FakeArchive:
            return self

        def __exit__(self, *_args: object) -> None:
            pass

    stale_status = SessionInsightStatusSnapshot(
        profile_rows_ready=False,
        latency_profile_rows_ready=True,
        work_event_inference_rows_ready=False,
        work_event_inference_fts_ready=True,
        phase_inference_rows_ready=True,
        threads_ready=True,
        threads_fts_ready=True,
        tag_rollups_ready=True,
        stale_profile_row_count=2,
        stale_work_event_inference_count=2,
        work_event_inference_fts_count=4,
        work_event_inference_count=4,
        thread_fts_count=1,
        thread_count=1,
    )
    statuses = iter((stale_status, _ready_session_insight_status()))

    def fake_rebuild(_archive: FakeArchive, *, session_ids: tuple[str, ...] | None, **_kwargs: object) -> Any:
        calls.append(("rebuild", session_ids))
        return SessionInsightCounts(profiles=2, work_events=2)

    monkeypatch.setattr(
        "polylogue.storage.sqlite.archive_tiers.archive.ArchiveStore.open_existing",
        lambda _archive_root, read_only=False: FakeArchive(),
    )
    monkeypatch.setattr(
        "polylogue.api.archive._rebuild_archive_session_insights",
        fake_rebuild,
    )

    result = repair_mod.repair_session_insights(_config(tmp_path), dry_run=False)

    assert result.success is True
    assert result.repaired_count == 4
    assert ("rebuild", None) in calls


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


def test_repair_dangling_fts_uses_targeted_missing_row_repair(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls: list[tuple[str, object]] = []

    class _Cursor:
        def __init__(self, value: object) -> None:
            self.value = value

        def fetchone(self) -> tuple[object, ...]:
            return (self.value,)

    class FakeConn:
        message_count = 8
        action_count = 4

        def execute(self, sql: str, params: object = ()) -> _Cursor:
            if sql.startswith("PRAGMA "):
                return _Cursor(None)
            if "sqlite_master" in sql and params == ("content_blocks",):
                return _Cursor(1)
            if "sqlite_master" in sql and params in {("action_events",), ("action_events_fts_docsize",)}:
                return _Cursor(1)
            if "sqlite_master" in sql and "messages_fts" in sql and "trigger" not in sql:
                return _Cursor("messages_fts")
            if "sqlite_master" in sql and "action_events" in sql:
                return _Cursor("action_events")
            if "sqlite_master" in sql and "content_blocks" in sql:
                return _Cursor("content_blocks")
            if "COUNT(*) FROM messages_fts_docsize" in sql:
                return _Cursor(self.message_count)
            if "COUNT(*) FROM action_events_fts_docsize" in sql:
                return _Cursor(self.action_count)
            if "FROM messages AS m" in sql:
                return _Cursor(10)
            if "SELECT COUNT(*) FROM action_events" in sql:
                return _Cursor(5)
            raise AssertionError(f"unexpected SQL: {sql}")

        def commit(self) -> None:
            calls.append(("commit", ()))

    @contextmanager
    def fake_connection_context(_path: Path) -> Iterator[FakeConn]:
        yield FakeConn()

    def repair_messages(conn: FakeConn) -> int:
        conn.message_count = 10
        calls.append(("message", ()))
        return 2

    def repair_actions(conn: FakeConn) -> int:
        conn.action_count = 5
        calls.append(("action", ()))
        return 1

    monkeypatch.setattr("polylogue.storage.sqlite.connection.connection_context", fake_connection_context)
    monkeypatch.setattr(
        "polylogue.storage.fts.dangling_repair.insert_missing_message_fts_rows_sync",
        repair_messages,
    )
    monkeypatch.setattr(
        "polylogue.storage.fts.dangling_repair.insert_missing_action_fts_rows_sync",
        repair_actions,
    )
    monkeypatch.setattr(
        "polylogue.storage.fts.dangling_repair._repair_session_work_events_fts_rows_sync",
        lambda _conn: (0, 0, 0, 0),
    )
    monkeypatch.setattr(
        "polylogue.storage.fts.dangling_repair._repair_work_threads_fts_rows_sync",
        lambda _conn: (0, 0, 0, 0),
    )
    monkeypatch.setattr(
        "polylogue.storage.fts.dangling_repair._record_optional_derived_surface",
        lambda _conn, **_kwargs: True,
    )
    monkeypatch.setattr("polylogue.storage.fts.dangling_repair._triggers_present_sync", lambda _conn, _names: True)
    monkeypatch.setattr(
        "polylogue.storage.fts.dangling_repair.record_fts_surface_state_sync",
        lambda _conn, **kwargs: calls.append(("record", kwargs["surface"])),
    )

    result = repair_mod.repair_dangling_fts(_config(tmp_path), dry_run=False)

    assert result.success is True
    assert "repaired index" in result.detail
    assert ("message", ()) in calls
    assert ("action", ()) in calls
    assert ("record", "messages_fts") in calls
    assert ("record", "action_events_fts") in calls
    assert ("commit", ()) in calls


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
                "valid_source_session_count": 1,
                "materialized_session_count": 1,
                "count": 1,
                "action_fts_count": 1,
                "stale_count": 0,
                "orphan_tool_block_count": 0,
                "matches_version": True,
            }
        return {
            "ready": False,
            "valid_source_session_count": 1,
            "materialized_session_count": 1,
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
            AssertionError("full FTS rebuild should not enumerate per-session targets")
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


def test_repair_action_event_read_model_is_archive_noop(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_database
    from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier

    initialize_archive_database(tmp_path / "index.db", ArchiveTier.INDEX)
    monkeypatch.setattr(
        "polylogue.storage.sqlite.connection.connection_context",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("legacy repair path must not open")),
    )

    result = repair_mod.repair_action_event_read_model(_config(tmp_path), dry_run=False)
    preview = repair_mod.repair_action_event_read_model(_config(tmp_path), dry_run=True)

    assert result.success is True
    assert result.repaired_count == 0
    assert "retired for archives" in result.detail
    assert preview.success is True
    assert preview.repaired_count == 0
    assert preview.detail.startswith("Would: action-event materialized read model retired")
