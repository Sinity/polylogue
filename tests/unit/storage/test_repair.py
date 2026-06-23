from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, cast

import pytest

from polylogue.config import Config
from polylogue.maintenance.models import DerivedModelStatus
from polylogue.storage import repair as repair_mod
from polylogue.storage.fts.dangling_repair import DanglingFtsRepairOutcome
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
        def execute(self, sql: str, params: object = ()) -> _Cursor:
            if sql.startswith("PRAGMA "):
                return _Cursor(None)
            if "sqlite_master" in sql and "messages_fts" in sql:
                return _Cursor("messages_fts")
            raise AssertionError(f"unexpected SQL: {sql}")

        def commit(self) -> None:
            calls.append(("commit", ()))

    @contextmanager
    def fake_connection_context() -> Iterator[FakeConn]:
        yield FakeConn()

    ops_db = tmp_path / "ops.db"
    with sqlite3.connect(ops_db) as conn:
        conn.execute(
            """
            CREATE TABLE convergence_debt (
                debt_id TEXT PRIMARY KEY,
                stage TEXT NOT NULL,
                target_type TEXT NOT NULL,
                target_id TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'failed' CHECK(status IN ('failed', 'deferred')),
                priority INTEGER NOT NULL DEFAULT 0,
                attempts INTEGER NOT NULL DEFAULT 0,
                last_error TEXT,
                next_retry_at TEXT,
                materializer_version TEXT,
                created_at_ms INTEGER NOT NULL,
                updated_at_ms INTEGER NOT NULL,
                UNIQUE(stage, target_type, target_id)
            )
            """
        )
        conn.execute(
            """
            INSERT INTO convergence_debt (
                debt_id, stage, target_type, target_id, status, priority,
                attempts, last_error, next_retry_at, materializer_version,
                created_at_ms, updated_at_ms
            )
            VALUES (
                'debt-1', 'fts', 'fts_surface', 'messages_fts', 'failed', 0,
                1, 'stale ledger', '1970-01-01T00:00:00+00:00', NULL, 1, 1
            )
            """
        )

    monkeypatch.setattr(repair_mod, "_open_archive_index_connection", fake_connection_context)

    def fail_full_rebuild(_conn: FakeConn) -> None:
        raise AssertionError("repair_dangling_fts must not run the full FTS rebuild path")

    monkeypatch.setattr(
        "polylogue.storage.fts.fts_lifecycle.rebuild_fts_index_sync",
        fail_full_rebuild,
    )
    monkeypatch.setattr(
        "polylogue.storage.fts.dangling_repair.repair_missing_fts_rows",
        lambda _conn: DanglingFtsRepairOutcome(
            repaired_count=2,
            success=True,
            detail="FTS sync: repaired index",
        ),
    )

    result = repair_mod.repair_dangling_fts(_config(tmp_path), dry_run=False)

    assert result.success is True
    assert result.repaired_count == 2
    assert result.detail == "FTS sync: repaired index"
    assert ("commit", ()) in calls
    with sqlite3.connect(ops_db) as conn:
        row = conn.execute(
            """
            SELECT status, last_error, next_retry_at, updated_at_ms
            FROM convergence_debt
            WHERE stage = 'fts' AND target_type = 'fts_surface' AND target_id = 'messages_fts'
            """
        ).fetchone()
    assert row is None
