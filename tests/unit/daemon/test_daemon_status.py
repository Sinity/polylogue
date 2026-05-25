from __future__ import annotations

import sqlite3
from datetime import timedelta
from pathlib import Path
from typing import Any, cast
from unittest.mock import Mock, patch

import pytest

from polylogue.core.json import JSONDocument
from polylogue.daemon import status as status_module
from polylogue.daemon.status import build_daemon_status, daemon_status_payload, format_daemon_status_lines
from polylogue.daemon.status_snapshot import get_status_snapshot_payload, refresh_status_snapshot
from polylogue.sources.live.cursor import CursorStore
from tests.infra.frozen_clock import FrozenClock


def test_status_snapshot_serves_cached_payload_without_rebuilding_status(monkeypatch: pytest.MonkeyPatch) -> None:
    payload: JSONDocument = {"ok": True, "daemon_liveness": True, "checked_at": "cached"}
    refresh_status_snapshot(payload=payload)

    monkeypatch.setattr(
        "polylogue.daemon.status.daemon_status_payload",
        lambda: (_ for _ in ()).throw(AssertionError("request path must not rebuild rich status")),
    )

    result = get_status_snapshot_payload()

    assert result["checked_at"] == "cached"
    snapshot = result["status_snapshot"]
    assert isinstance(snapshot, dict)
    assert snapshot["state"] == "fresh"


def test_status_snapshot_refresh_default_stays_request_safe(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db = tmp_path / "polylogue.db"
    db.touch()
    monkeypatch.setattr("polylogue.daemon.status_snapshot.db_path", lambda: db)
    monkeypatch.setattr(
        "polylogue.daemon.status.daemon_status_payload",
        lambda: (_ for _ in ()).throw(AssertionError("periodic snapshot must not build rich status")),
    )

    snapshot = refresh_status_snapshot()

    status_snapshot = snapshot.payload["status_snapshot"]
    assert isinstance(status_snapshot, dict)
    assert status_snapshot["state"] == "minimal"
    assert snapshot.payload["db_path"] == str(db)


def test_build_daemon_status_reports_failed_live_cursor_files(tmp_path: Path) -> None:
    db = tmp_path / "polylogue.db"
    failed = tmp_path / "failed.jsonl"
    failed.write_text('{"bad":true}\n')
    cursor = CursorStore(db)
    cursor.mark_failed(failed)

    with (
        patch("polylogue.daemon.status.db_path", return_value=db),
        patch("polylogue.daemon.status._check_daemon_liveness", return_value=False),
        patch("polylogue.daemon.status._blob_size_info", return_value=0),
        patch("polylogue.daemon.status._fts_readiness_info", return_value={}),
        patch("polylogue.daemon.status._insight_freshness_info", return_value={}),
    ):
        status = build_daemon_status(sources=())

    assert status.failing_files == [str(failed)]
    assert status.live_cursor.tracked_file_count == 1
    assert status.live_cursor.failed_file_count == 1
    assert status.live_cursor.in_backoff_file_count == 1
    assert status.live_cursor.failing_files[0].source_path == str(failed)
    assert status.live_cursor.failing_files[0].failure_count == 1
    assert status.live_cursor.failing_files[0].next_retry_at is not None


def test_daemon_status_payload_and_plain_output_include_failed_files(tmp_path: Path) -> None:
    db = tmp_path / "polylogue.db"
    failed = tmp_path / "failed.jsonl"
    failed.write_text('{"bad":true}\n')
    cursor = CursorStore(db)
    cursor.mark_failed(failed)

    with (
        patch("polylogue.daemon.status.db_path", return_value=db),
        patch("polylogue.daemon.status._check_daemon_liveness", return_value=False),
        patch("polylogue.daemon.status._blob_size_info", return_value=0),
        patch("polylogue.daemon.status._fts_readiness_info", return_value={}),
        patch("polylogue.daemon.status._insight_freshness_info", return_value={}),
    ):
        payload = daemon_status_payload(sources=())

    assert payload["failing_files"] == [str(failed)]
    live_cursor = payload["live_cursor"]
    assert isinstance(live_cursor, dict)
    assert live_cursor["tracked_file_count"] == 1
    assert live_cursor["failed_file_count"] == 1
    failing_files = live_cursor["failing_files"]
    assert isinstance(failing_files, list)
    first_failure = failing_files[0]
    assert isinstance(first_failure, dict)
    assert first_failure["source_path"] == str(failed)
    lines = format_daemon_status_lines(payload)
    assert "Live cursor: 1 tracked, 1 failed, 0 excluded, 0 retry due, 1 in backoff" in lines
    assert "Failing files: 1" in lines
    assert f"  {failed}" in lines


def test_plain_daemon_status_reports_bounded_embedding_pending_messages() -> None:
    payload: JSONDocument = {
        "embedding_readiness": {
            "embedding_enabled": False,
            "embedding_has_voyage_key": True,
            "embedding_status": "none",
            "embedding_freshness_status": "none",
            "embedding_retrieval_ready": False,
            "embedding_pending_count": 7,
            "embedding_pending_message_count": 0,
            "embedding_pending_message_count_exact": False,
        }
    }

    lines = format_daemon_status_lines(payload)

    assert (
        "Embeddings: disabled (key present; none/none, not ready; 7 pending convs, pending msgs not calculated)"
        in lines
    )


def test_daemon_status_caps_failed_file_samples(tmp_path: Path) -> None:
    db = tmp_path / "polylogue.db"
    cursor = CursorStore(db)
    for index in range(55):
        failed = tmp_path / f"failed-{index:02d}.jsonl"
        failed.write_text('{"bad":true}\n')
        cursor.mark_failed(failed)

    with (
        patch("polylogue.daemon.status.db_path", return_value=db),
        patch("polylogue.daemon.status._check_daemon_liveness", return_value=False),
        patch("polylogue.daemon.status._blob_size_info", return_value=0),
        patch("polylogue.daemon.status._fts_readiness_info", return_value={}),
        patch("polylogue.daemon.status._insight_freshness_info", return_value={}),
    ):
        payload = daemon_status_payload(sources=())

    failing_files = payload["failing_files"]
    assert isinstance(failing_files, list)
    assert len(failing_files) == 50
    live_cursor = payload["live_cursor"]
    assert isinstance(live_cursor, dict)
    assert live_cursor["failed_file_count"] == 55
    assert live_cursor["sampled_file_count"] == 50
    assert live_cursor["omitted_file_count"] == 5
    lines = format_daemon_status_lines(payload)
    assert "Failing files: 50 shown, 5 omitted" in lines


def test_daemon_status_reports_live_ingest_attempts(tmp_path: Path) -> None:
    db = tmp_path / "polylogue.db"
    source = tmp_path / "session.jsonl"
    source.write_text('{"a":1}\n')
    cursor = CursorStore(db)
    attempt_id = cursor.begin_ingest_attempt(
        paths=[source],
        input_bytes=source.stat().st_size,
        queued_file_count=1,
    )
    cursor.update_ingest_attempt(
        attempt_id,
        phase="full_parse",
        succeeded_file_count=0,
        failed_file_count=0,
        source_payload_read_bytes=0,
        cursor_fingerprint_read_bytes=0,
        parse_time_s=0.0,
        current_source="codex",
        current_path=source,
        rss_current_mb=42.0,
        cgroup_path="/user.slice/test.scope",
        cgroup_memory_current_mb=2048.0,
        cgroup_memory_peak_mb=4096.0,
    )
    cursor.record_ingest_stage_event(
        attempt_id,
        phase="full_parse",
        status="running",
        queued_file_count=1,
        needed_file_count=1,
        skipped_file_count=0,
        succeeded_file_count=0,
        failed_file_count=0,
        input_bytes=source.stat().st_size,
        source_payload_read_bytes=0,
        cursor_fingerprint_read_bytes=0,
        archive_write_bytes_delta=4096,
        parse_time_s=0.0,
        total_time_s=2.0,
        current_source="codex",
        current_path=source,
        stage_timings_json='{"full_parse": 1.25, "convergence": 0.75}',
    )

    with (
        patch("polylogue.daemon.status.db_path", return_value=db),
        patch("polylogue.daemon.status._check_daemon_liveness", return_value=False),
        patch("polylogue.daemon.status._blob_size_info", return_value=0),
        patch("polylogue.daemon.status._fts_readiness_info", return_value={}),
        patch("polylogue.daemon.status._insight_freshness_info", return_value={}),
    ):
        payload = daemon_status_payload(sources=())

    attempts = payload["live_ingest_attempts"]
    assert isinstance(attempts, dict)
    assert attempts["running_count"] == 1
    recent = attempts["recent"]
    assert isinstance(recent, list)
    latest = recent[0]
    assert isinstance(latest, dict)
    assert latest["phase"] == "full_parse"
    assert latest["current_path"] == str(source)
    assert latest["rss_current_mb"] == 42.0
    assert latest["cgroup_path"] == "/user.slice/test.scope"
    assert latest["cgroup_memory_current_mb"] == 2048.0
    assert latest["cgroup_memory_peak_mb"] == 4096.0
    assert latest["total_read_bytes"] == 0
    assert latest["read_amplification"] == 0.0
    assert latest["files_per_second"] == 0.0
    assert latest["archive_write_bytes_delta"] == 4096
    assert latest["total_time_s"] == 2.0
    assert latest["stage_timings_s"] == {"full_parse": 1.25, "convergence": 0.75}
    catchup = payload["catchup"]
    assert isinstance(catchup, dict)
    assert catchup["mode"] == "catching_up"
    assert catchup["current_phase"] == "full_parse"
    assert catchup["queued_file_count"] == 1
    assert catchup["read_amplification"] == 0.0
    recent_events = catchup["recent_events"]
    assert isinstance(recent_events, list)
    first_event = recent_events[0]
    assert isinstance(first_event, dict)
    assert first_event["current_path"] == str(source)
    lines = format_daemon_status_lines(payload)
    assert "Live ingest attempts: 1 running" in lines
    assert "  latest: running full_parse 0/1 files" in lines
    assert "  workload: read amp 0.00x, 0.00 MiB/s source, 0.00 files/s" in lines
    assert "  memory: cgroup 2048.0 MiB peak 4096.0 MiB" in lines
    assert "Catch-up: catching_up 0/1 files, read amp 0.0x" in lines


def test_daemon_status_reports_convergence_debt_separately(tmp_path: Path) -> None:
    db = tmp_path / "polylogue.db"
    source = tmp_path / "session.jsonl"
    source.write_text('{"a":1}\n')
    cursor = CursorStore(db)
    cursor.record_convergence_debt(
        stage="insights",
        subject_type="source_path",
        subject_id=str(source),
        error="legacy payload missing provenance",
    )

    with (
        patch("polylogue.daemon.status.db_path", return_value=db),
        patch("polylogue.daemon.status._check_daemon_liveness", return_value=False),
        patch("polylogue.daemon.status._blob_size_info", return_value=0),
        patch("polylogue.daemon.status._fts_readiness_info", return_value={}),
        patch("polylogue.daemon.status._insight_freshness_info", return_value={}),
    ):
        payload = daemon_status_payload(sources=())

    convergence = payload["convergence"]
    assert isinstance(convergence, dict)
    assert convergence["failed_count"] == 1
    stages = convergence["stage_summaries"]
    assert isinstance(stages, list)
    first_stage = stages[0]
    assert isinstance(first_stage, dict)
    assert first_stage["stage"] == "insights"
    recent = convergence["recent"]
    assert isinstance(recent, list)
    first_recent = recent[0]
    assert isinstance(first_recent, dict)
    assert first_recent["subject_id"] == str(source)
    lines = format_daemon_status_lines(payload)
    assert "Convergence debt: 1 failed, 0 retry due" in lines
    assert "  insights: 1 failed, 0 retry due" in lines


def test_daemon_status_payload_reuses_bounded_probe_results(tmp_path: Path) -> None:
    db = tmp_path / "polylogue.db"
    db.touch()
    db_info = Mock(
        return_value={
            "db_path": str(db),
            "db_size_bytes": 11,
            "wal_size_bytes": 7,
            "disk_free_bytes": 99,
        }
    )
    blob_info = Mock(return_value=0)
    fts_info = Mock(return_value={"messages_ready": True, "action_events_ready": True})
    freshness_info = Mock(return_value={"sessions_with_profiles": 3, "total_sessions": 4})

    with (
        patch("polylogue.daemon.status.db_path", return_value=db),
        patch("polylogue.daemon.status._check_daemon_liveness", return_value=True),
        patch("polylogue.daemon.status._db_size_info", db_info),
        patch("polylogue.daemon.status._blob_size_info", blob_info),
        patch("polylogue.daemon.status._fts_readiness_info", fts_info),
        patch("polylogue.daemon.status._insight_freshness_info", freshness_info),
    ):
        payload = daemon_status_payload(sources=())

    assert payload["db_path"] == str(db)
    assert payload["db_size_bytes"] == 11
    assert payload["wal_size_bytes"] == 7
    assert payload["blob_dir_size_bytes"] == 0
    assert payload["disk_free_bytes"] == 99
    fts_readiness = payload["fts_readiness"]
    assert isinstance(fts_readiness, dict)
    assert fts_readiness["messages_ready"] is True
    assert fts_readiness["action_events_ready"] is True
    assert db_info.call_count == 1
    assert blob_info.call_count == 1
    assert fts_info.call_count == 1
    # Bounded probe results are reused across the status payload assembly,
    # but freshness probing is invoked a second time for the deep insight
    # readiness band. Both calls reuse the cached probe results.
    assert freshness_info.call_count >= 1


def test_daemon_status_fts_readiness_uses_lightweight_table_probe(tmp_path: Path) -> None:
    db = tmp_path / "polylogue.db"
    with sqlite3.connect(db) as conn:
        conn.executescript(
            """
            CREATE TABLE messages_fts (text TEXT);
            CREATE TABLE action_events_fts (text TEXT);
            """
        )

    with patch("polylogue.daemon.status.db_path", return_value=db):
        readiness = status_module._fts_readiness_info()

    assert readiness["messages_ready"] is False
    assert readiness["action_events_ready"] is False
    assert readiness["invariant_ready"] is False


def test_daemon_status_fts_readiness_uses_bounded_structural_probes(tmp_path: Path) -> None:
    db = tmp_path / "polylogue.db"
    queries: list[str] = []
    with sqlite3.connect(db) as conn:
        conn.executescript(
            """
            CREATE TABLE messages (text TEXT);
            CREATE TABLE conversation_stats (conversation_id TEXT PRIMARY KEY, message_count INTEGER NOT NULL);
            CREATE TABLE messages_fts (text TEXT);
            CREATE TABLE messages_fts_docsize (id INTEGER PRIMARY KEY, sz BLOB);
            CREATE TRIGGER messages_fts_ai AFTER INSERT ON messages BEGIN SELECT 1; END;
            CREATE TRIGGER messages_fts_ad AFTER DELETE ON messages BEGIN SELECT 1; END;
            CREATE TRIGGER messages_fts_au AFTER UPDATE ON messages BEGIN SELECT 1; END;
            CREATE TABLE action_events (event_id TEXT);
            CREATE TABLE action_events_fts (text TEXT);
            CREATE TABLE action_events_fts_docsize (id INTEGER PRIMARY KEY, sz BLOB);
            CREATE TRIGGER action_events_fts_ai AFTER INSERT ON action_events BEGIN SELECT 1; END;
            CREATE TRIGGER action_events_fts_ad AFTER DELETE ON action_events BEGIN SELECT 1; END;
            CREATE TRIGGER action_events_fts_au AFTER UPDATE ON action_events BEGIN SELECT 1; END;
            INSERT INTO conversation_stats VALUES ('c1', 2), ('c2', 3);
            INSERT INTO messages(rowid, text) VALUES (1, 'a'), (2, 'b'), (3, 'c'), (4, 'd'), (5, 'e');
            INSERT INTO messages_fts_docsize VALUES (1, x''), (2, x''), (3, x''), (4, x''), (5, x'');
            """
        )

    original_connect = sqlite3.connect

    def traced_connect(*args: Any, **kwargs: Any) -> sqlite3.Connection:
        conn = original_connect(*args, **kwargs)
        conn.set_trace_callback(queries.append)
        return cast(sqlite3.Connection, conn)

    with (
        patch("polylogue.daemon.status.db_path", return_value=db),
        patch("polylogue.storage.sqlite.connection_profile.sqlite3.connect", side_effect=traced_connect),
    ):
        readiness = status_module._fts_readiness_info()

    assert readiness["messages_ready"] is True
    assert readiness["action_events_ready"] is True
    assert readiness["invariant_ready"] is True
    assert readiness["coverage_exact"] is False
    assert all("COUNT(*) FROM messages" not in query for query in queries)
    assert all("COUNT(*) FROM messages_fts_docsize" not in query for query in queries)
    assert all("LEFT JOIN messages_fts_docsize" not in query for query in queries)


def test_fts_readiness_exact_detects_missing_docsize_row(tmp_path: Path) -> None:
    from polylogue.daemon.fts_status import fts_readiness_info
    from polylogue.storage.sqlite.connection import open_connection

    db_path = tmp_path / "polylogue.db"
    with open_connection(db_path) as conn:
        conn.execute(
            "INSERT INTO conversations(conversation_id, provider_name, provider_conversation_id, version) VALUES(?,?,?,1)",
            ("conv-stale-fts", "codex", "provider-conv"),
        )
        conn.execute(
            "INSERT INTO messages(message_id, conversation_id, role, text, provider_name, version) VALUES(?,?,?,?,?,1)",
            ("msg-stale-fts", "conv-stale-fts", "user", "needle stale index", "codex"),
        )
        conn.commit()
        rowid = conn.execute("SELECT rowid FROM messages WHERE message_id = ?", ("msg-stale-fts",)).fetchone()[0]
        conn.execute("DELETE FROM messages_fts WHERE rowid = ?", (rowid,))
        conn.commit()

    structural = fts_readiness_info(db_path)
    exact = fts_readiness_info(db_path, exact=True)

    assert structural["messages_ready"] is True
    assert exact["messages_ready"] is False
    surfaces = exact["surfaces"]
    assert isinstance(surfaces, dict)
    messages = surfaces["messages_fts"]
    assert isinstance(messages, dict)
    assert messages["missing_rows"] == 1
    assert messages["ready"] is False


def test_fts_readiness_requires_recorded_freshness_when_available(tmp_path: Path) -> None:
    from polylogue.daemon.fts_status import fts_readiness_info

    db_path = tmp_path / "polylogue.db"
    with sqlite3.connect(db_path) as conn:
        conn.executescript(
            """
            CREATE TABLE messages (message_id TEXT, text TEXT);
            CREATE TABLE messages_fts (text TEXT);
            CREATE TRIGGER messages_fts_ai AFTER INSERT ON messages BEGIN SELECT 1; END;
            CREATE TRIGGER messages_fts_ad AFTER DELETE ON messages BEGIN SELECT 1; END;
            CREATE TRIGGER messages_fts_au AFTER UPDATE ON messages BEGIN SELECT 1; END;
            CREATE TABLE fts_freshness_state (
                surface TEXT PRIMARY KEY,
                state TEXT NOT NULL,
                checked_at TEXT NOT NULL
            );
            INSERT INTO fts_freshness_state VALUES ('messages_fts', 'stale', '2026-05-24T00:00:00+00:00');
            """
        )

    readiness = fts_readiness_info(db_path)

    assert readiness["messages_ready"] is False
    assert readiness["invariant_ready"] is False
    surfaces = readiness["surfaces"]
    assert isinstance(surfaces, dict)
    messages = surfaces["messages_fts"]
    assert isinstance(messages, dict)
    assert messages["freshness_known"] is True
    assert messages["freshness_state"] == "stale"
    assert messages["ready"] is False


def test_fts_readiness_reports_recorded_freshness_counts_without_exact_scan(tmp_path: Path) -> None:
    from polylogue.daemon.fts_status import fts_readiness_info

    db_path = tmp_path / "polylogue.db"
    queries: list[str] = []
    with sqlite3.connect(db_path) as conn:
        conn.executescript(
            """
            CREATE TABLE messages (message_id TEXT, text TEXT);
            CREATE TABLE messages_fts (text TEXT);
            CREATE TABLE messages_fts_docsize (id INTEGER PRIMARY KEY, sz BLOB);
            CREATE TRIGGER messages_fts_ai AFTER INSERT ON messages BEGIN SELECT 1; END;
            CREATE TRIGGER messages_fts_ad AFTER DELETE ON messages BEGIN SELECT 1; END;
            CREATE TRIGGER messages_fts_au AFTER UPDATE ON messages BEGIN SELECT 1; END;
            CREATE TABLE action_events (event_id TEXT);
            CREATE TABLE action_events_fts (text TEXT);
            CREATE TABLE action_events_fts_docsize (id INTEGER PRIMARY KEY, sz BLOB);
            CREATE TRIGGER action_events_fts_ai AFTER INSERT ON action_events BEGIN SELECT 1; END;
            CREATE TRIGGER action_events_fts_ad AFTER DELETE ON action_events BEGIN SELECT 1; END;
            CREATE TRIGGER action_events_fts_au AFTER UPDATE ON action_events BEGIN SELECT 1; END;
            CREATE TABLE fts_freshness_state (
                surface TEXT PRIMARY KEY,
                state TEXT NOT NULL,
                checked_at TEXT NOT NULL,
                source_rows INTEGER NOT NULL DEFAULT 0,
                indexed_rows INTEGER NOT NULL DEFAULT 0,
                missing_rows INTEGER NOT NULL DEFAULT 0,
                excess_rows INTEGER NOT NULL DEFAULT 0,
                duplicate_rows INTEGER NOT NULL DEFAULT 0,
                detail TEXT
            );
            INSERT INTO fts_freshness_state
                (surface, state, checked_at, source_rows, indexed_rows, missing_rows, excess_rows, duplicate_rows, detail)
            VALUES
                ('messages_fts', 'ready', '2026-05-24T00:00:00+00:00', 200, 199, 1, 0, 0, 'repair pending'),
                ('action_events_fts', 'ready', '2026-05-24T00:00:00+00:00', 7, 7, 0, 0, 0, NULL);
            """
        )

    original_connect = sqlite3.connect

    def traced_connect(*args: Any, **kwargs: Any) -> sqlite3.Connection:
        conn = original_connect(*args, **kwargs)
        conn.set_trace_callback(queries.append)
        return cast(sqlite3.Connection, conn)

    with patch("polylogue.storage.sqlite.connection_profile.sqlite3.connect", side_effect=traced_connect):
        readiness = fts_readiness_info(db_path)

    assert readiness["message_indexable_count"] == 200
    assert readiness["message_indexed_count"] == 199
    assert readiness["action_event_count"] == 7
    assert readiness["action_event_indexed_count"] == 7
    assert readiness["coverage_pct"] == 99.5
    assert readiness["messages_ready"] is False
    assert readiness["invariant_ready"] is False
    surfaces = readiness["surfaces"]
    assert isinstance(surfaces, dict)
    messages = surfaces["messages_fts"]
    assert isinstance(messages, dict)
    assert messages["missing_rows"] == 1
    assert messages["freshness_state"] == "stale"
    assert messages["freshness_recorded_state"] == "ready"
    assert messages["freshness_trusted"] is False
    assert messages["freshness_detail"] == "repair pending"
    assert all("COUNT(*) FROM messages" not in query for query in queries)
    assert all("messages_fts_docsize" not in query for query in queries)


def test_fts_readiness_rejects_zero_count_ready_freshness_when_source_has_rows(tmp_path: Path) -> None:
    from polylogue.daemon.fts_status import fts_readiness_info

    db_path = tmp_path / "polylogue.db"
    queries: list[str] = []
    with sqlite3.connect(db_path) as conn:
        conn.executescript(
            """
            CREATE TABLE messages (message_id TEXT, text TEXT);
            CREATE TABLE messages_fts (text TEXT);
            CREATE TRIGGER messages_fts_ai AFTER INSERT ON messages BEGIN SELECT 1; END;
            CREATE TRIGGER messages_fts_ad AFTER DELETE ON messages BEGIN SELECT 1; END;
            CREATE TRIGGER messages_fts_au AFTER UPDATE ON messages BEGIN SELECT 1; END;
            INSERT INTO messages VALUES ('m1', 'needs indexing');
            CREATE TABLE fts_freshness_state (
                surface TEXT PRIMARY KEY,
                state TEXT NOT NULL,
                checked_at TEXT NOT NULL,
                source_rows INTEGER NOT NULL DEFAULT 0,
                indexed_rows INTEGER NOT NULL DEFAULT 0,
                missing_rows INTEGER NOT NULL DEFAULT 0,
                excess_rows INTEGER NOT NULL DEFAULT 0,
                duplicate_rows INTEGER NOT NULL DEFAULT 0,
                detail TEXT
            );
            INSERT INTO fts_freshness_state
                (surface, state, checked_at, source_rows, indexed_rows, missing_rows, excess_rows, duplicate_rows)
            VALUES ('messages_fts', 'ready', '2026-05-24T00:00:00+00:00', 0, 0, 0, 0, 0);
            """
        )

    original_connect = sqlite3.connect

    def traced_connect(*args: Any, **kwargs: Any) -> sqlite3.Connection:
        conn = original_connect(*args, **kwargs)
        conn.set_trace_callback(queries.append)
        return cast(sqlite3.Connection, conn)

    with patch("polylogue.storage.sqlite.connection_profile.sqlite3.connect", side_effect=traced_connect):
        readiness = fts_readiness_info(db_path)

    assert readiness["messages_ready"] is False
    assert readiness["invariant_ready"] is False
    assert readiness["coverage_pct"] == 0.0
    surfaces = readiness["surfaces"]
    assert isinstance(surfaces, dict)
    messages = surfaces["messages_fts"]
    assert isinstance(messages, dict)
    assert messages["freshness_state"] == "unknown"
    assert messages["freshness_recorded_state"] == "ready"
    assert messages["freshness_trusted"] is False
    assert all("count(*) from messages_fts" not in query.lower() for query in queries)


def test_fts_readiness_tolerates_malformed_recorded_counts(tmp_path: Path) -> None:
    from polylogue.daemon.fts_status import fts_readiness_info

    db_path = tmp_path / "polylogue.db"
    with sqlite3.connect(db_path) as conn:
        conn.executescript(
            """
            CREATE TABLE messages (message_id TEXT, text TEXT);
            CREATE TABLE messages_fts (text TEXT);
            CREATE TRIGGER messages_fts_ai AFTER INSERT ON messages BEGIN SELECT 1; END;
            CREATE TRIGGER messages_fts_ad AFTER DELETE ON messages BEGIN SELECT 1; END;
            CREATE TRIGGER messages_fts_au AFTER UPDATE ON messages BEGIN SELECT 1; END;
            CREATE TABLE fts_freshness_state (
                surface TEXT PRIMARY KEY,
                state TEXT NOT NULL,
                checked_at TEXT NOT NULL,
                source_rows TEXT,
                indexed_rows TEXT,
                missing_rows TEXT,
                excess_rows TEXT,
                duplicate_rows TEXT
            );
            INSERT INTO fts_freshness_state
                (surface, state, checked_at, source_rows, indexed_rows, missing_rows, excess_rows, duplicate_rows)
            VALUES
                ('messages_fts', 'ready', '2026-05-24T00:00:00+00:00', 'not-int', NULL, 'bad', '0', 'bad');
            """
        )

    readiness = fts_readiness_info(db_path)
    surfaces = readiness["surfaces"]
    assert isinstance(surfaces, dict)
    messages = surfaces["messages_fts"]
    assert isinstance(messages, dict)
    assert messages["source_rows"] == 0
    assert messages["missing_rows"] == 0
    assert readiness["message_indexable_count"] == 0


def test_daemon_status_insight_freshness_uses_lightweight_counts(tmp_path: Path) -> None:
    db = tmp_path / "polylogue.db"
    with sqlite3.connect(db) as conn:
        conn.executescript(
            """
            CREATE TABLE conversations (conversation_id TEXT PRIMARY KEY);
            CREATE TABLE session_profiles (conversation_id TEXT PRIMARY KEY);
            INSERT INTO conversations (conversation_id) VALUES ('a'), ('b');
            INSERT INTO session_profiles (conversation_id) VALUES ('a');
            """
        )

    with patch("polylogue.daemon.status.db_path", return_value=db):
        freshness = status_module._insight_freshness_info()

    assert freshness == {"sessions_with_profiles": 1, "total_sessions": 2}


@pytest.mark.frozen_clock_modules("polylogue.daemon.status")
def test_daemon_status_flags_stale_live_ingest_attempts(tmp_path: Path, frozen_clock: FrozenClock) -> None:
    db = tmp_path / "polylogue.db"
    source = tmp_path / "session.jsonl"
    source.write_text('{"a":1}\n')
    cursor = CursorStore(db)
    attempt_id = cursor.begin_ingest_attempt(
        paths=[source],
        input_bytes=source.stat().st_size,
        queued_file_count=1,
    )
    old_updated_at = (frozen_clock.now() - timedelta(minutes=10)).isoformat()
    with sqlite3.connect(db) as conn:
        conn.execute(
            "UPDATE live_ingest_attempt SET updated_at = ? WHERE attempt_id = ?",
            (old_updated_at, attempt_id),
        )
        conn.commit()

    with (
        patch("polylogue.daemon.status.db_path", return_value=db),
        patch("polylogue.daemon.status._check_daemon_liveness", return_value=False),
        patch("polylogue.daemon.status._blob_size_info", return_value=0),
        patch("polylogue.daemon.status._fts_readiness_info", return_value={}),
        patch("polylogue.daemon.status._insight_freshness_info", return_value={}),
    ):
        payload = daemon_status_payload(sources=())

    attempts = payload["live_ingest_attempts"]
    assert isinstance(attempts, dict)
    assert attempts["running_count"] == 1
    assert attempts["stale_running_count"] == 1
    recent = attempts["recent"]
    assert isinstance(recent, list)
    latest = recent[0]
    assert isinstance(latest, dict)
    assert latest["stale"] is True
    # #1246: ``stale`` (legacy) and the typed ``progress_classification``
    # together encode the same condition for an attempt that has not made
    # progress for at least ``STUCK_AFTER_S``.
    assert latest["progress_classification"] == "stuck"
    updated_age_s = latest["updated_age_s"]
    assert isinstance(updated_age_s, int | float)
    assert updated_age_s >= 600
    assert attempts["stuck_running_count"] == 1
    assert attempts["slow_running_count"] == 0
    lines = format_daemon_status_lines(payload)
    assert "Live ingest attempts: 1 running, 1 stuck" in lines
    assert "  latest: running stuck planning 0/1 files" in lines


@pytest.mark.frozen_clock_modules("polylogue.daemon.status")
def test_daemon_status_flags_slow_but_progressing_live_ingest_attempt(
    tmp_path: Path, frozen_clock: FrozenClock
) -> None:
    """Running attempts that exceed p95 historical duration but still
    report fresh progress are reported as ``slow``, not ``stuck`` (#1246)."""

    db = tmp_path / "polylogue.db"
    source = tmp_path / "session.jsonl"
    source.write_text('{"a":1}\n')
    cursor = CursorStore(db)
    # Seed completed attempts with short, uniform durations so the p95
    # baseline is well below the running attempt's elapsed time.
    base = frozen_clock.now() - timedelta(hours=1)
    with sqlite3.connect(db) as conn:
        for i in range(8):
            start = (base + timedelta(minutes=i)).isoformat()
            end = (base + timedelta(minutes=i, seconds=2)).isoformat()
            conn.execute(
                """
                INSERT INTO live_ingest_attempt (
                    attempt_id, started_at, updated_at, completed_at,
                    status, phase, input_bytes
                ) VALUES (?, ?, ?, ?, 'completed', 'convergence', 0)
                """,
                (f"hist-{i}", start, end, end),
            )
        conn.commit()

    attempt_id = cursor.begin_ingest_attempt(
        paths=[source],
        input_bytes=source.stat().st_size,
        queued_file_count=1,
    )
    # Make the running attempt look like it has been ticking for 90s —
    # well above the 2-second historical p95, but well under the 180s
    # stuck threshold. Set updated_at to "now" so it is not stale.
    started_at = (frozen_clock.now() - timedelta(seconds=90)).isoformat()
    updated_at = frozen_clock.now().isoformat()
    with sqlite3.connect(db) as conn:
        conn.execute(
            "UPDATE live_ingest_attempt SET started_at = ?, updated_at = ? WHERE attempt_id = ?",
            (started_at, updated_at, attempt_id),
        )
        conn.commit()

    with (
        patch("polylogue.daemon.status.db_path", return_value=db),
        patch("polylogue.daemon.status._check_daemon_liveness", return_value=False),
        patch("polylogue.daemon.status._blob_size_info", return_value=0),
        patch("polylogue.daemon.status._fts_readiness_info", return_value={}),
        patch("polylogue.daemon.status._insight_freshness_info", return_value={}),
    ):
        payload = daemon_status_payload(sources=())

    attempts = payload["live_ingest_attempts"]
    assert isinstance(attempts, dict)
    assert attempts["running_count"] == 1
    assert attempts["slow_running_count"] == 1
    assert attempts["stuck_running_count"] == 0
    assert attempts["stale_running_count"] == 0
    threshold = attempts["slow_threshold_s"]
    assert isinstance(threshold, int | float)
    assert threshold < 90.0
    recent = attempts["recent"]
    assert isinstance(recent, list)
    latest = recent[0]
    assert isinstance(latest, dict)
    assert latest["progress_classification"] == "slow"
    assert latest["stale"] is False
    lines = format_daemon_status_lines(payload)
    assert "Live ingest attempts: 1 running, 1 slow" in lines
    assert any(line.startswith("  latest: running slow ") for line in lines)


def test_daemon_status_summarizes_retry_due_and_excluded_live_cursor_files(tmp_path: Path) -> None:
    db = tmp_path / "polylogue.db"
    failed = tmp_path / "failed.jsonl"
    failed.write_text('{"bad":true}\n')
    excluded = tmp_path / "excluded.jsonl"
    excluded.write_text('{"skip":true}\n')
    cursor = CursorStore(db)
    cursor.mark_failed(failed)
    cursor.set(excluded, excluded.stat().st_size)
    cursor.mark_excluded(excluded)

    with sqlite3.connect(db) as conn:
        conn.execute(
            "UPDATE live_cursor SET next_retry_at = ? WHERE source_path = ?",
            ("2000-01-01T00:00:00+00:00", str(failed)),
        )
        conn.commit()

    with (
        patch("polylogue.daemon.status.db_path", return_value=db),
        patch("polylogue.daemon.status._check_daemon_liveness", return_value=False),
        patch("polylogue.daemon.status._blob_size_info", return_value=0),
        patch("polylogue.daemon.status._fts_readiness_info", return_value={}),
        patch("polylogue.daemon.status._insight_freshness_info", return_value={}),
    ):
        status = build_daemon_status(sources=())

    assert status.live_cursor.tracked_file_count == 2
    assert status.live_cursor.failed_file_count == 1
    assert status.live_cursor.excluded_file_count == 1
    assert status.live_cursor.retry_due_file_count == 1
    assert status.live_cursor.in_backoff_file_count == 0
    assert [item.source_path for item in status.live_cursor.failing_files] == [str(excluded), str(failed)]
