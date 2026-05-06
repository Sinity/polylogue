from __future__ import annotations

import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import patch

from polylogue.daemon.status import build_daemon_status, daemon_status_payload, format_daemon_status_lines
from polylogue.sources.live.cursor import CursorStore


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
    lines = format_daemon_status_lines(payload)
    assert "Live ingest attempts: 1 running" in lines
    assert "  latest: running full_parse 0/1 files" in lines
    assert "  memory: cgroup 2048.0 MiB peak 4096.0 MiB" in lines


def test_daemon_status_flags_stale_live_ingest_attempts(tmp_path: Path) -> None:
    db = tmp_path / "polylogue.db"
    source = tmp_path / "session.jsonl"
    source.write_text('{"a":1}\n')
    cursor = CursorStore(db)
    attempt_id = cursor.begin_ingest_attempt(
        paths=[source],
        input_bytes=source.stat().st_size,
        queued_file_count=1,
    )
    old_updated_at = (datetime.now(UTC) - timedelta(minutes=10)).isoformat()
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
    updated_age_s = latest["updated_age_s"]
    assert isinstance(updated_age_s, int | float)
    assert updated_age_s >= 600
    lines = format_daemon_status_lines(payload)
    assert "Live ingest attempts: 1 running, 1 stale" in lines
    assert "  latest: running stale planning 0/1 files" in lines


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
