from __future__ import annotations

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
    lines = format_daemon_status_lines(payload)
    assert "Failing files: 1" in lines
    assert f"  {failed}" in lines
