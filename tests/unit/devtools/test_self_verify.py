from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import pytest

from devtools import self_verify
from tests.infra.storage_records import SessionBuilder


def _seed_archive(db_path: Path) -> None:
    (
        SessionBuilder(db_path, "chatgpt:alpha")
        .provider("chatgpt")
        .title("Alpha")
        .created_at("2026-01-01T00:00:00+00:00")
        .updated_at("2026-01-01T00:00:00+00:00")
        .add_message(message_id="alpha-m1", role="user", text="analysis prompt")
        .add_message(message_id="alpha-m2", role="assistant", text="analysis answer")
        .save()
    )
    (
        SessionBuilder(db_path, "claude-code:beta")
        .provider("claude-code")
        .title("Beta")
        .created_at("2026-01-02T00:00:00+00:00")
        .updated_at("2026-01-02T00:00:00+00:00")
        .add_message(message_id="beta-m1", role="user", text="error report", has_tool_use=1)
        .add_message(message_id="beta-m2", role="assistant", text="fix summary")
        .save()
    )


def test_capture_snapshot_records_core_read_envelopes(tmp_path: Path) -> None:
    db_path = tmp_path / "index.db"
    _seed_archive(db_path)

    payload = self_verify.capture_snapshot_sync_for_test(
        db_path, limit=2, message_limit=1, search_queries=("analysis",)
    )

    assert payload["report_version"] == self_verify.REPORT_VERSION
    snapshot = cast(dict[str, Any], payload["snapshot"])
    assert isinstance(snapshot, dict)
    assert set(snapshot) == {"archive_stats", "lists", "messages", "searches", "stats_by", "topology"}
    archive_stats = cast(dict[str, Any], snapshot["archive_stats"])
    stats_by = cast(dict[str, Any], snapshot["stats_by"])
    lists = cast(list[dict[str, Any]], snapshot["lists"])
    searches = cast(list[dict[str, Any]], snapshot["searches"])
    messages = cast(list[dict[str, Any]], snapshot["messages"])
    assert archive_stats["session_count"] == 2
    assert stats_by["provider"] == {"chatgpt-export": 1, "claude-code-session": 1}
    assert lists[0]["items"]
    first_search_envelope = cast(dict[str, Any], searches[0]["envelope"])
    assert first_search_envelope["query"] == "analysis"
    assert "hits" in first_search_envelope
    assert messages[0]["messages"]


def test_compare_snapshots_ignores_capture_metadata() -> None:
    baseline = {"captured_at": "before", "snapshot": {"archive_stats": {"session_count": 1}}}
    candidate = {"captured_at": "after", "snapshot": {"archive_stats": {"session_count": 1}}}

    result = self_verify.compare_snapshots(baseline, candidate)

    assert result["ok"] is True
    assert result["differing_sections"] == []


def test_compare_snapshots_reports_differing_sections() -> None:
    baseline = {"snapshot": {"archive_stats": {"session_count": 1}, "lists": []}}
    candidate = {"snapshot": {"archive_stats": {"session_count": 2}, "lists": []}}

    result = self_verify.compare_snapshots(baseline, candidate)

    assert result["ok"] is False
    assert result["differing_sections"] == ["archive_stats"]
    assert result["unexpected_differing_sections"] == ["archive_stats"]


def test_compare_snapshots_can_allow_intended_section_differences() -> None:
    baseline = {"snapshot": {"archive_stats": {"session_count": 1}, "lists": []}}
    candidate = {"snapshot": {"archive_stats": {"session_count": 2}, "lists": []}}

    result = self_verify.compare_snapshots(baseline, candidate, allowed_sections=("archive_stats",))

    assert result["ok"] is True
    assert result["differing_sections"] == ["archive_stats"]
    assert result["allowed_differing_sections"] == ["archive_stats"]
    assert result["unexpected_differing_sections"] == []


def test_self_verify_capture_and_compare_cli(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    db_path = tmp_path / "index.db"
    out_path = tmp_path / "baseline.json"
    _seed_archive(db_path)

    assert (
        self_verify.main(
            [
                "capture",
                "--db",
                str(db_path),
                "--out",
                str(out_path),
                "--limit",
                "2",
                "--message-limit",
                "1",
                "--search-query",
                "analysis",
            ]
        )
        == 0
    )
    assert json.loads(out_path.read_text(encoding="utf-8"))["snapshot"]["archive_stats"]["session_count"] == 2
    assert self_verify.main(["compare", str(out_path), str(out_path), "--json"]) == 0
    captured = capsys.readouterr()
    assert '"ok": true' in captured.out
