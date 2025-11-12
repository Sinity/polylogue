from __future__ import annotations

from polylogue import util
from tests.conftest import _configure_state


def test_load_runs_missing(tmp_path, monkeypatch):
    _configure_state(monkeypatch, tmp_path)
    assert util.load_runs() == []


def test_load_runs_limit(tmp_path, monkeypatch):
    _configure_state(monkeypatch, tmp_path)
    for cmd in ["a", "b", "c"]:
        util.add_run({"cmd": cmd})
    loaded = util.load_runs(limit=2)
    assert [entry["cmd"] for entry in loaded] == ["b", "c"]


def test_add_run_persists_metadata(tmp_path, monkeypatch):
    _configure_state(monkeypatch, tmp_path)
    util.add_run({"cmd": "render", "driveRetries": 2, "driveFailures": 1, "out": tmp_path / "out"})
    payload = util.load_runs(limit=1)[0]
    assert payload["cmd"] == "render"
    assert payload["driveRetries"] == 2
    assert payload["driveFailures"] == 1


def test_latest_run_filters_provider(tmp_path, monkeypatch):
    _configure_state(monkeypatch, tmp_path)
    util.add_run({"cmd": "sync drive", "provider": "drive", "count": 3})
    util.add_run({"cmd": "sync codex", "provider": "codex", "count": 1})
    latest_drive = util.latest_run(provider="drive")
    assert latest_drive["count"] == 3
    assert util.latest_run(provider="claude") is None


def test_format_run_brief_outputs_summary(tmp_path, monkeypatch):
    _configure_state(monkeypatch, tmp_path)
    util.add_run({"cmd": "sync drive", "provider": "drive", "count": 2, "skipped": 1, "duration": 1.5})
    entry = util.latest_run(provider="drive")
    summary = util.format_run_brief(entry)
    assert summary is not None
    assert "2 item" in summary
    assert "skipped" in summary
