from __future__ import annotations

import json

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


def test_add_run_emits_structured_log(tmp_path, monkeypatch, capsys):
    _configure_state(monkeypatch, tmp_path)
    monkeypatch.setenv("POLYLOGUE_RUN_LOG", "1")
    util.add_run({"cmd": "render", "driveRetries": 2, "driveFailures": 1, "out": tmp_path / "out"})
    err = capsys.readouterr().err.strip()
    assert err
    payload = json.loads(err.splitlines()[-1])
    assert payload["event"] == "polylogue_run"
    assert payload["cmd"] == "render"
    assert payload["retries"] == 2
    assert payload["failures"] == 1
