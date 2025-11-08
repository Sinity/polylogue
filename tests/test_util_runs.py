from __future__ import annotations

import json

from polylogue import util


def test_load_runs_missing(tmp_path, monkeypatch):
    runs_path = tmp_path / "runs.json"
    monkeypatch.setattr(util, "RUNS_PATH", runs_path, raising=False)
    assert util.load_runs() == []


def test_load_runs_limit(tmp_path, monkeypatch):
    runs_path = tmp_path / "runs.json"
    runs = [{"cmd": "a"}, {"cmd": "b"}, {"cmd": "c"}]
    runs_path.write_text(json.dumps(runs), encoding="utf-8")
    monkeypatch.setattr(util, "RUNS_PATH", runs_path, raising=False)
    assert util.load_runs(limit=2) == runs[-2:]
