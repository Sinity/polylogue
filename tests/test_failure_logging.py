import json
from argparse import Namespace
from pathlib import Path

from polylogue.cli.app import _record_failure
from polylogue.paths import STATE_HOME
from polylogue.ui import create_ui
from polylogue.commands import CommandEnv


def test_record_failure_stamps_schema(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("polylogue.cli.app.STATE_HOME", tmp_path)
    args = Namespace(cmd="render")

    _record_failure(args, RuntimeError("boom"))

    log_path = tmp_path / "failures.jsonl"
    assert log_path.exists()
    line = log_path.read_text(encoding="utf-8").strip().splitlines()[-1]
    payload = json.loads(line)
    assert "schemaVersion" in payload
    assert "polylogueVersion" in payload
    assert payload.get("cmd") == "render"
