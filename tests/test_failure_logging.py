import json
from pathlib import Path
from types import SimpleNamespace

from polylogue.cli.failure_logging import record_failure


def test_record_failure_stamps_schema(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("polylogue.cli.failure_logging.STATE_HOME", tmp_path)
    args = SimpleNamespace(cmd="render")

    record_failure(args, RuntimeError("boom"))

    log_path = tmp_path / "failures.jsonl"
    assert log_path.exists()
    line = log_path.read_text(encoding="utf-8").strip().splitlines()[-1]
    payload = json.loads(line)
    assert "schemaVersion" in payload
    assert "polylogueVersion" in payload
    assert payload.get("cmd") == "render"
