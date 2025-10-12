import json
from pathlib import Path

from polylogue.doctor import run_doctor


def test_doctor_detects_invalid_codex(tmp_path):
    codex_dir = tmp_path / "codex"
    codex_dir.mkdir()
    bad_session = codex_dir / "bad.jsonl"
    bad_session.write_text("{not json}\n", encoding="utf-8")

    report = run_doctor(codex_dir=codex_dir, claude_code_dir=tmp_path / "claude", limit=None)
    assert any(issue.provider == "codex" for issue in report.issues)
    assert report.checked.get("codex") == 1
