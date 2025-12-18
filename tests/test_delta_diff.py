from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from polylogue.util import write_delta_diff


def test_write_delta_diff_uses_delta_output(monkeypatch, tmp_path: Path):
    old_path = tmp_path / "old.txt"
    new_path = tmp_path / "new.txt"
    old_path.write_text("a\n", encoding="utf-8")
    new_path.write_text("b\n", encoding="utf-8")

    def fake_run(cmd, **_kwargs):  # noqa: ANN001
        assert cmd[:1] == ["delta"]
        return SimpleNamespace(returncode=0, stdout="delta\n", stderr="")

    monkeypatch.setattr("polylogue.util.subprocess.run", fake_run)

    diff_path = write_delta_diff(old_path, new_path)
    assert diff_path is not None
    assert diff_path.read_text(encoding="utf-8").strip() == "delta"


def test_write_delta_diff_falls_back_to_unified_diff(monkeypatch, tmp_path: Path):
    old_path = tmp_path / "old.txt"
    new_path = tmp_path / "new.txt"
    old_path.write_text("hello\n", encoding="utf-8")
    new_path.write_text("hello world\n", encoding="utf-8")

    def fake_run(_cmd, **_kwargs):  # noqa: ANN001
        return SimpleNamespace(returncode=1, stdout="", stderr="boom")

    monkeypatch.setattr("polylogue.util.subprocess.run", fake_run)

    diff_path = write_delta_diff(old_path, new_path)
    assert diff_path is not None
    body = diff_path.read_text(encoding="utf-8")
    assert "---" in body
    assert "+hello world" in body

