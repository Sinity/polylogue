from __future__ import annotations

from polylogue.cli import editor


def test_open_in_editor_parses_args(tmp_path, monkeypatch):
    target = tmp_path / "file.txt"
    target.write_text("demo", encoding="utf-8")

    captured = {}

    def fake_run(cmd):
        captured["cmd"] = cmd
        return True

    monkeypatch.setenv("EDITOR", "code --wait")
    monkeypatch.setattr(editor, "_run_editor", fake_run)

    assert editor.open_in_editor(target, line=12)
    assert captured["cmd"] == ["code", "--wait", f"{target}:12"]
