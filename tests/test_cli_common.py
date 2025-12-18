from __future__ import annotations

import sys
from types import SimpleNamespace

from polylogue.cli_common import filter_chats
from polylogue.cli_common import choose_single_entry


def test_filter_chats_warns_on_missing_modified_time(capsys):
    chats = [
        {"id": "a", "name": "Alpha", "modifiedTime": "2024-01-01T00:00:00Z"},
        {"id": "b", "name": "Bravo"},  # missing modifiedTime
    ]

    filtered = filter_chats(chats, None, "2024-01-01", None)

    assert len(filtered) == 1
    assert filtered[0]["id"] == "a"

    err = capsys.readouterr().err
    assert "missing modifiedTime" in err


class DummyConsole:
    def __init__(self):
        self.lines: list[str] = []

    def print(self, *args, **kwargs):  # noqa: ANN001, ARG002
        self.lines.append(" ".join(str(a) for a in args))


class DummyUI:
    plain = True

    def __init__(self):
        self.console = DummyConsole()


def test_choose_single_entry_non_tty_skips(monkeypatch):
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    ui = DummyUI()

    selection, cancelled = choose_single_entry(ui, ["a", "b"], format_line=lambda v, _: v)

    assert selection is None
    assert cancelled is True
    assert any("cannot prompt" in line for line in ui.console.lines)


def test_choose_single_entry_interactive_uses_sk(monkeypatch):
    import subprocess

    monkeypatch.setattr("sys.stdin.isatty", lambda: True)

    calls = []

    def fake_run(cmd, **kwargs):
        calls.append((cmd, kwargs))
        return subprocess.CompletedProcess(cmd, 0, stdout=b"0\tAlpha\n", stderr=b"")

    monkeypatch.setattr("polylogue.cli_common.subprocess.run", fake_run)

    ui = SimpleNamespace(plain=False)
    selection, cancelled = choose_single_entry(ui, ["alpha", "bravo"], format_line=lambda v, _: v, prompt="pick")

    assert cancelled is False
    assert selection == "alpha"
    assert calls
    assert calls[0][0][:2] == ["sk", "--ansi"]


def test_choose_single_entry_interactive_cancelled(monkeypatch):
    import subprocess

    monkeypatch.setattr("sys.stdin.isatty", lambda: True)

    def fake_run(cmd, **kwargs):
        raise subprocess.CalledProcessError(130, cmd, output=b"", stderr=b"cancelled")

    monkeypatch.setattr("polylogue.cli_common.subprocess.run", fake_run)

    ui = SimpleNamespace(plain=False)
    selection, cancelled = choose_single_entry(ui, ["alpha"], format_line=lambda v, _: v)
    assert selection is None
    assert cancelled is True
