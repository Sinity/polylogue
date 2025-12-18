from __future__ import annotations

from io import StringIO
from types import SimpleNamespace

import pytest

from polylogue import ui as ui_module
from polylogue.ui import UI


def test_ui_summary_handles_brackets_without_markup(monkeypatch):
    monkeypatch.setattr(ui_module.shutil, "which", lambda _cmd: "/usr/bin/fake")

    def fake_run(cmd, **_kwargs):
        if cmd[:2] == ["gum", "format"]:
            return SimpleNamespace(stdout="## Doctor\nPaths: [/tmp/example.json, /tmp/fallback.json]", stderr="", returncode=0)
        return SimpleNamespace(stdout="", stderr="", returncode=0)

    monkeypatch.setattr(ui_module.subprocess, "run", fake_run)

    capture = StringIO()
    console = ui_module.Console(file=capture, force_terminal=False, legacy_windows=False)
    view = UI(plain=False)
    view.console = console

    lines = ["Copy config to [/tmp/example.json, /tmp/fallback.json]"]
    view.summary("Doctor", lines)

    output = capture.getvalue()
    assert "Paths: [/tmp/example.json, /tmp/fallback.json]" in output


def test_ui_confirm_uses_default_flag(monkeypatch):
    monkeypatch.setattr(ui_module.shutil, "which", lambda _cmd: "/usr/bin/fake")

    calls = []

    def fake_run(cmd, **_kwargs):
        calls.append(cmd)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(ui_module.subprocess, "run", fake_run)

    view = UI(plain=False)
    assert view.confirm("Proceed?", default=True) is True
    assert calls[0] == ["gum", "confirm", "--prompt", "Proceed?", "--default"]

    calls.clear()
    view.confirm("Proceed?", default=False)
    assert calls[0] == ["gum", "confirm", "--prompt", "Proceed?"]


def test_ui_confirm_plain_mode(monkeypatch):
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    responses = iter(["y", ""])
    monkeypatch.setattr("builtins.input", lambda _: next(responses))
    view = UI(plain=True)
    assert view.confirm("Proceed?", default=False) is True
    assert view.confirm("Proceed?", default=True) is True


def test_ui_choose_plain_mode(monkeypatch):
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    responses = iter(["3"])
    monkeypatch.setattr("builtins.input", lambda _: next(responses))
    view = UI(plain=True)
    choice = view.choose("Select option", ["one", "two", "three"])
    assert choice == "three"


def test_ui_choose_plain_mode_default(monkeypatch):
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    monkeypatch.setattr("builtins.input", lambda _: "")
    view = UI(plain=True)
    choice = view.choose("Select option", ["alpha", "beta"])
    assert choice is None


def test_ui_plain_non_tty_skips_prompts(monkeypatch):
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    monkeypatch.setattr("builtins.input", lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("input should not be called")))

    view = UI(plain=True)

    with pytest.raises(SystemExit):
        view.confirm("Proceed?", default=False)
    with pytest.raises(SystemExit):
        view.choose("Select", ["a", "b"])
    with pytest.raises(SystemExit):
        view.input("Enter", default="x")


def test_ui_missing_dependencies_raise(monkeypatch):
    calls = []

    def fake_which(cmd):
        calls.append(cmd)
        return None

    monkeypatch.setattr(ui_module.shutil, "which", fake_which)

    with pytest.raises(SystemExit) as excinfo:
        UI(plain=False)

    assert "Interactive dependencies missing" in str(excinfo.value)
    assert {"gum", "sk", "bat", "glow", "delta"}.issubset(set(calls))


def test_ui_choose_interactive_uses_sk(monkeypatch):
    monkeypatch.setattr(ui_module.shutil, "which", lambda _cmd: "/usr/bin/fake")
    calls = []

    def fake_run(cmd, **_kwargs):
        calls.append(cmd)
        if cmd[:1] == ["sk"]:
            return SimpleNamespace(stdout="two\n", stderr="", returncode=0)
        return SimpleNamespace(stdout="", stderr="", returncode=0)

    monkeypatch.setattr(ui_module.subprocess, "run", fake_run)

    view = UI(plain=False)
    assert view.choose("Pick one", ["one", "two", "three"]) == "two"
    assert calls and calls[-1][:2] == ["sk", "--prompt"]


def test_ui_input_interactive_uses_gum(monkeypatch):
    monkeypatch.setattr(ui_module.shutil, "which", lambda _cmd: "/usr/bin/fake")
    calls = []

    def fake_run(cmd, **_kwargs):
        calls.append(cmd)
        if cmd[:2] == ["gum", "input"]:
            return SimpleNamespace(stdout="value\n", stderr="", returncode=0)
        return SimpleNamespace(stdout="", stderr="", returncode=0)

    monkeypatch.setattr(ui_module.subprocess, "run", fake_run)

    view = UI(plain=False)
    assert view.input("Enter", default="fallback") == "value"
    assert any(call[:2] == ["gum", "input"] for call in calls)
