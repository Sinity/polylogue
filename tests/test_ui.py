from __future__ import annotations

import re

import pytest

from polylogue import ui as ui_module
from polylogue.ui import UI


def test_ui_summary_handles_brackets_without_markup():
    view = UI(plain=False)
    lines = ["Copy config to [/tmp/example.json, /tmp/fallback.json]"]
    with view.console.capture() as capture:
        view.summary("Doctor", lines)
    assert "/tmp/example.json" in capture.get()


def test_ui_confirm_interactive_uses_questionary(monkeypatch):
    calls = []

    class DummyPrompt:
        def __init__(self, prompt: str, default: bool) -> None:
            calls.append((prompt, default))

        def ask(self):
            return True

    monkeypatch.setattr(
        "polylogue.ui.facade.questionary.confirm",
        lambda prompt, default=True: DummyPrompt(prompt, default),
    )

    view = UI(plain=False)
    assert view.confirm("Proceed?", default=True) is True
    assert calls[0] == ("Proceed?", True)

    view.confirm("Proceed?", default=False)
    assert calls[-1] == ("Proceed?", False)


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


def test_ui_choose_interactive_uses_questionary(monkeypatch):
    calls = []

    class DummySelect:
        def __init__(self, prompt: str, choices: list[str]):
            calls.append((prompt, tuple(choices)))

        def ask(self):
            return "two"

    monkeypatch.setattr(
        "polylogue.ui.facade.questionary.select",
        lambda prompt, choices: DummySelect(prompt, choices),
    )

    view = UI(plain=False)
    assert view.choose("Pick one", ["one", "two", "three"]) == "two"
    assert calls and calls[0][0] == "Pick one"


def test_ui_input_interactive_uses_questionary(monkeypatch):
    calls = []

    class DummyText:
        def __init__(self, prompt: str, default: str):
            calls.append((prompt, default))

        def ask(self):
            return "value"

    monkeypatch.setattr(
        "polylogue.ui.facade.questionary.text",
        lambda prompt, default="": DummyText(prompt, default),
    )

    view = UI(plain=False)
    assert view.input("Enter", default="fallback") == "value"
    assert calls and calls[0] == ("Enter", "fallback")
