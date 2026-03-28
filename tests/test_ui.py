from __future__ import annotations

import re
from contextlib import contextmanager

import pytest

from polylogue import ui as ui_module
from polylogue.ui import UI
from polylogue.ui.facade import ConsoleFacade


class _RecordingConsole:
    def __init__(self) -> None:
        self.lines: list[str] = []

    def print(self, *args: object, **_kwargs: object) -> None:
        self.lines.append(" ".join(str(arg) for arg in args))


_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[mK]")


def _strip_ansi(text: str) -> str:
    return _ANSI_RE.sub("", text)


def test_ui_summary_handles_brackets_without_markup():
    view = UI(plain=False)
    lines = ["Copy config to [/tmp/example.json, /tmp/fallback.json]"]
    with view.console.capture() as capture:
        view.summary("Doctor", lines)
    assert "/tmp/example.json" in _strip_ansi(capture.get())


def test_ui_summary_plain_mode_prints_body():
    view = UI(plain=True)
    console = _RecordingConsole()
    view.console = console

    view.summary("Summary", ["Line one", "Line two"])

    assert console.lines[0] == "-- Summary --"
    assert console.lines[1] == "Line one\nLine two"


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


def test_plain_progress_formats_integerish_values():
    tracker = ui_module._PlainProgressTracker
    assert tracker._coerce_int(5.0000001) == 5
    assert tracker._format_value(12.0000001) == "12"
    assert tracker._format_value(12.5) == "12.5"


def test_ui_banner_plain_mode_prints_lines():
    view = UI(plain=True)
    console = _RecordingConsole()
    view.console = console

    view.banner("Title", "Subtitle")

    assert console.lines[0] == "== Title =="
    assert console.lines[1] == "Subtitle"


def test_ui_banner_rich_mode_includes_title():
    view = UI(plain=False)
    with view.console.capture() as capture:
        view.banner("Title", "Subtitle")
    output = _strip_ansi(capture.get())
    assert "Title" in output
    assert "Subtitle" in output


def test_ui_render_markdown_plain_mode_prints_body():
    view = UI(plain=True)
    console = _RecordingConsole()
    view.console = console

    view.render_markdown("**Hello**")

    assert console.lines == ["**Hello**"]


def test_ui_render_markdown_rich_mode_includes_body():
    view = UI(plain=False)
    with view.console.capture() as capture:
        view.render_markdown("**Hello**")
    assert "Hello" in _strip_ansi(capture.get())


def test_ui_render_code_plain_mode_prints_body():
    view = UI(plain=True)
    console = _RecordingConsole()
    view.console = console

    view.render_code("print('ok')", language="python")

    assert console.lines == ["print('ok')"]


def test_ui_render_code_rich_mode_includes_source():
    view = UI(plain=False)
    with view.console.capture() as capture:
        view.render_code("print('ok')", language="python")
    assert "print('ok')" in _strip_ansi(capture.get())


def test_ui_render_diff_plain_mode_prints_unified_diff():
    view = UI(plain=True)
    console = _RecordingConsole()
    view.console = console

    view.render_diff("old\n", "new\n", filename="sample.txt")

    assert console.lines
    diff_text = "\n".join(console.lines)
    assert "a/sample.txt" in diff_text
    assert "b/sample.txt" in diff_text
    assert "-old" in diff_text
    assert "+new" in diff_text


def test_ui_render_diff_rich_mode_includes_filenames(monkeypatch):
    view = UI(plain=False)

    @contextmanager
    def _pager(*_args, **_kwargs):
        yield None

    monkeypatch.setattr(view.console, "pager", _pager)
    with view.console.capture() as capture:
        view.render_diff("old\n", "new\n", filename="sample.txt")
    output = _strip_ansi(capture.get())
    assert "sample.txt" in output


def test_ui_progress_plain_emits_completion_line():
    view = UI(plain=True)
    console = _RecordingConsole()
    view.console = console

    with view.progress("Syncing items", total=2) as tracker:
        tracker.advance()
        tracker.advance()

    assert console.lines[0] == "Syncing items..."
    assert any("complete" in line and "(2/2)" in line for line in console.lines)


def test_ui_progress_rich_uses_integer_format(monkeypatch):
    captured = {}

    class DummyProgress:
        def __init__(self, *columns, **_kwargs):
            captured["columns"] = columns

        def add_task(self, description, total=None):
            captured["description"] = description
            captured["total"] = total
            return 1

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def advance(self, _task_id, advance):
            captured["advance"] = advance

        def update(self, _task_id, **kwargs):
            captured["update"] = kwargs

    monkeypatch.setattr(ui_module, "Progress", DummyProgress)

    view = UI(plain=False)
    tracker = view.progress("Syncing items", total=3)

    count_column = next(
        column
        for column in captured["columns"]
        if isinstance(column, ui_module.TextColumn) and "task.completed" in column.text_format
    )
    assert count_column.text_format == "{task.completed:.0f}/{task.total:.0f}"

    with tracker:
        tracker.advance(1)
        tracker.update(description="Next", total=4)

    assert captured["description"] == "Syncing items"
    assert captured["total"] == 3
    assert captured["advance"] == 1
    assert captured["update"] == {"description": "Next", "total": 4}


def test_console_facade_status_plain_outputs_message_only():
    facade = ConsoleFacade(plain=True)
    console = _RecordingConsole()
    facade.console = console

    facade.error("error")
    facade.warning("warn")
    facade.success("ok")
    facade.info("info")

    assert console.lines[0].split(" ", 1)[1] == "error"
    assert console.lines[1].split(" ", 1)[1] == "warn"
    assert console.lines[2].split(" ", 1)[1] == "ok"
    assert console.lines[3] == "info"


def test_console_facade_status_rich_includes_messages():
    facade = ConsoleFacade(plain=False)
    with facade.console.capture() as capture:
        facade.error("error")
        facade.warning("warn")
        facade.success("ok")
        facade.info("info")
    output = _strip_ansi(capture.get())
    assert "error" in output
    assert "warn" in output
    assert "ok" in output
    assert "info" in output
