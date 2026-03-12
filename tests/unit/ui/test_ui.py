"""Focused UI facade and wrapper contracts.

Pure renderer behavior lives in ``test_rendering.py``. This file owns prompting,
plain-mode fallbacks, wrapper delegation, and the minimal facade rendering
surface that is specific to ``ConsoleFacade`` / ``UI``.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from polylogue.ui import UI, _PlainProgressTracker, _RichProgressTracker, create_ui
from polylogue.ui.facade import ConsoleFacade, PlainConsole, UIError, create_console_facade


@pytest.fixture
def mock_prompt_file(tmp_path, monkeypatch):
    prompt_file = tmp_path / "prompts.jsonl"
    monkeypatch.setenv("POLYLOGUE_TEST_PROMPT_FILE", str(prompt_file))
    return prompt_file


@pytest.fixture
def mock_facade():
    with patch("polylogue.ui.create_console_facade") as mock_create:
        facade = MagicMock(spec=ConsoleFacade)
        facade.console = MagicMock()
        facade.plain = False
        mock_create.return_value = facade
        yield facade


@pytest.fixture
def rich_facade():
    return create_console_facade(plain=False)


@pytest.fixture
def plain_facade():
    return create_console_facade(plain=True)


PROMPT_TOPICS = {
    "confirm": "confirmation prompts",
    "choose": "menu selections",
    "input": "text input",
}


def _write_stubs(prompt_file, *entries: dict[str, object]) -> None:
    prompt_file.write_text("\n".join(json.dumps(entry) for entry in entries) + "\n")


def _questionary_stub(monkeypatch, method: str, result: object) -> MagicMock:
    import questionary

    question = MagicMock()
    question.ask.return_value = result
    monkeypatch.setattr(questionary, method, lambda *args, **kwargs: question)
    return question


class TestPlainConsole:
    @pytest.mark.parametrize(
        ("objects", "expected", "unexpected"),
        [
            (("hello", "world"), "hello world", None),
            (("[bold]Important[/bold]",), "Important", "[bold]"),
            (("[green]Success[/green]",), "Success", "[green]"),
            (("[#d97757]colored[/#d97757]",), "colored", "[#d97757]"),
            (("[unclosed markup",), "[unclosed markup", None),
        ],
    )
    def test_print_contract(self, capsys, objects, expected, unexpected):
        console = PlainConsole()
        console.print(*objects)
        output = capsys.readouterr().out
        assert expected in output
        if unexpected:
            assert unexpected not in output

    def test_print_ignores_extra_kwargs(self, capsys):
        PlainConsole().print("text", sep="|", end="!\n")
        assert "text" in capsys.readouterr().out

    def test_init_accepts_extra_args(self):
        assert PlainConsole("arg1", key="value") is not None


class TestConsoleFacadePromptStubs:
    @pytest.mark.parametrize(
        ("entries", "expected_count"),
        [([], 0), ([{"type": "confirm", "value": True}], 1), ([{"type": "confirm", "value": True}, {"type": "choose", "value": "a"}, {"type": "input", "value": "x"}], 3)],
    )
    def test_stub_loading_contract(self, tmp_path, monkeypatch, entries, expected_count):
        if entries:
            prompt_file = tmp_path / "stubs.jsonl"
            _write_stubs(prompt_file, *entries)
            monkeypatch.setenv("POLYLOGUE_TEST_PROMPT_FILE", str(prompt_file))
        else:
            monkeypatch.delenv("POLYLOGUE_TEST_PROMPT_FILE", raising=False)
        assert len(ConsoleFacade(plain=True)._prompt_responses) == expected_count

    def test_invalid_json_raises_uierror(self, tmp_path, monkeypatch):
        prompt_file = tmp_path / "bad.jsonl"
        prompt_file.write_text("not json\n")
        monkeypatch.setenv("POLYLOGUE_TEST_PROMPT_FILE", str(prompt_file))
        with pytest.raises(UIError, match="Invalid prompt stub"):
            ConsoleFacade(plain=True)

    def test_pop_prompt_response_contract(self, mock_prompt_file):
        _write_stubs(mock_prompt_file, {"type": "confirm", "value": True})
        facade = ConsoleFacade(plain=True)
        with pytest.raises(UIError, match="expected 'confirm' but got 'choose'"):
            facade._pop_prompt_response("choose")

        _write_stubs(mock_prompt_file, {"value": True})
        facade = ConsoleFacade(plain=True)
        assert facade._pop_prompt_response("anything") == {"value": True}

        mock_prompt_file.write_text("")
        facade = ConsoleFacade(plain=True)
        assert facade._pop_prompt_response("confirm") is None


class TestConsoleFacadePrompts:
    @pytest.mark.parametrize(
        ("kind", "plain_value", "default", "expected"),
        [
            ("confirm", "y", True, True),
            ("confirm", "n", True, False),
            ("confirm", "", False, False),
            ("choose", "2", None, "b"),
            ("choose", "", None, None),
            ("input", "typed", None, "typed"),
            ("input", "", "fallback", "fallback"),
        ],
        ids=["confirm_yes", "confirm_no", "confirm_default", "choose_second", "choose_empty", "input_typed", "input_default"],
    )
    def test_plain_prompt_matrix(self, monkeypatch, kind, plain_value, default, expected):
        monkeypatch.setattr("sys.stdin.isatty", lambda: True)
        monkeypatch.setattr("builtins.input", lambda _: plain_value)
        facade = ConsoleFacade(plain=True)

        if kind == "confirm":
            assert facade.confirm("Continue?", default=default) is expected
        elif kind == "choose":
            assert facade.choose("Pick:", ["a", "b", "c"]) == expected
        else:
            assert facade.input("Name:", default=default) == expected

    @pytest.mark.parametrize("kind", ["confirm", "choose", "input"])
    def test_plain_non_tty_raises_uierror(self, monkeypatch, kind):
        monkeypatch.setattr("sys.stdin.isatty", lambda: False)
        facade = ConsoleFacade(plain=True)
        with pytest.raises(UIError, match=PROMPT_TOPICS[kind]):
            getattr(facade, kind)("Prompt:", default=True) if kind == "confirm" else (
                facade.choose("Prompt:", ["a", "b"]) if kind == "choose" else facade.input("Prompt:")
            )

    @pytest.mark.parametrize(
        ("entry", "expected"),
        [
            ({"type": "confirm", "value": True}, True),
            ({"type": "confirm", "value": "yes"}, True),
            ({"type": "confirm", "value": "0"}, False),
            ({"type": "confirm", "use_default": True}, False),
        ],
        ids=["bool_true", "str_yes", "str_zero", "use_default"],
    )
    def test_confirm_stub_contract(self, mock_prompt_file, entry, expected):
        _write_stubs(mock_prompt_file, entry)
        facade = ConsoleFacade(plain=False)
        assert facade.confirm("Continue?", default=False) is expected

    @pytest.mark.parametrize(
        ("entry", "options", "expected"),
        [
            ({"type": "choose", "value": "option2"}, ["option1", "option2", "option3"], "option2"),
            ({"type": "choose", "index": 2}, ["a", "b", "c"], "c"),
            ({"type": "choose", "index": "2"}, ["a", "b", "c"], "c"),
            ({"type": "choose", "use_default": True}, ["first", "second"], "first"),
        ],
    )
    def test_choose_stub_contract(self, mock_prompt_file, entry, options, expected):
        _write_stubs(mock_prompt_file, entry)
        assert ConsoleFacade(plain=False).choose("Pick:", options) == expected

    @pytest.mark.parametrize(
        ("entry", "default", "expected"),
        [
            ({"type": "input", "value": "typed"}, None, "typed"),
            ({"type": "input", "value": 42}, None, "42"),
            ({"type": "input", "use_default": True}, "fallback", "fallback"),
            ({"type": "input", "value": None}, None, None),
            ({"type": "input", "value": ""}, "default", ""),
        ],
    )
    def test_input_stub_contract(self, mock_prompt_file, entry, default, expected):
        _write_stubs(mock_prompt_file, entry)
        assert ConsoleFacade(plain=False).input("Prompt:", default=default) == expected

    def test_rich_choose_fallback_paths(self, monkeypatch):
        question = _questionary_stub(monkeypatch, "select", None)
        facade = ConsoleFacade(plain=False)
        assert facade.choose("Pick:", ["a", "b", "c"]) is None
        question.ask.assert_called_once()

        question = _questionary_stub(monkeypatch, "autocomplete", "opt13")
        assert facade.choose("Pick:", [f"opt{i}" for i in range(15)]) == "opt13"
        question.ask.assert_called_once()

    def test_rich_confirm_and_input_fallback_paths(self, monkeypatch):
        confirm = _questionary_stub(monkeypatch, "confirm", None)
        input_box = _questionary_stub(monkeypatch, "text", "")
        facade = ConsoleFacade(plain=False)
        assert facade.confirm("Continue?", default=True) is True
        assert facade.input("Name:", default="fallback") == "fallback"
        confirm.ask.assert_called_once()
        input_box.ask.assert_called_once()


class TestConsoleFacadeRendering:
    def test_console_facade_creation_contract(self):
        assert create_console_facade(plain=True).plain is True
        assert create_console_facade(plain=False).plain is False

    @pytest.mark.parametrize(
        ("method", "args", "expected"),
        [
            ("banner", ("Title", "Subtitle"), ["Title", "Subtitle"]),
            ("summary", ("Stats", ["Line 1", "[green]✓[/green] Success"]), ["Stats", "Line 1", "Success"]),
            ("render_markdown", ("# Hello\n\nWorld",), ["Hello", "World"]),
            ("render_code", ("print('hello')", "python"), ["print"]),
            ("render_diff", ("old\n", "new\n", "test.txt"), ["test.txt"]),
            ("success", ("Done",), ["Done"]),
            ("warning", ("Careful",), ["Careful"]),
            ("error", ("Broken",), ["Broken"]),
            ("info", ("FYI",), ["FYI"]),
        ],
    )
    def test_plain_rendering_surface_contract(self, capsys, method, args, expected):
        facade = ConsoleFacade(plain=True)
        getattr(facade, method)(*args)
        output = capsys.readouterr().out
        for text in expected:
            assert text in output

    def test_rich_rendering_surface_contract(self, rich_facade, capsys):
        rich_facade.banner("Welcome", "Mission Control")
        rich_facade.summary("Checklist", ["Item 1", "[red]Item 2[/red]"])
        rich_facade.render_markdown("# Title\n\nBody")
        rich_facade.render_code("print('hi')", "python")
        rich_facade.render_diff("old\n", "new\n", "test.txt")
        rich_facade.success("Good")
        rich_facade.warning("Warn")
        rich_facade.error("Oops")
        rich_facade.info("FYI")
        output = capsys.readouterr().out
        assert "Welcome" in output
        assert "Mission Control" in output
        assert "Checklist" in output
        assert "Item 2" in output
        assert "test.txt" in output
        assert "Good" in output
        assert "Warn" in output
        assert "Oops" in output
        assert "FYI" in output

    def test_render_diff_falls_back_when_pager_missing(self, plain_facade):
        plain_facade.render_diff("line1\n", "line2\n", "file.txt")

    def test_protocol_theme_contract(self):
        plain = ConsoleFacade(plain=True)
        rich = ConsoleFacade(plain=False)
        assert callable(getattr(plain.console, "print", None))
        assert callable(getattr(rich.console, "print", None))
        assert plain.theme is not None
        assert rich.theme is not None
        assert rich._panel_box is not None
        assert rich._banner_box is not None
        assert rich._banner_box != rich._panel_box


class TestUIWrapper:
    def test_ui_init_and_create_contract(self, mock_facade):
        ui = UI(plain=False)
        assert ui._facade == mock_facade
        assert ui.plain is False
        assert ui.console == mock_facade.console
        assert isinstance(create_ui(plain=True), UI)

        with patch("polylogue.ui.create_console_facade", side_effect=RuntimeError("boom")):
            with pytest.raises(SystemExit, match="boom"):
                UI(plain=False)

    def test_ui_delegation_contract(self, mock_facade):
        ui = UI(plain=False)
        ui.banner("Title", "Subtitle")
        ui.summary("Summary", ["line1"])
        ui.render_markdown("# Title")
        ui.render_code("print('hi')")
        ui.render_diff("old", "new")
        mock_facade.banner.assert_called_with("Title", "Subtitle")
        mock_facade.summary.assert_called_with("Summary", ["line1"])
        mock_facade.render_markdown.assert_called_with("# Title")
        mock_facade.render_code.assert_called_with("print('hi')", "python")
        mock_facade.render_diff.assert_called_with("old", "new", "file")

    @pytest.mark.parametrize(
        ("method", "args", "kwargs"),
        [
            ("confirm", ("Are you sure?",), {"default": True}),
            ("choose", ("Pick", ["A", "B"]), {}),
            ("input", ("Prompt",), {"default": None}),
        ],
    )
    def test_ui_prompt_delegation_contract(self, mock_facade, method, args, kwargs):
        ui = UI(plain=False)
        getattr(mock_facade, method).return_value = True if method == "confirm" else "A" if method == "choose" else "val"
        result = getattr(ui, method)(*args, **kwargs)
        assert result is not None
        getattr(mock_facade, method).assert_called_once_with(*args, **kwargs)

    @pytest.mark.parametrize(
        ("method", "call"),
        [
            ("confirm", lambda ui: ui.confirm("Q?")),
            ("choose", lambda ui: ui.choose("Pick", ["A", "B"])),
            ("input", lambda ui: ui.input("Prompt")),
        ],
    )
    def test_ui_plain_prompt_abort_contract(self, mock_facade, method, call):
        mock_facade.plain = True
        mock_facade.console = MagicMock()
        getattr(mock_facade, method).side_effect = UIError(f"Plain mode cannot prompt for {PROMPT_TOPICS[method]}", prompt_topic=PROMPT_TOPICS[method])
        ui = UI(plain=True)
        with pytest.raises(SystemExit):
            call(ui)
        mock_facade.console.print.assert_called()


class TestProgressTrackers:
    def test_plain_progress_tracker_contract(self):
        console = MagicMock()
        tracker = _PlainProgressTracker(console, "Task", 10)
        with tracker:
            tracker.advance(1)
            tracker.update(description="New Task")
            tracker.update(total=20)
        assert console.print.call_count >= 2

        console.reset_mock()
        tracker = _PlainProgressTracker(console, "Task2", None)
        with tracker:
            tracker.advance(1.5)
            tracker.update(total=5.5)
        assert console.print.call_count >= 2

    def test_rich_progress_tracker_contract(self):
        progress = MagicMock()
        task_id = "task-1"
        tracker = _RichProgressTracker(progress, task_id)
        with tracker:
            tracker.advance(5)
            tracker.update(total=100, description="Processing")
        progress.__enter__.assert_called_once()
        progress.advance.assert_called_with(task_id, 5)
        progress.update.assert_called()
        progress.__exit__.assert_called_once()
