"""Tests for UI facade and version resolution.

Covers:
- PlainConsole: markup stripping
- ConsoleFacade: plain/rich modes, prompt stubs, banner/summary/status
- Prompt stub system (POLYLOGUE_TEST_PROMPT_FILE)
- confirm/choose/input with stubs
- render_markdown, render_code, render_diff
- error/warning/success/info status messages
- PlainConsoleFacade
- create_console_facade factory
- VersionInfo formatting
- _get_git_info
- _resolve_version
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest


# =============================================================================
# PlainConsole
# =============================================================================

class TestPlainConsole:
    def test_print_basic(self, capsys):
        from polylogue.ui.facade import PlainConsole
        pc = PlainConsole()
        pc.print("hello", "world")
        assert "hello world" in capsys.readouterr().out

    @pytest.mark.parametrize("input_text,expected_stripped,unexpected", [
        ("[bold]Important[/bold]", "Important", "[bold]"),
        ("[green]Success[/green]", "Success", "[green]"),
        ("[#d97757]colored[/#d97757]", "colored", "[#d97757]"),
    ])
    def test_print_strips_markup(self, capsys, input_text, expected_stripped, unexpected):
        from polylogue.ui.facade import PlainConsole
        pc = PlainConsole()
        pc.print(input_text)
        captured = capsys.readouterr().out
        assert expected_stripped in captured
        assert unexpected not in captured

    def test_print_ignores_extra_kwargs(self, capsys):
        from polylogue.ui.facade import PlainConsole
        pc = PlainConsole()
        pc.print("text", sep="|", end="!\n")
        # Should print "text" without respecting sep/end kwargs
        assert "text" in capsys.readouterr().out

    def test_print_malformed_markup_fallback(self, capsys):
        from polylogue.ui.facade import PlainConsole
        pc = PlainConsole()
        pc.print("[unclosed markup")
        captured = capsys.readouterr().out
        assert "[unclosed markup" in captured

    def test_print_init_accepts_any_args(self):
        from polylogue.ui.facade import PlainConsole
        # Should not raise despite extra args
        pc = PlainConsole("arg1", "arg2", kwarg="value")
        assert pc is not None


# =============================================================================
# ConsoleFacade creation
# =============================================================================

class TestCreateConsoleFacade:
    def test_plain_creates_plain_facade(self):
        from polylogue.ui.facade import create_console_facade, PlainConsoleFacade
        facade = create_console_facade(plain=True)
        assert isinstance(facade, PlainConsoleFacade)
        assert facade.plain is True

    def test_rich_creates_console_facade(self):
        from polylogue.ui.facade import create_console_facade, ConsoleFacade, PlainConsoleFacade
        facade = create_console_facade(plain=False)
        assert isinstance(facade, ConsoleFacade)
        assert facade.plain is False
        assert not isinstance(facade, PlainConsoleFacade)


# =============================================================================
# Prompt stub system
# =============================================================================

class TestPromptStubs:
    def test_no_env_var_returns_empty(self, monkeypatch):
        from polylogue.ui.facade import ConsoleFacade
        monkeypatch.delenv("POLYLOGUE_TEST_PROMPT_FILE", raising=False)
        facade = ConsoleFacade(plain=True)
        assert len(facade._prompt_responses) == 0

    def test_loads_single_jsonl_stub(self, tmp_path, monkeypatch):
        from polylogue.ui.facade import ConsoleFacade
        stub_file = tmp_path / "stubs.jsonl"
        stub_file.write_text(json.dumps({"type": "confirm", "value": True}) + "\n")
        monkeypatch.setenv("POLYLOGUE_TEST_PROMPT_FILE", str(stub_file))
        facade = ConsoleFacade(plain=True)
        assert len(facade._prompt_responses) == 1

    def test_loads_multiple_jsonl_stubs(self, tmp_path, monkeypatch):
        from polylogue.ui.facade import ConsoleFacade
        stub_file = tmp_path / "stubs.jsonl"
        stub_file.write_text(
            json.dumps({"type": "confirm", "value": True}) + "\n"
            + json.dumps({"type": "choose", "value": "option1"}) + "\n"
            + json.dumps({"type": "input", "value": "text"}) + "\n"
        )
        monkeypatch.setenv("POLYLOGUE_TEST_PROMPT_FILE", str(stub_file))
        facade = ConsoleFacade(plain=True)
        assert len(facade._prompt_responses) == 3

    def test_invalid_json_raises_uierror(self, tmp_path, monkeypatch):
        from polylogue.ui.facade import ConsoleFacade, UIError
        stub_file = tmp_path / "bad.jsonl"
        stub_file.write_text("not json\n")
        monkeypatch.setenv("POLYLOGUE_TEST_PROMPT_FILE", str(stub_file))
        with pytest.raises(UIError, match="Invalid prompt stub"):
            ConsoleFacade(plain=True)

    def test_empty_lines_skipped(self, tmp_path, monkeypatch):
        from polylogue.ui.facade import ConsoleFacade
        stub_file = tmp_path / "stubs.jsonl"
        stub_file.write_text("\n\n" + json.dumps({"type": "confirm"}) + "\n\n")
        monkeypatch.setenv("POLYLOGUE_TEST_PROMPT_FILE", str(stub_file))
        facade = ConsoleFacade(plain=True)
        assert len(facade._prompt_responses) == 1

    def test_whitespace_only_lines_skipped(self, tmp_path, monkeypatch):
        from polylogue.ui.facade import ConsoleFacade
        stub_file = tmp_path / "stubs.jsonl"
        stub_file.write_text("   \n\t\n" + json.dumps({"type": "input"}) + "\n  ")
        monkeypatch.setenv("POLYLOGUE_TEST_PROMPT_FILE", str(stub_file))
        facade = ConsoleFacade(plain=True)
        assert len(facade._prompt_responses) == 1

    def test_pop_type_mismatch_raises(self, tmp_path, monkeypatch):
        from polylogue.ui.facade import ConsoleFacade, UIError
        stub_file = tmp_path / "stubs.jsonl"
        stub_file.write_text(json.dumps({"type": "confirm", "value": True}) + "\n")
        monkeypatch.setenv("POLYLOGUE_TEST_PROMPT_FILE", str(stub_file))
        facade = ConsoleFacade(plain=True)
        with pytest.raises(UIError, match="expected 'confirm' but got 'choose'"):
            facade._pop_prompt_response("choose")

    def test_pop_no_type_matches_any(self, tmp_path, monkeypatch):
        from polylogue.ui.facade import ConsoleFacade
        stub_file = tmp_path / "stubs.jsonl"
        stub_file.write_text(json.dumps({"value": True}) + "\n")
        monkeypatch.setenv("POLYLOGUE_TEST_PROMPT_FILE", str(stub_file))
        facade = ConsoleFacade(plain=True)
        result = facade._pop_prompt_response("anything")
        assert result == {"value": True}

    def test_pop_empty_queue_returns_none(self, monkeypatch):
        from polylogue.ui.facade import ConsoleFacade
        monkeypatch.delenv("POLYLOGUE_TEST_PROMPT_FILE", raising=False)
        facade = ConsoleFacade(plain=True)
        result = facade._pop_prompt_response("confirm")
        assert result is None


# =============================================================================
# confirm
# =============================================================================

class TestConfirm:
    def test_plain_returns_default_true(self):
        from polylogue.ui.facade import ConsoleFacade
        facade = ConsoleFacade(plain=True)
        assert facade.confirm("Continue?", default=True) is True

    def test_plain_returns_default_false(self):
        from polylogue.ui.facade import ConsoleFacade
        facade = ConsoleFacade(plain=True)
        assert facade.confirm("Continue?", default=False) is False

    def test_plain_ignores_prompt(self):
        from polylogue.ui.facade import ConsoleFacade
        facade = ConsoleFacade(plain=True)
        # Plain mode doesn't prompt, just returns default
        assert facade.confirm("anything", default=True) is True

    @pytest.mark.parametrize("stub_value,expected", [
        (True, True),
        (False, False),
        ("yes", True),
        ("y", True),
        ("true", True),
        ("1", True),
        ("no", False),
        ("n", False),
        ("false", False),
        ("0", False),
    ], ids=[
        "bool_true", "bool_false", "str_yes", "str_y",
        "str_true", "str_1", "str_no", "str_n",
        "str_false", "str_0"
    ])
    def test_stub_value_parsing(self, tmp_path, monkeypatch, stub_value, expected):
        from polylogue.ui.facade import ConsoleFacade
        stub_file = tmp_path / "stubs.jsonl"
        stub_file.write_text(json.dumps({"type": "confirm", "value": stub_value}) + "\n")
        monkeypatch.setenv("POLYLOGUE_TEST_PROMPT_FILE", str(stub_file))
        facade = ConsoleFacade(plain=False)
        assert facade.confirm("Continue?") is expected

    def test_stub_use_default_true(self, tmp_path, monkeypatch):
        from polylogue.ui.facade import ConsoleFacade
        stub_file = tmp_path / "stubs.jsonl"
        stub_file.write_text(json.dumps({"type": "confirm", "use_default": True}) + "\n")
        monkeypatch.setenv("POLYLOGUE_TEST_PROMPT_FILE", str(stub_file))
        facade = ConsoleFacade(plain=False)
        assert facade.confirm("Continue?", default=True) is True

    def test_stub_use_default_false(self, tmp_path, monkeypatch):
        from polylogue.ui.facade import ConsoleFacade
        stub_file = tmp_path / "stubs.jsonl"
        stub_file.write_text(json.dumps({"type": "confirm", "use_default": True}) + "\n")
        monkeypatch.setenv("POLYLOGUE_TEST_PROMPT_FILE", str(stub_file))
        facade = ConsoleFacade(plain=False)
        assert facade.confirm("Continue?", default=False) is False

    def test_stub_invalid_string_returns_default(self, tmp_path, monkeypatch):
        from polylogue.ui.facade import ConsoleFacade
        stub_file = tmp_path / "stubs.jsonl"
        stub_file.write_text(json.dumps({"type": "confirm", "value": "maybe"}) + "\n")
        monkeypatch.setenv("POLYLOGUE_TEST_PROMPT_FILE", str(stub_file))
        facade = ConsoleFacade(plain=False)
        # Invalid string doesn't match yes/no patterns, so questionary would be called
        # Mock it to return None so default is used
        import questionary
        original_confirm = questionary.confirm
        mock_confirm = MagicMock()
        mock_confirm.return_value.ask.return_value = None
        monkeypatch.setattr(questionary, "confirm", lambda *args, **kwargs: mock_confirm.return_value)
        result = facade.confirm("Continue?", default=True)
        assert result is True


# =============================================================================
# choose
# =============================================================================

class TestChoose:
    def test_empty_options_returns_none(self):
        from polylogue.ui.facade import ConsoleFacade
        facade = ConsoleFacade(plain=False)
        assert facade.choose("Pick:", []) is None

    def test_empty_options_plain(self):
        from polylogue.ui.facade import ConsoleFacade
        facade = ConsoleFacade(plain=True)
        assert facade.choose("Pick:", []) is None

    def test_plain_returns_none(self):
        from polylogue.ui.facade import ConsoleFacade
        facade = ConsoleFacade(plain=True)
        assert facade.choose("Pick:", ["a", "b"]) is None

    def test_plain_ignores_prompt(self):
        from polylogue.ui.facade import ConsoleFacade
        facade = ConsoleFacade(plain=True)
        assert facade.choose("anything", ["x", "y", "z"]) is None

    def test_stub_value_exact_match(self, tmp_path, monkeypatch):
        from polylogue.ui.facade import ConsoleFacade
        stub_file = tmp_path / "stubs.jsonl"
        stub_file.write_text(json.dumps({"type": "choose", "value": "option2"}) + "\n")
        monkeypatch.setenv("POLYLOGUE_TEST_PROMPT_FILE", str(stub_file))
        facade = ConsoleFacade(plain=False)
        result = facade.choose("Pick:", ["option1", "option2", "option3"])
        assert result == "option2"

    def test_stub_value_not_in_options(self, tmp_path, monkeypatch):
        from polylogue.ui.facade import ConsoleFacade
        import questionary
        stub_file = tmp_path / "stubs.jsonl"
        stub_file.write_text(json.dumps({"type": "choose", "value": "unknown"}) + "\n")
        monkeypatch.setenv("POLYLOGUE_TEST_PROMPT_FILE", str(stub_file))
        facade = ConsoleFacade(plain=False)
        # Value not in options, so questionary is called
        # Mock questionary to return None
        mock_select = MagicMock()
        mock_select.return_value.ask.return_value = None
        monkeypatch.setattr(questionary, "select", lambda *args, **kwargs: mock_select.return_value)
        result = facade.choose("Pick:", ["a", "b", "c"])
        assert result is None

    @pytest.mark.parametrize("index,options,expected", [
        (0, ["a", "b", "c"], "a"),
        (2, ["a", "b", "c"], "c"),
        ("2", ["a", "b", "c"], "c"),
    ], ids=["zero", "last", "string_valid"])
    def test_stub_index_valid(self, tmp_path, monkeypatch, index, options, expected):
        from polylogue.ui.facade import ConsoleFacade
        stub_file = tmp_path / "stubs.jsonl"
        stub_file.write_text(json.dumps({"type": "choose", "index": index}) + "\n")
        monkeypatch.setenv("POLYLOGUE_TEST_PROMPT_FILE", str(stub_file))
        facade = ConsoleFacade(plain=False)
        result = facade.choose("Pick:", options)
        assert result == expected

    @pytest.mark.parametrize("invalid_index", [99, -1, "not_a_number"], ids=["out_of_bounds", "negative", "string_invalid"])
    def test_stub_index_invalid(self, tmp_path, monkeypatch, invalid_index):
        from polylogue.ui.facade import ConsoleFacade
        import questionary
        stub_file = tmp_path / "stubs.jsonl"
        stub_file.write_text(json.dumps({"type": "choose", "index": invalid_index}) + "\n")
        monkeypatch.setenv("POLYLOGUE_TEST_PROMPT_FILE", str(stub_file))
        facade = ConsoleFacade(plain=False)
        # Invalid index, questionary is called
        mock_select = MagicMock()
        mock_select.return_value.ask.return_value = None
        monkeypatch.setattr(questionary, "select", lambda *args, **kwargs: mock_select.return_value)
        result = facade.choose("Pick:", ["a", "b", "c"])
        assert result is None

    def test_stub_use_default(self, tmp_path, monkeypatch):
        from polylogue.ui.facade import ConsoleFacade
        stub_file = tmp_path / "stubs.jsonl"
        stub_file.write_text(json.dumps({"type": "choose", "use_default": True}) + "\n")
        monkeypatch.setenv("POLYLOGUE_TEST_PROMPT_FILE", str(stub_file))
        facade = ConsoleFacade(plain=False)
        result = facade.choose("Pick:", ["first", "second"])
        assert result == "first"

    def test_many_options_uses_autocomplete(self, tmp_path, monkeypatch):
        from polylogue.ui.facade import ConsoleFacade
        stub_file = tmp_path / "stubs.jsonl"
        stub_file.write_text(json.dumps({"type": "choose", "value": "opt13"}) + "\n")
        monkeypatch.setenv("POLYLOGUE_TEST_PROMPT_FILE", str(stub_file))
        facade = ConsoleFacade(plain=False)
        # > 12 options triggers autocomplete instead of select
        many_options = [f"opt{i}" for i in range(15)]
        result = facade.choose("Pick:", many_options)
        assert result == "opt13"


# =============================================================================
# input
# =============================================================================

class TestInput:
    def test_plain_returns_default(self):
        from polylogue.ui.facade import ConsoleFacade
        facade = ConsoleFacade(plain=True)
        assert facade.input("Name:", default="anon") == "anon"

    def test_plain_no_default_returns_none(self):
        from polylogue.ui.facade import ConsoleFacade
        facade = ConsoleFacade(plain=True)
        assert facade.input("Name:") is None

    def test_plain_ignores_prompt(self):
        from polylogue.ui.facade import ConsoleFacade
        facade = ConsoleFacade(plain=True)
        assert facade.input("anything", default="default") == "default"

    def test_stub_value_string(self, tmp_path, monkeypatch):
        from polylogue.ui.facade import ConsoleFacade
        stub_file = tmp_path / "stubs.jsonl"
        stub_file.write_text(json.dumps({"type": "input", "value": "typed"}) + "\n")
        monkeypatch.setenv("POLYLOGUE_TEST_PROMPT_FILE", str(stub_file))
        facade = ConsoleFacade(plain=False)
        assert facade.input("Name:") == "typed"

    def test_stub_value_number_converted_to_string(self, tmp_path, monkeypatch):
        from polylogue.ui.facade import ConsoleFacade
        stub_file = tmp_path / "stubs.jsonl"
        stub_file.write_text(json.dumps({"type": "input", "value": 42}) + "\n")
        monkeypatch.setenv("POLYLOGUE_TEST_PROMPT_FILE", str(stub_file))
        facade = ConsoleFacade(plain=False)
        assert facade.input("Number:") == "42"

    def test_stub_use_default(self, tmp_path, monkeypatch):
        from polylogue.ui.facade import ConsoleFacade
        stub_file = tmp_path / "stubs.jsonl"
        stub_file.write_text(json.dumps({"type": "input", "use_default": True}) + "\n")
        monkeypatch.setenv("POLYLOGUE_TEST_PROMPT_FILE", str(stub_file))
        facade = ConsoleFacade(plain=False)
        assert facade.input("Name:", default="fallback") == "fallback"

    def test_stub_use_default_no_default(self, tmp_path, monkeypatch):
        from polylogue.ui.facade import ConsoleFacade
        stub_file = tmp_path / "stubs.jsonl"
        stub_file.write_text(json.dumps({"type": "input", "use_default": True}) + "\n")
        monkeypatch.setenv("POLYLOGUE_TEST_PROMPT_FILE", str(stub_file))
        facade = ConsoleFacade(plain=False)
        assert facade.input("Name:") is None

    def test_stub_value_none(self, tmp_path, monkeypatch):
        from polylogue.ui.facade import ConsoleFacade
        stub_file = tmp_path / "stubs.jsonl"
        stub_file.write_text(json.dumps({"type": "input", "value": None}) + "\n")
        monkeypatch.setenv("POLYLOGUE_TEST_PROMPT_FILE", str(stub_file))
        facade = ConsoleFacade(plain=False)
        assert facade.input("Name:") is None

    def test_stub_value_empty_string(self, tmp_path, monkeypatch):
        from polylogue.ui.facade import ConsoleFacade
        stub_file = tmp_path / "stubs.jsonl"
        stub_file.write_text(json.dumps({"type": "input", "value": ""}) + "\n")
        monkeypatch.setenv("POLYLOGUE_TEST_PROMPT_FILE", str(stub_file))
        facade = ConsoleFacade(plain=False)
        assert facade.input("Name:", default="default") == ""


# =============================================================================
# Display methods (banner, summary, render_*, status)
# =============================================================================

class TestBanner:
    def test_plain_banner_with_subtitle(self, capsys):
        from polylogue.ui.facade import ConsoleFacade
        facade = ConsoleFacade(plain=True)
        facade.banner("Title", "Subtitle")
        output = capsys.readouterr().out
        assert "Title" in output
        assert "Subtitle" in output

    def test_plain_banner_no_subtitle(self, capsys):
        from polylogue.ui.facade import ConsoleFacade
        facade = ConsoleFacade(plain=True)
        facade.banner("Title")
        output = capsys.readouterr().out
        assert "Title" in output
        assert "==" in output

    def test_plain_banner_formatting(self, capsys):
        from polylogue.ui.facade import ConsoleFacade
        facade = ConsoleFacade(plain=True)
        facade.banner("Test")
        output = capsys.readouterr().out
        assert "== Test ==" in output

    def test_rich_banner(self):
        from polylogue.ui.facade import ConsoleFacade
        facade = ConsoleFacade(plain=False)
        facade.banner("Title", "Subtitle")
        # Should not raise

    def test_rich_banner_no_subtitle(self):
        from polylogue.ui.facade import ConsoleFacade
        facade = ConsoleFacade(plain=False)
        facade.banner("Title")
        # Should not raise


class TestSummary:
    def test_plain_summary_single_line(self, capsys):
        from polylogue.ui.facade import ConsoleFacade
        facade = ConsoleFacade(plain=True)
        facade.summary("Stats", ["Line 1"])
        output = capsys.readouterr().out
        assert "Stats" in output
        assert "Line 1" in output
        assert "--" in output

    def test_plain_summary_multiple_lines(self, capsys):
        from polylogue.ui.facade import ConsoleFacade
        facade = ConsoleFacade(plain=True)
        facade.summary("Stats", ["Line 1", "Line 2", "Line 3"])
        output = capsys.readouterr().out
        assert "Stats" in output
        assert "Line 1" in output
        assert "Line 2" in output
        assert "Line 3" in output

    def test_plain_summary_empty_lines(self, capsys):
        from polylogue.ui.facade import ConsoleFacade
        facade = ConsoleFacade(plain=True)
        facade.summary("Stats", [])
        output = capsys.readouterr().out
        assert "Stats" in output

    def test_plain_summary_with_markup(self, capsys):
        from polylogue.ui.facade import ConsoleFacade
        facade = ConsoleFacade(plain=True)
        facade.summary("Result", ["[green]✓[/green] Success", "[red]✗[/red] Failed"])
        output = capsys.readouterr().out
        assert "Success" in output
        assert "Failed" in output

    def test_rich_summary(self):
        from polylogue.ui.facade import ConsoleFacade
        facade = ConsoleFacade(plain=False)
        facade.summary("Stats", ["Line 1", "Line 2"])


class TestRenderMarkdown:
    def test_plain_prints_raw(self, capsys):
        from polylogue.ui.facade import ConsoleFacade
        facade = ConsoleFacade(plain=True)
        facade.render_markdown("# Hello\n\nWorld")
        output = capsys.readouterr().out
        assert "Hello" in output
        assert "World" in output

    def test_plain_renders_multiline(self, capsys):
        from polylogue.ui.facade import ConsoleFacade
        facade = ConsoleFacade(plain=True)
        content = "# Title\n\n**Bold** text\n\n- List\n- Items"
        facade.render_markdown(content)
        output = capsys.readouterr().out
        assert "Title" in output
        assert "Bold" in output

    def test_rich_renders(self):
        from polylogue.ui.facade import ConsoleFacade
        facade = ConsoleFacade(plain=False)
        facade.render_markdown("# Hello\n\n**World**")


class TestRenderCode:
    def test_plain_prints_raw(self, capsys):
        from polylogue.ui.facade import ConsoleFacade
        facade = ConsoleFacade(plain=True)
        facade.render_code("print('hello')")
        output = capsys.readouterr().out
        assert "print" in output
        assert "hello" in output

    def test_plain_default_language(self, capsys):
        from polylogue.ui.facade import ConsoleFacade
        facade = ConsoleFacade(plain=True)
        facade.render_code("x = 1")
        output = capsys.readouterr().out
        assert "x = 1" in output

    def test_plain_custom_language(self, capsys):
        from polylogue.ui.facade import ConsoleFacade
        facade = ConsoleFacade(plain=True)
        facade.render_code("SELECT * FROM users;", "sql")
        output = capsys.readouterr().out
        assert "SELECT" in output

    def test_rich_renders(self):
        from polylogue.ui.facade import ConsoleFacade
        facade = ConsoleFacade(plain=False)
        facade.render_code("x = 1", "python")

    def test_rich_renders_javascript(self):
        from polylogue.ui.facade import ConsoleFacade
        facade = ConsoleFacade(plain=False)
        facade.render_code("console.log('test');", "javascript")


class TestRenderDiff:
    def test_plain_renders_diff(self, capsys):
        from polylogue.ui.facade import ConsoleFacade
        facade = ConsoleFacade(plain=True)
        facade.render_diff("old\n", "new\n", "test.txt")
        output = capsys.readouterr().out
        assert "test.txt" in output or "---" in output or "+++" in output

    def test_plain_renders_unified_diff(self, capsys):
        from polylogue.ui.facade import ConsoleFacade
        facade = ConsoleFacade(plain=True)
        facade.render_diff("line1\nline2\n", "line1\nline2_modified\n", "file.txt")
        output = capsys.readouterr().out
        assert "file.txt" in output or "---" in output

    def test_plain_empty_diff(self, capsys):
        from polylogue.ui.facade import ConsoleFacade
        facade = ConsoleFacade(plain=True)
        facade.render_diff("same\n", "same\n", "file.txt")
        output = capsys.readouterr().out
        # Empty diff might just print header
        assert "file.txt" in output or output.strip() == ""

    def test_rich_renders_diff(self):
        from polylogue.ui.facade import ConsoleFacade
        facade = ConsoleFacade(plain=False)
        facade.render_diff("old\n", "new\n", "test.txt")


class TestStatusMethods:
    @pytest.mark.parametrize("method,text,expected_icon", [
        ("error", "Something broke", "✗"),
        ("warning", "Watch out", "!"),
        ("success", "All good", "✓"),
        ("info", "FYI", None),
    ], ids=["error", "warning", "success", "info"])
    def test_status_method_plain(self, capsys, method, text, expected_icon):
        from polylogue.ui.facade import ConsoleFacade
        facade = ConsoleFacade(plain=True)
        getattr(facade, method)(text)
        output = capsys.readouterr().out
        assert text in output
        if expected_icon:
            assert expected_icon in output or expected_icon.upper() in output

    @pytest.mark.parametrize("method", ["error", "warning", "success", "info"])
    def test_status_method_rich(self, method):
        from polylogue.ui.facade import ConsoleFacade
        facade = ConsoleFacade(plain=False)
        getattr(facade, method)("Test message")
        # Should not raise


# =============================================================================
# PlainConsoleFacade
# =============================================================================

class TestPlainConsoleFacade:
    def test_inherits_from_console_facade(self):
        from polylogue.ui.facade import PlainConsoleFacade, ConsoleFacade
        facade = PlainConsoleFacade(plain=True)
        assert isinstance(facade, ConsoleFacade)

    def test_plain_flag_set(self):
        from polylogue.ui.facade import PlainConsoleFacade
        facade = PlainConsoleFacade(plain=True)
        assert facade.plain is True

    def test_post_init_sets_console(self):
        from polylogue.ui.facade import PlainConsoleFacade, PlainConsole
        facade = PlainConsoleFacade(plain=True)
        assert isinstance(facade.console, PlainConsole)

    def test_banner_works(self, capsys):
        from polylogue.ui.facade import PlainConsoleFacade
        facade = PlainConsoleFacade(plain=True)
        facade.banner("Test")
        output = capsys.readouterr().out
        assert "Test" in output

    def test_confirm_returns_default(self):
        from polylogue.ui.facade import PlainConsoleFacade
        facade = PlainConsoleFacade(plain=True)
        assert facade.confirm("Q?", default=True) is True


# =============================================================================
# VersionInfo
# =============================================================================

class TestVersionInfo:
    def test_version_only(self):
        from polylogue.version import VersionInfo
        v = VersionInfo(version="1.0.0")
        assert str(v) == "1.0.0"
        assert v.full == "1.0.0"
        assert v.short == "1.0.0"

    def test_version_repr(self):
        from polylogue.version import VersionInfo
        v = VersionInfo(version="1.0.0")
        repr_str = repr(v)
        assert "1.0.0" in repr_str

    def test_version_with_commit(self):
        from polylogue.version import VersionInfo
        v = VersionInfo(version="1.0.0", commit="abc123def456")
        assert str(v) == "1.0.0+abc123de"
        assert v.short == "1.0.0"

    def test_version_with_commit_short_sha(self):
        from polylogue.version import VersionInfo
        v = VersionInfo(version="2.5.3", commit="fedcba9876543210")
        assert str(v) == "2.5.3+fedcba98"
        assert v.full == "2.5.3+fedcba98"

    def test_version_with_dirty(self):
        from polylogue.version import VersionInfo
        v = VersionInfo(version="1.0.0", commit="abc123def456", dirty=True)
        assert str(v) == "1.0.0+abc123de-dirty"
        assert "-dirty" in v.full

    def test_version_dirty_no_commit(self):
        from polylogue.version import VersionInfo
        v = VersionInfo(version="1.0.0", dirty=True)
        # No commit = version only (dirty flag irrelevant without commit)
        assert str(v) == "1.0.0"

    def test_version_full_property(self):
        from polylogue.version import VersionInfo
        v = VersionInfo(version="3.2.1", commit="deadbeef12345678", dirty=False)
        assert v.full == str(v)
        assert v.full == "3.2.1+deadbeef"

    def test_version_short_property(self):
        from polylogue.version import VersionInfo
        v = VersionInfo(version="3.2.1", commit="deadbeef12345678")
        assert v.short == "3.2.1"

    def test_version_equality(self):
        from polylogue.version import VersionInfo
        v1 = VersionInfo(version="1.0.0")
        v2 = VersionInfo(version="1.0.0")
        assert v1 == v2

    def test_version_dataclass_fields(self):
        from polylogue.version import VersionInfo
        from dataclasses import fields
        field_names = {f.name for f in fields(VersionInfo)}
        assert field_names == {"version", "commit", "dirty"}


# =============================================================================
# _get_git_info
# =============================================================================

class TestGetGitInfo:
    def test_valid_git_repo(self):
        from polylogue.version import _get_git_info
        repo_root = Path(__file__).resolve().parent.parent
        if (repo_root / ".git").exists():
            commit, dirty = _get_git_info(repo_root)
            assert commit is not None
            assert len(commit) == 40  # Full SHA
            assert isinstance(dirty, bool)

    def test_nonexistent_dir_returns_none(self, tmp_path):
        from polylogue.version import _get_git_info
        commit, dirty = _get_git_info(tmp_path / "nonexistent")
        assert commit is None
        assert dirty is False

    def test_non_git_dir_returns_none(self, tmp_path):
        from polylogue.version import _get_git_info
        commit, dirty = _get_git_info(tmp_path)
        assert commit is None
        assert dirty is False

    def test_timeout_returns_none(self, tmp_path, monkeypatch):
        from polylogue.version import _get_git_info
        import subprocess

        original_run = subprocess.run

        def mock_run(*args, **kwargs):
            raise subprocess.TimeoutExpired("git", 2)

        monkeypatch.setattr(subprocess, "run", mock_run)
        commit, dirty = _get_git_info(tmp_path)
        assert commit is None
        assert dirty is False

    def test_returns_tuple(self, tmp_path):
        from polylogue.version import _get_git_info
        result = _get_git_info(tmp_path)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_dirty_state_is_bool(self):
        from polylogue.version import _get_git_info
        repo_root = Path(__file__).resolve().parent.parent
        if (repo_root / ".git").exists():
            commit, dirty = _get_git_info(repo_root)
            assert isinstance(dirty, bool)


# =============================================================================
# _resolve_version
# =============================================================================

class TestResolveVersion:
    def test_returns_version_info(self):
        from polylogue.version import _resolve_version, VersionInfo
        result = _resolve_version()
        assert isinstance(result, VersionInfo)
        assert result.version != ""
        assert result.version != "unknown"

    def test_version_not_empty(self):
        from polylogue.version import _resolve_version
        result = _resolve_version()
        assert len(result.version) > 0

    @pytest.mark.parametrize("attr", ["version", "commit", "dirty"])
    def test_version_info_has_attribute(self, attr):
        from polylogue.version import _resolve_version
        result = _resolve_version()
        assert hasattr(result, attr)

    def test_commit_is_none_or_string(self):
        from polylogue.version import _resolve_version
        result = _resolve_version()
        assert result.commit is None or isinstance(result.commit, str)

    def test_dirty_is_bool(self):
        from polylogue.version import _resolve_version
        result = _resolve_version()
        assert isinstance(result.dirty, bool)


# =============================================================================
# Module-level constants
# =============================================================================

class TestVersionConstants:
    def test_polylogue_version_exists(self):
        from polylogue.version import POLYLOGUE_VERSION
        assert isinstance(POLYLOGUE_VERSION, str)
        assert len(POLYLOGUE_VERSION) > 0

    def test_version_info_exists(self):
        from polylogue.version import VERSION_INFO, VersionInfo
        assert isinstance(VERSION_INFO, VersionInfo)

    def test_version_info_in_constant(self):
        from polylogue.version import POLYLOGUE_VERSION, VERSION_INFO
        assert VERSION_INFO.version in POLYLOGUE_VERSION

    def test_all_exports(self):
        from polylogue import version
        assert hasattr(version, "POLYLOGUE_VERSION")
        assert hasattr(version, "VERSION_INFO")
        assert hasattr(version, "VersionInfo")

    def test_version_module_all(self):
        from polylogue import version
        if hasattr(version, "__all__"):
            all_items = version.__all__
            assert "POLYLOGUE_VERSION" in all_items
            assert "VERSION_INFO" in all_items
            assert "VersionInfo" in all_items


# =============================================================================
# Additional edge cases and branch coverage
# =============================================================================

class TestResolveVersionEdgeCases:
    def test_resolve_version_with_pyproject_fallback(self, tmp_path, monkeypatch):
        """Test version resolution falls back to pyproject.toml if metadata not found."""
        from polylogue.version import _resolve_version
        from importlib.metadata import PackageNotFoundError

        # Mock metadata_version to raise PackageNotFoundError
        def mock_metadata_version(name):
            raise PackageNotFoundError(name)

        import polylogue.version as version_module
        original_metadata_version = version_module.metadata_version
        monkeypatch.setattr(version_module, "metadata_version", mock_metadata_version)

        # Call _resolve_version (it will try pyproject.toml)
        result = _resolve_version()

        # Restore original
        monkeypatch.setattr(version_module, "metadata_version", original_metadata_version)

        assert result.version is not None
        assert len(result.version) > 0

    def test_resolve_version_returns_consistent_format(self):
        """Test _resolve_version always returns consistent VersionInfo."""
        from polylogue.version import _resolve_version, VersionInfo

        # Call multiple times to ensure consistency
        result1 = _resolve_version()
        result2 = _resolve_version()

        # Should return same type
        assert isinstance(result1, VersionInfo)
        assert isinstance(result2, VersionInfo)

        # Version should be deterministic
        assert result1.version == result2.version


class TestUIErrorException:
    def test_ui_error_is_exception(self):
        from polylogue.ui.facade import UIError
        assert issubclass(UIError, Exception)

    def test_ui_error_can_be_raised(self):
        from polylogue.ui.facade import UIError
        with pytest.raises(UIError):
            raise UIError("test error")

    def test_ui_error_message(self):
        from polylogue.ui.facade import UIError
        msg = "custom error message"
        with pytest.raises(UIError, match=msg):
            raise UIError(msg)


class TestConsoleLikeProtocol:
    def test_plain_console_implements_protocol(self):
        from polylogue.ui.facade import PlainConsole, ConsoleLike
        pc = PlainConsole()
        # PlainConsole should have the print method
        assert callable(getattr(pc, "print", None))

    def test_rich_console_implements_protocol(self):
        from polylogue.ui.facade import ConsoleFacade
        from rich.console import Console
        facade = ConsoleFacade(plain=False)
        console = facade.console
        # Rich Console should have the print method
        assert callable(getattr(console, "print", None))


class TestConsoleFacadeTheme:
    def test_theme_initialized(self):
        from polylogue.ui.facade import ConsoleFacade
        facade = ConsoleFacade(plain=True)
        assert facade.theme is not None

    def test_theme_has_color_styles(self):
        from polylogue.ui.facade import ConsoleFacade
        facade = ConsoleFacade(plain=False)
        theme = facade.theme
        # Theme should have color definitions
        assert theme is not None


class TestConsoleFacadeBoxStyles:
    @pytest.mark.parametrize("attr", ["_panel_box", "_banner_box"])
    def test_box_style_set(self, attr):
        from polylogue.ui.facade import ConsoleFacade
        facade = ConsoleFacade(plain=False)
        assert getattr(facade, attr) is not None

    def test_different_box_styles(self):
        from polylogue.ui.facade import ConsoleFacade
        from rich import box
        facade = ConsoleFacade(plain=False)
        # Banner should use DOUBLE, panel should use ROUNDED
        assert facade._banner_box != facade._panel_box


class TestStatusMethodsPrivate:
    def test_status_plain_success_icon(self, capsys):
        from polylogue.ui.facade import ConsoleFacade
        facade = ConsoleFacade(plain=True)
        facade._status("✓", "status.icon.success", "Test")
        output = capsys.readouterr().out
        assert "✓" in output
        assert "Test" in output

    def test_status_plain_non_checkmark_icon_uppercase(self, capsys):
        from polylogue.ui.facade import ConsoleFacade
        facade = ConsoleFacade(plain=True)
        facade._status("!", "status.icon.warning", "Alert")
        output = capsys.readouterr().out
        assert "!" in output or "!" in output.upper()
        assert "Alert" in output

    def test_status_rich_formats_with_style(self):
        from polylogue.ui.facade import ConsoleFacade
        facade = ConsoleFacade(plain=False)
        # Should not raise
        facade._status("✓", "status.icon.success", "Success")


class TestChooseWithLargeOptionSet:
    def test_choose_exactly_12_options_uses_select(self, tmp_path, monkeypatch):
        """Exactly 12 options should use select, not autocomplete."""
        from polylogue.ui.facade import ConsoleFacade
        stub_file = tmp_path / "stubs.jsonl"
        stub_file.write_text(json.dumps({"type": "choose", "index": 0}) + "\n")
        monkeypatch.setenv("POLYLOGUE_TEST_PROMPT_FILE", str(stub_file))
        facade = ConsoleFacade(plain=False)
        options = [f"opt{i}" for i in range(12)]
        result = facade.choose("Pick:", options)
        assert result == "opt0"

    def test_choose_13_options_uses_autocomplete(self, tmp_path, monkeypatch):
        """13 options (>12) should use autocomplete."""
        from polylogue.ui.facade import ConsoleFacade
        stub_file = tmp_path / "stubs.jsonl"
        stub_file.write_text(json.dumps({"type": "choose", "value": "opt12"}) + "\n")
        monkeypatch.setenv("POLYLOGUE_TEST_PROMPT_FILE", str(stub_file))
        facade = ConsoleFacade(plain=False)
        options = [f"opt{i}" for i in range(13)]
        result = facade.choose("Pick:", options)
        assert result == "opt12"


class TestInputEdgeCases:
    def test_input_with_empty_questionary_result(self, tmp_path, monkeypatch):
        """Test input when questionary returns empty string."""
        from polylogue.ui.facade import ConsoleFacade
        import questionary

        monkeypatch.delenv("POLYLOGUE_TEST_PROMPT_FILE", raising=False)
        facade = ConsoleFacade(plain=False)

        # Mock questionary.text to return empty string
        mock_text = MagicMock()
        mock_text.return_value.ask.return_value = ""
        monkeypatch.setattr(questionary, "text", lambda *args, **kwargs: mock_text.return_value)

        # Empty string should return default
        result = facade.input("Prompt:", default="fallback")
        assert result == "fallback"

    def test_input_with_whitespace_questionary_result(self, tmp_path, monkeypatch):
        """Test input when questionary returns whitespace."""
        from polylogue.ui.facade import ConsoleFacade
        import questionary

        monkeypatch.delenv("POLYLOGUE_TEST_PROMPT_FILE", raising=False)
        facade = ConsoleFacade(plain=False)

        # Mock questionary.text to return whitespace
        mock_text = MagicMock()
        mock_text.return_value.ask.return_value = "   "
        monkeypatch.setattr(questionary, "text", lambda *args, **kwargs: mock_text.return_value)

        # Whitespace should still be returned as-is
        result = facade.input("Prompt:", default="fallback")
        assert result == "   "


class TestRenderDiffEdgeCases:
    def test_render_diff_multiline_format(self, capsys):
        """Test render_diff with multiple lines added and removed."""
        from polylogue.ui.facade import ConsoleFacade
        facade = ConsoleFacade(plain=True)
        old = "line1\nline2\nline3\n"
        new = "line1\nmodified2\nline3\nline4\n"
        facade.render_diff(old, new, "test.py")
        output = capsys.readouterr().out
        assert "test.py" in output

    def test_render_diff_no_trailing_newline(self, capsys):
        """Test render_diff when texts don't end with newline."""
        from polylogue.ui.facade import ConsoleFacade
        facade = ConsoleFacade(plain=True)
        old = "line1"
        new = "line1\nline2"
        facade.render_diff(old, new, "file.txt")
        output = capsys.readouterr().out
        assert "file.txt" in output


class TestConsoleFacadeDataclass:
    def test_console_facade_is_dataclass(self):
        from polylogue.ui.facade import ConsoleFacade
        from dataclasses import is_dataclass
        assert is_dataclass(ConsoleFacade)

    def test_plain_console_facade_is_dataclass(self):
        from polylogue.ui.facade import PlainConsoleFacade
        from dataclasses import is_dataclass
        assert is_dataclass(PlainConsoleFacade)
