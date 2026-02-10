"""Consolidated UI tests (facade, UX flows, wrapper API)."""

from __future__ import annotations

import json
from dataclasses import fields
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from rich.console import Console

from polylogue.ui import UI
from polylogue.ui.facade import (
    ConsoleFacade,
    PlainConsole,
    PlainConsoleFacade,
    UIError,
    create_console_facade,
)
from polylogue.version import VersionInfo, _get_git_info, _resolve_version


# --- Merged from test_ui_facade_coverage.py ---


# =============================================================================
# PlainConsole
# =============================================================================

class TestPlainConsole:
    def test_print_basic(self, capsys):
        pc = PlainConsole()
        pc.print("hello", "world")
        assert "hello world" in capsys.readouterr().out

    @pytest.mark.parametrize("input_text,expected_stripped,unexpected", [
        ("[bold]Important[/bold]", "Important", "[bold]"),
        ("[green]Success[/green]", "Success", "[green]"),
        ("[#d97757]colored[/#d97757]", "colored", "[#d97757]"),
    ])
    def test_print_strips_markup(self, capsys, input_text, expected_stripped, unexpected):
        pc = PlainConsole()
        pc.print(input_text)
        captured = capsys.readouterr().out
        assert expected_stripped in captured
        assert unexpected not in captured

    def test_print_ignores_extra_kwargs(self, capsys):
        pc = PlainConsole()
        pc.print("text", sep="|", end="!\n")
        # Should print "text" without respecting sep/end kwargs
        assert "text" in capsys.readouterr().out

    def test_print_malformed_markup_fallback(self, capsys):
        pc = PlainConsole()
        pc.print("[unclosed markup")
        captured = capsys.readouterr().out
        assert "[unclosed markup" in captured

    def test_print_init_accepts_any_args(self):
        # Should not raise despite extra args
        pc = PlainConsole("arg1", "arg2", kwarg="value")
        assert pc is not None


# =============================================================================
# ConsoleFacade creation
# =============================================================================

class TestCreateConsoleFacade:
    def test_plain_creates_plain_facade(self):
        facade = create_console_facade(plain=True)
        assert isinstance(facade, PlainConsoleFacade)
        assert facade.plain is True

    def test_rich_creates_console_facade(self):
        facade = create_console_facade(plain=False)
        assert isinstance(facade, ConsoleFacade)
        assert facade.plain is False
        assert not isinstance(facade, PlainConsoleFacade)


# =============================================================================
# Prompt stub system
# =============================================================================

class TestPromptStubs:
    def test_no_env_var_returns_empty(self, monkeypatch):
        monkeypatch.delenv("POLYLOGUE_TEST_PROMPT_FILE", raising=False)
        facade = ConsoleFacade(plain=True)
        assert len(facade._prompt_responses) == 0

    def test_loads_single_jsonl_stub(self, tmp_path, monkeypatch):
        stub_file = tmp_path / "stubs.jsonl"
        stub_file.write_text(json.dumps({"type": "confirm", "value": True}) + "\n")
        monkeypatch.setenv("POLYLOGUE_TEST_PROMPT_FILE", str(stub_file))
        facade = ConsoleFacade(plain=True)
        assert len(facade._prompt_responses) == 1

    def test_loads_multiple_jsonl_stubs(self, tmp_path, monkeypatch):
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
        stub_file = tmp_path / "bad.jsonl"
        stub_file.write_text("not json\n")
        monkeypatch.setenv("POLYLOGUE_TEST_PROMPT_FILE", str(stub_file))
        with pytest.raises(UIError, match="Invalid prompt stub"):
            ConsoleFacade(plain=True)

    def test_empty_lines_skipped(self, tmp_path, monkeypatch):
        stub_file = tmp_path / "stubs.jsonl"
        stub_file.write_text("\n\n" + json.dumps({"type": "confirm"}) + "\n\n")
        monkeypatch.setenv("POLYLOGUE_TEST_PROMPT_FILE", str(stub_file))
        facade = ConsoleFacade(plain=True)
        assert len(facade._prompt_responses) == 1

    def test_whitespace_only_lines_skipped(self, tmp_path, monkeypatch):
        stub_file = tmp_path / "stubs.jsonl"
        stub_file.write_text("   \n\t\n" + json.dumps({"type": "input"}) + "\n  ")
        monkeypatch.setenv("POLYLOGUE_TEST_PROMPT_FILE", str(stub_file))
        facade = ConsoleFacade(plain=True)
        assert len(facade._prompt_responses) == 1

    def test_pop_type_mismatch_raises(self, tmp_path, monkeypatch):
        stub_file = tmp_path / "stubs.jsonl"
        stub_file.write_text(json.dumps({"type": "confirm", "value": True}) + "\n")
        monkeypatch.setenv("POLYLOGUE_TEST_PROMPT_FILE", str(stub_file))
        facade = ConsoleFacade(plain=True)
        with pytest.raises(UIError, match="expected 'confirm' but got 'choose'"):
            facade._pop_prompt_response("choose")

    def test_pop_no_type_matches_any(self, tmp_path, monkeypatch):
        stub_file = tmp_path / "stubs.jsonl"
        stub_file.write_text(json.dumps({"value": True}) + "\n")
        monkeypatch.setenv("POLYLOGUE_TEST_PROMPT_FILE", str(stub_file))
        facade = ConsoleFacade(plain=True)
        result = facade._pop_prompt_response("anything")
        assert result == {"value": True}

    def test_pop_empty_queue_returns_none(self, monkeypatch):
        monkeypatch.delenv("POLYLOGUE_TEST_PROMPT_FILE", raising=False)
        facade = ConsoleFacade(plain=True)
        result = facade._pop_prompt_response("confirm")
        assert result is None


# =============================================================================
# confirm
# =============================================================================

class TestConfirm:
    def test_plain_returns_default_true(self):
        facade = ConsoleFacade(plain=True)
        assert facade.confirm("Continue?", default=True) is True

    def test_plain_returns_default_false(self):
        facade = ConsoleFacade(plain=True)
        assert facade.confirm("Continue?", default=False) is False

    def test_plain_ignores_prompt(self):
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
        stub_file = tmp_path / "stubs.jsonl"
        stub_file.write_text(json.dumps({"type": "confirm", "value": stub_value}) + "\n")
        monkeypatch.setenv("POLYLOGUE_TEST_PROMPT_FILE", str(stub_file))
        facade = ConsoleFacade(plain=False)
        assert facade.confirm("Continue?") is expected

    def test_stub_use_default_true(self, tmp_path, monkeypatch):
        stub_file = tmp_path / "stubs.jsonl"
        stub_file.write_text(json.dumps({"type": "confirm", "use_default": True}) + "\n")
        monkeypatch.setenv("POLYLOGUE_TEST_PROMPT_FILE", str(stub_file))
        facade = ConsoleFacade(plain=False)
        assert facade.confirm("Continue?", default=True) is True

    def test_stub_use_default_false(self, tmp_path, monkeypatch):
        stub_file = tmp_path / "stubs.jsonl"
        stub_file.write_text(json.dumps({"type": "confirm", "use_default": True}) + "\n")
        monkeypatch.setenv("POLYLOGUE_TEST_PROMPT_FILE", str(stub_file))
        facade = ConsoleFacade(plain=False)
        assert facade.confirm("Continue?", default=False) is False

    def test_stub_invalid_string_returns_default(self, tmp_path, monkeypatch):
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
        facade = ConsoleFacade(plain=False)
        assert facade.choose("Pick:", []) is None

    def test_empty_options_plain(self):
        facade = ConsoleFacade(plain=True)
        assert facade.choose("Pick:", []) is None

    def test_plain_returns_none(self):
        facade = ConsoleFacade(plain=True)
        assert facade.choose("Pick:", ["a", "b"]) is None

    def test_plain_ignores_prompt(self):
        facade = ConsoleFacade(plain=True)
        assert facade.choose("anything", ["x", "y", "z"]) is None

    def test_stub_value_exact_match(self, tmp_path, monkeypatch):
        stub_file = tmp_path / "stubs.jsonl"
        stub_file.write_text(json.dumps({"type": "choose", "value": "option2"}) + "\n")
        monkeypatch.setenv("POLYLOGUE_TEST_PROMPT_FILE", str(stub_file))
        facade = ConsoleFacade(plain=False)
        result = facade.choose("Pick:", ["option1", "option2", "option3"])
        assert result == "option2"

    def test_stub_value_not_in_options(self, tmp_path, monkeypatch):
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
        stub_file = tmp_path / "stubs.jsonl"
        stub_file.write_text(json.dumps({"type": "choose", "index": index}) + "\n")
        monkeypatch.setenv("POLYLOGUE_TEST_PROMPT_FILE", str(stub_file))
        facade = ConsoleFacade(plain=False)
        result = facade.choose("Pick:", options)
        assert result == expected

    @pytest.mark.parametrize("invalid_index", [99, -1, "not_a_number"], ids=["out_of_bounds", "negative", "string_invalid"])
    def test_stub_index_invalid(self, tmp_path, monkeypatch, invalid_index):
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
        stub_file = tmp_path / "stubs.jsonl"
        stub_file.write_text(json.dumps({"type": "choose", "use_default": True}) + "\n")
        monkeypatch.setenv("POLYLOGUE_TEST_PROMPT_FILE", str(stub_file))
        facade = ConsoleFacade(plain=False)
        result = facade.choose("Pick:", ["first", "second"])
        assert result == "first"

    def test_many_options_uses_autocomplete(self, tmp_path, monkeypatch):
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
        facade = ConsoleFacade(plain=True)
        assert facade.input("Name:", default="anon") == "anon"

    def test_plain_no_default_returns_none(self):
        facade = ConsoleFacade(plain=True)
        assert facade.input("Name:") is None

    def test_plain_ignores_prompt(self):
        facade = ConsoleFacade(plain=True)
        assert facade.input("anything", default="default") == "default"

    def test_stub_value_string(self, tmp_path, monkeypatch):
        stub_file = tmp_path / "stubs.jsonl"
        stub_file.write_text(json.dumps({"type": "input", "value": "typed"}) + "\n")
        monkeypatch.setenv("POLYLOGUE_TEST_PROMPT_FILE", str(stub_file))
        facade = ConsoleFacade(plain=False)
        assert facade.input("Name:") == "typed"

    def test_stub_value_number_converted_to_string(self, tmp_path, monkeypatch):
        stub_file = tmp_path / "stubs.jsonl"
        stub_file.write_text(json.dumps({"type": "input", "value": 42}) + "\n")
        monkeypatch.setenv("POLYLOGUE_TEST_PROMPT_FILE", str(stub_file))
        facade = ConsoleFacade(plain=False)
        assert facade.input("Number:") == "42"

    def test_stub_use_default(self, tmp_path, monkeypatch):
        stub_file = tmp_path / "stubs.jsonl"
        stub_file.write_text(json.dumps({"type": "input", "use_default": True}) + "\n")
        monkeypatch.setenv("POLYLOGUE_TEST_PROMPT_FILE", str(stub_file))
        facade = ConsoleFacade(plain=False)
        assert facade.input("Name:", default="fallback") == "fallback"

    def test_stub_use_default_no_default(self, tmp_path, monkeypatch):
        stub_file = tmp_path / "stubs.jsonl"
        stub_file.write_text(json.dumps({"type": "input", "use_default": True}) + "\n")
        monkeypatch.setenv("POLYLOGUE_TEST_PROMPT_FILE", str(stub_file))
        facade = ConsoleFacade(plain=False)
        assert facade.input("Name:") is None

    def test_stub_value_none(self, tmp_path, monkeypatch):
        stub_file = tmp_path / "stubs.jsonl"
        stub_file.write_text(json.dumps({"type": "input", "value": None}) + "\n")
        monkeypatch.setenv("POLYLOGUE_TEST_PROMPT_FILE", str(stub_file))
        facade = ConsoleFacade(plain=False)
        assert facade.input("Name:") is None

    def test_stub_value_empty_string(self, tmp_path, monkeypatch):
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
        facade = ConsoleFacade(plain=True)
        facade.banner("Title", "Subtitle")
        output = capsys.readouterr().out
        assert "Title" in output
        assert "Subtitle" in output

    def test_plain_banner_no_subtitle(self, capsys):
        facade = ConsoleFacade(plain=True)
        facade.banner("Title")
        output = capsys.readouterr().out
        assert "Title" in output
        assert "==" in output

    def test_plain_banner_formatting(self, capsys):
        facade = ConsoleFacade(plain=True)
        facade.banner("Test")
        output = capsys.readouterr().out
        assert "== Test ==" in output

    def test_rich_banner(self):
        facade = ConsoleFacade(plain=False)
        facade.banner("Title", "Subtitle")
        # Should not raise

    def test_rich_banner_no_subtitle(self):
        facade = ConsoleFacade(plain=False)
        facade.banner("Title")
        # Should not raise


class TestSummary:
    def test_plain_summary_single_line(self, capsys):
        facade = ConsoleFacade(plain=True)
        facade.summary("Stats", ["Line 1"])
        output = capsys.readouterr().out
        assert "Stats" in output
        assert "Line 1" in output
        assert "--" in output

    def test_plain_summary_multiple_lines(self, capsys):
        facade = ConsoleFacade(plain=True)
        facade.summary("Stats", ["Line 1", "Line 2", "Line 3"])
        output = capsys.readouterr().out
        assert "Stats" in output
        assert "Line 1" in output
        assert "Line 2" in output
        assert "Line 3" in output

    def test_plain_summary_empty_lines(self, capsys):
        facade = ConsoleFacade(plain=True)
        facade.summary("Stats", [])
        output = capsys.readouterr().out
        assert "Stats" in output

    def test_plain_summary_with_markup(self, capsys):
        facade = ConsoleFacade(plain=True)
        facade.summary("Result", ["[green]✓[/green] Success", "[red]✗[/red] Failed"])
        output = capsys.readouterr().out
        assert "Success" in output
        assert "Failed" in output

    def test_rich_summary(self):
        facade = ConsoleFacade(plain=False)
        facade.summary("Stats", ["Line 1", "Line 2"])


class TestRenderMarkdown:
    def test_plain_prints_raw(self, capsys):
        facade = ConsoleFacade(plain=True)
        facade.render_markdown("# Hello\n\nWorld")
        output = capsys.readouterr().out
        assert "Hello" in output
        assert "World" in output

    def test_plain_renders_multiline(self, capsys):
        facade = ConsoleFacade(plain=True)
        content = "# Title\n\n**Bold** text\n\n- List\n- Items"
        facade.render_markdown(content)
        output = capsys.readouterr().out
        assert "Title" in output
        assert "Bold" in output

    def test_rich_renders(self):
        facade = ConsoleFacade(plain=False)
        facade.render_markdown("# Hello\n\n**World**")


class TestRenderCode:
    def test_plain_prints_raw(self, capsys):
        facade = ConsoleFacade(plain=True)
        facade.render_code("print('hello')")
        output = capsys.readouterr().out
        assert "print" in output
        assert "hello" in output

    def test_plain_default_language(self, capsys):
        facade = ConsoleFacade(plain=True)
        facade.render_code("x = 1")
        output = capsys.readouterr().out
        assert "x = 1" in output

    def test_plain_custom_language(self, capsys):
        facade = ConsoleFacade(plain=True)
        facade.render_code("SELECT * FROM users;", "sql")
        output = capsys.readouterr().out
        assert "SELECT" in output

    def test_rich_renders(self):
        facade = ConsoleFacade(plain=False)
        facade.render_code("x = 1", "python")

    def test_rich_renders_javascript(self):
        facade = ConsoleFacade(plain=False)
        facade.render_code("console.log('test');", "javascript")


class TestRenderDiff:
    def test_plain_renders_diff(self, capsys):
        facade = ConsoleFacade(plain=True)
        facade.render_diff("old\n", "new\n", "test.txt")
        output = capsys.readouterr().out
        assert "test.txt" in output or "---" in output or "+++" in output

    def test_plain_renders_unified_diff(self, capsys):
        facade = ConsoleFacade(plain=True)
        facade.render_diff("line1\nline2\n", "line1\nline2_modified\n", "file.txt")
        output = capsys.readouterr().out
        assert "file.txt" in output or "---" in output

    def test_plain_empty_diff(self, capsys):
        facade = ConsoleFacade(plain=True)
        facade.render_diff("same\n", "same\n", "file.txt")
        output = capsys.readouterr().out
        # Empty diff might just print header
        assert "file.txt" in output or output.strip() == ""

    def test_rich_renders_diff(self):
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
        facade = ConsoleFacade(plain=True)
        getattr(facade, method)(text)
        output = capsys.readouterr().out
        assert text in output
        if expected_icon:
            assert expected_icon in output or expected_icon.upper() in output

    @pytest.mark.parametrize("method", ["error", "warning", "success", "info"])
    def test_status_method_rich(self, method):
        facade = ConsoleFacade(plain=False)
        getattr(facade, method)("Test message")
        # Should not raise


# =============================================================================
# PlainConsoleFacade
# =============================================================================

class TestPlainConsoleFacade:
    def test_inherits_from_console_facade(self):
        facade = PlainConsoleFacade(plain=True)
        assert isinstance(facade, ConsoleFacade)

    def test_plain_flag_set(self):
        facade = PlainConsoleFacade(plain=True)
        assert facade.plain is True

    def test_post_init_sets_console(self):
        facade = PlainConsoleFacade(plain=True)
        assert isinstance(facade.console, PlainConsole)

    def test_banner_works(self, capsys):
        facade = PlainConsoleFacade(plain=True)
        facade.banner("Test")
        output = capsys.readouterr().out
        assert "Test" in output

    def test_confirm_returns_default(self):
        facade = PlainConsoleFacade(plain=True)
        assert facade.confirm("Q?", default=True) is True


# =============================================================================
# VersionInfo
# =============================================================================

class TestVersionInfo:
    def test_version_only(self):
        v = VersionInfo(version="1.0.0")
        assert str(v) == "1.0.0"
        assert v.full == "1.0.0"
        assert v.short == "1.0.0"

    def test_version_repr(self):
        v = VersionInfo(version="1.0.0")
        repr_str = repr(v)
        assert "1.0.0" in repr_str

    def test_version_with_commit(self):
        v = VersionInfo(version="1.0.0", commit="abc123def456")
        assert str(v) == "1.0.0+abc123de"
        assert v.short == "1.0.0"

    def test_version_with_commit_short_sha(self):
        v = VersionInfo(version="2.5.3", commit="fedcba9876543210")
        assert str(v) == "2.5.3+fedcba98"
        assert v.full == "2.5.3+fedcba98"

    def test_version_with_dirty(self):
        v = VersionInfo(version="1.0.0", commit="abc123def456", dirty=True)
        assert str(v) == "1.0.0+abc123de-dirty"
        assert "-dirty" in v.full

    def test_version_dirty_no_commit(self):
        v = VersionInfo(version="1.0.0", dirty=True)
        # No commit = version only (dirty flag irrelevant without commit)
        assert str(v) == "1.0.0"

    def test_version_full_property(self):
        v = VersionInfo(version="3.2.1", commit="deadbeef12345678", dirty=False)
        assert v.full == str(v)
        assert v.full == "3.2.1+deadbeef"

    def test_version_short_property(self):
        v = VersionInfo(version="3.2.1", commit="deadbeef12345678")
        assert v.short == "3.2.1"

    def test_version_equality(self):
        v1 = VersionInfo(version="1.0.0")
        v2 = VersionInfo(version="1.0.0")
        assert v1 == v2

    def test_version_dataclass_fields(self):
        field_names = {f.name for f in fields(VersionInfo)}
        assert field_names == {"version", "commit", "dirty"}


# =============================================================================
# _get_git_info
# =============================================================================

class TestGetGitInfo:
    def test_valid_git_repo(self):
        repo_root = Path(__file__).resolve().parent.parent
        if (repo_root / ".git").exists():
            commit, dirty = _get_git_info(repo_root)
            assert commit is not None
            assert len(commit) == 40  # Full SHA
            assert isinstance(dirty, bool)

    def test_nonexistent_dir_returns_none(self, tmp_path):
        commit, dirty = _get_git_info(tmp_path / "nonexistent")
        assert commit is None
        assert dirty is False

    def test_non_git_dir_returns_none(self, tmp_path):
        commit, dirty = _get_git_info(tmp_path)
        assert commit is None
        assert dirty is False

    def test_timeout_returns_none(self, tmp_path, monkeypatch):
        import subprocess

        original_run = subprocess.run

        def mock_run(*args, **kwargs):
            raise subprocess.TimeoutExpired("git", 2)

        monkeypatch.setattr(subprocess, "run", mock_run)
        commit, dirty = _get_git_info(tmp_path)
        assert commit is None
        assert dirty is False

    def test_returns_tuple(self, tmp_path):
        result = _get_git_info(tmp_path)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_dirty_state_is_bool(self):
        repo_root = Path(__file__).resolve().parent.parent
        if (repo_root / ".git").exists():
            commit, dirty = _get_git_info(repo_root)
            assert isinstance(dirty, bool)


# =============================================================================
# _resolve_version
# =============================================================================

class TestResolveVersion:
    def test_returns_version_info(self):
        result = _resolve_version()
        assert isinstance(result, VersionInfo)
        assert result.version != ""
        assert result.version != "unknown"

    def test_version_not_empty(self):
        result = _resolve_version()
        assert len(result.version) > 0

    @pytest.mark.parametrize("attr", ["version", "commit", "dirty"])
    def test_version_info_has_attribute(self, attr):
        result = _resolve_version()
        assert hasattr(result, attr)

    def test_commit_is_none_or_string(self):
        result = _resolve_version()
        assert result.commit is None or isinstance(result.commit, str)

    def test_dirty_is_bool(self):
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
        assert issubclass(UIError, Exception)

    def test_ui_error_can_be_raised(self):
        with pytest.raises(UIError):
            raise UIError("test error")

    def test_ui_error_message(self):
        msg = "custom error message"
        with pytest.raises(UIError, match=msg):
            raise UIError(msg)


class TestConsoleLikeProtocol:
    def test_plain_console_implements_protocol(self):
        pc = PlainConsole()
        # PlainConsole should have the print method
        assert callable(getattr(pc, "print", None))

    def test_rich_console_implements_protocol(self):
        facade = ConsoleFacade(plain=False)
        console = facade.console
        # Rich Console should have the print method
        assert callable(getattr(console, "print", None))


class TestConsoleFacadeTheme:
    def test_theme_initialized(self):
        facade = ConsoleFacade(plain=True)
        assert facade.theme is not None

    def test_theme_has_color_styles(self):
        facade = ConsoleFacade(plain=False)
        theme = facade.theme
        # Theme should have color definitions
        assert theme is not None


class TestConsoleFacadeBoxStyles:
    @pytest.mark.parametrize("attr", ["_panel_box", "_banner_box"])
    def test_box_style_set(self, attr):
        facade = ConsoleFacade(plain=False)
        assert getattr(facade, attr) is not None

    def test_different_box_styles(self):
        from rich import box
        facade = ConsoleFacade(plain=False)
        # Banner should use DOUBLE, panel should use ROUNDED
        assert facade._banner_box != facade._panel_box


class TestStatusMethodsPrivate:
    def test_status_plain_success_icon(self, capsys):
        facade = ConsoleFacade(plain=True)
        facade._status("✓", "status.icon.success", "Test")
        output = capsys.readouterr().out
        assert "✓" in output
        assert "Test" in output

    def test_status_plain_non_checkmark_icon_uppercase(self, capsys):
        facade = ConsoleFacade(plain=True)
        facade._status("!", "status.icon.warning", "Alert")
        output = capsys.readouterr().out
        assert "!" in output or "!" in output.upper()
        assert "Alert" in output

    def test_status_rich_formats_with_style(self):
        facade = ConsoleFacade(plain=False)
        # Should not raise
        facade._status("✓", "status.icon.success", "Success")


class TestChooseWithLargeOptionSet:
    def test_choose_exactly_12_options_uses_select(self, tmp_path, monkeypatch):
        """Exactly 12 options should use select, not autocomplete."""
        stub_file = tmp_path / "stubs.jsonl"
        stub_file.write_text(json.dumps({"type": "choose", "index": 0}) + "\n")
        monkeypatch.setenv("POLYLOGUE_TEST_PROMPT_FILE", str(stub_file))
        facade = ConsoleFacade(plain=False)
        options = [f"opt{i}" for i in range(12)]
        result = facade.choose("Pick:", options)
        assert result == "opt0"

    def test_choose_13_options_uses_autocomplete(self, tmp_path, monkeypatch):
        """13 options (>12) should use autocomplete."""
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
        facade = ConsoleFacade(plain=True)
        old = "line1\nline2\nline3\n"
        new = "line1\nmodified2\nline3\nline4\n"
        facade.render_diff(old, new, "test.py")
        output = capsys.readouterr().out
        assert "test.py" in output

    def test_render_diff_no_trailing_newline(self, capsys):
        """Test render_diff when texts don't end with newline."""
        facade = ConsoleFacade(plain=True)
        old = "line1"
        new = "line1\nline2"
        facade.render_diff(old, new, "file.txt")
        output = capsys.readouterr().out
        assert "file.txt" in output


class TestConsoleFacadeDataclass:
    def test_console_facade_is_dataclass(self):
        from dataclasses import is_dataclass
        assert is_dataclass(ConsoleFacade)

    def test_plain_console_facade_is_dataclass(self):
        from dataclasses import is_dataclass
        assert is_dataclass(PlainConsoleFacade)


# --- Merged from test_ui_ux.py ---


@pytest.fixture
def mock_prompt_file(tmp_path, monkeypatch):
    """Setup a mock prompt response file."""
    prompt_file = tmp_path / "prompts.jsonl"
    monkeypatch.setenv("POLYLOGUE_TEST_PROMPT_FILE", str(prompt_file))
    return prompt_file


@pytest.fixture
def console_facade():
    """Create a real ConsoleFacade with captured output."""
    facade = create_console_facade(plain=False)
    # Replace console with a capture console
    facade.console = Console(force_terminal=True, no_color=False, width=80)
    # We'll use capture context in tests
    return facade


class TestConsoleFacadeInteractions:
    """Test interactive prompts using file-based mocking."""

    def test_confirm_true(self, mock_prompt_file):
        """Test confirm() returns True from mock."""
        mock_prompt_file.write_text(json.dumps({"type": "confirm", "value": True}) + "\n")

        facade = create_console_facade(plain=False)
        assert facade.confirm("Continue?") is True

    def test_confirm_false(self, mock_prompt_file):
        """Test confirm() returns False from mock."""
        mock_prompt_file.write_text(json.dumps({"type": "confirm", "value": False}) + "\n")

        facade = create_console_facade(plain=False)
        assert facade.confirm("Continue?") is False

    def test_confirm_default_override(self, mock_prompt_file):
        """Test confirm() uses default from mock instruction."""
        mock_prompt_file.write_text(json.dumps({"type": "confirm", "use_default": True}) + "\n")

        facade = create_console_facade(plain=False)
        assert facade.confirm("Continue?", default=True) is True

        # Reset and test default=False
        mock_prompt_file.write_text(json.dumps({"type": "confirm", "use_default": True}) + "\n")
        facade = create_console_facade(plain=False)
        assert facade.confirm("Continue?", default=False) is False

    def test_input_value(self, mock_prompt_file):
        """Test input() returns mocked string."""
        mock_prompt_file.write_text(json.dumps({"type": "input", "value": "test_input"}) + "\n")

        facade = create_console_facade(plain=False)
        assert facade.input("Name?") == "test_input"

    def test_choose_value(self, mock_prompt_file):
        """Test choose() selects by value."""
        mock_prompt_file.write_text(json.dumps({"type": "choose", "value": "Option B"}) + "\n")

        facade = create_console_facade(plain=False)
        result = facade.choose("Pick one", ["Option A", "Option B", "Option C"])
        assert result == "Option B"

    def test_choose_index(self, mock_prompt_file):
        """Test choose() selects by index."""
        mock_prompt_file.write_text(json.dumps({"type": "choose", "index": 2}) + "\n")

        facade = create_console_facade(plain=False)
        result = facade.choose("Pick one", ["Option A", "Option B", "Option C"])
        assert result == "Option C"

    def test_interaction_sequence(self, mock_prompt_file):
        """Test a sequence of different interactions."""
        lines = [
            json.dumps({"type": "input", "value": "User"}),
            json.dumps({"type": "confirm", "value": True}),
            json.dumps({"type": "choose", "value": "Red"}),
        ]
        mock_prompt_file.write_text("\n".join(lines) + "\n")

        facade = create_console_facade(plain=False)

        assert facade.input("Name?") == "User"
        assert facade.confirm("Save?") is True
        assert facade.choose("Color?", ["Red", "Blue"]) == "Red"

    def test_mismatched_prompt_type_raises_error(self, mock_prompt_file):
        """Test that mismatching prompt type raises UIError."""
        mock_prompt_file.write_text(json.dumps({"type": "input", "value": "User"}) + "\n")

        facade = create_console_facade(plain=False)

        with pytest.raises(UIError, match="expected 'input' but got 'confirm'"):
            facade.confirm("Save?")


class TestConsoleFacadeRendering:
    """Test rich rendering methods."""

    def test_banner_render(self):
        """Test banner rendering."""
        facade = create_console_facade(plain=False)
        with facade.console.capture() as capture:
            facade.banner("Welcome", "To Mission Control")

        output = capture.get()
        assert "Welcome" in output
        assert "To Mission Control" in output
        # Border characters check (approximate)
        assert "◈" in output or "mission" in output.lower()

    def test_summary_render(self):
        """Test list summary rendering."""
        facade = create_console_facade(plain=False)
        items = ["Item 1", "[red]Item 2[/red]"]
        with facade.console.capture() as capture:
            facade.summary("Checklist", items)

        output = capture.get()
        assert "Checklist" in output
        assert "Item 1" in output
        assert "Item 2" in output

    def test_render_diff(self):
        """Test diff rendering."""
        facade = create_console_facade(plain=False)

        # Mock pager to avoid actual paging which interferes with capture
        facade.console.pager = MagicMock()
        facade.console.pager.return_value.__enter__ = MagicMock()
        facade.console.pager.return_value.__exit__ = MagicMock()

        old = "line 1\nline 2"
        new = "line 1\nline 3"

        with facade.console.capture() as capture:
            facade.render_diff(old, new, filename="test.txt")

        output = capture.get()
        assert "line 2" in output
        assert "line 3" in output
        # Check for diff symbols? Color codes make exact matching hard,
        # but content should be there.

    def test_status_messages(self):
        """Test success/warning/error messages."""
        facade = create_console_facade(plain=False)
        with facade.console.capture() as capture:
            facade.success("Good job")
            facade.warning("Be careful")
            facade.error("Oh no")
            facade.info("FYI")

        output = capture.get()
        assert "Good job" in output
        assert "Be careful" in output
        assert "Oh no" in output
        assert "FYI" in output


class TestPlainFacade:
    """Test plain mode fallback."""

    def test_plain_interactions_defaults(self):
        """Plain mode should return defaults without prompting."""
        facade = create_console_facade(plain=True)
        assert isinstance(facade, PlainConsoleFacade)

        assert facade.confirm("Ctx?", default=True) is True
        assert facade.confirm("Ctx?", default=False) is False
        assert facade.input("Input?", default="Default") == "Default"
        assert facade.input("Input?", default=None) is None
        assert facade.choose("Pick", ["A", "B"]) is None  # Defaults to None in code? Or None if no interaction?
        # Checked code: choose returns None in plain mode.

    def test_plain_rendering(self, capsys):
        """Plain mode should print simple text."""
        facade = create_console_facade(plain=True)

        facade.banner("Title", "Subtitle")
        captured = capsys.readouterr()
        assert "== Title ==" in captured.out
        assert "Subtitle" in captured.out

        facade.success("Done")
        captured = capsys.readouterr()
        assert "✓ Done" in captured.out


# --- Merged from test_ui_wrapper.py ---


@pytest.fixture
def mock_facade():
    with patch("polylogue.ui.create_console_facade") as mock_create:
        facade = MagicMock(spec=ConsoleFacade)
        # MagicMock spec doesn't include fields that are initialized in __post_init__ or field(init=False)
        # unless we explicitly set them on the instance.
        console_mock = MagicMock()
        facade.console = console_mock
        facade.plain = False
        mock_create.return_value = facade
        yield facade


def test_ui_init_success(mock_facade):
    ui = UI(plain=False)
    assert ui._facade == mock_facade
    assert ui.plain is False
    assert ui.console == mock_facade.console


def test_ui_init_failure():
    with patch("polylogue.ui.create_console_facade", side_effect=RuntimeError("Test error")):
        with pytest.raises(SystemExit) as exc:
            UI(plain=False)
        assert "Test error" in str(exc.value)


def test_create_ui(mock_facade):
    from polylogue.ui import create_ui

    ui = create_ui(plain=True)
    assert isinstance(ui, UI)
    mock_facade.plain = True  # Setup specific for this test if needed


def test_ui_delegation(mock_facade):
    ui = UI(plain=False)

    ui.banner("Title", "Subtitle")
    mock_facade.banner.assert_called_with("Title", "Subtitle")

    ui.summary("Summary", ["line1"])
    mock_facade.summary.assert_called_with("Summary", ["line1"])

    ui.render_markdown("# Title")
    mock_facade.render_markdown.assert_called_with("# Title")

    ui.render_code("print('hi')")
    mock_facade.render_code.assert_called_with("print('hi')", "python")

    ui.render_diff("old", "new")
    mock_facade.render_diff.assert_called_with("old", "new", "file")


def test_ui_confirm_delegation(mock_facade):
    ui = UI(plain=False)
    mock_facade.confirm.return_value = True
    assert ui.confirm("Are you sure?") is True
    mock_facade.confirm.assert_called_with("Are you sure?", default=True)


def test_ui_confirm_plain(mock_facade):
    mock_facade.plain = True
    ui = UI(plain=True)

    # Mock isatty to be True so input is used
    with patch("sys.stdin.isatty", return_value=True), patch("builtins.input", side_effect=["y", "n", "", "foo"]):
        assert ui.confirm("Q1") is True
        assert ui.confirm("Q2") is False
        assert ui.confirm("Q3", default=True) is True  # Empty input -> default
        # "foo" -> not in {"y", "yes"} -> False (default logic logic: return response.lower() in {"y", "yes"})
        assert ui.confirm("Q4") is False


def test_ui_confirm_plain_eof(mock_facade):
    mock_facade.plain = True
    ui = UI(plain=True)
    with patch("sys.stdin.isatty", return_value=True), patch("builtins.input", side_effect=EOFError):
        assert ui.confirm("Q") is True  # Default


def test_ui_choose_empty(mock_facade):
    ui = UI(plain=False)
    assert ui.choose("Pick", []) is None


def test_ui_choose_plain(mock_facade):
    mock_facade.plain = True
    ui = UI(plain=True)
    options = ["A", "B", "C"]

    with patch("sys.stdin.isatty", return_value=True):
        # 1. Valid choice "1" -> "A"
        # 2. Invalid choice "99" -> loop -> "2" -> "B"
        # 3. Non-digit -> loop -> "3" -> "C"
        # 4. Empty -> None
        with patch("builtins.input", side_effect=["1", "99", "2", "foo", "3", ""]):
            assert ui.choose("Pick", options) == "A"
            assert ui.choose("Pick", options) == "B"
            assert ui.choose("Pick", options) == "C"
            assert ui.choose("Pick", options) is None


def test_ui_choose_plain_eof(mock_facade):
    mock_facade.plain = True
    ui = UI(plain=True)
    options = ["A"]
    with patch("sys.stdin.isatty", return_value=True), patch("builtins.input", side_effect=EOFError):
        assert ui.choose("Pick", options) is None


def test_ui_input_delegation(mock_facade):
    ui = UI(plain=False)
    mock_facade.input.return_value = "val"
    assert ui.input("Prompt") == "val"
    mock_facade.input.assert_called_with("Prompt", default=None)


def test_ui_input_plain(mock_facade):
    mock_facade.plain = True
    ui = UI(plain=True)
    with patch("sys.stdin.isatty", return_value=True), patch("builtins.input", side_effect=["val", ""]):
        assert ui.input("P1") == "val"
        assert ui.input("P2", default="def") == "def"


def test_ui_input_plain_eof(mock_facade):
    mock_facade.plain = True
    ui = UI(plain=True)
    with patch("sys.stdin.isatty", return_value=True), patch("builtins.input", side_effect=EOFError):
        assert ui.input("P") is None
        assert ui.input("P", default="def") == "def"


def test_plain_progress_tracker():
    console_mock = MagicMock()
    from polylogue.ui import _PlainProgressTracker

    # Test with total (int)
    tracker = _PlainProgressTracker(console_mock, "Task", 10)
    tracker.advance(1)
    tracker.update(description="New Task")
    tracker.update(total=20)
    tracker.__exit__(None, None, None)

    assert console_mock.print.call_count >= 2

    # Test with None total
    tracker = _PlainProgressTracker(console_mock, "Task2", None)
    tracker.advance(1.5)  # float
    tracker.update(total=5.5)
    tracker.__exit__(None, None, None)


class TestPlainConsoleMarkupStripping:
    """Regression tests: PlainConsole must strip Rich markup for CI/plain output."""

    def test_strips_bold_markup(self, capsys):
        console = PlainConsole()
        console.print("[bold]Archive:[/bold] 1,234 conversations")
        captured = capsys.readouterr()
        assert "[bold]" not in captured.out
        assert "Archive: 1,234 conversations" in captured.out

    def test_strips_color_markup(self, capsys):
        console = PlainConsole()
        console.print("[green]✓[/green] All ok")
        captured = capsys.readouterr()
        assert "[green]" not in captured.out
        assert "✓ All ok" in captured.out

    def test_strips_hex_color_markup(self, capsys):
        console = PlainConsole()
        console.print("[#d97757]████████[/#d97757]")
        captured = capsys.readouterr()
        assert "#d97757" not in captured.out
        assert "████████" in captured.out

    def test_preserves_plain_text(self, capsys):
        console = PlainConsole()
        console.print("No markup at all")
        captured = capsys.readouterr()
        assert captured.out.strip() == "No markup at all"

    def test_handles_empty_string(self, capsys):
        console = PlainConsole()
        console.print("")
        captured = capsys.readouterr()
        assert captured.out.strip() == ""


def test_rich_progress_tracker():
    progress_mock = MagicMock()
    task_id = "task1"

    from polylogue.ui import _RichProgressTracker

    tracker = _RichProgressTracker(progress_mock, task_id)

    with tracker:
        tracker.advance(5)
        tracker.update(total=100, description="Processing")

    progress_mock.__enter__.assert_called()
    progress_mock.advance.assert_called_with(task_id, 5)
    progress_mock.update.assert_called()
    progress_mock.__exit__.assert_called()
