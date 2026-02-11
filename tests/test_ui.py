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
    @pytest.mark.parametrize("content,expected_count,desc", [
        (None, 0, "no_env_var"),
        (json.dumps({"type": "confirm", "value": True}) + "\n", 1, "single_stub"),
        (json.dumps({"type": "confirm", "value": True}) + "\n" + json.dumps({"type": "choose", "value": "option1"}) + "\n" + json.dumps({"type": "input", "value": "text"}) + "\n", 3, "multiple_stubs"),
        ("\n\n" + json.dumps({"type": "confirm"}) + "\n\n", 1, "empty_lines"),
        ("   \n\t\n" + json.dumps({"type": "input"}) + "\n  ", 1, "whitespace_lines"),
    ], ids=["no_env_var", "single", "multiple", "empty_lines", "whitespace"])
    def test_stub_loading(self, tmp_path, monkeypatch, content, expected_count, desc):
        if content is None:
            monkeypatch.delenv("POLYLOGUE_TEST_PROMPT_FILE", raising=False)
        else:
            stub_file = tmp_path / "stubs.jsonl"
            stub_file.write_text(content)
            monkeypatch.setenv("POLYLOGUE_TEST_PROMPT_FILE", str(stub_file))
        facade = ConsoleFacade(plain=True)
        assert len(facade._prompt_responses) == expected_count

    def test_invalid_json_raises_uierror(self, tmp_path, monkeypatch):
        stub_file = tmp_path / "bad.jsonl"
        stub_file.write_text("not json\n")
        monkeypatch.setenv("POLYLOGUE_TEST_PROMPT_FILE", str(stub_file))
        with pytest.raises(UIError, match="Invalid prompt stub"):
            ConsoleFacade(plain=True)

    def test_pop_behaviors(self, tmp_path, monkeypatch):
        # Test type mismatch raises
        stub_file = tmp_path / "stubs.jsonl"
        stub_file.write_text(json.dumps({"type": "confirm", "value": True}) + "\n")
        monkeypatch.setenv("POLYLOGUE_TEST_PROMPT_FILE", str(stub_file))
        facade = ConsoleFacade(plain=True)
        with pytest.raises(UIError, match="expected 'confirm' but got 'choose'"):
            facade._pop_prompt_response("choose")

        # Test no type matches any
        stub_file.write_text(json.dumps({"value": True}) + "\n")
        facade = ConsoleFacade(plain=True)
        result = facade._pop_prompt_response("anything")
        assert result == {"value": True}

        # Test empty queue returns none
        monkeypatch.delenv("POLYLOGUE_TEST_PROMPT_FILE", raising=False)
        facade = ConsoleFacade(plain=True)
        result = facade._pop_prompt_response("confirm")
        assert result is None


# =============================================================================
# confirm
# =============================================================================

class TestConfirm:
    @pytest.mark.parametrize("prompt,default,expected,ignore", [
        ("Continue?", True, True, False),
        ("Continue?", False, False, False),
        ("anything", True, True, True),
    ], ids=["plain_default_true", "plain_default_false", "plain_ignores_prompt"])
    def test_plain_returns_default(self, prompt, default, expected, ignore):
        facade = ConsoleFacade(plain=True)
        assert facade.confirm(prompt, default=default) is expected

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

    @pytest.mark.parametrize("use_default,default,expected", [
        (True, True, True),
        (True, False, False),
    ])
    def test_stub_use_default(self, tmp_path, monkeypatch, use_default, default, expected):
        stub_file = tmp_path / "stubs.jsonl"
        stub_file.write_text(json.dumps({"type": "confirm", "use_default": use_default}) + "\n")
        monkeypatch.setenv("POLYLOGUE_TEST_PROMPT_FILE", str(stub_file))
        facade = ConsoleFacade(plain=False)
        assert facade.confirm("Continue?", default=default) is expected

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
    @pytest.mark.parametrize("plain,options,prompt", [
        (True, [], "Pick:"),
        (False, [], "Pick:"),
        (True, ["a", "b"], "Pick:"),
        (True, ["x", "y", "z"], "anything"),
    ], ids=["plain_empty", "rich_empty", "plain_returns_none", "plain_ignores_prompt"])
    def test_plain_returns_none(self, plain, options, prompt):
        facade = ConsoleFacade(plain=plain)
        assert facade.choose(prompt, options) is None

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
    @pytest.mark.parametrize("prompt,default,expected", [
        ("Name:", "anon", "anon"),
        ("Name:", None, None),
        ("anything", "default", "default"),
    ], ids=["plain_returns_default", "plain_no_default", "plain_ignores_prompt"])
    def test_plain_behavior(self, prompt, default, expected):
        facade = ConsoleFacade(plain=True)
        result = facade.input(prompt) if default is None else facade.input(prompt, default=default)
        assert result == expected

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

    @pytest.mark.parametrize("stub_value", [None, ""])
    def test_stub_value_falsy(self, tmp_path, monkeypatch, stub_value):
        stub_file = tmp_path / "stubs.jsonl"
        stub_file.write_text(json.dumps({"type": "input", "value": stub_value}) + "\n")
        monkeypatch.setenv("POLYLOGUE_TEST_PROMPT_FILE", str(stub_file))
        facade = ConsoleFacade(plain=False)
        if stub_value is None:
            assert facade.input("Name:") is None
        else:
            assert facade.input("Name:", default="default") == ""


# =============================================================================
# Display methods (banner, summary, render_*, status)
# =============================================================================

class TestBanner:
    @pytest.mark.parametrize("plain,has_subtitle", [
        (True, True),
        (True, False),
        (False, True),
        (False, False),
    ], ids=["plain_with_subtitle", "plain_no_subtitle", "rich_with_subtitle", "rich_no_subtitle"])
    def test_banner(self, capsys, plain, has_subtitle):
        facade = ConsoleFacade(plain=plain)
        if has_subtitle:
            facade.banner("Title", "Subtitle")
            if plain:
                output = capsys.readouterr().out
                assert "Title" in output
                assert "Subtitle" in output
        else:
            facade.banner("Title")
            if plain:
                output = capsys.readouterr().out
                assert "Title" in output
                assert "==" in output

    def test_plain_banner_formatting(self, capsys):
        facade = ConsoleFacade(plain=True)
        facade.banner("Test")
        output = capsys.readouterr().out
        assert "== Test ==" in output


class TestSummary:
    @pytest.mark.parametrize("lines_count,expected_lines", [
        (1, ["Line 1"]),
        (3, ["Line 1", "Line 2", "Line 3"]),
        (0, []),
    ], ids=["single_line", "multiple_lines", "empty"])
    def test_plain_summary(self, capsys, lines_count, expected_lines):
        facade = ConsoleFacade(plain=True)
        lines = [f"Line {i+1}" for i in range(lines_count)]
        facade.summary("Stats", lines)
        output = capsys.readouterr().out
        assert "Stats" in output
        for expected_line in expected_lines:
            assert expected_line in output
        if lines_count > 0:
            assert "--" in output

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
    @pytest.mark.parametrize("plain,multiline", [
        (True, False),
        (True, True),
        (False, False),
    ], ids=["plain_simple", "plain_multiline", "rich"])
    def test_render_markdown(self, capsys, plain, multiline):
        facade = ConsoleFacade(plain=plain)
        if multiline:
            content = "# Title\n\n**Bold** text\n\n- List\n- Items"
        else:
            content = "# Hello\n\nWorld"
        facade.render_markdown(content)
        if plain:
            output = capsys.readouterr().out
            assert "Hello" in output or "Title" in output


class TestRenderCode:
    @pytest.mark.parametrize("plain,language", [
        (True, None),
        (True, "sql"),
        (True, "python"),
        (False, "python"),
        (False, "javascript"),
    ], ids=["plain_default", "plain_sql", "plain_python", "rich_python", "rich_javascript"])
    def test_render_code(self, capsys, plain, language):
        facade = ConsoleFacade(plain=plain)
        if language == "sql":
            code = "SELECT * FROM users;"
            expected_text = "SELECT"
        elif language == "javascript":
            code = "console.log('test');"
            expected_text = "console"
        else:
            code = "x = 1" if plain or language is None else "print('hello')" if language == "python" else "x = 1"
            expected_text = "x" if "x" in code else "print"

        facade.render_code(code, language) if language else facade.render_code(code)
        if plain:
            output = capsys.readouterr().out
            assert expected_text in output or code in output


class TestRenderDiff:
    @pytest.mark.parametrize("old,new,filename,desc", [
        ("old\n", "new\n", "test.txt", "basic_diff"),
        ("line1\nline2\n", "line1\nline2_modified\n", "file.txt", "unified_diff"),
        ("same\n", "same\n", "file.txt", "empty_diff"),
    ], ids=["basic", "unified", "empty"])
    def test_render_diff(self, capsys, old, new, filename, desc):
        # Plain mode
        facade = ConsoleFacade(plain=True)
        facade.render_diff(old, new, filename)
        output = capsys.readouterr().out
        if desc == "empty_diff":
            assert filename in output or output.strip() == ""
        else:
            assert filename in output or "---" in output

        # Rich mode (just verify no error)
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
    def test_plain_console_facade_properties_and_methods(self, capsys):
        facade = PlainConsoleFacade(plain=True)
        # Test inheritance and properties
        assert isinstance(facade, ConsoleFacade)
        assert facade.plain is True
        assert isinstance(facade.console, PlainConsole)

        # Test banner method
        facade.banner("Test")
        output = capsys.readouterr().out
        assert "Test" in output

        # Test confirm method
        result = facade.confirm("Q?", default=True)
        assert result is True


# =============================================================================
# VersionInfo
# =============================================================================

class TestVersionInfo:
    def test_version_only_and_repr(self):
        v = VersionInfo(version="1.0.0")
        assert str(v) == "1.0.0"
        assert v.full == "1.0.0"
        assert v.short == "1.0.0"
        assert "1.0.0" in repr(v)

    @pytest.mark.parametrize("commit,dirty,has_dirty_suffix", [
        ("abc123def456", False, False),
        ("abc123def456", True, True),
    ], ids=["with_commit_clean", "with_commit_dirty"])
    def test_version_with_commit(self, commit, dirty, has_dirty_suffix):
        v = VersionInfo(version="1.0.0", commit=commit, dirty=dirty)
        v_str = str(v)
        assert "1.0.0+" in v_str
        assert v.short == "1.0.0"
        if has_dirty_suffix:
            assert "-dirty" in v_str
        else:
            assert "-dirty" not in v_str

    @pytest.mark.parametrize("version,commit", [
        ("2.5.3", "fedcba9876543210"),
        ("3.2.1", "deadbeef12345678"),
    ], ids=["short_sha_1", "short_sha_2"])
    def test_version_properties(self, version, commit):
        v = VersionInfo(version=version, commit=commit)
        assert str(v) == f"{version}+{commit[:8]}"
        assert v.full == f"{version}+{commit[:8]}"
        assert v.short == version

    def test_version_dirty_no_commit_and_equality(self):
        # No commit = version only
        v = VersionInfo(version="1.0.0", dirty=True)
        assert str(v) == "1.0.0"

        # Test equality
        v1 = VersionInfo(version="1.0.0")
        v2 = VersionInfo(version="1.0.0")
        assert v1 == v2

        # Test dataclass fields
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

    @pytest.mark.parametrize("scenario,setup", [
        ("nonexistent", lambda tmp_path: tmp_path / "nonexistent"),
        ("non_git_dir", lambda tmp_path: tmp_path),
    ], ids=["nonexistent_dir", "non_git_dir"])
    def test_returns_none(self, tmp_path, scenario, setup):
        path = setup(tmp_path)
        commit, dirty = _get_git_info(path)
        assert commit is None
        assert dirty is False

    def test_timeout_returns_none(self, tmp_path, monkeypatch):
        import subprocess

        def mock_run(*args, **kwargs):
            raise subprocess.TimeoutExpired("git", 2)

        monkeypatch.setattr(subprocess, "run", mock_run)
        commit, dirty = _get_git_info(tmp_path)
        assert commit is None
        assert dirty is False

    def test_returns_tuple_and_dirty_is_bool(self, tmp_path):
        result = _get_git_info(tmp_path)
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[1], bool)

        repo_root = Path(__file__).resolve().parent.parent
        if (repo_root / ".git").exists():
            commit, dirty = _get_git_info(repo_root)
            assert isinstance(dirty, bool)


# =============================================================================
# _resolve_version
# =============================================================================

class TestResolveVersion:
    def test_returns_version_info_and_attributes(self):
        result = _resolve_version()
        assert isinstance(result, VersionInfo)
        assert result.version != ""
        assert result.version != "unknown"
        assert len(result.version) > 0

        # Check all attributes exist and have correct types
        assert hasattr(result, "version")
        assert hasattr(result, "commit")
        assert hasattr(result, "dirty")
        assert result.commit is None or isinstance(result.commit, str)
        assert isinstance(result.dirty, bool)


# =============================================================================
# Module-level constants
# =============================================================================

class TestVersionConstants:
    def test_all_version_exports(self):
        from polylogue.version import POLYLOGUE_VERSION, VERSION_INFO, VersionInfo
        from polylogue import version

        # Test POLYLOGUE_VERSION
        assert isinstance(POLYLOGUE_VERSION, str)
        assert len(POLYLOGUE_VERSION) > 0

        # Test VERSION_INFO
        assert isinstance(VERSION_INFO, VersionInfo)
        assert VERSION_INFO.version in POLYLOGUE_VERSION

        # Test module exports
        assert hasattr(version, "POLYLOGUE_VERSION")
        assert hasattr(version, "VERSION_INFO")
        assert hasattr(version, "VersionInfo")

        # Test __all__ if present
        if hasattr(version, "__all__"):
            all_items = version.__all__
            assert "POLYLOGUE_VERSION" in all_items
            assert "VERSION_INFO" in all_items
            assert "VersionInfo" in all_items


# =============================================================================
# Additional edge cases and branch coverage
# =============================================================================

class TestResolveVersionEdgeCases:
    def test_resolve_version_consistency_and_fallback(self, monkeypatch):
        """Test version resolution consistency and fallback behavior."""
        # Test consistency
        result1 = _resolve_version()
        result2 = _resolve_version()
        assert isinstance(result1, VersionInfo)
        assert isinstance(result2, VersionInfo)
        assert result1.version == result2.version

        # Test fallback to pyproject.toml
        from importlib.metadata import PackageNotFoundError
        import polylogue.version as version_module

        def mock_metadata_version(name):
            raise PackageNotFoundError(name)

        original_metadata_version = version_module.metadata_version
        monkeypatch.setattr(version_module, "metadata_version", mock_metadata_version)
        result = _resolve_version()
        monkeypatch.setattr(version_module, "metadata_version", original_metadata_version)

        assert result.version is not None
        assert len(result.version) > 0


class TestUIErrorException:
    def test_ui_error_behavior(self):
        assert issubclass(UIError, Exception)

        with pytest.raises(UIError):
            raise UIError("test error")

        msg = "custom error message"
        with pytest.raises(UIError, match=msg):
            raise UIError(msg)


class TestConsoleProtocolThemeAndStyles:
    def test_console_protocol_theme_and_styles(self):
        from dataclasses import is_dataclass
        # PlainConsole protocol
        pc = PlainConsole()
        assert callable(getattr(pc, "print", None))

        # Facade dataclass
        assert is_dataclass(ConsoleFacade)
        assert is_dataclass(PlainConsoleFacade)

        # Plain facade theme and styles
        facade_plain = ConsoleFacade(plain=True)
        assert facade_plain.theme is not None

        # Rich facade protocol, theme, and styles
        facade_rich = ConsoleFacade(plain=False)
        assert callable(getattr(facade_rich.console, "print", None))
        assert facade_rich.theme is not None
        assert facade_rich._panel_box is not None
        assert facade_rich._banner_box is not None
        assert facade_rich._banner_box != facade_rich._panel_box


class TestStatusMethodsPrivate:
    @pytest.mark.parametrize("plain,icon,style,text", [
        (True, "✓", "status.icon.success", "Test"),
        (True, "!", "status.icon.warning", "Alert"),
        (False, "✓", "status.icon.success", "Success"),
    ], ids=["plain_success", "plain_warning", "rich"])
    def test_status_private(self, capsys, plain, icon, style, text):
        facade = ConsoleFacade(plain=plain)
        facade._status(icon, style, text)
        if plain:
            output = capsys.readouterr().out
            assert text in output


class TestChooseWithLargeOptionSet:
    def test_choose_select_vs_autocomplete_threshold(self, tmp_path, monkeypatch):
        """Test threshold between select (<=12) and autocomplete (>12)."""
        # Test exactly 12 options (uses select)
        stub_file = tmp_path / "stubs.jsonl"
        stub_file.write_text(json.dumps({"type": "choose", "index": 0}) + "\n")
        monkeypatch.setenv("POLYLOGUE_TEST_PROMPT_FILE", str(stub_file))
        facade = ConsoleFacade(plain=False)
        options = [f"opt{i}" for i in range(12)]
        result = facade.choose("Pick:", options)
        assert result == "opt0"

        # Test 13 options (uses autocomplete)
        stub_file.write_text(json.dumps({"type": "choose", "value": "opt12"}) + "\n")
        facade = ConsoleFacade(plain=False)
        options = [f"opt{i}" for i in range(13)]
        result = facade.choose("Pick:", options)
        assert result == "opt12"


class TestInputEdgeCases:
    @pytest.mark.parametrize("questionary_return,expected,desc", [
        ("", "fallback", "empty_string_returns_default"),
        ("   ", "   ", "whitespace_returned_as_is"),
    ], ids=["empty_questionary", "whitespace_questionary"])
    def test_input_questionary_results(self, monkeypatch, questionary_return, expected, desc):
        """Test input with different questionary responses."""
        import questionary

        monkeypatch.delenv("POLYLOGUE_TEST_PROMPT_FILE", raising=False)
        facade = ConsoleFacade(plain=False)

        mock_text = MagicMock()
        mock_text.return_value.ask.return_value = questionary_return
        monkeypatch.setattr(questionary, "text", lambda *args, **kwargs: mock_text.return_value)

        result = facade.input("Prompt:", default="fallback")
        assert result == expected


class TestRenderDiffEdgeCases:
    @pytest.mark.parametrize("old,new,filename,desc", [
        ("line1\nline2\nline3\n", "line1\nmodified2\nline3\nline4\n", "test.py", "multiline"),
        ("line1", "line1\nline2", "file.txt", "no_trailing_newline"),
    ], ids=["multiline_format", "no_trailing_newline"])
    def test_render_diff_edge_cases(self, capsys, old, new, filename, desc):
        """Test render_diff edge cases."""
        facade = ConsoleFacade(plain=True)
        facade.render_diff(old, new, filename)
        output = capsys.readouterr().out
        assert filename in output


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

    @pytest.mark.parametrize("stub_value,expected", [(True, True), (False, False)], ids=["confirm_true", "confirm_false"])
    def test_confirm(self, mock_prompt_file, stub_value, expected):
        """Test confirm() returns values from mock."""
        mock_prompt_file.write_text(json.dumps({"type": "confirm", "value": stub_value}) + "\n")
        facade = create_console_facade(plain=False)
        assert facade.confirm("Continue?") is expected

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

    @pytest.mark.parametrize("value,index,expected", [
        ("Option B", None, "Option B"),
        (None, 2, "Option C"),
    ], ids=["by_value", "by_index"])
    def test_choose(self, mock_prompt_file, value, index, expected):
        """Test choose() selects by value or index."""
        stub_dict = {"type": "choose"}
        if value:
            stub_dict["value"] = value
        if index is not None:
            stub_dict["index"] = index
        mock_prompt_file.write_text(json.dumps(stub_dict) + "\n")

        facade = create_console_facade(plain=False)
        result = facade.choose("Pick one", ["Option A", "Option B", "Option C"])
        assert result == expected

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
    """Test rich and plain rendering methods."""

    def test_banner_render(self):
        """Test banner rendering."""
        facade = create_console_facade(plain=False)
        with facade.console.capture() as capture:
            facade.banner("Welcome", "To Mission Control")

        output = capture.get()
        assert "Welcome" in output
        assert "To Mission Control" in output
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

    def test_render_diff_and_status(self):
        """Test diff rendering and status messages."""
        facade = create_console_facade(plain=False)

        # Test render_diff
        facade.console.pager = MagicMock()
        facade.console.pager.return_value.__enter__ = MagicMock()
        facade.console.pager.return_value.__exit__ = MagicMock()

        with facade.console.capture() as capture:
            facade.render_diff("line 1\nline 2", "line 1\nline 3", filename="test.txt")
        output = capture.get()
        assert "line 2" in output
        assert "line 3" in output

        # Test status messages
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

    def test_plain_mode_interactions_and_rendering(self, capsys):
        """Test plain mode fallback for interactions and rendering."""
        facade = create_console_facade(plain=True)
        assert isinstance(facade, PlainConsoleFacade)

        # Test interactions
        assert facade.confirm("Ctx?", default=True) is True
        assert facade.confirm("Ctx?", default=False) is False
        assert facade.input("Input?", default="Default") == "Default"
        assert facade.input("Input?", default=None) is None
        assert facade.choose("Pick", ["A", "B"]) is None

        # Test rendering
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


def test_ui_init_and_creation(mock_facade):
    # Test successful initialization
    ui = UI(plain=False)
    assert ui._facade == mock_facade
    assert ui.plain is False
    assert ui.console == mock_facade.console

    # Test init failure
    with patch("polylogue.ui.create_console_facade", side_effect=RuntimeError("Test error")):
        with pytest.raises(SystemExit) as exc:
            UI(plain=False)
        assert "Test error" in str(exc.value)

    # Test create_ui factory
    from polylogue.ui import create_ui
    ui = create_ui(plain=True)
    assert isinstance(ui, UI)


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


def test_ui_confirm(mock_facade):
    # Test delegation
    ui = UI(plain=False)
    mock_facade.confirm.return_value = True
    assert ui.confirm("Are you sure?") is True
    mock_facade.confirm.assert_called_with("Are you sure?", default=True)

    # Test plain mode
    mock_facade.plain = True
    ui = UI(plain=True)
    with patch("sys.stdin.isatty", return_value=True), patch("builtins.input", side_effect=["y", "n", "", "foo"]):
        assert ui.confirm("Q1") is True
        assert ui.confirm("Q2") is False
        assert ui.confirm("Q3", default=True) is True
        assert ui.confirm("Q4") is False

    # Test plain mode EOF
    mock_facade.plain = True
    ui = UI(plain=True)
    with patch("sys.stdin.isatty", return_value=True), patch("builtins.input", side_effect=EOFError):
        assert ui.confirm("Q") is True


def test_ui_choose(mock_facade):
    # Test empty options
    ui = UI(plain=False)
    assert ui.choose("Pick", []) is None

    # Test plain mode
    mock_facade.plain = True
    ui = UI(plain=True)
    options = ["A", "B", "C"]

    with patch("sys.stdin.isatty", return_value=True):
        with patch("builtins.input", side_effect=["1", "99", "2", "foo", "3", ""]):
            assert ui.choose("Pick", options) == "A"
            assert ui.choose("Pick", options) == "B"
            assert ui.choose("Pick", options) == "C"
            assert ui.choose("Pick", options) is None

    # Test plain mode EOF
    mock_facade.plain = True
    ui = UI(plain=True)
    options = ["A"]
    with patch("sys.stdin.isatty", return_value=True), patch("builtins.input", side_effect=EOFError):
        assert ui.choose("Pick", options) is None


def test_ui_input(mock_facade):
    # Test delegation
    ui = UI(plain=False)
    mock_facade.input.return_value = "val"
    assert ui.input("Prompt") == "val"
    mock_facade.input.assert_called_with("Prompt", default=None)

    # Test plain mode
    mock_facade.plain = True
    ui = UI(plain=True)
    with patch("sys.stdin.isatty", return_value=True), patch("builtins.input", side_effect=["val", ""]):
        assert ui.input("P1") == "val"
        assert ui.input("P2", default="def") == "def"

    # Test plain mode EOF
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
    console_mock.reset_mock()
    tracker = _PlainProgressTracker(console_mock, "Task2", None)
    tracker.advance(1.5)
    tracker.update(total=5.5)
    tracker.__exit__(None, None, None)


class TestPlainConsoleMarkupStripping:
    """Regression tests: PlainConsole must strip Rich markup for CI/plain output."""

    @pytest.mark.parametrize("input_text,should_not_have,should_have", [
        ("[bold]Archive:[/bold] 1,234 conversations", "[bold]", "Archive: 1,234 conversations"),
        ("[green]✓[/green] All ok", "[green]", "✓ All ok"),
        ("[#d97757]████████[/#d97757]", "#d97757", "████████"),
        ("No markup at all", "", "No markup at all"),
        ("", "", ""),
    ], ids=["bold_markup", "color_markup", "hex_color_markup", "plain_text", "empty_string"])
    def test_markup_handling(self, capsys, input_text, should_not_have, should_have):
        console = PlainConsole()
        console.print(input_text)
        captured = capsys.readouterr()
        if should_not_have:
            assert should_not_have not in captured.out
        assert should_have in captured.out


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
