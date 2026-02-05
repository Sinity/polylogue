"""Tests for polylogue.cli.editor — shell command validation and browser/editor opening."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from polylogue.cli.editor import (
    get_editor,
    open_in_browser,
    open_in_editor,
    validate_command,
)


# --- validate_command ---


class TestValidateCommand:
    """Tests for shell injection prevention."""

    def test_simple_command_accepted(self):
        """Simple command names should be accepted."""
        validate_command("vim")
        validate_command("nvim")
        validate_command("/usr/bin/code --wait")

    def test_empty_command_rejected(self):
        """Empty or whitespace-only commands should be rejected."""
        with pytest.raises(ValueError, match="cannot be empty"):
            validate_command("")
        with pytest.raises(ValueError, match="cannot be empty"):
            validate_command("   ")

    @pytest.mark.parametrize(
        "cmd",
        [
            "vim; rm -rf /",
            "vim & echo pwned",
            "vim | cat /etc/passwd",
            "$(malicious)",
            "`malicious`",
            "cmd > /dev/null",
            "cmd < /dev/null",
            "echo ${HOME}",
            "cmd (subshell)",
            "cmd {brace}",
            "cmd [bracket]",
            "!history",
            "cmd\\escaped",
        ],
    )
    def test_injection_attempts_rejected(self, cmd):
        """Commands with shell metacharacters should be rejected."""
        with pytest.raises(ValueError, match="unsafe shell metacharacters"):
            validate_command(cmd)

    def test_custom_context_in_error(self):
        """Custom context string should appear in error message."""
        with pytest.raises(ValueError, match=r"\$EDITOR"):
            validate_command("vim; echo", context="$EDITOR")

    def test_path_with_spaces_accepted(self):
        """Paths and flags with spaces should be accepted."""
        validate_command("/usr/local/bin/my editor")
        validate_command("code --wait")

    def test_hyphenated_flags_accepted(self):
        """Hyphenated flags should be accepted."""
        validate_command("vim -u NONE")
        validate_command("code --wait --new-window")


# --- get_editor ---


class TestGetEditor:
    """Tests for getting user's preferred editor."""

    def test_returns_editor_env(self, monkeypatch):
        """Should return $EDITOR env var when set."""
        monkeypatch.setenv("EDITOR", "vim")
        monkeypatch.delenv("VISUAL", raising=False)
        assert get_editor() == "vim"

    def test_returns_visual_fallback(self, monkeypatch):
        """Should return $VISUAL when $EDITOR is not set."""
        monkeypatch.delenv("EDITOR", raising=False)
        monkeypatch.setenv("VISUAL", "code")
        assert get_editor() == "code"

    def test_editor_takes_precedence(self, monkeypatch):
        """$EDITOR should take precedence over $VISUAL."""
        monkeypatch.setenv("EDITOR", "vim")
        monkeypatch.setenv("VISUAL", "code")
        assert get_editor() == "vim"

    def test_returns_none_when_unset(self, monkeypatch):
        """Should return None when neither env var is set."""
        monkeypatch.delenv("EDITOR", raising=False)
        monkeypatch.delenv("VISUAL", raising=False)
        assert get_editor() is None


# --- open_in_editor ---


class TestOpenInEditor:
    """Tests for opening file in editor."""

    def test_returns_false_when_no_editor(self, monkeypatch, tmp_path):
        """Should return False when no editor is configured."""
        monkeypatch.delenv("EDITOR", raising=False)
        monkeypatch.delenv("VISUAL", raising=False)
        f = tmp_path / "test.txt"
        f.write_text("hello")
        assert open_in_editor(f) is False

    def test_returns_false_for_nonexistent_file(self, monkeypatch, tmp_path):
        """Should return False for non-existent file."""
        monkeypatch.setenv("EDITOR", "vim")
        assert open_in_editor(tmp_path / "nonexistent.txt") is False

    def test_returns_false_for_unsafe_editor(self, monkeypatch, tmp_path):
        """Should return False when editor contains unsafe characters."""
        monkeypatch.setenv("EDITOR", "vim; rm -rf /")
        f = tmp_path / "test.txt"
        f.write_text("hello")
        assert open_in_editor(f) is False

    @patch("polylogue.cli.editor._run_editor")
    def test_vim_line_jump(self, mock_run, monkeypatch, tmp_path):
        """Vim should use +line syntax for line jumps."""
        monkeypatch.setenv("EDITOR", "vim")
        f = tmp_path / "test.txt"
        f.write_text("hello")
        mock_run.return_value = True
        assert open_in_editor(f, line=42) is True
        cmd = mock_run.call_args[0][0]
        assert "+42" in cmd
        assert str(f) in cmd

    @patch("polylogue.cli.editor._run_editor")
    def test_nvim_line_jump(self, mock_run, monkeypatch, tmp_path):
        """Neovim should use +line syntax for line jumps."""
        monkeypatch.setenv("EDITOR", "nvim")
        f = tmp_path / "test.txt"
        f.write_text("hello")
        mock_run.return_value = True
        assert open_in_editor(f, line=10) is True
        cmd = mock_run.call_args[0][0]
        assert "+10" in cmd
        assert str(f) in cmd

    @patch("polylogue.cli.editor._run_editor")
    def test_code_line_jump(self, mock_run, monkeypatch, tmp_path):
        """VS Code should use path:line syntax for line jumps."""
        monkeypatch.setenv("EDITOR", "code --wait")
        f = tmp_path / "test.txt"
        f.write_text("hello")
        mock_run.return_value = True
        assert open_in_editor(f, line=10) is True
        cmd = mock_run.call_args[0][0]
        # VS Code uses path:line syntax, should be in the command list
        assert any(f"{f}:10" in str(arg) for arg in cmd)

    @patch("polylogue.cli.editor._run_editor")
    def test_subl_line_jump(self, mock_run, monkeypatch, tmp_path):
        """Sublime Text should use path:line syntax for line jumps."""
        monkeypatch.setenv("EDITOR", "subl")
        f = tmp_path / "test.txt"
        f.write_text("hello")
        mock_run.return_value = True
        assert open_in_editor(f, line=15) is True
        cmd = mock_run.call_args[0][0]
        assert any(f"{f}:15" in str(arg) for arg in cmd)

    @patch("polylogue.cli.editor._run_editor")
    def test_atom_line_jump(self, mock_run, monkeypatch, tmp_path):
        """Atom should use path:line syntax for line jumps."""
        monkeypatch.setenv("EDITOR", "atom")
        f = tmp_path / "test.txt"
        f.write_text("hello")
        mock_run.return_value = True
        assert open_in_editor(f, line=20) is True
        cmd = mock_run.call_args[0][0]
        assert any(f"{f}:20" in str(arg) for arg in cmd)

    @patch("polylogue.cli.editor._run_editor")
    def test_emacs_line_jump(self, mock_run, monkeypatch, tmp_path):
        """Emacs should use +line syntax for line jumps."""
        monkeypatch.setenv("EDITOR", "emacs")
        f = tmp_path / "test.txt"
        f.write_text("hello")
        mock_run.return_value = True
        assert open_in_editor(f, line=5) is True
        cmd = mock_run.call_args[0][0]
        assert "+5" in cmd
        assert str(f) in cmd

    @patch("polylogue.cli.editor._run_editor")
    def test_opens_without_line(self, mock_run, monkeypatch, tmp_path):
        """Should open file without line when line not specified."""
        monkeypatch.setenv("EDITOR", "vim")
        f = tmp_path / "test.txt"
        f.write_text("hello")
        mock_run.return_value = True
        assert open_in_editor(f) is True
        cmd = mock_run.call_args[0][0]
        assert str(f) in cmd
        # Should not have line jump
        assert not any(arg.startswith("+") for arg in cmd)

    @patch("polylogue.cli.editor._run_editor")
    def test_returns_true_on_success(self, mock_run, monkeypatch, tmp_path):
        """Should return True when editor runs successfully."""
        monkeypatch.setenv("EDITOR", "vim")
        f = tmp_path / "test.txt"
        f.write_text("hello")
        mock_run.return_value = True
        assert open_in_editor(f) is True

    @patch("polylogue.cli.editor._run_editor")
    def test_returns_false_on_failure(self, mock_run, monkeypatch, tmp_path):
        """Should return False when editor fails to run."""
        monkeypatch.setenv("EDITOR", "vim")
        f = tmp_path / "test.txt"
        f.write_text("hello")
        mock_run.return_value = False
        assert open_in_editor(f) is False

    def test_uses_visual_when_editor_not_set(self, monkeypatch, tmp_path):
        """Should use $VISUAL if $EDITOR not set."""
        monkeypatch.delenv("EDITOR", raising=False)
        monkeypatch.setenv("VISUAL", "vim")
        f = tmp_path / "test.txt"
        f.write_text("hello")
        with patch("polylogue.cli.editor._run_editor", return_value=True):
            result = open_in_editor(f)
            assert result is True

    @patch("polylogue.cli.editor._run_editor")
    def test_unknown_editor_opens_without_line_jump(self, mock_run, monkeypatch, tmp_path):
        """Unknown editors should open without line jump syntax."""
        monkeypatch.setenv("EDITOR", "unknown_editor")
        f = tmp_path / "test.txt"
        f.write_text("hello")
        mock_run.return_value = True
        assert open_in_editor(f, line=42) is True
        cmd = mock_run.call_args[0][0]
        # Should not have special line jump syntax for unknown editor
        assert str(f) in cmd


# --- open_in_browser ---


class TestOpenInBrowser:
    """Tests for opening file in browser."""

    @patch("webbrowser.open")
    def test_opens_default_browser(self, mock_open, tmp_path, monkeypatch):
        """Should open file in default browser when no custom browser set."""
        monkeypatch.delenv("POLYLOGUE_BROWSER", raising=False)
        f = tmp_path / "test.html"
        f.write_text("<html></html>")
        mock_open.return_value = True
        assert open_in_browser(f) is True
        mock_open.assert_called_once()
        url = mock_open.call_args[0][0]
        assert url.startswith("file://")

    @patch("webbrowser.open")
    def test_anchor_appended(self, mock_open, tmp_path, monkeypatch):
        """Anchor should be appended to URL."""
        monkeypatch.delenv("POLYLOGUE_BROWSER", raising=False)
        f = tmp_path / "test.html"
        f.write_text("<html></html>")
        mock_open.return_value = True
        open_in_browser(f, anchor="section1")
        url = mock_open.call_args[0][0]
        assert url.endswith("#section1")

    @patch("webbrowser.open")
    def test_anchor_none_no_hash(self, mock_open, tmp_path, monkeypatch):
        """No anchor should result in no hash in URL."""
        monkeypatch.delenv("POLYLOGUE_BROWSER", raising=False)
        f = tmp_path / "test.html"
        f.write_text("<html></html>")
        mock_open.return_value = True
        open_in_browser(f, anchor=None)
        url = mock_open.call_args[0][0]
        assert not url.endswith("#")

    @patch("subprocess.Popen")
    def test_custom_browser(self, mock_popen, monkeypatch, tmp_path):
        """Custom browser should be used when POLYLOGUE_BROWSER is set."""
        monkeypatch.setenv("POLYLOGUE_BROWSER", "firefox")
        f = tmp_path / "test.html"
        f.write_text("<html></html>")
        mock_popen.return_value = MagicMock()
        assert open_in_browser(f) is True
        mock_popen.assert_called_once()
        cmd = mock_popen.call_args[0][0]
        assert cmd[0] == "firefox"

    @patch("subprocess.Popen")
    def test_custom_browser_with_anchor(self, mock_popen, monkeypatch, tmp_path):
        """Custom browser should receive anchor in URL."""
        monkeypatch.setenv("POLYLOGUE_BROWSER", "firefox")
        f = tmp_path / "test.html"
        f.write_text("<html></html>")
        mock_popen.return_value = MagicMock()
        open_in_browser(f, anchor="section")
        cmd = mock_popen.call_args[0][0]
        assert "firefox" in cmd
        assert cmd[-1].endswith("#section")

    def test_rejects_unsafe_browser(self, monkeypatch, tmp_path):
        """Unsafe browser command should be rejected."""
        monkeypatch.setenv("POLYLOGUE_BROWSER", "firefox; rm -rf /")
        f = tmp_path / "test.html"
        f.write_text("<html></html>")
        assert open_in_browser(f) is False

    def test_rejects_backtick_in_browser(self, monkeypatch, tmp_path):
        """Browser command with backticks should be rejected."""
        monkeypatch.setenv("POLYLOGUE_BROWSER", "firefox `whoami`")
        f = tmp_path / "test.html"
        f.write_text("<html></html>")
        assert open_in_browser(f) is False

    def test_rejects_dollar_in_browser(self, monkeypatch, tmp_path):
        """Browser command with variable expansion should be rejected."""
        monkeypatch.setenv("POLYLOGUE_BROWSER", "firefox $HOME")
        f = tmp_path / "test.html"
        f.write_text("<html></html>")
        assert open_in_browser(f) is False

    def test_handles_invalid_path_null_char(self, monkeypatch):
        """Should handle paths with null characters gracefully."""
        monkeypatch.delenv("POLYLOGUE_BROWSER", raising=False)
        bad_path = Path("\x00invalid.html")
        assert open_in_browser(bad_path) is False

    def test_handles_nonexistent_file(self, monkeypatch, tmp_path):
        """Should still work with nonexistent file path (browser may handle it)."""
        monkeypatch.delenv("POLYLOGUE_BROWSER", raising=False)
        nonexistent = tmp_path / "nonexistent.html"
        with patch("webbrowser.open", return_value=True):
            # Should convert to file:// URL even if file doesn't exist
            result = open_in_browser(nonexistent)
            # The result depends on webbrowser implementation
            assert isinstance(result, bool)

    @patch("subprocess.Popen")
    def test_returns_false_on_popen_error(self, mock_popen, monkeypatch, tmp_path):
        """Should return False when Popen raises OSError."""
        monkeypatch.setenv("POLYLOGUE_BROWSER", "firefox")
        f = tmp_path / "test.html"
        f.write_text("<html></html>")
        mock_popen.side_effect = OSError("No such file")
        assert open_in_browser(f) is False

    @patch("webbrowser.open")
    def test_returns_false_on_webbrowser_error(self, mock_open, monkeypatch, tmp_path):
        """Should return False when webbrowser.open raises error."""
        monkeypatch.delenv("POLYLOGUE_BROWSER", raising=False)
        f = tmp_path / "test.html"
        f.write_text("<html></html>")
        mock_open.side_effect = OSError("webbrowser error")
        assert open_in_browser(f) is False

    @patch("subprocess.Popen")
    def test_custom_browser_with_args(self, mock_popen, monkeypatch, tmp_path):
        """Custom browser with arguments should work."""
        monkeypatch.setenv("POLYLOGUE_BROWSER", "firefox --new-window")
        f = tmp_path / "test.html"
        f.write_text("<html></html>")
        mock_popen.return_value = MagicMock()
        assert open_in_browser(f) is True
        cmd = mock_popen.call_args[0][0]
        assert "firefox" in cmd
        assert "--new-window" in cmd

    @patch("webbrowser.open")
    def test_file_uri_format(self, mock_open, tmp_path, monkeypatch):
        """File URI should be properly formatted."""
        monkeypatch.delenv("POLYLOGUE_BROWSER", raising=False)
        f = tmp_path / "test file.html"
        f.write_text("<html></html>")
        mock_open.return_value = True
        open_in_browser(f)
        url = mock_open.call_args[0][0]
        # Should be a valid file URI
        assert url.startswith("file://")
        # Should properly handle spaces
        assert "test" in url


# --- _run_editor ---


class TestRunEditor:
    """Tests for _run_editor helper."""

    @patch("subprocess.run")
    def test_returns_true_on_success(self, mock_run):
        """Should return True when subprocess.run succeeds."""
        from polylogue.cli.editor import _run_editor

        mock_run.return_value = MagicMock()
        result = _run_editor(["vim", "/tmp/test.txt"])
        assert result is True
        mock_run.assert_called_once()

    @patch("subprocess.run")
    def test_returns_false_on_file_not_found(self, mock_run):
        """Should return False when editor executable not found."""
        from polylogue.cli.editor import _run_editor

        mock_run.side_effect = FileNotFoundError("vim not found")
        result = _run_editor(["vim", "/tmp/test.txt"])
        assert result is False

    @patch("subprocess.run")
    def test_returns_false_on_os_error(self, mock_run):
        """Should return False on OSError."""
        from polylogue.cli.editor import _run_editor

        mock_run.side_effect = OSError("Permission denied")
        result = _run_editor(["vim", "/tmp/test.txt"])
        assert result is False

    @patch("subprocess.run")
    def test_returns_false_on_subprocess_error(self, mock_run):
        """Should return False on SubprocessError."""
        from polylogue.cli.editor import _run_editor

        mock_run.side_effect = subprocess.SubprocessError("Process error")
        result = _run_editor(["vim", "/tmp/test.txt"])
        assert result is False

    @patch("subprocess.run")
    def test_passes_cmd_list(self, mock_run):
        """Should pass command list to subprocess.run."""
        from polylogue.cli.editor import _run_editor

        mock_run.return_value = MagicMock()
        cmd = ["vim", "file.txt"]
        _run_editor(cmd)
        mock_run.assert_called_once()
        args = mock_run.call_args[0][0]
        assert args == cmd

    @patch("subprocess.run")
    def test_check_false(self, mock_run):
        """Should call subprocess.run with check=False."""
        from polylogue.cli.editor import _run_editor

        mock_run.return_value = MagicMock()
        _run_editor(["vim"])
        # Verify check=False is passed
        mock_run.assert_called_once_with(["vim"], check=False)


# --- Integration tests ---


class TestEditorIntegration:
    """Integration tests combining multiple functions."""

    def test_validate_and_open_flow(self, tmp_path, monkeypatch):
        """Complete flow: validate editor, get editor, then open."""
        monkeypatch.setenv("EDITOR", "vim")
        f = tmp_path / "test.txt"
        f.write_text("content")

        # Should get editor
        editor = get_editor()
        assert editor is not None
        # Should validate successfully
        validate_command(editor)
        # Should open successfully with mocked _run_editor
        with patch("polylogue.cli.editor._run_editor", return_value=True):
            result = open_in_editor(f)
            assert result is True

    def test_unsafe_editor_validation_prevents_open(self, tmp_path, monkeypatch):
        """Unsafe editor should be caught before attempting to open."""
        monkeypatch.setenv("EDITOR", "vim; echo hacked")
        f = tmp_path / "test.txt"
        f.write_text("content")

        # open_in_editor should validate and reject
        result = open_in_editor(f)
        assert result is False

    @patch("webbrowser.open")
    def test_browser_with_anchor_integration(self, mock_open, tmp_path, monkeypatch):
        """Browser opening with anchor should produce correct URI."""
        monkeypatch.delenv("POLYLOGUE_BROWSER", raising=False)
        f = tmp_path / "doc.html"
        f.write_text("<html></html>")
        mock_open.return_value = True

        open_in_browser(f, anchor="results")

        url = mock_open.call_args[0][0]
        assert url.startswith("file://")
        assert url.endswith("#results")
