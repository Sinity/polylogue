"""Tests for polylogue.cli.editor module security."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from polylogue.cli.editor import validate_command

# =============================================================================
# Parametrized test cases for command validation
# =============================================================================

UNSAFE_COMMAND_CASES = [
    ("vim; rm -rf /tmp/pwned", "unsafe shell metacharacters", "semicolon injection"),
    ("vim | cat /etc/passwd", "unsafe shell metacharacters", "pipe injection"),
    ("vim `whoami`", "unsafe shell metacharacters", "backtick injection"),
    ("vim $(cat /etc/passwd)", "unsafe shell metacharacters", "dollar paren injection"),
    ("vim & malicious_command", "unsafe shell metacharacters", "ampersand background"),
    ("vim && rm -rf /", "unsafe shell metacharacters", "double ampersand chain"),
    ("vim || evil_command", "unsafe shell metacharacters", "double pipe fallback"),
    ("vim > /tmp/output", "unsafe shell metacharacters", "redirect out"),
    ("vim < /tmp/input", "unsafe shell metacharacters", "redirect in"),
    ("vim {/tmp/a,/tmp/b}", "unsafe shell metacharacters", "brace expansion"),
    ("vim /tmp/[abc]", "unsafe shell metacharacters", "bracket glob"),
    ("vim \\n", "unsafe shell metacharacters", "backslash escape"),
    ("vim !!", "unsafe shell metacharacters", "history expansion"),
    ("", "cannot be empty", "empty string"),
    ("   ", "cannot be empty", "whitespace only"),
]

SAFE_COMMAND_CASES = [
    ("vim", "simple vim"),
    ("/usr/bin/vim", "vim with path"),
    ("vim -u NONE", "vim with options"),
    ("nano", "nano editor"),
    ("nvim", "neovim"),
    ("code --wait", "vscode with wait"),
    ("emacs -nw", "emacs terminal mode"),
]


class TestEditorCommandValidation:
    """Parametrized tests for editor command validation."""

    @pytest.mark.parametrize("command,expected_error,description", UNSAFE_COMMAND_CASES)
    def test_validate_command_rejects_unsafe(self, command: str, expected_error: str, description: str):
        """Command with unsafe patterns should be rejected."""
        with pytest.raises(ValueError, match=expected_error):
            validate_command(command)

    @pytest.mark.parametrize("command,description", SAFE_COMMAND_CASES)
    def test_validate_command_allows_safe(self, command: str, description: str):
        """Safe editor command should be allowed."""
        # Should not raise
        validate_command(command)

    def test_validate_command_custom_context(self):
        """Custom context should appear in error message."""
        with pytest.raises(ValueError, match="CUSTOM_VAR"):
            validate_command("vim; evil", context="$CUSTOM_VAR")


class TestOpenInEditorSecurity:
    """Tests for open_in_editor function security."""

    def test_open_in_editor_rejects_injection_in_env(self, tmp_path: Path, monkeypatch):
        """open_in_editor should reject malicious $EDITOR."""
        from polylogue.cli.editor import open_in_editor

        monkeypatch.setenv("EDITOR", "vim; rm -rf /tmp/pwned")
        monkeypatch.delenv("VISUAL", raising=False)

        test_file = tmp_path / "test.txt"
        test_file.write_text("safe content")

        # Should return False (validation failed), not raise
        result = open_in_editor(test_file)
        assert result is False

    def test_open_in_editor_allows_safe_editor(self, tmp_path: Path, monkeypatch):
        """open_in_editor should handle safe $EDITOR without throwing."""
        from polylogue.cli.editor import open_in_editor

        # Use a non-existent but safely-named editor
        monkeypatch.setenv("EDITOR", "nonexistent_safe_editor")
        monkeypatch.delenv("VISUAL", raising=False)

        test_file = tmp_path / "test.txt"
        test_file.write_text("content")

        # Should return False (editor doesn't exist), but not from validation error
        result = open_in_editor(test_file)
        assert result is False

    def test_open_in_editor_returns_false_when_no_editor(self, tmp_path: Path, monkeypatch):
        """open_in_editor should return False when no editor is set."""
        from polylogue.cli.editor import open_in_editor

        monkeypatch.delenv("EDITOR", raising=False)
        monkeypatch.delenv("VISUAL", raising=False)

        test_file = tmp_path / "test.txt"
        test_file.write_text("content")

        result = open_in_editor(test_file)
        assert result is False

    def test_open_in_editor_returns_false_when_file_missing(self, tmp_path: Path, monkeypatch):
        """open_in_editor should return False when file doesn't exist."""
        from polylogue.cli.editor import open_in_editor

        monkeypatch.setenv("EDITOR", "vim")
        monkeypatch.delenv("VISUAL", raising=False)

        missing_file = tmp_path / "missing.txt"

        result = open_in_editor(missing_file)
        assert result is False


class TestOpenInBrowserSecurity:
    """Tests for open_in_browser function security."""

    def test_open_in_browser_rejects_injection_in_polylogue_browser(self, tmp_path: Path, monkeypatch):
        """open_in_browser should reject malicious $POLYLOGUE_BROWSER."""
        from polylogue.cli.editor import open_in_browser

        monkeypatch.setenv("POLYLOGUE_BROWSER", "firefox; rm -rf /tmp")

        test_file = tmp_path / "test.html"
        test_file.write_text("<html></html>")

        # Should return False (validation failed), not raise
        result = open_in_browser(test_file)
        assert result is False

    def test_open_in_browser_rejects_backtick_injection(self, tmp_path: Path, monkeypatch):
        """open_in_browser should reject backtick injection."""
        from polylogue.cli.editor import open_in_browser

        monkeypatch.setenv("POLYLOGUE_BROWSER", "firefox `whoami`")

        test_file = tmp_path / "test.html"
        test_file.write_text("<html></html>")

        # Should return False (validation failed)
        result = open_in_browser(test_file)
        assert result is False

    def test_open_in_browser_allows_safe_browser(self, tmp_path: Path, monkeypatch):
        """open_in_browser should allow safe POLYLOGUE_BROWSER without throwing."""
        from polylogue.cli.editor import open_in_browser

        monkeypatch.setenv("POLYLOGUE_BROWSER", "firefox")

        test_file = tmp_path / "test.html"
        test_file.write_text("<html></html>")

        # Mock subprocess.Popen to avoid actually opening a browser
        with patch("polylogue.cli.editor.subprocess.Popen", return_value=MagicMock()) as mock_popen:
            result = open_in_browser(test_file)
            # Should succeed with mocked Popen
            assert result is True
            # Verify Popen was called with firefox and the file URI
            mock_popen.assert_called_once()
            cmd = mock_popen.call_args[0][0]
            assert cmd[0] == "firefox"
            assert "file://" in cmd[1]

    def test_open_in_browser_returns_false_on_invalid_path(self, monkeypatch):
        """open_in_browser should handle invalid paths gracefully."""
        from polylogue.cli.editor import open_in_browser

        monkeypatch.setenv("POLYLOGUE_BROWSER", "firefox")

        invalid_path = Path("\x00invalid")

        # Should return False gracefully
        result = open_in_browser(invalid_path)
        assert result is False
