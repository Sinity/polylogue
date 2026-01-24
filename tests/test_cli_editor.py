"""Tests for polylogue.cli.editor module security."""

from pathlib import Path

import pytest


class TestEditorCommandValidation:
    """Tests for editor command validation."""

    def test_validate_command_rejects_semicolon(self):
        """Command with semicolon should be rejected."""
        from polylogue.cli.editor import validate_command

        with pytest.raises(ValueError, match="unsafe shell metacharacters"):
            validate_command("vim; rm -rf /tmp/pwned")

    def test_validate_command_rejects_pipe(self):
        """Command with pipe should be rejected."""
        from polylogue.cli.editor import validate_command

        with pytest.raises(ValueError, match="unsafe shell metacharacters"):
            validate_command("vim | cat /etc/passwd")

    def test_validate_command_rejects_backticks(self):
        """Command with backticks should be rejected."""
        from polylogue.cli.editor import validate_command

        with pytest.raises(ValueError, match="unsafe shell metacharacters"):
            validate_command("vim `whoami`")

    def test_validate_command_rejects_dollar_parens(self):
        """Command with $() should be rejected."""
        from polylogue.cli.editor import validate_command

        with pytest.raises(ValueError, match="unsafe shell metacharacters"):
            validate_command("vim $(cat /etc/passwd)")

    def test_validate_command_rejects_ampersand(self):
        """Command with ampersand should be rejected."""
        from polylogue.cli.editor import validate_command

        with pytest.raises(ValueError, match="unsafe shell metacharacters"):
            validate_command("vim & malicious_command")

    def test_validate_command_rejects_double_ampersand(self):
        """Command with && should be rejected."""
        from polylogue.cli.editor import validate_command

        with pytest.raises(ValueError, match="unsafe shell metacharacters"):
            validate_command("vim && rm -rf /")

    def test_validate_command_rejects_double_pipe(self):
        """Command with || should be rejected."""
        from polylogue.cli.editor import validate_command

        with pytest.raises(ValueError, match="unsafe shell metacharacters"):
            validate_command("vim || evil_command")

    def test_validate_command_rejects_redirect_out(self):
        """Command with > redirect should be rejected."""
        from polylogue.cli.editor import validate_command

        with pytest.raises(ValueError, match="unsafe shell metacharacters"):
            validate_command("vim > /tmp/output")

    def test_validate_command_rejects_redirect_in(self):
        """Command with < redirect should be rejected."""
        from polylogue.cli.editor import validate_command

        with pytest.raises(ValueError, match="unsafe shell metacharacters"):
            validate_command("vim < /tmp/input")

    def test_validate_command_rejects_brace_expansion(self):
        """Command with braces should be rejected."""
        from polylogue.cli.editor import validate_command

        with pytest.raises(ValueError, match="unsafe shell metacharacters"):
            validate_command("vim {/tmp/a,/tmp/b}")

    def test_validate_command_rejects_bracket_glob(self):
        """Command with brackets should be rejected."""
        from polylogue.cli.editor import validate_command

        with pytest.raises(ValueError, match="unsafe shell metacharacters"):
            validate_command("vim /tmp/[abc]")

    def test_validate_command_rejects_backslash_escape(self):
        """Command with backslash should be rejected."""
        from polylogue.cli.editor import validate_command

        with pytest.raises(ValueError, match="unsafe shell metacharacters"):
            validate_command("vim \\n")

    def test_validate_command_rejects_exclamation(self):
        """Command with exclamation (history expansion) should be rejected."""
        from polylogue.cli.editor import validate_command

        with pytest.raises(ValueError, match="unsafe shell metacharacters"):
            validate_command("vim !!")

    def test_validate_command_rejects_empty_string(self):
        """Empty command should be rejected."""
        from polylogue.cli.editor import validate_command

        with pytest.raises(ValueError, match="cannot be empty"):
            validate_command("")

    def test_validate_command_rejects_whitespace_only(self):
        """Whitespace-only command should be rejected."""
        from polylogue.cli.editor import validate_command

        with pytest.raises(ValueError, match="cannot be empty"):
            validate_command("   ")

    def test_validate_command_allows_simple_editor(self):
        """Simple editor name should be allowed."""
        from polylogue.cli.editor import validate_command

        # Should not raise
        validate_command("vim")

    def test_validate_command_allows_editor_with_path(self):
        """Editor with path should be allowed."""
        from polylogue.cli.editor import validate_command

        # Should not raise
        validate_command("/usr/bin/vim")

    def test_validate_command_allows_editor_with_options(self):
        """Editor with safe options should be allowed."""
        from polylogue.cli.editor import validate_command

        # Should not raise
        validate_command("vim -u NONE")

    def test_validate_command_allows_nano(self):
        """nano editor should be allowed."""
        from polylogue.cli.editor import validate_command

        # Should not raise
        validate_command("nano")

    def test_validate_command_allows_nvim(self):
        """nvim editor should be allowed."""
        from polylogue.cli.editor import validate_command

        # Should not raise
        validate_command("nvim")

    def test_validate_command_allows_vscode(self):
        """VS Code with --wait should be allowed."""
        from polylogue.cli.editor import validate_command

        # Should not raise
        validate_command("code --wait")

    def test_validate_command_allows_emacs(self):
        """emacs with options should be allowed."""
        from polylogue.cli.editor import validate_command

        # Should not raise
        validate_command("emacs -nw")

    def test_validate_command_custom_context(self):
        """Custom context should appear in error message."""
        from polylogue.cli.editor import validate_command

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
        from unittest.mock import MagicMock, patch

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
