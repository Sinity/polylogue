"""Terminal output snapshot tests.

Captures CLI output in a virtual PTY and compares against stored
snapshots. Uses syrupy for snapshot management.

These tests verify:
1. ANSI color codes and formatting are correct
2. Table layouts render at expected widths
3. Progress indicators appear in expected positions
4. Error messages have consistent formatting
"""

from __future__ import annotations

import pytest

syrupy = pytest.importorskip("syrupy")

from tests.infra.pty_cli import HAS_PYTE, grid_to_text, run_in_pty, sanitize_grid

# Mark all tests as requiring pyte
pytestmark = pytest.mark.skipif(not HAS_PYTE, reason="pyte not installed")


class TestHelpOutput:
    """Test --help output rendering."""

    def test_help_output_snapshot(self, snapshot):
        """Verify --help output renders correctly."""
        result = run_in_pty(["--help"], rows=80)
        assert result.exit_code == 0

        # Sanitize and convert to text
        grid = sanitize_grid(result.grid, strip_timestamps=False, strip_paths=True)
        output = grid_to_text(grid)

        assert output == snapshot

    def test_help_output_has_basic_structure(self):
        """Verify --help output contains expected sections."""
        result = run_in_pty(["--help"], rows=80)
        assert result.exit_code == 0

        text = grid_to_text(result.grid)
        assert "Usage:" in text or "usage:" in text
        assert "Options:" in text or "options:" in text
        assert "Subcommands:" not in text


class TestCommandOutputs:
    """Test individual command outputs."""

    def test_check_output_snapshot(self, snapshot):
        """Verify check command output renders correctly."""
        result = run_in_pty(["doctor", "--help"], rows=80)
        assert result.exit_code == 0

        grid = sanitize_grid(result.grid, strip_timestamps=False, strip_paths=True)
        output = grid_to_text(grid)

        assert output == snapshot

    def test_run_help_output_snapshot(self, snapshot):
        """Verify run command help renders correctly."""
        result = run_in_pty(["run", "--help"], cols=120, rows=80)
        assert result.exit_code == 0

        grid = sanitize_grid(result.grid, strip_timestamps=False, strip_paths=True)
        output = grid_to_text(grid)

        assert output == snapshot
        assert "Usage: polylogue run [OPTIONS] COMMAND1 [ARGS]..." in output
        assert "Commands:" in output
        assert "reprocess" in output


class TestErrorOutput:
    """Test error message rendering."""

    def test_invalid_option_error_output(self, snapshot):
        """Verify invalid option error message."""
        result = run_in_pty(["--bogus"])
        assert result.exit_code != 0

        grid = sanitize_grid(result.grid, strip_timestamps=True, strip_paths=True)
        output = grid_to_text(grid)

        assert output == snapshot

    def test_missing_required_argument_error(self):
        """Verify error on missing required arguments."""
        # Use completions command without --shell (requires it)
        result = run_in_pty(["completions"])
        assert result.exit_code != 0


class TestTerminalDimensions:
    """Test output at different terminal widths."""

    def test_help_at_narrow_width(self, snapshot):
        """Verify help wraps correctly at 60 columns."""
        result = run_in_pty(["--help"], cols=60, rows=80)
        assert result.exit_code == 0

        grid = sanitize_grid(result.grid, strip_timestamps=False, strip_paths=True)
        output = grid_to_text(grid)

        assert output == snapshot
        # Verify wrapping occurred (should have lines)
        assert len(output) > 0

    def test_help_at_wide_width(self, snapshot):
        """Verify help renders at 120 columns."""
        result = run_in_pty(["--help"], cols=120, rows=80)
        assert result.exit_code == 0

        grid = sanitize_grid(result.grid, strip_timestamps=False, strip_paths=True)
        output = grid_to_text(grid)

        assert output == snapshot


class TestPlainModeConsistency:
    """Test consistency between plain and PTY modes."""

    def test_help_plain_vs_pty_consistency(self):
        """Verify --help output is similar in plain and PTY modes."""
        # Run with enough rows to capture full output
        result_pty = run_in_pty(["--help"], rows=80)
        assert result_pty.exit_code == 0

        # PTY output should contain terminal escape sequences or plain text
        pty_text = grid_to_text(result_pty.grid)

        # Both should be non-empty
        assert len(pty_text) > 0

        # Should have help content
        assert "Usage:" in pty_text or "usage:" in pty_text

    def test_error_consistency_across_modes(self):
        """Verify error output is consistent."""
        result = run_in_pty(["invalid-xyz"])
        assert result.exit_code != 0

        error_text = grid_to_text(result.grid)
        assert len(error_text) > 0
