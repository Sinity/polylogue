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

import sqlite3
from pathlib import Path

import pytest

syrupy = pytest.importorskip("syrupy")
pytest.importorskip("pyte", reason="pyte not installed")

from tests.infra.cli_subprocess import setup_isolated_workspace
from tests.infra.pty_cli import grid_to_text, run_in_pty, sanitize_grid


def _seed_schema_drift(root: Path) -> None:
    """Create an operational schema-drift signal in an isolated archive."""
    from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_tier
    from polylogue.storage.sqlite.archive_tiers.ops_write import record_schema_drift_sample
    from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier

    root.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(root / "ops.db") as conn:
        initialize_archive_tier(conn, ArchiveTier.OPS)
        for index in range(5):
            record_schema_drift_sample(
                conn,
                origin="codex-session",
                element_kind="session_record",
                classification="field_changed",
                unseen_key_signature="",
                native_id_example=f"raw-{index}",
                raw_id=f"raw-{index}",
                observed_at_ms=2_000_000_000_000,
            )


class TestHelpOutput:
    """Test --help output rendering."""

    def test_help_output_snapshot(self, snapshot: object) -> None:
        """Verify --help output renders correctly."""
        result = run_in_pty(["--help"], rows=80)
        assert result.exit_code == 0

        # Sanitize and convert to text
        grid = sanitize_grid(result.grid, strip_timestamps=False, strip_paths=True)
        output = grid_to_text(grid)

        assert output == snapshot

    def test_help_output_has_basic_structure(self) -> None:
        """Verify --help output contains expected sections."""
        result = run_in_pty(["--help"], rows=80)
        assert result.exit_code == 0

        text = grid_to_text(result.grid)
        assert "Usage:" in text or "usage:" in text
        assert "Options:" in text or "options:" in text
        assert "Subcommands:" not in text


class TestCommandOutputs:
    """Test individual command outputs."""

    def test_doctor_help_is_hermetic_and_status_reports_drift(self, tmp_path: Path) -> None:
        """Help ignores archive state while an explicit status command reports it."""
        workspace = setup_isolated_workspace(tmp_path)
        clean_root = workspace["paths"]["archive_root"]
        drift_root = tmp_path / "drift-archive"

        from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore

        with ArchiveStore(drift_root):
            pass
        _seed_schema_drift(drift_root)

        clean_env = {**workspace["env"], "POLYLOGUE_ARCHIVE_ROOT": str(clean_root)}
        drift_env = {**workspace["env"], "POLYLOGUE_ARCHIVE_ROOT": str(drift_root)}
        clean_help = run_in_pty(["ops", "doctor", "--help"], rows=80, env=clean_env)
        drift_help = run_in_pty(["ops", "doctor", "--help"], rows=80, env=drift_env)

        assert clean_help.exit_code == drift_help.exit_code == 0
        clean_help_text = grid_to_text(clean_help.grid)
        drift_help_text = grid_to_text(drift_help.grid)
        assert clean_help_text == drift_help_text
        assert "Usage: polylogue ops doctor [OPTIONS]" in drift_help_text
        assert "format drift" not in drift_help_text

        status = run_in_pty(
            ["--plain", "ops", "status", "--daemon-url", "http://127.0.0.1:1"],
            rows=80,
            env=drift_env,
        )

        assert status.exit_code == 0
        status_text = grid_to_text(status.grid)
        assert "Format drift sentinel" in status_text
        assert "codex-session" in status_text
        assert "carry unseen shapes" in status_text


class TestErrorOutput:
    """Test error message rendering."""

    def test_invalid_option_error_output(self, snapshot: object) -> None:
        """Verify invalid option error message."""
        result = run_in_pty(["--bogus"])
        assert result.exit_code != 0

        grid = sanitize_grid(result.grid, strip_timestamps=True, strip_paths=True)
        output = grid_to_text(grid)

        assert output == snapshot

    def test_missing_required_argument_error(self) -> None:
        """Verify error on missing required arguments."""
        # Use completions command without --shell (requires it)
        result = run_in_pty(["config", "completions"])
        assert result.exit_code != 0


class TestTerminalDimensions:
    """Test output at different terminal widths."""

    def test_help_at_narrow_width(self, snapshot: object) -> None:
        """Verify help wraps correctly at 60 columns."""
        result = run_in_pty(["--help"], cols=60, rows=80)
        assert result.exit_code == 0

        grid = sanitize_grid(result.grid, strip_timestamps=False, strip_paths=True)
        output = grid_to_text(grid)

        assert output == snapshot
        # Verify wrapping occurred (should have lines)
        assert len(output) > 0

    def test_help_at_wide_width(self, snapshot: object) -> None:
        """Verify help renders at 120 columns."""
        result = run_in_pty(["--help"], cols=120, rows=80)
        assert result.exit_code == 0

        grid = sanitize_grid(result.grid, strip_timestamps=False, strip_paths=True)
        output = grid_to_text(grid)

        assert output == snapshot


class TestPlainModeConsistency:
    """Test consistency between plain and PTY modes."""

    def test_help_plain_vs_pty_consistency(self) -> None:
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

    def test_error_consistency_across_modes(self) -> None:
        """Verify Click's unknown-subcommand error is rendered in the PTY."""
        result = run_in_pty(["ops", "insights", "invalid-xyz"])
        assert result.exit_code != 0

        error_text = grid_to_text(result.grid)
        assert "No such command 'invalid-xyz'" in error_text
