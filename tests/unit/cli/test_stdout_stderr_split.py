"""Tests for stdout/stderr channel separation in the polylogue CLI.

Contract: when polylogue is used in a pipeline (e.g. ``polylogue --format json | jq .``),
result data must go to stdout and diagnostic/error text must go to stderr.
No Python tracebacks must appear in either channel.

CliRunner mixes stdout/stderr into result.output (Click 8.2+ removed mix_stderr).
Subprocess-based tests use capture_output=True for actual channel separation and
are marked @pytest.mark.slow.
"""

from __future__ import annotations

import subprocess
import sys

import pytest
from click.testing import CliRunner

from polylogue.cli.click_app import cli

TRACEBACK_SENTINEL = "Traceback (most recent call last)"


def _looks_like_json(text: str) -> bool:
    stripped = text.strip()
    return stripped.startswith("{") or stripped.startswith("[")


# ---------------------------------------------------------------------------
# Help output: exit code and no-traceback (CliRunner, fast)
# ---------------------------------------------------------------------------


_ALL_COMMANDS: tuple[tuple[str, ...], ...] = (
    (),  # polylogue root
    ("config",),
    ("count",),
    ("delete",),
    ("import",),
    (
        "ops",
        "insights",
    ),
    ("ops", "insights", "audit"),
    ("ops", "insights", "export"),
    ("ops", "insights", "status"),
    ("analyze", "insights"),
    ("analyze", "insights", "cost-rollups"),
    ("analyze", "insights", "costs"),
    ("analyze", "insights", "coverage"),
    ("analyze", "insights", "debt"),
    ("analyze", "insights", "phases"),
    ("analyze", "insights", "profiles"),
    ("analyze", "insights", "tags"),
    ("analyze", "insights", "threads"),
    ("analyze", "insights", "timeline"),
    ("analyze", "insights", "tool-usage"),
    ("analyze", "insights", "work-events"),
    ("ops",),
    ("ops", "auth"),
    ("ops", "backup"),
    ("config", "completions"),
    ("dashboard",),
    ("ops", "diagnostics"),
    ("analyze", "pace"),
    ("ops", "diagnostics", "space"),
    ("analyze", "tools"),
    ("analyze", "turns"),
    ("ops", "diagnostics", "workload"),
    ("ops", "doctor"),
    ("ops", "maintenance"),
    ("ops", "maintenance", "plan"),
    ("ops", "maintenance", "run"),
    ("ops", "reset"),
    ("ops", "status"),
    ("tutorial",),
    ("read",),
    ("continue",),
    ("mark",),
    ("mark", "candidates"),
)

_CMD_IDS = [" ".join(args) if args else "root" for args in _ALL_COMMANDS]


@pytest.mark.contract
class TestHelpExitCodeAndContent:
    """--help exits 0 with non-empty output and no traceback (covers cli.command.help claim)."""

    def test_version_exit_zero(self) -> None:
        result = CliRunner().invoke(cli, ["--version"])
        assert result.exit_code == 0, f"--version failed: {result.output!r}"
        assert result.output.strip()

    @pytest.mark.parametrize("cmd_args", _ALL_COMMANDS, ids=_CMD_IDS)
    def test_help_exits_zero_with_usage(self, cmd_args: tuple[str, ...]) -> None:
        """Every command: --help exits 0, shows Usage:, no traceback."""
        args = [*cmd_args, "--help"]
        result = CliRunner().invoke(cli, args)
        label = " ".join(args)
        assert result.exit_code == 0, f"{label} exited {result.exit_code}: {result.output!r}"
        assert result.output.strip(), f"{label} produced empty output"
        assert "Usage:" in result.output, f"{label} missing 'Usage:' in output"
        assert TRACEBACK_SENTINEL not in result.output, f"{label} leaked traceback"


# ---------------------------------------------------------------------------
# Error output: invalid args exit nonzero with no traceback (CliRunner, fast)
# ---------------------------------------------------------------------------


@pytest.mark.contract
class TestErrorOutputContent:
    """Invalid arguments produce non-zero exit with no traceback in combined output."""

    def test_invalid_limit_exits_nonzero(self) -> None:
        result = CliRunner().invoke(cli, ["--limit", "not_a_number"])
        assert result.exit_code != 0

    def test_invalid_limit_no_traceback(self) -> None:
        result = CliRunner().invoke(cli, ["--limit", "not_a_number"])
        assert TRACEBACK_SENTINEL not in result.output

    def test_unknown_format_exits_nonzero(self) -> None:
        result = CliRunner().invoke(cli, ["mark", "candidates", "list", "--format", "totally_invalid_xyz"])
        assert result.exit_code != 0

    def test_unknown_format_no_traceback(self) -> None:
        result = CliRunner().invoke(cli, ["mark", "candidates", "list", "--format", "totally_invalid_xyz"])
        assert TRACEBACK_SENTINEL not in result.output

    def test_run_unknown_subcommand_no_traceback(self) -> None:
        result = CliRunner().invoke(cli, ["run", "__invalid_sub_xyz__"])
        assert result.exit_code != 0
        assert TRACEBACK_SENTINEL not in result.output

    def test_error_output_not_json_on_invalid_arg(self) -> None:
        """On invalid arg error, output must not look like JSON (would break pipe consumers)."""
        result = CliRunner().invoke(cli, ["mark", "candidates", "list", "--format", "totally_invalid_xyz"])
        if result.output.strip():
            assert not _looks_like_json(result.output), (
                f"error output looks like JSON, which breaks pipe consumers: {result.output!r}"
            )


# ---------------------------------------------------------------------------
# Actual channel separation (subprocess, @pytest.mark.slow)
# These tests verify real stdout/stderr isolation for pipeline consumers.
# ---------------------------------------------------------------------------


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "polylogue", *args],
        capture_output=True,
        text=True,
        timeout=15,
    )


@pytest.mark.slow
@pytest.mark.contract
class TestChannelSeparation:
    """Subprocess-level channel separation: stdout for results, stderr for diagnostics."""

    def test_help_stdout_nonempty_stderr_empty(self) -> None:
        """--help: stdout has content, stderr is empty."""
        r = _run("--help")
        assert r.returncode == 0
        assert r.stdout.strip(), "--help produced empty stdout"
        assert not r.stderr.strip(), f"--help produced stderr: {r.stderr!r}"

    def test_version_stdout_nonempty_stderr_empty(self) -> None:
        """--version: stdout has content, stderr is empty."""
        r = _run("--version")
        assert r.returncode == 0
        assert r.stdout.strip()
        assert not r.stderr.strip(), f"--version produced stderr: {r.stderr!r}"

    def test_invalid_limit_stdout_has_no_json(self) -> None:
        """Non-numeric --limit: stdout must not contain JSON (would break pipe consumers)."""
        r = _run("--limit", "not_a_number")
        assert r.returncode != 0
        assert not _looks_like_json(r.stdout), f"Error stdout contains JSON: {r.stdout!r}"

    def test_no_traceback_in_stderr_on_invalid_arg(self) -> None:
        """Invalid argument: no Python traceback in stderr."""
        r = _run("--limit", "not_a_number")
        assert TRACEBACK_SENTINEL not in r.stderr

    def test_no_traceback_in_stdout_on_invalid_arg(self) -> None:
        """Invalid argument: no Python traceback in stdout."""
        r = _run("--limit", "not_a_number")
        assert TRACEBACK_SENTINEL not in r.stdout

    def test_help_no_traceback_in_either_channel(self) -> None:
        """--help: no traceback in stdout or stderr."""
        r = _run("--help")
        assert TRACEBACK_SENTINEL not in r.stdout
        assert TRACEBACK_SENTINEL not in r.stderr
