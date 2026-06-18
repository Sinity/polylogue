"""Contracts for the ``--help`` surface of every registered CLI command.

These tests pin the human-facing help surface: every command path responds
to ``--help`` with exit 0, the output contains conventional Click sections,
no raw tracebacks ever appear in help text, and ``--version`` returns a
development-build identity that includes the commit hash (per
``CONTRIBUTING.md``).

Companion to ``test_plain_output_contract.py`` (plain rendering, #1060) and
``test_json_envelope_contract.py`` (machine surface, #1080).
"""

from __future__ import annotations

import re

import pytest
from click.testing import CliRunner

from polylogue.cli.click_app import cli
from polylogue.cli.command_inventory import CommandPath, iter_command_paths

pytestmark = pytest.mark.contract

TRACEBACK_SENTINEL = "Traceback (most recent call last)"

# Full recursive inventory of command paths (excludes the root group itself).
_COMMAND_PATHS = iter_command_paths(cli, include_root=False)
_COMMAND_IDS = [" ".join(p.path) for p in _COMMAND_PATHS]

# ``polylogue --version`` output shape, e.g. ``polylogue, version 0.1.0+abcd1234[-dirty]``.
_VERSION_RE = re.compile(
    r"^polylogue,\s+version\s+\d+\.\d+\.\d+\+[0-9a-f]{7,40}(?:-dirty)?\s*$",
)


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


class TestRootHelpStructure:
    """``polylogue --help`` exposes the conventional Click sections."""

    def test_root_help_contains_usage_and_commands(
        self,
        runner: CliRunner,
    ) -> None:
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert TRACEBACK_SENTINEL not in result.output
        assert "Usage:" in result.output, f"missing 'Usage:' section: {result.output!r}"
        assert "Options:" in result.output, f"missing 'Options:' section: {result.output!r}"
        assert "Commands:" in result.output, f"missing 'Commands:' section: {result.output!r}"

    def test_root_help_alias_short_flag(self, runner: CliRunner) -> None:
        """``-h`` is an alias for ``--help`` per the root group config."""
        long_form = runner.invoke(cli, ["--help"])
        short_form = runner.invoke(cli, ["-h"])
        assert long_form.exit_code == short_form.exit_code == 0
        assert long_form.output == short_form.output


class TestEverySubcommandHasHelp:
    """Every registered command path responds to ``--help`` without error."""

    @pytest.mark.parametrize("path", _COMMAND_PATHS, ids=_COMMAND_IDS)
    def test_subcommand_help_exits_zero_with_no_traceback(
        self,
        path: CommandPath,
        runner: CliRunner,
    ) -> None:
        result = runner.invoke(cli, [*path.path, "--help"])
        assert result.exit_code == 0, f"{path.display_name} --help exited {result.exit_code}: {result.output!r}"
        assert TRACEBACK_SENTINEL not in result.output, (
            f"{path.display_name} --help contained traceback: {result.output!r}"
        )
        # Click always emits a Usage line; assert it's present so a regression
        # to empty help text would fail this contract.
        assert "Usage:" in result.output, f"{path.display_name} --help missing Usage: section"

    @pytest.mark.parametrize("path", _COMMAND_PATHS, ids=_COMMAND_IDS)
    def test_subcommand_help_has_nonempty_description(
        self,
        path: CommandPath,
        runner: CliRunner,
    ) -> None:
        """Each command must have at least one non-boilerplate help line.

        A help text containing only ``Usage:`` and ``Options:`` is treated as
        a missing description and fails this contract; commands must document
        their purpose.
        """
        result = runner.invoke(cli, [*path.path, "--help"])
        # Count substantive lines: non-empty, not the Usage line, not a
        # standard section header.
        substantive: list[str] = []
        for raw in result.output.splitlines():
            line = raw.strip()
            if not line:
                continue
            if line.startswith("Usage:"):
                continue
            if line in ("Options:", "Commands:"):
                continue
            substantive.append(line)
        assert substantive, f"{path.display_name} --help has no substantive description lines: {result.output!r}"


class TestVersionContract:
    """``--version`` returns a development-build identity with commit hash."""

    def test_version_includes_commit_hash(
        self,
        runner: CliRunner,
    ) -> None:
        """Per CONTRIBUTING.md: version output must include commit + dirty marker."""
        result = runner.invoke(cli, ["--version"])
        assert result.exit_code == 0
        assert TRACEBACK_SENTINEL not in result.output
        line = result.output.strip()
        assert _VERSION_RE.match(line), (
            f"version line does not match expected shape 'polylogue, version <semver>+<sha>[-dirty]': {line!r}"
        )


class TestHelpStaysOnStdout:
    """``--help`` is a successful query, not an error -> stdout, not stderr."""

    def test_root_help_goes_to_stdout(self, runner: CliRunner) -> None:
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "Usage:" in result.stdout
        # Click writes successful --help to stdout; stderr stays empty.
        assert "Usage:" not in result.stderr

    def test_subcommand_help_goes_to_stdout(self, runner: CliRunner) -> None:
        result = runner.invoke(cli, ["ops", "doctor", "--help"])
        assert result.exit_code == 0
        assert "Usage:" in result.stdout
        assert "Usage:" not in result.stderr
