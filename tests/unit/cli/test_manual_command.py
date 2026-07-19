"""Tests for the ``polylogue manual`` root command (polylogue-jnj.8).

``polylogue manual`` is the discoverable, human-facing route onto the same
generated CLI reference that ``polylogue --help-markdown`` already produces
(``polylogue/cli/help_markdown.py``) — there is exactly one generator, this
command only adds a name a cold reader would actually type.
"""

from __future__ import annotations

import json

from click.testing import CliRunner

from polylogue.cli.click_app import cli
from polylogue.version import POLYLOGUE_VERSION


def test_manual_command_registered_at_root() -> None:
    runner = CliRunner()
    result = runner.invoke(cli, ["--help"])
    assert result.exit_code == 0
    assert "manual" in result.output


def test_manual_text_matches_help_markdown_content() -> None:
    """Same generator, same content — no second hand-authored manual.

    Both invocations go through the same ``CliRunner`` so Click's terminal-
    width-dependent help wrapping is identical on both sides; comparing
    against a bare out-of-runner ``render_help_markdown()`` call is flaky
    because that call sees the *real* terminal width instead of the
    isolated one ``CliRunner`` establishes.
    """
    runner = CliRunner()
    manual_result = runner.invoke(cli, ["manual"], prog_name="polylogue")
    assert manual_result.exit_code == 0, manual_result.output
    flag_result = runner.invoke(cli, ["--help-markdown"], prog_name="polylogue")
    assert flag_result.exit_code == 0, flag_result.output

    assert flag_result.output in manual_result.output
    # Version/source identity is explicit (AC #3).
    assert POLYLOGUE_VERSION in manual_result.output
    assert "installed `polylogue`" in manual_result.output


def test_manual_covers_every_root_subcommand() -> None:
    runner = CliRunner()
    result = runner.invoke(cli, ["manual"], prog_name="polylogue")
    assert result.exit_code == 0, result.output
    assert "`polylogue tutorial`" in result.output
    assert "`polylogue demo" in result.output
    assert "`polylogue manual`" in result.output


def test_manual_json_format_is_machine_readable() -> None:
    runner = CliRunner()
    result = runner.invoke(cli, ["manual", "--format", "json"], prog_name="polylogue")
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["content_version"] == POLYLOGUE_VERSION
    assert "# `polylogue`" in payload["content"]
    assert payload["source"] == "installed `polylogue` command tree"


def test_manual_requires_no_archive() -> None:
    """Works offline, with no configured or seeded archive at all."""
    runner = CliRunner()
    result = runner.invoke(cli, ["manual"], prog_name="polylogue")
    assert result.exit_code == 0, result.output
