"""Behavioral proof for the ``polylogue setting`` get/set/list command (polylogue-at44).

``setting set`` writes ``user.db``, the archive's one irreplaceable tier, so it
lowers to the declared ``mutation.user.setting.set`` operation and the daemon
is its sole writer (polylogue-gjwto / polylogue-r29bv). The write tests here
therefore run a real daemon stack rather than an in-process writer; the reads
stay direct.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import pytest
from click.testing import CliRunner

from polylogue.cli import cli
from polylogue.cli.commands.setting import setting_command
from tests.infra.daemon_operations import cli_daemon_archive


def _run(args: list[str]) -> dict[str, object] | list[object]:
    result = CliRunner().invoke(cli, ["--plain", "setting", *args], catch_exceptions=False)

    assert result.exit_code == 0, result.output
    return cast("dict[str, object] | list[object]", json.loads(result.output))


def test_setting_get_reports_unset(cli_workspace: dict[str, Path]) -> None:
    payload = _run(["get", "subscription_tier", "--format", "json"])
    assert payload == {"setting_key": "subscription_tier", "value": None}


def test_setting_set_then_get_round_trips(cli_workspace: dict[str, Path], monkeypatch: pytest.MonkeyPatch) -> None:
    """The daemon write is what the direct reads then see.

    Anti-vacuity: point ``setting set`` at a second in-process writer and this
    still passes, which is why ``test_setting_set_refuses_without_a_daemon``
    below is the load-bearing half -- together they say the write happened
    *and* that it could only have happened through the daemon.
    """
    with cli_daemon_archive(cli_workspace["archive_root"], monkeypatch):
        written = _run(["set", "subscription_tier", "max_5x", "--format", "json"])
        assert isinstance(written, dict)
        assert written["setting_key"] == "subscription_tier"
        assert written["value"] == "max_5x"

        fetched = _run(["get", "subscription_tier", "--format", "json"])
        assert isinstance(fetched, dict)
        assert fetched["value"] == "max_5x"

        listed = _run(["list", "--format", "json"])
        assert isinstance(listed, list)
        assert listed == [written]


def test_setting_set_refuses_without_a_daemon(cli_workspace: dict[str, Path]) -> None:
    """No daemon means no write, not a second writer in the CLI process.

    Anti-vacuity: restore ``Polylogue.set_setting`` behind this command and the
    row lands with no daemon at all, so the assertion on the empty registry
    goes red. That is the whole defect polylogue-gjwto names.
    """
    result = CliRunner().invoke(cli, ["--plain", "setting", "set", "subscription_tier", "pro"])

    assert result.exit_code != 0, result.output
    assert "daemon" in result.output.lower()
    assert _run(["get", "subscription_tier", "--format", "json"]) == {
        "setting_key": "subscription_tier",
        "value": None,
    }


def test_setting_set_rejects_unknown_key(cli_workspace: dict[str, Path], monkeypatch: pytest.MonkeyPatch) -> None:
    with cli_daemon_archive(cli_workspace["archive_root"], monkeypatch):
        result = CliRunner().invoke(cli, ["--plain", "setting", "set", "not_a_real_setting", "x"])
    assert result.exit_code != 0
    assert "unknown setting key" in result.output


def test_setting_set_rejects_invalid_tier_value(
    cli_workspace: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    with cli_daemon_archive(cli_workspace["archive_root"], monkeypatch):
        result = CliRunner().invoke(cli, ["--plain", "setting", "set", "subscription_tier", "not-a-tier"])
    assert result.exit_code != 0
    assert "subscription_tier must be one of" in result.output


@pytest.mark.parametrize("subcommand", ("get", "set", "list"))
def test_setting_subcommands_expose_standard_format_alias(subcommand: str) -> None:
    """Every settings read/write route accepts the CLI-wide ``-f`` shorthand."""
    result = CliRunner().invoke(setting_command, [subcommand, "--help"])

    assert result.exit_code == 0, result.output
    assert "-f, --format" in result.output
