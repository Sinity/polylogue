"""Behavioral proof for the ``polylogue setting`` get/set/list command (polylogue-at44)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import cast

from click.testing import CliRunner

from polylogue.cli import cli


def _run(args: list[str]) -> dict[str, object] | list[object]:
    result = CliRunner().invoke(cli, ["--plain", "setting", *args], catch_exceptions=False)
    assert result.exit_code == 0, result.output
    return cast("dict[str, object] | list[object]", json.loads(result.output))


def test_setting_get_reports_unset(cli_workspace: dict[str, Path]) -> None:
    payload = _run(["get", "subscription_tier", "--format", "json"])
    assert payload == {"setting_key": "subscription_tier", "value": None}


def test_setting_set_then_get_round_trips(cli_workspace: dict[str, Path]) -> None:
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


def test_setting_set_rejects_unknown_key(cli_workspace: dict[str, Path]) -> None:
    result = CliRunner().invoke(cli, ["--plain", "setting", "set", "not_a_real_setting", "x"])
    assert result.exit_code != 0
    assert "unknown setting key" in result.output


def test_setting_set_rejects_invalid_tier_value(cli_workspace: dict[str, Path]) -> None:
    result = CliRunner().invoke(cli, ["--plain", "setting", "set", "subscription_tier", "not-a-tier"])
    assert result.exit_code != 0
    assert "subscription_tier must be one of" in result.output
