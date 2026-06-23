"""Dashboard command product-surface contracts."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from polylogue.cli.commands.dashboard import dashboard_command
from polylogue.cli.shared.types import AppEnv


def _make_env() -> AppEnv:
    ui: Any = MagicMock()
    ui.plain = True
    return AppEnv(ui=ui)


def test_dashboard_status_json_reports_terminal_surface_and_daemon_url(monkeypatch: pytest.MonkeyPatch) -> None:
    """``dashboard --status --format json`` is an operator-visible contract."""

    monkeypatch.setenv("POLYLOGUE_DAEMON_URL", "http://127.0.0.1:8766")
    runner = CliRunner()
    with patch("polylogue.cli.commands.dashboard.urlopen", side_effect=OSError("offline")):
        result = runner.invoke(dashboard_command, ["--status", "--format", "json"], obj=_make_env())

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["surface"] == "terminal_tui"
    assert payload["daemon_api_url"] == "http://127.0.0.1:8766"
    assert payload["web_reader_url"] == "http://127.0.0.1:8766"
    assert payload["daemon_api_reachable"] is False
    assert "OSError" in payload["failure_reason"]


def test_dashboard_default_prints_evidence_before_launching_tui() -> None:
    """Default launch prints what surface is starting before entering Textual."""

    class FakeApp:
        def __init__(self, polylogue: object) -> None:
            self.polylogue = polylogue

        def run(self) -> None:
            pass

    runner = CliRunner()
    with (
        patch("polylogue.cli.commands.dashboard.urlopen", side_effect=OSError("offline")),
        patch("polylogue.ui.tui.app.PolylogueApp", FakeApp),
    ):
        result = runner.invoke(dashboard_command, [], obj=_make_env())

    assert result.exit_code == 0, result.output
    assert "Dashboard surface: terminal TUI" in result.output
    assert "Readiness: degraded" in result.output
    assert "Launching Textual dashboard in this terminal." in result.output


def test_dashboard_json_requires_status_mode() -> None:
    runner = CliRunner()
    result = runner.invoke(dashboard_command, ["--format", "json"], obj=_make_env())
    assert result.exit_code != 0
    assert "dashboard --format json requires --status" in result.output
