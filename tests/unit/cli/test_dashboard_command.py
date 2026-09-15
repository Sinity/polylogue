"""Dashboard command product-surface contracts."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from polylogue.cli.commands.dashboard import dashboard_command
from polylogue.cli.operation_kernel import OperationResult
from polylogue.cli.shared.types import AppEnv
from polylogue.config import Config
from polylogue.services import RuntimeServices


def _make_env() -> AppEnv:
    ui: Any = MagicMock()
    ui.plain = True
    # #3079 removed the ambient config fallback ``AppEnv``/``RuntimeServices``
    # used to have -- a bare ``AppEnv(ui=ui)`` now raises ConfigError the
    # first time ``env.polylogue``/``env.config`` is touched (the default
    # launch path constructs ``PolylogueApp(polylogue=env.polylogue)``).
    # Supply an explicit, disposable Config via the documented "explicit
    # library caller" compatibility path (polylogue-c66i).
    root = Path(tempfile.mkdtemp(prefix="polylogue-dashboard-test-"))
    config = Config(archive_root=root, render_root=root, sources=[])
    return AppEnv(ui=ui, services=RuntimeServices(config=config))


def _served(mode: str) -> Any:
    """Stand in for the status operation, served by the named authority."""

    def _call(config: Any, operation: str, payload: dict[str, object], **kwargs: Any) -> OperationResult:
        del config, payload, kwargs
        return OperationResult(operation, {}, {"mode": mode, "class": "read"})

    return _call


def _patch_probe(mode: str) -> Any:
    return patch("polylogue.cli.operation_kernel.configured_read_operation", new=_served(mode))


def _probe_raises(exc: BaseException) -> Any:
    def _call(config: Any, operation: str, payload: dict[str, object], **kwargs: Any) -> OperationResult:
        del config, operation, payload, kwargs
        raise exc

    return patch("polylogue.cli.operation_kernel.configured_read_operation", new=_call)


def test_dashboard_status_json_reports_terminal_surface_and_no_web_launch(monkeypatch: pytest.MonkeyPatch) -> None:
    """``dashboard --status --format json`` is an operator-visible contract."""

    monkeypatch.setenv("POLYLOGUE_DAEMON_URL", "http://127.0.0.1:8766")
    runner = CliRunner()
    with _probe_raises(OSError("offline")):
        result = runner.invoke(dashboard_command, ["--status", "--format", "json"], obj=_make_env())

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["surface"] == "terminal_tui"
    assert payload["daemon_api_url"] == "http://127.0.0.1:8766"
    assert payload["reader_surface"] == "terminal_tui"
    assert payload["web_reader_url"] is None
    assert payload["web_reader_launch_attempted"] is False
    assert payload["daemon_api_reachable"] is False
    assert "OSError" in payload["failure_reason"]


def test_dashboard_direct_served_status_is_not_reported_as_a_reachable_daemon() -> None:
    """A direct in-process read is not evidence that a daemon is running.

    The ``status`` operation declares ``DaemonFallback.DIRECT_READ``, so it
    succeeds with no daemon at all. Reachability is read off the result's
    authority mode instead.

    Anti-vacuity: reporting reachability from "the dispatch did not raise"
    makes this red, because this probe succeeds and is served directly.
    """
    runner = CliRunner()
    with _patch_probe("direct"):
        result = runner.invoke(dashboard_command, ["--status", "--format", "json"], obj=_make_env())

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["daemon_api_reachable"] is False
    assert "direct" in str(payload["failure_reason"])


def test_dashboard_default_prints_evidence_before_launching_tui() -> None:
    """Default launch prints what surface is starting before entering Textual."""

    class FakeApp:
        def __init__(self, polylogue: object) -> None:
            self.polylogue = polylogue

        def run(self) -> None:
            pass

    runner = CliRunner()
    with (
        _probe_raises(OSError("offline")),
        patch("polylogue.ui.tui.app.PolylogueApp", FakeApp),
    ):
        result = runner.invoke(dashboard_command, [], obj=_make_env())

    assert result.exit_code == 0, result.output
    assert "Dashboard surface: terminal TUI (Textual)" in result.output
    assert "Daemon status-operation probe" in result.output
    assert "Web reader launch: not attempted by this command" in result.output
    assert "Readiness: degraded" in result.output
    # polylogue-jnj.8: degraded readiness teaches the prerequisite and a
    # no-TUI, no-daemon fallback instead of only stating "degraded".
    assert "Prerequisite: start the daemon with `polylogued run`" in result.output
    assert "CLI fallback: `polylogue find QUERY then read`" in result.output
    assert "Launching Textual dashboard in this terminal." in result.output


def test_dashboard_status_degraded_teaches_prerequisite_and_cli_fallback() -> None:
    """``--status`` (plain text) is the readiness surface without launching the TUI.

    Exercised via ``--status`` rather than the default launch path so this
    stays independent of ``PolylogueApp`` construction (polylogue-jnj.8 AC4
    only concerns the printed readiness guidance, not the TUI launch itself).
    """
    runner = CliRunner()
    with _probe_raises(OSError("offline")):
        result = runner.invoke(dashboard_command, ["--status"], obj=_make_env())

    assert result.exit_code == 0, result.output
    assert "Readiness: degraded" in result.output
    assert "Prerequisite: start the daemon with `polylogued run`" in result.output
    assert "CLI fallback: `polylogue find QUERY then read`" in result.output


def test_dashboard_status_reachable_daemon_skips_degraded_guidance() -> None:
    """A reachable daemon does not print the degraded-only fallback hints."""
    runner = CliRunner()
    with _patch_probe("daemon"):
        result = runner.invoke(dashboard_command, ["--status"], obj=_make_env())

    assert result.exit_code == 0, result.output
    assert "Readiness: daemon API reachable" in result.output
    assert "Prerequisite:" not in result.output
    assert "CLI fallback:" not in result.output


def test_dashboard_json_requires_status_mode() -> None:
    runner = CliRunner()
    result = runner.invoke(dashboard_command, ["--format", "json"], obj=_make_env())
    assert result.exit_code != 0
    assert "dashboard --format json requires --status" in result.output
