"""Tests for the daemon format-drift sentinel health check (polylogue-da1)."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from polylogue.daemon import health as health_module
from polylogue.daemon.health import HealthSeverity, _check_schema_drift_medium


@pytest.fixture(autouse=True)
def _reset_failure_counts() -> Iterator[None]:
    health_module._failure_counts.clear()
    yield
    health_module._failure_counts.clear()


def test_schema_drift_ok_when_sentinel_unavailable(
    workspace_env: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A fresh/synthetic archive with no ops.db data degrades to OK, not an alert."""
    import polylogue.cli.commands.status as status_module

    monkeypatch.setattr(status_module, "_schema_drift_status", lambda *a, **kw: {"available": False})
    alert = _check_schema_drift_medium()
    assert alert.severity == HealthSeverity.OK
    assert "unavailable" in alert.message


def test_schema_drift_ok_when_no_risky_origins(
    workspace_env: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import polylogue.cli.commands.status as status_module

    monkeypatch.setattr(
        status_module,
        "_schema_drift_status",
        lambda *a, **kw: {
            "available": True,
            "origins": [
                {"origin": "chatgpt-export", "total": 40, "risky": 0, "risky_rate": 0.0, "severity": "ok"},
            ],
        },
    )
    alert = _check_schema_drift_medium()
    assert alert.severity == HealthSeverity.OK
    assert alert.message == "no risky format drift"


def test_schema_drift_warning_when_one_origin_crosses_warn_threshold(
    workspace_env: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import polylogue.cli.commands.status as status_module

    monkeypatch.setattr(
        status_module,
        "_schema_drift_status",
        lambda *a, **kw: {
            "available": True,
            "origins": [
                {
                    "origin": "claude-code-session",
                    "total": 100,
                    "risky": 7,
                    "risky_rate": 0.07,
                    "severity": "warning",
                },
            ],
        },
    )
    alert = _check_schema_drift_medium()
    assert alert.severity == HealthSeverity.WARNING
    assert "claude-code-session" in alert.message
    assert "devtools lab schema generate/promote" in alert.message
    assert alert.consecutive_failures == 1


def test_schema_drift_error_outranks_warning_across_origins(
    workspace_env: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When multiple origins carry drift, the ERROR-severity one wins the message."""
    import polylogue.cli.commands.status as status_module

    monkeypatch.setattr(
        status_module,
        "_schema_drift_status",
        lambda *a, **kw: {
            "available": True,
            "origins": [
                {
                    "origin": "chatgpt-export",
                    "total": 100,
                    "risky": 7,
                    "risky_rate": 0.07,
                    "severity": "warning",
                },
                {
                    "origin": "codex-session",
                    "total": 100,
                    "risky": 25,
                    "risky_rate": 0.25,
                    "severity": "error",
                },
            ],
        },
    )
    alert = _check_schema_drift_medium()
    assert alert.severity == HealthSeverity.ERROR
    assert "codex-session" in alert.message


def test_schema_drift_recovers_after_error(
    workspace_env: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import polylogue.cli.commands.status as status_module

    monkeypatch.setattr(
        status_module,
        "_schema_drift_status",
        lambda *a, **kw: {
            "available": True,
            "origins": [
                {"origin": "codex-session", "total": 100, "risky": 25, "risky_rate": 0.25, "severity": "error"},
            ],
        },
    )
    bad = _check_schema_drift_medium()
    assert bad.consecutive_failures == 1

    monkeypatch.setattr(
        status_module,
        "_schema_drift_status",
        lambda *a, **kw: {
            "available": True,
            "origins": [
                {"origin": "codex-session", "total": 100, "risky": 0, "risky_rate": 0.0, "severity": "ok"},
            ],
        },
    )
    good = _check_schema_drift_medium()
    assert good.severity == HealthSeverity.OK
    assert good.consecutive_failures == 0
