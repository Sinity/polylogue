"""Daemon health aggregation contract tests."""

from __future__ import annotations

import pytest

from polylogue.daemon import health as health_module
from polylogue.daemon.health import HealthAlert, HealthSeverity, HealthTier, check_health


def _alert(name: str, tier: HealthTier, severity: HealthSeverity) -> HealthAlert:
    return HealthAlert(
        check_name=name,
        tier=tier,
        severity=severity,
        message=f"{name} {severity.value}",
        checked_at="2026-05-15T00:00:00+00:00",
        consecutive_failures=1 if severity != HealthSeverity.OK else 0,
    )


@pytest.mark.contract
def test_check_health_aggregates_requested_tiers_and_worst_status(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fast_alerts = [
        _alert("daemon_liveness", HealthTier.FAST, HealthSeverity.OK),
        _alert("wal_size", HealthTier.FAST, HealthSeverity.WARNING),
    ]
    medium_alerts = [_alert("raw_failures", HealthTier.MEDIUM, HealthSeverity.ERROR)]
    expensive_alerts = [_alert("db_integrity", HealthTier.EXPENSIVE, HealthSeverity.CRITICAL)]

    monkeypatch.setattr(health_module, "_run_fast_checks", lambda: fast_alerts)
    monkeypatch.setattr(health_module, "_run_medium_checks", lambda: medium_alerts)
    monkeypatch.setattr(health_module, "_run_expensive_checks", lambda: expensive_alerts)

    health = check_health(tiers={HealthTier.FAST, HealthTier.MEDIUM})

    assert health.overall_status == HealthSeverity.ERROR
    assert [alert.check_name for alert in health.alerts] == ["daemon_liveness", "wal_size", "raw_failures"]
    assert health.tier_summary == {
        "fast": {"ok": 1, "warning": 1, "error": 0, "critical": 0},
        "medium": {"ok": 0, "warning": 0, "error": 1, "critical": 0},
    }


@pytest.mark.contract
def test_check_health_default_runs_fast_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fast_alerts = [_alert("daemon_liveness", HealthTier.FAST, HealthSeverity.OK)]
    medium_called = False

    def fail_medium() -> list[HealthAlert]:
        nonlocal medium_called
        medium_called = True
        return [_alert("fts_readiness", HealthTier.MEDIUM, HealthSeverity.ERROR)]

    monkeypatch.setattr(health_module, "_run_fast_checks", lambda: fast_alerts)
    monkeypatch.setattr(health_module, "_run_medium_checks", fail_medium)
    monkeypatch.setattr(health_module, "_run_expensive_checks", lambda: [])

    health = check_health()

    assert [alert.check_name for alert in health.alerts] == ["daemon_liveness"]
    assert health.tier_summary == {"fast": {"ok": 1, "warning": 0, "error": 0, "critical": 0}}
    assert medium_called is False
