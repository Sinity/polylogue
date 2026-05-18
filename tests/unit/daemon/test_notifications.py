"""Daemon notification dispatch tests."""

from __future__ import annotations

import pytest

from polylogue.daemon.health import HealthAlert, HealthSeverity, HealthTier
from polylogue.daemon.notifications import send_notifications


class RecordingBackend:
    def __init__(self) -> None:
        self.calls: list[tuple[list[HealthAlert], dict[str, object] | None]] = []

    def notify(self, alerts: list[HealthAlert], *, config: dict[str, object] | None = None) -> None:
        self.calls.append((alerts, config))


class FailingBackend:
    def notify(self, alerts: list[HealthAlert], *, config: dict[str, object] | None = None) -> None:
        raise RuntimeError("backend unavailable")


def _alert(name: str, severity: HealthSeverity) -> HealthAlert:
    return HealthAlert(
        check_name=name,
        tier=HealthTier.FAST,
        severity=severity,
        message=f"{name} {severity.value}",
        checked_at="2026-05-15T00:00:00+00:00",
        consecutive_failures=1 if severity != HealthSeverity.OK else 0,
    )


@pytest.mark.contract
def test_send_notifications_routes_alert_batch_to_backend() -> None:
    backend = RecordingBackend()
    config: dict[str, object] = {"notification_backend": "recording", "health_check_interval_s": 30}
    alerts = [
        _alert("daemon_liveness", HealthSeverity.OK),
        _alert("wal_size", HealthSeverity.WARNING),
        _alert("schema_version", HealthSeverity.CRITICAL),
    ]

    send_notifications(alerts, backend=backend, config=config)

    assert len(backend.calls) == 1
    delivered_alerts, delivered_config = backend.calls[0]
    assert delivered_alerts == alerts
    assert delivered_config == config


def test_send_notifications_rejects_unknown_backend() -> None:
    with pytest.raises(ValueError, match="unknown notification backend"):
        send_notifications([_alert("schema_version", HealthSeverity.ERROR)], config={"notification_backend": "smtp"})


def test_send_notifications_propagates_backend_failure() -> None:
    with pytest.raises(RuntimeError, match="backend unavailable"):
        send_notifications([_alert("schema_version", HealthSeverity.ERROR)], backend=FailingBackend())
