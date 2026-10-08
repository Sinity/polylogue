"""Daemon notification dispatch tests."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from polylogue.daemon.health import HealthAlert, HealthSeverity, HealthTier
from polylogue.daemon.notifications import ConfiguredNotificationBackend, send_notifications


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
        ConfiguredNotificationBackend().notify(
            [_alert("schema_version", HealthSeverity.ERROR)], config={"notification_backend": "smtp"}
        )


def test_send_notifications_propagates_backend_failure() -> None:
    with pytest.raises(RuntimeError, match="backend unavailable"):
        send_notifications([_alert("schema_version", HealthSeverity.ERROR)], backend=FailingBackend())


@pytest.mark.asyncio
async def test_configured_daemon_email_budget_survives_reloads_and_fanout(
    monkeypatch: pytest.MonkeyPatch, frozen_clock: Any
) -> None:
    """Rebuilding configured SMTP per tick sends three emails instead of one."""
    from polylogue import config as config_module
    from polylogue.daemon import cli, health, notifications
    from polylogue.daemon.notification_backends import email
    from tests.unit.daemon.test_notification_backends import _factory_returning, _RecordingSMTP

    smtp = _RecordingSMTP()
    monkeypatch.setattr(email, "_default_factory", lambda _ssl: _factory_returning(smtp))
    monkeypatch.setitem(notifications._BUILDERS, "webhook", lambda _config: RecordingBackend())
    settings: dict[str, object] = {
        "notification_backend": "email,log,webhook",
        "notification_email_host": "smtp.example",
        "notification_email_from": "alerts@example",
        "notification_email_to": ["ops@example"],
        "notification_email_max_per_hour": 1,
    }
    monkeypatch.setattr(
        config_module, "load_polylogue_config", lambda: SimpleNamespace(raw=dict(settings), health_check_tiers="fast")
    )
    monkeypatch.setattr(
        health,
        "check_health",
        lambda **_kwargs: health.DaemonHealth(
            overall_status=HealthSeverity.ERROR, alerts=[_alert("fixture", HealthSeverity.ERROR)]
        ),
    )
    monkeypatch.setattr(cli, "_daemon_stage_write_admission", lambda: None)
    monkeypatch.setattr(cli, "_run_with_stage_admission", lambda _admission, work: work())
    logs: list[list[HealthAlert]] = []
    monkeypatch.setattr(
        notifications.LogNotificationBackend, "notify", lambda self, alerts, **_kwargs: logs.append(alerts)
    )

    class Runner:
        async def run(self, _name: str, once: Any, **_kwargs: Any) -> None:
            await once()
            await once()  # fresh but equal runtime configuration
            settings["health_check_interval_s"] = 999
            settings["notification_backend"] = ["email", "log", "webhook"]  # equivalent selection syntax
            await once()
            assert len(smtp.messages) == 1
            assert len(logs) == 3  # email refusal does not stop other destinations
            settings["notification_webhook_url"] = "https://example/changed"
            await once()
            assert len(smtp.messages) == 1
            settings["notification_email_to"] = ["other@example"]
            await once()  # a genuine notification setting change replaces the backend graph
            assert len(smtp.messages) == 2
            assert smtp.messages[-1]["To"] == "other@example"
            await once()
            assert len(smtp.messages) == 2
            frozen_clock.advance(3601)
            await once()
            assert len(smtp.messages) == 3
            assert len(logs) == 7

    monkeypatch.setattr(cli, "daemon_periodic_runner", Runner)
    await cli._periodic_health_check(backend=ConfiguredNotificationBackend(), sources=())
