"""Notification backends for daemon health alerts and operational events.

The initial backend is log-based — all alerts are written to the structured
logger. This provides a baseline notifications surface that can be extended
with webhook, email, or other transports without changing the alert model.

Backend selection is driven by the runtime config (#829).
"""

from __future__ import annotations

from typing import Protocol

from polylogue.daemon.health import HealthAlert, HealthSeverity
from polylogue.logging import get_logger

logger = get_logger(__name__)


class NotificationBackend(Protocol):
    """Protocol for notification backends."""

    def notify(self, alerts: list[HealthAlert], *, config: dict[str, object] | None = None) -> None:
        """Deliver one or more health alerts."""
        ...


class LogNotificationBackend:
    """Logs all non-OK alerts to the structured logger."""

    def notify(self, alerts: list[HealthAlert], *, config: dict[str, object] | None = None) -> None:
        for alert in alerts:
            if alert.severity == HealthSeverity.OK:
                logger.debug(
                    "daemon.health",
                    check_name=alert.check_name,
                    severity=alert.severity.value,
                    tier=alert.tier.value,
                    consecutive_failures=alert.consecutive_failures,
                )
            elif alert.severity == HealthSeverity.WARNING:
                logger.warning(
                    "daemon.health: %s [%s] %s",
                    alert.check_name,
                    alert.severity.value,
                    alert.message,
                )
            elif alert.severity == HealthSeverity.ERROR or alert.severity == HealthSeverity.CRITICAL:
                logger.error(
                    "daemon.health: %s [%s] %s",
                    alert.check_name,
                    alert.severity.value,
                    alert.message,
                )


def send_notifications(
    alerts: list[HealthAlert],
    *,
    backend: NotificationBackend | None = None,
    config: dict[str, object] | None = None,
) -> None:
    """Send health alert notifications through the configured backend.

    Args:
        alerts: Health alerts to deliver.
        backend: Notification backend. Defaults to ``LogNotificationBackend``.
        config: Optional runtime config dict (reserved for future backends).
    """
    _backend = backend or LogNotificationBackend()
    if alerts:
        _backend.notify(alerts, config=config)
    else:
        logger.debug("daemon.health: no alerts to notify")


__all__ = [
    "LogNotificationBackend",
    "NotificationBackend",
    "send_notifications",
]
