"""Notification dispatch for daemon health alerts and operational events.

This module is the routing layer; the concrete adapters live under
:mod:`polylogue.daemon.notification_backends`. Five backends ship:

- ``"log"`` (default) — structured logger.
- ``"webhook"`` — POST a typed JSON envelope (HMAC signed when a secret is set).
- ``"journald"`` — ``systemd.journal.send()`` with severity-priority mapping.
- ``"email"`` — SMTP/TLS with token-bucket rate limiting.
- ``"apprise"`` — fan-out to 100+ services via the Apprise library.

The runtime config key ``notification_backend`` accepts either a single
name or a list/comma-separated string of names. When more than one backend
is configured, the dispatcher wraps them in :class:`FanOutNotificationBackend`
so a failure in one does not abort the others.

The periodic health loop's ``except Exception`` boundary catches persistent
single-backend failures; the fan-out wrapper logs and isolates failures
per-backend so multi-destination configurations stay resilient.
"""

from __future__ import annotations

from collections.abc import Callable

from polylogue.daemon.health import HealthAlert, HealthSeverity
from polylogue.daemon.notification_backends import (
    BackendConfigError,
    BackendUnavailableError,
    NotificationBackend,
    build_envelope,
)
from polylogue.daemon.notification_backends.apprise_backend import (
    AppriseConfigError,
    AppriseNotificationBackend,
    build_apprise_backend,
)
from polylogue.daemon.notification_backends.email import (
    EmailConfigError,
    EmailNotificationBackend,
    build_email_backend,
)
from polylogue.daemon.notification_backends.journald import (
    JournaldNotificationBackend,
    build_journald_backend,
)
from polylogue.daemon.notification_backends.webhook import (
    WEBHOOK_RETRY_BACKOFF_S,
    WEBHOOK_TIMEOUT_S,
    WebhookConfigError,
    WebhookNotificationBackend,
    build_webhook_backend,
)
from polylogue.logging import DEBUG, ERROR, WARNING, emit

# ---------------------------------------------------------------------------
# Built-in backends
# ---------------------------------------------------------------------------


#: Alert severity to event level. A severity absent from this table is not
#: emitted at all, which is how an OK-with-no-interest alert stays silent.
_ALERT_LEVELS: dict[HealthSeverity, int] = {
    HealthSeverity.OK: DEBUG,
    HealthSeverity.WARNING: WARNING,
    HealthSeverity.ERROR: ERROR,
    HealthSeverity.CRITICAL: ERROR,
}


class LogNotificationBackend:
    """Emit each alert as a ``daemon.health.alert`` event.

    The severity decides the level (see :data:`_ALERT_LEVELS`), so an operator
    filtering the stream at WARNING sees exactly the alerts that are not OK
    without the backend having to decide what is worth printing.
    """

    def notify(self, alerts: list[HealthAlert], *, config: dict[str, object] | None = None) -> None:
        for alert in alerts:
            level = _ALERT_LEVELS.get(alert.severity)
            if level is None:
                continue
            emit(
                "daemon.health.alert",
                level=level,
                outcome="ok" if alert.severity is HealthSeverity.OK else "degraded",
                check_name=alert.check_name,
                severity=alert.severity.value,
                tier=alert.tier.value,
                failed=alert.consecutive_failures,
                error_detail=alert.message,
            )


# ---------------------------------------------------------------------------
# Fan-out wrapper
# ---------------------------------------------------------------------------


class FanOutNotificationBackend:
    """Dispatch alerts to multiple backends; isolate per-backend failures.

    Each child backend's ``notify`` is invoked in declaration order. A
    raised exception is logged with the backend's class name and the loop
    continues so a single broken destination does not block the others.
    The first observed exception is re-raised after all backends have been
    given a chance, so callers (and the periodic health loop) still see
    that *something* failed.
    """

    def __init__(self, backends: list[NotificationBackend]) -> None:
        if not backends:
            raise ValueError("FanOutNotificationBackend requires at least one backend")
        self._backends = list(backends)

    @property
    def backends(self) -> tuple[NotificationBackend, ...]:
        return tuple(self._backends)

    def notify(self, alerts: list[HealthAlert], *, config: dict[str, object] | None = None) -> None:
        first_error: BaseException | None = None
        for backend in self._backends:
            try:
                backend.notify(alerts, config=config)
            except Exception as err:  # per-backend isolation is the point of fan-out
                emit(
                    "daemon.notifications.backend_failed",
                    level=WARNING,
                    outcome="error",
                    reason="backend_raised",
                    backend=type(backend).__name__,
                    error_type=type(err).__name__,
                    error_detail=str(err),
                )
                if first_error is None:
                    first_error = err
        if first_error is not None:
            raise first_error


# ---------------------------------------------------------------------------
# Backend registry
# ---------------------------------------------------------------------------


_BackendBuilder = Callable[[dict[str, object] | None], NotificationBackend]

_BUILDERS: dict[str, _BackendBuilder] = {
    "log": lambda _config: LogNotificationBackend(),
    "webhook": build_webhook_backend,
    "journald": build_journald_backend,
    "email": build_email_backend,
    "apprise": build_apprise_backend,
}


def supported_backends() -> tuple[str, ...]:
    """Return the registered backend names in declaration order."""
    return tuple(_BUILDERS.keys())


def _parse_backend_spec(spec: object) -> list[str]:
    """Normalize the ``notification_backend`` config value into a name list."""
    if isinstance(spec, str):
        return [s.strip() for s in spec.split(",") if s.strip()]
    if isinstance(spec, (list, tuple)):
        return [str(s).strip() for s in spec if str(s).strip()]
    raise TypeError(f"notification_backend must be a string or list, got {type(spec).__name__}")


def _resolve_backend(backend_name: str, config: dict[str, object] | None = None) -> NotificationBackend:
    """Resolve a single backend name to an instance.

    Accepts a comma-separated string or a single name; if more than one name
    resolves, returns a :class:`FanOutNotificationBackend`.
    """
    names = _parse_backend_spec(backend_name)
    if not names:
        raise ValueError("notification_backend resolved to an empty list")
    instances: list[NotificationBackend] = []
    for name in names:
        builder = _BUILDERS.get(name)
        if builder is None:
            supported = ", ".join(_BUILDERS.keys())
            raise ValueError(f"unknown notification backend: {name!r}. Supported backends: {supported}")
        instances.append(builder(config))
    if len(instances) == 1:
        return instances[0]
    return FanOutNotificationBackend(instances)


def send_notifications(
    alerts: list[HealthAlert],
    *,
    backend: NotificationBackend | None = None,
    config: dict[str, object] | None = None,
) -> None:
    """Send health alert notifications through the configured backend(s).

    Args:
        alerts: Health alerts to deliver.
        backend: Optional pre-built backend instance. When provided, it is
            used verbatim and ``config`` is forwarded to ``notify``.
        config: Runtime config dict. The ``notification_backend`` key may be
            a name, comma-separated names, or a list; backend-specific keys
            (e.g. ``notification_webhook_url``,
            ``notification_apprise_urls``) are consumed during construction.
    """
    if backend is not None:
        _backend: NotificationBackend = backend
    elif config is not None and "notification_backend" in config:
        spec = config["notification_backend"]
        if isinstance(spec, (str, list, tuple)):
            _backend = _resolve_backend(
                spec if isinstance(spec, str) else ",".join(str(s) for s in spec), config=config
            )
        else:
            _backend = LogNotificationBackend()
    else:
        _backend = LogNotificationBackend()

    if alerts:
        _backend.notify(alerts, config=config)
    else:
        emit("daemon.health.notify", level=DEBUG, outcome="empty", alerts=0)


__all__ = [
    "AppriseConfigError",
    "AppriseNotificationBackend",
    "BackendConfigError",
    "BackendUnavailableError",
    "EmailConfigError",
    "EmailNotificationBackend",
    "FanOutNotificationBackend",
    "JournaldNotificationBackend",
    "LogNotificationBackend",
    "NotificationBackend",
    "WEBHOOK_RETRY_BACKOFF_S",
    "WEBHOOK_TIMEOUT_S",
    "WebhookConfigError",
    "WebhookNotificationBackend",
    "_resolve_backend",
    "build_envelope",
    "send_notifications",
    "supported_backends",
]
