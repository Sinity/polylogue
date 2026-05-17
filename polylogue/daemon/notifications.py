"""Notification backends for daemon health alerts and operational events.

Two backends ship today:

- ``"log"`` (default) — writes alerts to the structured logger.
- ``"webhook"`` — POSTs a typed JSON envelope to a user-configured URL.

Backend selection is driven by the runtime config's ``notification_backend``
key. The webhook backend reads ``notification_webhook_url`` from the same
config dict.

Backends are intentionally fire-and-forget per dispatch call. Dedup,
rate-limiting, and severity-routing live in sibling concerns; the periodic
health loop's ``except Exception`` boundary catches persistent failures.
"""

from __future__ import annotations

import time
from typing import Protocol

import httpx

from polylogue.daemon.health import HealthAlert, HealthSeverity
from polylogue.logging import get_logger
from polylogue.version import POLYLOGUE_VERSION

logger = get_logger(__name__)


WEBHOOK_TIMEOUT_S = 5.0
"""Per-attempt HTTP timeout for the webhook backend."""

WEBHOOK_RETRY_BACKOFF_S = 0.5
"""Backoff between the initial attempt and the single retry."""


class NotificationBackend(Protocol):
    """Protocol for notification backends."""

    def notify(self, alerts: list[HealthAlert], *, config: dict[str, object] | None = None) -> None:
        """Deliver one or more health alerts."""
        ...


class _HTTPPoster(Protocol):
    """Minimal client surface used by :class:`WebhookNotificationBackend`.

    ``httpx.Client`` satisfies this protocol structurally; tests inject
    lightweight stubs without depending on the full client surface.
    """

    def post(self, url: str, *, json: dict[str, object], timeout: float) -> httpx.Response: ...


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


class WebhookConfigError(ValueError):
    """Raised when the webhook backend is selected without a valid URL."""


class WebhookNotificationBackend:
    """POSTs a typed JSON envelope to a user-configured URL.

    Envelope shape:

    .. code-block:: json

        {
          "alerts": [{"check_name": "...", "tier": "...", "severity": "...",
                       "message": "...", "checked_at": "...",
                       "consecutive_failures": N}, ...],
          "emitted_at": "<unix epoch seconds, float>",
          "daemon_version": "<polylogue version string>"
        }

    The backend uses a single bounded-timeout HTTP POST. On transient
    failure (network error or HTTP 5xx), it retries exactly once after a
    short backoff. Persistent failure raises ``httpx.HTTPError``; callers
    are expected to swallow the exception at the periodic-loop boundary
    (see ``polylogue/daemon/cli.py`` health loop).

    The HTTP client is injected for tests; production uses
    :class:`httpx.Client`.
    """

    def __init__(
        self,
        url: str,
        *,
        timeout: float = WEBHOOK_TIMEOUT_S,
        max_retries: int = 1,
        backoff_s: float = WEBHOOK_RETRY_BACKOFF_S,
        client: _HTTPPoster | None = None,
    ) -> None:
        if not url or not isinstance(url, str):
            raise WebhookConfigError(
                "notification_webhook_url must be a non-empty string when notification_backend='webhook'"
            )
        self._url = url
        self._timeout = float(timeout)
        self._max_retries = int(max_retries)
        self._backoff_s = float(backoff_s)
        self._client = client

    def notify(self, alerts: list[HealthAlert], *, config: dict[str, object] | None = None) -> None:
        if not alerts:
            return
        envelope = _build_envelope(alerts)
        attempts = self._max_retries + 1
        last_error: Exception | None = None
        for attempt in range(attempts):
            try:
                response = self._post(envelope)
            except httpx.HTTPError as err:
                last_error = err
                logger.warning(
                    "daemon.notifications.webhook: transient error on attempt %d/%d: %s",
                    attempt + 1,
                    attempts,
                    err,
                )
            else:
                if response.status_code < 500:
                    response.raise_for_status()
                    logger.debug(
                        "daemon.notifications.webhook: delivered %d alert(s) (status=%d)",
                        len(alerts),
                        response.status_code,
                    )
                    return
                last_error = httpx.HTTPStatusError(
                    f"server error {response.status_code}", request=response.request, response=response
                )
                logger.warning(
                    "daemon.notifications.webhook: server %d on attempt %d/%d",
                    response.status_code,
                    attempt + 1,
                    attempts,
                )
            if attempt < attempts - 1 and self._backoff_s > 0:
                time.sleep(self._backoff_s)
        assert last_error is not None  # one of the branches above set it
        raise last_error

    def _post(self, envelope: dict[str, object]) -> httpx.Response:
        if self._client is not None:
            return self._client.post(self._url, json=envelope, timeout=self._timeout)
        with httpx.Client(timeout=self._timeout) as client:
            return client.post(self._url, json=envelope)


def _build_envelope(alerts: list[HealthAlert]) -> dict[str, object]:
    """Build the JSON envelope POSTed by :class:`WebhookNotificationBackend`."""
    return {
        "alerts": [alert.model_dump(mode="json") for alert in alerts],
        "emitted_at": time.time(),
        "daemon_version": POLYLOGUE_VERSION,
    }


def _resolve_backend(backend_name: str, config: dict[str, object] | None = None) -> NotificationBackend:
    """Resolve a notification backend name to an instance.

    Args:
        backend_name: Backend identifier (e.g. ``"log"`` or ``"webhook"``).
        config: Optional runtime config dict, consulted by backends that
            require additional keys (the webhook backend reads
            ``notification_webhook_url``).

    Returns:
        An instance of the requested backend.

    Raises:
        ValueError: If the backend name is unknown.
        WebhookConfigError: If ``"webhook"`` is selected without a usable
            ``notification_webhook_url`` value.
    """
    if backend_name == "log":
        return LogNotificationBackend()
    if backend_name == "webhook":
        url_value: object = (config or {}).get("notification_webhook_url")
        if not isinstance(url_value, str) or not url_value:
            raise WebhookConfigError("notification_backend='webhook' requires notification_webhook_url in config")
        return WebhookNotificationBackend(url_value)
    supported = ["log", "webhook"]
    raise ValueError(f"unknown notification backend: {backend_name!r}. Supported backends: {', '.join(supported)}")


def send_notifications(
    alerts: list[HealthAlert],
    *,
    backend: NotificationBackend | None = None,
    config: dict[str, object] | None = None,
) -> None:
    """Send health alert notifications through the configured backend.

    Args:
        alerts: Health alerts to deliver.
        backend: Notification backend. If None, resolved from config or
                 defaults to ``LogNotificationBackend``.
        config: Optional runtime config dict (forwarded to the resolved
                backend; required for the webhook backend to read its URL).
    """
    if backend is not None:
        _backend = backend
    elif config is not None and isinstance(config.get("notification_backend"), str):
        _backend = _resolve_backend(str(config["notification_backend"]), config=config)
    else:
        _backend = LogNotificationBackend()

    if alerts:
        _backend.notify(alerts, config=config)
    else:
        logger.debug("daemon.health: no alerts to notify")


__all__ = [
    "LogNotificationBackend",
    "NotificationBackend",
    "WEBHOOK_RETRY_BACKOFF_S",
    "WEBHOOK_TIMEOUT_S",
    "WebhookConfigError",
    "WebhookNotificationBackend",
    "_resolve_backend",
    "send_notifications",
]
