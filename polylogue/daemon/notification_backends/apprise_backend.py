"""Apprise notification backend — fan-out to 100+ services.

`Apprise <https://github.com/caronc/apprise>`_ is a unified notification
library. One adapter covers Pushover, Discord, Slack, ntfy, Matrix,
Telegram, Mastodon, Gotify, Pushbullet, Mattermost, and dozens more by
URL scheme.

Configuration is a list of Apprise URLs in
``notification_apprise_urls`` (TOML alias ``apprise_urls``). Each URL is
added to a single :class:`apprise.Apprise` instance and notified per
alert. Per-alert dispatch (rather than per batch) lets Apprise format the
title from the severity-aware subject without manual aggregation.

The ``apprise`` package is optional; ``BackendUnavailableError`` is
raised at construction time when the import fails. The dispatcher
isolates the error from sibling backends.
"""

from __future__ import annotations

import json
from typing import Any, Protocol

from polylogue.daemon.health import HealthAlert, HealthSeverity
from polylogue.daemon.notification_backends import (
    BackendConfigError,
    BackendUnavailableError,
    build_envelope,
)
from polylogue.logging import get_logger

logger = get_logger(__name__)


class AppriseConfigError(BackendConfigError):
    """Raised when no Apprise URLs are configured."""


SEVERITY_TO_NOTIFY_TYPE: dict[HealthSeverity, str] = {
    HealthSeverity.OK: "info",
    HealthSeverity.WARNING: "warning",
    HealthSeverity.ERROR: "failure",
    HealthSeverity.CRITICAL: "failure",
}


class _AppriseClient(Protocol):
    """Minimal Apprise surface used by the backend (tests inject a stub)."""

    def add(self, url: str) -> bool: ...
    def notify(
        self,
        body: str,
        *,
        title: str,
        notify_type: str,
        body_format: str,
    ) -> bool: ...


def _resolve_default_client() -> _AppriseClient:
    try:
        import apprise
    except ImportError as err:  # pragma: no cover - environment-dependent
        raise BackendUnavailableError(
            "notification_backend='apprise' requires the 'apprise' Python package (pip install apprise)"
        ) from err
    client: _AppriseClient = apprise.Apprise()
    return client


class AppriseNotificationBackend:
    """Fan out alerts through a single ``apprise.Apprise`` instance."""

    def __init__(
        self,
        urls: tuple[str, ...],
        *,
        client: _AppriseClient | None = None,
        include_envelope: bool = True,
    ) -> None:
        if not urls:
            raise AppriseConfigError("notification_backend='apprise' requires notification_apprise_urls in config")
        self._client = client if client is not None else _resolve_default_client()
        for url in urls:
            ok = self._client.add(url)
            if not ok:
                logger.warning("daemon.notifications.apprise: failed to add URL %r", url)
        self._urls = tuple(urls)
        self._include_envelope = bool(include_envelope)

    def notify(self, alerts: list[HealthAlert], *, config: dict[str, object] | None = None) -> None:
        if not alerts:
            return
        # Per-alert dispatch so each platform's "title" maps cleanly.
        for alert in alerts:
            title = f"[polylogue] {alert.severity.value.upper()} {alert.check_name}"
            body = alert.message
            if self._include_envelope:
                payload: dict[str, Any] = dict(build_envelope([alert]))
                body = f"{alert.message}\n\n{json.dumps(payload, sort_keys=True, default=str)}"
            ok = self._client.notify(
                body=body,
                title=title,
                notify_type=SEVERITY_TO_NOTIFY_TYPE.get(alert.severity, "info"),
                body_format="text",
            )
            if not ok:
                logger.warning(
                    "daemon.notifications.apprise: notify returned false for %s",
                    alert.check_name,
                )


def build_apprise_backend(config: dict[str, object] | None) -> AppriseNotificationBackend:
    cfg = config or {}
    raw = cfg.get("notification_apprise_urls")
    if isinstance(raw, str):
        urls = tuple(s.strip() for s in raw.split(",") if s.strip())
    elif isinstance(raw, (list, tuple)):
        urls = tuple(str(s).strip() for s in raw if str(s).strip())
    else:
        urls = ()
    if not urls:
        raise AppriseConfigError(
            "notification_backend='apprise' requires notification_apprise_urls (list or comma-separated string)"
        )
    return AppriseNotificationBackend(urls)


__all__ = [
    "SEVERITY_TO_NOTIFY_TYPE",
    "AppriseConfigError",
    "AppriseNotificationBackend",
    "build_apprise_backend",
]
