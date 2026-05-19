"""Notification backend implementations for the daemon health loop.

Each backend is an independently importable adapter that implements the
:class:`~polylogue.daemon.notifications.NotificationBackend` protocol:
``notify(alerts, *, config=None) -> None``.

The factory :func:`build_backend` constructs a single backend from a name
and a config dict; the dispatcher in
:mod:`polylogue.daemon.notifications` composes one or more backends into a
fan-out wrapper.

Backends shipped:

- ``log`` (default) — structured logger
- ``webhook`` — POST a typed JSON envelope with HMAC-SHA256 signature
- ``journald`` — ``systemd.journal.send()`` with severity-priority mapping
- ``email`` — SMTP/TLS with token-bucket rate limiting
- ``apprise`` — fan-out via the Apprise library to 100+ services

Backends that depend on optional third-party packages (``systemd``,
``apprise``) raise a typed :class:`BackendUnavailableError` at construction
time when the package is not importable. The dispatcher reports the error
but does not abort fan-out across other backends.
"""

from __future__ import annotations

import time
from typing import Protocol

from polylogue.daemon.health import HealthAlert
from polylogue.version import POLYLOGUE_VERSION


class BackendUnavailableError(RuntimeError):
    """Raised when a backend's optional dependency is not installed."""


class BackendConfigError(ValueError):
    """Raised when a backend is selected without the config keys it needs."""


class NotificationBackend(Protocol):
    """Structural protocol every backend satisfies."""

    def notify(self, alerts: list[HealthAlert], *, config: dict[str, object] | None = None) -> None:
        """Deliver one or more health alerts."""
        ...


def build_envelope(alerts: list[HealthAlert]) -> dict[str, object]:
    """Build the canonical alert envelope shared by webhook/apprise/email payloads."""
    return {
        "alerts": [alert.model_dump(mode="json") for alert in alerts],
        "emitted_at": time.time(),
        "daemon_version": POLYLOGUE_VERSION,
    }


__all__ = [
    "BackendConfigError",
    "BackendUnavailableError",
    "NotificationBackend",
    "build_envelope",
]
