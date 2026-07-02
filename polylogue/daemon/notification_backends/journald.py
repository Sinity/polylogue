"""journald notification backend.

Uses ``systemd.journal.send()`` to emit one journald entry per alert,
mapping :class:`~polylogue.daemon.health.HealthSeverity` to a syslog
priority. Structured ``HealthAlert`` fields (check name, tier, severity,
consecutive failure count, checked-at timestamp) are forwarded as journald
metadata so ``journalctl POLYLOGUE_CHECK=fts_readiness`` works.

The backend is hard-dependent on the optional ``systemd`` Python binding
because that is the only public Python entrypoint into the journal
protocol. When the binding is absent the constructor raises
:class:`BackendUnavailableError` with operator guidance and the dispatcher
isolates it from other backends.

Severity → priority mapping (see ``man 3 sd-daemon``):

==========  ============  ===========
severity    priority      meaning
==========  ============  ===========
ok          6 LOG_INFO    informational
warning     4 LOG_WARNING warning condition
error       3 LOG_ERR     error condition
critical    2 LOG_CRIT    critical condition
==========  ============  ===========
"""

from __future__ import annotations

from typing import Protocol

from polylogue.daemon.health import HealthAlert, HealthSeverity
from polylogue.daemon.notification_backends import BackendUnavailableError
from polylogue.logging import get_logger

logger = get_logger(__name__)


SEVERITY_TO_PRIORITY: dict[HealthSeverity, int] = {
    HealthSeverity.OK: 6,
    HealthSeverity.WARNING: 4,
    HealthSeverity.ERROR: 3,
    HealthSeverity.CRITICAL: 2,
}


class _JournalSender(Protocol):
    def __call__(self, message: str, **fields: object) -> None: ...


def _resolve_default_sender() -> _JournalSender:
    try:
        from systemd import journal  # type: ignore[attr-defined, unused-ignore]
    except ImportError as err:  # pragma: no cover - exercised in test_no_systemd
        raise BackendUnavailableError(
            "notification_backend='journald' requires the 'systemd' Python package "
            "(install systemd-python or run inside a systemd-enabled environment)"
        ) from err
    sender: _JournalSender = journal.send
    return sender


class JournaldNotificationBackend:
    """Emit alerts to systemd-journald as structured entries."""

    def __init__(self, *, sender: _JournalSender | None = None) -> None:
        self._send = sender if sender is not None else _resolve_default_sender()

    def notify(self, alerts: list[HealthAlert], *, config: dict[str, object] | None = None) -> None:
        for alert in alerts:
            priority = SEVERITY_TO_PRIORITY.get(alert.severity, 5)
            try:
                self._send(
                    alert.message,
                    PRIORITY=priority,
                    SYSLOG_IDENTIFIER="polylogued",
                    POLYLOGUE_CHECK=alert.check_name,
                    POLYLOGUE_TIER=alert.tier.value,
                    POLYLOGUE_SEVERITY=alert.severity.value,
                    POLYLOGUE_CONSECUTIVE_FAILURES=str(alert.consecutive_failures),
                    POLYLOGUE_CHECKED_AT=alert.checked_at,
                )
            except Exception as err:  # log and continue per-alert: one bad entry should not abort batch
                logger.warning(
                    "daemon.notifications.journald: send failed for %s: %s",
                    alert.check_name,
                    err,
                )


def build_journald_backend(config: dict[str, object] | None) -> JournaldNotificationBackend:
    return JournaldNotificationBackend()


__all__ = [
    "SEVERITY_TO_PRIORITY",
    "JournaldNotificationBackend",
    "build_journald_backend",
]
