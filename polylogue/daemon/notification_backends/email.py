"""Email (SMTP) notification backend.

Sends one ``multipart/alternative`` email per ``notify()`` call carrying
the rendered alert envelope as both plain text and JSON. The transport is
``smtplib.SMTP`` with STARTTLS by default (set ``use_tls=False`` for
``smtplib.SMTP_SSL`` implicit TLS on port 465; ``use_starttls=False`` to
disable upgrade).

A simple token-bucket rate limiter caps deliveries at
``max_per_hour`` per process (default 12). Drops are logged at WARNING.
The bucket is in-memory and per-instance so each daemon restart resets;
that matches the "operator gets notified at most N times per hour"
intent without persisting state.

The SMTP client class is injected for tests; production uses ``smtplib``.
"""

from __future__ import annotations

import contextlib
import json
import smtplib
import time
from collections import deque
from collections.abc import Callable
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.utils import formatdate, make_msgid
from typing import Protocol

from polylogue.daemon.health import HealthAlert
from polylogue.daemon.notification_backends import BackendConfigError, build_envelope
from polylogue.logging import get_logger

logger = get_logger(__name__)


EMAIL_DEFAULT_PORT = 587
EMAIL_DEFAULT_TIMEOUT_S = 10.0
EMAIL_DEFAULT_MAX_PER_HOUR = 12


class EmailConfigError(BackendConfigError):
    """Raised when the email backend is missing required SMTP keys."""


class _SMTPClient(Protocol):
    def starttls(self) -> object: ...
    def login(self, user: str, password: str) -> object: ...
    def send_message(self, msg: MIMEMultipart) -> object: ...
    def quit(self) -> object: ...


class _SMTPFactory(Protocol):
    def __call__(self, host: str, port: int, *, timeout: float) -> _SMTPClient: ...


def _default_factory(use_ssl: bool) -> _SMTPFactory:
    def _factory(host: str, port: int, *, timeout: float) -> _SMTPClient:
        client: smtplib.SMTP = (
            smtplib.SMTP_SSL(host, port, timeout=timeout) if use_ssl else smtplib.SMTP(host, port, timeout=timeout)
        )
        return client

    return _factory


class EmailNotificationBackend:
    """Send a single email per ``notify()`` call carrying the alert envelope."""

    def __init__(
        self,
        *,
        host: str,
        port: int = EMAIL_DEFAULT_PORT,
        username: str | None = None,
        password: str | None = None,
        sender: str,
        recipients: tuple[str, ...],
        subject_prefix: str = "[polylogue]",
        use_tls: bool = True,
        use_starttls: bool = True,
        timeout: float = EMAIL_DEFAULT_TIMEOUT_S,
        max_per_hour: int = EMAIL_DEFAULT_MAX_PER_HOUR,
        smtp_factory: _SMTPFactory | None = None,
        now: Callable[[], float] | None = None,
    ) -> None:
        if not host:
            raise EmailConfigError("notification email backend requires notification_email_host")
        if not sender:
            raise EmailConfigError("notification email backend requires notification_email_from")
        if not recipients:
            raise EmailConfigError("notification email backend requires notification_email_to")
        self._host = host
        self._port = int(port)
        self._username = username
        self._password = password
        self._sender = sender
        self._recipients = tuple(recipients)
        self._subject_prefix = subject_prefix
        self._use_tls = bool(use_tls)
        self._use_starttls = bool(use_starttls)
        self._timeout = float(timeout)
        self._max_per_hour = max(0, int(max_per_hour))
        use_ssl = self._use_tls and self._port == 465
        # On implicit-TLS port 465 the SSL handshake is already complete; STARTTLS would be a noop.
        if use_ssl:
            self._use_starttls = False
        self._smtp_factory: _SMTPFactory = smtp_factory if smtp_factory is not None else _default_factory(use_ssl)
        self._now: Callable[[], float] = now if now is not None else time.time
        self._recent_sends: deque[float] = deque()

    # ------------------------------------------------------------------
    # rate limiter
    # ------------------------------------------------------------------

    def _allow_send(self) -> bool:
        if self._max_per_hour <= 0:
            return True
        now = float(self._now())
        cutoff = now - 3600.0
        while self._recent_sends and self._recent_sends[0] < cutoff:
            self._recent_sends.popleft()
        if len(self._recent_sends) >= self._max_per_hour:
            return False
        self._recent_sends.append(now)
        return True

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    def notify(self, alerts: list[HealthAlert], *, config: dict[str, object] | None = None) -> None:
        if not alerts:
            return
        if not self._allow_send():
            logger.warning(
                "daemon.notifications.email: rate limit reached (%d/hour); dropped %d alert(s)",
                self._max_per_hour,
                len(alerts),
            )
            return
        msg = self._build_message(alerts)
        client = self._smtp_factory(self._host, self._port, timeout=self._timeout)
        try:
            if self._use_starttls:
                client.starttls()
            if self._username and self._password is not None:
                client.login(self._username, self._password)
            client.send_message(msg)
        finally:
            with contextlib.suppress(Exception):
                client.quit()

    # ------------------------------------------------------------------
    # rendering
    # ------------------------------------------------------------------

    def _build_message(self, alerts: list[HealthAlert]) -> MIMEMultipart:
        envelope = build_envelope(alerts)
        text_body = self._render_text(alerts)
        json_body = json.dumps(envelope, indent=2, sort_keys=True, default=str)
        msg = MIMEMultipart("alternative")
        worst = max(alerts, key=lambda a: _SEVERITY_RANK.get(a.severity.value, 0))
        msg["Subject"] = f"{self._subject_prefix} {worst.severity.value.upper()} {worst.check_name}"
        msg["From"] = self._sender
        msg["To"] = ", ".join(self._recipients)
        msg["Date"] = formatdate(localtime=True)
        msg["Message-ID"] = make_msgid(domain="polylogue.local")
        msg.attach(MIMEText(text_body, "plain", "utf-8"))
        msg.attach(MIMEText(json_body, "json", "utf-8"))
        return msg

    @staticmethod
    def _render_text(alerts: list[HealthAlert]) -> str:
        lines = [f"Polylogue daemon emitted {len(alerts)} alert(s):", ""]
        for alert in alerts:
            lines.append(
                f"- [{alert.severity.value.upper()}] {alert.check_name} (tier={alert.tier.value}, "
                f"consecutive_failures={alert.consecutive_failures})"
            )
            lines.append(f"    {alert.message}")
            lines.append(f"    checked_at={alert.checked_at}")
            lines.append("")
        return "\n".join(lines)


_SEVERITY_RANK = {"ok": 0, "warning": 1, "error": 2, "critical": 3}


def build_email_backend(config: dict[str, object] | None) -> EmailNotificationBackend:
    cfg = config or {}

    def _string(key: str) -> str | None:
        value = cfg.get(key)
        return value if isinstance(value, str) and value else None

    def _int(key: str, default: int) -> int:
        value = cfg.get(key, default)
        if isinstance(value, int) and not isinstance(value, bool):
            return value
        if isinstance(value, str):
            try:
                return int(value)
            except ValueError:
                return default
        return default

    def _bool(key: str, default: bool) -> bool:
        value = cfg.get(key, default)
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.lower() in ("1", "true", "yes")
        return default

    host = _string("notification_email_host")
    sender = _string("notification_email_from")
    recipients_raw = cfg.get("notification_email_to")
    if isinstance(recipients_raw, str):
        recipients = tuple(s.strip() for s in recipients_raw.split(",") if s.strip())
    elif isinstance(recipients_raw, (list, tuple)):
        recipients = tuple(str(s).strip() for s in recipients_raw if str(s).strip())
    else:
        recipients = ()

    if not host:
        raise EmailConfigError("notification_backend='email' requires notification_email_host in config")
    if not sender:
        raise EmailConfigError("notification_backend='email' requires notification_email_from in config")
    if not recipients:
        raise EmailConfigError("notification_backend='email' requires notification_email_to in config")

    return EmailNotificationBackend(
        host=host,
        port=_int("notification_email_port", EMAIL_DEFAULT_PORT),
        username=_string("notification_email_username"),
        password=_string("notification_email_password"),
        sender=sender,
        recipients=recipients,
        subject_prefix=_string("notification_email_subject_prefix") or "[polylogue]",
        use_tls=_bool("notification_email_use_tls", True),
        use_starttls=_bool("notification_email_use_starttls", True),
        max_per_hour=_int("notification_email_max_per_hour", EMAIL_DEFAULT_MAX_PER_HOUR),
    )


__all__ = [
    "EMAIL_DEFAULT_MAX_PER_HOUR",
    "EMAIL_DEFAULT_PORT",
    "EMAIL_DEFAULT_TIMEOUT_S",
    "EmailConfigError",
    "EmailNotificationBackend",
    "build_email_backend",
]
