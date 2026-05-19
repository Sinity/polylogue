"""Adapter tests for the non-log notification backends (#1233).

Covers the journald, email, and Apprise adapters plus the fan-out
dispatcher in :mod:`polylogue.daemon.notifications`. The webhook adapter
keeps its own dedicated file (``test_notification_webhook.py``) which is
where the HMAC + retry/backoff coverage lives.
"""

from __future__ import annotations

import smtplib
from email.message import Message
from typing import Any

import pytest

from polylogue.daemon.health import HealthAlert, HealthSeverity, HealthTier
from polylogue.daemon.notification_backends import (
    BackendConfigError,
    BackendUnavailableError,
)
from polylogue.daemon.notification_backends.apprise_backend import (
    AppriseConfigError,
    AppriseNotificationBackend,
)
from polylogue.daemon.notification_backends.email import (
    EmailConfigError,
    EmailNotificationBackend,
)
from polylogue.daemon.notification_backends.journald import (
    SEVERITY_TO_PRIORITY,
    JournaldNotificationBackend,
)
from polylogue.daemon.notifications import (
    FanOutNotificationBackend,
    LogNotificationBackend,
    _resolve_backend,
    send_notifications,
    supported_backends,
)


def _alert(
    name: str = "schema_version",
    severity: HealthSeverity = HealthSeverity.ERROR,
) -> HealthAlert:
    return HealthAlert(
        check_name=name,
        tier=HealthTier.FAST,
        severity=severity,
        message=f"{name} {severity.value}",
        checked_at="2026-05-17T00:00:00+00:00",
        consecutive_failures=2 if severity != HealthSeverity.OK else 0,
    )


# ---------------------------------------------------------------------------
# journald
# ---------------------------------------------------------------------------


class _RecordingJournalSender:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, object]]] = []

    def __call__(self, message: str, **fields: object) -> None:
        self.calls.append((message, dict(fields)))


def test_journald_emits_one_entry_per_alert_with_priority_mapping() -> None:
    sender = _RecordingJournalSender()
    backend = JournaldNotificationBackend(sender=sender)

    backend.notify(
        [
            _alert("a", HealthSeverity.OK),
            _alert("b", HealthSeverity.WARNING),
            _alert("c", HealthSeverity.ERROR),
            _alert("d", HealthSeverity.CRITICAL),
        ]
    )

    assert [c[0] for c in sender.calls] == [
        "a ok",
        "b warning",
        "c error",
        "d critical",
    ]
    priorities = [c[1]["PRIORITY"] for c in sender.calls]
    assert priorities == [
        SEVERITY_TO_PRIORITY[HealthSeverity.OK],
        SEVERITY_TO_PRIORITY[HealthSeverity.WARNING],
        SEVERITY_TO_PRIORITY[HealthSeverity.ERROR],
        SEVERITY_TO_PRIORITY[HealthSeverity.CRITICAL],
    ]
    fields = sender.calls[2][1]
    assert fields["SYSLOG_IDENTIFIER"] == "polylogued"
    assert fields["POLYLOGUE_CHECK"] == "c"
    assert fields["POLYLOGUE_TIER"] == "fast"
    assert fields["POLYLOGUE_SEVERITY"] == "error"


def test_journald_isolates_per_alert_send_failures() -> None:
    """A raise from one ``send`` does not abort the loop."""

    def _spotty(message: str, **fields: object) -> None:
        if "boom" in message:
            raise RuntimeError("simulated journal failure")

    backend = JournaldNotificationBackend(sender=_spotty)
    # Should not propagate; degraded delivery is logged in the backend.
    backend.notify([_alert("boom", HealthSeverity.ERROR), _alert("ok", HealthSeverity.WARNING)])


def test_journald_default_construction_skips_when_systemd_unavailable() -> None:
    """Without the systemd package, construction raises the typed error.

    On a host that actually has systemd installed this test is a no-op.
    """
    try:
        import systemd.journal  # noqa: F401
    except ImportError:
        with pytest.raises(BackendUnavailableError):
            JournaldNotificationBackend()
    else:
        pytest.skip("systemd python binding installed; cannot exercise unavailable path")


# ---------------------------------------------------------------------------
# email
# ---------------------------------------------------------------------------


class _RecordingSMTP:
    """Test double for smtplib.SMTP exposing only the methods we call."""

    def __init__(self) -> None:
        self.starttls_called = 0
        self.login_calls: list[tuple[str, str]] = []
        self.messages: list[Message] = []
        self.quit_called = 0

    def starttls(self) -> object:
        self.starttls_called += 1
        return None

    def login(self, user: str, password: str) -> object:
        self.login_calls.append((user, password))
        return None

    def send_message(self, msg: Message) -> object:
        self.messages.append(msg)
        return None

    def quit(self) -> object:
        self.quit_called += 1
        return None


def _factory_returning(client: _RecordingSMTP) -> Any:
    def _factory(host: str, port: int, *, timeout: float) -> _RecordingSMTP:
        _factory.last = (host, port, timeout)  # type: ignore[attr-defined]
        return client

    return _factory


def test_email_sends_multipart_message_with_text_and_json_parts() -> None:
    client = _RecordingSMTP()
    backend = EmailNotificationBackend(
        host="smtp.example",
        port=587,
        username="user",
        password="pw",
        sender="alerts@example",
        recipients=("ops@example", "oncall@example"),
        smtp_factory=_factory_returning(client),
    )

    backend.notify([_alert("schema_version", HealthSeverity.CRITICAL)])

    assert len(client.messages) == 1
    msg = client.messages[0]
    assert msg["From"] == "alerts@example"
    assert msg["To"] == "ops@example, oncall@example"
    assert msg["Subject"].startswith("[polylogue] CRITICAL schema_version")
    parts = list(msg.walk())
    subtypes = [p.get_content_subtype() for p in parts if not p.is_multipart()]
    assert "plain" in subtypes
    assert "json" in subtypes
    assert client.starttls_called == 1
    assert client.login_calls == [("user", "pw")]
    assert client.quit_called == 1


def test_email_rate_limiter_drops_after_max_per_hour() -> None:
    client = _RecordingSMTP()
    fake_now = [1_000_000.0]

    def _now() -> float:
        return fake_now[0]

    backend = EmailNotificationBackend(
        host="smtp.example",
        sender="alerts@example",
        recipients=("ops@example",),
        smtp_factory=_factory_returning(client),
        max_per_hour=2,
        now=_now,
    )

    backend.notify([_alert("a")])
    backend.notify([_alert("b")])
    backend.notify([_alert("c")])  # dropped
    assert len(client.messages) == 2

    # Advance past the 1h window; bucket resets.
    fake_now[0] += 3601.0
    backend.notify([_alert("d")])
    assert len(client.messages) == 3


def test_email_missing_required_keys_raise_typed_config_error() -> None:
    with pytest.raises(EmailConfigError):
        EmailNotificationBackend(
            host="",
            sender="x@example",
            recipients=("y@example",),
            smtp_factory=_factory_returning(_RecordingSMTP()),
        )
    with pytest.raises(EmailConfigError):
        EmailNotificationBackend(
            host="smtp.example",
            sender="",
            recipients=("y@example",),
            smtp_factory=_factory_returning(_RecordingSMTP()),
        )
    with pytest.raises(EmailConfigError):
        EmailNotificationBackend(
            host="smtp.example",
            sender="x@example",
            recipients=(),
            smtp_factory=_factory_returning(_RecordingSMTP()),
        )


def test_email_uses_implicit_tls_on_port_465() -> None:
    """``port=465`` selects SMTP_SSL via the factory and skips STARTTLS."""
    client = _RecordingSMTP()
    factory = _factory_returning(client)
    backend = EmailNotificationBackend(
        host="smtp.example",
        port=465,
        sender="alerts@example",
        recipients=("ops@example",),
        smtp_factory=factory,
    )
    backend.notify([_alert()])
    assert factory.last[1] == 465
    assert client.starttls_called == 0


def test_email_no_auth_when_credentials_absent() -> None:
    client = _RecordingSMTP()
    backend = EmailNotificationBackend(
        host="smtp.example",
        sender="alerts@example",
        recipients=("ops@example",),
        smtp_factory=_factory_returning(client),
    )
    backend.notify([_alert()])
    assert client.login_calls == []


def test_email_smtplib_classes_are_present_for_documentation() -> None:
    # Pinning here so a future refactor that drops the import surface
    # surfaces in CI rather than via runtime AttributeError.
    assert hasattr(smtplib, "SMTP")
    assert hasattr(smtplib, "SMTP_SSL")


# ---------------------------------------------------------------------------
# apprise
# ---------------------------------------------------------------------------


class _RecordingApprise:
    def __init__(self) -> None:
        self.added: list[str] = []
        self.notifies: list[dict[str, object]] = []
        self.notify_result = True

    def add(self, url: str) -> bool:
        self.added.append(url)
        return True

    def notify(
        self,
        body: str,
        *,
        title: str,
        notify_type: str,
        body_format: str,
    ) -> bool:
        self.notifies.append({"body": body, "title": title, "notify_type": notify_type, "body_format": body_format})
        return self.notify_result


def test_apprise_adds_all_urls_and_notifies_per_alert() -> None:
    client = _RecordingApprise()
    backend = AppriseNotificationBackend(
        ("pover://user@token", "discord://webhookid/token", "ntfy://hostname/topic"),
        client=client,
    )

    backend.notify(
        [
            _alert("a", HealthSeverity.WARNING),
            _alert("b", HealthSeverity.CRITICAL),
        ]
    )

    assert client.added == [
        "pover://user@token",
        "discord://webhookid/token",
        "ntfy://hostname/topic",
    ]
    assert len(client.notifies) == 2
    assert client.notifies[0]["title"] == "[polylogue] WARNING a"
    assert client.notifies[0]["notify_type"] == "warning"
    assert client.notifies[1]["title"] == "[polylogue] CRITICAL b"
    assert client.notifies[1]["notify_type"] == "failure"
    # Envelope inclusion: body carries the message and a JSON payload.
    body = client.notifies[0]["body"]
    assert isinstance(body, str)
    assert "a warning" in body
    assert '"alerts"' in body


def test_apprise_construction_rejects_empty_url_list() -> None:
    with pytest.raises(AppriseConfigError):
        AppriseNotificationBackend((), client=_RecordingApprise())


def test_apprise_notify_returning_false_does_not_raise() -> None:
    client = _RecordingApprise()
    client.notify_result = False
    backend = AppriseNotificationBackend(("ntfy://x/y",), client=client)

    backend.notify([_alert("c", HealthSeverity.ERROR)])

    assert len(client.notifies) == 1


# ---------------------------------------------------------------------------
# fan-out + registry
# ---------------------------------------------------------------------------


class _RecordingBackend:
    def __init__(self, *, raises: Exception | None = None) -> None:
        self.calls: list[list[HealthAlert]] = []
        self._raises = raises

    def notify(self, alerts: list[HealthAlert], *, config: dict[str, object] | None = None) -> None:
        self.calls.append(list(alerts))
        if self._raises is not None:
            raise self._raises


def test_fanout_invokes_every_backend_in_order() -> None:
    a = _RecordingBackend()
    b = _RecordingBackend()
    fan = FanOutNotificationBackend([a, b])
    fan.notify([_alert()])
    assert len(a.calls) == 1
    assert len(b.calls) == 1


def test_fanout_isolates_per_backend_failures_and_reraises_first() -> None:
    a = _RecordingBackend(raises=RuntimeError("first"))
    b = _RecordingBackend()
    c = _RecordingBackend(raises=RuntimeError("third"))
    fan = FanOutNotificationBackend([a, b, c])

    with pytest.raises(RuntimeError, match="first"):
        fan.notify([_alert()])

    # 'b' still received the batch despite 'a' failing.
    assert len(b.calls) == 1
    assert len(c.calls) == 1


def test_fanout_requires_at_least_one_backend() -> None:
    with pytest.raises(ValueError):
        FanOutNotificationBackend([])


def test_supported_backends_includes_all_new_adapters() -> None:
    assert set(supported_backends()) >= {"log", "webhook", "journald", "email", "apprise"}


def test_resolve_backend_accepts_comma_separated_string_for_fanout() -> None:
    backend = _resolve_backend(
        "log,webhook",
        config={
            "notification_backend": "log,webhook",
            "notification_webhook_url": "https://example/notify",
        },
    )
    assert isinstance(backend, FanOutNotificationBackend)
    assert isinstance(backend.backends[0], LogNotificationBackend)


def test_resolve_backend_accepts_single_log_name() -> None:
    backend = _resolve_backend("log")
    assert isinstance(backend, LogNotificationBackend)


def test_resolve_backend_rejects_unknown_name_inside_fanout() -> None:
    with pytest.raises(ValueError, match="unknown notification backend"):
        _resolve_backend("log,bogus")


def test_send_notifications_accepts_list_typed_backend_spec() -> None:
    """A list value for ``notification_backend`` is accepted (TOML array)."""
    seen: list[list[HealthAlert]] = []

    class _Sink:
        def notify(
            self,
            alerts: list[HealthAlert],
            *,
            config: dict[str, object] | None = None,
        ) -> None:
            seen.append(list(alerts))

    send_notifications([_alert()], backend=_Sink(), config={"notification_backend": ["log"]})
    assert len(seen) == 1


def test_email_backend_unknown_via_resolve_requires_config_keys() -> None:
    with pytest.raises(BackendConfigError):
        _resolve_backend("email", config={"notification_backend": "email"})


def test_apprise_backend_unknown_via_resolve_requires_urls() -> None:
    with pytest.raises(BackendConfigError):
        _resolve_backend("apprise", config={"notification_backend": "apprise"})
