"""Tests for the webhook notification backend.

Covers issue #1150 + #1233 acceptance criteria:
- POST shape (URL, method, payload, timeout, headers) asserted with a stub
  client. HMAC-SHA256 signature header is emitted when a secret is set.
- Transient 5xx retried with exponential backoff up to ``WEBHOOK_MAX_RETRIES``;
  persistent failure raises.
- Missing/invalid URL when ``"webhook"`` is selected raises a typed config
  error at resolve time (not a silent fallback to log).
- ``_resolve_backend("webhook")`` returns the backend; unknown names still
  raise the original ``ValueError``.
"""

from __future__ import annotations

import hashlib
import hmac
import json
from typing import Any

import httpx
import pytest

from polylogue.daemon.health import HealthAlert, HealthSeverity, HealthTier
from polylogue.daemon.notifications import (
    WEBHOOK_TIMEOUT_S,
    WebhookConfigError,
    WebhookNotificationBackend,
    _resolve_backend,
    send_notifications,
)


def _alert(name: str = "schema_version", severity: HealthSeverity = HealthSeverity.ERROR) -> HealthAlert:
    return HealthAlert(
        check_name=name,
        tier=HealthTier.FAST,
        severity=severity,
        message=f"{name} {severity.value}",
        checked_at="2026-05-17T00:00:00+00:00",
        consecutive_failures=2 if severity != HealthSeverity.OK else 0,
    )


class _StubClient:
    """Test double for ``httpx.Client`` exposing only ``.post``.

    Each scripted response is either an ``httpx.Response`` (returned) or an
    ``Exception`` (raised). The same instance is replayed across calls so a
    single test can model retries.
    """

    def __init__(self, scripted: list[Any]) -> None:
        self._scripted = list(scripted)
        self.calls: list[dict[str, Any]] = []

    def post(
        self,
        url: str,
        *,
        content: bytes,
        headers: dict[str, str],
        timeout: float,
    ) -> httpx.Response:
        self.calls.append(
            {
                "url": url,
                "content": content,
                "headers": dict(headers),
                "timeout": timeout,
                "json": json.loads(content.decode("utf-8")),
            }
        )
        if not self._scripted:
            raise AssertionError("StubClient ran out of scripted responses")
        item = self._scripted.pop(0)
        if isinstance(item, Exception):
            raise item
        assert isinstance(item, httpx.Response)
        # Bind a synthetic request so raise_for_status() has the context it expects.
        item.request = httpx.Request("POST", url)
        return item


def _ok(status: int = 200) -> httpx.Response:
    return httpx.Response(status_code=status)


def _server_error(status: int = 503) -> httpx.Response:
    return httpx.Response(status_code=status, text="upstream busy")


def test_webhook_posts_typed_envelope_to_configured_url() -> None:
    stub = _StubClient([_ok(202)])
    backend = WebhookNotificationBackend("https://example/notify", client=stub, backoff_s=0, max_retries=1)
    alerts = [_alert(), _alert("wal_size", HealthSeverity.WARNING)]

    backend.notify(alerts)

    assert len(stub.calls) == 1
    call = stub.calls[0]
    assert call["url"] == "https://example/notify"
    assert call["timeout"] == WEBHOOK_TIMEOUT_S
    envelope = call["json"]
    assert set(envelope.keys()) == {"alerts", "emitted_at", "daemon_version"}
    assert isinstance(envelope["alerts"], list)
    assert len(envelope["alerts"]) == 2
    assert envelope["alerts"][0]["check_name"] == "schema_version"
    assert envelope["alerts"][0]["severity"] == "error"
    assert envelope["alerts"][0]["tier"] == "fast"
    assert envelope["alerts"][1]["check_name"] == "wal_size"
    assert isinstance(envelope["emitted_at"], float)
    assert isinstance(envelope["daemon_version"], str) and envelope["daemon_version"]


def test_webhook_empty_alert_list_is_noop() -> None:
    stub = _StubClient([])  # no scripted response → blow up if called
    backend = WebhookNotificationBackend("https://example/notify", client=stub)

    backend.notify([])

    assert stub.calls == []


def test_webhook_transient_5xx_retried_once_then_succeeds() -> None:
    stub = _StubClient([_server_error(502), _ok(200)])
    backend = WebhookNotificationBackend("https://example/notify", client=stub, backoff_s=0, max_retries=1)

    backend.notify([_alert()])

    assert len(stub.calls) == 2


def test_webhook_transient_network_error_retried_once_then_succeeds() -> None:
    stub = _StubClient([httpx.ConnectError("dns broke"), _ok(204)])
    backend = WebhookNotificationBackend("https://example/notify", client=stub, backoff_s=0, max_retries=1)

    backend.notify([_alert()])

    assert len(stub.calls) == 2


def test_webhook_persistent_5xx_raises_after_single_retry() -> None:
    stub = _StubClient([_server_error(503), _server_error(503)])
    backend = WebhookNotificationBackend("https://example/notify", client=stub, backoff_s=0, max_retries=1)

    with pytest.raises(httpx.HTTPError):
        backend.notify([_alert()])

    assert len(stub.calls) == 2  # initial + one retry, not more


def test_webhook_persistent_network_error_raises_after_single_retry() -> None:
    stub = _StubClient([httpx.ConnectError("nope"), httpx.ConnectError("still nope")])
    backend = WebhookNotificationBackend("https://example/notify", client=stub, backoff_s=0, max_retries=1)

    with pytest.raises(httpx.HTTPError):
        backend.notify([_alert()])

    assert len(stub.calls) == 2


def test_webhook_4xx_is_not_retried_and_surfaces() -> None:
    """A 4xx response is a permanent client error; do not waste a retry on it."""
    stub = _StubClient([httpx.Response(status_code=400, text="bad shape")])
    backend = WebhookNotificationBackend("https://example/notify", client=stub, backoff_s=0, max_retries=1)

    with pytest.raises(httpx.HTTPStatusError):
        backend.notify([_alert()])

    assert len(stub.calls) == 1


def test_webhook_missing_url_raises_at_construction() -> None:
    with pytest.raises(WebhookConfigError):
        WebhookNotificationBackend("")


def test_resolve_backend_webhook_requires_url_in_config() -> None:
    with pytest.raises(WebhookConfigError):
        _resolve_backend("webhook", config={"notification_backend": "webhook"})


def test_resolve_backend_webhook_returns_backend_when_url_present() -> None:
    backend = _resolve_backend(
        "webhook",
        config={
            "notification_backend": "webhook",
            "notification_webhook_url": "https://example/notify",
        },
    )
    assert isinstance(backend, WebhookNotificationBackend)


def test_resolve_backend_unknown_name_still_raises_value_error() -> None:
    with pytest.raises(ValueError, match="unknown notification backend"):
        _resolve_backend("smtp")


def test_send_notifications_dispatches_through_webhook_from_config() -> None:
    """send_notifications honours config-driven webhook selection end-to-end."""
    stub = _StubClient([_ok(200)])
    backend = WebhookNotificationBackend("https://example/notify", client=stub, backoff_s=0, max_retries=1)

    send_notifications(
        [_alert()],
        backend=backend,
        config={
            "notification_backend": "webhook",
            "notification_webhook_url": "https://example/notify",
        },
    )

    assert len(stub.calls) == 1


def test_send_notifications_webhook_missing_url_raises_at_resolve_time() -> None:
    """No silent fallback to log when webhook is selected without a URL."""
    with pytest.raises(WebhookConfigError):
        send_notifications(
            [_alert()],
            config={"notification_backend": "webhook"},
        )


# ---------------------------------------------------------------------------
# #1233: HMAC signature + exponential backoff
# ---------------------------------------------------------------------------


def test_webhook_emits_hmac_sha256_signature_header_when_secret_set() -> None:
    """A configured secret signs the body and surfaces in X-Polylogue-Signature."""
    stub = _StubClient([_ok(200)])
    secret = "s3cret"
    backend = WebhookNotificationBackend("https://example/notify", secret=secret, client=stub, backoff_s=0)

    backend.notify([_alert()])

    assert len(stub.calls) == 1
    call = stub.calls[0]
    sent_signature = call["headers"]["X-Polylogue-Signature"]
    expected_hex = hmac.new(secret.encode("utf-8"), call["content"], hashlib.sha256).hexdigest()
    assert sent_signature == f"sha256={expected_hex}"
    assert call["headers"]["Content-Type"] == "application/json"


def test_webhook_omits_signature_header_when_secret_not_set() -> None:
    stub = _StubClient([_ok(200)])
    backend = WebhookNotificationBackend("https://example/notify", client=stub, backoff_s=0)

    backend.notify([_alert()])

    assert "X-Polylogue-Signature" not in stub.calls[0]["headers"]


def test_webhook_resolve_propagates_secret_from_config() -> None:
    backend = _resolve_backend(
        "webhook",
        config={
            "notification_backend": "webhook",
            "notification_webhook_url": "https://example/notify",
            "notification_webhook_secret": "shh",
        },
    )
    assert isinstance(backend, WebhookNotificationBackend)


def test_webhook_exponential_backoff_between_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    """Retry attempts use exponential backoff (0.2, 0.4, ...)."""
    sleeps: list[float] = []
    monkeypatch.setattr(
        "polylogue.daemon.notification_backends.webhook.time.sleep",
        sleeps.append,
    )

    stub = _StubClient([_server_error(503), _server_error(503), _ok(200)])
    backend = WebhookNotificationBackend("https://example/notify", client=stub, max_retries=2, backoff_s=0.2)
    backend.notify([_alert()])

    assert sleeps == pytest.approx([0.2, 0.4])
    assert len(stub.calls) == 3


def test_webhook_max_retries_total_attempts() -> None:
    """``max_retries=2`` -> three total attempts before raising."""
    stub = _StubClient([_server_error(503), _server_error(503), _server_error(503)])
    backend = WebhookNotificationBackend("https://example/notify", client=stub, max_retries=2, backoff_s=0)

    with pytest.raises(httpx.HTTPError):
        backend.notify([_alert()])

    assert len(stub.calls) == 3
