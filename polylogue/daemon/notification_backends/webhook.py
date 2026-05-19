"""Webhook notification backend.

POSTs a typed JSON envelope to a configured URL. Adds an
``X-Polylogue-Signature`` header carrying ``sha256=<hex>`` when a shared
secret is configured so the receiver can verify the payload was produced by
this daemon.

Transient failures (network errors, HTTP 5xx) are retried with exponential
backoff up to ``max_retries`` additional attempts. 4xx responses are
treated as permanent and surface immediately. Persistent failure raises
``httpx.HTTPError``; the caller (the periodic health loop) is expected to
isolate the exception.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import time
from typing import Protocol

import httpx

from polylogue.daemon.health import HealthAlert
from polylogue.daemon.notification_backends import (
    BackendConfigError,
    build_envelope,
)
from polylogue.logging import get_logger

logger = get_logger(__name__)


WEBHOOK_TIMEOUT_S = 5.0
"""Per-attempt HTTP timeout for the webhook backend."""

WEBHOOK_RETRY_BACKOFF_S = 0.5
"""Initial backoff between retries (doubled on each successive retry)."""

WEBHOOK_MAX_RETRIES = 2
"""Number of additional attempts after the first; total attempts = 1 + N."""


class WebhookConfigError(BackendConfigError):
    """Raised when the webhook backend is selected without a usable URL."""


class _HTTPPoster(Protocol):
    """Minimal client surface (``.post(...)``) so tests can inject stubs."""

    def post(
        self,
        url: str,
        *,
        content: bytes,
        headers: dict[str, str],
        timeout: float,
    ) -> httpx.Response: ...


class WebhookNotificationBackend:
    """POST a typed JSON envelope to a user-configured URL."""

    def __init__(
        self,
        url: str,
        *,
        secret: str | None = None,
        timeout: float = WEBHOOK_TIMEOUT_S,
        max_retries: int = WEBHOOK_MAX_RETRIES,
        backoff_s: float = WEBHOOK_RETRY_BACKOFF_S,
        client: _HTTPPoster | None = None,
    ) -> None:
        if not url or not isinstance(url, str):
            raise WebhookConfigError(
                "notification_webhook_url must be a non-empty string when notification_backend='webhook'"
            )
        self._url = url
        self._secret = secret.encode("utf-8") if secret else None
        self._timeout = float(timeout)
        self._max_retries = int(max_retries)
        self._backoff_s = float(backoff_s)
        self._client = client

    def notify(self, alerts: list[HealthAlert], *, config: dict[str, object] | None = None) -> None:
        if not alerts:
            return
        envelope = build_envelope(alerts)
        body = json.dumps(envelope, sort_keys=True, default=str).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self._secret is not None:
            mac = hmac.new(self._secret, body, hashlib.sha256).hexdigest()
            headers["X-Polylogue-Signature"] = f"sha256={mac}"

        attempts = self._max_retries + 1
        last_error: Exception | None = None
        backoff = self._backoff_s
        for attempt in range(attempts):
            try:
                response = self._post(body, headers)
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
                    f"server error {response.status_code}",
                    request=response.request,
                    response=response,
                )
                logger.warning(
                    "daemon.notifications.webhook: server %d on attempt %d/%d",
                    response.status_code,
                    attempt + 1,
                    attempts,
                )
            if attempt < attempts - 1 and backoff > 0:
                time.sleep(backoff)
                backoff *= 2
        assert last_error is not None
        raise last_error

    def _post(self, body: bytes, headers: dict[str, str]) -> httpx.Response:
        if self._client is not None:
            return self._client.post(self._url, content=body, headers=headers, timeout=self._timeout)
        with httpx.Client(timeout=self._timeout) as client:
            return client.post(self._url, content=body, headers=headers)


def build_webhook_backend(config: dict[str, object] | None) -> WebhookNotificationBackend:
    """Construct from a config dict using ``notification_webhook_url`` / ``notification_webhook_secret``."""
    cfg = config or {}
    url_value: object = cfg.get("notification_webhook_url")
    if not isinstance(url_value, str) or not url_value:
        raise WebhookConfigError("notification_backend='webhook' requires notification_webhook_url in config")
    secret_value: object = cfg.get("notification_webhook_secret")
    secret: str | None = secret_value if isinstance(secret_value, str) and secret_value else None
    return WebhookNotificationBackend(url_value, secret=secret)


__all__ = [
    "WEBHOOK_MAX_RETRIES",
    "WEBHOOK_RETRY_BACKOFF_S",
    "WEBHOOK_TIMEOUT_S",
    "WebhookConfigError",
    "WebhookNotificationBackend",
    "build_webhook_backend",
]
