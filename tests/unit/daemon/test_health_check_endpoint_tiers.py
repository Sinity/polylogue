"""``/api/health/check`` tier resolution."""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from email.message import Message
from io import BytesIO
from typing import TYPE_CHECKING, cast
from unittest.mock import MagicMock

import pytest

from polylogue.daemon.health import DaemonHealth, HealthSeverity, HealthTier, resolve_health_tiers

if TYPE_CHECKING:
    from polylogue.daemon.http import DaemonAPIHandler, DaemonAPIHTTPServer


class _MockServer:
    auth_token = ""
    api_host = "127.0.0.1"
    archive_query_executor = ThreadPoolExecutor(max_workers=1)
    archive_query_admission = threading.BoundedSemaphore(64)  # generous: not under test


class _MockHeaders:
    def __init__(self, headers: dict[str, str] | None = None) -> None:
        self._headers = headers or {}

    def get(self, key: str, default: str | None = None) -> str | None:
        return self._headers.get(key, default)


def _make_handler() -> DaemonAPIHandler:
    from polylogue.daemon.http import DaemonAPIHandler

    handler = DaemonAPIHandler.__new__(DaemonAPIHandler)
    handler.server = cast("DaemonAPIHTTPServer", _MockServer())
    handler.client_address = ("127.0.0.1", 12345)
    handler.path = "/api/health/check"
    handler.command = "GET"
    handler.requestline = "GET /api/health/check HTTP/1.1"
    handler.headers = cast("Message[str, str]", _MockHeaders({"Content-Length": "0"}))
    handler.rfile = BytesIO(b"")
    handler.wfile = BytesIO()
    return handler


@pytest.mark.parametrize(
    ("tier_str", "expected"),
    [
        ("fast", {HealthTier.FAST}),
        ("FAST", {HealthTier.FAST}),
        ("fast,medium", {HealthTier.FAST, HealthTier.MEDIUM}),
        (" fast , expensive ", {HealthTier.FAST, HealthTier.EXPENSIVE}),
        ("", {HealthTier.FAST}),
        ("bogus", {HealthTier.FAST}),
        ("medium", {HealthTier.MEDIUM}),
    ],
)
def test_resolve_health_tiers(tier_str: str, expected: set[HealthTier]) -> None:
    assert resolve_health_tiers(tier_str) == expected


def _ok_health() -> DaemonHealth:
    return DaemonHealth(overall_status=HealthSeverity.OK, checked_at="now", alerts=[], tier_summary={})


def test_health_check_endpoint_defaults_to_fast_tier(monkeypatch: pytest.MonkeyPatch) -> None:
    """With the default config the endpoint must run FAST-only checks."""
    import polylogue.daemon.health as health_module

    captured: dict[str, object] = {}

    def _fake_check_health(*, tiers: set[HealthTier] | None = None) -> DaemonHealth:
        captured["tiers"] = tiers
        return _ok_health()

    monkeypatch.setattr(health_module, "check_health", _fake_check_health)

    handler = _make_handler()
    send_json = MagicMock()
    handler._send_json = send_json  # type: ignore[method-assign]
    handler._handle_health_check()

    assert captured["tiers"] == {HealthTier.FAST}
    assert send_json.call_args.args[1]["status"] == "healthy"


def test_health_check_endpoint_honors_config_override(monkeypatch: pytest.MonkeyPatch) -> None:
    """Operators can opt into a heavier polled tier set via config."""
    import polylogue.daemon.health as health_module
    from polylogue import config as config_module

    captured: dict[str, object] = {}

    def _fake_check_health(*, tiers: set[HealthTier] | None = None) -> DaemonHealth:
        captured["tiers"] = tiers
        return _ok_health()

    class _Cfg:
        health_check_tiers = "fast,medium"

    monkeypatch.setattr(health_module, "check_health", _fake_check_health)
    monkeypatch.setattr(config_module, "load_polylogue_config", lambda: _Cfg())

    handler = _make_handler()
    handler._send_json = MagicMock()  # type: ignore[method-assign]
    handler._handle_health_check()

    assert captured["tiers"] == {HealthTier.FAST, HealthTier.MEDIUM}
