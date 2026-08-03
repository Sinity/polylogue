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


def test_health_check_endpoint_defaults_to_fast_and_medium_tiers(monkeypatch: pytest.MonkeyPatch) -> None:
    """With the default config the endpoint must run FAST+MEDIUM (polylogue-y0ven)."""
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

    assert captured["tiers"] == {HealthTier.FAST, HealthTier.MEDIUM}
    assert send_json.call_args.args[1]["status"] == "healthy"


def test_health_check_endpoint_honors_config_override(monkeypatch: pytest.MonkeyPatch) -> None:
    """Operators can opt into the EXPENSIVE tier via config."""
    import polylogue.daemon.health as health_module
    from polylogue import config as config_module

    captured: dict[str, object] = {}

    def _fake_check_health(*, tiers: set[HealthTier] | None = None) -> DaemonHealth:
        captured["tiers"] = tiers
        return _ok_health()

    class _Cfg:
        health_check_tiers = "fast,medium,expensive"

    monkeypatch.setattr(health_module, "check_health", _fake_check_health)
    monkeypatch.setattr(config_module, "load_polylogue_config", lambda: _Cfg())

    handler = _make_handler()
    handler._send_json = MagicMock()  # type: ignore[method-assign]
    handler._handle_health_check()

    assert captured["tiers"] == {HealthTier.FAST, HealthTier.MEDIUM, HealthTier.EXPENSIVE}


def test_health_tier_coverage_names_off_tiers_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """Default config runs fast+medium; the FAST notice must name EXPENSIVE as off.

    This is the polylogue-y0ven silence-labeling requirement: a stalled
    MEDIUM/EXPENSIVE tier used to be indistinguishable from "never
    configured". The FAST-tier coverage check must always say which tiers
    are off instead of the daemon's health payload staying quiet about it.
    """
    from polylogue import config as config_module
    from polylogue.daemon.health import HealthSeverity, _check_health_tier_coverage_fast

    class _Cfg:
        health_check_tiers = "fast,medium"

    monkeypatch.setattr(config_module, "load_polylogue_config", lambda: _Cfg())

    alert = _check_health_tier_coverage_fast()

    assert alert.check_name == "health_tier_coverage"
    assert alert.tier == HealthTier.FAST
    assert alert.severity == HealthSeverity.OK
    assert "expensive" in alert.message
    assert "medium" not in alert.message.split(":", 1)[-1]


def test_health_tier_coverage_reports_nothing_off_when_all_tiers_configured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from polylogue import config as config_module
    from polylogue.daemon.health import HealthSeverity, _check_health_tier_coverage_fast

    class _Cfg:
        health_check_tiers = "fast,medium,expensive"

    monkeypatch.setattr(config_module, "load_polylogue_config", lambda: _Cfg())

    alert = _check_health_tier_coverage_fast()

    assert alert.severity == HealthSeverity.OK
    assert "off" not in alert.message
    assert "fast, medium, expensive" in alert.message


def test_health_tier_coverage_names_multiple_off_tiers(monkeypatch: pytest.MonkeyPatch) -> None:
    from polylogue import config as config_module
    from polylogue.daemon.health import _check_health_tier_coverage_fast

    class _Cfg:
        health_check_tiers = "fast"

    monkeypatch.setattr(config_module, "load_polylogue_config", lambda: _Cfg())

    alert = _check_health_tier_coverage_fast()

    assert "medium" in alert.message
    assert "expensive" in alert.message
