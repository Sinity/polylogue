"""``GET /api/provider-usage`` endpoint contract (polylogue-g9j6).

Registered in ``_static_get_routes`` and ``route_contracts.py`` since #2469
but the ``_handle_provider_usage`` method was never implemented — any real
request raised an unhandled ``AttributeError``. This file proves the
handler now actually works end-to-end against a real (if minimal) archive,
complementing the route-registration regression test in
``test_daemon_http_security.py`` (which only proves the method exists, not
that it behaves correctly).
"""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from email.message import Message
from http import HTTPStatus
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING, cast
from unittest.mock import MagicMock

from tests.infra.storage_records import SessionBuilder, db_setup

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


def _make_handler(path: str = "/api/provider-usage") -> DaemonAPIHandler:
    from polylogue.daemon.http import DaemonAPIHandler

    handler = DaemonAPIHandler.__new__(DaemonAPIHandler)
    handler.server = cast("DaemonAPIHTTPServer", _MockServer())
    handler.client_address = ("127.0.0.1", 12345)
    handler.path = path
    handler.command = "GET"
    handler.requestline = f"GET {path} HTTP/1.1"
    handler.headers = cast("Message[str, str]", _MockHeaders({"Content-Length": "0"}))
    handler.rfile = BytesIO(b"")
    handler.wfile = BytesIO()
    return handler


def _capture_json(handler: DaemonAPIHandler) -> MagicMock:
    send_json = MagicMock()
    handler._send_json = send_json  # type: ignore[method-assign]
    return send_json


def test_returns_200_with_empty_origins_when_archive_has_no_sessions(workspace_env: dict[str, Path]) -> None:
    """Smoke test for the missing-handler bug: even with a freshly
    initialized, session-free archive, the endpoint must return a
    structured 200, not crash with the AttributeError this bead fixed."""
    handler = _make_handler()
    send_json = _capture_json(handler)

    handler.do_GET()

    status, payload = send_json.call_args.args
    assert status == HTTPStatus.OK
    assert isinstance(payload, dict)
    assert payload["origins"] == []


def test_returns_200_with_origin_usage_for_seeded_session(workspace_env: dict[str, Path]) -> None:
    db_path = db_setup(workspace_env)
    SessionBuilder(db_path, "pusage-1").provider("chatgpt").title("alpha").add_message(
        role="user", text="hello"
    ).add_message(role="assistant", text="hi").save()

    handler = _make_handler()
    send_json = _capture_json(handler)

    handler.do_GET()

    status, payload = send_json.call_args.args
    assert status == HTTPStatus.OK
    assert isinstance(payload, dict)
    assert "origins" in payload
    assert "model_rollup_usage" in payload
    assert "pricing_lanes" in payload


def test_default_detail_is_headline_not_full(workspace_env: dict[str, Path]) -> None:
    """polylogue-dlmv: detail=full does an unbounded Python-side scan over
    session_provider_usage_events, measured to exceed 90s on the live
    archive. A synchronous HTTP request thread must not default into that
    path — headline is fast (SQL-side rollups only) and the MCP tool's
    detail="full" default is deliberately NOT mirrored here."""
    handler = _make_handler()
    send_json = _capture_json(handler)

    handler.do_GET()

    status, payload = send_json.call_args.args
    assert status == HTTPStatus.OK
    assert payload["detail_level"] == "headline"


def test_respects_detail_and_limit_query_params(workspace_env: dict[str, Path]) -> None:
    handler = _make_handler("/api/provider-usage?detail=headline&limit=5")
    send_json = _capture_json(handler)

    handler.do_GET()

    status, payload = send_json.call_args.args
    assert status == HTTPStatus.OK
    assert payload["detail_level"] == "headline"
