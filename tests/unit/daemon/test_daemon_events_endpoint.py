"""Contract tests for the realtime ``/api/events`` channel (#957).

The daemon exposes daemon-event notifications to the web reader via two
shapes off the same handler:

- ``GET /api/events?poll=1&since=<id>`` — JSON snapshot of events with
  ``id > since``. Used by ETag/poll fallback when ``EventSource`` is
  unavailable.
- ``GET /api/events?since=<id>`` — Server-Sent Events stream of the same
  payload, bounded by ``max_seconds`` so HTTP idle timeouts and tests
  cannot deadlock.

``GET /api/status`` advertises the same monotonic ``last_event_id`` and
sets a weak ``ETag`` so clients can long-poll without paying the full
status payload on every probe.
"""

from __future__ import annotations

from email.message import Message
from http import HTTPStatus
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING, cast
from unittest.mock import MagicMock

import pytest

if TYPE_CHECKING:
    from polylogue.daemon.http import DaemonAPIHandler, DaemonAPIHTTPServer


class _MockServer:
    auth_token = ""
    api_host = "127.0.0.1"


class _MockHeaders:
    def __init__(self, headers: dict[str, str] | None = None) -> None:
        self._headers = headers or {}

    def get(self, key: str, default: str | None = None) -> str | None:
        return self._headers.get(key, default)


def _make_handler(
    method: str,
    path: str,
    *,
    body: bytes = b"",
    extra_headers: dict[str, str] | None = None,
) -> DaemonAPIHandler:
    from polylogue.daemon.http import DaemonAPIHandler

    handler = DaemonAPIHandler.__new__(DaemonAPIHandler)
    handler.server = cast("DaemonAPIHTTPServer", _MockServer())
    handler.client_address = ("127.0.0.1", 12345)
    handler.path = path
    handler.command = method
    handler.requestline = f"{method} {path} HTTP/1.1"
    handler.request_version = "HTTP/1.1"
    handler.protocol_version = "HTTP/1.1"
    headers: dict[str, str] = {"Content-Length": str(len(body))}
    if extra_headers:
        headers.update(extra_headers)
    handler.headers = cast("Message", _MockHeaders(headers))
    handler.rfile = BytesIO(body)
    handler.wfile = BytesIO()
    return handler


def _capture_json(handler: DaemonAPIHandler) -> MagicMock:
    send_json = MagicMock()
    handler._send_json = send_json  # type: ignore[method-assign]
    return send_json


@pytest.fixture
def empty_events_db(workspace_env: dict[str, Path], monkeypatch: pytest.MonkeyPatch) -> Path:
    """Force the daemon-events DB into an isolated workspace."""
    from polylogue.daemon import events as events_mod

    events_path = workspace_env["archive_root"] / "daemon_events.db"

    def _path() -> Path:
        return events_path

    monkeypatch.setattr(events_mod, "_events_db_path", _path)
    return events_path


class TestEventsPollFallback:
    """``GET /api/events?poll=1`` returns JSON envelopes for ETag-style polling."""

    def test_poll_with_no_events_returns_empty_envelope(self, empty_events_db: Path) -> None:
        handler = _make_handler("GET", "/api/events?poll=1&since=0")
        send_json = _capture_json(handler)
        handler.do_GET()

        send_json.assert_called_once()
        status, payload = send_json.call_args.args
        assert status == HTTPStatus.OK
        assert payload == {"events": [], "last_event_id": 0}

    def test_poll_returns_events_after_since(self, empty_events_db: Path) -> None:
        from polylogue.daemon.events import emit_daemon_event

        emit_daemon_event("ingestion_batch", payload={"files": 1})
        emit_daemon_event("ingest", operation_id="op-2", payload={"path": "/tmp/x"})

        handler = _make_handler("GET", "/api/events?poll=1&since=0")
        send_json = _capture_json(handler)
        handler.do_GET()

        status, payload = send_json.call_args.args
        assert status == HTTPStatus.OK
        events = payload["events"]
        assert [e["kind"] for e in events] == ["ingestion_batch", "ingest"]
        assert payload["last_event_id"] == events[-1]["id"]

    def test_poll_kinds_filter_whitelist(self, empty_events_db: Path) -> None:
        from polylogue.daemon.events import emit_daemon_event

        emit_daemon_event("ingestion_batch", payload={"n": 1})
        emit_daemon_event("noise", payload={"n": 2})

        handler = _make_handler("GET", "/api/events?poll=1&since=0&kinds=ingestion_batch,ingest")
        send_json = _capture_json(handler)
        handler.do_GET()

        payload = send_json.call_args.args[1]
        kinds = {e["kind"] for e in payload["events"]}
        assert kinds == {"ingestion_batch"}

    def test_poll_since_is_strict_gt(self, empty_events_db: Path) -> None:
        from polylogue.daemon.events import emit_daemon_event

        emit_daemon_event("ingestion_batch", payload={})
        handler = _make_handler("GET", "/api/events?poll=1&since=0")
        send_json = _capture_json(handler)
        handler.do_GET()
        first_id = send_json.call_args.args[1]["events"][0]["id"]

        handler = _make_handler("GET", f"/api/events?poll=1&since={first_id}")
        send_json = _capture_json(handler)
        handler.do_GET()
        assert send_json.call_args.args[1] == {"events": [], "last_event_id": first_id}


class TestEventsSSEStream:
    """``GET /api/events`` (no ``poll``) writes a Server-Sent Events stream."""

    def test_sse_stream_emits_pending_events_and_closes(self, empty_events_db: Path) -> None:
        from polylogue.daemon.events import emit_daemon_event

        emit_daemon_event("ingestion_batch", payload={"files": 3})
        emit_daemon_event("ingest", operation_id="op-99", payload={})

        handler = _make_handler("GET", "/api/events?since=0&max_seconds=1")
        handler.do_GET()

        out = cast("BytesIO", handler.wfile).getvalue()
        assert b"HTTP/1.0 200" in out or b"HTTP/1.1 200" in out
        assert b"Content-Type: text/event-stream" in out
        assert b"Cache-Control: no-cache" in out
        # Each emitted event becomes one SSE frame.
        assert b"event: ingestion_batch\n" in out
        assert b"event: ingest\n" in out
        assert out.count(b"\nid: ") >= 2

    def test_sse_resumes_from_last_event_id_header(self, empty_events_db: Path) -> None:
        from polylogue.daemon.events import emit_daemon_event

        emit_daemon_event("ingestion_batch", payload={"n": 1})
        emit_daemon_event("ingestion_batch", payload={"n": 2})

        # Last-Event-ID set to the first event's id should suppress it.
        handler_first = _make_handler("GET", "/api/events?poll=1&since=0")
        send_json_first = _capture_json(handler_first)
        handler_first.do_GET()
        first_id = send_json_first.call_args.args[1]["events"][0]["id"]

        handler = _make_handler(
            "GET",
            "/api/events?max_seconds=1",
            extra_headers={"Last-Event-ID": str(first_id)},
        )
        handler.do_GET()
        out = cast("BytesIO", handler.wfile).getvalue()
        # Only the second event should be present.
        assert out.count(b"event: ingestion_batch\n") == 1


class TestStatusEventEtag:
    """``GET /api/status`` advertises ``last_event_id`` + ``ETag``; 304 on match."""

    def test_status_includes_last_event_id_field(self, empty_events_db: Path) -> None:
        from polylogue.daemon.events import emit_daemon_event

        emit_daemon_event("ingestion_batch", payload={})
        handler = _make_handler("GET", "/api/status")
        handler.do_GET()
        out = cast("BytesIO", handler.wfile).getvalue()
        assert b'ETag: W/"events-' in out
        assert b'"last_event_id":' in out

    def test_status_returns_304_when_etag_matches(self, empty_events_db: Path) -> None:
        from polylogue.daemon.events import emit_daemon_event, get_latest_event_id

        emit_daemon_event("ingestion_batch", payload={})
        etag = f'W/"events-{get_latest_event_id()}"'

        handler = _make_handler("GET", "/api/status", extra_headers={"If-None-Match": etag})
        handler.do_GET()
        out = cast("BytesIO", handler.wfile).getvalue()
        assert b" 304 " in out
        assert b"Content-Type: application/json" not in out


class TestAccessTokenQueryFallback:
    """When a token is configured, EventSource clients can use ``?access_token=``."""

    def test_access_token_in_query_string_authenticates(self, empty_events_db: Path) -> None:
        from polylogue.daemon.http import _check_auth_logic

        # The pure logic accepts a Bearer header; the handler-level fallback
        # rewrites ?access_token=X into Bearer X before delegating.
        handler = _make_handler("GET", "/api/events?poll=1&since=0&access_token=secret")
        handler.server = cast(
            "DaemonAPIHTTPServer",
            type(
                "_Srv",
                (),
                {"auth_token": "secret", "api_host": "127.0.0.1"},
            )(),
        )
        send_json = _capture_json(handler)
        handler.do_GET()
        # Should succeed (200), not 401.
        status, _ = send_json.call_args.args
        assert status == HTTPStatus.OK

        # Sanity: the raw logic still rejects when no header is present.
        assert _check_auth_logic("secret", "127.0.0.1", "").allowed is False
