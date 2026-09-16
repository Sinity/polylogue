"""SSE framing contract for ``GET /api/events`` (polylogue-gmskd).

A streaming route has already put its status line and headers on the wire by
the time its body can fail. The JSON error path writes a *second* status line
and header block, which a connected ``EventSource`` reads as stream data --
the response stops being HTTP at that point. These tests pin the byte stream,
not the handler's return value, because the defect is only visible in bytes.
"""

from __future__ import annotations

import sqlite3
from io import BytesIO
from typing import TYPE_CHECKING, Any, cast

import pytest

if TYPE_CHECKING:
    from polylogue.daemon.http import DaemonAPIHandler


class _Headers:
    def __init__(self, values: dict[str, str] | None = None) -> None:
        self._values = values or {}

    def get(self, key: str, default: str | None = None) -> str | None:
        return self._values.get(key, default)


def _stream_handler(path: str) -> DaemonAPIHandler:
    """A real ``DaemonAPIHandler`` writing to an in-memory socket file."""
    from polylogue.daemon.http import DaemonAPIHandler

    handler = DaemonAPIHandler.__new__(DaemonAPIHandler)
    handler.path = path
    handler.command = "GET"
    handler.requestline = f"GET {path} HTTP/1.1"
    handler.request_version = "HTTP/1.1"
    handler.client_address = ("127.0.0.1", 12345)
    handler.rfile = BytesIO(b"")
    handler.wfile = BytesIO()
    handler.headers = cast("Any", _Headers())
    handler.log_request = lambda *args, **kwargs: None  # type: ignore[method-assign]
    return handler


def test_events_handler_carries_exactly_one_safe_handler_decorator() -> None:
    """``_handle_events`` is wrapped once.

    At the reported head a stray ``@daemon_safe_handler`` sat where a section
    header had been deleted, so the decorator was applied twice and every
    failure ran the JSON error path twice.

    Anti-vacuity: re-adding the second ``@daemon_safe_handler`` above
    ``_handle_events`` lengthens the ``__wrapped__`` chain and turns this red.
    """
    from polylogue.daemon.http import DaemonAPIHandler

    depth = 0
    fn = DaemonAPIHandler._handle_events
    while hasattr(fn, "__wrapped__"):
        depth += 1
        fn = fn.__wrapped__
    assert depth == 1, f"_handle_events is wrapped {depth} times"


def test_mid_stream_sqlite_error_keeps_the_body_valid_sse(monkeypatch: pytest.MonkeyPatch) -> None:
    """A read failure after the SSE headers terminates the stream, not the framing.

    Input: ``index.db`` is locked or replaced while a client is connected, so
    ``query_events_since`` raises ``sqlite3.OperationalError`` inside the
    stream loop.

    Anti-vacuity: removing the stream-aware ``except`` in ``_stream_events``
    (letting ``daemon_safe_handler`` answer with ``_send_json(503)``) writes a
    second ``HTTP/1.x`` status line into the ``text/event-stream`` body, and
    the single-status-line assertion below turns red.
    """
    from polylogue.daemon import events_http

    calls = {"n": 0}

    def _boom(*args: object, **kwargs: object) -> list[dict[str, object]]:
        calls["n"] += 1
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(events_http, "query_events_since", _boom)

    handler = _stream_handler("/api/events?since=0&max_seconds=5")
    handler._handle_events({"since": ["0"], "max_seconds": ["5"]})

    raw = cast("Any", handler.wfile).getvalue()
    assert calls["n"] == 1
    assert raw.count(b"HTTP/1.") == 1, raw
    head, _, body = raw.partition(b"\r\n\r\n")
    assert b"text/event-stream" in head
    assert b"HTTP/" not in body, body
    assert b"event: error" in body, body
    assert b"stream_read_failed" in body, body
    # Every body frame is SSE framing: a comment, a field line, or a blank.
    for line in body.split(b"\n"):
        stripped = line.rstrip(b"\r")
        if not stripped or stripped.startswith(b":"):
            continue
        assert b":" in stripped, stripped
        field = stripped.split(b":", 1)[0]
        assert field in {b"id", b"event", b"data", b"retry"}, stripped
