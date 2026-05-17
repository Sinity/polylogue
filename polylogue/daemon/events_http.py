"""HTTP handlers for the realtime daemon-event channel (#957).

Two shapes share the ``/api/events`` route:

- ``GET /api/events?poll=1&since=<id>`` — JSON snapshot of new events.
  Used by the ETag-style polling fallback when ``EventSource`` is not
  available in the client.
- ``GET /api/events?since=<id>`` — Server-Sent Events stream. Long-polls
  the daemon-event ledger for new rows and emits one SSE frame per
  event, with a heartbeat comment between idle ticks. Bounded by
  ``max_seconds`` so HTTP idle timeouts and tests cannot deadlock.

The handlers are pulled out of :mod:`polylogue.daemon.http` to keep that
module's file-size budget headroom for the broader archive read API.
"""

from __future__ import annotations

import contextlib
import time
from http import HTTPStatus
from typing import TYPE_CHECKING, cast

from polylogue.daemon.events import query_events_since

if TYPE_CHECKING:
    from polylogue.daemon.http import DaemonAPIHandler


def handle_events(handler: DaemonAPIHandler, params: dict[str, list[str]]) -> None:
    """Dispatch ``GET /api/events`` to either the poll or SSE shape."""
    since_param = handler._get_int(params, "since", 0)
    if since_param == 0:
        header_id = handler.headers.get("Last-Event-ID", "")
        with contextlib.suppress(ValueError, TypeError):
            if header_id:
                since_param = int(header_id)
    kinds_param = handler._get_param(params, "kinds")
    kinds: tuple[str, ...] = tuple(k.strip() for k in (kinds_param or "").split(",") if k.strip())

    if handler._get_bool(params, "poll"):
        events = list(query_events_since(since_param, kinds=kinds, limit=500))
        latest = events[-1]["id"] if events else since_param
        handler._send_json(
            HTTPStatus.OK,
            {"events": events, "last_event_id": latest},
        )
        return

    max_seconds = handler._get_int(params, "max_seconds", 30)
    if max_seconds <= 0 or max_seconds > 300:
        max_seconds = 30
    _stream_events(handler, since_param, kinds, max_seconds)


def _stream_events(
    handler: DaemonAPIHandler,
    since: int,
    kinds: tuple[str, ...],
    max_seconds: int,
) -> None:
    """Long-poll the event ledger and stream SSE frames to the client."""
    handler.send_response(HTTPStatus.OK.value)
    handler.send_header("Content-Type", "text/event-stream")
    handler.send_header("Cache-Control", "no-cache, no-store")
    handler.send_header("X-Accel-Buffering", "no")
    handler.send_header("Connection", "close")
    handler.end_headers()

    cursor = since
    deadline = time.monotonic() + max_seconds
    try:
        _write_sse_comment(handler, b"open")
        while time.monotonic() < deadline:
            events = list(query_events_since(cursor, kinds=kinds, limit=200))
            if events:
                for event in events:
                    _write_sse_event(handler, event)
                    cursor = int(cast("int", event["id"]))
            else:
                _write_sse_comment(handler, b"tick")
            time.sleep(1.0)
    except (BrokenPipeError, ConnectionResetError):
        return


def _write_sse_comment(handler: DaemonAPIHandler, comment: bytes) -> None:
    handler.wfile.write(b": " + comment + b"\n\n")
    handler.wfile.flush()


def _write_sse_event(handler: DaemonAPIHandler, event: dict[str, object]) -> None:
    import orjson

    event_id = event.get("id")
    kind = event.get("kind") or "message"
    payload = orjson.dumps(event)
    data_lines = payload.split(b"\n")
    chunks = [
        b"id: " + str(event_id).encode() + b"\n",
        b"event: " + str(kind).encode() + b"\n",
    ]
    for line in data_lines:
        chunks.append(b"data: " + line + b"\n")
    chunks.append(b"\n")
    handler.wfile.write(b"".join(chunks))
    handler.wfile.flush()


__all__ = ["handle_events"]
