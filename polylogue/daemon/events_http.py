"""HTTP handlers for the realtime daemon-event channel (#957 / #1204).

Two shapes share the ``/api/events`` route:

- ``GET /api/events?poll=1&since=<id>`` — JSON snapshot of new events.
  Used by the ETag-style polling fallback when ``EventSource`` is not
  available in the client.
- ``GET /api/events?since=<id>`` — Server-Sent Events stream. Long-polls
  the daemon-event ledger for new rows and emits one SSE frame per
  event, with a heartbeat comment between idle ticks. Bounded by
  ``max_seconds`` so HTTP idle timeouts and tests cannot deadlock.

Granular topics (#1204): callers may filter by ``?kinds=`` (comma-list).
The reader subscribes by view — list view filters to ``session.*``;
the session view filters to ``message.appended`` plus
``insight.updated``.

Backpressure coalescing (#1204): when the ledger has produced more than
``coalesce_threshold`` events since ``since`` (default 100, override via
``?coalesce=<int>``), the SSE/poll path collapses the burst into a
single ``snapshot`` event carrying ``{kind: count}`` rather than
streaming each row to a slow client.

The handlers are pulled out of :mod:`polylogue.daemon.http` to keep that
module's file-size budget headroom for the broader archive read API.
"""

from __future__ import annotations

import contextlib
import time
from collections import Counter
from http import HTTPStatus
from typing import TYPE_CHECKING, cast

from polylogue.daemon.events import query_events_since

if TYPE_CHECKING:
    from polylogue.daemon.http import DaemonAPIHandler


_DEFAULT_COALESCE_THRESHOLD = 100
_MAX_COALESCE_THRESHOLD = 1000


def _resolve_coalesce_threshold(handler: DaemonAPIHandler, params: dict[str, list[str]]) -> int:
    threshold = handler._get_int(params, "coalesce", _DEFAULT_COALESCE_THRESHOLD)
    if threshold <= 0:
        return _DEFAULT_COALESCE_THRESHOLD
    return min(threshold, _MAX_COALESCE_THRESHOLD)


def _build_snapshot_event(events: list[dict[str, object]]) -> dict[str, object]:
    """Collapse a burst of events into a single ``snapshot`` envelope."""
    counts: Counter[str] = Counter()
    for event in events:
        kind = cast("str", event.get("kind", "")) or "unknown"
        counts[kind] += 1
    last = events[-1]
    return {
        "id": last["id"],
        "ts": last["ts"],
        "kind": "snapshot",
        "operation_id": None,
        "payload": {
            "event_count": len(events),
            "first_event_id": events[0]["id"],
            "last_event_id": last["id"],
            "kind_counts": dict(counts),
            "coalesced": True,
        },
    }


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
    coalesce_threshold = _resolve_coalesce_threshold(handler, params)

    if handler._get_bool(params, "poll"):
        events = list(query_events_since(since_param, kinds=kinds, limit=500))
        if len(events) > coalesce_threshold:
            snapshot = _build_snapshot_event(events)
            handler._send_json(
                HTTPStatus.OK,
                {
                    "events": [snapshot],
                    "last_event_id": snapshot["id"],
                    "coalesced": True,
                    "coalesced_count": len(events),
                },
            )
            return
        latest = events[-1]["id"] if events else since_param
        handler._send_json(
            HTTPStatus.OK,
            {"events": events, "last_event_id": latest},
        )
        return

    max_seconds = handler._get_int(params, "max_seconds", 30)
    if max_seconds <= 0 or max_seconds > 300:
        max_seconds = 30
    _stream_events(handler, since_param, kinds, max_seconds, coalesce_threshold)


def _stream_events(
    handler: DaemonAPIHandler,
    since: int,
    kinds: tuple[str, ...],
    max_seconds: int,
    coalesce_threshold: int,
) -> None:
    """Long-poll the event ledger and stream SSE frames to the client.

    When a single batch exceeds ``coalesce_threshold`` the burst is
    collapsed into one ``snapshot`` SSE frame instead of being streamed
    row-by-row. This is the slow-client backpressure path required by
    #1204; clients receiving a snapshot are expected to refetch their
    materialised view rather than animate row-level diffs.
    """
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
                if len(events) > coalesce_threshold:
                    snapshot = _build_snapshot_event(events)
                    _write_sse_event(handler, snapshot)
                    cursor = int(cast("int", snapshot["id"]))
                else:
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
