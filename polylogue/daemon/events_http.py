"""HTTP handlers for the realtime daemon-event channel (#957 / #1204).

Two shapes share the ``/api/events`` route:

- ``GET /api/events?poll=1&since=<cursor>`` — JSON snapshot of new events.
  Used by the ETag-style polling fallback when ``EventSource`` is not
  available in the client.
- ``GET /api/events?since=<cursor>`` — Server-Sent Events stream. Long-polls
  the daemon-event ledger for new rows and emits one SSE frame per
  event, with a heartbeat comment between idle ticks. Bounded by
  ``max_seconds`` so HTTP idle timeouts and tests cannot deadlock.

Granular topics (#1204): callers may filter by ``?kinds=`` (comma-list).
The reader subscribes by view — list view filters to ``session.*``;
the session view filters to ``message.appended``.

Backpressure coalescing (#1204): when the ledger has produced more than
``coalesce_threshold`` events since ``since`` (default 100, override via
``?coalesce=<int>``), the SSE/poll path collapses the burst into a
single ``snapshot`` event carrying ``{kind: count}`` rather than
streaming each row to a slow client.

Aged-out cursors (polylogue-20d.13.6): an open SSE stream is a live
subscriber, and retention keeps every row after its cursor. When
``since``/``Last-Event-ID`` names history the ledger no longer retains (a
poll, or a stream that reconnected after rows it had not read were pruned),
the handler answers with the
same ``snapshot`` envelope carrying ``resync: true`` and a reason instead
of a short or empty page, and the poll shape reports ``outcome:
"degraded"``. An empty page would read to the client as "nothing
happened".

The handlers are pulled out of :mod:`polylogue.daemon.http` to keep that
module's file-size budget headroom for the broader archive read API.
"""

from __future__ import annotations

import contextlib
import time
from collections import Counter
from http import HTTPStatus
from typing import TYPE_CHECKING, cast

from polylogue.daemon.events import (
    EVENT_SUBSCRIBERS,
    EventCursorStatus,
    build_snapshot_envelope,
    parse_event_cursor,
    query_events_since,
)

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
    """Collapse a burst of events into a single ``snapshot`` envelope.

    Shares :func:`polylogue.daemon.events.build_snapshot_envelope` with the
    aged-out-cursor resync path so a client learns one envelope shape, not two.
    """
    counts: Counter[str] = Counter()
    for event in events:
        kind = cast("str", event.get("kind", "")) or "unknown"
        counts[kind] += 1
    last = events[-1]
    snapshot = build_snapshot_envelope(
        event_id=cast("int", last["id"]),
        ts=cast("str", last["ts"]),
        event_count=len(events),
        first_event_id=cast("int", events[0]["id"]),
        last_event_id=cast("int", last["id"]),
        kind_counts=dict(counts),
    )
    snapshot["cursor"] = last["cursor"]
    return snapshot


def handle_events(handler: DaemonAPIHandler, params: dict[str, list[str]]) -> None:
    """Dispatch ``GET /api/events`` to either the poll or SSE shape."""
    since_param = handler._get_param(params, "since") or handler.headers.get("Last-Event-ID")
    try:
        parse_event_cursor(since_param)
    except ValueError:
        handler._send_json(HTTPStatus.BAD_REQUEST, {"error": "invalid_event_cursor"})
        return
    kinds_param = handler._get_param(params, "kinds")
    kinds: tuple[str, ...] = tuple(k.strip() for k in (kinds_param or "").split(",") if k.strip())
    coalesce_threshold = _resolve_coalesce_threshold(handler, params)

    if handler._get_bool(params, "poll"):
        page = query_events_since(since_param, kinds=kinds, limit=500)
        if page.status is EventCursorStatus.AGED_OUT:
            resync = cast("dict[str, object]", page.resync)
            # ``degraded`` outranks ``empty``: the subscriber asked from a
            # cursor the ledger no longer retains, so answering with rows --
            # or with none -- would report pruned history as "no change".
            handler._send_json(
                HTTPStatus.OK,
                {
                    "events": [resync],
                    "last_event_id": page.latest_cursor,
                    "outcome": "degraded",
                    "resync": True,
                    "resync_reason": cast("dict[str, object]", resync["payload"])["reason"],
                },
            )
            return
        events = list(page.events)
        if len(events) > coalesce_threshold:
            snapshot = _build_snapshot_event(events)
            handler._send_json(
                HTTPStatus.OK,
                {
                    "events": [snapshot],
                    "last_event_id": snapshot["cursor"],
                    "coalesced": True,
                    "coalesced_count": len(events),
                },
            )
            return
        latest = events[-1]["cursor"] if events else page.latest_cursor
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
    since: str | None,
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
    # While the stream is open it is a live subscriber: retention keeps every
    # row after its cursor (polylogue-20d.13.6).
    subscription = EVENT_SUBSCRIBERS.subscribe(cursor)
    try:
        _write_sse_comment(handler, b"open")
        while time.monotonic() < deadline:
            page = query_events_since(cursor, kinds=kinds, limit=200)
            if page.status is EventCursorStatus.AGED_OUT:
                resync = cast("dict[str, object]", page.resync)
                _write_sse_event(handler, resync)
                # The subscriber must refetch its materialized view; advancing
                # to the newest retained id is the cursor that refetch is
                # consistent with, and it cannot re-trip the same refusal.
                cursor = page.latest_cursor
                subscription.advance(cursor)
                time.sleep(1.0)
                continue
            events = list(page.events)
            if events:
                if len(events) > coalesce_threshold:
                    snapshot = _build_snapshot_event(events)
                    _write_sse_event(handler, snapshot)
                    cursor = cast("str", snapshot["cursor"])
                else:
                    for event in events:
                        _write_sse_event(handler, event)
                        cursor = cast("str", event["cursor"])
                subscription.advance(cursor)
            else:
                _write_sse_comment(handler, b"tick")
            time.sleep(1.0)
    except (BrokenPipeError, ConnectionResetError):
        return
    except Exception as exc:
        # polylogue-gmskd: the status line and headers are already on the
        # wire, so the JSON error path (``_send_json`` -> ``send_response``)
        # would write a second HTTP status line and header block *into* the
        # text/event-stream body. A mid-stream failure -- index.db locked or
        # replaced while a client is connected -- terminates the stream with
        # an SSE error frame, which is the only framing this connection can
        # still express.
        _terminate_stream_with_error(handler, exc)
        return
    finally:
        subscription.close()


def _terminate_stream_with_error(handler: DaemonAPIHandler, exc: BaseException) -> None:
    """End an open SSE stream with a typed ``error`` frame, never a status line."""
    from polylogue.logging import ERROR, emit

    emit(
        "daemon.http.event_stream_failed",
        level=ERROR,
        outcome="error",
        reason="stream_read_failed",
        route="_stream_events",
        error_type=type(exc).__name__,
        error_detail=str(exc),
    )
    with contextlib.suppress(BrokenPipeError, ConnectionResetError, OSError, ValueError):
        _write_sse_error(handler, "stream_read_failed", type(exc).__name__)


def _write_sse_error(handler: DaemonAPIHandler, reason: str, error_type: str) -> None:
    from polylogue.core.json import dumps_bytes

    payload = dumps_bytes({"error": reason, "error_type": error_type})
    chunks = [b"event: error\n"]
    for line in payload.split(b"\n"):
        chunks.append(b"data: " + line + b"\n")
    chunks.append(b"\n")
    handler.wfile.write(b"".join(chunks))
    handler.wfile.flush()


def _write_sse_comment(handler: DaemonAPIHandler, comment: bytes) -> None:
    handler.wfile.write(b": " + comment + b"\n\n")
    handler.wfile.flush()


def _write_sse_event(handler: DaemonAPIHandler, event: dict[str, object]) -> None:
    from polylogue.core.json import dumps_bytes

    event_id = event.get("cursor")
    if event_id is None:
        event_id = ""
    kind = event.get("kind") or "message"
    payload = dumps_bytes(event)
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
