"""Contract tests for the realtime ``/api/events`` channel (#957).

The daemon exposes daemon-event notifications to the web reader via two
shapes off the same handler:

- ``GET /api/events?poll=1&since=<cursor>`` — JSON snapshot of events with
  the opaque lifetime-bound continuation. Used by ETag/poll fallback when ``EventSource`` is
  unavailable.
- ``GET /api/events?since=<cursor>`` — Server-Sent Events stream of the same
  payload, bounded by ``max_seconds`` so HTTP idle timeouts and tests
  cannot deadlock.

``GET /api/status`` advertises the same lifetime-bound ``last_event_id`` and
sets a weak ``ETag`` so clients can long-poll without paying the full
status payload on every probe.
"""

from __future__ import annotations

import ast
import dataclasses
import hashlib
import importlib
import inspect
import json
import re
import sqlite3
from http import HTTPStatus
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING, cast
from unittest.mock import MagicMock, patch

import pytest

from polylogue.core.json import JSONDocument
from polylogue.daemon.web_auth import WebCredentialRegistry
from tests.infra.daemon_http_harness import MockDaemonServer, capture_json_response, make_daemon_handler

if TYPE_CHECKING:
    from polylogue.daemon.http import DaemonAPIHandler


def _make_handler(
    method: str,
    path: str,
    *,
    body: bytes = b"",
    extra_headers: dict[str, str] | None = None,
    server: object | None = None,
) -> DaemonAPIHandler:
    return make_daemon_handler(method, path, body=body, extra_headers=extra_headers, server=server)


def _complete_healthy_frontier() -> JSONDocument:
    return {
        "available": True,
        "overall_status": "healthy",
        "broken_head_status": "healthy",
        "broken_head_count": 0,
        "broken_head_checked_count": 1,
        "broken_head_samples": [],
        "broken_head_reason": "",
        "missing_source_raw_status": "healthy",
        "missing_source_raw_count": 0,
        "missing_source_raw_samples": [],
        "missing_source_raw_reason": "",
        "cursor_ahead_status": "healthy",
        "cursor_ahead_count": 0,
        "cursor_ahead_checked_count": 1,
        "cursor_head_comparison_count": 1,
        "cursor_ahead_comparison_count": 0,
        "cursor_ahead_samples": [],
        "cursor_authority_gap_count": 0,
        "cursor_authority_gap_samples": [],
        "cursor_ahead_reason": "",
    }


def _response_etag(response: bytes) -> str:
    match = re.search(rb"\r\nETag: ([^\r\n]+)", response)
    assert match is not None
    return match.group(1).decode("ascii")


def _response_json(response: bytes) -> dict[str, object]:
    _, body = response.split(b"\r\n\r\n", 1)
    payload = json.loads(body)
    assert isinstance(payload, dict)
    return cast(dict[str, object], payload)


def _cursor(position: int) -> str:
    from polylogue.daemon.events import get_latest_event_cursor

    current = get_latest_event_cursor()
    assert current is not None
    return current.rsplit(":", 1)[0] + f":{position}"


@pytest.fixture
def empty_events_db(workspace_env: dict[str, Path], monkeypatch: pytest.MonkeyPatch) -> Path:
    """Force the daemon-events DB into an isolated workspace."""
    from polylogue.daemon import events as events_mod

    events_path = workspace_env["archive_root"] / "daemon_events.db"

    def _path() -> Path:
        return events_path

    monkeypatch.setattr(events_mod, "_events_db_path", _path)
    return events_path


@pytest.fixture
def live_batch_archive(workspace_env: dict[str, Path], monkeypatch: pytest.MonkeyPatch) -> Path:
    """An isolated archive whose ``ops.db`` ledger the daemon's batch emitter names.

    The daemon passes its own archive root to ``_emit_live_batch_event``;
    readers here resolve the same ledger.
    """
    from polylogue.daemon import events as events_mod

    archive_root = workspace_env["archive_root"]
    monkeypatch.setattr(events_mod, "_events_db_path", lambda: archive_root / "ops.db")
    return archive_root


class TestEventsPollFallback:
    """``GET /api/events?poll=1`` returns JSON envelopes for ETag-style polling."""

    def test_poll_with_no_events_returns_empty_envelope(self, empty_events_db: Path) -> None:
        handler = _make_handler("GET", "/api/events?poll=1")
        send_json = capture_json_response(handler)
        handler.do_GET()

        send_json.assert_called_once()
        status, payload = send_json.call_args.args
        assert status == HTTPStatus.OK
        assert payload == {"events": [], "last_event_id": None}
        assert not empty_events_db.exists()

    def test_poll_returns_events_after_since(self, empty_events_db: Path) -> None:
        from polylogue.daemon.events import emit_daemon_event

        emit_daemon_event("ingestion_batch", payload={"files": 1})
        emit_daemon_event("ingest", operation_id="op-2", payload={"path": "/tmp/x"})

        handler = _make_handler("GET", "/api/events?poll=1")
        send_json = capture_json_response(handler)
        handler.do_GET()

        status, payload = send_json.call_args.args
        assert status == HTTPStatus.OK
        events = payload["events"]
        assert [e["kind"] for e in events] == ["ingestion_batch", "ingest"]
        assert payload["last_event_id"] == events[-1]["cursor"]

    def test_poll_kinds_filter_whitelist(self, empty_events_db: Path) -> None:
        from polylogue.daemon.events import emit_daemon_event

        emit_daemon_event("ingestion_batch", payload={"n": 1})
        emit_daemon_event("noise", payload={"n": 2})

        handler = _make_handler("GET", "/api/events?poll=1&kinds=ingestion_batch,ingest")
        send_json = capture_json_response(handler)
        handler.do_GET()

        payload = send_json.call_args.args[1]
        kinds = {e["kind"] for e in payload["events"]}
        assert kinds == {"ingestion_batch"}

    def test_poll_since_is_strict_gt(self, empty_events_db: Path) -> None:
        from polylogue.daemon.events import emit_daemon_event

        emit_daemon_event("ingestion_batch", payload={})
        handler = _make_handler("GET", "/api/events?poll=1")
        send_json = capture_json_response(handler)
        handler.do_GET()
        first_id = send_json.call_args.args[1]["events"][0]["cursor"]

        handler = _make_handler("GET", f"/api/events?poll=1&since={first_id}")
        send_json = capture_json_response(handler)
        handler.do_GET()
        assert send_json.call_args.args[1] == {"events": [], "last_event_id": first_id}


class TestEventsSSEStream:
    """``GET /api/events`` (no ``poll``) writes a Server-Sent Events stream."""

    def test_sse_stream_emits_pending_events_and_closes(self, empty_events_db: Path) -> None:
        from polylogue.daemon.events import emit_daemon_event

        emit_daemon_event("ingestion_batch", payload={"files": 3})
        emit_daemon_event("ingest", operation_id="op-99", payload={})

        handler = _make_handler("GET", "/api/events?max_seconds=1")
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
        handler_first = _make_handler("GET", "/api/events?poll=1")
        send_json_first = capture_json_response(handler_first)
        handler_first.do_GET()
        first_id = send_json_first.call_args.args[1]["events"][0]["cursor"]

        handler = _make_handler(
            "GET",
            "/api/events?max_seconds=1",
            extra_headers={"Last-Event-ID": str(first_id)},
        )
        handler.do_GET()
        out = cast("BytesIO", handler.wfile).getvalue()
        # Only the second event should be present.
        assert out.count(b"event: ingestion_batch\n") == 1


class TestEventLedgerReadIsolation:
    """Event/status observation never initializes or contends for ops writes."""

    def test_read_helpers_leave_missing_tier_and_parent_absent(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from polylogue.daemon import events as events_mod

        events_path = tmp_path / "fresh" / "ops.db"
        monkeypatch.setattr(events_mod, "_events_db_path", lambda: events_path)

        def unexpected(*_args: object, **_kwargs: object) -> None:
            pytest.fail("event readers must not initialize or open an ops writer")

        monkeypatch.setattr(events_mod, "initialize_archive_database", unexpected)
        monkeypatch.setattr(events_mod, "open_daemon_connection", unexpected)

        assert list(events_mod.iter_daemon_events()) == []
        assert events_mod.query_events_since(None).events == ()
        assert events_mod.get_latest_event_cursor() is None
        assert events_mod.get_daemon_event_counts() == {}
        assert events_mod.get_last_ingestion_batch() is None
        assert list(events_mod.get_recent_operations()) == []
        assert not events_path.parent.exists()

    def test_read_helpers_leave_schema_less_ops_file_unchanged(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from polylogue.daemon import events as events_mod

        events_path = tmp_path / "ops.db"
        with sqlite3.connect(events_path) as conn:
            conn.execute("CREATE TABLE sentinel (value TEXT NOT NULL)")
            conn.execute("INSERT INTO sentinel VALUES ('preserved')")
        size_before = events_path.stat().st_size
        monkeypatch.setattr(events_mod, "_events_db_path", lambda: events_path)

        def unexpected(*_args: object, **_kwargs: object) -> None:
            pytest.fail("event readers must not initialize or open an ops writer")

        monkeypatch.setattr(events_mod, "initialize_archive_database", unexpected)
        monkeypatch.setattr(events_mod, "open_daemon_connection", unexpected)

        assert list(events_mod.iter_daemon_events()) == []
        assert events_mod.query_events_since(None).events == ()
        assert events_mod.get_latest_event_cursor() is None
        assert events_mod.get_daemon_event_counts() == {}
        assert events_path.stat().st_size == size_before
        with sqlite3.connect(f"file:{events_path}?mode=ro", uri=True) as conn:
            assert conn.execute("SELECT value FROM sentinel").fetchone() == ("preserved",)
            assert conn.execute("SELECT 1 FROM sqlite_master WHERE name = 'daemon_events'").fetchone() is None

    def test_reads_remain_query_only_during_active_writer_transaction(
        self,
        empty_events_db: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from polylogue.daemon import events as events_mod

        events_mod.emit_daemon_event("committed", payload={"value": 1})
        writer = sqlite3.connect(empty_events_db, timeout=0.1)
        try:
            writer.execute("BEGIN IMMEDIATE")
            writer.execute(
                "INSERT INTO daemon_events (ts_ms, kind, operation_id, payload_json) VALUES (2, 'uncommitted', NULL, '{}')"
            )
            writer_changes = writer.total_changes

            def unexpected(*_args: object, **_kwargs: object) -> None:
                pytest.fail("event readers must not initialize or open an ops writer")

            monkeypatch.setattr(events_mod, "initialize_archive_database", unexpected)
            monkeypatch.setattr(events_mod, "open_daemon_connection", unexpected)

            assert [event["kind"] for event in events_mod.iter_daemon_events()] == ["committed"]
            assert [event["kind"] for event in events_mod.query_events_since(None).events] == ["committed"]
            assert events_mod.get_latest_event_cursor() == _cursor(1)
            assert events_mod.get_daemon_event_counts() == {"committed": 1}
            assert writer.in_transaction is True
            assert writer.total_changes == writer_changes
            writer.execute(
                "INSERT INTO daemon_events (ts_ms, kind, operation_id, payload_json) VALUES (3, 'writer-still-active', NULL, '{}')"
            )
            writer.commit()
        finally:
            writer.close()

        assert [event["kind"] for event in events_mod.query_events_since(_cursor(1)).events] == [
            "uncommitted",
            "writer-still-active",
        ]


class TestStatusEventEtag:
    """``GET /api/status`` ETags include event and normalized snapshot identity."""

    def test_status_includes_last_event_id_field(self, empty_events_db: Path) -> None:
        from polylogue.daemon.events import emit_daemon_event

        emit_daemon_event("ingestion_batch", payload={})
        handler = _make_handler("GET", "/api/status")
        handler.do_GET()
        out = cast("BytesIO", handler.wfile).getvalue()
        assert b'ETag: W/"status-' in out
        assert b'"last_event_id":' in out

    def test_status_returns_304_when_etag_matches(self, empty_events_db: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        from polylogue.daemon.events import emit_daemon_event
        from polylogue.daemon.status_snapshot import refresh_status_snapshot

        monkeypatch.setattr("polylogue.daemon.status_snapshot.time.monotonic", lambda: 100.0)
        monkeypatch.setattr("polylogue.daemon.status_snapshot._status_frame", lambda: "frame-A")
        emit_daemon_event("ingestion_batch", payload={})
        refresh_status_snapshot(payload={"ok": False, "daemon_liveness": True})
        first = _make_handler("GET", "/api/status")
        first.do_GET()
        etag = _response_etag(cast("BytesIO", first.wfile).getvalue())

        handler = _make_handler("GET", "/api/status", extra_headers={"If-None-Match": etag})
        handler.do_GET()
        out = cast("BytesIO", handler.wfile).getvalue()
        assert b" 304 " in out
        assert b"Content-Type: application/json" not in out

    def test_status_etag_changes_when_snapshot_becomes_stale(
        self,
        empty_events_db: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from polylogue.daemon.events import emit_daemon_event
        from polylogue.daemon.status_snapshot import refresh_status_snapshot

        emit_daemon_event("ingestion_batch", payload={})
        monkeypatch.setattr("polylogue.daemon.status_snapshot.time.monotonic", lambda: 100.0)
        refresh_status_snapshot(
            payload={
                "ok": True,
                "daemon_liveness": True,
                "raw_frontier_integrity": _complete_healthy_frontier(),
            }
        )
        first = _make_handler("GET", "/api/status")
        first.do_GET()
        etag = _response_etag(cast("BytesIO", first.wfile).getvalue())

        monkeypatch.setattr("polylogue.daemon.status_snapshot.time.monotonic", lambda: 131.0)
        second = _make_handler("GET", "/api/status", extra_headers={"If-None-Match": etag})
        second.do_GET()
        response = cast("BytesIO", second.wfile).getvalue()
        payload = _response_json(response)

        assert b" 200 " in response
        assert payload["ok"] is False
        snapshot = cast(dict[str, object], payload["status_snapshot"])
        frontier = cast(dict[str, object], payload["raw_frontier_integrity"])
        assert snapshot["state"] == "stale"
        assert frontier["overall_status"] == "unknown"

    def test_status_etag_changes_when_archive_frame_changes(
        self,
        empty_events_db: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from polylogue.daemon.events import emit_daemon_event
        from polylogue.daemon.status_snapshot import refresh_status_snapshot

        frame = {"value": "A"}
        monkeypatch.setattr("polylogue.daemon.status_snapshot.time.monotonic", lambda: 100.0)
        monkeypatch.setattr("polylogue.daemon.status_snapshot._status_frame", lambda: frame["value"])
        emit_daemon_event("ingestion_batch", payload={})
        refresh_status_snapshot(payload={"ok": False, "checked_at": "A"})
        first = _make_handler("GET", "/api/status")
        first.do_GET()
        etag = _response_etag(cast("BytesIO", first.wfile).getvalue())

        frame["value"] = "B"
        second = _make_handler("GET", "/api/status", extra_headers={"If-None-Match": etag})
        second.do_GET()
        response = cast("BytesIO", second.wfile).getvalue()
        payload = _response_json(response)
        assert b" 200 " in response
        assert cast(dict[str, object], payload["status_snapshot"])["state"] == "stale"
        assert cast(dict[str, object], payload["status_snapshot"])["current_frame"] == "B"

        frame["value"] = "C"
        third = _make_handler("GET", "/api/status", extra_headers={"If-None-Match": _response_etag(response)})
        third.do_GET()
        third_response = cast("BytesIO", third.wfile).getvalue()
        assert b" 200 " in third_response
        assert cast(dict[str, object], _response_json(third_response)["status_snapshot"])["current_frame"] == "C"

    def test_status_etag_changes_with_live_discovery_without_event(
        self,
        empty_events_db: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from polylogue.daemon.discovery_progress import (
            advance_discovery,
            begin_discovery,
            end_discovery,
            reset_discovery_progress,
        )
        from polylogue.daemon.status_snapshot import refresh_status_snapshot

        monkeypatch.setattr("polylogue.daemon.status_snapshot.time.monotonic", lambda: 100.0)
        monkeypatch.setattr("polylogue.daemon.status_snapshot._status_frame", lambda: "frame-A")
        refresh_status_snapshot(payload={"ok": False, "catchup": {"mode": "idle"}})
        token = begin_discovery("synthetic")
        try:
            first = _make_handler("GET", "/api/status")
            first.do_GET()
            first_payload = _response_json(cast("BytesIO", first.wfile).getvalue())
            first_etag = _response_etag(cast("BytesIO", first.wfile).getvalue())
            advance_discovery(token, inspected=1, disposition="accepted")

            second = _make_handler("GET", "/api/status", extra_headers={"If-None-Match": first_etag})
            second.do_GET()
            response = cast("BytesIO", second.wfile).getvalue()
            second_payload = _response_json(response)
            assert b" 200 " in response
            assert second_payload["last_event_id"] == first_payload["last_event_id"]
            assert cast(dict[str, object], first_payload["catchup"])["discovery_inspected_count"] == 0
            assert cast(dict[str, object], second_payload["catchup"])["discovery_inspected_count"] == 1
            assert _response_etag(response) != first_etag
            from polylogue.daemon.http import _json_bytes, _stable_status_identity

            identity = _json_bytes(_stable_status_identity(second_payload))
            assert _response_etag(response) == f'W/"status-{hashlib.sha256(identity).hexdigest()[:24]}"'
        finally:
            end_discovery(token)
            reset_discovery_progress()

    def test_status_etag_changes_when_snapshot_refreshes_to_violation(
        self,
        empty_events_db: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from polylogue.daemon.events import emit_daemon_event
        from polylogue.daemon.status_snapshot import refresh_status_snapshot

        emit_daemon_event("ingestion_batch", payload={})
        monkeypatch.setattr("polylogue.daemon.status_snapshot.time.monotonic", lambda: 100.0)
        refresh_status_snapshot(
            payload={
                "ok": True,
                "daemon_liveness": True,
                "raw_frontier_integrity": _complete_healthy_frontier(),
            }
        )
        first = _make_handler("GET", "/api/status")
        first.do_GET()
        etag = _response_etag(cast("BytesIO", first.wfile).getvalue())

        violated = _complete_healthy_frontier()
        violated.update(
            {
                "overall_status": "violated",
                "broken_head_status": "violated",
                "broken_head_count": 1,
                "broken_head_samples": [
                    {"logical_source_key": "codex:one", "accepted_raw_id": "raw-one", "reason": "broken"}
                ],
                "broken_head_reason": "1 active seed is broken",
            }
        )
        monkeypatch.setattr("polylogue.daemon.status_snapshot.time.monotonic", lambda: 101.0)
        refresh_status_snapshot(payload={"ok": False, "daemon_liveness": True, "raw_frontier_integrity": violated})
        second = _make_handler("GET", "/api/status", extra_headers={"If-None-Match": etag})
        second.do_GET()
        response = cast("BytesIO", second.wfile).getvalue()
        payload = _response_json(response)

        assert b" 200 " in response
        frontier = cast(dict[str, object], payload["raw_frontier_integrity"])
        assert frontier["overall_status"] == "violated"

    def test_status_etag_changes_with_live_write_coordinator_state(
        self,
        empty_events_db: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from polylogue.daemon.events import emit_daemon_event
        from polylogue.daemon.status_snapshot import refresh_status_snapshot

        emit_daemon_event("ingestion_batch", payload={})
        refresh_status_snapshot(
            payload={
                "ok": True,
                "daemon_liveness": True,
                "raw_frontier_integrity": _complete_healthy_frontier(),
            }
        )
        monkeypatch.setattr(
            "polylogue.daemon.status_snapshot._daemon_write_coordinator_payload",
            lambda: {"state": "idle"},
        )
        first = _make_handler("GET", "/api/status")
        first.do_GET()
        etag = _response_etag(cast("BytesIO", first.wfile).getvalue())

        monkeypatch.setattr(
            "polylogue.daemon.status_snapshot._daemon_write_coordinator_payload",
            lambda: {"state": "writing"},
        )
        second = _make_handler("GET", "/api/status", extra_headers={"If-None-Match": etag})
        second.do_GET()
        response = cast("BytesIO", second.wfile).getvalue()
        payload = _response_json(response)

        assert b" 200 " in response
        assert payload["daemon_write_coordinator"] == {"state": "writing"}


class TestGranularEventKinds:
    """#1204 — granular SSE topics for selective subscription and live tail."""

    def test_emit_session_appended_payload_shape(self, empty_events_db: Path) -> None:
        from polylogue.daemon.events import emit_session_appended, iter_daemon_events

        emit_session_appended(
            source_name="claude-code-session",
            succeeded_file_count=3,
            failed_file_count=1,
            source_paths=["/tmp/a.jsonl", "/tmp/b.jsonl"],
            session_id="claude-code-session:conv-abc",
        )
        events = list(iter_daemon_events(limit=10))
        assert events[0]["kind"] == "session.appended"
        payload = cast("dict[str, object]", events[0]["payload"])
        assert payload["source_name"] == "claude-code-session"
        assert payload["succeeded_file_count"] == 3
        assert payload["failed_file_count"] == 1
        assert payload["source_paths"] == ["/tmp/a.jsonl", "/tmp/b.jsonl"]
        # Identity-scoped ref (polylogue-20d.13): a reader must be able to
        # tell whether this event describes the session it has open.
        assert payload["session_id"] == "claude-code-session:conv-abc"

    def test_emit_session_updated_payload_shape(self, empty_events_db: Path) -> None:
        from polylogue.daemon.events import emit_session_updated, iter_daemon_events

        emit_session_updated(
            session_id="codex:conv-xyz",
            source_name="codex",
            appended_count=2,
        )
        events = list(iter_daemon_events(limit=10))
        assert events[0]["kind"] == "session.updated"
        payload = cast("dict[str, object]", events[0]["payload"])
        assert payload["session_id"] == "codex:conv-xyz"
        assert payload["source_name"] == "codex"
        assert payload["appended_count"] == 2

    def test_emit_message_appended_payload_shape(self, empty_events_db: Path) -> None:
        from polylogue.daemon.events import emit_message_appended, iter_daemon_events

        emit_message_appended(
            session_id="conv-abc",
            source_name="codex",
            appended_count=4,
            source_path="/tmp/session.json",
        )
        events = list(iter_daemon_events(limit=10))
        assert events[0]["kind"] == "message.appended"
        payload = cast("dict[str, object]", events[0]["payload"])
        assert payload["session_id"] == "conv-abc"
        assert payload["appended_count"] == 4
        assert payload["source_path"] == "/tmp/session.json"

    def test_selective_subscription_via_kinds(self, empty_events_db: Path) -> None:
        from polylogue.daemon.events import (
            emit_message_appended,
            emit_session_appended,
            emit_session_updated,
        )

        emit_session_appended(source_name=None, succeeded_file_count=1)
        emit_message_appended(session_id="c", appended_count=1)
        emit_session_updated(session_id="c", appended_count=1)

        handler = _make_handler(
            "GET",
            "/api/events?poll=1&kinds=message.appended,session.updated",
        )
        send_json = capture_json_response(handler)
        handler.do_GET()
        kinds = {e["kind"] for e in send_json.call_args.args[1]["events"]}
        assert kinds == {"message.appended", "session.updated"}


class TestLiveBatchEventFanOut:
    """polylogue-20d.13 — live-ingest batches fan out identity-scoped events.

    ``_emit_live_batch_event`` is the daemon-side translator between the
    generic ``ingestion_batch`` metrics payload (produced deep in
    ``polylogue.sources.live.batch``) and the granular SSE topics. These
    tests exercise it directly against the real ``daemon.events`` emitters
    and the real event ledger, so a regression that drops ``session_id``
    threading (e.g. reverting to the pre-#20d.13 aggregate-only emission)
    fails here even without a full live-ingest fixture.
    """

    def test_batch_with_new_and_updated_sessions_emits_scoped_events(self, live_batch_archive: Path) -> None:
        from polylogue.daemon.cli import _emit_live_batch_event
        from polylogue.daemon.events import iter_daemon_events

        _emit_live_batch_event(
            "ingestion_batch",
            {
                "succeeded_file_count": 2,
                "failed_file_count": 0,
                "new_sessions": [{"source_name": "codex", "session_id": "codex:new-1"}],
                "updated_sessions": [{"source_name": "claude-code", "session_id": "claude-code:existing-1"}],
            },
            archive_root_path=live_batch_archive,
        )
        events = list(iter_daemon_events(limit=10))
        by_kind: dict[str, list[dict[str, object]]] = {}
        for event in events:
            by_kind.setdefault(cast("str", event["kind"]), []).append(cast("dict[str, object]", event["payload"]))

        assert len(by_kind["session.appended"]) == 1
        assert by_kind["session.appended"][0]["session_id"] == "codex:new-1"
        assert by_kind["session.appended"][0]["source_name"] == "codex"

        assert len(by_kind["session.updated"]) == 1
        assert by_kind["session.updated"][0]["session_id"] == "claude-code:existing-1"
        assert by_kind["session.updated"][0]["source_name"] == "claude-code"

        # message.appended fires once per distinct touched session, each
        # scoped to that session's own id -- never the aggregate None the
        # description names as the identity defect ("an unscoped message
        # event currently refreshes whichever session a browser has open").
        message_session_ids = {payload["session_id"] for payload in by_kind["message.appended"]}
        assert message_session_ids == {"codex:new-1", "claude-code:existing-1"}
        assert None not in message_session_ids

    def test_batch_touching_only_session_b_never_names_session_a(self, live_batch_archive: Path) -> None:
        """The exact regression the bead describes: session A must be unaffected."""
        from polylogue.daemon.cli import _emit_live_batch_event
        from polylogue.daemon.events import iter_daemon_events

        _emit_live_batch_event(
            "ingestion_batch",
            {
                "succeeded_file_count": 1,
                "failed_file_count": 0,
                "new_sessions": [],
                "updated_sessions": [{"source_name": "codex", "session_id": "codex:session-b"}],
            },
            archive_root_path=live_batch_archive,
        )
        events = list(iter_daemon_events(limit=10))
        seen_session_ids = {
            cast("dict[str, object]", event["payload"])["session_id"]
            for event in events
            if event["kind"] in ("session.updated", "message.appended")
        }
        assert seen_session_ids == {"codex:session-b"}
        assert "codex:session-a" not in seen_session_ids

    def test_batch_without_resolved_identity_falls_back_to_unscoped_aggregate(self, live_batch_archive: Path) -> None:
        """No source path yet threads identity through -- preserve the old signal."""
        from polylogue.daemon.cli import _emit_live_batch_event
        from polylogue.daemon.events import iter_daemon_events

        _emit_live_batch_event(
            "ingestion_batch",
            {"succeeded_file_count": 1, "failed_file_count": 0},
            archive_root_path=live_batch_archive,
        )
        events = list(iter_daemon_events(limit=10))
        kinds = {cast("str", event["kind"]) for event in events}
        assert kinds == {"ingestion_batch", "session.appended", "message.appended"}
        for event in events:
            if event["kind"] in ("session.appended", "message.appended"):
                assert cast("dict[str, object]", event["payload"])["session_id"] is None

    def test_batch_and_its_session_events_land_in_one_ledger_transaction(
        self, live_batch_archive: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Anti-vacuity: emitting each fanned-out event on its own opens the
        ledger once per event (five here) instead of once for the batch."""
        from polylogue.daemon import events as events_module
        from polylogue.daemon.cli import _emit_live_batch_event

        opened: list[object] = []
        real_ensure = events_module._ensure_events_db

        def counting_ensure(*args: object) -> sqlite3.Connection:
            opened.append(args)
            return real_ensure(*args)  # type: ignore[arg-type]

        monkeypatch.setattr(events_module, "_ensure_events_db", counting_ensure)
        _emit_live_batch_event(
            "ingestion_batch",
            {
                "succeeded_file_count": 2,
                "failed_file_count": 0,
                "new_sessions": [
                    {"source_name": "codex", "session_id": "codex:a"},
                    {"source_name": "codex", "session_id": "codex:b"},
                ],
            },
            archive_root_path=live_batch_archive,
        )
        assert len(opened) == 1
        kinds = [cast("str", event["kind"]) for event in reversed(list(events_module.iter_daemon_events(limit=10)))]
        assert kinds == [
            "ingestion_batch",
            "session.appended",
            "message.appended",
            "session.appended",
            "message.appended",
        ]

    def test_zero_succeeded_batch_emits_no_granular_events(self, live_batch_archive: Path) -> None:
        from polylogue.daemon.cli import _emit_live_batch_event
        from polylogue.daemon.events import iter_daemon_events

        _emit_live_batch_event(
            "ingestion_batch",
            {"succeeded_file_count": 0, "failed_file_count": 3},
            archive_root_path=live_batch_archive,
        )
        events = list(iter_daemon_events(limit=10))
        assert {cast("str", event["kind"]) for event in events} == {"ingestion_batch"}


class TestBackpressureCoalescing:
    """#1204 — bursts collapse into one ``snapshot`` envelope for slow clients."""

    def test_poll_coalesces_burst_into_snapshot(self, empty_events_db: Path) -> None:
        from polylogue.daemon.events import emit_message_appended

        for _ in range(20):
            emit_message_appended(session_id="c", appended_count=1)

        handler = _make_handler("GET", "/api/events?poll=1&coalesce=5")
        send_json = capture_json_response(handler)
        handler.do_GET()
        payload = send_json.call_args.args[1]
        assert payload["coalesced"] is True
        assert payload["coalesced_count"] == 20
        assert len(payload["events"]) == 1
        snapshot = payload["events"][0]
        assert snapshot["kind"] == "snapshot"
        assert snapshot["payload"]["event_count"] == 20
        assert snapshot["payload"]["kind_counts"] == {"message.appended": 20}
        # last_event_id advances past every coalesced row so the next
        # request doesn't replay the same burst.
        assert payload["last_event_id"] == snapshot["cursor"]

    def test_poll_below_threshold_returns_individual_events(self, empty_events_db: Path) -> None:
        from polylogue.daemon.events import emit_message_appended

        for _ in range(3):
            emit_message_appended(session_id="c", appended_count=1)

        handler = _make_handler("GET", "/api/events?poll=1&coalesce=10")
        send_json = capture_json_response(handler)
        handler.do_GET()
        payload = send_json.call_args.args[1]
        assert payload.get("coalesced") is None
        assert len(payload["events"]) == 3
        assert all(e["kind"] == "message.appended" for e in payload["events"])

    def test_sse_stream_coalesces_burst(self, empty_events_db: Path) -> None:
        from polylogue.daemon.events import emit_message_appended

        for _ in range(15):
            emit_message_appended(session_id="c", appended_count=1)

        handler = _make_handler("GET", "/api/events?max_seconds=1&coalesce=5")
        handler.do_GET()
        out = cast("BytesIO", handler.wfile).getvalue()
        # A coalesced burst emits exactly one snapshot frame, not 15.
        assert out.count(b"event: snapshot\n") == 1
        assert b'"coalesced": true' in out or b'"coalesced":true' in out


class TestAccessTokenQueryRejected:
    """SSE credentials never travel in URLs; EventSource uses the web cookie."""

    @pytest.mark.parametrize("credential_param", ["access_token", "api_key", "secret"])
    def test_credential_in_query_string_is_rejected(
        self,
        empty_events_db: Path,
        credential_param: str,
    ) -> None:
        handler = _make_handler(
            "GET",
            f"/api/events?poll=1&{credential_param}=secret",
            server=MockDaemonServer(auth_token="secret"),
        )
        send_error = MagicMock()
        handler._send_error = send_error  # type: ignore[method-assign]
        handler.do_GET()
        send_error.assert_called_once_with(
            HTTPStatus.BAD_REQUEST,
            "credential_in_query",
            "credentials must use Authorization or the protected first-party cookie",
        )

    def test_access_token_in_query_string_is_rejected_on_non_sse_routes(self, empty_events_db: Path) -> None:
        handler = _make_handler(
            "GET",
            "/api/status?access_token=secret",
            server=MockDaemonServer(auth_token="secret"),
        )
        send_error = MagicMock()
        handler._send_error = send_error  # type: ignore[method-assign]
        handler.do_GET()
        send_error.assert_called_once_with(
            HTTPStatus.BAD_REQUEST,
            "credential_in_query",
            "credentials must use Authorization or the protected first-party cookie",
        )

    def test_valid_cookie_cannot_make_query_credential_reach_dispatch(self, empty_events_db: Path) -> None:
        registry = WebCredentialRegistry()
        issued = registry.issue("http://127.0.0.1:8766")
        handler = _make_handler(
            "GET",
            "/api/status?access_token=must-not-surface",
            extra_headers={
                "Cookie": f"polylogue_web_credential={issued.token}",
                "Host": "127.0.0.1:8766",
                "Sec-Fetch-Site": "same-origin",
                "X-Polylogue-Web-Client": "1",
            },
            server=MockDaemonServer(auth_token="secret", web_credentials=registry),
        )
        status_handler = MagicMock()
        handler._handle_status = status_handler  # type: ignore[method-assign]
        send_error = MagicMock()
        handler._send_error = send_error  # type: ignore[method-assign]

        handler.do_GET()

        send_error.assert_called_once_with(
            HTTPStatus.BAD_REQUEST,
            "credential_in_query",
            "credentials must use Authorization or the protected first-party cookie",
        )
        status_handler.assert_not_called()

    def test_route_metadata_redacts_and_disconnect_log_path_drops_query_values(self) -> None:
        from polylogue.daemon.http import _public_route_from_request_path, _request_path_for_log

        raw = "/api/sessions?query=normal&unknown_name=must-not-surface"
        assert _public_route_from_request_path(raw) == "/api/sessions"
        assert _request_path_for_log(raw) == "/api/sessions"


def test_emitter_converges_ops_tier_once_per_process(
    workspace_env: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    # The multi-statement tier initialization can wait out SQLite lock
    # timeouts statement by statement; running it on every emit put that
    # aggregate wait inside the daemon's SIGTERM exit window
    # (polylogue-b9oi8). First emit converges, later emits reuse it.
    from typing import Any

    from polylogue.daemon import events as events_module
    from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_database

    calls: list[object] = []

    def counting_initialize(path: Path, tier: Any, **kwargs: Any) -> None:
        calls.append(path)
        initialize_archive_database(path, tier, **kwargs)

    monkeypatch.setattr(events_module, "initialize_archive_database", counting_initialize)
    events_module._CONVERGED_EVENT_DBS.clear()
    events_module.emit_daemon_event("test-event", payload={"n": 1})
    events_module.emit_daemon_event("test-event", payload={"n": 2})

    assert len(calls) == 1


def _events_module_source() -> ast.Module:
    """Parse ``polylogue/daemon/events.py`` from disk, without importing it."""
    from polylogue.daemon import events as events_mod

    source_file = inspect.getsourcefile(events_mod)
    assert source_file is not None
    return ast.parse(Path(source_file).read_text(encoding="utf-8"))


def _advertised_topic_constants() -> dict[str, str]:
    """Derive the advertised-topic denominator from the code that advertises them.

    Every module-level ``EVENT_*`` assignment whose value is a string literal in
    ``polylogue/daemon/events.py`` is an advertised SSE topic. The denominator
    is read out of the constant *declarations* -- not out of ``EVENT_SPECS``,
    and not out of ``GRANULAR_EVENT_KINDS`` (which is derived from the
    registry) -- so the enumeration below cannot see the thing it checks. Add a
    fourth ``EVENT_FOO = "foo.bar"`` constant and this denominator grows even
    though the registry did not.
    """
    topics: dict[str, str] = {}
    for node in _events_module_source().body:
        if not isinstance(node, ast.Assign):
            continue
        if not isinstance(node.value, ast.Constant) or not isinstance(node.value.value, str):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id.startswith("EVENT_"):
                topics[target.id] = node.value.value
    return topics


def _production_emit_targets() -> set[str]:
    """Names of ``EVENT_*`` constants production emits.

    A second denominator, read from the call sites rather than the declarations:
    a topic constant that no production function ever emits is advertised with
    zero producers -- the regression that retired ``insight.updated`` and the
    two ``progress.*`` topics. A topic is emitted either directly through
    ``emit_daemon_event`` or as a ``DaemonEventRecord`` that a builder hands to
    ``emit_daemon_events`` (one ledger transaction per live batch).
    """
    emitted: set[str] = set()
    for node in ast.walk(_events_module_source()):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not (isinstance(func, ast.Name) and func.id in {"emit_daemon_event", "DaemonEventRecord"}):
            continue
        if node.args and isinstance(node.args[0], ast.Name):
            emitted.add(node.args[0].id)
    return emitted


class TestAdvertisedTopicContracts:
    """Every advertised SSE topic carries a declared, production-backed contract."""

    def test_every_advertised_topic_constant_has_a_registry_entry(self) -> None:
        """Anti-vacuity: a topic constant with no ``EVENT_SPECS`` entry fails here.

        The denominator comes from the module's own ``EVENT_*`` string
        declarations via ``ast``; the registry is the numerator. Deleting a
        spec entry, or adding a topic constant without one, is red.
        """
        from polylogue.daemon.events import EVENT_SPECS

        advertised = set(_advertised_topic_constants().values())
        assert advertised, "no advertised topic constants were discovered -- the denominator is broken"
        assert advertised == set(EVENT_SPECS), (
            f"topics advertised without a declared contract: {sorted(advertised - set(EVENT_SPECS))}; "
            f"contracts for unadvertised topics: {sorted(set(EVENT_SPECS) - advertised)}"
        )

    def test_granular_kinds_are_exactly_the_advertised_topics(self) -> None:
        from polylogue.daemon.events import GRANULAR_EVENT_KINDS

        assert set(_advertised_topic_constants().values()) == set(GRANULAR_EVENT_KINDS)

    def test_every_advertised_topic_has_a_production_emitter_in_this_module(self) -> None:
        """A topic nobody emits is an advertised channel with no producer."""
        emitted_constants = _production_emit_targets()
        advertised_constants = set(_advertised_topic_constants())
        assert advertised_constants <= emitted_constants, (
            f"advertised topics with no production emit call: {sorted(advertised_constants - emitted_constants)}"
        )

    def test_declared_emitter_resolves_to_production_code_not_tests(self) -> None:
        """Anti-vacuity: pointing a spec's emitter at a tests-only function fails here."""
        import polylogue
        from polylogue.daemon.events import EVENT_SPECS

        package_root = Path(inspect.getsourcefile(polylogue) or "").parent.resolve()
        for spec in EVENT_SPECS.values():
            module_path, _, attribute = spec.emitter.rpartition(".")
            module = importlib.import_module(module_path)
            emitter = getattr(module, attribute, None)
            assert callable(emitter), f"{spec.kind}: declared emitter {spec.emitter} is not callable"
            source_file = inspect.getsourcefile(emitter)
            assert source_file is not None
            resolved = Path(source_file).resolve()
            assert resolved.is_relative_to(package_root), (
                f"{spec.kind}: emitter {spec.emitter} lives outside the polylogue package at {resolved}"
            )
            assert "tests" not in resolved.parts, (
                f"{spec.kind}: emitter {spec.emitter} is defined under tests/, not production code"
            )

    def test_emitted_payload_matches_the_declared_projection(self, empty_events_db: Path) -> None:
        """Run each declared production emitter and check its payload against its spec.

        Makes ``payload_projection``/``required_payload_fields`` load-bearing:
        an emitter that starts publishing an undeclared key, or stops carrying
        its object ref, goes red here rather than shipping an undeclared
        projection to subscribers.
        """
        from polylogue.daemon.events import (
            EVENT_MESSAGE_APPENDED,
            EVENT_SESSION_APPENDED,
            EVENT_SESSION_UPDATED,
            EVENT_SPECS,
            emit_message_appended,
            emit_session_appended,
            emit_session_updated,
            query_events_since,
        )

        emit_session_appended(source_name="synthetic", succeeded_file_count=1, session_id="synthetic:s1")
        emit_session_updated(session_id="synthetic:s1", source_name="synthetic", appended_count=2)
        emit_message_appended(session_id="synthetic:s1", source_name="synthetic", appended_count=3)

        by_kind = {event["kind"]: event for event in query_events_since(None).events}
        assert set(by_kind) == {EVENT_SESSION_APPENDED, EVENT_SESSION_UPDATED, EVENT_MESSAGE_APPENDED}
        for kind, event in by_kind.items():
            spec = EVENT_SPECS[cast("str", kind)]
            payload = cast("dict[str, object]", event["payload"])
            undeclared = set(payload) - set(spec.payload_projection)
            assert not undeclared, f"{kind}: emitted undeclared payload keys {sorted(undeclared)}"
            missing = set(spec.required_payload_fields) - set(payload)
            assert not missing, f"{kind}: emitted payload is missing required fields {sorted(missing)}"
            assert payload[spec.object_ref] == "synthetic:s1"

    def test_declared_frame_is_the_frame_written_on_the_wire(self, empty_events_db: Path) -> None:
        """``EventSpec.frame`` describes the real SSE ``event:`` line, not a wish."""
        from polylogue.daemon.events import EVENT_SPECS, emit_session_updated

        emit_session_updated(session_id="synthetic:s1", source_name="synthetic", appended_count=1)
        handler = _make_handler("GET", "/api/events?max_seconds=1")
        handler.do_GET()
        out = cast("BytesIO", handler.wfile).getvalue()
        frame = EVENT_SPECS["session.updated"].frame
        assert f"event: {frame}\n".encode() in out

    def test_spec_rejects_an_emitter_outside_the_polylogue_package(self) -> None:
        from polylogue.daemon.events import EVENT_SPECS

        spec = EVENT_SPECS["session.updated"]
        with pytest.raises(ValueError, match="non-production emitter"):
            dataclasses.replace(spec, emitter="tests.unit.daemon.test_daemon_events_endpoint.fake_emitter")

    def test_spec_rejects_an_object_ref_outside_its_projection(self) -> None:
        from polylogue.daemon.events import EVENT_SPECS

        spec = EVENT_SPECS["session.updated"]
        with pytest.raises(ValueError, match="object_ref"):
            dataclasses.replace(spec, object_ref="not_a_payload_key")


class TestDaemonEventRetention:
    """The ledger keeps what live subscribers have not read, with no row count or age."""

    def test_an_open_stream_holds_its_unread_rows_against_the_owners_prune(self, empty_events_db: Path) -> None:
        """A stream registered as a live subscriber reads every row emitted after it opened.

        Anti-vacuity: without the subscription in ``_stream_events`` the owner's
        next emit prunes the unread frames and the stream sees an aged-out resync.
        """
        from polylogue.daemon.events import EVENT_SUBSCRIBERS, emit_message_appended
        from polylogue.daemon.events import query_events_since as real_query

        emitted: list[int] = []

        def emit_then_query(cursor: str | None, **kwargs: object) -> object:
            if not emitted:
                for index in range(3):
                    emit_message_appended(session_id=f"s-{index}", source_name="codex", appended_count=1)
                emitted.append(1)
            return real_query(cursor, **kwargs)  # type: ignore[arg-type]

        with EVENT_SUBSCRIBERS.owning(), patch("polylogue.daemon.events_http.query_events_since", emit_then_query):
            handler = _make_handler("GET", "/api/events?max_seconds=1")
            handler.do_GET()
        out = cast("BytesIO", handler.wfile).getvalue()

        assert out.count(b"event: message.appended\n") == 3
        assert b"cursor_aged_out" not in out

    def test_a_closed_stream_no_longer_holds_rows(self, empty_events_db: Path) -> None:
        from polylogue.daemon.events import EVENT_SUBSCRIBERS, emit_message_appended, query_events_since

        with EVENT_SUBSCRIBERS.owning():
            subscription = EVENT_SUBSCRIBERS.subscribe(None)
            emit_message_appended(session_id="s-1", source_name="codex", appended_count=1)
            assert len(query_events_since(None).events) == 1
            subscription.close()
            emit_message_appended(session_id="s-2", source_name="codex", appended_count=1)
            assert query_events_since(None).events == ()


class TestAgedOutCursorResync:
    """A pruned replay gap is an explicit resync, never a short or empty answer."""

    def _force_gap(self) -> str:
        """Emit, prune below the client's cursor, and return that stale cursor.

        Uses the production enforcement point to do the pruning, so the fixture
        reproduces the real route rather than a hand-rolled DELETE.
        """
        from polylogue.daemon.events import EVENT_SUBSCRIBERS, emit_daemon_event, query_events_since

        for index in range(4):
            emit_daemon_event("ingestion_batch", payload={"n": index})
        client_cursor = cast("str", query_events_since(None).events[0]["cursor"])
        # The daemon owns the ledger and no subscriber is live, so the next emit
        # removes every superseded batch record the client has not read.
        with EVENT_SUBSCRIBERS.owning():
            emit_daemon_event("ingestion_batch", payload={"n": "after-prune"})
        return client_cursor

    def test_cursor_below_the_retained_minimum_is_refused_with_a_resync(self, empty_events_db: Path) -> None:
        """Anti-vacuity: without the watermark check this returns a short
        page of surviving rows and reports it as a complete ``ok`` answer."""
        from polylogue.daemon.events import EventCursorStatus, query_events_since

        stale_cursor = self._force_gap()
        page = query_events_since(stale_cursor)

        assert page.status is EventCursorStatus.AGED_OUT
        assert page.events == ()
        assert page.resync is not None
        payload = cast("dict[str, object]", page.resync["payload"])
        assert payload["resync"] is True
        assert payload["reason"] == "cursor_aged_out"
        assert payload["requested_since"] == stale_cursor
        assert payload["first_event_id"] == page.retained_min_id
        assert payload["last_event_id"] == page.latest_id
        assert page.retained_min_id is not None and page.retained_min_id > int(stale_cursor.rsplit(":", 1)[1]) + 1

    def test_cursor_inside_the_retained_range_is_served_normally(self, empty_events_db: Path) -> None:
        from polylogue.daemon.events import EventCursorStatus, emit_daemon_event, query_events_since

        emit_daemon_event("ingestion_batch", payload={"n": 0})
        emit_daemon_event("ingestion_batch", payload={"n": 1})
        first_id = cast("str", query_events_since(None).events[0]["cursor"])

        page = query_events_since(first_id)
        assert page.status is EventCursorStatus.OK
        assert page.resync is None
        assert [cast("dict[str, object]", event["payload"])["n"] for event in page.events] == [1]

    def test_a_cursor_ahead_of_a_reset_ledger_is_refused(self, empty_events_db: Path) -> None:
        """A disposable tier replaced under a subscriber is loss, not "no change"."""
        from polylogue.daemon.events import EventCursorStatus, emit_daemon_event, query_events_since

        emit_daemon_event("ingestion_batch", payload={"n": 0})
        page = query_events_since(_cursor(9_999))

        assert page.status is EventCursorStatus.AGED_OUT
        assert cast("dict[str, object]", cast("dict[str, object]", page.resync)["payload"])["reason"] == "ledger_reset"

    def test_missing_ledger_resets_a_resumed_cursor(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        from polylogue.daemon import events

        monkeypatch.setattr(events, "_events_db_path", lambda: tmp_path / "removed-ops.db")
        page = events.query_events_since("a" * 32 + ":42")
        assert page.status is events.EventCursorStatus.AGED_OUT
        assert page.resync is not None
        assert cast(dict[str, object], page.resync["payload"])["reason"] == "ledger_reset"

    def test_empty_anchor_parameter_conflicts_with_explicit_offset(self) -> None:
        """Anti-vacuity: checking anchor truthiness accepts ``around=&offset=500``."""
        handler = _make_handler("GET", "/api/sessions/x/read")
        assert handler._accept_message_window_anchor("", None, 500) is False
        assert b"400" in cast(BytesIO, handler.wfile).getvalue()

    def test_query_parsing_keeps_a_blank_anchor(self) -> None:
        """Anti-vacuity: parse_qs without keep_blank_values drops ``around=`` before the conflict check."""
        handler = _make_handler("GET", "/api/sessions/x/read?view=messages&around=&offset=500")
        _path, params = handler._parse_path()
        assert params["around"] == [""]
        assert handler._get_param(params, "around") is None

    def test_messages_route_refuses_a_blank_anchor_with_an_offset(self) -> None:
        """Anti-vacuity: reading ``around`` through ``_get_param`` turns ``around=`` into None and serves the offset."""
        from polylogue.daemon.route_families.read_detail import _handle_get_messages

        handler = _make_handler("GET", "/api/sessions/x/messages?around=&offset=500")
        _path, params = handler._parse_path()
        _handle_get_messages(handler, "x", params)
        assert b"400" in cast(BytesIO, handler.wfile).getvalue()

    def test_an_aged_out_page_cannot_also_deliver_rows(self) -> None:
        from polylogue.daemon.events import DaemonEventPage, EventCursorStatus

        with pytest.raises(ValueError, match="partial row page"):
            DaemonEventPage(
                status=EventCursorStatus.AGED_OUT,
                events=({"id": 1},),
                retained_min_id=5,
                latest_id=9,
                latest_cursor="a" * 32 + ":9",
                resync={"kind": "snapshot"},
            )

    def test_an_ok_page_cannot_carry_a_resync_envelope(self) -> None:
        from polylogue.daemon.events import DaemonEventPage, EventCursorStatus

        with pytest.raises(ValueError, match="resync envelope"):
            DaemonEventPage(
                status=EventCursorStatus.OK,
                events=(),
                retained_min_id=5,
                latest_id=9,
                latest_cursor="a" * 32 + ":9",
                resync={"kind": "snapshot"},
            )

    def test_poll_reports_an_aged_out_cursor_as_a_degraded_resync(self, empty_events_db: Path) -> None:
        stale_cursor = self._force_gap()

        handler = _make_handler("GET", f"/api/events?poll=1&since={stale_cursor}")
        send_json = capture_json_response(handler)
        handler.do_GET()

        status, payload = send_json.call_args.args
        assert status == HTTPStatus.OK
        assert payload["outcome"] == "degraded"
        assert payload["resync"] is True
        assert payload["resync_reason"] == "cursor_aged_out"
        assert [event["kind"] for event in payload["events"]] == ["snapshot"]

    def test_sse_stream_emits_a_resync_frame_for_an_aged_out_cursor(self, empty_events_db: Path) -> None:
        stale_cursor = self._force_gap()

        handler = _make_handler(
            "GET",
            "/api/events?max_seconds=1",
            extra_headers={"Last-Event-ID": str(stale_cursor)},
        )
        handler.do_GET()
        out = cast("BytesIO", handler.wfile).getvalue()

        assert b"event: snapshot\n" in out
        assert b'"reason": "cursor_aged_out"' in out or b'"reason":"cursor_aged_out"' in out
        assert b"event: ingestion_batch\n" not in out

    def test_resync_and_coalesced_snapshots_share_one_envelope_shape(self, empty_events_db: Path) -> None:
        """One wire shape for "refetch your view", not two signals to learn."""
        from polylogue.daemon.events import build_snapshot_envelope
        from polylogue.daemon.events_http import _build_snapshot_event

        stale_cursor = self._force_gap()
        from polylogue.daemon.events import query_events_since

        resync = cast("dict[str, object]", query_events_since(stale_cursor).resync)
        coalesced = _build_snapshot_event([{"id": 1, "cursor": "a" * 32 + ":1", "ts": "t", "kind": "ingestion_batch"}])

        assert set(coalesced) == set(resync)
        assert set(cast("dict[str, object]", coalesced["payload"])) <= set(cast("dict[str, object]", resync["payload"]))
        assert (
            build_snapshot_envelope(
                event_id=1,
                cursor="a" * 32 + ":1",
                ts="t",
                event_count=1,
                first_event_id=1,
                last_event_id=1,
                kind_counts={"ingestion_batch": 1},
            )
            == coalesced
        )
