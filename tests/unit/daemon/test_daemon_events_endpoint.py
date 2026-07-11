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

import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor
from email.message import Message
from http import HTTPStatus
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING, cast
from unittest.mock import MagicMock

import pytest

from polylogue.daemon.web_auth import WebCredentialRegistry

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
        assert not empty_events_db.exists()

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

        assert events_mod.query_daemon_events() == []
        assert events_mod.query_events_since(0) == []
        assert events_mod.get_latest_event_id() == 0
        assert events_mod.get_daemon_event_counts() == {}
        assert events_mod.get_last_ingestion_batch() is None
        assert events_mod.get_recent_operations() == []
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

        assert events_mod.query_daemon_events() == []
        assert events_mod.query_events_since(0) == []
        assert events_mod.get_latest_event_id() == 0
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

            assert [event["kind"] for event in events_mod.query_daemon_events()] == ["committed"]
            assert [event["kind"] for event in events_mod.query_events_since(0)] == ["committed"]
            assert events_mod.get_latest_event_id() == 1
            assert events_mod.get_daemon_event_counts() == {"committed": 1}
            assert writer.in_transaction is True
            assert writer.total_changes == writer_changes
            writer.execute(
                "INSERT INTO daemon_events (ts_ms, kind, operation_id, payload_json) VALUES (3, 'writer-still-active', NULL, '{}')"
            )
            writer.commit()
        finally:
            writer.close()

        assert [event["kind"] for event in events_mod.query_events_since(1)] == [
            "uncommitted",
            "writer-still-active",
        ]


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


class TestGranularEventKinds:
    """#1204 — granular SSE topics for selective subscription and live tail."""

    def test_emit_session_appended_payload_shape(self, empty_events_db: Path) -> None:
        from polylogue.daemon.events import emit_session_appended, query_daemon_events

        emit_session_appended(
            source_name="claude-code-session",
            succeeded_file_count=3,
            failed_file_count=1,
            source_paths=["/tmp/a.jsonl", "/tmp/b.jsonl"],
        )
        events = query_daemon_events(limit=10)
        assert events[0]["kind"] == "session.appended"
        payload = cast("dict[str, object]", events[0]["payload"])
        assert payload["source_name"] == "claude-code-session"
        assert payload["succeeded_file_count"] == 3
        assert payload["failed_file_count"] == 1
        assert payload["source_paths"] == ["/tmp/a.jsonl", "/tmp/b.jsonl"]

    def test_emit_message_appended_payload_shape(self, empty_events_db: Path) -> None:
        from polylogue.daemon.events import emit_message_appended, query_daemon_events

        emit_message_appended(
            session_id="conv-abc",
            source_name="codex",
            appended_count=4,
            source_path="/tmp/session.json",
        )
        events = query_daemon_events(limit=10)
        assert events[0]["kind"] == "message.appended"
        payload = cast("dict[str, object]", events[0]["payload"])
        assert payload["session_id"] == "conv-abc"
        assert payload["appended_count"] == 4
        assert payload["source_path"] == "/tmp/session.json"

    def test_emit_progress_update_includes_fraction(self, empty_events_db: Path) -> None:
        from polylogue.daemon.events import emit_progress_update, query_daemon_events

        emit_progress_update(
            operation_id="replay-42",
            operation_kind="replay",
            completed=47,
            total=100,
            eta_seconds=180.0,
            detail="reapplying schema validation",
        )
        events = query_daemon_events(limit=10)
        assert events[0]["kind"] == "progress.update"
        assert events[0]["operation_id"] == "replay-42"
        payload = cast("dict[str, object]", events[0]["payload"])
        assert payload["completed"] == 47
        assert payload["total"] == 100
        assert payload["fraction"] == 0.47
        assert payload["eta_seconds"] == 180.0

    def test_emit_progress_complete(self, empty_events_db: Path) -> None:
        from polylogue.daemon.events import emit_progress_complete, query_daemon_events

        emit_progress_complete(operation_id="replay-42", operation_kind="replay", status="completed")
        events = query_daemon_events(limit=10)
        assert events[0]["kind"] == "progress.complete"
        assert events[0]["operation_id"] == "replay-42"
        assert cast("dict[str, object]", events[0]["payload"])["status"] == "completed"

    def test_selective_subscription_via_kinds(self, empty_events_db: Path) -> None:
        from polylogue.daemon.events import (
            emit_message_appended,
            emit_progress_update,
            emit_session_appended,
        )

        emit_session_appended(source_name=None, succeeded_file_count=1)
        emit_message_appended(session_id="c", appended_count=1)
        emit_progress_update(operation_id="op", operation_kind="x", completed=1)

        handler = _make_handler(
            "GET",
            "/api/events?poll=1&since=0&kinds=message.appended,progress.update",
        )
        send_json = _capture_json(handler)
        handler.do_GET()
        kinds = {e["kind"] for e in send_json.call_args.args[1]["events"]}
        assert kinds == {"message.appended", "progress.update"}


class TestBackpressureCoalescing:
    """#1204 — bursts collapse into one ``snapshot`` envelope for slow clients."""

    def test_poll_coalesces_burst_into_snapshot(self, empty_events_db: Path) -> None:
        from polylogue.daemon.events import emit_message_appended

        for _ in range(20):
            emit_message_appended(session_id="c", appended_count=1)

        handler = _make_handler("GET", "/api/events?poll=1&since=0&coalesce=5")
        send_json = _capture_json(handler)
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
        assert payload["last_event_id"] == snapshot["id"]

    def test_poll_below_threshold_returns_individual_events(self, empty_events_db: Path) -> None:
        from polylogue.daemon.events import emit_message_appended

        for _ in range(3):
            emit_message_appended(session_id="c", appended_count=1)

        handler = _make_handler("GET", "/api/events?poll=1&since=0&coalesce=10")
        send_json = _capture_json(handler)
        handler.do_GET()
        payload = send_json.call_args.args[1]
        assert payload.get("coalesced") is None
        assert len(payload["events"]) == 3
        assert all(e["kind"] == "message.appended" for e in payload["events"])

    def test_sse_stream_coalesces_burst(self, empty_events_db: Path) -> None:
        from polylogue.daemon.events import emit_message_appended

        for _ in range(15):
            emit_message_appended(session_id="c", appended_count=1)

        handler = _make_handler("GET", "/api/events?since=0&max_seconds=1&coalesce=5")
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
        handler = _make_handler("GET", f"/api/events?poll=1&since=0&{credential_param}=secret")
        handler.server = cast(
            "DaemonAPIHTTPServer",
            type(
                "_Srv",
                (),
                {
                    "auth_token": "secret",
                    "api_host": "127.0.0.1",
                    "web_credentials": WebCredentialRegistry(),
                },
            )(),
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
        handler = _make_handler("GET", "/api/status?access_token=secret")
        handler.server = cast(
            "DaemonAPIHTTPServer",
            type(
                "_Srv",
                (),
                {"auth_token": "secret", "api_host": "127.0.0.1"},
            )(),
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
        )
        handler.server = cast(
            "DaemonAPIHTTPServer",
            type(
                "_Srv",
                (),
                {"auth_token": "secret", "api_host": "127.0.0.1", "web_credentials": registry},
            )(),
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
