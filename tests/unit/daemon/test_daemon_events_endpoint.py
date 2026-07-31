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

import json
import re
import sqlite3
from http import HTTPStatus
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING, cast
from unittest.mock import MagicMock

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
        send_json = capture_json_response(handler)
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
        send_json = capture_json_response(handler)
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
        send_json = capture_json_response(handler)
        handler.do_GET()

        payload = send_json.call_args.args[1]
        kinds = {e["kind"] for e in payload["events"]}
        assert kinds == {"ingestion_batch"}

    def test_poll_since_is_strict_gt(self, empty_events_db: Path) -> None:
        from polylogue.daemon.events import emit_daemon_event

        emit_daemon_event("ingestion_batch", payload={})
        handler = _make_handler("GET", "/api/events?poll=1&since=0")
        send_json = capture_json_response(handler)
        handler.do_GET()
        first_id = send_json.call_args.args[1]["events"][0]["id"]

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
        send_json_first = capture_json_response(handler_first)
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
    """``GET /api/status`` ETags include event and normalized snapshot identity."""

    def test_status_includes_last_event_id_field(self, empty_events_db: Path) -> None:
        from polylogue.daemon.events import emit_daemon_event

        emit_daemon_event("ingestion_batch", payload={})
        handler = _make_handler("GET", "/api/status")
        handler.do_GET()
        out = cast("BytesIO", handler.wfile).getvalue()
        assert b'ETag: W/"status-' in out
        assert b'"last_event_id":' in out

    def test_status_returns_304_when_etag_matches(self, empty_events_db: Path) -> None:
        from polylogue.daemon.events import emit_daemon_event
        from polylogue.daemon.status_snapshot import refresh_status_snapshot

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
        from polylogue.daemon.events import emit_session_appended, query_daemon_events

        emit_session_appended(
            source_name="claude-code-session",
            succeeded_file_count=3,
            failed_file_count=1,
            source_paths=["/tmp/a.jsonl", "/tmp/b.jsonl"],
            session_id="claude-code-session:conv-abc",
        )
        events = query_daemon_events(limit=10)
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
        from polylogue.daemon.events import emit_session_updated, query_daemon_events

        emit_session_updated(
            session_id="codex:conv-xyz",
            source_name="codex",
            appended_count=2,
        )
        events = query_daemon_events(limit=10)
        assert events[0]["kind"] == "session.updated"
        payload = cast("dict[str, object]", events[0]["payload"])
        assert payload["session_id"] == "codex:conv-xyz"
        assert payload["source_name"] == "codex"
        assert payload["appended_count"] == 2

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
            "/api/events?poll=1&since=0&kinds=message.appended,session.updated",
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

    def test_batch_with_new_and_updated_sessions_emits_scoped_events(self, empty_events_db: Path) -> None:
        from polylogue.daemon.cli import _emit_live_batch_event
        from polylogue.daemon.events import query_daemon_events

        _emit_live_batch_event(
            "ingestion_batch",
            {
                "succeeded_file_count": 2,
                "failed_file_count": 0,
                "new_sessions": [{"source_name": "codex", "session_id": "codex:new-1"}],
                "updated_sessions": [{"source_name": "claude-code", "session_id": "claude-code:existing-1"}],
            },
        )
        events = query_daemon_events(limit=10)
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

    def test_batch_touching_only_session_b_never_names_session_a(self, empty_events_db: Path) -> None:
        """The exact regression the bead describes: session A must be unaffected."""
        from polylogue.daemon.cli import _emit_live_batch_event
        from polylogue.daemon.events import query_daemon_events

        _emit_live_batch_event(
            "ingestion_batch",
            {
                "succeeded_file_count": 1,
                "failed_file_count": 0,
                "new_sessions": [],
                "updated_sessions": [{"source_name": "codex", "session_id": "codex:session-b"}],
            },
        )
        events = query_daemon_events(limit=10)
        seen_session_ids = {
            cast("dict[str, object]", event["payload"])["session_id"]
            for event in events
            if event["kind"] in ("session.updated", "message.appended")
        }
        assert seen_session_ids == {"codex:session-b"}
        assert "codex:session-a" not in seen_session_ids

    def test_batch_without_resolved_identity_falls_back_to_unscoped_aggregate(self, empty_events_db: Path) -> None:
        """No source path yet threads identity through -- preserve the old signal."""
        from polylogue.daemon.cli import _emit_live_batch_event
        from polylogue.daemon.events import query_daemon_events

        _emit_live_batch_event(
            "ingestion_batch",
            {"succeeded_file_count": 1, "failed_file_count": 0},
        )
        events = query_daemon_events(limit=10)
        kinds = {cast("str", event["kind"]) for event in events}
        assert kinds == {"ingestion_batch", "session.appended", "message.appended"}
        for event in events:
            if event["kind"] in ("session.appended", "message.appended"):
                assert cast("dict[str, object]", event["payload"])["session_id"] is None

    def test_zero_succeeded_batch_emits_no_granular_events(self, empty_events_db: Path) -> None:
        from polylogue.daemon.cli import _emit_live_batch_event
        from polylogue.daemon.events import query_daemon_events

        _emit_live_batch_event("ingestion_batch", {"succeeded_file_count": 0, "failed_file_count": 3})
        events = query_daemon_events(limit=10)
        assert {cast("str", event["kind"]) for event in events} == {"ingestion_batch"}


class TestBackpressureCoalescing:
    """#1204 — bursts collapse into one ``snapshot`` envelope for slow clients."""

    def test_poll_coalesces_burst_into_snapshot(self, empty_events_db: Path) -> None:
        from polylogue.daemon.events import emit_message_appended

        for _ in range(20):
            emit_message_appended(session_id="c", appended_count=1)

        handler = _make_handler("GET", "/api/events?poll=1&since=0&coalesce=5")
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
        assert payload["last_event_id"] == snapshot["id"]

    def test_poll_below_threshold_returns_individual_events(self, empty_events_db: Path) -> None:
        from polylogue.daemon.events import emit_message_appended

        for _ in range(3):
            emit_message_appended(session_id="c", appended_count=1)

        handler = _make_handler("GET", "/api/events?poll=1&since=0&coalesce=10")
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
        handler = _make_handler(
            "GET",
            f"/api/events?poll=1&since=0&{credential_param}=secret",
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
