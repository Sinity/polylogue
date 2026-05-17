"""Insights browser endpoint contracts for the reader (#1120).

``GET /api/insights/sessions/{id}[?include=profile,timeline,phases,threads]``
returns a single typed envelope joining four per-session insight kinds:

- session profile (#1018)
- work-event timeline (#1133/#1135)
- session phases
- work-thread membership

Each kind carries a readiness chip from the closed vocabulary
(``q-ready`` / ``q-partial`` / ``q-missing``). Unknown conversations are a
hard 404. Existing conversations without any materialized insight return
200 with explicit ``q-missing`` shapes per kind (panel never blank —
AC#1120).

These tests exercise the pure helpers (``_readiness_tag``,
``_parse_insight_includes``, panel projectors) and the end-to-end dispatch
path through the in-process handler harness (same shape as
``test_cost_panel_endpoint.py``).
"""

from __future__ import annotations

import json
import sqlite3
from email.message import Message
from http import HTTPStatus
from io import BytesIO
from pathlib import Path
from typing import cast
from unittest.mock import MagicMock

from polylogue.daemon.http import (
    INSIGHT_KINDS,
    DaemonAPIHandler,
    DaemonAPIHTTPServer,
    _empty_profile_panel_payload,
    _parse_insight_includes,
    _phase_panel_payload,
    _readiness_tag,
    _thread_panel_payload,
    _work_event_panel_payload,
)

# ---------------------------------------------------------------------------
# In-process handler harness (mirrors test_cost_panel_endpoint.py)
# ---------------------------------------------------------------------------


class _MockServer:
    auth_token = ""
    api_host = "127.0.0.1"


class _MockHeaders:
    def __init__(self, headers: dict[str, str] | None = None) -> None:
        self._headers = headers or {}

    def get(self, key: str, default: str | None = None) -> str | None:
        return self._headers.get(key, default)


def _make_handler(method: str, path: str) -> DaemonAPIHandler:
    handler = DaemonAPIHandler.__new__(DaemonAPIHandler)
    handler.server = cast(DaemonAPIHTTPServer, _MockServer())
    handler.client_address = ("127.0.0.1", 12345)
    handler.path = path
    handler.command = method
    handler.requestline = f"{method} {path} HTTP/1.1"
    headers: dict[str, str] = {"Content-Length": "0"}
    handler.headers = cast(Message, _MockHeaders(headers))
    handler.rfile = BytesIO(b"")
    handler.wfile = BytesIO()
    return handler


def _capture_responses(handler: DaemonAPIHandler) -> tuple[MagicMock, MagicMock]:
    send_error = MagicMock()
    send_json = MagicMock()
    handler._send_error = send_error  # type: ignore[method-assign]
    handler._send_json = send_json  # type: ignore[method-assign]
    return send_error, send_json


def _seed_minimum_archive(workspace_env: dict[str, Path]) -> None:
    """Seed the minimum DB schema with one conversation + one message."""
    from polylogue.paths import db_path
    from polylogue.storage.sqlite.schema_ddl_archive import (
        ARCHIVE_STORAGE_DDL,
        MESSAGE_FTS_DDL,
        RECALL_PACKS_DDL,
        SAVED_VIEWS_DDL,
        USER_ANNOTATIONS_DDL,
        USER_MARKS_DDL,
    )

    dbp = db_path()
    dbp.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(dbp))
    try:
        conn.executescript(ARCHIVE_STORAGE_DDL)
        conn.executescript(MESSAGE_FTS_DDL)
        conn.executescript(USER_MARKS_DDL)
        conn.executescript(USER_ANNOTATIONS_DDL)
        conn.executescript(SAVED_VIEWS_DDL)
        conn.executescript(RECALL_PACKS_DDL)
        conn.execute(
            "INSERT INTO conversations(conversation_id, provider_name, provider_conversation_id,"
            " title, content_hash, version) VALUES(?,?,?,?,?,?)",
            ("ins-1", "claude-code", "p-ins-1", "An ins conv", "hash-ins-1", 1),
        )
        conn.execute(
            "INSERT INTO messages(message_id, conversation_id, role, text, provider_name,"
            " content_hash, version) VALUES(?,?,?,?,?,?,?)",
            ("m-ins-1", "ins-1", "user", "hello", "claude-code", "mhash-ins-1", 1),
        )
        conn.commit()
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Pure helpers: readiness chip vocabulary
# ---------------------------------------------------------------------------


class TestReadinessTagMapping:
    """``_readiness_tag`` maps materialized/row-count to the readiness chips."""

    def test_unmaterialized_is_missing(self) -> None:
        assert _readiness_tag(materialized=False) == "q-missing"

    def test_unmaterialized_with_zero_rows_is_missing(self) -> None:
        # The materialized flag wins — a missing surface cannot be partial.
        assert _readiness_tag(materialized=False, row_count=0) == "q-missing"

    def test_materialized_with_zero_rows_is_partial(self) -> None:
        # Rebuild ran but produced nothing — explicit "partial" surface.
        assert _readiness_tag(materialized=True, row_count=0) == "q-partial"

    def test_materialized_with_rows_is_ready(self) -> None:
        assert _readiness_tag(materialized=True, row_count=3) == "q-ready"

    def test_materialized_without_row_count_is_ready(self) -> None:
        # row_count=None (e.g. session profile) defaults to ready.
        assert _readiness_tag(materialized=True) == "q-ready"


# ---------------------------------------------------------------------------
# Pure helpers: include= parser
# ---------------------------------------------------------------------------


class TestParseIncludes:
    """``_parse_insight_includes`` resolves include= into a canonical tuple."""

    def test_none_defaults_to_all_kinds(self) -> None:
        assert _parse_insight_includes(None) == INSIGHT_KINDS

    def test_empty_string_defaults_to_all_kinds(self) -> None:
        assert _parse_insight_includes("") == INSIGHT_KINDS
        assert _parse_insight_includes("   ") == INSIGHT_KINDS

    def test_subset_preserves_canonical_order(self) -> None:
        # Caller passes threads,profile but the canonical order is
        # profile,timeline,phases,threads — we must normalize.
        assert _parse_insight_includes("threads,profile") == ("profile", "threads")

    def test_unknown_tokens_are_dropped(self) -> None:
        assert _parse_insight_includes("profile,not-a-kind,phases") == ("profile", "phases")

    def test_whitespace_and_case_tolerant(self) -> None:
        assert _parse_insight_includes("  PROFILE , Timeline  ") == ("profile", "timeline")


# ---------------------------------------------------------------------------
# Pure helpers: empty/missing panel shapes
# ---------------------------------------------------------------------------


class TestEmptyPayloads:
    """Empty-state panels must surface explicit q-missing — never blank."""

    def test_empty_profile_panel_is_q_missing(self) -> None:
        payload = _empty_profile_panel_payload()
        assert payload["readiness_tag"] == "q-missing"
        assert payload["materialized"] is False
        assert payload["profile"] is None

    def test_empty_work_event_panel_is_q_missing(self) -> None:
        payload = _work_event_panel_payload([])
        assert payload["readiness_tag"] == "q-missing"
        assert payload["materialized"] is False
        assert payload["count"] == 0
        assert payload["events"] == []

    def test_empty_phase_panel_is_q_missing(self) -> None:
        payload = _phase_panel_payload([])
        assert payload["readiness_tag"] == "q-missing"
        assert payload["count"] == 0
        assert payload["phases"] == []

    def test_empty_thread_panel_is_q_missing(self) -> None:
        payload = _thread_panel_payload([])
        assert payload["readiness_tag"] == "q-missing"
        assert payload["count"] == 0
        assert payload["threads"] == []

    def test_payloads_round_trip_through_json(self) -> None:
        for payload in (
            _empty_profile_panel_payload(),
            _work_event_panel_payload([]),
            _phase_panel_payload([]),
            _thread_panel_payload([]),
        ):
            json.dumps(payload)


# ---------------------------------------------------------------------------
# End-to-end endpoint dispatch
# ---------------------------------------------------------------------------


class TestInsightsEndpointDispatch:
    """``GET /api/insights/sessions/{id}`` routes to the insights handler."""

    def test_unknown_conversation_returns_404(self, workspace_env: dict[str, Path]) -> None:
        _seed_minimum_archive(workspace_env)
        handler = _make_handler("GET", "/api/insights/sessions/does-not-exist")
        send_error, send_json = _capture_responses(handler)
        handler.do_GET()
        send_json.assert_not_called()
        send_error.assert_called_once()
        status, code = send_error.call_args.args
        assert status == HTTPStatus.NOT_FOUND
        assert code == "not_found"

    def test_known_conversation_returns_typed_envelope(self, workspace_env: dict[str, Path]) -> None:
        _seed_minimum_archive(workspace_env)
        handler = _make_handler("GET", "/api/insights/sessions/ins-1")
        send_error, send_json = _capture_responses(handler)
        handler.do_GET()
        send_error.assert_not_called()
        send_json.assert_called_once()
        status, payload = send_json.call_args.args
        assert status == HTTPStatus.OK
        assert payload["conversation_id"] == "ins-1"
        assert payload["include"] == list(INSIGHT_KINDS)
        kinds = payload["kinds"]
        assert set(kinds.keys()) == set(INSIGHT_KINDS)
        # No insights are materialized for the seed; every kind must report
        # an explicit q-missing readiness chip rather than being absent.
        for kind in INSIGHT_KINDS:
            assert kinds[kind]["readiness_tag"] in {"q-ready", "q-partial", "q-missing"}
            assert "materialized" in kinds[kind]
        assert kinds["profile"]["readiness_tag"] == "q-missing"
        assert kinds["timeline"]["readiness_tag"] == "q-missing"
        assert kinds["phases"]["readiness_tag"] == "q-missing"
        assert kinds["threads"]["readiness_tag"] == "q-missing"

    def test_include_param_restricts_kinds(self, workspace_env: dict[str, Path]) -> None:
        _seed_minimum_archive(workspace_env)
        handler = _make_handler("GET", "/api/insights/sessions/ins-1?include=profile,phases")
        send_error, send_json = _capture_responses(handler)
        handler.do_GET()
        send_error.assert_not_called()
        send_json.assert_called_once()
        status, payload = send_json.call_args.args
        assert status == HTTPStatus.OK
        assert payload["include"] == ["profile", "phases"]
        # Only the requested kinds appear — restriction must be honored.
        assert set(payload["kinds"].keys()) == {"profile", "phases"}

    def test_include_param_with_unknown_tokens_drops_them(self, workspace_env: dict[str, Path]) -> None:
        _seed_minimum_archive(workspace_env)
        handler = _make_handler("GET", "/api/insights/sessions/ins-1?include=profile,bogus,timeline")
        _, send_json = _capture_responses(handler)
        handler.do_GET()
        status, payload = send_json.call_args.args
        assert status == HTTPStatus.OK
        assert set(payload["kinds"].keys()) == {"profile", "timeline"}

    def test_envelope_carries_provider_and_id(self, workspace_env: dict[str, Path]) -> None:
        _seed_minimum_archive(workspace_env)
        handler = _make_handler("GET", "/api/insights/sessions/ins-1")
        _, send_json = _capture_responses(handler)
        handler.do_GET()
        _, payload = send_json.call_args.args
        assert payload["conversation_id"] == "ins-1"
        # provider may be string-coerced from the Provider enum.
        assert payload["provider"]

    def test_envelope_is_json_serialisable(self, workspace_env: dict[str, Path]) -> None:
        _seed_minimum_archive(workspace_env)
        handler = _make_handler("GET", "/api/insights/sessions/ins-1")
        _, send_json = _capture_responses(handler)
        handler.do_GET()
        _, payload = send_json.call_args.args
        # All panel data ships over HTTP as JSON.
        json.dumps(payload)
