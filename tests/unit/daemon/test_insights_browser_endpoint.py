"""Insights browser endpoint contracts for the reader (#1120).

``GET /api/insights/sessions/{id}[?include=profile,timeline,phases,threads]``
returns a single typed envelope joining four per-session insight kinds:

- session profile (#1018)
- work-event timeline (#1133/#1135)
- session phases
- thread membership

Each kind carries a canonical terminal outcome plus a readiness chip from
the closed vocabulary (``q-error`` / ``q-missing`` / ``q-partial`` /
``q-ready``). Unknown sessions are a hard 404. Existing sessions without any
materialized insight return 200 with explicit ``q-missing`` shapes per kind
(panel never blank — AC#1120), and a kind whose insight surface could not
answer returns ``q-error`` so it is never read as zero rows.

These tests exercise the pure helpers (``_readiness_tag``,
``_parse_insight_includes``, panel projectors) and the end-to-end dispatch
path through the in-process handler harness (same shape as
``test_cost_panel_endpoint.py``).
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from email.message import Message
from http import HTTPStatus
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
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
from polylogue.surfaces.outcome import decide_outcome

_OK = decide_outcome(matched=1)
_EMPTY = decide_outcome(matched=0)
_FAILED = decide_outcome(matched=0, error="insight_unavailable:probe")

# ---------------------------------------------------------------------------
# In-process handler harness (mirrors test_cost_panel_endpoint.py)
# ---------------------------------------------------------------------------


class _MockServer:
    auth_token = ""
    api_host = "127.0.0.1"
    archive_query_executor = ThreadPoolExecutor(max_workers=1)


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


def _seed_minimum_archive(workspace_env: dict[str, Path]) -> str:
    """Seed the archive with one session + one message.

    Returns the archive session id (``origin:native_id``) the insights
    endpoint resolves through ``poly.get_session``.
    """
    from tests.infra.storage_records import SessionBuilder, db_setup

    builder = (
        SessionBuilder(db_setup(workspace_env), "ins-1")
        .provider("claude-code")
        .title("An ins conv")
        .add_message(message_id="m-ins-1", role="user", text="hello")
    )
    builder.save()
    return builder.native_session_id()


# ---------------------------------------------------------------------------
# Pure helpers: readiness chip vocabulary
# ---------------------------------------------------------------------------


class TestReadinessTagMapping:
    """``_readiness_tag`` maps a panel's outcome and row count to the chips."""

    def test_unmaterialized_is_missing(self) -> None:
        assert _readiness_tag(_EMPTY, materialized=False) == "q-missing"

    def test_unmaterialized_with_zero_rows_is_missing(self) -> None:
        # The materialized flag wins — a missing surface cannot be partial.
        assert _readiness_tag(_EMPTY, materialized=False, row_count=0) == "q-missing"

    def test_materialized_with_zero_rows_is_partial(self) -> None:
        # Rebuild ran but produced nothing — explicit "partial" surface.
        assert _readiness_tag(_EMPTY, materialized=True, row_count=0) == "q-partial"

    def test_materialized_with_rows_is_ready(self) -> None:
        assert _readiness_tag(_OK, materialized=True, row_count=3) == "q-ready"

    def test_materialized_without_row_count_is_ready(self) -> None:
        # row_count=None (e.g. session profile) defaults to ready.
        assert _readiness_tag(_OK, materialized=True) == "q-ready"

    def test_failed_surface_is_q_error_not_missing(self) -> None:
        # An insight surface that could not answer must not be reported with
        # the same chip as a session that genuinely has no rows.
        assert _readiness_tag(_FAILED, materialized=False) == "q-error"
        assert _readiness_tag(_FAILED, materialized=False, row_count=0) == "q-error"


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
        payload = _empty_profile_panel_payload(_EMPTY)
        assert payload["readiness_tag"] == "q-missing"
        assert payload["materialized"] is False
        assert payload["profile"] is None
        assert payload["provenance"] is None

    def test_empty_work_event_panel_is_q_missing(self) -> None:
        payload = _work_event_panel_payload([], _EMPTY)
        assert payload["readiness_tag"] == "q-missing"
        assert payload["materialized"] is False
        assert payload["count"] == 0
        assert payload["events"] == []

    def test_empty_phase_panel_is_q_missing(self) -> None:
        payload = _phase_panel_payload([], _EMPTY)
        assert payload["readiness_tag"] == "q-missing"
        assert payload["count"] == 0
        assert payload["phases"] == []

    def test_empty_thread_panel_is_q_missing(self) -> None:
        payload = _thread_panel_payload([], _EMPTY)
        assert payload["readiness_tag"] == "q-missing"
        assert payload["count"] == 0
        assert payload["threads"] == []

    def test_payloads_round_trip_through_json(self) -> None:
        for payload in (
            _empty_profile_panel_payload(_EMPTY),
            _work_event_panel_payload([], _EMPTY),
            _phase_panel_payload([], _EMPTY),
            _thread_panel_payload([], _EMPTY),
        ):
            json.dumps(payload)

    def test_work_event_payload_uses_origin(self) -> None:
        class _Dump:
            def model_dump(self, *, mode: str) -> dict[str, object]:
                assert mode == "json"
                return {}

        payload = _work_event_panel_payload(
            [
                SimpleNamespace(
                    event_id="event-1",
                    event_index=0,
                    session_id="claude-code-session:ins-1",
                    origin="claude-code-session",
                    evidence=_Dump(),
                    inference=_Dump(),
                    provenance=SimpleNamespace(materializer_version=1),
                )
            ],
            _OK,
        )
        event = cast(list[dict[str, object]], payload["events"])[0]
        assert event["origin"] == "claude-code-session"
        assert "provider" not in event

    def test_phase_payload_uses_origin(self) -> None:
        class _Dump:
            def model_dump(self, *, mode: str) -> dict[str, object]:
                assert mode == "json"
                return {}

        payload = _phase_panel_payload(
            [
                SimpleNamespace(
                    phase_id="phase-1",
                    phase_index=0,
                    session_id="codex-session:ins-1",
                    origin="codex-session",
                    evidence=_Dump(),
                    inference=_Dump(),
                    provenance=SimpleNamespace(materializer_version=1),
                )
            ],
            _OK,
        )
        phase = cast(list[dict[str, object]], payload["phases"])[0]
        assert phase["origin"] == "codex-session"
        assert "provider" not in phase


# ---------------------------------------------------------------------------
# End-to-end endpoint dispatch
# ---------------------------------------------------------------------------


class TestInsightsEndpointDispatch:
    """``GET /api/insights/sessions/{id}`` routes to the insights handler."""

    def test_unknown_session_returns_404(self, workspace_env: dict[str, Path]) -> None:
        _seed_minimum_archive(workspace_env)
        handler = _make_handler("GET", "/api/insights/sessions/does-not-exist")
        send_error, send_json = _capture_responses(handler)
        handler.do_GET()
        send_json.assert_not_called()
        send_error.assert_called_once()
        status, code = send_error.call_args.args
        assert status == HTTPStatus.NOT_FOUND
        assert code == "not_found"

    def test_known_session_returns_typed_envelope(self, workspace_env: dict[str, Path]) -> None:
        session_id = _seed_minimum_archive(workspace_env)
        handler = _make_handler("GET", f"/api/insights/sessions/{session_id}")
        send_error, send_json = _capture_responses(handler)
        handler.do_GET()
        send_error.assert_not_called()
        send_json.assert_called_once()
        status, payload = send_json.call_args.args
        assert status == HTTPStatus.OK
        assert payload["session_id"] == session_id
        assert payload["include"] == list(INSIGHT_KINDS)
        kinds = payload["kinds"]
        assert set(kinds.keys()) == set(INSIGHT_KINDS)
        # Every kind must report an explicit readiness chip from the closed
        # vocabulary rather than being absent (panel never blank, AC#1120).
        # Native ingest materializes some derived read models (e.g. work
        # threads) at write time, so the precise chip per kind reflects what
        # the archive store produced; the contract is that none is missing
        # from the envelope.
        for kind in INSIGHT_KINDS:
            assert kinds[kind]["readiness_tag"] in {"q-ready", "q-partial", "q-missing"}
            assert "materialized" in kinds[kind]
            assert kinds[kind]["outcome"]["state"] in {"ok", "empty"}
        assert payload["outcome"]["state"] in {"ok", "empty"}

    def test_include_param_restricts_kinds(self, workspace_env: dict[str, Path]) -> None:
        session_id = _seed_minimum_archive(workspace_env)
        handler = _make_handler("GET", f"/api/insights/sessions/{session_id}?include=profile,phases")
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
        session_id = _seed_minimum_archive(workspace_env)
        handler = _make_handler("GET", f"/api/insights/sessions/{session_id}?include=profile,bogus,timeline")
        _, send_json = _capture_responses(handler)
        handler.do_GET()
        status, payload = send_json.call_args.args
        assert status == HTTPStatus.OK
        assert set(payload["kinds"].keys()) == {"profile", "timeline"}

    def test_envelope_carries_origin_and_id(self, workspace_env: dict[str, Path]) -> None:
        session_id = _seed_minimum_archive(workspace_env)
        handler = _make_handler("GET", f"/api/insights/sessions/{session_id}")
        _, send_json = _capture_responses(handler)
        handler.do_GET()
        _, payload = send_json.call_args.args
        assert payload["session_id"] == session_id
        assert payload["origin"]

    def test_envelope_is_json_serialisable(self, workspace_env: dict[str, Path]) -> None:
        session_id = _seed_minimum_archive(workspace_env)
        handler = _make_handler("GET", f"/api/insights/sessions/{session_id}")
        _, send_json = _capture_responses(handler)
        handler.do_GET()
        _, payload = send_json.call_args.args
        # All panel data ships over HTTP as JSON.
        json.dumps(payload)


class TestUnavailableInsightSurface:
    """An insight surface that cannot answer is never rendered as zero rows.

    Anti-vacuity: restoring the pre-fix handler — swallowing
    ``ArchiveInsightUnavailableError`` into an empty list and deriving
    readiness from ``bool(rows)`` — makes every assertion below fail, because
    the failed panel would then be byte-identical to the genuinely empty one.
    """

    @staticmethod
    def _panels(workspace_env: dict[str, Path], *, fail: bool) -> dict[str, object]:
        import pytest as _pytest

        from polylogue.analysis.archive import ArchiveInsightUnavailableError

        session_id = _seed_minimum_archive(workspace_env)
        handler = _make_handler("GET", f"/api/insights/sessions/{session_id}?include=timeline,phases")
        _, send_json = _capture_responses(handler)
        if fail:
            from polylogue.api import Polylogue

            monkeypatch = _pytest.MonkeyPatch()

            async def _unavailable(*_args: object, **_kwargs: object) -> object:
                raise ArchiveInsightUnavailableError("work-event insight surface is unavailable")

            try:
                monkeypatch.setattr(Polylogue, "list_session_work_event_insights", _unavailable, raising=True)
                handler.do_GET()
            finally:
                monkeypatch.undo()
        else:
            handler.do_GET()
        _, payload = send_json.call_args.args
        return cast(dict[str, object], payload)

    def test_failed_panel_is_distinguishable_from_empty(self, workspace_env: dict[str, Path]) -> None:
        failed = self._panels(workspace_env, fail=True)
        failed_timeline = cast(dict[str, object], cast(dict[str, object], failed["kinds"])["timeline"])
        assert failed_timeline["outcome"] == {
            "state": "error",
            "reason": "insight_unavailable:timeline",
            "detail": {},
        }
        assert failed_timeline["readiness_tag"] == "q-error"
        assert failed_timeline["count"] == 0

    def test_empty_panel_stays_empty(self, workspace_env: dict[str, Path]) -> None:
        healthy = self._panels(workspace_env, fail=False)
        healthy_timeline = cast(dict[str, object], cast(dict[str, object], healthy["kinds"])["timeline"])
        assert cast(dict[str, object], healthy_timeline["outcome"])["state"] == "empty"
        assert healthy_timeline["readiness_tag"] == "q-missing"
        assert healthy_timeline["count"] == 0

    def test_envelope_outcome_is_degraded_when_one_kind_fails(self, workspace_env: dict[str, Path]) -> None:
        failed = self._panels(workspace_env, fail=True)
        envelope_outcome = cast(dict[str, object], failed["outcome"])
        assert envelope_outcome["state"] == "degraded"
        assert envelope_outcome["reason"] == "insight_unavailable:timeline"
        # The unaffected kind still answered, which is what "degraded" means.
        phases = cast(dict[str, object], cast(dict[str, object], failed["kinds"])["phases"])
        assert cast(dict[str, object], phases["outcome"])["state"] in {"ok", "empty"}

    def test_failure_is_logged(self, workspace_env: dict[str, Path], caplog: object) -> None:
        import logging

        import pytest as _pytest

        typed = cast(_pytest.LogCaptureFixture, caplog)
        with typed.at_level(logging.WARNING, logger="polylogue.daemon.http"):
            self._panels(workspace_env, fail=True)
        assert any("timeline" in record.getMessage() for record in typed.records)
