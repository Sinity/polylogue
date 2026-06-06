"""Per-session embedding similarity endpoint contracts (#1123).

The similarity read surface returns ranked similar sessions
through the embedding pipeline established by #828. The pipeline is
dormant by default, so the endpoint's primary job is to render that
state explicitly — "embeddings disabled", "embedding runtime
unavailable", "this session not yet embedded" — rather than
collapsing all of those into an empty success.

Tests use the in-process handler pattern from
``tests/unit/daemon/test_provenance_endpoint.py``: no real daemon, no
socket listener, just the route dispatch against a freshly seeded
SQLite archive. ``sqlite-vec``'s ``MATCH`` engine is not exercised in
unit tests (the extension may not be available in the verify
environment); the contract tests instead pin the route, the absent
states, and the helper-level scoring/aggregation logic.
"""

from __future__ import annotations

import sqlite3
from email.message import Message
from http import HTTPStatus
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING, cast
from unittest.mock import MagicMock

import pytest

from polylogue.daemon.similarity import (
    SIMILAR_RESULTS_DEFAULT,
    SIMILAR_RESULTS_MAX,
    _aggregate_hits,
    _clamp_limit,
    _confidence_for_score,
    _disabled_reason,
    _l2_to_cosine_similarity,
    build_similar_payload,
)
from polylogue.paths import active_index_db_path

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


def _make_handler(method: str, path: str, *, body: bytes = b"") -> DaemonAPIHandler:
    from polylogue.daemon.http import DaemonAPIHandler

    handler = DaemonAPIHandler.__new__(DaemonAPIHandler)
    handler.server = cast("DaemonAPIHTTPServer", _MockServer())
    handler.client_address = ("127.0.0.1", 12345)
    handler.path = path
    handler.command = method
    handler.requestline = f"{method} {path} HTTP/1.1"
    headers: dict[str, str] = {"Content-Length": str(len(body))}
    handler.headers = cast("Message[str, str]", _MockHeaders(headers))
    handler.rfile = BytesIO(body)
    handler.wfile = BytesIO()
    return handler


def _capture_responses(handler: DaemonAPIHandler) -> tuple[MagicMock, MagicMock]:
    send_error = MagicMock()
    send_json = MagicMock()
    handler._send_error = send_error  # type: ignore[method-assign]
    handler._send_json = send_json  # type: ignore[method-assign]
    return send_error, send_json


def _index_db() -> Path:
    return active_index_db_path()


def _seed_archive_session(
    session_id: str,
    *,
    origin: str = "claude-code",
    title: str = "stub",
) -> None:
    """Seed an archive `sessions` row in index.db.

    The similarity reader (``polylogue/daemon/similarity.py``) routes to
    the archive path whenever ``index.db`` exists; it only needs the
    ``sessions`` row to confirm the session exists before rendering
    the disabled/unavailable envelope.
    """
    archive_db = _index_db()
    archive_db.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(archive_db) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                title TEXT,
                origin TEXT NOT NULL
            );
            """
        )
        conn.execute(
            "INSERT OR IGNORE INTO sessions (session_id, title, origin) VALUES (?, ?, ?)",
            (session_id, title, origin),
        )
        conn.commit()


def _init_archive() -> None:
    """Create an empty index.db with the ``sessions`` table only.

    Used by the missing-session cases so the reader routes to the
    archive path and returns ``None`` (404) for an unknown id.
    """
    archive_db = _index_db()
    archive_db.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(archive_db) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                title TEXT,
                origin TEXT NOT NULL
            );
            """
        )
        conn.commit()


def _disable_embeddings(monkeypatch: pytest.MonkeyPatch) -> None:
    """Force ``load_polylogue_config`` to return an embeddings-off config."""

    import polylogue.daemon.similarity as similarity_mod

    class _Cfg:
        embedding_enabled = False
        voyage_api_key = None

    monkeypatch.setattr(similarity_mod, "load_polylogue_config", lambda: _Cfg())


def _enable_embeddings(monkeypatch: pytest.MonkeyPatch) -> None:
    """Force ``load_polylogue_config`` to report embeddings as enabled."""

    import polylogue.daemon.similarity as similarity_mod

    class _Cfg:
        embedding_enabled = True
        voyage_api_key = "test-key"

    monkeypatch.setattr(similarity_mod, "load_polylogue_config", lambda: _Cfg())


# ---------------------------------------------------------------------------
# Pure helper contracts
# ---------------------------------------------------------------------------


def test_confidence_bands_partition_score_space() -> None:
    assert _confidence_for_score(0.9) == "q-canonical"
    assert _confidence_for_score(0.75) == "q-canonical"
    assert _confidence_for_score(0.65) == "q-estimated"
    assert _confidence_for_score(0.55) == "q-estimated"
    assert _confidence_for_score(0.40) == "q-heuristic"
    assert _confidence_for_score(0.0) == "q-heuristic"


def test_disabled_reason_distinguishes_failure_modes() -> None:
    assert _disabled_reason(embedding_enabled=False, voyage_api_key=None) == "embeddings_not_enabled"
    assert _disabled_reason(embedding_enabled=False, voyage_api_key="k") == "embeddings_not_enabled"
    assert _disabled_reason(embedding_enabled=True, voyage_api_key=None) == "no_voyage_api_key"
    assert _disabled_reason(embedding_enabled=True, voyage_api_key="") == "no_voyage_api_key"
    assert _disabled_reason(embedding_enabled=True, voyage_api_key="key") is None


def test_clamp_limit_bounds_and_defaults() -> None:
    assert _clamp_limit(None) == SIMILAR_RESULTS_DEFAULT
    assert _clamp_limit(0) == SIMILAR_RESULTS_DEFAULT
    assert _clamp_limit(-5) == SIMILAR_RESULTS_DEFAULT
    assert _clamp_limit(5) == 5
    assert _clamp_limit(10**6) == SIMILAR_RESULTS_MAX


def test_l2_to_cosine_similarity_clamps_to_unit_interval() -> None:
    # Identical unit vectors → distance 0 → cosine 1.
    assert _l2_to_cosine_similarity(0.0) == 1.0
    # Orthogonal unit vectors → distance sqrt(2) → cosine 0.
    assert abs(_l2_to_cosine_similarity(2.0**0.5) - 0.0) < 1e-9
    # Antipodal unit vectors → distance 2 → cosine -1, clamped to 0.
    assert _l2_to_cosine_similarity(2.0) == 0.0
    # Pathological large distances clamp to 0 — never produce a
    # negative cosine that would slip past the heuristic chip band.
    assert _l2_to_cosine_similarity(100.0) == 0.0
    assert _l2_to_cosine_similarity(1000.0) == 0.0


def test_aggregate_hits_skips_self_and_takes_best_distance() -> None:
    per_message = [
        [("m1", "self", 0.0), ("m2", "other-a", 0.5), ("m3", "other-b", 1.2)],
        [("m4", "self", 0.1), ("m5", "other-a", 0.3), ("m6", "other-b", 1.0)],
    ]
    agg = _aggregate_hits(per_message, self_session_id="self")
    assert set(agg.keys()) == {"other-a", "other-b"}
    # other-a was matched by two source messages, best distance 0.3.
    assert agg["other-a"]["best_distance"] == pytest.approx(0.3)
    assert agg["other-a"]["matched_messages"] == 2
    # other-b matched once per source, best distance 1.0.
    assert agg["other-b"]["best_distance"] == pytest.approx(1.0)
    assert agg["other-b"]["matched_messages"] == 2


def test_aggregate_hits_handles_empty_input() -> None:
    assert _aggregate_hits([], self_session_id="self") == {}
    assert _aggregate_hits([[]], self_session_id="self") == {}


# ---------------------------------------------------------------------------
# Substrate envelope contracts
# ---------------------------------------------------------------------------


@pytest.mark.contract
class TestSimilarPayloadStates:
    """``build_similar_payload`` surfaces every absent state explicitly."""

    def test_returns_none_for_missing_session(
        self, workspace_env: dict[str, Path], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _enable_embeddings(monkeypatch)
        _init_archive()
        assert build_similar_payload("ghost") is None

    def test_disabled_envelope_when_embeddings_off(
        self, workspace_env: dict[str, Path], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _disable_embeddings(monkeypatch)
        _seed_archive_session("c1")
        result = build_similar_payload("c1")
        assert result is not None
        assert result["status"] == "disabled"
        assert result["reason"] == "embeddings_not_enabled"
        assert result["results"] == []
        assert result["session_id"] == "c1"

    def test_disabled_envelope_distinguishes_missing_api_key(
        self, workspace_env: dict[str, Path], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import polylogue.daemon.similarity as similarity_mod

        class _Cfg:
            embedding_enabled = True
            voyage_api_key = None

        monkeypatch.setattr(similarity_mod, "load_polylogue_config", lambda: _Cfg())
        _seed_archive_session("c1")
        result = build_similar_payload("c1")
        assert result is not None
        assert result["status"] == "disabled"
        assert result["reason"] == "no_voyage_api_key"

    def test_unavailable_envelope_when_vec_table_missing(
        self, workspace_env: dict[str, Path], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _enable_embeddings(monkeypatch)
        _seed_archive_session("c1")
        # No ``message_embeddings`` table has been created — the
        # embedding stage has never run on this archive.
        result = build_similar_payload("c1")
        assert result is not None
        assert result["status"] == "unavailable"
        assert result["reason"] == "vec0_table_missing"
        assert result["results"] == []

    def test_clamps_limit_in_envelope(self, workspace_env: dict[str, Path], monkeypatch: pytest.MonkeyPatch) -> None:
        _disable_embeddings(monkeypatch)
        _seed_archive_session("c1")
        result = build_similar_payload("c1", limit=10**6)
        assert result is not None
        assert result["limit"] == SIMILAR_RESULTS_MAX

    def test_archive_file_set_disabled_envelope_without_polylogue_db(
        self, workspace_env: dict[str, Path], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _disable_embeddings(monkeypatch)
        _seed_archive_session("codex-session:v1", origin="codex-session", title="Archive")

        result = build_similar_payload("codex-session:v1")

        assert result is not None
        assert result["status"] == "disabled"
        assert result["reason"] == "embeddings_not_enabled"
        assert result["session_id"] == "codex-session:v1"

    def test_archive_tiers_unavailable_when_vec_table_missing(
        self, workspace_env: dict[str, Path], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import polylogue.daemon.similarity as similarity_mod

        class _Cfg:
            embedding_enabled = True
            voyage_api_key = "test-key"

        monkeypatch.setattr(similarity_mod, "load_polylogue_config", lambda: _Cfg())
        _seed_archive_session("codex-session:v1", origin="codex-session", title="Archive")

        result = build_similar_payload("codex-session:v1")

        assert result is not None
        assert result["status"] == "unavailable"
        assert result["reason"] == "vec0_table_missing"
        assert result["session_id"] == "codex-session:v1"


# ---------------------------------------------------------------------------
# HTTP endpoint contracts
# ---------------------------------------------------------------------------


@pytest.mark.contract
class TestSimilarEndpoint:
    """``GET /api/sessions/{id}/similar`` HTTP route contract."""

    def test_missing_session_returns_404(self, workspace_env: dict[str, Path], monkeypatch: pytest.MonkeyPatch) -> None:
        _enable_embeddings(monkeypatch)
        _init_archive()
        handler = _make_handler("GET", "/api/sessions/ghost/similar")
        send_error, send_json = _capture_responses(handler)
        handler.do_GET()

        send_error.assert_called_once()
        status, code = send_error.call_args.args
        assert status == HTTPStatus.NOT_FOUND
        assert code == "not_found"
        send_json.assert_not_called()

    def test_disabled_envelope_routes_through_200(
        self, workspace_env: dict[str, Path], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Disabled-state is a real response, not an error.

        The reader expects ``200`` with ``status="disabled"`` so it can
        render the operator-facing guidance string. A 5xx here would
        cause the inspector tab to render an opaque "fetch failed"
        message and hide the actionable disabled state.
        """
        _disable_embeddings(monkeypatch)
        _seed_archive_session("c1")

        handler = _make_handler("GET", "/api/sessions/c1/similar")
        send_error, send_json = _capture_responses(handler)
        handler.do_GET()

        send_error.assert_not_called()
        send_json.assert_called_once()
        status, payload = send_json.call_args.args
        assert status == HTTPStatus.OK
        assert isinstance(payload, dict)
        assert payload["status"] == "disabled"
        assert payload["reason"] == "embeddings_not_enabled"
        assert payload["results"] == []

    def test_limit_query_param_propagates_to_envelope(
        self, workspace_env: dict[str, Path], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _disable_embeddings(monkeypatch)
        _seed_archive_session("c1")

        handler = _make_handler("GET", "/api/sessions/c1/similar?limit=3")
        _, send_json = _capture_responses(handler)
        handler.do_GET()

        _, payload = send_json.call_args.args
        assert payload["limit"] == 3

    def test_unparseable_limit_falls_back_to_default(
        self, workspace_env: dict[str, Path], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _disable_embeddings(monkeypatch)
        _seed_archive_session("c1")

        handler = _make_handler("GET", "/api/sessions/c1/similar?limit=banana")
        _, send_json = _capture_responses(handler)
        handler.do_GET()

        _, payload = send_json.call_args.args
        assert payload["limit"] == SIMILAR_RESULTS_DEFAULT

    def test_unavailable_envelope_when_pipeline_dormant(
        self, workspace_env: dict[str, Path], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _enable_embeddings(monkeypatch)
        _seed_archive_session("c1")

        handler = _make_handler("GET", "/api/sessions/c1/similar")
        _, send_json = _capture_responses(handler)
        handler.do_GET()

        _, payload = send_json.call_args.args
        assert payload["status"] == "unavailable"
        assert payload["reason"] == "vec0_table_missing"
