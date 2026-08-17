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
environment); the ready-state parity test uses the real provider when
sqlite-vec is available, while the other tests pin the route and absent
states.
"""

from __future__ import annotations

import hashlib
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

from polylogue.core.enums import Origin
from polylogue.daemon.similarity import (
    SIMILAR_RESULTS_DEFAULT,
    SIMILAR_RESULTS_MAX,
    _clamp_limit,
    _confidence_for_score,
    _disabled_reason,
    build_similar_payload,
)
from polylogue.paths import archive_root
from polylogue.storage.archive_identity import resolve_active_index_path
from polylogue.storage.search_providers.sqlite_vec import SqliteVecProvider
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_tier
from polylogue.storage.sqlite.archive_tiers.embedding_write import upsert_message_embedding
from polylogue.storage.sqlite.archive_tiers.embeddings import EMBEDDING_DIMENSION
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier

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
    return resolve_active_index_path(archive_root())


def _session_parts(session_id: str, origin: str) -> tuple[str, str]:
    prefix = f"{origin}:"
    native_id = session_id[len(prefix) :] if session_id.startswith(prefix) else session_id
    return native_id, f"{origin}:{native_id}"


def _seed_archive_session(
    session_id: str,
    *,
    origin: str = "claude-code-session",
    title: str = "stub",
) -> str:
    """Seed an archive `sessions` row in index.db.

    The similarity reader (``polylogue/daemon/similarity.py``) routes to
    the archive path whenever ``index.db`` exists; it only needs the
    ``sessions`` row to confirm the session exists before rendering
    the disabled/unavailable envelope.
    """
    archive_db = _index_db()
    archive_db.parent.mkdir(parents=True, exist_ok=True)
    native_id, archive_session_id = _session_parts(session_id, origin)
    with sqlite3.connect(archive_db) as conn:
        conn.execute(
            """
            INSERT OR IGNORE INTO sessions (
                native_id, origin, title, content_hash
            ) VALUES (?, ?, ?, ?)
            """,
            (native_id, origin, title, b"x" * 32),
        )
        conn.commit()
    return archive_session_id


def _unit_vector(*, axis0: float, axis1: float) -> list[float]:
    vector = [0.0] * EMBEDDING_DIMENSION
    vector[0] = axis0
    vector[1] = axis1
    return vector


def _seed_ready_similarity_archive() -> tuple[str, Path, dict[str, str]]:
    """Seed canonical index and embedding tiers for the route parity test.

    Embeddings must reference the message ids the index actually generates.
    ``messages.message_id`` is a generated column -- ``session_id || ':' ||
    ('n:' || native_id)`` when a native id exists -- so hand-writing
    ``<session>:<native_id>`` produces ids that match no row, and the route
    silently drops every vector hit while still reporting ``ready``. Read the
    generated ids back and seed against those.
    """
    root = archive_root()
    index_db = root / "index.db"
    with sqlite3.connect(index_db) as conn:
        initialize_archive_tier(conn, ArchiveTier.INDEX)
        for native_id, title in (("seed", "Seed"), ("near", "Near"), ("far", "Far")):
            conn.execute(
                "INSERT OR REPLACE INTO sessions (native_id, origin, title, content_hash) VALUES (?, ?, ?, ?)",
                (native_id, "codex-session", title, b"x" * 32),
            )
        for native_id in ("seed", "near", "far"):
            session_id = f"codex-session:{native_id}"
            conn.execute(
                """
                INSERT OR REPLACE INTO messages (
                    session_id, native_id, position, role, content_hash
                ) VALUES (?, ?, 0, 'user', ?)
                """,
                (session_id, "m1", b"x" * 32),
            )
        message_id_by_session = {
            str(row[1]): str(row[0]) for row in conn.execute("SELECT message_id, session_id FROM messages")
        }
    session_by_message_id = {message_id: session for session, message_id in message_id_by_session.items()}

    embeddings_db = root / "embeddings.db"
    try:
        with sqlite3.connect(embeddings_db) as conn:
            initialize_archive_tier(conn, ArchiveTier.EMBEDDINGS)
            for session_id, message_id, vector in (
                ("codex-session:seed", message_id_by_session["codex-session:seed"], _unit_vector(axis0=1.0, axis1=0.0)),
                (
                    "codex-session:near",
                    message_id_by_session["codex-session:near"],
                    _unit_vector(axis0=0.99, axis1=0.141),
                ),
                ("codex-session:far", message_id_by_session["codex-session:far"], _unit_vector(axis0=0.0, axis1=1.0)),
            ):
                upsert_message_embedding(
                    conn,
                    message_id=message_id,
                    session_id=session_id,
                    origin=Origin.CODEX_SESSION,
                    embedding=vector,
                    model="voyage-4",
                    embedded_at_ms=1_767_225_700_000,
                    embedding_input_hash=hashlib.sha256(message_id.encode()).digest(),
                )
    except RuntimeError as exc:
        if "sqlite-vec" in str(exc) or "vec0" in str(exc):
            pytest.skip("sqlite-vec extension is unavailable")
        raise
    return "codex-session:seed", embeddings_db, session_by_message_id


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
        voyage_api_key: str | None = None

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
        session_id = _seed_archive_session("c1")
        result = build_similar_payload(session_id)
        assert result is not None
        assert result["status"] == "disabled"
        assert result["reason"] == "embeddings_not_enabled"
        assert result["results"] == []
        assert result["session_id"] == session_id

    def test_disabled_envelope_distinguishes_missing_api_key(
        self, workspace_env: dict[str, Path], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import polylogue.daemon.similarity as similarity_mod

        class _Cfg:
            embedding_enabled = True
            voyage_api_key: str | None = None

        monkeypatch.setattr(similarity_mod, "load_polylogue_config", lambda: _Cfg())
        session_id = _seed_archive_session("c1")
        result = build_similar_payload(session_id)
        assert result is not None
        assert result["status"] == "disabled"
        assert result["reason"] == "no_voyage_api_key"

    def test_not_embedded_envelope_when_session_has_no_vectors(
        self, workspace_env: dict[str, Path], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _enable_embeddings(monkeypatch)
        session_id = _seed_archive_session("c1")
        result = build_similar_payload(session_id)
        assert result is not None
        assert result["status"] == "not_embedded"
        assert result["reason"] is None
        assert result["results"] == []

    def test_clamps_limit_in_envelope(self, workspace_env: dict[str, Path], monkeypatch: pytest.MonkeyPatch) -> None:
        _disable_embeddings(monkeypatch)
        session_id = _seed_archive_session("c1")
        result = build_similar_payload(session_id, limit=10**6)
        assert result is not None
        assert result["limit"] == SIMILAR_RESULTS_MAX

    def test_archive_file_set_disabled_envelope_from_archive_tiers(
        self, workspace_env: dict[str, Path], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _disable_embeddings(monkeypatch)
        session_id = _seed_archive_session("codex-session:v1", origin="codex-session", title="Archive")

        result = build_similar_payload(session_id)

        assert result is not None
        assert result["status"] == "disabled"
        assert result["reason"] == "embeddings_not_enabled"
        assert result["session_id"] == session_id

    def test_archive_tiers_not_embedded_when_session_has_no_vectors(
        self, workspace_env: dict[str, Path], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import polylogue.daemon.similarity as similarity_mod

        class _Cfg:
            embedding_enabled = True
            voyage_api_key = "test-key"

        monkeypatch.setattr(similarity_mod, "load_polylogue_config", lambda: _Cfg())
        session_id = _seed_archive_session("codex-session:v1", origin="codex-session", title="Archive")

        result = build_similar_payload(session_id)

        assert result is not None
        assert result["status"] == "not_embedded"
        assert result["reason"] is None
        assert result["session_id"] == session_id


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
        session_id = _seed_archive_session("c1")

        handler = _make_handler("GET", f"/api/sessions/{session_id}/similar")
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
        session_id = _seed_archive_session("c1")

        handler = _make_handler("GET", f"/api/sessions/{session_id}/similar?limit=3")
        _, send_json = _capture_responses(handler)
        handler.do_GET()

        _, payload = send_json.call_args.args
        assert payload["limit"] == 3

    def test_unparseable_limit_falls_back_to_default(
        self, workspace_env: dict[str, Path], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _disable_embeddings(monkeypatch)
        session_id = _seed_archive_session("c1")

        handler = _make_handler("GET", f"/api/sessions/{session_id}/similar?limit=banana")
        _, send_json = _capture_responses(handler)
        handler.do_GET()

        _, payload = send_json.call_args.args
        assert payload["limit"] == SIMILAR_RESULTS_DEFAULT

    def test_not_embedded_envelope_when_pipeline_dormant(
        self, workspace_env: dict[str, Path], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _enable_embeddings(monkeypatch)
        session_id = _seed_archive_session("c1")

        handler = _make_handler("GET", f"/api/sessions/{session_id}/similar")
        _, send_json = _capture_responses(handler)
        handler.do_GET()

        _, payload = send_json.call_args.args
        assert payload["status"] == "not_embedded"
        assert payload["reason"] is None

    def test_ready_route_preserves_provider_query_by_session_order(
        self, workspace_env: dict[str, Path], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The live route projects the provider's session-seeded KNN order."""
        _enable_embeddings(monkeypatch)
        seed_session_id, embeddings_db, session_by_message_id = _seed_ready_similarity_archive()
        provider = SqliteVecProvider(voyage_key="test-key", db_path=embeddings_db, model="voyage-4")
        expected_message_hits = provider.query_by_session(seed_session_id, limit=3)
        # Resolve message -> session through the index's own mapping. Splitting the
        # id on ':' assumes a positional shape and mangles the `n:<native_id>` form.
        expected_session_order = list(
            dict.fromkeys(
                session_by_message_id[message_id]
                for message_id, _distance in expected_message_hits
                if message_id in session_by_message_id
            )
        )
        assert expected_session_order, "provider returned no resolvable session hits to compare against"

        handler = _make_handler("GET", f"/api/sessions/{seed_session_id}/similar?limit=3")
        _, send_json = _capture_responses(handler)
        handler.do_GET()

        _, payload = send_json.call_args.args
        assert payload["status"] == "ready"
        actual_session_order = [row["session_id"] for row in payload["results"]]
        assert actual_session_order == expected_session_order
