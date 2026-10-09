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

import sqlite3
from concurrent.futures import ThreadPoolExecutor
from email.message import Message
from http import HTTPStatus
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast
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
from polylogue.storage.embeddings.identity import vector_derivation_hash
from polylogue.storage.search_providers.sqlite_vec import SqliteVecProvider
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_tier
from polylogue.storage.sqlite.archive_tiers.embedding_write import upsert_message_embedding
from polylogue.storage.sqlite.archive_tiers.embeddings import EMBEDDING_DIMENSION
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from polylogue.storage.sqlite.connection_profile import CheckpointEscalation
from tests.infra.identity import fixture_block_content_identity
from tests.infra.vector_archive import record_owned_vector_closes, record_similarity_read_closes

if TYPE_CHECKING:
    from polylogue.daemon.http import DaemonAPIHandler, DaemonAPIHTTPServer


class _MockServer:
    auth_token = ""
    api_host = "127.0.0.1"
    archive_query_executor = ThreadPoolExecutor(max_workers=1)


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
    text_by_native_id = {
        "seed": "Alpha seed session with durable authored prose.",
        "near": "Alpha near neighbor session with durable authored prose.",
        "far": "Zeta unrelated topic with durable authored prose.",
    }
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
                    session_id, native_id, position, role, material_origin,
                    word_count, content_hash
                ) VALUES (?, ?, 0, 'user', 'human_authored', ?, ?)
                """,
                (session_id, "m1", len(text_by_native_id[native_id].split()), b"x" * 32),
            )
            conn.execute(
                "INSERT OR REPLACE INTO blocks (\n                    session_id, message_id, position, block_type, text, content_hash\n                , content_identity, content_occurrence) VALUES (?, ?, 0, 'text', ?, ?, ?, 0)",
                (
                    session_id,
                    f"{session_id}:n:m1",
                    text_by_native_id[native_id],
                    b"x" * 32,
                    fixture_block_content_identity("text", text_by_native_id[native_id]),
                ),
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
                    model="voyage-4-lite",
                    embedded_at_ms=1_767_225_700_000,
                    vector_derivation_hash=vector_derivation_hash(
                        model="voyage-4-lite", input_text=text_by_native_id[session_id.rsplit(":", 1)[-1]]
                    ),
                )
    except RuntimeError as exc:
        if "sqlite-vec" in str(exc) or "vec0" in str(exc):
            pytest.skip("sqlite-vec extension is unavailable")
        raise
    return "codex-session:seed", embeddings_db, session_by_message_id


def _disable_embeddings(monkeypatch: pytest.MonkeyPatch) -> None:
    """Force ``load_polylogue_config`` to return an embeddings-off config."""

    import polylogue.daemon.similarity as similarity_mod

    class _Cfg:
        embedding_enabled = False
        embedding_model = "voyage-4-lite"
        embedding_dimension = EMBEDDING_DIMENSION
        voyage_api_key: str | None = None

    monkeypatch.setattr(similarity_mod, "load_polylogue_config", lambda: _Cfg())


def _enable_embeddings(monkeypatch: pytest.MonkeyPatch) -> None:
    """Force ``load_polylogue_config`` to report embeddings as enabled."""

    import polylogue.daemon.similarity as similarity_mod

    class _Cfg:
        embedding_enabled = True
        embedding_model = "voyage-4-lite"
        embedding_dimension = EMBEDDING_DIMENSION
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


def test_disabled_reason_depends_only_on_embedding_policy() -> None:
    assert _disabled_reason(embedding_enabled=False) == "embeddings_not_enabled"
    assert _disabled_reason(embedding_enabled=True) is None


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
        _seed_ready_similarity_archive()
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

    def test_missing_api_key_does_not_disable_vector_reads(
        self, workspace_env: dict[str, Path], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import polylogue.daemon.similarity as similarity_mod

        class _Cfg:
            embedding_enabled = True
            embedding_model = "voyage-4-lite"
            embedding_dimension = EMBEDDING_DIMENSION
            voyage_api_key: str | None = None

        monkeypatch.setattr(similarity_mod, "load_polylogue_config", lambda: _Cfg())
        session_id = _seed_archive_session("c1")
        result = build_similar_payload(session_id)
        assert result is not None
        assert result["status"] == "not_embedded"
        assert result["reason"] is None

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
            embedding_model = "voyage-4-lite"
            embedding_dimension = EMBEDDING_DIMENSION
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
        _seed_ready_similarity_archive()
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

    def test_unresolvable_embedding_hits_report_inconsistent_not_ready(
        self, workspace_env: dict[str, Path], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Neighbors that match no indexed message are a broken join, not "nothing similar".

        A stale embedding generation, or a reindex that changed message-identity
        derivation, leaves the embeddings tier pointing at message ids the index
        no longer carries. Ranking over that join yields zero survivors, and
        answering ``ready`` with an empty list is indistinguishable from a
        healthy archive that simply holds nothing similar.
        """
        _enable_embeddings(monkeypatch)
        seed_session_id, _embeddings_db, _mapping = _seed_ready_similarity_archive()

        # Inject a broken ranking join at the provider seam. Production projection
        # filters stale vectors; a malformed provider result must still stay typed.
        original_read = SqliteVecProvider.read_similarity

        async def broken_join(self: SqliteVecProvider, **kwargs: Any) -> Any:
            project = kwargs["project"]
            kwargs["project"] = lambda connection, count, _hits: project(connection, count, [("orphan-message", 0.1)])
            return await original_read(self, **kwargs)

        monkeypatch.setattr(SqliteVecProvider, "read_similarity", broken_join)

        handler = _make_handler("GET", f"/api/sessions/{seed_session_id}/similar?limit=3")
        _, send_json = _capture_responses(handler)
        handler.do_GET()

        _, payload = send_json.call_args.args
        assert payload["status"] == "inconsistent"
        assert payload["reason"] == "embedded_messages_missing_from_index"
        assert payload["results"] == []
        assert payload["unresolved_message_hits"] > 0

    def test_ready_route_preserves_provider_query_by_session_order(
        self, workspace_env: dict[str, Path], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The live route projects the provider's session-seeded KNN order."""
        _enable_embeddings(monkeypatch)
        seed_session_id, embeddings_db, session_by_message_id = _seed_ready_similarity_archive()
        provider = SqliteVecProvider(
            voyage_key="test-key",
            db_path=embeddings_db,
            model="voyage-4-lite",
            archive_root=archive_root(),
        )
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


@pytest.mark.contract
def test_retained_vectors_are_queryable_without_acquisition_credentials(
    workspace_env: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Restoring either credential gate breaks the ordinary HTTP read."""
    import polylogue.daemon.similarity as similarity_mod
    from polylogue.config import PolylogueConfig

    config = PolylogueConfig(_data={"embedding_enabled": True})
    monkeypatch.setattr(similarity_mod, "load_polylogue_config", lambda: config)
    monkeypatch.setattr("polylogue.config.load_polylogue_config", lambda: config)
    monkeypatch.delenv("VOYAGE_API_KEY", raising=False)
    session_id, _, _ = _seed_ready_similarity_archive()
    preflight_closed = record_similarity_read_closes(monkeypatch)
    provider_call = MagicMock(side_effect=AssertionError("retained reads must not acquire vectors"))
    monkeypatch.setattr(SqliteVecProvider, "_get_embeddings", provider_call)
    handler = _make_handler("GET", f"/api/sessions/{session_id}/similar?limit=3")
    send_error, send_json = _capture_responses(handler)

    handler.do_GET()

    send_error.assert_not_called()
    _, payload = send_json.call_args.args
    assert payload["status"] == "ready"
    assert payload["source_embedded_messages"] == 1
    assert [hit["session_id"] for hit in payload["results"]] == ["codex-session:near", "codex-session:far"]
    assert payload["results"][0]["score"] > 0.98
    provider_call.assert_not_called()
    assert preflight_closed == [True]


@pytest.mark.contract
@pytest.mark.parametrize("first_promotion", [False, True])
def test_similarity_publication_keeps_count_ranking_and_hydration_on_selected_generation(
    workspace_env: dict[str, Path], monkeypatch: pytest.MonkeyPatch, first_promotion: bool
) -> None:
    """Reopening the active index between count and ranking changes this result."""
    import polylogue.storage.index_generation as generation_module
    from polylogue.config import PolylogueConfig
    from polylogue.core.sqlite_locking import is_transient_sqlite_lock
    from polylogue.storage.index_generation import IndexGenerationStore
    from polylogue.storage.sqlite.wal_checkpoint import checkpoint_connection as original_checkpoint

    config = PolylogueConfig(_data={"embedding_enabled": True})
    monkeypatch.setattr("polylogue.daemon.similarity.load_polylogue_config", lambda: config)
    monkeypatch.setattr("polylogue.config.load_polylogue_config", lambda: config)
    monkeypatch.delenv("VOYAGE_API_KEY", raising=False)
    session_id, _, _ = _seed_ready_similarity_archive()
    root = archive_root()
    store = IndexGenerationStore.for_archive_root(root)
    first = store.create(owner_id="similarity-first", source_snapshot="synthetic-first")
    with sqlite3.connect(root / "index.db") as source, sqlite3.connect(first.index_path) as target:
        source.backup(target)
    if not first_promotion:
        store.promote(first)
    second = store.create(owner_id="similarity-second", source_snapshot="synthetic-second")
    with sqlite3.connect(first.index_path) as source, sqlite3.connect(second.index_path) as target:
        source.backup(target)
        target.execute("UPDATE sessions SET title = 'New generation'")
        target.execute("UPDATE blocks SET text = 'Changed prose without a matching retained vector'")
    original_count = SqliteVecProvider.count_session_embeddings
    published: list[Path] = []
    busy_checkpoints: list[tuple[int, int, int]] = []
    blocked: list[BaseException] = []

    def checkpoint(
        connection: sqlite3.Connection, mode: str, *, boundary: CheckpointEscalation
    ) -> tuple[int, int, int]:
        result = original_checkpoint(connection, mode, boundary=boundary)
        if result[0]:
            busy_checkpoints.append(result)
        return result

    monkeypatch.setattr(generation_module, "checkpoint_connection", checkpoint)
    closed = record_owned_vector_closes(monkeypatch)

    def count_and_publish(provider: SqliteVecProvider, seed: str) -> int:
        count = original_count(provider, seed)
        try:
            store.promote(second)
        except (sqlite3.Error, RuntimeError) as exc:
            if not (is_transient_sqlite_lock(exc) or busy_checkpoints):
                raise
            blocked.append(exc)
        else:
            published.append(resolve_active_index_path(root).resolve(strict=True))
        return count

    monkeypatch.setattr(SqliteVecProvider, "count_session_embeddings", count_and_publish)
    provider_call = MagicMock(side_effect=AssertionError("retained reads must not acquire vectors"))
    monkeypatch.setattr(SqliteVecProvider, "_get_embeddings", provider_call)
    handler = _make_handler("GET", f"/api/sessions/{session_id}/similar?limit=3")
    send_error, send_json = _capture_responses(handler)

    handler.do_GET()

    send_error.assert_not_called()
    _, payload = send_json.call_args.args
    assert len(published) + len(blocked) == 1, payload
    assert payload["status"] == "ready"
    assert payload["source_embedded_messages"] == 1
    assert [(hit["session_id"], hit["title"]) for hit in payload["results"]] == [
        ("codex-session:near", "Near"),
        ("codex-session:far", "Far"),
    ]
    assert closed == [True]
    if blocked:
        store.promote(second)
        published.append(resolve_active_index_path(root).resolve(strict=True))
    assert published == [Path(second.index_path)]
    provider_call.assert_not_called()


@pytest.mark.contract
@pytest.mark.parametrize("failure", ["corrupt", "missing", "projection", "contention", "stale", "runtime"])
def test_unreadable_retained_vectors_never_certify_absence(
    workspace_env: dict[str, Path], monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    """Replacing unavailable failures with not_embedded turns this red."""
    _enable_embeddings(monkeypatch)
    session_id, embeddings_db, _ = _seed_ready_similarity_archive()
    closed = record_owned_vector_closes(monkeypatch)
    preflight_closed = record_similarity_read_closes(monkeypatch)
    provider_call = MagicMock(side_effect=AssertionError("retained reads must not acquire vectors"))
    monkeypatch.setattr(SqliteVecProvider, "_get_embeddings", provider_call)
    if failure == "runtime":
        monkeypatch.setattr("polylogue.storage.search_providers._sqlite_vec_available", lambda: False)
    elif failure == "corrupt":
        embeddings_db.write_bytes(b"synthetic unreadable database")
    elif failure == "missing":
        embeddings_db.unlink()
    elif failure == "stale":
        with sqlite3.connect(_index_db()) as conn:
            conn.execute(
                "UPDATE blocks SET text = ? WHERE session_id = ?",
                ("Changed current prose with no retained vector for this recipe.", session_id),
            )
    elif failure == "projection":
        with sqlite3.connect(_index_db()) as conn:
            conn.execute("DROP TABLE blocks")
    else:
        from polylogue.storage.search_providers.sqlite_vec_support import SqliteVecError

        def unavailable(self: SqliteVecProvider, session_id: str) -> int:
            raise SqliteVecError("stored vectors could not be read") from sqlite3.OperationalError("database is locked")

        monkeypatch.setattr(SqliteVecProvider, "count_session_embeddings", unavailable)
    handler = _make_handler("GET", f"/api/sessions/{session_id}/similar")
    send_error, send_json = _capture_responses(handler)

    handler.do_GET()

    send_error.assert_not_called()
    _, payload = send_json.call_args.args
    assert payload["status"] == "unavailable"
    assert (
        payload["reason"]
        == {
            "corrupt": "embeddings_db_unreadable",
            "missing": "vec0_table_missing",
            "projection": "embedding_read_failed",
            "stale": "embedding_read_failed",
            "contention": "sqlite_contention",
            "runtime": "sqlite_vec_not_loaded",
        }[failure]
    )
    assert payload["results"] == []
    assert closed == ([True] if failure in {"contention", "stale"} else [])
    provider_call.assert_not_called()
    assert preflight_closed == ([] if failure in {"missing", "corrupt"} else [True])


@pytest.mark.contract
def test_http_daemon_binds_vector_snapshot_without_acquisition_credentials(
    workspace_env: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Restoring the HTTP server's key gate removes the machine read binding."""
    from polylogue.config import PolylogueConfig
    from polylogue.daemon.http import DaemonAPIHandler, DaemonAPIHTTPServer

    config = PolylogueConfig(_data={"embedding_enabled": True, "archive_root": str(archive_root())})
    monkeypatch.setattr("polylogue.config.load_polylogue_config", lambda: config)
    with DaemonAPIHTTPServer(("127.0.0.1", 0), DaemonAPIHandler, archive_root=archive_root()) as server:
        factory = server.operation_runtime._read_dependencies_factory
        assert factory is not None
        binding = factory().vector_binding
        assert binding is not None
        assert binding.voyage_key is None


@pytest.mark.contract
def test_similarity_first_publication_checks_seed_on_query_snapshot(
    workspace_env: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Checking existence before opening the vector snapshot gives false not_embedded."""
    from polylogue import Polylogue
    from polylogue.storage.index_generation import IndexGenerationStore
    from tests.infra.archive_templates import run_off_event_loop

    _enable_embeddings(monkeypatch)
    monkeypatch.delenv("VOYAGE_API_KEY", raising=False)
    session_id, _, _ = _seed_ready_similarity_archive()
    root = archive_root()
    store = IndexGenerationStore.for_archive_root(root)
    successor = store.create(owner_id="similarity-absence", source_snapshot="synthetic-absence")
    with sqlite3.connect(root / "index.db") as source, sqlite3.connect(successor.index_path) as target:
        source.backup(target)
        target.execute("DELETE FROM blocks WHERE session_id = ?", (session_id,))
        target.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
        target.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
    with sqlite3.connect(root / "embeddings.db") as vectors:
        assert (
            vectors.execute(
                "SELECT COUNT(*) FROM message_embedding_refs WHERE session_id = ?", (session_id,)
            ).fetchone()[0]
            == 1
        )
    original_query = Polylogue.search_similar_sessions
    published: list[Path] = []

    async def publish_before_query(archive: Polylogue, seed: str, *, limit: int = 10) -> dict[str, object]:
        # Promotion takes the synchronous writer lease, which refuses to block
        # the query's running event loop; publish from a loop-free thread.
        run_off_event_loop(lambda: store.promote(successor))
        published.append(resolve_active_index_path(root).resolve(strict=True))
        return await original_query(archive, seed, limit=limit)

    monkeypatch.setattr(Polylogue, "search_similar_sessions", publish_before_query)
    provider_call = MagicMock(side_effect=AssertionError("retained reads must not acquire vectors"))
    monkeypatch.setattr(SqliteVecProvider, "_get_embeddings", provider_call)
    closed = record_owned_vector_closes(monkeypatch)
    handler = _make_handler("GET", f"/api/sessions/{session_id}/similar")
    send_error, send_json = _capture_responses(handler)

    handler.do_GET()

    assert published == [Path(successor.index_path)]
    send_error.assert_called_once_with(HTTPStatus.NOT_FOUND, "not_found")
    send_json.assert_not_called()
    assert closed == [True]
    provider_call.assert_not_called()
