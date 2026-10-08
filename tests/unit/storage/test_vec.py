"""Focused capability contracts for the sqlite-vec provider."""

from __future__ import annotations

import sqlite3
import struct
from collections.abc import Callable
from pathlib import Path
from typing import Protocol, TypeAlias
from unittest.mock import MagicMock, patch

import httpx
import pytest

from polylogue.storage.embeddings.identity import EmbeddingRecipe
from polylogue.storage.search_providers.sqlite_vec import SqliteVecProvider
from polylogue.storage.search_providers.sqlite_vec_runtime import open_vector_read_snapshot
from polylogue.storage.search_providers.sqlite_vec_support import SqliteVecError
from polylogue.storage.sqlite.archive_tiers.embeddings import EMBEDDING_DIMENSION

Embedding: TypeAlias = list[float]


class EmbeddingFetcher(Protocol):
    def __call__(self, texts: list[str], input_type: str = "document") -> list[Embedding]: ...


class VectorReadConnection(Protocol):
    def __call__(
        self,
        *,
        index_path: Path | None = None,
        index_connection: sqlite3.Connection | None = None,
        configure_connection: Callable[[sqlite3.Connection], None] | None = None,
    ) -> sqlite3.Connection: ...


class MutableSqliteVecProvider(SqliteVecProvider):
    _ensure_vec_available: Callable[[], None]
    _ensure_tables: Callable[[], None]
    _get_embeddings: EmbeddingFetcher
    _get_connection: Callable[[], sqlite3.Connection]
    _get_read_connection: VectorReadConnection


@pytest.fixture
def mock_provider(tmp_path: Path) -> MutableSqliteVecProvider:
    from tests.infra.vector_archive import seed_vector_archive

    seed_vector_archive(tmp_path, [])
    provider = MutableSqliteVecProvider(
        voyage_key="test-voyage-key", db_path=tmp_path / "embeddings.db", model="voyage-4", archive_root=tmp_path
    )
    provider.dimension = 1024
    provider._vec_available = None
    provider._tables_ensured = True
    return provider


def test_operation_snapshot_provider_never_closes_or_writes_its_supplied_handle(tmp_path: Path) -> None:
    """Mutation: make snapshot reads open/close their own connection and this fails."""

    from polylogue.storage.embeddings.identity import EmbeddingRecipe
    from polylogue.storage.search_providers.sqlite_vec_runtime import open_vector_read_snapshot
    from tests.infra.vector_archive import seed_vector_archive

    seed_vector_archive(tmp_path, [])
    connection = open_vector_read_snapshot(
        embeddings_path=tmp_path / "embeddings.db",
        index_path=tmp_path / "index.db",
        recipe=EmbeddingRecipe.current(model="voyage-4", dimensions=1024),
    )
    provider = SqliteVecProvider.from_vector_read_snapshot(
        voyage_key="test-voyage-key",
        connection=connection,
        model="voyage-4",
    )

    assert provider._get_connection() is connection
    provider._release_connection(connection)
    assert tuple(connection.execute("SELECT 1").fetchone()) == (1,)
    assert tuple(connection.execute("SELECT 1").fetchone()) == (1,)
    connection.close()


def test_open_vector_read_snapshot_uses_only_the_explicit_pinned_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Mutation: resolve the active index instead of the supplied path and this fails."""

    from tests.infra.vector_archive import seed_vector_archive

    identities = seed_vector_archive(
        tmp_path,
        [("session-1", "message-1", "a sufficiently long pinned snapshot message", [1.0] + [0.0] * 1023)],
    )
    _, message_id = identities[("session-1", "message-1")]
    embeddings_path = tmp_path / "pinned-embeddings.db"
    pinned_index_path = tmp_path / "pinned-index.db"
    (tmp_path / "embeddings.db").rename(embeddings_path)
    (tmp_path / "index.db").rename(pinned_index_path)
    embeddings = sqlite3.connect(embeddings_path)
    try:
        embeddings.execute("PRAGMA journal_mode = WAL")
        embeddings.execute("CREATE TABLE snapshot_probe (value INTEGER NOT NULL)")
        embeddings.execute("INSERT INTO snapshot_probe VALUES (1)")
        embeddings.commit()
    finally:
        embeddings.close()
    index = sqlite3.connect(pinned_index_path)
    try:
        index.execute("PRAGMA journal_mode = WAL")
    finally:
        index.close()

    import polylogue.storage.search_providers.sqlite_vec_runtime as runtime

    def unexpected_active_index_resolution(*args: object, **kwargs: object) -> Path:
        raise AssertionError("snapshot reader must not resolve an active index")

    monkeypatch.setattr(runtime, "resolve_active_index_path", unexpected_active_index_resolution)
    configure_projection = runtime._configure_current_embedding_messages

    def publish_after_pin(
        connection: sqlite3.Connection,
        *,
        index_path: Path | None = None,
        recipe: EmbeddingRecipe,
        attach_index: bool = True,
        register_identity: bool = True,
        index_connection: sqlite3.Connection | None = None,
    ) -> None:
        # A TEMP setup executescript would implicitly commit, admitting these
        # later writes into the allegedly pinned semantic snapshot.
        assert index_path is None
        with sqlite3.connect(embeddings_path) as writer:
            writer.execute("UPDATE snapshot_probe SET value = 2")
        assert attach_index is False
        assert register_identity is False
        with sqlite3.connect(pinned_index_path) as writer:
            writer.execute("UPDATE messages SET role = 'system'")
        configure_projection(
            connection,
            recipe=recipe,
            attach_index=attach_index,
            register_identity=register_identity,
            index_connection=index_connection,
        )

    monkeypatch.setattr(runtime, "_configure_current_embedding_messages", publish_after_pin)

    connection = open_vector_read_snapshot(
        embeddings_path=embeddings_path,
        index_path=pinned_index_path,
        recipe=EmbeddingRecipe.current(model="voyage-4", dimensions=EMBEDDING_DIMENSION),
    )
    try:
        assert connection.in_transaction
        assert connection.execute("PRAGMA query_only").fetchone()[0] == 1
        assert connection.execute("SELECT value FROM snapshot_probe").fetchone()[0] == 1
        assert connection.execute("SELECT message_id FROM current_embedding_messages").fetchone()[0] == message_id
    finally:
        connection.close()


def test_get_embeddings_request_contract(mock_provider: MutableSqliteVecProvider) -> None:
    """Embedding requests must send the canonical payload, headers, and optional dimension."""
    cases = [
        (1024, "document", None),
        (512, "query", 512),
    ]

    for dimension, input_type, expected_dimension in cases:
        mock_provider.dimension = dimension
        captured_payload: dict[str, object] = {}
        captured_headers: dict[str, str] = {}
        response = MagicMock()
        response.json.return_value = {"data": [{"embedding": [0.1, 0.2, 0.3]}]}
        response.raise_for_status = MagicMock()

        def capture_post(
            *args: object,
            _captured_payload: dict[str, object] = captured_payload,
            _captured_headers: dict[str, str] = captured_headers,
            _response: MagicMock = response,
            **kwargs: object,
        ) -> MagicMock:
            del args
            payload = kwargs.get("json")
            assert isinstance(payload, dict)
            _captured_payload.update({str(key): value for key, value in payload.items()})
            headers = kwargs.get("headers")
            assert isinstance(headers, dict)
            _captured_headers.update({str(key): str(value) for key, value in headers.items()})
            return _response

        with patch("httpx.Client") as mock_client_cls:
            client = MagicMock()
            client.post = capture_post
            client.__enter__ = MagicMock(return_value=client)
            client.__exit__ = MagicMock(return_value=False)
            mock_client_cls.return_value = client

            result = mock_provider._get_embeddings(["test text"], input_type=input_type)

        assert result == [[0.1, 0.2, 0.3]]
        assert captured_payload["input"] == ["test text"]
        assert captured_payload["model"] == mock_provider.model
        assert captured_payload["input_type"] == input_type
        if expected_dimension is None:
            assert "output_dimension" not in captured_payload
        else:
            assert captured_payload["output_dimension"] == expected_dimension
        assert captured_headers["Authorization"] == f"Bearer {mock_provider.voyage_key}"


@pytest.mark.slow
def test_get_embeddings_batches_large_input(mock_provider: MutableSqliteVecProvider) -> None:
    """Large requests must batch without dropping embeddings."""
    from polylogue.storage.search_providers.sqlite_vec import BATCH_SIZE

    texts = [f"text {i} is a long enough message for embedding purposes" for i in range(BATCH_SIZE + 10)]
    call_sizes: list[int] = []

    def fake_post(*args: object, **kwargs: object) -> MagicMock:
        del args
        payload = kwargs.get("json")
        assert isinstance(payload, dict)
        input_payload = payload.get("input")
        assert isinstance(input_payload, list)
        batch_size = len(input_payload)
        call_sizes.append(batch_size)
        response = MagicMock()
        response.json.return_value = {"data": [{"embedding": [0.1] * 3} for _ in range(batch_size)]}
        response.raise_for_status = MagicMock()
        return response

    with patch("httpx.Client") as mock_client_cls, patch("time.sleep") as sleep:
        client = MagicMock()
        client.post = fake_post
        client.__enter__ = MagicMock(return_value=client)
        client.__exit__ = MagicMock(return_value=False)
        mock_client_cls.return_value = client
        result = mock_provider._get_embeddings(texts)

    assert call_sizes == [BATCH_SIZE, 10]
    assert len(result) == len(texts)
    sleep.assert_called_once()


@pytest.mark.parametrize(
    ("error", "pattern"),
    [
        (
            httpx.HTTPStatusError(
                "rate limited",
                request=MagicMock(),
                response=MagicMock(status_code=429),
            ),
            "HTTP 429",
        ),
        (httpx.TimeoutException("Connection timed out"), "TimeoutException"),
    ],
    ids=["http-status", "timeout"],
)
def test_get_embeddings_error_contract(
    mock_provider: MutableSqliteVecProvider,
    error: httpx.HTTPError,
    pattern: str,
) -> None:
    """HTTP-layer failures must surface as sanitized sqlite-vec errors."""
    from polylogue.storage.search_providers.sqlite_vec import SqliteVecError

    with patch("httpx.Client") as mock_client_cls, patch("time.sleep"):
        client = MagicMock()
        client.post.side_effect = error
        client.__enter__ = MagicMock(return_value=client)
        client.__exit__ = MagicMock(return_value=False)
        mock_client_cls.return_value = client

        with pytest.raises(SqliteVecError, match=pattern):
            mock_provider._get_embeddings(["test text"])


def test_get_embeddings_error_does_not_leak_api_key(mock_provider: MutableSqliteVecProvider) -> None:
    """Sanitized errors must not expose the configured API key."""
    from polylogue.storage.search_providers.sqlite_vec import SqliteVecError

    error = httpx.HTTPStatusError(
        f"Unauthorized: Bearer {mock_provider.voyage_key}",
        request=MagicMock(),
        response=MagicMock(status_code=401),
    )

    with patch("httpx.Client") as mock_client_cls, patch("time.sleep"):
        client = MagicMock()
        client.post.side_effect = error
        client.__enter__ = MagicMock(return_value=client)
        client.__exit__ = MagicMock(return_value=False)
        mock_client_cls.return_value = client

        with pytest.raises(SqliteVecError) as exc_info:
            mock_provider._get_embeddings(["test text"])

    assert mock_provider.voyage_key is not None
    assert mock_provider.voyage_key not in str(exc_info.value)


@pytest.mark.parametrize(
    ("method_name", "provider", "embedding_result"),
    [
        ("query", None, [[0.1, 0.2]]),
        ("query", None, []),
    ],
    ids=["query", "query-empty"],
)
def test_query_route_contract(
    mock_provider: MutableSqliteVecProvider,
    method_name: str,
    provider: str | None,
    embedding_result: list[Embedding],
) -> None:
    """Query methods must generate query embeddings, optionally filter by provider, and close connections."""
    embedding_calls: list[tuple[list[str], str | None]] = []
    executed_queries: list[tuple[str, tuple[object, ...] | None]] = []

    def capture_embeddings(texts: list[str], input_type: str = "document") -> list[Embedding]:
        embedding_calls.append((texts, input_type))
        return embedding_result

    def capture_execute(sql: str, params: tuple[object, ...] | None = None) -> MagicMock:
        executed_queries.append((sql, params))
        row_1 = MagicMock(spec=sqlite3.Row)
        row_1.__getitem__.side_effect = lambda key: "msg-1" if key == "message_id" else 0.5
        row_2 = MagicMock(spec=sqlite3.Row)
        row_2.__getitem__.side_effect = lambda key: "msg-2" if key == "message_id" else 0.7
        result = MagicMock()
        result.__iter__.return_value = iter([row_1, row_2])
        return result

    mock_provider._get_embeddings = capture_embeddings
    connection = MagicMock()
    connection.execute = capture_execute
    cursor = MagicMock()

    def cursor_execute(sql: str, params: tuple[object, ...] | None = None) -> MagicMock:
        result = capture_execute(sql, params)
        cursor.__iter__.return_value = iter(result)
        return cursor

    cursor.execute = cursor_execute
    connection.cursor.return_value = cursor
    connection.close = MagicMock()
    mock_provider._get_read_connection = MagicMock(return_value=connection)

    result = mock_provider.query("search text", limit=10)

    assert embedding_calls == [(["search text"], "query")]
    if embedding_result:
        assert result == [("msg-1", 0.5), ("msg-2", 0.7)]
        assert connection.close.called
        assert all("r.origin = ?" not in sql for sql, _ in executed_queries)
    else:
        assert result == []
        # Provider may open a connection for early existence check before
        # bailing out. The result contract is still empty, and the connection
        # should have been closed by the caller.
        if mock_provider._get_read_connection.called:
            assert connection.close.called


@pytest.mark.parametrize(
    ("initial_state", "probed_state", "should_raise"),
    [(True, True, False), (False, False, True), (None, True, False), (None, False, True)],
    ids=["cached-available", "cached-unavailable", "probe-available", "probe-unavailable"],
)
def test_ensure_vec_available_contract(
    mock_provider: MutableSqliteVecProvider,
    initial_state: bool | None,
    probed_state: bool,
    should_raise: bool,
) -> None:
    """Availability probing should cache the result and raise with a helpful message when unavailable."""
    from polylogue.storage.search_providers.sqlite_vec import SqliteVecError

    mock_provider._vec_available = initial_state
    connection = MagicMock()
    connection.close = MagicMock()

    def get_connection() -> MagicMock:
        mock_provider._vec_available = probed_state
        return connection

    mock_provider._get_connection = MagicMock(side_effect=get_connection)

    if should_raise:
        with pytest.raises(SqliteVecError, match="sqlite-vec extension not available|pip install"):
            mock_provider._ensure_vec_available()
    else:
        mock_provider._ensure_vec_available()

    if initial_state is None:
        mock_provider._get_connection.assert_called_once()
        connection.close.assert_called_once()
    else:
        mock_provider._get_connection.assert_not_called()


def test_serialize_f32_contract() -> None:
    """Vector serialization must preserve float32 payloads and the empty-vector case."""
    from polylogue.storage.search_providers.sqlite_vec import _serialize_f32

    assert _serialize_f32([]) == b""
    packed = _serialize_f32([1.0, 2.0, 3.0, 4.0])
    assert len(packed) == 16
    assert struct.unpack("<4f", packed) == (1.0, 2.0, 3.0, 4.0)


def test_retained_reader_refuses_acquisition_without_key(tmp_path: Path) -> None:
    """Removing the acquisition guard would attempt a provider call."""
    from polylogue.storage.search_providers.sqlite_vec import SqliteVecProvider
    from polylogue.storage.search_providers.sqlite_vec_support import SqliteVecError

    provider = SqliteVecProvider(voyage_key=None, db_path=tmp_path / "embeddings.db")
    with pytest.raises(SqliteVecError):
        provider._get_embeddings(["synthetic query"], input_type="query")


def test_snapshot_provider_refuses_raw_connection_without_owner_proof(tmp_path: Path) -> None:
    """A raw handle cannot be certified by statting a pathname after publication."""
    from contextlib import closing

    from tests.infra.vector_archive import seed_vector_archive

    seed_vector_archive(tmp_path, [])
    with closing(sqlite3.connect(":memory:")) as connection:
        connection.execute("ATTACH DATABASE ? AS archive_index", (str(tmp_path / "index.db"),))
        with pytest.raises(SqliteVecError):
            SqliteVecProvider.from_vector_read_snapshot(voyage_key=None, connection=connection, model="voyage-4")
        assert connection.execute("SELECT 1").fetchone() == (1,)


def test_snapshot_admission_refuses_index_replacement_and_closes_its_handle(tmp_path: Path) -> None:
    """Recording a fresh post-attach identity would certify the replaced file."""
    import shutil

    from polylogue.storage.embeddings.identity import EmbeddingRecipe
    from polylogue.storage.search_providers.sqlite_vec_runtime import open_vector_read_snapshot
    from tests.infra.vector_archive import seed_vector_archive

    seed_vector_archive(tmp_path, [("seed", "m1", "Synthetic selected admission prose.", [1.0] + [0.0] * 1023)])
    acquired: list[sqlite3.Connection] = []

    def replace_selected_index(connection: sqlite3.Connection) -> None:
        acquired.append(connection)
        (tmp_path / "index.db").rename(tmp_path / "prior-index.db")
        shutil.copyfile(tmp_path / "prior-index.db", tmp_path / "index.db")

    with pytest.raises(SqliteVecError):
        open_vector_read_snapshot(
            embeddings_path=tmp_path / "embeddings.db",
            index_path=tmp_path / "index.db",
            recipe=EmbeddingRecipe.current(model="voyage-4", dimensions=1024),
            configure_connection=replace_selected_index,
        )
    assert len(acquired) == 1
    with pytest.raises(sqlite3.ProgrammingError):
        acquired[0].execute("SELECT 1")
