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

from polylogue.lib.roles import Role
from polylogue.storage.runtime import MessageRecord
from polylogue.storage.search_providers.sqlite_vec import SqliteVecProvider
from polylogue.types import ContentHash, ConversationId, MessageId

Embedding: TypeAlias = list[float]


class EmbeddingFetcher(Protocol):
    def __call__(self, texts: list[str], input_type: str = "document") -> list[Embedding]: ...


class MutableSqliteVecProvider(SqliteVecProvider):
    _ensure_vec_available: Callable[[], None]
    _ensure_tables: Callable[[], None]
    _get_embeddings: EmbeddingFetcher
    _get_connection: Callable[[], sqlite3.Connection]


def make_message(
    message_id: str = "msg-1",
    conversation_id: str = "conv-1",
    role: str = "user",
    text: str = "This is a sufficiently long test message for embedding.",
    content_hash: str = "hash-1",
    provider_name: str = "test-provider",
) -> MessageRecord:
    return MessageRecord(
        message_id=MessageId(message_id),
        conversation_id=ConversationId(conversation_id),
        role=Role.normalize(role),
        text=text,
        content_hash=ContentHash(content_hash),
        provider_name=provider_name,
        version=1,
    )


@pytest.fixture
def mock_provider(tmp_path: Path) -> MutableSqliteVecProvider:
    provider = MutableSqliteVecProvider(voyage_key="test-voyage-key", db_path=tmp_path / "test.db", model="voyage-4")
    provider.dimension = 1024
    provider._vec_available = None
    provider._tables_ensured = True
    return provider


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

    assert mock_provider.voyage_key not in str(exc_info.value)


@pytest.mark.parametrize(
    ("role", "text", "should_embed"),
    [
        ("tool_result", "ok", False),
        ("tool_result", "SUCCESS", False),
        ("tool_result", "File contents: def hello(): print('world')", True),
        ("system", "This is a sufficiently long system message.", False),
        ("user", "   \n\t  ", False),
        ("user", "1234567890123456789", False),
        ("user", "12345678901234567890", True),
    ],
)
def test_should_embed_message_contract(
    mock_provider: MutableSqliteVecProvider,
    role: str,
    text: str,
    should_embed: bool,
) -> None:
    """Only semantically useful messages should be embedded."""
    assert mock_provider._should_embed_message(make_message(role=role, text=text)) is should_embed


@pytest.mark.parametrize(
    ("messages", "should_ensure"),
    [([], False), ([make_message(text="short")], True)],
    ids=["empty", "no-embeddable"],
)
def test_upsert_noop_contract(
    mock_provider: MutableSqliteVecProvider,
    messages: list[MessageRecord],
    should_ensure: bool,
) -> None:
    """Upsert should short-circuit on empty or non-embeddable input."""
    ensure_vec_available = MagicMock()
    ensure_tables = MagicMock()
    get_connection = MagicMock()
    mock_provider._ensure_vec_available = ensure_vec_available
    mock_provider._ensure_tables = ensure_tables
    mock_provider._get_connection = get_connection

    mock_provider.upsert("conv-1", messages)

    if should_ensure:
        ensure_vec_available.assert_called_once()
        ensure_tables.assert_called_once()
    else:
        ensure_vec_available.assert_not_called()
        ensure_tables.assert_not_called()
    get_connection.assert_not_called()


@pytest.mark.parametrize(
    ("provider_name", "expected_provider"),
    [("claude-ai", "claude-ai"), (None, "unknown")],
    ids=["provider-row", "missing-provider-row"],
)
def test_upsert_persistence_contract(
    mock_provider: MutableSqliteVecProvider,
    provider_name: str | None,
    expected_provider: str,
) -> None:
    """Upsert must filter messages, persist embeddings, and stamp provider metadata."""
    messages = [
        make_message(message_id="m1", text="This is a long embeddable message."),
        make_message(message_id="m2", text="short"),
        make_message(message_id="m3", text="This is another long embeddable message."),
    ]
    embeddings_called_with: list[str] = []
    insert_calls: list[tuple[str, tuple[object, ...] | None]] = []
    mock_provider._ensure_vec_available = MagicMock()
    mock_provider._ensure_tables = MagicMock()

    def capture_embeddings(texts: list[str], input_type: str = "document") -> list[Embedding]:
        del input_type
        embeddings_called_with.extend(texts)
        return [[0.1, 0.2] for _ in texts]

    def capture_execute(sql: str, params: tuple[object, ...] | None = None) -> MagicMock:
        insert_calls.append((sql, params))
        cursor = MagicMock()
        if "SELECT provider_name FROM conversations" in sql:
            cursor.fetchone.return_value = (provider_name,) if provider_name is not None else None
        return cursor

    mock_provider._get_embeddings = capture_embeddings
    connection = MagicMock()
    connection.execute = capture_execute
    connection.commit = MagicMock()
    connection.close = MagicMock()
    mock_provider._get_connection = MagicMock(return_value=connection)

    mock_provider.upsert("conv-1", messages)

    assert embeddings_called_with == [
        "This is a long embeddable message.",
        "This is another long embeddable message.",
    ]
    assert connection.commit.called
    assert connection.close.called
    embedding_inserts = [params for sql, params in insert_calls if "INSERT INTO message_embeddings" in sql]
    assert len(embedding_inserts) == 2
    assert all(expected_provider in str(params) for params in embedding_inserts)
    status_updates = [sql for sql, _ in insert_calls if "INSERT INTO embedding_status" in sql]
    assert len(status_updates) == 1


def test_upsert_closes_connection_on_embedding_error(mock_provider: MutableSqliteVecProvider) -> None:
    """Connection cleanup must happen even when embedding generation fails."""
    mock_provider._ensure_vec_available = MagicMock()
    mock_provider._ensure_tables = MagicMock()
    mock_provider._get_embeddings = MagicMock(side_effect=RuntimeError("API error"))

    with pytest.raises(RuntimeError):
        mock_provider.upsert("conv-1", [make_message()])


@pytest.mark.parametrize(
    ("method_name", "provider", "embedding_result"),
    [
        ("query", None, [[0.1, 0.2]]),
        ("query_by_provider", "claude-ai", [[0.1, 0.2]]),
        ("query", None, []),
        ("query_by_provider", "claude-ai", []),
    ],
    ids=["query", "query-by-provider", "query-empty", "query-by-provider-empty"],
)
def test_query_route_contract(
    mock_provider: MutableSqliteVecProvider,
    method_name: str,
    provider: str | None,
    embedding_result: list[Embedding],
) -> None:
    """Query methods must generate query embeddings, optionally filter by provider, and close connections."""
    mock_provider._ensure_vec_available = MagicMock()
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
        return MagicMock(fetchall=MagicMock(return_value=[row_1, row_2]))

    mock_provider._get_embeddings = capture_embeddings
    connection = MagicMock()
    connection.execute = capture_execute
    connection.close = MagicMock()
    mock_provider._get_connection = MagicMock(return_value=connection)

    if method_name == "query":
        result = mock_provider.query("search text", limit=10)
    else:
        assert provider is not None
        result = mock_provider.query_by_provider("search text", provider=provider, limit=10)

    assert embedding_calls == [(["search text"], "query")]
    if embedding_result:
        assert result == [("msg-1", 0.5), ("msg-2", 0.7)]
        assert connection.close.called
        if provider is None:
            assert all("provider_name = ?" not in sql for sql, _ in executed_queries)
        else:
            assert any("provider_name = ?" in sql for sql, _ in executed_queries)
            assert any(provider in str(params) for _, params in executed_queries)
    else:
        assert result == []
        mock_provider._get_connection.assert_not_called()


@pytest.mark.parametrize(
    ("msg_count", "pending", "operational_error"),
    [(42, 5, False), (0, 0, False), (0, 0, True)],
    ids=["counts", "zeros", "missing-tables"],
)
def test_get_embedding_stats_contract(
    mock_provider: MutableSqliteVecProvider,
    msg_count: int,
    pending: int,
    operational_error: bool,
) -> None:
    """Stats queries must tolerate absent tables and always close the connection."""
    connection = MagicMock()

    def execute(sql: str, params: tuple[object, ...] | None = None) -> MagicMock:
        del params
        if operational_error:
            raise sqlite3.OperationalError("Table not found")
        if "sqlite_master" in sql and "conversations" in sql:
            return MagicMock(fetchone=MagicMock(return_value=None))
        if "message_embeddings" in sql:
            return MagicMock(fetchone=MagicMock(return_value=[msg_count]))
        if "embedding_status" in sql:
            return MagicMock(fetchone=MagicMock(return_value=[pending]))
        return MagicMock()

    connection.execute = execute
    connection.close = MagicMock()
    mock_provider._get_connection = MagicMock(return_value=connection)

    stats = mock_provider.get_embedding_stats()

    assert stats == {"embedded_messages": msg_count, "pending_conversations": pending}
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
