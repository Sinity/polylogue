"""Tests for SqliteVecProvider operations (API calls, upsert, query).

Complements test_sqlite_vec.py which covers filtering and serialization.
These tests focus on:
- Voyage API interaction (mocked httpx)
- Embedding batch processing
- Upsert database operations
- Query execution paths
- Error handling and retry logic
"""

from __future__ import annotations

import struct
from unittest.mock import MagicMock, patch, call

import pytest

from polylogue.storage.store import MessageRecord


def make_message(
    message_id: str = "msg-1",
    conversation_id: str = "conv-1",
    role: str = "user",
    text: str = "This is a sufficiently long test message for embedding.",
    content_hash: str = "hash-1",
    provider_meta: dict | None = None,
) -> MessageRecord:
    """Create a test MessageRecord."""
    if provider_meta is None:
        provider_meta = {"provider_name": "test-provider"}
    return MessageRecord(
        message_id=message_id,
        conversation_id=conversation_id,
        role=role,
        text=text,
        content_hash=content_hash,
        provider_meta=provider_meta,
        version=1,
    )


@pytest.fixture
def provider_cls():
    """Import SqliteVecProvider class."""
    from polylogue.storage.search_providers.sqlite_vec import SqliteVecProvider

    return SqliteVecProvider


@pytest.fixture
def mock_provider(tmp_path, provider_cls):
    """Create a SqliteVecProvider with mocked internals."""
    provider = object.__new__(provider_cls)
    provider.db_path = tmp_path / "test.db"
    provider.voyage_key = "test-voyage-key"
    provider.model = "voyage-4"
    provider.dimension = 1024
    provider._vec_available = None
    return provider


# =============================================================================
# _get_embeddings
# =============================================================================


class TestGetEmbeddings:
    """Test Voyage API interaction logic."""

    def test_empty_texts_returns_empty(self, mock_provider):
        """Empty input should short-circuit without API call."""
        result = mock_provider._get_embeddings([])
        assert result == []

    def test_single_text_calls_api(self, mock_provider):
        """Single text should make one API call and return embedding."""
        fake_response = MagicMock()
        fake_response.status_code = 200
        fake_response.json.return_value = {"data": [{"embedding": [0.1, 0.2, 0.3]}]}
        fake_response.raise_for_status = MagicMock()

        with patch("httpx.Client") as MockClient:
            mock_client = MagicMock()
            mock_client.post.return_value = fake_response
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            MockClient.return_value = mock_client

            result = mock_provider._get_embeddings(["test text"])
            assert len(result) == 1
            assert result[0] == [0.1, 0.2, 0.3]

    def test_batching_splits_large_input(self, mock_provider):
        """Texts exceeding BATCH_SIZE should be split into multiple API calls."""
        from polylogue.storage.search_providers.sqlite_vec import BATCH_SIZE

        texts = [
            f"text {i} is a long enough message for embedding purposes"
            for i in range(BATCH_SIZE + 10)
        ]

        call_count = 0

        def fake_post(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            batch_size = len(kwargs.get("json", {}).get("input", []))
            fake_resp = MagicMock()
            fake_resp.json.return_value = {
                "data": [{"embedding": [0.1] * 3} for _ in range(batch_size)]
            }
            fake_resp.raise_for_status = MagicMock()
            return fake_resp

        with patch("httpx.Client") as MockClient:
            mock_client = MagicMock()
            mock_client.post = fake_post
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            MockClient.return_value = mock_client

            # Also mock time.sleep to not slow test
            with patch("time.sleep"):
                result = mock_provider._get_embeddings(texts)

        assert call_count == 2  # One batch of BATCH_SIZE, one of 10
        assert len(result) == BATCH_SIZE + 10

    def test_http_error_raises_sqlite_vec_error(self, mock_provider):
        """HTTP errors should be caught and re-raised as SqliteVecError."""
        import httpx

        from polylogue.storage.search_providers.sqlite_vec import SqliteVecError

        mock_response = MagicMock()
        mock_response.status_code = 429
        error = httpx.HTTPStatusError(
            "rate limited", request=MagicMock(), response=mock_response
        )

        with patch("httpx.Client") as MockClient:
            mock_client = MagicMock()
            mock_client.post.side_effect = error
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            MockClient.return_value = mock_client

            with pytest.raises(SqliteVecError, match="HTTP 429"):
                mock_provider._get_embeddings(["test text"])

    def test_http_timeout_error_raises_sqlite_vec_error(self, mock_provider):
        """HTTP timeout errors should be caught and re-raised."""
        import httpx

        from polylogue.storage.search_providers.sqlite_vec import SqliteVecError

        error = httpx.TimeoutException("Connection timed out")

        with patch("httpx.Client") as MockClient:
            mock_client = MagicMock()
            mock_client.post.side_effect = error
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            MockClient.return_value = mock_client

            with pytest.raises(SqliteVecError, match="TimeoutException"):
                mock_provider._get_embeddings(["test text"])

    def test_api_key_not_leaked_in_error(self, mock_provider):
        """Error messages should not contain the API key."""
        import httpx

        from polylogue.storage.search_providers.sqlite_vec import SqliteVecError

        mock_response = MagicMock()
        mock_response.status_code = 401
        error = httpx.HTTPStatusError(
            f"Unauthorized: Bearer {mock_provider.voyage_key}",
            request=MagicMock(),
            response=mock_response,
        )

        with patch("httpx.Client") as MockClient:
            mock_client = MagicMock()
            mock_client.post.side_effect = error
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            MockClient.return_value = mock_client

            with pytest.raises(SqliteVecError) as exc_info:
                mock_provider._get_embeddings(["test text"])

            # The error message should NOT contain the API key
            assert mock_provider.voyage_key not in str(exc_info.value)

    def test_dimension_reduction_sent_in_payload(self, mock_provider):
        """Non-default dimension should add output_dimension to API payload."""
        mock_provider.dimension = 512  # Non-default

        fake_response = MagicMock()
        fake_response.json.return_value = {
            "data": [{"embedding": [0.1] * 512}]
        }
        fake_response.raise_for_status = MagicMock()

        captured_payload = {}

        def capture_post(*args, **kwargs):
            captured_payload.update(kwargs.get("json", {}))
            return fake_response

        with patch("httpx.Client") as MockClient:
            mock_client = MagicMock()
            mock_client.post = capture_post
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            MockClient.return_value = mock_client

            mock_provider._get_embeddings(["test text"])

            # Check that output_dimension was in the payload
            assert captured_payload.get("output_dimension") == 512

    def test_default_dimension_omitted_from_payload(self, mock_provider):
        """Default dimension should NOT add output_dimension to payload."""
        from polylogue.storage.search_providers.sqlite_vec import DEFAULT_DIMENSION

        mock_provider.dimension = DEFAULT_DIMENSION

        fake_response = MagicMock()
        fake_response.json.return_value = {
            "data": [{"embedding": [0.1] * DEFAULT_DIMENSION}]
        }
        fake_response.raise_for_status = MagicMock()

        captured_payload = {}

        def capture_post(*args, **kwargs):
            captured_payload.update(kwargs.get("json", {}))
            return fake_response

        with patch("httpx.Client") as MockClient:
            mock_client = MagicMock()
            mock_client.post = capture_post
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            MockClient.return_value = mock_client

            mock_provider._get_embeddings(["test text"])

            # Check that output_dimension was NOT in the payload
            assert "output_dimension" not in captured_payload

    def test_query_input_type_sent_in_payload(self, mock_provider):
        """Query mode should send input_type='query'."""
        fake_response = MagicMock()
        fake_response.json.return_value = {"data": [{"embedding": [0.1, 0.2]}]}
        fake_response.raise_for_status = MagicMock()

        captured_payload = {}

        def capture_post(*args, **kwargs):
            captured_payload.update(kwargs.get("json", {}))
            return fake_response

        with patch("httpx.Client") as MockClient:
            mock_client = MagicMock()
            mock_client.post = capture_post
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            MockClient.return_value = mock_client

            mock_provider._get_embeddings(["test text"], input_type="query")

            assert captured_payload.get("input_type") == "query"

    def test_document_input_type_sent_in_payload(self, mock_provider):
        """Document mode should send input_type='document'."""
        fake_response = MagicMock()
        fake_response.json.return_value = {"data": [{"embedding": [0.1, 0.2]}]}
        fake_response.raise_for_status = MagicMock()

        captured_payload = {}

        def capture_post(*args, **kwargs):
            captured_payload.update(kwargs.get("json", {}))
            return fake_response

        with patch("httpx.Client") as MockClient:
            mock_client = MagicMock()
            mock_client.post = capture_post
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            MockClient.return_value = mock_client

            mock_provider._get_embeddings(["test text"], input_type="document")

            assert captured_payload.get("input_type") == "document"

    def test_authorization_header_sent(self, mock_provider):
        """Authorization header should contain the API key."""
        fake_response = MagicMock()
        fake_response.json.return_value = {"data": [{"embedding": [0.1]}]}
        fake_response.raise_for_status = MagicMock()

        captured_headers = {}

        def capture_post(*args, **kwargs):
            captured_headers.update(kwargs.get("headers", {}))
            return fake_response

        with patch("httpx.Client") as MockClient:
            mock_client = MagicMock()
            mock_client.post = capture_post
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            MockClient.return_value = mock_client

            mock_provider._get_embeddings(["test text"])

            assert (
                captured_headers.get("Authorization")
                == f"Bearer {mock_provider.voyage_key}"
            )


# =============================================================================
# _should_embed_message edge cases (complement to test_sqlite_vec.py)
# =============================================================================


class TestShouldEmbedMessageEdgeCases:
    """Test message filtering edge cases not in test_sqlite_vec.py."""

    def test_tool_result_status_ok_filtered(self, mock_provider):
        msg = make_message(role="tool_result", text="ok")
        assert not mock_provider._should_embed_message(msg)

    def test_tool_result_status_success_filtered(self, mock_provider):
        msg = make_message(role="tool_result", text="success")
        assert not mock_provider._should_embed_message(msg)

    def test_tool_result_status_done_filtered(self, mock_provider):
        msg = make_message(role="tool_result", text="done")
        assert not mock_provider._should_embed_message(msg)

    def test_tool_result_status_error_filtered(self, mock_provider):
        msg = make_message(role="tool_result", text="error")
        assert not mock_provider._should_embed_message(msg)

    def test_tool_result_status_failed_filtered(self, mock_provider):
        msg = make_message(role="tool_result", text="failed")
        assert not mock_provider._should_embed_message(msg)

    def test_tool_result_status_case_insensitive(self, mock_provider):
        """Status check should be case-insensitive."""
        msg = make_message(role="tool_result", text="SUCCESS")
        assert not mock_provider._should_embed_message(msg)

    def test_tool_result_with_real_content_accepted(self, mock_provider):
        msg = make_message(
            role="tool_result",
            text="File contents: def hello(): print('world')",
        )
        assert mock_provider._should_embed_message(msg)

    def test_tool_result_status_with_trailing_space_filtered(self, mock_provider):
        """Status check should strip whitespace."""
        msg = make_message(role="tool_result", text="ok  ")
        assert not mock_provider._should_embed_message(msg)

    def test_whitespace_only_text_filtered(self, mock_provider):
        msg = make_message(text="   \n\t  ")
        assert not mock_provider._should_embed_message(msg)

    def test_none_text_filtered(self, mock_provider):
        msg = make_message(text="")
        assert not mock_provider._should_embed_message(msg)

    def test_exactly_20_chars_accepted(self, mock_provider):
        """Exactly 20 chars should pass (not < 20)."""
        msg = make_message(text="12345678901234567890")  # exactly 20
        assert mock_provider._should_embed_message(msg)

    def test_19_chars_filtered(self, mock_provider):
        msg = make_message(text="1234567890123456789")  # 19 chars
        assert not mock_provider._should_embed_message(msg)


# =============================================================================
# upsert flow with mocked API + DB
# =============================================================================


class TestUpsertFlow:
    """Test the upsert operation: API call -> DB insert."""

    def test_upsert_empty_messages_noop(self, mock_provider):
        """Upsert with empty list should be a no-op."""
        mock_provider.upsert("conv-1", [])
        # No errors, just returns early

    def test_upsert_no_embeddable_messages_noop(self, mock_provider):
        """Upsert with only non-embeddable messages should be a no-op."""
        mock_provider._ensure_vec_available = MagicMock()
        msg = make_message(text="short")  # Too short to embed
        mock_provider.upsert("conv-1", [msg])
        # Should have called _ensure_vec_available but not try to embed
        mock_provider._ensure_vec_available.assert_called_once()

    def test_upsert_calls_ensure_vec_available(self, mock_provider):
        """Upsert should verify vec is available."""
        msg = make_message()
        mock_provider._ensure_vec_available = MagicMock()
        mock_provider._get_embeddings = MagicMock(return_value=[[0.1, 0.2]])
        mock_provider._get_connection = MagicMock()
        mock_conn = MagicMock()
        mock_provider._get_connection.return_value = mock_conn

        mock_provider.upsert("conv-1", [msg])

        mock_provider._ensure_vec_available.assert_called_once()

    def test_upsert_filters_embeddable_messages(self, mock_provider):
        """Upsert should only embed messages that pass _should_embed_message."""
        messages = [
            make_message(message_id="m1", text="This is a long embeddable message."),
            make_message(message_id="m2", text="short"),  # Too short
            make_message(message_id="m3", text="This is another long embeddable message."),
        ]

        mock_provider._ensure_vec_available = MagicMock()

        embeddings_called_with = []

        def capture_embeddings(texts, **kwargs):
            embeddings_called_with.extend(texts)
            return [[0.1] * 1024 for _ in texts]

        mock_provider._get_embeddings = capture_embeddings
        mock_provider._get_connection = MagicMock()
        mock_conn = MagicMock()
        mock_provider._get_connection.return_value = mock_conn

        mock_provider.upsert("conv-1", messages)

        # Should only embed the two long messages
        assert len(embeddings_called_with) == 2

    def test_upsert_calls_db_operations(self, mock_provider):
        """Upsert should execute DELETE, INSERT for embeddings and metadata."""
        msg = make_message()

        mock_provider._ensure_vec_available = MagicMock()
        mock_provider._get_embeddings = MagicMock(return_value=[[0.1, 0.2]])
        mock_provider._get_connection = MagicMock()
        mock_conn = MagicMock()
        mock_conn.execute = MagicMock(return_value=MagicMock())
        mock_conn.commit = MagicMock()
        mock_provider._get_connection.return_value = mock_conn

        mock_provider.upsert("conv-1", [msg])

        # Should have called execute multiple times (DELETE, INSERT embedding, INSERT metadata, INSERT status)
        assert mock_conn.execute.call_count >= 3
        mock_conn.commit.assert_called_once()

    def test_upsert_uses_provider_name_from_metadata(self, mock_provider):
        """Upsert should extract provider_name from message metadata."""
        msg = make_message(provider_meta={"provider_name": "claude"})

        mock_provider._ensure_vec_available = MagicMock()
        mock_provider._get_embeddings = MagicMock(return_value=[[0.1, 0.2]])
        mock_provider._get_connection = MagicMock()
        mock_conn = MagicMock()

        # Capture INSERT calls to check provider_name
        insert_calls = []

        def capture_execute(sql, params=None):
            if "INSERT INTO message_embeddings" in sql:
                insert_calls.append(params)
            return MagicMock()

        mock_conn.execute = capture_execute
        mock_conn.commit = MagicMock()
        mock_provider._get_connection.return_value = mock_conn

        mock_provider.upsert("conv-1", [msg])

        # Should have called INSERT with provider_name='claude'
        assert any("claude" in str(call) for call in insert_calls)

    def test_upsert_sanitizes_unknown_provider_name(self, mock_provider):
        """Upsert should use 'unknown' when provider_meta lacks provider_name."""
        msg = make_message(provider_meta={})

        mock_provider._ensure_vec_available = MagicMock()
        mock_provider._get_embeddings = MagicMock(return_value=[[0.1, 0.2]])
        mock_provider._get_connection = MagicMock()
        mock_conn = MagicMock()

        insert_calls = []

        def capture_execute(sql, params=None):
            if "INSERT INTO message_embeddings" in sql:
                insert_calls.append(params)
            return MagicMock()

        mock_conn.execute = capture_execute
        mock_conn.commit = MagicMock()
        mock_provider._get_connection.return_value = mock_conn

        mock_provider.upsert("conv-1", [msg])

        # Should have called INSERT with provider_name='unknown'
        assert any("unknown" in str(call) for call in insert_calls)

    def test_upsert_closes_connection(self, mock_provider):
        """Upsert should close the connection in finally block."""
        msg = make_message()

        mock_provider._ensure_vec_available = MagicMock()
        mock_provider._get_embeddings = MagicMock(return_value=[[0.1, 0.2]])
        mock_provider._get_connection = MagicMock()
        mock_conn = MagicMock()
        mock_conn.execute = MagicMock(return_value=MagicMock())
        mock_conn.commit = MagicMock()
        mock_provider._get_connection.return_value = mock_conn

        mock_provider.upsert("conv-1", [msg])

        mock_conn.close.assert_called_once()

    def test_upsert_closes_connection_on_error(self, mock_provider):
        """Upsert should close connection even if embedding fails."""
        msg = make_message()

        mock_provider._ensure_vec_available = MagicMock()
        mock_provider._get_embeddings = MagicMock(
            side_effect=RuntimeError("API error")
        )

        with pytest.raises(RuntimeError):
            mock_provider.upsert("conv-1", [msg])


# =============================================================================
# query/query_by_provider execution paths
# =============================================================================


class TestQueryExecutionPaths:
    """Test query and query_by_provider methods."""

    def test_query_calls_ensure_vec_available(self, mock_provider):
        """Query should verify vec is available."""
        mock_provider._ensure_vec_available = MagicMock()
        mock_provider._get_embeddings = MagicMock(return_value=[[0.1, 0.2]])
        mock_provider._get_connection = MagicMock()
        mock_conn = MagicMock()
        mock_conn.execute = MagicMock(
            return_value=MagicMock(fetchall=MagicMock(return_value=[]))
        )
        mock_provider._get_connection.return_value = mock_conn

        mock_provider.query("search text")

        mock_provider._ensure_vec_available.assert_called_once()

    def test_query_generates_query_embedding(self, mock_provider):
        """Query should generate embedding for the search text."""
        embeddings_called_with = []

        def capture_embeddings(texts, input_type=None):
            embeddings_called_with.append((texts, input_type))
            return [[0.1, 0.2]]

        mock_provider._ensure_vec_available = MagicMock()
        mock_provider._get_embeddings = capture_embeddings
        mock_provider._get_connection = MagicMock()
        mock_conn = MagicMock()
        mock_conn.execute = MagicMock(
            return_value=MagicMock(fetchall=MagicMock(return_value=[]))
        )
        mock_provider._get_connection.return_value = mock_conn

        mock_provider.query("search text")

        # Should have been called with input_type='query'
        assert embeddings_called_with[0] == (["search text"], "query")

    def test_query_empty_embedding_returns_empty(self, mock_provider):
        """Query should return empty if embedding generation fails."""
        mock_provider._ensure_vec_available = MagicMock()
        mock_provider._get_embeddings = MagicMock(return_value=[])
        mock_provider._get_connection = MagicMock()

        result = mock_provider.query("search text")

        assert result == []
        # Should not try to connect if no embedding
        mock_provider._get_connection.assert_not_called()

    def test_query_returns_message_ids_and_distances(self, mock_provider):
        """Query should return list of (message_id, distance) tuples."""
        import sqlite3

        mock_provider._ensure_vec_available = MagicMock()
        mock_provider._get_embeddings = MagicMock(return_value=[[0.1, 0.2]])
        mock_provider._get_connection = MagicMock()
        mock_conn = MagicMock()

        # Create mock Row objects that support both dict-like and tuple access
        mock_row_1 = MagicMock(spec=sqlite3.Row)
        mock_row_1.__getitem__.side_effect = lambda k: "msg-1" if k == "message_id" else 0.5
        mock_row_1.message_id = "msg-1"
        mock_row_1.distance = 0.5

        mock_row_2 = MagicMock(spec=sqlite3.Row)
        mock_row_2.__getitem__.side_effect = lambda k: "msg-2" if k == "message_id" else 0.7
        mock_row_2.message_id = "msg-2"
        mock_row_2.distance = 0.7

        mock_conn.execute = MagicMock(
            return_value=MagicMock(fetchall=MagicMock(return_value=[mock_row_1, mock_row_2]))
        )
        mock_provider._get_connection.return_value = mock_conn

        result = mock_provider.query("search text", limit=10)

        assert len(result) == 2
        assert result[0] == ("msg-1", 0.5)
        assert result[1] == ("msg-2", 0.7)

    def test_query_closes_connection(self, mock_provider):
        """Query should close connection in finally block."""
        mock_provider._ensure_vec_available = MagicMock()
        mock_provider._get_embeddings = MagicMock(return_value=[[0.1, 0.2]])
        mock_provider._get_connection = MagicMock()
        mock_conn = MagicMock()
        mock_conn.execute = MagicMock(
            return_value=MagicMock(fetchall=MagicMock(return_value=[]))
        )
        mock_provider._get_connection.return_value = mock_conn

        mock_provider.query("search text")

        mock_conn.close.assert_called_once()

    def test_query_by_provider_filters_by_provider(self, mock_provider):
        """query_by_provider should filter results by provider name."""
        import sqlite3

        mock_provider._ensure_vec_available = MagicMock()
        mock_provider._get_embeddings = MagicMock(return_value=[[0.1, 0.2]])
        mock_provider._get_connection = MagicMock()
        mock_conn = MagicMock()

        executed_queries = []

        def capture_execute(sql, params=None):
            executed_queries.append((sql, params))
            return MagicMock(fetchall=MagicMock(return_value=[]))

        mock_conn.execute = capture_execute
        mock_provider._get_connection.return_value = mock_conn

        mock_provider.query_by_provider("search text", provider="claude", limit=5)

        # Should have executed a query with the provider filter
        assert any("provider_name = ?" in sql for sql, _ in executed_queries)
        # Should have passed "claude" as a parameter
        assert any("claude" in str(params) for _, params in executed_queries)

    def test_query_by_provider_closes_connection(self, mock_provider):
        """query_by_provider should close connection in finally block."""
        mock_provider._ensure_vec_available = MagicMock()
        mock_provider._get_embeddings = MagicMock(return_value=[[0.1, 0.2]])
        mock_provider._get_connection = MagicMock()
        mock_conn = MagicMock()
        mock_conn.execute = MagicMock(
            return_value=MagicMock(fetchall=MagicMock(return_value=[]))
        )
        mock_provider._get_connection.return_value = mock_conn

        mock_provider.query_by_provider("search text", provider="claude")

        mock_conn.close.assert_called_once()

    def test_query_by_provider_empty_embedding_returns_empty(self, mock_provider):
        """query_by_provider should return empty if embedding generation fails."""
        mock_provider._ensure_vec_available = MagicMock()
        mock_provider._get_embeddings = MagicMock(return_value=[])
        mock_provider._get_connection = MagicMock()

        result = mock_provider.query_by_provider("search text", provider="claude")

        assert result == []
        mock_provider._get_connection.assert_not_called()


# =============================================================================
# get_embedding_stats
# =============================================================================


class TestGetEmbeddingStats:
    """Test embedding statistics retrieval."""

    def test_stats_with_both_tables_present(self, mock_provider):
        """Should return counts from both tables when they exist."""
        mock_provider._get_connection = MagicMock()
        mock_conn = MagicMock()

        def execute_sql(sql, params=None):
            if "message_embeddings" in sql:
                return MagicMock(fetchone=MagicMock(return_value=[42]))
            elif "embedding_status" in sql:
                return MagicMock(fetchone=MagicMock(return_value=[5]))
            return MagicMock()

        mock_conn.execute = execute_sql
        mock_provider._get_connection.return_value = mock_conn

        stats = mock_provider.get_embedding_stats()

        assert stats["embedded_messages"] == 42
        assert stats["pending_conversations"] == 5

    def test_stats_with_missing_tables(self, mock_provider):
        """Should gracefully handle missing tables."""
        mock_provider._get_connection = MagicMock()
        mock_conn = MagicMock()
        mock_conn.execute = MagicMock(side_effect=Exception("Table not found"))
        mock_provider._get_connection.return_value = mock_conn

        stats = mock_provider.get_embedding_stats()

        assert stats["embedded_messages"] == 0
        assert stats["pending_conversations"] == 0

    def test_stats_closes_connection(self, mock_provider):
        """Should close connection in finally block."""
        mock_provider._get_connection = MagicMock()
        mock_conn = MagicMock()
        mock_conn.execute = MagicMock(return_value=MagicMock(fetchone=MagicMock(return_value=[0])))
        mock_provider._get_connection.return_value = mock_conn

        mock_provider.get_embedding_stats()

        mock_conn.close.assert_called_once()

    def test_stats_returns_dict_with_expected_keys(self, mock_provider):
        """Stats should return dict with specific keys."""
        mock_provider._get_connection = MagicMock()
        mock_conn = MagicMock()
        mock_conn.execute = MagicMock(return_value=MagicMock(fetchone=MagicMock(return_value=[0])))
        mock_provider._get_connection.return_value = mock_conn

        stats = mock_provider.get_embedding_stats()

        assert "embedded_messages" in stats
        assert "pending_conversations" in stats
        assert isinstance(stats, dict)


# =============================================================================
# _ensure_vec_available
# =============================================================================


class TestEnsureVecAvailable:
    """Test sqlite-vec availability checking."""

    def test_raises_when_vec_not_available(self, mock_provider):
        from polylogue.storage.search_providers.sqlite_vec import SqliteVecError

        mock_provider._vec_available = False
        with pytest.raises(SqliteVecError, match="not available"):
            mock_provider._ensure_vec_available()

    def test_raises_with_helpful_message(self, mock_provider):
        from polylogue.storage.search_providers.sqlite_vec import SqliteVecError

        mock_provider._vec_available = False
        with pytest.raises(SqliteVecError, match="pip install"):
            mock_provider._ensure_vec_available()

    def test_probes_on_first_call(self, mock_provider):
        """First call with _vec_available=None should probe via _get_connection."""
        from polylogue.storage.search_providers.sqlite_vec import SqliteVecError

        mock_provider._vec_available = None

        # Mock _get_connection to set _vec_available to False
        mock_conn = MagicMock()
        mock_conn.close = MagicMock()

        def mock_get_conn():
            mock_provider._vec_available = False
            return mock_conn

        mock_provider._get_connection = mock_get_conn

        with pytest.raises(SqliteVecError, match="not available"):
            mock_provider._ensure_vec_available()

        # Should have been called once to probe
        mock_conn.close.assert_called_once()

    def test_skips_probe_when_already_unavailable(self, mock_provider):
        """If _vec_available is False, should not probe again."""
        from polylogue.storage.search_providers.sqlite_vec import SqliteVecError

        mock_provider._vec_available = False
        mock_provider._get_connection = MagicMock()

        with pytest.raises(SqliteVecError):
            mock_provider._ensure_vec_available()

        # Should not have called _get_connection
        mock_provider._get_connection.assert_not_called()

    def test_skips_probe_when_already_available(self, mock_provider):
        """If _vec_available is True, should not raise."""
        mock_provider._vec_available = True
        mock_provider._get_connection = MagicMock()

        # Should not raise
        mock_provider._ensure_vec_available()

        # Should not have called _get_connection
        mock_provider._get_connection.assert_not_called()

    def test_stores_vec_available_after_probe(self, mock_provider):
        """After first call, _vec_available should be cached."""
        mock_provider._vec_available = None

        mock_conn = MagicMock()
        mock_conn.close = MagicMock()

        def mock_get_conn():
            mock_provider._vec_available = True
            return mock_conn

        mock_provider._get_connection = mock_get_conn

        mock_provider._ensure_vec_available()

        # Should have set _vec_available to True (from mock_get_conn)
        assert mock_provider._vec_available is True
