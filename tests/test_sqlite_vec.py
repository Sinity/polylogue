"""Tests for SqliteVecProvider with mocked Voyage API."""

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


# MERGED FROM test_sqlite_vec_operations.py


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
    provider._tables_ensured = True
    return provider


# =============================================================================
# _get_embeddings
# =============================================================================


@pytest.mark.slow
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

    @pytest.mark.parametrize(
        "status_code,error_type,error_msg,match_pattern",
        [
            (429, "HTTPStatusError", "rate limited", "HTTP 429"),
        ],
    )
    def test_http_error_raises_sqlite_vec_error(
        self, mock_provider, status_code, error_type, error_msg, match_pattern
    ):
        """HTTP errors should be caught and re-raised as SqliteVecError."""
        import httpx

        from polylogue.storage.search_providers.sqlite_vec import SqliteVecError

        mock_response = MagicMock()
        mock_response.status_code = status_code
        error = httpx.HTTPStatusError(
            error_msg, request=MagicMock(), response=mock_response
        )

        with patch("httpx.Client") as MockClient:
            mock_client = MagicMock()
            mock_client.post.side_effect = error
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            MockClient.return_value = mock_client

            with pytest.raises(SqliteVecError, match=match_pattern):
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

    @pytest.mark.parametrize(
        "dimension,expected_in_payload",
        [
            (512, True),  # non-default, should be in payload
        ],
    )
    def test_dimension_reduction_sent_in_payload(
        self, mock_provider, dimension, expected_in_payload
    ):
        """Non-default dimension should add output_dimension to API payload."""
        mock_provider.dimension = dimension

        fake_response = MagicMock()
        fake_response.json.return_value = {
            "data": [{"embedding": [0.1] * dimension}]
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

            if expected_in_payload:
                assert captured_payload.get("output_dimension") == dimension
            else:
                assert "output_dimension" not in captured_payload

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

    @pytest.mark.parametrize(
        "input_type,expected_value",
        [
            ("query", "query"),
            ("document", "document"),
        ],
    )
    def test_input_type_sent_in_payload(
        self, mock_provider, input_type, expected_value
    ):
        """Input type should be sent in API payload."""
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

            mock_provider._get_embeddings(["test text"], input_type=input_type)

            assert captured_payload.get("input_type") == expected_value

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

    @pytest.mark.parametrize(
        "role,text,should_embed",
        [
            ("tool_result", "ok", False),
            ("tool_result", "success", False),
            ("tool_result", "done", False),
            ("tool_result", "error", False),
            ("tool_result", "failed", False),
            ("tool_result", "SUCCESS", False),  # case-insensitive
            ("tool_result", "ok  ", False),  # trailing space
            ("tool_result", "File contents: def hello(): print('world')", True),  # real content
            ("user", "   \n\t  ", False),  # whitespace only
            ("user", "", False),  # empty
            ("user", "12345678901234567890", True),  # exactly 20 chars
            ("user", "1234567890123456789", False),  # 19 chars
        ],
    )
    def test_message_filtering(self, mock_provider, role, text, should_embed):
        """Test various message filtering conditions."""
        msg = make_message(role=role, text=text)
        assert mock_provider._should_embed_message(msg) == should_embed


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

    @pytest.mark.parametrize(
        "provider_name,expected_provider",
        [
            ("claude", "claude"),
            (None, "unknown"),  # missing provider_name
        ],
    )
    def test_upsert_provider_name_handling(
        self, mock_provider, provider_name, expected_provider
    ):
        """Upsert should extract and sanitize provider_name from metadata."""
        provider_meta = (
            {"provider_name": provider_name} if provider_name else {}
        )
        msg = make_message(provider_meta=provider_meta)

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

        # Should have called INSERT with expected provider_name
        assert any(expected_provider in str(call) for call in insert_calls)

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

    @pytest.mark.parametrize(
        "embedding_result,expected_empty",
        [
            ([], True),
            ([[0.1, 0.2]], False),
        ],
    )
    def test_query_empty_embedding_behavior(
        self, mock_provider, embedding_result, expected_empty
    ):
        """Query should return empty if embedding generation fails."""
        mock_provider._ensure_vec_available = MagicMock()
        mock_provider._get_embeddings = MagicMock(return_value=embedding_result)
        mock_provider._get_connection = MagicMock()
        mock_conn = MagicMock()
        mock_conn.execute = MagicMock(
            return_value=MagicMock(fetchall=MagicMock(return_value=[]))
        )
        mock_provider._get_connection.return_value = mock_conn

        result = mock_provider.query("search text")

        if expected_empty:
            assert result == []
            mock_provider._get_connection.assert_not_called()
        else:
            # Should have called connection
            mock_provider._get_connection.assert_called()

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

    @pytest.mark.parametrize(
        "msg_count,status_count",
        [
            (42, 5),  # both tables present with data
            (0, 0),   # both tables with zeros
        ],
    )
    def test_stats_table_queries(self, mock_provider, msg_count, status_count):
        """Should return counts from both tables when they exist."""
        mock_provider._get_connection = MagicMock()
        mock_conn = MagicMock()

        def execute_sql(sql, params=None):
            if "message_embeddings" in sql:
                return MagicMock(fetchone=MagicMock(return_value=[msg_count]))
            elif "embedding_status" in sql:
                return MagicMock(fetchone=MagicMock(return_value=[status_count]))
            return MagicMock()

        mock_conn.execute = execute_sql
        mock_provider._get_connection.return_value = mock_conn

        stats = mock_provider.get_embedding_stats()

        assert stats["embedded_messages"] == msg_count
        assert stats["pending_conversations"] == status_count

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

    @pytest.mark.parametrize(
        "vec_available,should_raise",
        [
            (False, True),
            (True, False),
        ],
    )
    def test_vec_availability_state(self, mock_provider, vec_available, should_raise):
        """Test behavior based on vec availability state."""
        from polylogue.storage.search_providers.sqlite_vec import SqliteVecError

        mock_provider._vec_available = vec_available
        mock_provider._get_connection = MagicMock()

        if should_raise:
            with pytest.raises(SqliteVecError):
                mock_provider._ensure_vec_available()
        else:
            mock_provider._ensure_vec_available()
            mock_provider._get_connection.assert_not_called()

    def test_raises_with_helpful_message(self, mock_provider):
        """Error should contain helpful message."""
        from polylogue.storage.search_providers.sqlite_vec import SqliteVecError

        mock_provider._vec_available = False
        with pytest.raises(SqliteVecError, match="pip install"):
            mock_provider._ensure_vec_available()

    def test_probes_on_first_call(self, mock_provider):
        """First call with _vec_available=None should probe via _get_connection."""
        from polylogue.storage.search_providers.sqlite_vec import SqliteVecError

        mock_provider._vec_available = None

        mock_conn = MagicMock()
        mock_conn.close = MagicMock()

        def mock_get_conn():
            mock_provider._vec_available = False
            return mock_conn

        mock_provider._get_connection = mock_get_conn

        with pytest.raises(SqliteVecError, match="not available"):
            mock_provider._ensure_vec_available()

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


class TestSqliteVecProviderFiltering:
    """Tests for message filtering in SqliteVecProvider."""

    @pytest.fixture
    def provider_class(self):
        """Get SqliteVecProvider class (doesn't require sqlite-vec to be installed)."""
        from polylogue.storage.search_providers.sqlite_vec import SqliteVecProvider
        return SqliteVecProvider

    @pytest.mark.parametrize(
        "text,should_embed",
        [
            ("", False),  # empty
            ("Short", False),  # short text
            ("This is a longer user message that should be embedded.", True),  # acceptable
        ],
    )
    def test_should_embed_text_length(self, provider_class, text, should_embed):
        """Test filtering by text length."""
        provider = object.__new__(provider_class)
        provider.voyage_key = "test"
        provider.model = "voyage-4"
        provider.dimension = 1024

        msg = make_message("m1", "c1", text=text)
        assert provider._should_embed_message(msg) == should_embed

    @pytest.mark.parametrize(
        "role,should_embed",
        [
            ("system", False),  # system messages filtered
            ("user", True),  # user messages accepted
            ("assistant", True),  # assistant messages accepted
        ],
    )
    def test_should_embed_by_role(self, provider_class, role, should_embed):
        """Test filtering by message role."""
        provider = object.__new__(provider_class)
        provider.voyage_key = "test"
        provider.model = "voyage-4"
        provider.dimension = 1024

        msg = make_message("m1", "c1", role=role, text="This is a longer message that should be checked for role filtering purposes.")
        result = provider._should_embed_message(msg)
        assert result == should_embed


class TestSqliteVecProviderSerialization:
    """Tests for vector serialization."""

    def test_serialize_f32(self):
        """Test that float vectors are serialized correctly."""
        from polylogue.storage.search_providers.sqlite_vec import _serialize_f32

        vector = [1.0, 2.0, 3.0, 4.0]
        result = _serialize_f32(vector)

        # Should be 4 floats * 4 bytes = 16 bytes
        assert len(result) == 16
        # Little-endian float32
        import struct
        unpacked = struct.unpack("<4f", result)
        assert unpacked == (1.0, 2.0, 3.0, 4.0)

    def test_serialize_f32_empty(self):
        """Empty vector should serialize to empty bytes."""
        from polylogue.storage.search_providers.sqlite_vec import _serialize_f32

        result = _serialize_f32([])
        assert result == b""


class TestCreateVectorProvider:
    """Tests for create_vector_provider factory."""

    def test_returns_none_when_no_voyage_key(self):
        """Returns None when VOYAGE_API_KEY is not configured."""
        from polylogue.storage.search_providers import create_vector_provider

        with patch.dict("os.environ", {}, clear=True):
            provider = create_vector_provider()
            assert provider is None

    def test_returns_none_when_sqlite_vec_not_installed(self, monkeypatch):
        """Returns None when sqlite-vec is not installed."""
        from polylogue.storage.search_providers import create_vector_provider

        monkeypatch.setenv("VOYAGE_API_KEY", "voyage-key")

        # Mock sqlite_vec import to fail
        import builtins
        original_import = builtins.__import__

        def mock_import(name: str, *args, **kwargs):
            if name == "sqlite_vec":
                raise ImportError("No module named 'sqlite_vec'")
            return original_import(name, *args, **kwargs)

        with patch.object(builtins, "__import__", mock_import):
            provider = create_vector_provider()
            # Should return None because sqlite_vec import failed
            assert provider is None


class TestIndexConfigEmbedding:
    """Tests for IndexConfig embedding settings."""

    @pytest.mark.parametrize(
        "env_var,env_value,expected_value,attr_name",
        [
            ("POLYLOGUE_VOYAGE_MODEL", None, "voyage-4", "voyage_model"),  # default
            ("POLYLOGUE_VOYAGE_MODEL", "voyage-4-large", "voyage-4-large", "voyage_model"),
            ("POLYLOGUE_VOYAGE_DIMENSION", None, None, "voyage_dimension"),  # default None
            ("POLYLOGUE_VOYAGE_DIMENSION", "512", 512, "voyage_dimension"),
            ("POLYLOGUE_AUTO_EMBED", None, False, "auto_embed"),  # default False
            ("POLYLOGUE_AUTO_EMBED", "true", True, "auto_embed"),
        ],
    )
    def test_config_from_env(self, monkeypatch, env_var, env_value, expected_value, attr_name):
        """Test IndexConfig environment variable handling."""
        from polylogue.paths import IndexConfig

        if env_value is None:
            monkeypatch.delenv(env_var, raising=False)
        else:
            monkeypatch.setenv(env_var, env_value)

        config = IndexConfig.from_env()
        assert getattr(config, attr_name) == expected_value


class TestArchiveStats:
    """Tests for ArchiveStats dataclass."""

    @pytest.mark.parametrize(
        "total_convs,embedded_convs,expected_coverage",
        [
            (100, 75, 75.0),  # typical case
            (0, 0, 0.0),  # zero conversations
            (50, 50, 100.0),  # full coverage
        ],
    )
    def test_embedding_coverage_calculation(
        self, total_convs, embedded_convs, expected_coverage
    ):
        """Embedding coverage should be calculated correctly."""
        from polylogue.lib.stats import ArchiveStats

        stats = ArchiveStats(
            total_conversations=total_convs,
            total_messages=total_convs * 5,  # synthetic
            embedded_conversations=embedded_convs,
        )
        assert stats.embedding_coverage == expected_coverage

    def test_avg_messages_per_conversation(self):
        """Average messages should be calculated correctly."""
        from polylogue.lib.stats import ArchiveStats

        stats = ArchiveStats(
            total_conversations=10,
            total_messages=50,
        )
        assert stats.avg_messages_per_conversation == 5.0

    def test_to_dict(self):
        """to_dict should include all relevant fields."""
        from polylogue.lib.stats import ArchiveStats

        stats = ArchiveStats(
            total_conversations=100,
            total_messages=500,
            providers={"claude": 60, "chatgpt": 40},
            embedded_conversations=75,
            embedded_messages=400,
            db_size_bytes=1024 * 1024,
        )
        d = stats.to_dict()

        assert d["total_conversations"] == 100
        assert d["total_messages"] == 500
        assert d["provider_count"] == 2
        assert d["embedding_coverage_percent"] == 75.0
        assert d["db_size_bytes"] == 1024 * 1024
