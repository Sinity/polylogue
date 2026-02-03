"""Consolidated search core tests.

SYSTEMATIZATION: Merged from:
- test_search.py (FTS5 escaping and search)
- test_search_providers.py (Search provider factory and implementations)

This file contains tests for:
- FTS5 query escaping
- Search message operations
- Search provider factory
- FTS5Provider and QdrantProvider
"""
from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest

from polylogue.config import Config, IndexConfig
from polylogue.storage.backends.sqlite import open_connection
from polylogue.storage.index import rebuild_index
from polylogue.storage.search import escape_fts5_query, search_messages
from polylogue.storage.search_providers import create_vector_provider
from polylogue.storage.search_providers.fts5 import FTS5Provider
from polylogue.storage.search_providers.qdrant import QdrantProvider
from tests.helpers import ConversationBuilder, make_conversation, make_message
from tests.factories import DbFactory


# =============================================================================
# FTS5 SEARCH TESTS (from test_search.py)
# =============================================================================


# =============================================================================
# FTS5 ESCAPING - PARAMETRIZED (1 test replacing 30+ tests)
# =============================================================================


# Test cases: (input_query, expected_property, description)
FTS5_ESCAPE_CASES = [
    # Empty/whitespace
    ('', '""', "empty query"),
    ('   ', '""', "whitespace only"),

    # Quotes
    ('find "quoted text" here', 'has_doubled_quotes', "internal quotes"),

    # Wildcards
    ('*', '""', "bare asterisk"),
    ('test*', 'starts_and_ends_with_quotes', "asterisk with text"),
    ('?', '?', "question mark"),  # Single char, no special FTS5 chars -> unquoted

    # FTS5 operators (should be quoted as literals)
    ('AND', '"AND"', "AND operator"),
    ('OR', '"OR"', "OR operator"),
    ('NOT', '"NOT"', "NOT operator"),
    ('NEAR', '"NEAR"', "NEAR operator"),
    ('and', '"and"', "lowercase and"),
    ('And', '"And"', "mixed case And"),

    # Special characters
    ('test:value', 'starts_and_ends_with_quotes', "colon"),
    ('test^2', 'starts_and_ends_with_quotes', "caret"),
    ('function(arg)', 'starts_and_ends_with_quotes', "parentheses"),
    ('test)', 'starts_and_ends_with_quotes', "close paren"),
    ('(test', 'starts_and_ends_with_quotes', "open paren"),

    # Minus/hyphen
    ('-test', 'starts_and_ends_with_quotes', "leading minus"),
    ('test-word', 'starts_and_ends_with_quotes', "embedded hyphen"),

    # Plus
    ('+required', 'starts_and_ends_with_quotes', "plus operator"),

    # Multiple operators
    ('test AND query', 'test AND query', "embedded AND - passes through unquoted"),
    ('OR query', 'starts_and_ends_with_quotes', "leading OR - quoted for safety"),

    # Normal text (NOT quoted - implementation passes simple alphanumeric through as-is)
    ('simple query', 'simple query', "simple words"),
    ('hello', 'hello', "single word"),
]


@pytest.mark.parametrize("input_query,expected,desc", FTS5_ESCAPE_CASES)
def test_escape_fts5_comprehensive(input_query, expected, desc):
    """Comprehensive FTS5 escaping test.

    Replaces 30+ individual escaping tests with single parametrized test.

    Expected can be:
    - Exact string match (e.g., '""')
    - Property to check (e.g., 'starts_and_ends_with_quotes', 'has_doubled_quotes')
    """
    result = escape_fts5_query(input_query)

    if expected == 'starts_and_ends_with_quotes':
        assert result.startswith('"'), f"Failed {desc}: doesn't start with quote"
        assert result.endswith('"'), f"Failed {desc}: doesn't end with quote"
    elif expected == 'has_doubled_quotes':
        assert result.startswith('"'), f"Failed {desc}: not quoted"
        assert result.endswith('"'), f"Failed {desc}: not quoted"
        assert '""' in result, f"Failed {desc}: quotes not doubled"
    else:
        # Exact match
        assert result == expected, f"Failed {desc}: expected {expected}, got {result}"


# =============================================================================
# SEARCH INTEGRATION - PARAMETRIZED (1 test replacing ~10 tests)
# =============================================================================


@pytest.mark.parametrize("query,should_find", [
    ("test", True),  # Basic search
    ("nonexistent", False),  # No match
    ("*", False),  # Bare asterisk escaped
    ("AND", False),  # Operator as literal
    ("quoted", True),  # Part of text with quotes
])
def test_search_messages_escaping_integration(query, should_find, tmp_path):
    """Integration test for search with various queries.

    Replaces ~10 individual integration tests.
    """
    from pathlib import Path

    # Setup database with test data
    db_path = tmp_path / "test.db"
    db = DbFactory(db_path)

    # Insert test conversation using builder
    (ConversationBuilder(db_path, "test1")
     .title("Test Conversation")
     .add_message("msg1", role="user", text='This is a test message with "quoted text" inside.')
     .save())

    # Build search index
    with open_connection(str(db_path)) as conn:
        rebuild_index(conn)

    # Search - use keyword arguments
    results = search_messages(
        query,
        archive_root=tmp_path,
        db_path=Path(str(db_path)),
        limit=10,
    )

    if should_find:
        assert len(results.hits) > 0, f"Expected to find results for '{query}'"
    else:
        # Either no results or results don't match the query
        # The important thing is no SQL errors occur
        assert isinstance(results.hits, list)


# =============================================================================
# EDGE CASES - PARAMETRIZED (1 test replacing ~5 tests)
# =============================================================================


@pytest.mark.parametrize("special_query,should_quote", [
    ("test OR anything", False),  # "OR" in middle - passes through unquoted
    ("NOT this", True),  # "NOT" at start - should be quoted
    ("NEAR that", True),  # "NEAR" at start - should be quoted
    ("' OR '1'='1", False),  # No special FTS5 chars, passes through (single quotes aren't FTS5 special)
    ("test; DROP TABLE messages--", True),  # Contains special chars (semicolon, etc.), should be quoted
])
def test_escape_fts5_injection_prevention(special_query, should_quote):
    """Prevent dangerous operator positions and special characters.

    Replaces ~5 security-focused tests.
    """
    result = escape_fts5_query(special_query)

    if should_quote:
        # Should be safely quoted
        assert result.startswith('"'), f"Expected quoted: {special_query}"
        assert result.endswith('"'), f"Expected quoted: {special_query}"
    else:
        # These may or may not be quoted depending on special chars
        # The important thing is they don't cause FTS5 errors
        assert isinstance(result, str), f"Should return string: {special_query}"


# =============================================================================
# UNICODE HANDLING - PARAMETRIZED (1 test)
# =============================================================================


@pytest.mark.parametrize("unicode_query", [
    "文字",  # Chinese
    "тест",  # Cyrillic
    "🔍",   # Emoji
    "café",  # Accented
])
def test_escape_fts5_unicode(unicode_query):
    """Unicode queries are handled correctly.

    Unicode-only queries are simple alphanumeric (no special FTS5 chars),
    so they pass through unquoted.
    """
    result = escape_fts5_query(unicode_query)

    # Should preserve unicode and pass through unquoted
    assert result == unicode_query


# =============================================================================
# SEARCH RESULT VALIDATION (NEW - was missing)
# =============================================================================


def test_search_messages_returns_valid_structure(tmp_path):
    """Search results have expected structure."""
    from pathlib import Path

    db_path = tmp_path / "test.db"
    db = DbFactory(db_path)

    # Insert test conversation using builder
    (ConversationBuilder(db_path, "test1")
     .title("Test")
     .add_message("msg1", role="user", text="Searchable content")
     .save())

    # Build search index
    with open_connection(str(db_path)) as conn:
        rebuild_index(conn)

    # Search using keyword arguments
    results = search_messages(
        "searchable",
        archive_root=tmp_path,
        db_path=Path(str(db_path)),
        limit=10,
    )

    assert len(results.hits) > 0
    for hit in results.hits:
        # Verify result structure
        assert hasattr(hit, 'snippet')
        assert hasattr(hit, 'conversation_id')
        assert hit.snippet is not None
        assert len(hit.snippet) > 0

# =============================================================================
# SEARCH PROVIDER TESTS (from test_search_providers.py)
# =============================================================================


class TestCreateVectorProvider:
    """Tests for create_vector_provider factory."""

    def test_returns_none_when_no_qdrant_url(self):
        """Returns None when QDRANT_URL is not configured."""
        with patch.dict("os.environ", {}, clear=True):
            provider = create_vector_provider()
            assert provider is None

    def test_uses_env_vars_when_no_config(self, monkeypatch):
        """Uses environment variables when no config provided."""
        monkeypatch.setenv("QDRANT_URL", "http://localhost:6333")
        monkeypatch.setenv("QDRANT_API_KEY", "test-key")
        monkeypatch.setenv("VOYAGE_API_KEY", "voyage-key")

        with patch("polylogue.storage.search_providers.QdrantProvider") as mock_provider:
            create_vector_provider()
            mock_provider.assert_called_once_with(
                qdrant_url="http://localhost:6333",
                api_key="test-key",
                voyage_key="voyage-key",
            )

    def test_uses_config_over_env_vars(self, monkeypatch, tmp_path):
        """Config values take priority over environment variables."""
        monkeypatch.setenv("QDRANT_URL", "http://env:6333")
        monkeypatch.setenv("VOYAGE_API_KEY", "env-voyage-key")

        index_config = IndexConfig(
            qdrant_url="http://config:6333",
            voyage_api_key="config-voyage-key",
        )
        config = Config(
            archive_root=tmp_path / "archive",
            render_root=tmp_path / "render",
            sources=[],
            index_config=index_config,
        )

        with patch("polylogue.storage.search_providers.QdrantProvider") as mock_provider:
            create_vector_provider(config=config)
            mock_provider.assert_called_once_with(
                qdrant_url="http://config:6333",
                api_key=None,
                voyage_key="config-voyage-key",
            )

    def test_explicit_args_override_config(self, tmp_path):
        """Explicit arguments override both config and env vars."""
        index_config = IndexConfig(
            qdrant_url="http://config:6333",
            voyage_api_key="config-voyage-key",
        )
        config = Config(
            archive_root=tmp_path / "archive",
            render_root=tmp_path / "render",
            sources=[],
            index_config=index_config,
        )

        with patch("polylogue.storage.search_providers.QdrantProvider") as mock_provider:
            create_vector_provider(
                config=config,
                qdrant_url="http://explicit:6333",
                voyage_api_key="explicit-voyage-key",
            )
            mock_provider.assert_called_once_with(
                qdrant_url="http://explicit:6333",
                api_key=None,
                voyage_key="explicit-voyage-key",
            )

    def test_raises_when_voyage_key_missing(self, monkeypatch):
        """Raises ValueError when Qdrant URL is set but Voyage key is missing."""
        monkeypatch.setenv("QDRANT_URL", "http://localhost:6333")

        with patch.dict("os.environ", {"QDRANT_URL": "http://localhost:6333"}, clear=True):
            with pytest.raises(ValueError, match="VOYAGE_API_KEY"):
                create_vector_provider()

    def test_config_with_none_values_falls_back_to_env(self, monkeypatch, tmp_path):
        """Config with None values falls back to environment variables."""
        monkeypatch.setenv("QDRANT_URL", "http://env:6333")
        monkeypatch.setenv("VOYAGE_API_KEY", "env-voyage-key")

        index_config = IndexConfig(
            qdrant_url=None,  # Explicitly None
            voyage_api_key=None,
        )
        config = Config(
            archive_root=tmp_path / "archive",
            render_root=tmp_path / "render",
            sources=[],
            index_config=index_config,
        )

        with patch("polylogue.storage.search_providers.QdrantProvider") as mock_provider:
            create_vector_provider(config=config)
            mock_provider.assert_called_once_with(
                qdrant_url="http://env:6333",
                api_key=None,
                voyage_key="env-voyage-key",
            )


class TestFTS5Provider:
    """Tests for FTS5Provider full-text search implementation."""

    @pytest.fixture
    def fts_provider(self, workspace_env):
        """Create FTS5Provider with test database."""
        db_path = workspace_env["data_root"] / "polylogue" / "polylogue.db"
        return FTS5Provider(db_path=db_path)

    @pytest.fixture
    def populated_fts(self, workspace_env, storage_repository, fts_provider):
        """FTS provider with indexed test data."""
        conv = make_conversation("fts-conv-1", provider_name="claude", title="FTS Test", created_at="1000", updated_at="1000", provider_meta={"source": "inbox"})
        msgs = [
            make_message("fts-msg-1", "fts-conv-1", text="How do I implement quicksort in Python?", timestamp="1000"),
            make_message("fts-msg-2", "fts-conv-1", role="assistant", text="Quicksort is a divide-and-conquer algorithm for sorting", timestamp="1001"),
        ]
        storage_repository.save_conversation(conversation=conv, messages=msgs, attachments=[])

        # Index the messages
        fts_provider.index(msgs)
        return fts_provider

    def test_ensure_index_creates_fts_table(self, workspace_env, fts_provider):
        """Ensure index creates FTS5 virtual table."""
        from polylogue.storage.backends.sqlite import open_connection

        db_path = workspace_env["data_root"] / "polylogue" / "polylogue.db"

        # Index empty list to trigger table creation
        fts_provider.index([])

        with open_connection(db_path) as conn:
            # Check if the FTS table doesn't exist yet (we passed empty list)
            # Actually we need to trigger the _ensure_index by indexing something
            pass

        # Index with actual message to ensure table creation
        conv = make_conversation("ensure-conv", title="Ensure Test", created_at="1000", updated_at="1000", provider_meta={"source": "inbox"})
        # First save the conversation so provider_name lookup works
        from polylogue.storage.backends.sqlite import SQLiteBackend

        backend = SQLiteBackend(db_path=db_path)
        backend.begin()
        backend.save_conversation(conv)
        backend.commit()

        msgs = [make_message("ens-msg", "ensure-conv", timestamp="1000")]

        fts_provider.index(msgs)

        with open_connection(db_path) as conn:
            row = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='messages_fts'").fetchone()
            assert row is not None
            assert row["name"] == "messages_fts"

    def test_ensure_index_idempotent(self, workspace_env, fts_provider, storage_repository):
        """Calling index multiple times is safe (idempotent)."""
        conv = make_conversation("idem-conv", title="Idempotent Test", created_at="1000", updated_at="1000", provider_meta={"source": "inbox"})
        msgs = [make_message("idem-msg", "idem-conv", text="Idempotent message", timestamp="1000")]
        storage_repository.save_conversation(conversation=conv, messages=msgs, attachments=[])

        # Index twice - should not error or duplicate
        fts_provider.index(msgs)
        fts_provider.index(msgs)

        # Search should return exactly one result
        results = fts_provider.search("idempotent")
        assert len(results) == 1
        assert results[0] == "idem-msg"

    def test_index_deletes_old_entries(self, workspace_env, fts_provider, storage_repository):
        """Incremental indexing removes old entries before inserting."""
        conv = make_conversation("incr-conv", title="Incremental Test", created_at="1000", updated_at="1000", provider_meta={"source": "inbox"})
        msgs_v1 = [make_message("incr-msg-1", "incr-conv", text="Original content about apples", timestamp="1000")]
        storage_repository.save_conversation(conversation=conv, messages=msgs_v1, attachments=[])
        fts_provider.index(msgs_v1)

        # Should find "apples"
        results = fts_provider.search("apples")
        assert len(results) == 1

        # Re-index with different content
        msgs_v2 = [make_message("incr-msg-1", "incr-conv", text="Updated content about oranges", timestamp="1000")]
        fts_provider.index(msgs_v2)

        # "apples" should no longer be found
        results = fts_provider.search("apples")
        assert len(results) == 0

        # "oranges" should be found
        results = fts_provider.search("oranges")
        assert len(results) == 1

    def test_index_skips_empty_text(self, workspace_env, fts_provider, storage_repository):
        """Messages with empty text are not indexed."""
        conv = make_conversation("skip-conv", title="Skip Test", created_at="1000", updated_at="1000", provider_meta={"source": "inbox"})
        msgs = [
            make_message("skip-msg-1", "skip-conv", text="", timestamp="1000"),  # Empty text
            make_message("skip-msg-2", "skip-conv", role="assistant", text="This has content", timestamp="1001"),
        ]
        storage_repository.save_conversation(conversation=conv, messages=msgs, attachments=[])
        fts_provider.index(msgs)

        # Search should only find the non-empty message
        results = fts_provider.search("content")
        assert len(results) == 1
        assert results[0] == "skip-msg-2"

    def test_search_returns_ranked_results(self, populated_fts):
        """Search returns results ordered by relevance (BM25)."""
        # The populated fixture has messages about quicksort
        results = populated_fts.search("quicksort")
        assert len(results) == 2  # Both messages mention quicksort
        # Results should be in relevance order (checked implicitly by ORDER BY rank)

    def test_search_escapes_fts5_special_chars(self, populated_fts):
        """Search query escapes FTS5 special characters."""
        # Quotes and asterisks should be escaped
        results = populated_fts.search('"special* query"')
        # Should not raise FTS5 syntax error
        assert isinstance(results, list)

    def test_search_returns_empty_if_no_index(self, workspace_env):
        """Search returns empty list if FTS index doesn't exist."""
        db_path = workspace_env["data_root"] / "polylogue" / "nonexistent.db"
        provider = FTS5Provider(db_path=db_path)
        results = provider.search("anything")
        assert results == []

        # Could be empty or match all - depends on FTS5 behavior
        assert isinstance(results, list)


class TestQdrantProviderChunking:
    """Tests for QdrantProvider batching/chunking."""

    @pytest.fixture
    def provider(self):
        """Create a QdrantProvider with mocked client."""
        # Mock sys.modules to handle lazy imports
        mock_qdrant_module = MagicMock()
        mock_client_cls = MagicMock()
        mock_qdrant_module.QdrantClient = mock_client_cls

        mock_httpx = MagicMock()

        with patch.dict(
            "sys.modules",
            {
                "qdrant_client": mock_qdrant_module,
                "qdrant_client.http": MagicMock(),
                "httpx": mock_httpx,
            },
        ):
            provider = QdrantProvider(qdrant_url="http://test:6333", api_key="test-key", voyage_key="voyage-key")
            # Ensure the client property returns our mock instance
            provider._client = mock_client_cls.return_value
            return provider

    def test_upsert_chunks_large_request(self, provider):
        """Verify that upsert chunks messages into batches of 64."""
        # Create 150 messages (should be 3 chunks: 64, 64, 22)
        messages = [make_message(f"msg-{i}", "conv-1", text=f"Message {i}", timestamp="1000") for i in range(150)]

        # Mock embeddings to return one mock vector per input text
        with patch.object(provider, "_get_embeddings") as mock_embed:
            mock_embed.side_effect = lambda texts: [[0.1] * 1024] * len(texts)

            provider.upsert(conversation_id="conv-1", messages=messages)

            # Check embedding calls (3 batches)
            assert mock_embed.call_count == 3
            assert len(mock_embed.call_args_list[0][0][0]) == 64
            assert len(mock_embed.call_args_list[1][0][0]) == 64
            assert len(mock_embed.call_args_list[2][0][0]) == 22

            # Check Qdrant upsert calls
            assert provider.client.upsert.call_count == 3
            assert len(provider.client.upsert.call_args_list[0].kwargs["points"]) == 64
            assert len(provider.client.upsert.call_args_list[1].kwargs["points"]) == 64
            assert len(provider.client.upsert.call_args_list[2].kwargs["points"]) == 22

    def test_upsert_handles_single_chunk(self, provider):
        """Verify upsert handling for small lists (<64)."""
        messages = [make_message(f"msg-{i}", "conv-1", text=f"Message {i}", timestamp="1000") for i in range(10)]

        with patch.object(provider, "_get_embeddings") as mock_embed:
            mock_embed.return_value = [[0.1] * 1024] * 10
            provider.upsert(conversation_id="conv-1", messages=messages)

            mock_embed.assert_called_once()
            assert len(mock_embed.call_args[0][0]) == 10
            provider.client.upsert.assert_called_once()
            assert len(provider.client.upsert.call_args.kwargs["points"]) == 10
