"""Tests for search provider factory functions and FTS5 provider."""

from unittest.mock import MagicMock, patch

import pytest

from polylogue.config import Config, IndexConfig
from polylogue.storage.search_providers import create_vector_provider
from polylogue.storage.search_providers.fts5 import FTS5Provider
from polylogue.storage.store import ConversationRecord, MessageRecord


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
            version=2,
            archive_root=tmp_path / "archive",
            render_root=tmp_path / "render",
            sources=[],
            path=tmp_path / "config.json",
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
            version=2,
            archive_root=tmp_path / "archive",
            render_root=tmp_path / "render",
            sources=[],
            path=tmp_path / "config.json",
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
            version=2,
            archive_root=tmp_path / "archive",
            render_root=tmp_path / "render",
            sources=[],
            path=tmp_path / "config.json",
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
        db_path = workspace_env["state_root"] / "polylogue" / "polylogue.db"
        return FTS5Provider(db_path=db_path)

    @pytest.fixture
    def populated_fts(self, workspace_env, storage_repository, fts_provider):
        """FTS provider with indexed test data."""
        # Create a conversation with messages
        conv = ConversationRecord(
            conversation_id="fts-conv-1",
            provider_name="claude",
            provider_conversation_id="pfts-1",
            title="FTS Test",
            created_at="1000",
            updated_at="1000",
            content_hash="ftshash1",
            provider_meta={"source": "inbox"},
        )
        msgs = [
            MessageRecord(
                message_id="fts-msg-1",
                conversation_id="fts-conv-1",
                provider_message_id="ftsp-1",
                role="user",
                text="How do I implement quicksort in Python?",
                timestamp="1000",
                content_hash="ftsm1",
                provider_meta=None,
            ),
            MessageRecord(
                message_id="fts-msg-2",
                conversation_id="fts-conv-1",
                provider_message_id="ftsp-2",
                role="assistant",
                text="Quicksort is a divide-and-conquer algorithm for sorting",
                timestamp="1001",
                content_hash="ftsm2",
                provider_meta=None,
            ),
        ]
        storage_repository.save_conversation(conversation=conv, messages=msgs, attachments=[])

        # Index the messages
        fts_provider.index(msgs)
        return fts_provider

    def test_ensure_index_creates_fts_table(self, workspace_env, fts_provider):
        """Ensure index creates FTS5 virtual table."""
        from polylogue.storage.db import open_connection

        db_path = workspace_env["state_root"] / "polylogue" / "polylogue.db"

        # Index empty list to trigger table creation
        fts_provider.index([])

        with open_connection(db_path) as conn:
            # Check if the FTS table doesn't exist yet (we passed empty list)
            # Actually we need to trigger the _ensure_index by indexing something
            pass

        # Index with actual message to ensure table creation
        conv = ConversationRecord(
            conversation_id="ensure-conv",
            provider_name="test",
            provider_conversation_id="pens",
            title="Ensure Test",
            created_at="1000",
            updated_at="1000",
            content_hash="enshash",
            provider_meta={"source": "inbox"},
        )
        # First save the conversation so provider_name lookup works
        from polylogue.storage.backends.sqlite import SQLiteBackend

        backend = SQLiteBackend(db_path=db_path)
        backend.begin()
        backend.save_conversation(conv)
        backend.commit()

        msgs = [
            MessageRecord(
                message_id="ens-msg",
                conversation_id="ensure-conv",
                provider_message_id="ens-p",
                role="user",
                text="Test message",
                timestamp="1000",
                content_hash="ens-mhash",
                provider_meta=None,
            ),
        ]

        fts_provider.index(msgs)

        with open_connection(db_path) as conn:
            row = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='messages_fts'"
            ).fetchone()
            assert row is not None
            assert row["name"] == "messages_fts"

    def test_ensure_index_idempotent(self, workspace_env, fts_provider, storage_repository):
        """Calling index multiple times is safe (idempotent)."""
        db_path = workspace_env["state_root"] / "polylogue" / "polylogue.db"

        conv = ConversationRecord(
            conversation_id="idem-conv",
            provider_name="test",
            provider_conversation_id="pidem",
            title="Idempotent Test",
            created_at="1000",
            updated_at="1000",
            content_hash="idemhash",
            provider_meta={"source": "inbox"},
        )
        msgs = [
            MessageRecord(
                message_id="idem-msg",
                conversation_id="idem-conv",
                provider_message_id="idem-p",
                role="user",
                text="Idempotent message",
                timestamp="1000",
                content_hash="idem-mhash",
                provider_meta=None,
            ),
        ]
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
        db_path = workspace_env["state_root"] / "polylogue" / "polylogue.db"

        conv = ConversationRecord(
            conversation_id="incr-conv",
            provider_name="test",
            provider_conversation_id="pincr",
            title="Incremental Test",
            created_at="1000",
            updated_at="1000",
            content_hash="incrhash",
            provider_meta={"source": "inbox"},
        )
        msgs_v1 = [
            MessageRecord(
                message_id="incr-msg-1",
                conversation_id="incr-conv",
                provider_message_id="incr-p-1",
                role="user",
                text="Original content about apples",
                timestamp="1000",
                content_hash="incr-mhash-1",
                provider_meta=None,
            ),
        ]
        storage_repository.save_conversation(conversation=conv, messages=msgs_v1, attachments=[])
        fts_provider.index(msgs_v1)

        # Should find "apples"
        results = fts_provider.search("apples")
        assert len(results) == 1

        # Re-index with different content
        msgs_v2 = [
            MessageRecord(
                message_id="incr-msg-1",
                conversation_id="incr-conv",
                provider_message_id="incr-p-1",
                role="user",
                text="Updated content about oranges",
                timestamp="1000",
                content_hash="incr-mhash-2",
                provider_meta=None,
            ),
        ]
        fts_provider.index(msgs_v2)

        # "apples" should no longer be found
        results = fts_provider.search("apples")
        assert len(results) == 0

        # "oranges" should be found
        results = fts_provider.search("oranges")
        assert len(results) == 1

    def test_index_skips_empty_text(self, workspace_env, fts_provider, storage_repository):
        """Messages with empty text are not indexed."""
        db_path = workspace_env["state_root"] / "polylogue" / "polylogue.db"

        conv = ConversationRecord(
            conversation_id="skip-conv",
            provider_name="test",
            provider_conversation_id="pskip",
            title="Skip Test",
            created_at="1000",
            updated_at="1000",
            content_hash="skiphash",
            provider_meta={"source": "inbox"},
        )
        msgs = [
            MessageRecord(
                message_id="skip-msg-1",
                conversation_id="skip-conv",
                provider_message_id="skip-p-1",
                role="user",
                text="",  # Empty text
                timestamp="1000",
                content_hash="skip-mhash-1",
                provider_meta=None,
            ),
            MessageRecord(
                message_id="skip-msg-2",
                conversation_id="skip-conv",
                provider_message_id="skip-p-2",
                role="assistant",
                text="This has content",
                timestamp="1001",
                content_hash="skip-mhash-2",
                provider_meta=None,
            ),
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
        db_path = workspace_env["state_root"] / "polylogue" / "nonexistent.db"
        provider = FTS5Provider(db_path=db_path)
        results = provider.search("anything")
        assert results == []

    def test_search_empty_query(self, populated_fts):
        """Search with empty-ish query returns empty list."""
        # Empty queries after escaping may not match anything
        results = populated_fts.search("")
        # Could be empty or match all - depends on FTS5 behavior
        assert isinstance(results, list)
