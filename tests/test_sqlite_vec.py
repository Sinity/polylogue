"""Tests for SqliteVecProvider with mocked Voyage API."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from polylogue.storage.store import MessageRecord


def make_message(
    message_id: str,
    conversation_id: str,
    role: str = "user",
    text: str = "Test message content",
) -> MessageRecord:
    """Create a test MessageRecord."""
    return MessageRecord(
        message_id=message_id,
        conversation_id=conversation_id,
        role=role,
        text=text,
        content_hash="hash",
        provider_meta={"provider_name": "test"},
        version=1,
    )


class TestSqliteVecProviderFiltering:
    """Tests for message filtering in SqliteVecProvider."""

    @pytest.fixture
    def provider_class(self):
        """Get SqliteVecProvider class (doesn't require sqlite-vec to be installed)."""
        from polylogue.storage.search_providers.sqlite_vec import SqliteVecProvider
        return SqliteVecProvider

    def test_should_embed_filters_empty_text(self, provider_class):
        """Empty text should not be embedded."""
        # Create provider without connecting (just test the method)
        provider = object.__new__(provider_class)
        provider.voyage_key = "test"
        provider.model = "voyage-4"
        provider.dimension = 1024

        msg = make_message("m1", "c1", text="")
        assert not provider._should_embed_message(msg)

    def test_should_embed_filters_short_text(self, provider_class):
        """Text shorter than 20 chars should not be embedded."""
        provider = object.__new__(provider_class)
        provider.voyage_key = "test"
        provider.model = "voyage-4"
        provider.dimension = 1024

        msg = make_message("m1", "c1", text="Short")
        assert not provider._should_embed_message(msg)

    def test_should_embed_filters_system_messages(self, provider_class):
        """System messages should not be embedded."""
        provider = object.__new__(provider_class)
        provider.voyage_key = "test"
        provider.model = "voyage-4"
        provider.dimension = 1024

        msg = make_message("m1", "c1", role="system", text="You are a helpful assistant.")
        assert not provider._should_embed_message(msg)

    def test_should_embed_accepts_user_messages(self, provider_class):
        """User messages with sufficient length should be embedded."""
        provider = object.__new__(provider_class)
        provider.voyage_key = "test"
        provider.model = "voyage-4"
        provider.dimension = 1024

        msg = make_message("m1", "c1", text="This is a longer user message that should be embedded.")
        assert provider._should_embed_message(msg)

    def test_should_embed_accepts_assistant_messages(self, provider_class):
        """Assistant messages with sufficient length should be embedded."""
        provider = object.__new__(provider_class)
        provider.voyage_key = "test"
        provider.model = "voyage-4"
        provider.dimension = 1024

        msg = make_message("m1", "c1", role="assistant", text="This is a longer assistant response that should be embedded.")
        assert provider._should_embed_message(msg)


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

    def test_default_voyage_model(self, monkeypatch):
        """Default voyage model should be voyage-4."""
        from polylogue.paths import IndexConfig

        monkeypatch.delenv("POLYLOGUE_VOYAGE_MODEL", raising=False)
        config = IndexConfig.from_env()
        assert config.voyage_model == "voyage-4"

    def test_custom_voyage_model(self, monkeypatch):
        """Custom voyage model via environment variable."""
        from polylogue.paths import IndexConfig

        monkeypatch.setenv("POLYLOGUE_VOYAGE_MODEL", "voyage-4-large")
        config = IndexConfig.from_env()
        assert config.voyage_model == "voyage-4-large"

    def test_voyage_dimension_default(self, monkeypatch):
        """Default dimension should be None (use model default)."""
        from polylogue.paths import IndexConfig

        monkeypatch.delenv("POLYLOGUE_VOYAGE_DIMENSION", raising=False)
        config = IndexConfig.from_env()
        assert config.voyage_dimension is None

    def test_voyage_dimension_custom(self, monkeypatch):
        """Custom dimension via environment variable."""
        from polylogue.paths import IndexConfig

        monkeypatch.setenv("POLYLOGUE_VOYAGE_DIMENSION", "512")
        config = IndexConfig.from_env()
        assert config.voyage_dimension == 512

    def test_auto_embed_default(self, monkeypatch):
        """Auto-embed should be disabled by default."""
        from polylogue.paths import IndexConfig

        monkeypatch.delenv("POLYLOGUE_AUTO_EMBED", raising=False)
        config = IndexConfig.from_env()
        assert config.auto_embed is False

    def test_auto_embed_enabled(self, monkeypatch):
        """Auto-embed can be enabled via environment."""
        from polylogue.paths import IndexConfig

        monkeypatch.setenv("POLYLOGUE_AUTO_EMBED", "true")
        config = IndexConfig.from_env()
        assert config.auto_embed is True


class TestArchiveStats:
    """Tests for ArchiveStats dataclass."""

    def test_embedding_coverage_calculation(self):
        """Embedding coverage should be calculated correctly."""
        from polylogue.lib.stats import ArchiveStats

        stats = ArchiveStats(
            total_conversations=100,
            total_messages=500,
            embedded_conversations=75,
        )
        assert stats.embedding_coverage == 75.0

    def test_embedding_coverage_zero_conversations(self):
        """Coverage should be 0 when no conversations exist."""
        from polylogue.lib.stats import ArchiveStats

        stats = ArchiveStats(
            total_conversations=0,
            total_messages=0,
        )
        assert stats.embedding_coverage == 0.0

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
