"""Tests for search provider factory functions."""

from unittest.mock import MagicMock, patch

import pytest

from polylogue.config import Config, IndexConfig
from polylogue.storage.search_providers import create_vector_provider


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
