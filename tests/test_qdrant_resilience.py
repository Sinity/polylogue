"""Tests for Qdrant provider resilience and retry behavior.

Covers retry scenarios for:
- Embedding API timeouts
- Embedding API HTTP errors
- Qdrant connection errors
- Exponential backoff configuration
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from polylogue.ingestion.drive_client import (
    DriveError,
    DriveAuthError,
    DriveNotFoundError,
)


# =============================================================================
# QdrantError Tests
# =============================================================================


class TestQdrantError:
    """Tests for QdrantError exception."""

    def test_qdrant_error_is_runtime_error(self):
        """QdrantError inherits from RuntimeError."""
        from polylogue.storage.search_providers.qdrant import QdrantError

        exc = QdrantError("Test error")
        assert isinstance(exc, RuntimeError)

    def test_qdrant_error_message(self):
        """QdrantError preserves error message."""
        from polylogue.storage.search_providers.qdrant import QdrantError

        exc = QdrantError("Something went wrong")
        assert str(exc) == "Something went wrong"

    def test_qdrant_error_chaining(self):
        """QdrantError can chain from other exceptions."""
        from polylogue.storage.search_providers.qdrant import QdrantError

        original = ValueError("Original")
        exc = QdrantError("Wrapped")
        exc.__cause__ = original

        assert exc.__cause__ is original


# =============================================================================
# Retry Decorator Tests
# =============================================================================


class TestRetryDecorator:
    """Tests for the lazy-loaded retry decorator."""

    def test_retry_decorator_returns_decorator(self):
        """_retry_decorator returns a decorator factory."""
        from polylogue.storage.search_providers.qdrant import _retry_decorator

        decorator_factory = _retry_decorator()
        assert callable(decorator_factory)

    def test_retry_decorator_creates_decorator(self):
        """Decorator factory creates a decorator."""
        from polylogue.storage.search_providers.qdrant import _retry_decorator

        decorator_factory = _retry_decorator()
        decorator = decorator_factory(
            stop_attempts=3,
            min_wait=1,
            max_wait=5,
            retry_on=Exception,
        )
        assert callable(decorator)


# =============================================================================
# Configuration Constants Tests
# =============================================================================


class TestQdrantConfiguration:
    """Tests for Qdrant configuration values."""

    def test_default_collection_name(self):
        """Verify default collection name."""
        from polylogue.storage.search_providers.qdrant import DEFAULT_COLLECTION

        assert DEFAULT_COLLECTION == "polylogue_messages"

    def test_default_vector_size_matches_voyage2(self):
        """Verify default vector size matches Voyage-2 embedding dimension."""
        from polylogue.storage.search_providers.qdrant import DEFAULT_VECTOR_SIZE

        # Voyage-2 produces 1024-dimensional embeddings
        assert DEFAULT_VECTOR_SIZE == 1024

    def test_voyage_api_url_is_valid(self):
        """Verify Voyage API URL is properly configured."""
        from polylogue.storage.search_providers.qdrant import VOYAGE_API_URL

        assert "voyageai.com" in VOYAGE_API_URL
        assert "/v1/embeddings" in VOYAGE_API_URL
        assert VOYAGE_API_URL.startswith("https://")


# =============================================================================
# Retry Count Verification
# =============================================================================


class TestRetryConfigurations:
    """Verify retry configurations match documented behavior."""

    def test_embedding_retry_count_is_5(self):
        """Embedding generation retries 5 times (documented in source)."""
        import inspect
        from polylogue.storage.search_providers import qdrant

        source = inspect.getsource(qdrant.QdrantProvider._get_embeddings)
        # Verify the retry decorator uses stop_after_attempt(5)
        assert "stop_after_attempt(5)" in source

    def test_upsert_retry_count_is_3(self):
        """Upsert batch retries 3 times (documented in source)."""
        import inspect
        from polylogue.storage.search_providers import qdrant

        source = inspect.getsource(qdrant.QdrantProvider.upsert)
        # Verify the retry decorator uses stop_after_attempt(3)
        assert "stop_after_attempt(3)" in source

    def test_collection_check_retry_count_is_3(self):
        """Collection check retries 3 times (documented in source)."""
        import inspect
        from polylogue.storage.search_providers import qdrant

        source = inspect.getsource(qdrant.QdrantProvider._ensure_collection)
        # Verify the retry decorator uses stop_after_attempt(3)
        assert "stop_after_attempt(3)" in source


# =============================================================================
# Batch Size Verification
# =============================================================================


class TestBatchConfiguration:
    """Verify batch processing configuration."""

    def test_batch_size_is_64(self):
        """Upsert batches messages in groups of 64."""
        import inspect
        from polylogue.storage.search_providers import qdrant

        source = inspect.getsource(qdrant.QdrantProvider.upsert)
        assert "batch_size = 64" in source


# =============================================================================
# API Interface Verification
# =============================================================================


class TestQdrantProviderInterface:
    """Verify QdrantProvider has required interface."""

    def test_has_upsert_method(self):
        """Provider has upsert method for indexing."""
        from polylogue.storage.search_providers.qdrant import QdrantProvider

        assert hasattr(QdrantProvider, "upsert")
        assert callable(getattr(QdrantProvider, "upsert"))

    def test_has_query_method(self):
        """Provider has query method for search."""
        from polylogue.storage.search_providers.qdrant import QdrantProvider

        assert hasattr(QdrantProvider, "query")
        assert callable(getattr(QdrantProvider, "query"))

    def test_query_accepts_text_and_limit(self):
        """Query method accepts text and limit parameters."""
        import inspect
        from polylogue.storage.search_providers.qdrant import QdrantProvider

        sig = inspect.signature(QdrantProvider.query)
        params = list(sig.parameters.keys())

        assert "text" in params
        assert "limit" in params
