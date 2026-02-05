"""Tests for polylogue.storage.search_cache module.

Covers:
- SearchCacheKey creation and immutability
- Cache invalidation via version counter
- Cache stats retrieval
- Thread safety of invalidation
"""

from __future__ import annotations

import threading
from pathlib import Path

from polylogue.storage.search_cache import (
    SearchCacheKey,
    get_cache_stats,
    invalidate_search_cache,
)


class TestSearchCacheKey:
    """Tests for SearchCacheKey creation and behavior."""

    def test_create_basic(self, tmp_path):
        """Create a basic cache key."""
        key = SearchCacheKey.create(
            query="hello",
            archive_root=tmp_path,
        )
        assert key.query == "hello"
        assert key.archive_root == str(tmp_path)
        assert key.limit == 20  # default
        assert key.source is None
        assert key.since is None

    def test_create_with_all_params(self, tmp_path):
        """Create a cache key with all parameters."""
        key = SearchCacheKey.create(
            query="test query",
            archive_root=tmp_path / "archive",
            render_root_path=tmp_path / "render",
            db_path=tmp_path / "test.db",
            limit=50,
            source="claude",
            since="2024-01-01",
        )
        assert key.query == "test query"
        assert key.limit == 50
        assert key.source == "claude"
        assert key.since == "2024-01-01"
        assert key.render_root_path == str(tmp_path / "render")
        assert key.db_path == str(tmp_path / "test.db")

    def test_key_is_frozen(self, tmp_path):
        """Cache key is immutable (frozen dataclass)."""
        key = SearchCacheKey.create(query="test", archive_root=tmp_path)
        # Frozen dataclass should raise on attribute assignment
        try:
            key.query = "changed"
            assert False, "Should have raised"
        except AttributeError:
            pass

    def test_same_params_same_key(self, tmp_path):
        """Same parameters produce equal keys (same cache version)."""
        key1 = SearchCacheKey.create(query="test", archive_root=tmp_path, limit=10)
        key2 = SearchCacheKey.create(query="test", archive_root=tmp_path, limit=10)
        assert key1 == key2

    def test_different_query_different_key(self, tmp_path):
        """Different queries produce different keys."""
        key1 = SearchCacheKey.create(query="hello", archive_root=tmp_path)
        key2 = SearchCacheKey.create(query="world", archive_root=tmp_path)
        assert key1 != key2

    def test_different_limit_different_key(self, tmp_path):
        """Different limits produce different keys."""
        key1 = SearchCacheKey.create(query="test", archive_root=tmp_path, limit=10)
        key2 = SearchCacheKey.create(query="test", archive_root=tmp_path, limit=20)
        assert key1 != key2

    def test_none_render_root(self, tmp_path):
        """None render_root_path stored as None."""
        key = SearchCacheKey.create(
            query="test", archive_root=tmp_path, render_root_path=None
        )
        assert key.render_root_path is None

    def test_key_is_hashable(self, tmp_path):
        """Cache key can be used as dict key (hashable)."""
        key = SearchCacheKey.create(query="test", archive_root=tmp_path)
        d = {key: "result"}
        assert d[key] == "result"


class TestInvalidateSearchCache:
    """Tests for cache invalidation."""

    def test_invalidation_increments_version(self, tmp_path):
        """Invalidation changes cache version."""
        key_before = SearchCacheKey.create(query="test", archive_root=tmp_path)
        invalidate_search_cache()
        key_after = SearchCacheKey.create(query="test", archive_root=tmp_path)

        # Keys should differ due to version change
        assert key_before != key_after
        assert key_before.cache_version < key_after.cache_version

    def test_multiple_invalidations(self, tmp_path):
        """Multiple invalidations increment version each time."""
        v1 = SearchCacheKey.create(query="test", archive_root=tmp_path).cache_version
        invalidate_search_cache()
        v2 = SearchCacheKey.create(query="test", archive_root=tmp_path).cache_version
        invalidate_search_cache()
        v3 = SearchCacheKey.create(query="test", archive_root=tmp_path).cache_version

        assert v1 < v2 < v3


class TestCacheStats:
    """Tests for cache statistics."""

    def test_stats_returns_dict(self):
        """get_cache_stats returns a dictionary."""
        stats = get_cache_stats()
        assert isinstance(stats, dict)
        assert "cache_version" in stats

    def test_stats_version_matches_current(self, tmp_path):
        """Stats version matches what keys use."""
        key = SearchCacheKey.create(query="test", archive_root=tmp_path)
        stats = get_cache_stats()
        assert stats["cache_version"] == key.cache_version


class TestCacheThreadSafety:
    """Thread safety tests for cache invalidation."""

    def test_concurrent_invalidation(self):
        """Concurrent invalidation doesn't corrupt state."""
        initial_stats = get_cache_stats()
        initial_version = initial_stats["cache_version"]

        errors: list[Exception] = []
        num_threads = 10
        invalidations_per_thread = 100

        def invalidate_many():
            try:
                for _ in range(invalidations_per_thread):
                    invalidate_search_cache()
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=invalidate_many) for _ in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        final_stats = get_cache_stats()
        expected_version = initial_version + (num_threads * invalidations_per_thread)
        assert final_stats["cache_version"] == expected_version
