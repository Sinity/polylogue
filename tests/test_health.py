"""Tests for polylogue.health module."""
from __future__ import annotations

import json
import os
import stat
from pathlib import Path
from unittest.mock import patch

import pytest


class TestHealthCacheErrorHandling:
    """Tests for health check cache error handling."""

    def test_cache_read_error_returns_none_not_stale_data(self, tmp_path: Path):
        """Corrupted cache file should return None (cache miss), not garbage data.

        This test verifies that error handling prevents returning invalid/stale data
        when the cache is corrupted.
        """
        from polylogue.health import _load_cached

        # Create corrupted cache file with invalid JSON (must be named health.json)
        cache_file = tmp_path / "health.json"
        cache_file.write_text("{ this is not valid json }")

        # Should return None (cache miss), not crash or return garbage
        result = _load_cached(tmp_path)

        assert result is None, "Corrupted cache should return None, not garbage"

    def test_cache_returns_none_on_non_dict_content(self, tmp_path: Path):
        """Valid JSON that isn't a dict should return None."""
        from polylogue.health import _load_cached

        # Create cache file with valid JSON but not a dict (e.g., a list)
        cache_file = tmp_path / "health.json"
        cache_file.write_text('["not", "a", "dict"]')

        # Should return None since we expect a dict
        result = _load_cached(tmp_path)

        assert result is None, "Non-dict JSON should return None"

    def test_cache_returns_none_on_missing_file(self, tmp_path: Path):
        """Missing cache file should return None, not raise."""
        from polylogue.health import _load_cached

        missing_dir = tmp_path / "does_not_exist"
        result = _load_cached(missing_dir)

        assert result is None, "Missing cache file should return None"

    def test_cache_read_permission_error_returns_none(self, tmp_path: Path):
        """Permission errors reading cache should return None, not raise."""
        from polylogue.health import _load_cached

        cache_file = tmp_path / "health.json"
        cache_file.write_text(json.dumps({"status": "ok"}))

        # Make file unreadable (on Unix-like systems)
        try:
            os.chmod(cache_file, 0o000)

            # Should return None, not raise
            result = _load_cached(tmp_path)
            assert result is None, "Permission error should return None"
        finally:
            # Restore permissions for cleanup
            os.chmod(cache_file, stat.S_IRUSR | stat.S_IWUSR)

    def test_cache_read_logs_corruption_warning(self, tmp_path: Path, caplog):
        """Cache read errors should be logged as warnings."""
        import logging
        from polylogue.health import _load_cached

        # Create corrupted cache file (must be named health.json)
        cache_file = tmp_path / "health.json"
        cache_file.write_text("{ invalid json")

        with caplog.at_level(logging.WARNING, logger="polylogue.health"):
            result = _load_cached(tmp_path)

        # Should return None
        assert result is None
        # Should have logged a warning about the corruption
        assert any("cache" in record.message.lower() for record in caplog.records), \
            f"Expected warning log about cache, got: {[r.message for r in caplog.records]}"

    def test_cache_read_logs_permission_error(self, tmp_path: Path, caplog):
        """Permission errors should be logged."""
        import logging
        from polylogue.health import _load_cached

        cache_file = tmp_path / "health.json"
        cache_file.write_text(json.dumps({"status": "ok"}))

        try:
            os.chmod(cache_file, 0o000)

            with caplog.at_level(logging.WARNING, logger="polylogue.health"):
                result = _load_cached(tmp_path)

            # Should return None
            assert result is None
            # Should have logged something about the error
            assert any(
                "permission" in record.message.lower() or "cache" in record.message.lower()
                for record in caplog.records
            ), f"Expected warning about permission, got: {[r.message for r in caplog.records]}"
        finally:
            os.chmod(cache_file, stat.S_IRUSR | stat.S_IWUSR)

    def test_cache_write_creates_parent_dirs(self, tmp_path: Path):
        """Cache write should create parent directories if needed."""
        from polylogue.health import _write_cache

        nested_path = tmp_path / "nested" / "deep" / "dir"

        # Write cache to nested path (directories don't exist yet)
        _write_cache(nested_path, {"status": "ok"})

        # Should have created the file
        cache_file = nested_path / "health.json"
        assert cache_file.exists(), "Cache write should create parent directories"

        # Verify content
        content = json.loads(cache_file.read_text())
        assert content.get("status") == "ok"

    def test_cache_write_error_logged_not_raised(self, tmp_path: Path, caplog, monkeypatch):
        """Cache write errors should be logged, not raised."""
        import logging
        from polylogue.health import _write_cache

        # Mock Path.write_text to raise an OSError
        original_write = Path.write_text

        def failing_write(self, data, encoding=None):
            if "health.json" in str(self):
                raise OSError("Simulated write error")
            return original_write(self, data, encoding=encoding)

        with caplog.at_level(logging.WARNING):
            monkeypatch.setattr(Path, "write_text", failing_write)
            # Should not raise, just log warning
            _write_cache(tmp_path, {"status": "ok"})

        # Should have logged a warning
        assert any(
            "failed" in record.message.lower() or "cache" in record.message.lower()
            for record in caplog.records
        ), f"Expected warning about write failure, got: {[r.message for r in caplog.records]}"

    def test_cache_with_valid_json_returns_dict(self, tmp_path: Path):
        """Valid cache file should return the dict."""
        from polylogue.health import _load_cached

        cache_file = tmp_path / "health.json"
        test_data = {
            "timestamp": 1234567890,
            "checks": [{"name": "test", "status": "ok", "detail": "ok"}],
        }
        cache_file.write_text(json.dumps(test_data))

        result = _load_cached(tmp_path)

        assert result == test_data, "Valid cache should return the dict"

    def test_cache_read_unicode_errors_handled(self, tmp_path: Path, caplog):
        """Unicode decode errors in cache should be handled gracefully."""
        import logging
        from polylogue.health import _load_cached

        # Write binary data that isn't valid UTF-8
        cache_file = tmp_path / "health.json"
        cache_file.write_bytes(b'\xff\xfe invalid utf8')

        with caplog.at_level(logging.WARNING):
            result = _load_cached(tmp_path)

        # Should return None
        assert result is None
        # Should have logged a warning
        assert any(
            "cache" in record.message.lower() or "error" in record.message.lower()
            for record in caplog.records
        )
