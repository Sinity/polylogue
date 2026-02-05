"""Simple, targeted tests to improve overall coverage."""

from __future__ import annotations

import json

import pytest


class TestVerifyStatusStr:
    """Test VerifyStatus enum __str__ method."""

    def test_verify_status_str_conversion(self):
        """Test __str__ returns the value."""
        from polylogue.health import VerifyStatus

        assert str(VerifyStatus.OK) == "ok"
        assert str(VerifyStatus.WARNING) == "warning"
        assert str(VerifyStatus.ERROR) == "error"


class TestHealthCachePaths:
    """Test health.py cache error paths."""

    def test_cache_load_corrupted_json(self, tmp_path):
        """Test _load_cached handles corrupted JSON."""
        from polylogue.health import _cache_path, _load_cached

        archive_root = tmp_path
        cache_file = _cache_path(archive_root)
        cache_file.parent.mkdir(parents=True, exist_ok=True)

        # Write invalid JSON
        cache_file.write_text("{invalid json", encoding="utf-8")

        result = _load_cached(archive_root)
        assert result is None

    def test_cache_load_encoding_error(self, tmp_path):
        """Test _load_cached handles encoding errors."""
        from polylogue.health import _cache_path, _load_cached

        archive_root = tmp_path
        cache_file = _cache_path(archive_root)
        cache_file.parent.mkdir(parents=True, exist_ok=True)

        # Write invalid UTF-8
        cache_file.write_bytes(b"\xff\xfe invalid utf-8")

        result = _load_cached(archive_root)
        assert result is None

    def test_cache_write_creates_dirs(self, tmp_path):
        """Test _write_cache creates parent directories."""
        from polylogue.health import _cache_path, _write_cache

        archive_root = tmp_path / "deep" / "nested"
        assert not archive_root.exists()

        payload = {"status": "ok"}
        _write_cache(archive_root, payload)

        cache_file = _cache_path(archive_root)
        assert cache_file.exists()
        loaded = json.loads(cache_file.read_text())
        assert loaded == payload


class TestTimestampParsing:
    """Test core/timestamps.py edge cases."""

    def test_parse_timestamp_none(self):
        """Test parse_timestamp with None."""
        from polylogue.lib.timestamps import parse_timestamp

        result = parse_timestamp(None)
        assert result is None

    def test_parse_timestamp_numeric(self):
        """Test parse_timestamp with numeric timestamp."""
        from polylogue.lib.timestamps import parse_timestamp

        # Valid numeric timestamp
        result = parse_timestamp(1704067200.0)
        assert result is not None


class TestJsonUtils:
    """Test core/json.py utilities."""

    def test_loads_valid_json(self):
        """Test loads with valid JSON."""
        from polylogue.lib.json import loads

        data = loads('{"key": "value"}')
        assert data == {"key": "value"}

    def test_loads_invalid_json(self):
        """Test loads with invalid JSON raises."""
        from polylogue.lib.json import loads

        with pytest.raises((ValueError, json.JSONDecodeError)):
            loads("{invalid}")
