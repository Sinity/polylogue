"""Tests for polylogue.health module."""

from __future__ import annotations

import json
import os
import stat
import time
from pathlib import Path

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

    def test_cache_read_logs_corruption_warning(self, tmp_path: Path, capsys):
        """Cache read errors should be logged as warnings."""
        from polylogue.health import _load_cached

        # Create corrupted cache file (must be named health.json)
        cache_file = tmp_path / "health.json"
        cache_file.write_text("{ invalid json")

        result = _load_cached(tmp_path)

        # Should return None
        assert result is None
        # Should have logged a warning about the corruption (structlog outputs to stdout)
        captured = capsys.readouterr()
        assert "cache" in captured.out.lower() or "cache" in captured.err.lower(), (
            f"Expected warning log about cache, got stdout: {captured.out}, stderr: {captured.err}"
        )

    def test_cache_read_logs_permission_error(self, tmp_path: Path, capsys):
        """Permission errors should be logged."""
        from polylogue.health import _load_cached

        cache_file = tmp_path / "health.json"
        cache_file.write_text(json.dumps({"status": "ok"}))

        try:
            os.chmod(cache_file, 0o000)

            result = _load_cached(tmp_path)

            # Should return None
            assert result is None
            # Should have logged something about the error (structlog outputs to stdout)
            captured = capsys.readouterr()
            assert (
                "permission" in captured.out.lower() or "cache" in captured.out.lower()
                or "permission" in captured.err.lower() or "cache" in captured.err.lower()
            ), f"Expected warning about permission/cache, got stdout: {captured.out}, stderr: {captured.err}"
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

    def test_cache_write_error_logged_not_raised(self, tmp_path: Path, capsys, monkeypatch):
        """Cache write errors should be logged, not raised."""
        from polylogue.health import _write_cache

        # Mock Path.write_text to raise an OSError
        original_write = Path.write_text

        def failing_write(self, data, encoding=None):
            if "health.json" in str(self):
                raise OSError("Simulated write error")
            return original_write(self, data, encoding=encoding)

        monkeypatch.setattr(Path, "write_text", failing_write)
        # Should not raise, just log warning
        _write_cache(tmp_path, {"status": "ok"})

        # Should have logged a warning (structlog outputs to stdout)
        captured = capsys.readouterr()
        assert (
            "failed" in captured.out.lower() or "cache" in captured.out.lower()
            or "failed" in captured.err.lower() or "cache" in captured.err.lower()
        ), f"Expected warning about write failure, got stdout: {captured.out}, stderr: {captured.err}"

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

    def test_cache_read_unicode_errors_handled(self, tmp_path: Path, capsys):
        """Unicode decode errors in cache should be handled gracefully."""
        from polylogue.health import _load_cached

        # Write binary data that isn't valid UTF-8
        cache_file = tmp_path / "health.json"
        cache_file.write_bytes(b"\xff\xfe invalid utf8")

        result = _load_cached(tmp_path)

        # Should return None
        assert result is None
        # Should have logged a warning (structlog outputs to stdout)
        captured = capsys.readouterr()
        assert (
            "cache" in captured.out.lower() or "error" in captured.out.lower()
            or "cache" in captured.err.lower() or "error" in captured.err.lower()
        ), f"Expected warning about cache/error, got stdout: {captured.out}, stderr: {captured.err}"


class TestHealthCheck:
    """Tests for HealthCheck dataclass."""

    def test_health_check_creation(self):
        """HealthCheck should be creatable with name, status, detail."""
        from polylogue.health import HealthCheck, VerifyStatus

        check = HealthCheck(name="test", status=VerifyStatus.OK, detail="All good")

        assert check.name == "test"
        assert check.status == VerifyStatus.OK
        assert check.detail == "All good"

    def test_health_check_to_dict(self):
        """HealthCheck should convert to dict for serialization."""
        from polylogue.health import HealthCheck, VerifyStatus

        check = HealthCheck(name="database", status=VerifyStatus.ERROR, detail="Connection failed")

        result = check.to_dict()
        assert result == {
            "name": "database",
            "status": "error",
            "count": 0,
            "detail": "Connection failed",
            "breakdown": {},
        }


class TestRunHealth:
    """Tests for run_health function."""

    def test_run_health_with_valid_config(self, cli_workspace):
        """run_health should return checks for valid configuration."""
        from polylogue.config import load_config
        from polylogue.health import run_health

        config = load_config(cli_workspace["config_path"])
        result = run_health(config)

        assert result.timestamp > 0
        assert isinstance(result.checks, list)
        assert len(result.checks) > 0

    def test_run_health_includes_config_check(self, cli_workspace):
        """run_health should include a config check."""
        from polylogue.config import load_config
        from polylogue.health import VerifyStatus, run_health

        config = load_config(cli_workspace["config_path"])
        result = run_health(config)

        checks = result.checks
        config_checks = [c for c in checks if c.name == "config"]
        assert len(config_checks) > 0
        assert config_checks[0].status == VerifyStatus.OK

    def test_run_health_includes_archive_root_check(self, cli_workspace):
        """run_health should include archive_root check (ok when exists)."""
        from polylogue.config import load_config
        from polylogue.health import VerifyStatus, run_health

        config = load_config(cli_workspace["config_path"])
        result = run_health(config)

        checks = result.checks
        archive_checks = [c for c in checks if c.name == "archive_root"]
        assert len(archive_checks) > 0
        assert archive_checks[0].status == VerifyStatus.OK

    def test_run_health_archive_root_warning_when_missing(self, tmp_path):
        """run_health should warn when archive_root doesn't exist."""
        from polylogue.config import Config, Source
        from polylogue.health import VerifyStatus, run_health

        missing_archive = tmp_path / "missing"
        config = Config(
            archive_root=missing_archive,
            render_root=missing_archive / "render",
            sources=[Source(name="test", path=tmp_path)],
        )

        result = run_health(config)
        checks = result.checks
        archive_checks = [c for c in checks if c.name == "archive_root"]
        assert len(archive_checks) > 0
        assert archive_checks[0].status == VerifyStatus.WARNING

    def test_run_health_includes_render_root_check(self, cli_workspace):
        """run_health should include render_root check."""
        from polylogue.config import load_config
        from polylogue.health import run_health

        config = load_config(cli_workspace["config_path"])
        result = run_health(config)

        checks = result.checks
        render_checks = [c for c in checks if c.name == "render_root"]
        assert len(render_checks) > 0

    def test_run_health_render_root_warning_when_missing(self, tmp_path):
        """run_health should warn when render_root doesn't exist."""
        from polylogue.config import Config, Source
        from polylogue.health import VerifyStatus, run_health

        archive_root = tmp_path / "archive"
        archive_root.mkdir()
        missing_render = tmp_path / "missing"

        config = Config(
            archive_root=archive_root,
            render_root=missing_render,
            sources=[Source(name="test", path=tmp_path)],
        )

        result = run_health(config)
        checks = result.checks
        render_checks = [c for c in checks if c.name == "render_root"]
        assert len(render_checks) > 0
        assert render_checks[0].status == VerifyStatus.WARNING

    def test_run_health_includes_database_check(self, cli_workspace):
        """run_health should include database check."""
        from polylogue.config import load_config
        from polylogue.health import VerifyStatus, run_health

        config = load_config(cli_workspace["config_path"])
        result = run_health(config)

        checks = result.checks
        db_checks = [c for c in checks if c.name == "database"]
        assert len(db_checks) > 0
        assert db_checks[0].status == VerifyStatus.OK

    def test_run_health_includes_index_check(self, cli_workspace):
        """run_health should include index check."""
        from polylogue.config import load_config
        from polylogue.health import run_health

        config = load_config(cli_workspace["config_path"])
        result = run_health(config)

        checks = result.checks
        index_checks = [c for c in checks if c.name == "index"]
        assert len(index_checks) > 0

    def test_run_health_includes_source_checks(self, cli_workspace):
        """run_health should include checks for each source."""
        from polylogue.config import load_config
        from polylogue.health import run_health

        config = load_config(cli_workspace["config_path"])
        result = run_health(config)

        checks = result.checks
        source_checks = [c for c in checks if c.name.startswith("source:")]
        assert len(source_checks) >= len(config.sources)

    def test_run_health_caches_result(self, cli_workspace):
        """run_health should write cache to health.json."""
        from polylogue.config import load_config
        from polylogue.health import _cache_path, run_health

        config = load_config(cli_workspace["config_path"])
        result = run_health(config)

        cache_file = _cache_path(config.archive_root)
        assert cache_file.exists(), "run_health should write cache file"

        cached_data = json.loads(cache_file.read_text())
        assert cached_data["timestamp"] == result.timestamp
        assert len(cached_data["checks"]) == len(result.checks)


class TestGetHealth:
    """Tests for get_health function."""

    def test_get_health_returns_fresh_on_cache_miss(self, cli_workspace):
        """get_health should run fresh checks when cache doesn't exist."""
        from polylogue.config import load_config
        from polylogue.health import get_health

        config = load_config(cli_workspace["config_path"])
        result = get_health(config)

        assert result.timestamp > 0
        assert isinstance(result.checks, list)
        assert result.cached is False
        assert result.age_seconds == 0

    def test_get_health_returns_cached_when_valid(self, cli_workspace):
        """get_health should return cached result when valid."""
        from polylogue.config import load_config
        from polylogue.health import get_health

        config = load_config(cli_workspace["config_path"])

        # First call populates cache
        result1 = get_health(config)
        timestamp1 = result1.timestamp

        # Second call should return cached result (same timestamp)
        # Cache TTL validation doesn't require delay; we verify cache is used via the cached flag
        result2 = get_health(config)
        timestamp2 = result2.timestamp

        assert timestamp1 == timestamp2, "Cached result should have same timestamp"
        assert result2.cached is True
        assert result2.age_seconds >= 0

    def test_get_health_refreshes_on_cache_expiry(self, cli_workspace, monkeypatch):
        """get_health should refresh when cache is too old."""
        import time

        from polylogue.config import load_config
        from polylogue.health import HEALTH_TTL_SECONDS, get_health

        config = load_config(cli_workspace["config_path"])

        # First call populates cache
        result1 = get_health(config)

        # Manually set cache timestamp to old
        from polylogue.health import _cache_path, _load_cached

        cache_file = _cache_path(config.archive_root)
        cached = _load_cached(config.archive_root)
        if cached:
            cached["timestamp"] = int(time.time()) - HEALTH_TTL_SECONDS - 100
            cache_file.write_text(json.dumps(cached))

        # Second call should refresh (not use cache)
        result2 = get_health(config)

        assert result2.cached is False

    def test_get_health_includes_cache_metadata(self, cli_workspace):
        """get_health should include cached and age_seconds fields."""
        from polylogue.config import load_config
        from polylogue.health import get_health

        config = load_config(cli_workspace["config_path"])
        result = get_health(config)

        assert isinstance(result.cached, bool)
        assert isinstance(result.age_seconds, int)


class TestCachedHealthSummary:
    """Tests for cached_health_summary function."""

    def test_cached_health_summary_not_run_when_no_cache(self, tmp_path):
        """Should return 'not run' when cache doesn't exist."""
        from polylogue.health import cached_health_summary

        result = cached_health_summary(tmp_path)
        assert result == "not run"

    def test_cached_health_summary_with_valid_cache(self, cli_workspace):
        """Should return summary string from valid cache."""
        from polylogue.config import load_config
        from polylogue.health import cached_health_summary, run_health

        config = load_config(cli_workspace["config_path"])
        run_health(config)  # Populate cache

        result = cached_health_summary(config.archive_root)

        # Should be cached X seconds ago (with check counts)
        assert "cached" in result
        assert "ago" in result
        assert "ok=" in result or "warning=" in result or "error=" in result

    def test_cached_health_summary_corrupted_cache(self, tmp_path):
        """Should return 'unknown' when cache is corrupted."""
        from polylogue.health import cached_health_summary

        # Create corrupted cache
        cache_file = tmp_path / "health.json"
        cache_file.write_text("{invalid json")

        result = cached_health_summary(tmp_path)
        assert result == "unknown" or result == "not run"

    def test_cached_health_summary_non_dict_cache(self, tmp_path):
        """Should handle non-dict JSON gracefully."""
        from polylogue.health import cached_health_summary

        cache_file = tmp_path / "health.json"
        cache_file.write_text('["not", "a", "dict"]')

        result = cached_health_summary(tmp_path)
        # Should not crash, returns fallback
        assert isinstance(result, str)

    def test_cached_health_summary_missing_timestamp(self, tmp_path):
        """Should handle missing timestamp safely."""
        from polylogue.health import cached_health_summary

        cache_file = tmp_path / "health.json"
        cache_file.write_text(json.dumps({"checks": []}))

        result = cached_health_summary(tmp_path)
        assert "cached" in result

    def test_cached_health_summary_non_int_timestamp(self, tmp_path):
        """Should handle non-int timestamp safely."""
        from polylogue.health import cached_health_summary

        cache_file = tmp_path / "health.json"
        cache_file.write_text(json.dumps({"timestamp": "not-an-int", "checks": []}))

        result = cached_health_summary(tmp_path)
        assert result == "unknown"

    def test_cached_health_summary_counts_statuses(self, tmp_path):
        """Should properly count ok, warning, error statuses."""
        from polylogue.health import cached_health_summary

        cache_file = tmp_path / "health.json"
        cache_file.write_text(
            json.dumps(
                {
                    "timestamp": int(time.time()),
                    "summary": {"ok": 2, "warning": 1, "error": 1},
                    "checks": [
                        {"name": "check1", "status": "ok", "detail": "ok"},
                        {"name": "check2", "status": "ok", "detail": "ok"},
                        {"name": "check3", "status": "warning", "detail": "warning"},
                        {"name": "check4", "status": "error", "detail": "error"},
                    ],
                }
            )
        )

        result = cached_health_summary(tmp_path)

        # Should contain count summary
        assert "ok=2" in result
        assert "warning=1" in result
        assert "error=1" in result

    def test_cached_health_summary_ignores_invalid_checks(self, tmp_path):
        """Should skip non-dict items in checks list."""
        from polylogue.health import cached_health_summary

        cache_file = tmp_path / "health.json"
        cache_file.write_text(
            json.dumps(
                {
                    "timestamp": int(time.time()),
                    "summary": {"ok": 1, "warning": 1},
                    "checks": [
                        {"name": "check1", "status": "ok", "detail": "ok"},
                        "not a dict",  # Invalid check
                        {"name": "check2", "status": "warning", "detail": "warning"},
                    ],
                }
            )
        )

        result = cached_health_summary(tmp_path)

        # Should still work, ignoring the invalid check
        assert isinstance(result, str)


class TestHealthTTL:
    """Tests for health check TTL constant."""

    def test_health_ttl_is_reasonable(self):
        """HEALTH_TTL_SECONDS should be a reasonable cache duration."""
        from polylogue.health import HEALTH_TTL_SECONDS

        # Should be reasonable (e.g., between 1 minute and 1 hour)
        assert 60 <= HEALTH_TTL_SECONDS <= 3600
        assert isinstance(HEALTH_TTL_SECONDS, int)


class TestCachePath:
    """Tests for _cache_path function."""

    def test_cache_path_returns_health_json_in_archive_root(self, tmp_path):
        """_cache_path should return archive_root/health.json."""
        from polylogue.health import _cache_path

        result = _cache_path(tmp_path)

        assert result == tmp_path / "health.json"
        assert result.name == "health.json"
