"""Tests for DriveConfig integration."""

from __future__ import annotations

from pathlib import Path

from polylogue.config import DriveConfig, default_config, load_config


class TestDriveConfig:
    """Tests for DriveConfig dataclass."""

    def test_drive_config_defaults(self):
        """DriveConfig should have sensible defaults."""
        config = DriveConfig()
        assert config.credentials_path is None
        assert config.token_path is None
        assert config.retry_count == 3
        assert config.timeout == 30

    def test_drive_config_custom_values(self):
        """DriveConfig should accept custom values."""
        config = DriveConfig(
            credentials_path=Path("/custom/creds.json"),
            token_path=Path("/custom/token.json"),
            retry_count=5,
            timeout=60,
        )
        assert config.credentials_path == Path("/custom/creds.json")
        assert config.token_path == Path("/custom/token.json")
        assert config.retry_count == 5
        assert config.timeout == 60


class TestConfigDriveIntegration:
    """Tests for DriveConfig integration in Config."""

    def test_default_config_includes_drive_config(self, tmp_path: Path):
        """default_config should create DriveConfig from environment."""
        config = default_config(path=tmp_path / "config.json")
        assert config.drive_config is not None
        assert isinstance(config.drive_config, DriveConfig)

    def test_default_config_loads_drive_env_vars(self, tmp_path: Path, monkeypatch):
        """default_config should load DriveConfig from environment variables."""
        monkeypatch.setenv("POLYLOGUE_CREDENTIAL_PATH", "/env/creds.json")
        monkeypatch.setenv("POLYLOGUE_TOKEN_PATH", "/env/token.json")
        monkeypatch.setenv("POLYLOGUE_DRIVE_RETRIES", "7")

        config = default_config(path=tmp_path / "config.json")
        assert config.drive_config is not None
        assert config.drive_config.credentials_path == Path("/env/creds.json")
        assert config.drive_config.token_path == Path("/env/token.json")
        assert config.drive_config.retry_count == 7

    def test_default_config_handles_invalid_retry_count(self, tmp_path: Path, monkeypatch):
        """default_config should use default retry count for invalid env var."""
        monkeypatch.setenv("POLYLOGUE_DRIVE_RETRIES", "invalid")

        config = default_config(path=tmp_path / "config.json")
        assert config.drive_config is not None
        assert config.drive_config.retry_count == 3  # default

    def test_load_config_includes_drive_config(self, tmp_path: Path):
        """load_config should create DriveConfig from environment."""
        config_path = tmp_path / "config.json"
        config_path.write_text(
            """{
            "version": 2,
            "archive_root": "/tmp/archive",
            "render_root": "/tmp/render",
            "sources": []
        }"""
        )

        config = load_config(config_path)
        assert config.drive_config is not None
        assert isinstance(config.drive_config, DriveConfig)

    def test_load_config_loads_drive_env_vars(self, tmp_path: Path, monkeypatch):
        """load_config should load DriveConfig from environment variables."""
        monkeypatch.setenv("POLYLOGUE_CREDENTIAL_PATH", "/env/creds.json")
        monkeypatch.setenv("POLYLOGUE_TOKEN_PATH", "/env/token.json")
        monkeypatch.setenv("POLYLOGUE_DRIVE_RETRIES", "9")

        config_path = tmp_path / "config.json"
        config_path.write_text(
            """{
            "version": 2,
            "archive_root": "/tmp/archive",
            "render_root": "/tmp/render",
            "sources": []
        }"""
        )

        config = load_config(config_path)
        assert config.drive_config is not None
        assert config.drive_config.credentials_path == Path("/env/creds.json")
        assert config.drive_config.token_path == Path("/env/token.json")
        assert config.drive_config.retry_count == 9


