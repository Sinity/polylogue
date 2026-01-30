"""Tests for DriveConfig integration."""

from __future__ import annotations

from pathlib import Path

from polylogue.paths import DriveConfig


class TestDriveConfig:
    """Tests for DriveConfig dataclass."""

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
