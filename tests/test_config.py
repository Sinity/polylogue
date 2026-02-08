"""Tests for polylogue.config module.

Covers:
- Config dataclass construction and defaults
- get_config() returns proper Config from paths module
- ConfigError for backward compatibility
- db_path property delegation to paths.DB_PATH
"""

from __future__ import annotations

from pathlib import Path

import pytest

from polylogue.config import Config, ConfigError, get_config
from polylogue.paths import Source


class TestConfig:
    """Tests for Config dataclass."""

    def test_config_basic_construction(self, tmp_path):
        """Config can be constructed with required fields."""
        config = Config(
            archive_root=tmp_path / "archive",
            render_root=tmp_path / "render",
            sources=[],
        )
        assert config.archive_root == tmp_path / "archive"
        assert config.render_root == tmp_path / "render"
        assert config.sources == []

    def test_config_with_sources(self, tmp_path):
        """Config stores source list."""
        sources = [
            Source(name="inbox", path=tmp_path / "inbox"),
            Source(name="claude-code", path=tmp_path / "claude"),
        ]
        config = Config(
            archive_root=tmp_path,
            render_root=tmp_path / "render",
            sources=sources,
        )
        assert len(config.sources) == 2
        assert config.sources[0].name == "inbox"
        assert config.sources[1].name == "claude-code"

    def test_config_db_path_property(self, workspace_env):
        """db_path property returns paths.DB_PATH."""
        config = Config(
            archive_root=Path(workspace_env["archive_root"]),
            render_root=Path(workspace_env["archive_root"]) / "render",
            sources=[],
        )
        # db_path delegates to paths.DB_PATH which uses XDG_DATA_HOME
        assert config.db_path.name == "polylogue.db"
        assert "polylogue" in str(config.db_path)

    def test_config_optional_fields_default_none(self, tmp_path):
        """Optional fields default to None."""
        config = Config(
            archive_root=tmp_path,
            render_root=tmp_path / "render",
            sources=[],
        )
        assert config.drive_config is None
        assert config.index_config is None


class TestGetConfig:
    """Tests for get_config() function."""

    def test_get_config_returns_config(self, workspace_env):
        """get_config() returns a Config instance."""
        # Re-import after workspace_env reloads modules to get same class identity
        from polylogue.config import Config as ReloadedConfig
        from polylogue.config import get_config as reloaded_get_config

        config = reloaded_get_config()
        assert isinstance(config, ReloadedConfig)

    def test_get_config_has_archive_root(self, workspace_env):
        """Config from get_config() has an archive_root."""
        config = get_config()
        assert config.archive_root is not None
        assert isinstance(config.archive_root, Path)

    def test_get_config_has_render_root(self, workspace_env):
        """Config from get_config() has a render_root."""
        config = get_config()
        assert config.render_root is not None
        assert isinstance(config.render_root, Path)

    def test_get_config_has_sources(self, workspace_env):
        """Config from get_config() has a sources list."""
        config = get_config()
        assert isinstance(config.sources, list)
        # Should always have at least inbox + gemini
        assert len(config.sources) >= 1

    def test_get_config_has_drive_config(self, workspace_env):
        """Config from get_config() includes drive configuration."""
        config = get_config()
        assert config.drive_config is not None

    def test_get_config_has_index_config(self, workspace_env):
        """Config from get_config() includes index configuration."""
        config = get_config()
        assert config.index_config is not None


class TestConfigError:
    """Tests for ConfigError exception."""

    def test_config_error_is_runtime_error(self):
        """ConfigError inherits from RuntimeError."""
        assert issubclass(ConfigError, RuntimeError)

    def test_config_error_message(self):
        """ConfigError preserves error message."""
        err = ConfigError("test error message")
        assert str(err) == "test error message"

    def test_config_error_can_be_raised_and_caught(self):
        """ConfigError works with try/except."""
        with pytest.raises(ConfigError, match="bad config"):
            raise ConfigError("bad config")
