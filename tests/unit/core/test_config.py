"""Tests for configuration classes.

Consolidated from test_config.py.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from polylogue.config import Config, ConfigError, get_config
from polylogue.paths import DriveConfig, IndexConfig, Source


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

    def test_config_error_is_polylogue_error(self):
        """ConfigError inherits from PolylogueError."""
        from polylogue.errors import PolylogueError

        assert issubclass(ConfigError, PolylogueError)

    def test_config_error_message(self):
        """ConfigError preserves error message."""
        err = ConfigError("test error message")
        assert str(err) == "test error message"

    def test_config_error_can_be_raised_and_caught(self):
        """ConfigError works with try/except."""
        with pytest.raises(ConfigError, match="bad config"):
            raise ConfigError("bad config")


class TestSource:
    """Tests for Source dataclass validation."""

    def test_source_with_path(self, tmp_path):
        """Source with path is valid."""
        src = Source(name="test", path=tmp_path)
        assert src.name == "test"
        assert src.path == tmp_path
        assert not src.is_drive

    def test_source_with_folder(self):
        """Source with folder (Drive) is valid."""
        src = Source(name="gemini", folder="Google AI Studio")
        assert src.name == "gemini"
        assert src.is_drive

    def test_source_empty_name_raises(self):
        """Empty name raises ValueError."""
        with pytest.raises(ValueError, match="cannot be empty"):
            Source(name="", path=Path("/tmp"))

    def test_source_whitespace_name_raises(self):
        """Whitespace-only name raises ValueError."""
        with pytest.raises(ValueError, match="cannot be empty"):
            Source(name="   ", path=Path("/tmp"))

    def test_source_no_path_no_folder_raises(self):
        """Source without path or folder raises ValueError."""
        with pytest.raises(ValueError, match="must have either"):
            Source(name="broken")

    def test_source_both_path_and_folder_raises(self):
        """Source with both path and folder raises ValueError."""
        with pytest.raises(ValueError, match="cannot have both"):
            Source(name="confused", path=Path("/tmp"), folder="Drive Folder")

    def test_source_name_stripped(self):
        """Source name is stripped of whitespace."""
        src = Source(name="  test  ", path=Path("/tmp"))
        assert src.name == "test"

    def test_source_folder_stripped(self):
        """Source folder is stripped of whitespace."""
        src = Source(name="test", folder="  My Folder  ")
        assert src.folder == "My Folder"


class TestDriveConfig:
    """Tests for DriveConfig defaults."""

    def test_default_retry_count(self):
        """Default retry count is 3."""
        config = DriveConfig()
        assert config.retry_count == 3

    def test_default_timeout(self):
        """Default timeout is 30 seconds."""
        config = DriveConfig()
        assert config.timeout == 30

    def test_credentials_path_is_in_config(self):
        """Default credentials path is in polylogue config dir."""
        config = DriveConfig()
        assert "polylogue" in str(config.credentials_path)


class TestIndexConfig:
    """Tests for IndexConfig from environment."""

    def test_from_env_defaults(self, monkeypatch):
        """Default IndexConfig has FTS enabled, no external services."""
        monkeypatch.delenv("POLYLOGUE_VOYAGE_API_KEY", raising=False)
        monkeypatch.delenv("VOYAGE_API_KEY", raising=False)
        monkeypatch.delenv("POLYLOGUE_VOYAGE_MODEL", raising=False)
        monkeypatch.delenv("POLYLOGUE_VOYAGE_DIMENSION", raising=False)
        monkeypatch.delenv("POLYLOGUE_AUTO_EMBED", raising=False)
        config = IndexConfig.from_env()
        assert config.fts_enabled is True
        assert config.voyage_api_key is None
        assert config.voyage_model == "voyage-4"
        assert config.voyage_dimension is None
        assert config.auto_embed is False

    def test_from_env_polylogue_prefixed(self, monkeypatch):
        """POLYLOGUE_* env vars are picked up."""
        monkeypatch.setenv("POLYLOGUE_VOYAGE_API_KEY", "voyage-key")
        monkeypatch.setenv("POLYLOGUE_VOYAGE_MODEL", "voyage-4-large")
        monkeypatch.setenv("POLYLOGUE_VOYAGE_DIMENSION", "512")
        monkeypatch.setenv("POLYLOGUE_AUTO_EMBED", "true")
        config = IndexConfig.from_env()
        assert config.voyage_api_key == "voyage-key"
        assert config.voyage_model == "voyage-4-large"
        assert config.voyage_dimension == 512
        assert config.auto_embed is True

    def test_from_env_unprefixed_fallback(self, monkeypatch):
        """Unprefixed env vars used when POLYLOGUE_* not set."""
        monkeypatch.delenv("POLYLOGUE_VOYAGE_API_KEY", raising=False)
        monkeypatch.setenv("VOYAGE_API_KEY", "fallback-key")
        config = IndexConfig.from_env()
        assert config.voyage_api_key == "fallback-key"

    def test_from_env_prefixed_takes_precedence(self, monkeypatch):
        """POLYLOGUE_* vars take precedence over unprefixed."""
        monkeypatch.setenv("POLYLOGUE_VOYAGE_API_KEY", "preferred-key")
        monkeypatch.setenv("VOYAGE_API_KEY", "fallback-key")
        config = IndexConfig.from_env()
        assert config.voyage_api_key == "preferred-key"


class TestXDGPaths:
    """Tests for XDG path resolution."""

    def test_xdg_data_home_respected(self, monkeypatch):
        """XDG_DATA_HOME env var overrides default."""
        monkeypatch.setenv("XDG_DATA_HOME", "/custom/data")

        import polylogue.paths

        assert Path("/custom/data") == polylogue.paths.data_root()

        monkeypatch.delenv("XDG_DATA_HOME", raising=False)

    def test_db_path_under_data_home(self, workspace_env):
        """DB_PATH is under XDG_DATA_HOME/polylogue/."""
        import polylogue.paths

        assert "polylogue" in str(polylogue.paths.db_path())
        assert polylogue.paths.db_path().name == "polylogue.db"
