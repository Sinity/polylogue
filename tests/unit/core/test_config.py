"""Tests for configuration classes and logging infrastructure.

Consolidated from test_config.py and test_logging.py.
"""

from __future__ import annotations

import sys
from io import StringIO
from pathlib import Path

import pytest

from polylogue.config import Config, ConfigError
from polylogue.logging import _StderrProxy, configure_logging, get_logger
from polylogue.paths import DriveConfig, IndexConfig, Source


class TestConfig:
    """Tests for Config dataclass."""

    def test_config_basic_construction(self, tmp_path: Path) -> None:
        """Config can be constructed with required fields."""
        config = Config(
            archive_root=tmp_path / "archive",
            render_root=tmp_path / "render",
            sources=[],
        )
        assert config.archive_root == tmp_path / "archive"
        assert config.render_root == tmp_path / "render"
        assert config.sources == []

    def test_config_with_sources(self, tmp_path: Path) -> None:
        """Config stores source list."""
        sources = [
            Source(name="inbox", path=tmp_path / "inbox"),
            Source(name="claude-code", path=tmp_path / "claude-ai"),
        ]
        config = Config(
            archive_root=tmp_path,
            render_root=tmp_path / "render",
            sources=sources,
        )
        assert len(config.sources) == 2
        assert config.sources[0].name == "inbox"
        assert config.sources[1].name == "claude-code"

    def test_config_db_path_property(self, workspace_env: dict[str, Path]) -> None:
        """db_path property returns paths.DB_PATH."""
        config = Config(
            archive_root=Path(workspace_env["archive_root"]),
            render_root=Path(workspace_env["archive_root"]) / "render",
            sources=[],
        )
        assert config.db_path.name == "polylogue.db"
        assert "polylogue" in str(config.db_path)

    def test_config_optional_fields_default_none(self, tmp_path: Path) -> None:
        """Optional fields default to None."""
        config = Config(
            archive_root=tmp_path,
            render_root=tmp_path / "render",
            sources=[],
        )
        assert config.drive_config is None
        assert config.index_config is None


class TestConfigError:
    """Tests for ConfigError exception."""

    def test_config_error_is_polylogue_error(self) -> None:
        """ConfigError inherits from PolylogueError."""
        from polylogue.errors import PolylogueError

        assert issubclass(ConfigError, PolylogueError)

    def test_config_error_message(self) -> None:
        """ConfigError preserves error message."""
        err = ConfigError("test error message")
        assert str(err) == "test error message"

    def test_config_error_can_be_raised_and_caught(self) -> None:
        """ConfigError works with try/except."""
        with pytest.raises(ConfigError, match="bad config"):
            raise ConfigError("bad config")


class TestSource:
    """Tests for Source dataclass validation."""

    def test_source_with_path(self, tmp_path: Path) -> None:
        """Source with path is valid."""
        src = Source(name="test", path=tmp_path)
        assert src.name == "test"
        assert src.path == tmp_path
        assert not src.is_drive

    def test_source_with_folder(self) -> None:
        """Source with folder (Drive) is valid."""
        src = Source(name="gemini", folder="Google AI Studio")
        assert src.name == "gemini"
        assert src.is_drive

    def test_source_empty_name_raises(self) -> None:
        """Empty name raises ValueError."""
        with pytest.raises(ValueError, match="cannot be empty"):
            Source(name="", path=Path("/tmp"))

    def test_source_whitespace_name_raises(self) -> None:
        """Whitespace-only name raises ValueError."""
        with pytest.raises(ValueError, match="cannot be empty"):
            Source(name="   ", path=Path("/tmp"))

    def test_source_no_path_no_folder_raises(self) -> None:
        """Source without path or folder raises ValueError."""
        with pytest.raises(ValueError, match="must have either"):
            Source(name="broken")

    def test_source_both_path_and_folder_valid(self) -> None:
        """Source with both path and folder is valid (Drive sources use local cache)."""
        src = Source(name="gemini", path=Path("/tmp/cache"), folder="Drive Folder")
        assert src.is_drive
        assert src.path == Path("/tmp/cache")
        assert src.folder == "Drive Folder"

    def test_source_name_stripped(self) -> None:
        """Source name is stripped of whitespace."""
        src = Source(name="  test  ", path=Path("/tmp"))
        assert src.name == "test"

    def test_source_folder_stripped(self) -> None:
        """Source folder is stripped of whitespace."""
        src = Source(name="test", folder="  My Folder  ")
        assert src.folder == "My Folder"


class TestDriveConfig:
    """Tests for DriveConfig defaults."""

    def test_default_retry_count(self) -> None:
        """Default retry count is 3."""
        config = DriveConfig()
        assert config.retry_count == 3

    def test_default_timeout(self) -> None:
        """Default timeout is 30 seconds."""
        config = DriveConfig()
        assert config.timeout == 30

    def test_credentials_path_is_in_config(self) -> None:
        """Default credentials path is in polylogue config dir."""
        config = DriveConfig()
        assert "polylogue" in str(config.credentials_path)


class TestIndexConfig:
    """Tests for IndexConfig from environment."""

    def test_from_env_defaults(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Default IndexConfig has FTS enabled and no vector provider configured."""
        monkeypatch.delenv("VOYAGE_API_KEY", raising=False)
        config = IndexConfig.from_env()
        assert config.fts_enabled is True
        assert config.voyage_api_key is None

    def test_from_env_voyage_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """VOYAGE_API_KEY is used for embeddings."""
        monkeypatch.setenv("VOYAGE_API_KEY", "voyage-key")
        config = IndexConfig.from_env()
        assert config.voyage_api_key == "voyage-key"


class TestXDGPaths:
    """Tests for XDG path resolution."""

    def test_xdg_data_home_respected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """XDG_DATA_HOME env var overrides default."""
        monkeypatch.setenv("XDG_DATA_HOME", "/custom/data")

        import polylogue.paths

        assert Path("/custom/data") == polylogue.paths.data_root()

        monkeypatch.delenv("XDG_DATA_HOME", raising=False)

    def test_db_path_under_data_home(self, workspace_env: dict[str, Path]) -> None:
        """DB_PATH is under XDG_DATA_HOME/polylogue/."""
        import polylogue.paths

        assert "polylogue" in str(polylogue.paths.db_path())
        assert polylogue.paths.db_path().name == "polylogue.db"


class TestConfiguredSources:
    def test_get_sources_skips_drive_source_without_cache_or_credentials(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        monkeypatch.setenv("HOME", str(tmp_path / "home"))
        monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
        monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))

        from polylogue.paths_config import get_sources

        sources = get_sources()
        assert [source.name for source in sources] == ["inbox"]

    def test_get_sources_includes_drive_source_when_credentials_exist(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        home = tmp_path / "home"
        data = tmp_path / "data"
        state = tmp_path / "state"
        config = tmp_path / "config"
        monkeypatch.setenv("HOME", str(home))
        monkeypatch.setenv("XDG_DATA_HOME", str(data))
        monkeypatch.setenv("XDG_STATE_HOME", str(state))
        monkeypatch.setenv("XDG_CONFIG_HOME", str(config))
        credentials = config / "polylogue" / "polylogue-credentials.json"
        credentials.parent.mkdir(parents=True, exist_ok=True)
        credentials.write_text("{}", encoding="utf-8")

        from polylogue.paths_config import get_sources

        sources = get_sources()
        assert [source.name for source in sources] == ["inbox", "gemini"]


# =============================================================================
# Merged from test_logging.py (2024-03-15)
# =============================================================================


# =============================================================================
# Runtime Services Tests (relocated from test_json.py)
# =============================================================================


class TestRuntimeServices:
    def test_repository_is_cached_per_runtime_scope(self, workspace_env: dict[str, Path]) -> None:
        from polylogue.services import build_runtime_services

        services = build_runtime_services()
        repo1 = services.get_repository()
        repo2 = services.get_repository()
        assert repo1 is repo2

    def test_backend_is_cached_per_runtime_scope(self, workspace_env: dict[str, Path]) -> None:
        from polylogue.services import build_runtime_services

        services = build_runtime_services()
        backend1 = services.get_backend()
        backend2 = services.get_backend()
        assert backend1 is backend2

    def test_repository_uses_runtime_backend(self, workspace_env: dict[str, Path]) -> None:
        from polylogue.services import build_runtime_services

        services = build_runtime_services()
        repo = services.get_repository()
        assert repo.backend is services.get_backend()

    def test_distinct_runtime_scopes_do_not_share_instances(self, workspace_env: dict[str, Path]) -> None:
        from polylogue.services import build_runtime_services

        services1 = build_runtime_services()
        services2 = build_runtime_services()
        assert services1.get_repository() is not services2.get_repository()
        assert services1.get_backend() is not services2.get_backend()


# =============================================================================
# Merged from test_logging.py (2024-03-15)
# =============================================================================


def test_configure_logging_accepts_verbose_mode() -> None:
    configure_logging(verbose=True, json_logs=False)


def test_configure_logging_accepts_json_mode() -> None:
    configure_logging(verbose=False, json_logs=True)


def test_get_logger_returns_structlog_logger() -> None:
    assert get_logger("test.module") is not None


def test_stderr_proxy_write_delegates_to_current_stderr() -> None:
    proxy = _StderrProxy()
    original_stderr = sys.stderr
    try:
        sys.stderr = StringIO()
        proxy.write("test message")
        assert sys.stderr.getvalue() == "test message"
    finally:
        sys.stderr = original_stderr


def test_stderr_proxy_exposes_terminal_capabilities() -> None:
    proxy = _StderrProxy()
    assert isinstance(proxy.isatty(), bool)
    assert isinstance(proxy.fileno(), int)
