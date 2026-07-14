"""Tests for configuration classes and logging infrastructure.

Consolidated from test_config.py and test_logging.py.
"""

from __future__ import annotations

import sys
from io import StringIO
from pathlib import Path

import pytest

from polylogue.config import Config, ConfigError, DriveConfig, IndexConfig, Source
from polylogue.logging import _StderrProxy, configure_logging, get_logger


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

    def test_config_db_path_default(self, workspace_env: dict[str, Path]) -> None:
        """db_path defaults to the resolved index.db database path."""
        config = Config(
            archive_root=Path(workspace_env["archive_root"]),
            render_root=Path(workspace_env["archive_root"]) / "render",
            sources=[],
        )
        assert config.db_path.name == "index.db"
        assert "polylogue" in str(config.db_path)

    def test_config_db_path_is_captured_at_construction(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """Config snapshots db_path instead of resolving it on every access."""
        first_data = tmp_path / "data-a"
        second_data = tmp_path / "data-b"
        monkeypatch.setenv("XDG_DATA_HOME", str(first_data))
        config = Config(
            archive_root=tmp_path / "archive",
            render_root=tmp_path / "render",
            sources=[],
        )

        monkeypatch.setenv("XDG_DATA_HOME", str(second_data))

        assert config.db_path == first_data / "polylogue" / "index.db"

    def test_active_index_resolver_uses_index_db(self, tmp_path: Path) -> None:
        """index.db is the active query store."""
        from polylogue.paths import resolve_active_index_db_path

        db_anchor = tmp_path / "custom.sqlite"
        index_db = tmp_path / "index.db"
        db_anchor.write_text("unrelated")

        assert resolve_active_index_db_path(db_anchor=db_anchor, index_db=index_db) == index_db

    def test_active_index_resolver_ignores_sibling_index_for_overrides(self, tmp_path: Path) -> None:
        """Configured archive_root/index.db is authoritative."""
        from polylogue.paths import resolve_active_index_db_path

        override_root = tmp_path / "override"
        archive_root = tmp_path / "archive"
        override_root.mkdir()
        archive_root.mkdir()
        db_anchor = override_root / "custom.sqlite"
        sibling_index_db = override_root / "index.db"
        canonical_index_db = archive_root / "index.db"
        db_anchor.write_text("unrelated")
        sibling_index_db.write_text("index")

        assert resolve_active_index_db_path(db_anchor=db_anchor, index_db=canonical_index_db) == canonical_index_db

    def test_active_index_resolver_does_not_fall_back_to_db_anchor(self, tmp_path: Path) -> None:
        """Missing index.db remains the active target instead of a non-index anchor."""
        from polylogue.paths import resolve_active_index_db_path

        db_anchor = tmp_path / "custom.sqlite"

        assert (
            resolve_active_index_db_path(db_anchor=db_anchor, index_db=tmp_path / "index.db") == tmp_path / "index.db"
        )

    def test_archive_file_set_index_availability_is_unconditional(self, tmp_path: Path) -> None:
        """Archive file-set availability does not depend on the DB anchor."""
        from polylogue.paths import archive_file_set_index_available_for_paths, archive_file_set_root_for_paths

        archive_root = tmp_path / "archive"
        override_root = tmp_path / "override"
        archive_root.mkdir()
        override_root.mkdir()
        db_anchor = override_root / "custom.sqlite"

        assert archive_file_set_index_available_for_paths(archive_root_path=archive_root, db_anchor=db_anchor)
        assert archive_file_set_root_for_paths(archive_root_path=archive_root, db_anchor=db_anchor) == archive_root

        (override_root / "index.db").write_text("index")

        assert archive_file_set_index_available_for_paths(archive_root_path=archive_root, db_anchor=db_anchor)
        assert archive_file_set_root_for_paths(archive_root_path=archive_root, db_anchor=db_anchor) == archive_root

    def test_config_optional_fields_default_none(self, tmp_path: Path) -> None:
        """Optional fields default to None."""
        config = Config(
            archive_root=tmp_path,
            render_root=tmp_path / "render",
            sources=[],
        )
        assert config.drive_config is None
        assert config.index_config is None

    def test_config_rejects_relative_archive_root(self, tmp_path: Path) -> None:
        """Relative paths silently shift meaning across processes; reject at construction."""
        with pytest.raises(ConfigError, match="archive_root must be an absolute path"):
            Config(
                archive_root=Path("relative/archive"),
                render_root=tmp_path / "render",
                sources=[],
            )

    def test_config_rejects_relative_render_root(self, tmp_path: Path) -> None:
        with pytest.raises(ConfigError, match="render_root must be an absolute path"):
            Config(
                archive_root=tmp_path,
                render_root=Path("relative/render"),
                sources=[],
            )

    def test_config_rejects_relative_db_path(self, tmp_path: Path) -> None:
        with pytest.raises(ConfigError, match="db_path must be an absolute path"):
            Config(
                archive_root=tmp_path,
                render_root=tmp_path / "render",
                sources=[],
                db_path=Path("relative/db.sqlite"),
            )


class TestConfigError:
    """Tests for ConfigError exception."""

    def test_config_error_is_polylogue_error(self) -> None:
        """ConfigError inherits from PolylogueError."""
        from polylogue.core.errors import PolylogueError

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
        assert polylogue.paths.db_path().name == "index.db"


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

        from polylogue.config import get_sources

        sources = get_sources()
        assert [source.name for source in sources] == []

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

        from polylogue.config import get_sources

        sources = get_sources()
        assert [source.name for source in sources] == ["aistudio"]


class TestConfigPublicBoundary:
    def test_config_exports_configuration_models_and_loaders(self) -> None:
        import polylogue.config as config

        expected = {
            "Config",
            "ConfigError",
            "DriveConfig",
            "IndexConfig",
            "Source",
            "get_config",
            "get_drive_config",
            "get_index_config",
            "get_sources",
        }
        assert expected.issubset(set(config.__all__))


# =============================================================================
# PolylogueConfig TOML/env/CLI precedence tests (#829)
# =============================================================================


class TestPolylogueConfigDefaults:
    """PolylogueConfig returns sensible defaults with no TOML or env."""

    def test_archive_root_default(self, workspace_env: dict[str, Path]) -> None:
        from polylogue.config import load_polylogue_config

        cfg = load_polylogue_config()
        assert cfg.archive_root

    def test_api_host_default(self, workspace_env: dict[str, Path]) -> None:
        from polylogue.config import load_polylogue_config

        cfg = load_polylogue_config()
        assert cfg.api_host == "127.0.0.1"

    def test_api_port_default(self, workspace_env: dict[str, Path]) -> None:
        from polylogue.config import load_polylogue_config

        cfg = load_polylogue_config()
        assert cfg.api_port == 8766

    def test_embedding_disabled_by_default(self, workspace_env: dict[str, Path]) -> None:
        from polylogue.config import load_polylogue_config

        cfg = load_polylogue_config()
        assert cfg.embedding_enabled is False

    def test_embedding_model_default(self, workspace_env: dict[str, Path]) -> None:
        from polylogue.config import load_polylogue_config

        cfg = load_polylogue_config()
        assert cfg.embedding_model == "voyage-4"

    def test_embedding_dimension_default(self, workspace_env: dict[str, Path]) -> None:
        from polylogue.config import load_polylogue_config

        cfg = load_polylogue_config()
        assert cfg.embedding_dimension == 1024

    def test_embedding_max_cost_default(self, workspace_env: dict[str, Path]) -> None:
        from polylogue.config import load_polylogue_config

        cfg = load_polylogue_config()
        assert cfg.embedding_max_cost_usd == 5.0

    def test_notification_backend_default(self, workspace_env: dict[str, Path]) -> None:
        from polylogue.config import load_polylogue_config

        cfg = load_polylogue_config()
        assert cfg.notification_backend == "log"

    def test_health_interval_default(self, workspace_env: dict[str, Path]) -> None:
        from polylogue.config import load_polylogue_config

        cfg = load_polylogue_config()
        assert cfg.health_check_interval_s == 300

    def test_health_tiers_default(self, workspace_env: dict[str, Path]) -> None:
        from polylogue.config import load_polylogue_config

        cfg = load_polylogue_config()
        assert cfg.health_check_tiers == "fast"

    def test_source_roots_default(self, workspace_env: dict[str, Path]) -> None:
        from polylogue.config import load_polylogue_config

        cfg = load_polylogue_config()
        assert cfg.source_roots == ()

    def test_watch_debounce_default(self, workspace_env: dict[str, Path]) -> None:
        from polylogue.config import load_polylogue_config

        cfg = load_polylogue_config()
        assert cfg.watch_debounce_s == 2.0


class TestPolylogueConfigEnvOverrides:
    """POLYLOGUE_* env vars override defaults."""

    def test_env_overrides_api_host(self, monkeypatch: pytest.MonkeyPatch, workspace_env: dict[str, Path]) -> None:
        from polylogue.config import load_polylogue_config

        monkeypatch.setenv("POLYLOGUE_API_HOST", "0.0.0.0")
        cfg = load_polylogue_config()
        assert cfg.api_host == "0.0.0.0"

    def test_env_overrides_api_port(self, monkeypatch: pytest.MonkeyPatch, workspace_env: dict[str, Path]) -> None:
        from polylogue.config import load_polylogue_config

        monkeypatch.setenv("POLYLOGUE_API_PORT", "9999")
        cfg = load_polylogue_config()
        assert cfg.api_port == 9999

    def test_env_overrides_api_auth_token(
        self, monkeypatch: pytest.MonkeyPatch, workspace_env: dict[str, Path]
    ) -> None:
        from polylogue.config import load_polylogue_config

        monkeypatch.setenv("POLYLOGUE_API_AUTH_TOKEN", "secret-token")
        cfg = load_polylogue_config()
        assert cfg.api_auth_token == "secret-token"

    def test_env_overrides_embedding_enabled(
        self, monkeypatch: pytest.MonkeyPatch, workspace_env: dict[str, Path]
    ) -> None:
        from polylogue.config import load_polylogue_config

        monkeypatch.setenv("POLYLOGUE_DAEMON_ENABLE_EMBEDDINGS", "true")
        cfg = load_polylogue_config()
        assert cfg.embedding_enabled is True

    def test_env_overrides_notification_backend(
        self, monkeypatch: pytest.MonkeyPatch, workspace_env: dict[str, Path]
    ) -> None:
        from polylogue.config import load_polylogue_config

        monkeypatch.setenv("POLYLOGUE_NOTIFICATION_BACKEND", "stdout")
        cfg = load_polylogue_config()
        assert cfg.notification_backend == "stdout"

    def test_env_overrides_health_interval(
        self, monkeypatch: pytest.MonkeyPatch, workspace_env: dict[str, Path]
    ) -> None:
        from polylogue.config import load_polylogue_config

        monkeypatch.setenv("POLYLOGUE_HEALTH_CHECK_INTERVAL_S", "60")
        cfg = load_polylogue_config()
        assert cfg.health_check_interval_s == 60

    def test_env_overrides_health_tiers(self, monkeypatch: pytest.MonkeyPatch, workspace_env: dict[str, Path]) -> None:
        from polylogue.config import load_polylogue_config

        monkeypatch.setenv("POLYLOGUE_HEALTH_CHECK_TIERS", "fast")
        cfg = load_polylogue_config()
        assert cfg.health_check_tiers == "fast"

    def test_env_overrides_browser_capture_port(
        self, monkeypatch: pytest.MonkeyPatch, workspace_env: dict[str, Path]
    ) -> None:
        from polylogue.config import load_polylogue_config

        monkeypatch.setenv("POLYLOGUE_BROWSER_CAPTURE_PORT", "8888")
        cfg = load_polylogue_config()
        assert cfg.browser_capture_port == 8888

    def test_env_overrides_watch_debounce(
        self, monkeypatch: pytest.MonkeyPatch, workspace_env: dict[str, Path]
    ) -> None:
        from polylogue.config import load_polylogue_config

        monkeypatch.setenv("POLYLOGUE_WATCH_DEBOUNCE_S", "5.0")
        cfg = load_polylogue_config()
        assert cfg.watch_debounce_s == 5.0


class TestPolylogueConfigCLIOverrides:
    """CLI overrides take highest precedence."""

    def test_cli_overrides_api_port(self, workspace_env: dict[str, Path]) -> None:
        from polylogue.config import load_polylogue_config

        cfg = load_polylogue_config(cli_overrides={"api_port": 7777})
        assert cfg.api_port == 7777

    def test_cli_overrides_auth_token(self, workspace_env: dict[str, Path]) -> None:
        from polylogue.config import load_polylogue_config

        cfg = load_polylogue_config(cli_overrides={"api_auth_token": "cli-token"})
        assert cfg.api_auth_token == "cli-token"

    def test_cli_trumps_env(self, monkeypatch: pytest.MonkeyPatch, workspace_env: dict[str, Path]) -> None:
        from polylogue.config import load_polylogue_config

        monkeypatch.setenv("POLYLOGUE_API_PORT", "9999")
        cfg = load_polylogue_config(cli_overrides={"api_port": 7777})
        assert cfg.api_port == 7777

    def test_cli_overrides_embedding_max_cost(self, workspace_env: dict[str, Path]) -> None:
        from polylogue.config import load_polylogue_config

        cfg = load_polylogue_config(cli_overrides={"embedding_max_cost_usd": 10.0})
        assert cfg.embedding_max_cost_usd == 10.0


class TestPolylogueConfigTOML:
    """TOML config file overrides defaults but is overridden by env/CLI."""

    def test_toml_sets_api_port(self, tmp_path: Path, workspace_env: dict[str, Path]) -> None:
        from polylogue.config import load_polylogue_config

        toml_path = tmp_path / "polylogue.toml"
        toml_path.write_text(
            "[daemon.api]\nport = 9998\n",
            encoding="utf-8",
        )
        cfg = load_polylogue_config(config_path=toml_path)
        assert cfg.api_port == 9998

    def test_toml_sets_browser_capture(self, tmp_path: Path, workspace_env: dict[str, Path]) -> None:
        from polylogue.config import load_polylogue_config

        toml_path = tmp_path / "polylogue.toml"
        toml_path.write_text(
            '[daemon.browser_capture]\nhost = "0.0.0.0"\nport = 9997\n',
            encoding="utf-8",
        )
        cfg = load_polylogue_config(config_path=toml_path)
        assert cfg.browser_capture_host == "0.0.0.0"
        assert cfg.browser_capture_port == 9997

    def test_toml_sets_source_roots(self, tmp_path: Path, workspace_env: dict[str, Path]) -> None:
        from polylogue.config import load_polylogue_config

        toml_path = tmp_path / "polylogue.toml"
        toml_path.write_text(
            '[sources]\nroots = ["/tmp/extra", "/tmp/more"]\n',
            encoding="utf-8",
        )
        cfg = load_polylogue_config(config_path=toml_path)
        assert cfg.source_roots == ("/tmp/extra", "/tmp/more")

    def test_toml_env_cli_precedence(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, workspace_env: dict[str, Path]
    ) -> None:
        from polylogue.config import load_polylogue_config

        toml_path = tmp_path / "polylogue.toml"
        toml_path.write_text(
            "[daemon.api]\nport = 9000\n",
            encoding="utf-8",
        )
        monkeypatch.setenv("POLYLOGUE_API_PORT", "9001")
        cfg = load_polylogue_config(config_path=toml_path)
        # env overrides TOML
        assert cfg.api_port == 9001

        # CLI overrides env
        cfg2 = load_polylogue_config(config_path=toml_path, cli_overrides={"api_port": 9002})
        assert cfg2.api_port == 9002

    def test_toml_sets_embedding(self, tmp_path: Path, workspace_env: dict[str, Path]) -> None:
        from polylogue.config import load_polylogue_config

        toml_path = tmp_path / "polylogue.toml"
        toml_path.write_text(
            '[embedding]\nenabled = true\nmodel = "voyage-3"\ndimension = 512\n',
            encoding="utf-8",
        )
        cfg = load_polylogue_config(config_path=toml_path)
        assert cfg.embedding_enabled is True
        assert cfg.embedding_model == "voyage-3"
        assert cfg.embedding_dimension == 512


class TestPolylogueConfigFormatTOML:
    """format_config_toml produces loadable TOML."""

    def test_roundtrip_defaults(self) -> None:
        from polylogue.config import format_config_toml, load_polylogue_config

        cfg = load_polylogue_config()
        formatted = format_config_toml(cfg.raw)
        assert "[archive]" in formatted
        assert "[daemon]" in formatted

    def test_source_roots_formatted_as_array(self, workspace_env: dict[str, Path]) -> None:
        from polylogue.config import format_config_toml, load_polylogue_config

        cfg = load_polylogue_config(cli_overrides={"source_roots": ("/a", "/b")})
        formatted = format_config_toml(cfg.raw)
        # The TOML serializer renders arrays multi-line; assert the array shape
        # and both members rather than a single-line spelling.
        assert "roots = [" in formatted
        assert '"/a"' in formatted
        assert '"/b"' in formatted


class TestPolylogueConfigLayerPrecedence:
    """Five-layer precedence (default → site → user → env → cli) per #829."""

    def _disable_site(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Empty POLYLOGUE_SITE_CONFIG disables site discovery so tests do not
        # accidentally read /etc/polylogue/polylogue.toml from the host.
        monkeypatch.setenv("POLYLOGUE_SITE_CONFIG", "")

    def test_default_layer_marks_unset_keys(
        self,
        monkeypatch: pytest.MonkeyPatch,
        workspace_env: dict[str, Path],
    ) -> None:
        from polylogue.config import load_polylogue_config

        self._disable_site(monkeypatch)
        cfg = load_polylogue_config()
        assert cfg.layer_of("api_port") == "default"
        assert cfg.layer_of("embedding_model") == "default"

    def test_site_layer_supplies_value(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        workspace_env: dict[str, Path],
    ) -> None:
        from polylogue.config import load_polylogue_config

        self._disable_site(monkeypatch)
        site = tmp_path / "site.toml"
        site.write_text("[daemon.api]\nport = 8001\n", encoding="utf-8")
        cfg = load_polylogue_config(site_config_path=site)
        assert cfg.api_port == 8001
        assert cfg.layer_of("api_port") == "site"

    def test_user_overrides_site(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        workspace_env: dict[str, Path],
    ) -> None:
        from polylogue.config import load_polylogue_config

        self._disable_site(monkeypatch)
        site = tmp_path / "site.toml"
        site.write_text("[daemon.api]\nport = 8001\n", encoding="utf-8")
        user = tmp_path / "user.toml"
        user.write_text("[daemon.api]\nport = 8002\n", encoding="utf-8")
        cfg = load_polylogue_config(site_config_path=site, config_path=user)
        assert cfg.api_port == 8002
        assert cfg.layer_of("api_port") == "user"

    def test_env_overrides_user(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        workspace_env: dict[str, Path],
    ) -> None:
        from polylogue.config import load_polylogue_config

        self._disable_site(monkeypatch)
        user = tmp_path / "user.toml"
        user.write_text("[daemon.api]\nport = 8002\n", encoding="utf-8")
        monkeypatch.setenv("POLYLOGUE_API_PORT", "8003")
        cfg = load_polylogue_config(config_path=user)
        assert cfg.api_port == 8003
        assert cfg.layer_of("api_port") == "env"

    def test_cli_overrides_env(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        workspace_env: dict[str, Path],
    ) -> None:
        from polylogue.config import load_polylogue_config

        self._disable_site(monkeypatch)
        monkeypatch.setenv("POLYLOGUE_API_PORT", "8003")
        cfg = load_polylogue_config(cli_overrides={"api_port": 8004})
        assert cfg.api_port == 8004
        assert cfg.layer_of("api_port") == "cli"

    def test_layers_map_lists_every_default_key(
        self,
        monkeypatch: pytest.MonkeyPatch,
        workspace_env: dict[str, Path],
    ) -> None:
        from polylogue.config import load_polylogue_config

        self._disable_site(monkeypatch)
        cfg = load_polylogue_config()
        for key in cfg.raw:
            assert key in cfg.layers, f"missing layer source for {key}"

    def test_missing_site_file_falls_through_to_default(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        workspace_env: dict[str, Path],
    ) -> None:
        from polylogue.config import load_polylogue_config

        self._disable_site(monkeypatch)
        missing = tmp_path / "nonexistent.toml"
        cfg = load_polylogue_config(site_config_path=missing)
        assert cfg.layer_of("api_port") == "default"

    def test_malformed_toml_does_not_crash(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        workspace_env: dict[str, Path],
    ) -> None:
        from polylogue.config import load_polylogue_config

        self._disable_site(monkeypatch)
        bad = tmp_path / "bad.toml"
        bad.write_text("this is not valid toml = = = [\n", encoding="utf-8")
        cfg = load_polylogue_config(config_path=bad)
        # Falls back to defaults silently rather than blowing up the CLI.
        assert cfg.api_port == 8766
        assert cfg.layer_of("api_port") == "default"


class TestDescribeConfigLayers:
    """``describe_config_layers`` reports site/user paths and existence."""

    def test_describe_with_no_files(
        self,
        monkeypatch: pytest.MonkeyPatch,
        workspace_env: dict[str, Path],
    ) -> None:
        from polylogue.config import describe_config_layers

        monkeypatch.setenv("POLYLOGUE_SITE_CONFIG", "")
        monkeypatch.delenv("POLYLOGUE_CONFIG", raising=False)
        report = describe_config_layers()
        assert report["site"] == {"path": None, "exists": False}

    def test_describe_with_files_present(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        workspace_env: dict[str, Path],
    ) -> None:
        from polylogue.config import describe_config_layers

        site = tmp_path / "site.toml"
        site.write_text("", encoding="utf-8")
        user = tmp_path / "user.toml"
        user.write_text("", encoding="utf-8")
        monkeypatch.setenv("POLYLOGUE_SITE_CONFIG", str(site))
        monkeypatch.setenv("POLYLOGUE_CONFIG", str(user))
        report = describe_config_layers()
        site_info = report["site"]
        user_info = report["user"]
        assert isinstance(site_info, dict)
        assert isinstance(user_info, dict)
        assert site_info["path"] == str(site)
        assert site_info["exists"] is True
        assert user_info["path"] == str(user)
        assert user_info["exists"] is True

    def test_empty_site_env_var_disables_default_path(
        self,
        monkeypatch: pytest.MonkeyPatch,
        workspace_env: dict[str, Path],
    ) -> None:
        from polylogue.config import describe_config_layers

        monkeypatch.setenv("POLYLOGUE_SITE_CONFIG", "")
        report = describe_config_layers()
        assert report["site"] == {"path": None, "exists": False}


class TestConfigInventoryPayload:
    """The public config inventory must stay executable, not decorative."""

    def _disable_site(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("POLYLOGUE_SITE_CONFIG", "")

    def test_inventory_payload_has_unique_keys_and_env_vars(self) -> None:
        from polylogue.config import config_inventory_payload

        rows = config_inventory_payload()
        keys = [str(row["key"]) for row in rows]
        env_vars = [str(row["env_var"]) for row in rows if row.get("env_var")]

        assert len(keys) == len(set(keys))
        assert len(env_vars) == len(set(env_vars))
        assert all(row.get("owner_class") for row in rows)
        assert all(row.get("reload_behavior") for row in rows)
        assert all(row.get("effective_path") for row in rows)

    def test_effective_payload_redacts_secret_and_reports_env_source(
        self,
        monkeypatch: pytest.MonkeyPatch,
        workspace_env: dict[str, Path],
    ) -> None:
        from polylogue.config import effective_config_payload, load_polylogue_config

        self._disable_site(monkeypatch)
        monkeypatch.setenv("POLYLOGUE_API_AUTH_TOKEN", "secret-do-not-leak")
        cfg = load_polylogue_config()
        payload = effective_config_payload(cfg)

        values = payload["values"]
        assert isinstance(values, dict)
        token = values["api_auth_token"]
        assert isinstance(token, dict)
        assert token["value"] == "<set>"
        assert token["secret"] is True
        assert token["secret_present"] is True
        assert token["source_layer"] == "env"
        assert "secret-do-not-leak" not in str(payload)

    def test_effective_payload_uses_inventory_env_mapping_for_typed_values(
        self,
        monkeypatch: pytest.MonkeyPatch,
        workspace_env: dict[str, Path],
    ) -> None:
        from polylogue.config import effective_config_payload, load_polylogue_config

        self._disable_site(monkeypatch)
        monkeypatch.setenv("POLYLOGUE_BROWSER_CAPTURE_PORT", "9987")
        cfg = load_polylogue_config()
        payload = effective_config_payload(cfg)

        values = payload["values"]
        assert isinstance(values, dict)
        port = values["browser_capture_port"]
        assert isinstance(port, dict)
        assert port["value"] == 9987
        assert port["source_layer"] == "env"
        assert port["env_var"] == "POLYLOGUE_BROWSER_CAPTURE_PORT"


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
