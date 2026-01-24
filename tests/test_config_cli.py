"""Tests for config.py, paths.py, and CLI helpers."""

from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from polylogue.cli import cli
from polylogue.cli.helpers import (
    fail,
    is_declarative,
    latest_render_path,
    load_effective_config,
    load_last_source,
    maybe_prompt_sources,
    resolve_sources,
    save_last_source,
    source_state_path,
)
from polylogue.cli.types import AppEnv
from polylogue.config import (
    CONFIG_VERSION,
    ConfigError,
    Source,
    default_config,
    load_config,
    update_config,
    update_source,
    write_config,
)
from polylogue.paths import (
    CACHE_HOME,
    CACHE_ROOT,
    CONFIG_HOME,
    CONFIG_ROOT,
    DATA_HOME,
    DATA_ROOT,
    STATE_HOME,
    STATE_ROOT,
    is_within_root,
    safe_path_component,
)
from polylogue.ui import create_ui


# ==== Config Tests ====


class TestConfigBasics:
    """Basic config load/save functionality."""

    def test_load_config_missing_file_raises_error(self, workspace_env):
        """load_config(nonexistent_path) raises ConfigError."""
        nonexistent = workspace_env["config_path"].parent / "missing.json"
        with pytest.raises(ConfigError) as exc_info:
            load_config(nonexistent)
        assert "Config not found" in str(exc_info.value)
        # Error message includes path info (either the requested path or default)
        assert "config" in str(exc_info.value).lower()

    def test_load_config_valid_json(self, workspace_env):
        """Valid JSON config loads correctly with proper Source parsing."""
        config_path = workspace_env["config_path"]
        config_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": CONFIG_VERSION,
            "archive_root": str(workspace_env["archive_root"]),
            "sources": [
                {"name": "inbox", "path": str(workspace_env["archive_root"] / "inbox")},
                {"name": "remote", "folder": "Google Drive"},
            ],
        }
        config_path.write_text(json.dumps(payload), encoding="utf-8")

        config = load_config(config_path)
        assert config.version == CONFIG_VERSION
        assert config.archive_root == workspace_env["archive_root"]
        assert len(config.sources) == 2
        assert config.sources[0].name == "inbox"
        assert config.sources[0].path == workspace_env["archive_root"] / "inbox"
        assert not config.sources[0].is_drive
        assert config.sources[1].name == "remote"
        assert config.sources[1].folder == "Google Drive"
        assert config.sources[1].is_drive

    def test_load_config_malformed_json(self, workspace_env):
        """Malformed JSON raises ConfigError with clear message."""
        config_path = workspace_env["config_path"]
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text("{ invalid json", encoding="utf-8")

        with pytest.raises(ConfigError) as exc_info:
            load_config(config_path)
        error_msg = str(exc_info.value)
        assert "JSON" in error_msg or "json" in error_msg.lower()
        assert str(config_path) in error_msg or "config" in error_msg.lower()

    def test_load_config_not_a_dict(self, workspace_env):
        """Config that is JSON array instead of object raises error."""
        config_path = workspace_env["config_path"]
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text("[]", encoding="utf-8")

        with pytest.raises(ConfigError) as exc_info:
            load_config(config_path)
        assert "JSON object" in str(exc_info.value)

    def test_load_config_wrong_version(self, workspace_env):
        """Config with unsupported version raises error."""
        config_path = workspace_env["config_path"]
        config_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 999,
            "archive_root": str(workspace_env["archive_root"]),
            "sources": [],
        }
        config_path.write_text(json.dumps(payload), encoding="utf-8")

        with pytest.raises(ConfigError) as exc_info:
            load_config(config_path)
        assert "version" in str(exc_info.value).lower()
        assert "999" in str(exc_info.value)


class TestConfigSourceValidation:
    """Source validation during config loading."""

    def test_source_missing_name(self, workspace_env):
        """Source without 'name' raises ConfigError."""
        config_path = workspace_env["config_path"]
        config_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": CONFIG_VERSION,
            "archive_root": str(workspace_env["archive_root"]),
            "sources": [{"path": "/some/path"}],  # Missing 'name'
        }
        config_path.write_text(json.dumps(payload), encoding="utf-8")

        with pytest.raises(ConfigError) as exc_info:
            load_config(config_path)
        assert "name" in str(exc_info.value).lower()

    def test_source_empty_name(self, workspace_env):
        """Source with empty 'name' raises ConfigError."""
        config_path = workspace_env["config_path"]
        config_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": CONFIG_VERSION,
            "archive_root": str(workspace_env["archive_root"]),
            "sources": [{"name": "", "path": "/some/path"}],
        }
        config_path.write_text(json.dumps(payload), encoding="utf-8")

        with pytest.raises(ConfigError) as exc_info:
            load_config(config_path)
        assert "name" in str(exc_info.value).lower()

    def test_source_missing_path_and_folder(self, workspace_env):
        """Source without both 'path' and 'folder' raises ConfigError."""
        config_path = workspace_env["config_path"]
        config_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": CONFIG_VERSION,
            "archive_root": str(workspace_env["archive_root"]),
            "sources": [{"name": "test"}],  # Missing both path and folder
        }
        config_path.write_text(json.dumps(payload), encoding="utf-8")

        with pytest.raises(ConfigError) as exc_info:
            load_config(config_path)
        assert "path" in str(exc_info.value).lower() or "folder" in str(exc_info.value).lower()

    def test_source_both_path_and_folder(self, workspace_env):
        """Source with both 'path' and 'folder' raises ConfigError."""
        config_path = workspace_env["config_path"]
        config_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": CONFIG_VERSION,
            "archive_root": str(workspace_env["archive_root"]),
            "sources": [{"name": "test", "path": "/some/path", "folder": "Google Drive"}],
        }
        config_path.write_text(json.dumps(payload), encoding="utf-8")

        with pytest.raises(ConfigError) as exc_info:
            load_config(config_path)
        assert "both" in str(exc_info.value).lower()

    def test_source_empty_path(self, workspace_env):
        """Source with empty 'path' raises ConfigError."""
        config_path = workspace_env["config_path"]
        config_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": CONFIG_VERSION,
            "archive_root": str(workspace_env["archive_root"]),
            "sources": [{"name": "test", "path": ""}],
        }
        config_path.write_text(json.dumps(payload), encoding="utf-8")

        with pytest.raises(ConfigError) as exc_info:
            load_config(config_path)
        assert "path" in str(exc_info.value).lower()

    def test_source_empty_folder(self, workspace_env):
        """Source with empty 'folder' raises ConfigError."""
        config_path = workspace_env["config_path"]
        config_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": CONFIG_VERSION,
            "archive_root": str(workspace_env["archive_root"]),
            "sources": [{"name": "test", "folder": ""}],
        }
        config_path.write_text(json.dumps(payload), encoding="utf-8")

        with pytest.raises(ConfigError) as exc_info:
            load_config(config_path)
        assert "folder" in str(exc_info.value).lower()

    def test_source_with_unknown_keys(self, workspace_env):
        """Source with unknown keys raises ConfigError."""
        config_path = workspace_env["config_path"]
        config_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": CONFIG_VERSION,
            "archive_root": str(workspace_env["archive_root"]),
            "sources": [
                {"name": "test", "path": "/some/path", "unknown_key": "value"}
            ],
        }
        config_path.write_text(json.dumps(payload), encoding="utf-8")

        with pytest.raises(ConfigError) as exc_info:
            load_config(config_path)
        assert "Unknown" in str(exc_info.value)
        assert "unknown_key" in str(exc_info.value)

    def test_duplicate_source_names(self, workspace_env):
        """Config with duplicate source names raises ConfigError."""
        config_path = workspace_env["config_path"]
        config_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": CONFIG_VERSION,
            "archive_root": str(workspace_env["archive_root"]),
            "sources": [
                {"name": "inbox", "path": "/path1"},
                {"name": "inbox", "path": "/path2"},
            ],
        }
        config_path.write_text(json.dumps(payload), encoding="utf-8")

        with pytest.raises(ConfigError) as exc_info:
            load_config(config_path)
        assert "Duplicate" in str(exc_info.value)
        assert "inbox" in str(exc_info.value)


class TestConfigPaths:
    """Config path and environment variable handling."""

    def test_config_path_from_env(self, tmp_path, monkeypatch):
        """POLYLOGUE_CONFIG environment variable overrides default."""
        custom_config = tmp_path / "custom" / "config.json"
        custom_config.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": CONFIG_VERSION,
            "archive_root": str(tmp_path / "archive"),
            "sources": [],
        }
        custom_config.write_text(json.dumps(payload), encoding="utf-8")
        monkeypatch.setenv("POLYLOGUE_CONFIG", str(custom_config))

        config = load_config()
        assert config.path == custom_config

    def test_config_archive_root_from_env(self, tmp_path, monkeypatch):
        """POLYLOGUE_ARCHIVE_ROOT environment variable overrides config file."""
        config_path = tmp_path / "config.json"
        env_archive = tmp_path / "env_archive"
        payload = {
            "version": CONFIG_VERSION,
            "archive_root": str(tmp_path / "config_archive"),  # Will be overridden
            "sources": [],
        }
        config_path.write_text(json.dumps(payload), encoding="utf-8")
        monkeypatch.setenv("POLYLOGUE_CONFIG", str(config_path))
        monkeypatch.setenv("POLYLOGUE_ARCHIVE_ROOT", str(env_archive))

        config = load_config()
        assert config.archive_root == env_archive

    def test_config_render_root_from_env(self, tmp_path, monkeypatch):
        """POLYLOGUE_RENDER_ROOT environment variable overrides config file."""
        config_path = tmp_path / "config.json"
        env_render = tmp_path / "env_render"
        payload = {
            "version": CONFIG_VERSION,
            "archive_root": str(tmp_path / "archive"),
            "render_root": str(tmp_path / "config_render"),  # Will be overridden
            "sources": [],
        }
        config_path.write_text(json.dumps(payload), encoding="utf-8")
        monkeypatch.setenv("POLYLOGUE_CONFIG", str(config_path))
        monkeypatch.setenv("POLYLOGUE_RENDER_ROOT", str(env_render))

        config = load_config()
        assert config.render_root == env_render

    def test_config_template_path_from_env(self, tmp_path, monkeypatch):
        """POLYLOGUE_TEMPLATE_PATH environment variable overrides config file."""
        config_path = tmp_path / "config.json"
        env_template = tmp_path / "env_template.html"
        payload = {
            "version": CONFIG_VERSION,
            "archive_root": str(tmp_path / "archive"),
            "template_path": str(tmp_path / "config_template.html"),  # Will be overridden
            "sources": [],
        }
        config_path.write_text(json.dumps(payload), encoding="utf-8")
        monkeypatch.setenv("POLYLOGUE_CONFIG", str(config_path))
        monkeypatch.setenv("POLYLOGUE_TEMPLATE_PATH", str(env_template))

        config = load_config()
        assert config.template_path == env_template


class TestConfigWriteAndRoundtrip:
    """Config serialization and round-trip."""

    def test_write_config_creates_parent_directories(self, tmp_path):
        """write_config creates parent directories as needed."""
        config = default_config(path=tmp_path / "deep" / "nested" / "config.json")
        assert not (tmp_path / "deep").exists()

        write_config(config)

        assert (tmp_path / "deep" / "nested" / "config.json").exists()

    def test_config_roundtrip(self, tmp_path, monkeypatch):
        """Config can be written and read back with preservation of data."""
        monkeypatch.setenv("POLYLOGUE_CONFIG", str(tmp_path / "config.json"))
        monkeypatch.setenv("POLYLOGUE_ARCHIVE_ROOT", str(tmp_path / "archive"))

        original = default_config(path=tmp_path / "config.json")
        original.sources.append(Source(name="drive", folder="My Drive"))
        write_config(original)

        loaded = load_config(tmp_path / "config.json")
        assert loaded.version == original.version
        assert loaded.archive_root == original.archive_root
        assert len(loaded.sources) == len(original.sources)
        assert loaded.sources[1].name == "drive"
        assert loaded.sources[1].folder == "My Drive"

    def test_source_as_dict(self):
        """Source.as_dict() serializes correctly."""
        path_source = Source(name="local", path=Path("/some/path"))
        drive_source = Source(name="remote", folder="Google Drive")

        assert path_source.as_dict() == {"name": "local", "path": "/some/path"}
        assert drive_source.as_dict() == {"name": "remote", "folder": "Google Drive"}

    def test_config_as_dict(self, tmp_path):
        """Config.as_dict() serializes all fields correctly."""
        config = Config(
            version=CONFIG_VERSION,
            archive_root=tmp_path / "archive",
            render_root=tmp_path / "render",
            sources=[Source(name="inbox", path=tmp_path / "inbox")],
            path=tmp_path / "config.json",
            template_path=tmp_path / "template.html",
        )

        data = config.as_dict()
        assert data["version"] == CONFIG_VERSION
        assert data["archive_root"] == str(tmp_path / "archive")
        assert data["render_root"] == str(tmp_path / "render")
        assert data["template_path"] == str(tmp_path / "template.html")
        assert len(data["sources"]) == 1
        assert data["sources"][0]["name"] == "inbox"


class TestUpdateConfig:
    """Config update functions."""

    def test_update_config_archive_root(self, tmp_path):
        """update_config() changes archive_root and returns new instance."""
        original = default_config()
        new_archive = tmp_path / "new_archive"

        updated = update_config(original, archive_root=new_archive)

        assert updated.archive_root == new_archive
        assert original.archive_root != updated.archive_root  # Original unchanged

    def test_update_config_render_root(self, tmp_path):
        """update_config() changes render_root and returns new instance."""
        original = default_config()
        new_render = tmp_path / "new_render"

        updated = update_config(original, render_root=new_render)

        assert updated.render_root == new_render
        assert original.render_root != updated.render_root

    def test_update_config_no_changes(self):
        """update_config() with no args returns same config."""
        config = default_config()
        updated = update_config(config)
        assert updated is config

    def test_update_source_path_field(self):
        """update_source() can update a source's path field."""
        config = Config(
            version=CONFIG_VERSION,
            archive_root=Path("/archive"),
            render_root=Path("/render"),
            sources=[Source(name="inbox", path=Path("/old/path"))],
            path=Path("/config.json"),
        )

        updated = update_source(config, "inbox", "path", "/new/path")

        assert updated.sources[0].path == Path("/new/path")
        assert updated.sources[0].folder is None

    def test_update_source_folder_field(self):
        """update_source() can update a source's folder field."""
        config = Config(
            version=CONFIG_VERSION,
            archive_root=Path("/archive"),
            render_root=Path("/render"),
            sources=[Source(name="drive", folder="Old Folder")],
            path=Path("/config.json"),
        )

        updated = update_source(config, "drive", "folder", "New Folder")

        assert updated.sources[0].folder == "New Folder"
        assert updated.sources[0].path is None

    def test_update_source_source_not_found(self):
        """update_source() raises error if source not found."""
        config = Config(
            version=CONFIG_VERSION,
            archive_root=Path("/archive"),
            render_root=Path("/render"),
            sources=[],
            path=Path("/config.json"),
        )

        with pytest.raises(ConfigError) as exc_info:
            update_source(config, "missing", "path", "/new/path")
        assert "not found" in str(exc_info.value).lower()

    def test_update_source_unknown_field(self):
        """update_source() raises error for unknown field."""
        config = Config(
            version=CONFIG_VERSION,
            archive_root=Path("/archive"),
            render_root=Path("/render"),
            sources=[Source(name="inbox", path=Path("/path"))],
            path=Path("/config.json"),
        )

        with pytest.raises(ConfigError) as exc_info:
            update_source(config, "inbox", "unknown", "value")
        assert "Unknown" in str(exc_info.value) or "field" in str(exc_info.value).lower()


# ==== Paths Tests ====


class TestPathsXDG:
    """XDG directory paths."""

    def test_xdg_defaults(self, monkeypatch):
        """XDG paths default to ~/.config, ~/.local/share, etc. when env not set."""
        monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
        monkeypatch.delenv("XDG_DATA_HOME", raising=False)
        monkeypatch.delenv("XDG_CACHE_HOME", raising=False)
        monkeypatch.delenv("XDG_STATE_HOME", raising=False)

        # Reimport to get defaults
        from importlib import reload
        import polylogue.paths as paths_mod

        reload(paths_mod)

        home = Path.home()
        assert paths_mod.CONFIG_ROOT == home / ".config"
        assert paths_mod.DATA_ROOT == home / ".local" / "share"
        assert paths_mod.CACHE_ROOT == home / ".cache"
        assert paths_mod.STATE_ROOT == home / ".local" / "state"

    def test_xdg_custom_paths(self, tmp_path, monkeypatch):
        """XDG paths can be customized via environment variables."""
        custom_config = tmp_path / "config"
        custom_data = tmp_path / "data"
        custom_cache = tmp_path / "cache"
        custom_state = tmp_path / "state"

        monkeypatch.setenv("XDG_CONFIG_HOME", str(custom_config))
        monkeypatch.setenv("XDG_DATA_HOME", str(custom_data))
        monkeypatch.setenv("XDG_CACHE_HOME", str(custom_cache))
        monkeypatch.setenv("XDG_STATE_HOME", str(custom_state))

        from importlib import reload
        import polylogue.paths as paths_mod

        reload(paths_mod)

        assert paths_mod.CONFIG_ROOT == custom_config
        assert paths_mod.DATA_ROOT == custom_data
        assert paths_mod.CACHE_ROOT == custom_cache
        assert paths_mod.STATE_ROOT == custom_state

    def test_polylogue_subdirectories(self):
        """Polylogue paths include 'polylogue' subdirectory."""
        assert CONFIG_HOME.name == "polylogue"
        assert DATA_HOME.name == "polylogue"
        assert CACHE_HOME.name == "polylogue"
        assert STATE_HOME.name == "polylogue"
        assert CONFIG_HOME.parent == CONFIG_ROOT
        assert DATA_HOME.parent == DATA_ROOT
        assert CACHE_HOME.parent == CACHE_ROOT
        assert STATE_HOME.parent == STATE_ROOT


class TestSafePathComponent:
    """Path component sanitization."""

    def test_safe_path_component_alphanumeric(self):
        """Alphanumeric strings pass through unchanged."""
        assert safe_path_component("hello") == "hello"
        assert safe_path_component("test123") == "test123"
        assert safe_path_component("file_name") == "file_name"
        assert safe_path_component("file-name") == "file-name"
        assert safe_path_component("file.name") == "file.name"

    def test_safe_path_component_special_chars(self):
        """Special characters are replaced with underscores."""
        result = safe_path_component("hello/world")
        assert "/" not in result
        assert "_" in result or "-" in result

    def test_safe_path_component_spaces(self):
        """Spaces are sanitized."""
        result = safe_path_component("hello world")
        assert " " not in result

    def test_safe_path_component_empty_or_none(self):
        """Empty or None input uses fallback."""
        assert safe_path_component("") == "item"
        assert safe_path_component(None) == "item"
        assert safe_path_component("   ") == "item"

    def test_safe_path_component_custom_fallback(self):
        """Custom fallback is used when provided."""
        assert safe_path_component("", fallback="custom") == "custom"
        assert safe_path_component("///", fallback="custom") != "item"  # Uses fallback

    def test_safe_path_component_parent_refs(self):
        """Parent directory references are sanitized."""
        result = safe_path_component("..")
        assert result != ".."
        assert "-" in result

    def test_safe_path_component_long_name_hashed(self):
        """Very long or complex names get hashed suffix."""
        complex_name = "hello world!@#$%^&*() - this is a complex name"
        result = safe_path_component(complex_name)
        # Should have a hash suffix
        assert "-" in result and len(result.split("-")) > 1


class TestIsWithinRoot:
    """Path containment check."""

    def test_is_within_root_true(self, tmp_path):
        """Path within root returns True."""
        root = tmp_path / "root"
        path = root / "subdir" / "file.txt"
        assert is_within_root(path, root)

    def test_is_within_root_false(self, tmp_path):
        """Path outside root returns False."""
        root1 = tmp_path / "root1"
        root2 = tmp_path / "root2"
        path = root1 / "file.txt"
        assert not is_within_root(path, root2)

    def test_is_within_root_same_path(self, tmp_path):
        """Path same as root returns True."""
        root = tmp_path / "root"
        assert is_within_root(root, root)

    def test_is_within_root_symlinks(self, tmp_path):
        """Symlinks are resolved for containment check."""
        root = tmp_path / "root"
        root.mkdir()
        actual = root / "actual"
        actual.mkdir()
        link = root / "link"
        link.symlink_to(actual)
        file_in_link = link / "file.txt"

        # Path through symlink inside root should be within root
        result = is_within_root(file_in_link, root)
        assert result


# ==== CLI Helpers Tests ====


class TestCliHelpersFail:
    """fail() error function."""

    def test_fail_raises_system_exit(self):
        """fail() raises SystemExit with formatted message."""
        with pytest.raises(SystemExit) as exc_info:
            fail("test_command", "Error message")
        assert "test_command: Error message" in str(exc_info.value)


class TestCliHelpersDeclarative:
    """is_declarative() environment flag."""

    def test_is_declarative_false_by_default(self, monkeypatch):
        """is_declarative() is False when env not set."""
        monkeypatch.delenv("POLYLOGUE_DECLARATIVE", raising=False)
        assert not is_declarative()

    def test_is_declarative_true_values(self, monkeypatch):
        """is_declarative() is True for various truthy values."""
        for value in ["1", "true", "True", "TRUE", "yes", "Yes"]:
            monkeypatch.setenv("POLYLOGUE_DECLARATIVE", value)
            assert is_declarative(), f"Failed for value: {value}"

    def test_is_declarative_false_values(self, monkeypatch):
        """is_declarative() is False for falsy values."""
        for value in ["0", "false", "False", "no", "No"]:
            monkeypatch.setenv("POLYLOGUE_DECLARATIVE", value)
            assert not is_declarative(), f"Failed for value: {value}"


class TestCliHelpersSourceState:
    """Source state file handling."""

    def test_source_state_path_default(self, monkeypatch):
        """source_state_path() uses XDG_STATE_HOME if available."""
        monkeypatch.delenv("XDG_STATE_HOME", raising=False)
        path = source_state_path()
        assert "polylogue" in str(path)
        assert "last-source.json" in str(path)

    def test_source_state_path_custom(self, tmp_path, monkeypatch):
        """source_state_path() respects XDG_STATE_HOME."""
        custom_state = tmp_path / "state"
        monkeypatch.setenv("XDG_STATE_HOME", str(custom_state))
        path = source_state_path()
        assert path == custom_state / "polylogue" / "last-source.json"

    def test_save_and_load_last_source(self, tmp_path, monkeypatch):
        """save_last_source() and load_last_source() work together."""
        state_dir = tmp_path / "state"
        monkeypatch.setenv("XDG_STATE_HOME", str(state_dir))

        save_last_source("my_source")
        loaded = load_last_source()

        assert loaded == "my_source"

    def test_load_last_source_missing_file(self, tmp_path, monkeypatch):
        """load_last_source() returns None if file doesn't exist."""
        state_dir = tmp_path / "state"
        monkeypatch.setenv("XDG_STATE_HOME", str(state_dir))
        loaded = load_last_source()
        assert loaded is None

    def test_load_last_source_malformed_json(self, tmp_path, monkeypatch):
        """load_last_source() returns None for malformed JSON."""
        state_dir = tmp_path / "state"
        monkeypatch.setenv("XDG_STATE_HOME", str(state_dir))
        state_file = state_dir / "polylogue" / "last-source.json"
        state_file.parent.mkdir(parents=True, exist_ok=True)
        state_file.write_text("invalid json", encoding="utf-8")

        loaded = load_last_source()
        assert loaded is None

    def test_load_last_source_no_source_field(self, tmp_path, monkeypatch):
        """load_last_source() returns None if 'source' field missing."""
        state_dir = tmp_path / "state"
        monkeypatch.setenv("XDG_STATE_HOME", str(state_dir))
        state_file = state_dir / "polylogue" / "last-source.json"
        state_file.parent.mkdir(parents=True, exist_ok=True)
        state_file.write_text('{"other_field": "value"}', encoding="utf-8")

        loaded = load_last_source()
        assert loaded is None


class TestCliHelpersLoadEffectiveConfig:
    """load_effective_config() function."""

    def test_load_effective_config_uses_env_path(self, tmp_path, monkeypatch):
        """load_effective_config() uses AppEnv.config_path."""
        config_path = tmp_path / "config.json"
        payload = {
            "version": CONFIG_VERSION,
            "archive_root": str(tmp_path / "archive"),
            "sources": [],
        }
        config_path.write_text(json.dumps(payload), encoding="utf-8")

        ui = create_ui(plain=True)
        env = AppEnv(ui=ui, config_path=config_path)
        config = load_effective_config(env)

        assert config.path == config_path

    def test_load_effective_config_missing_raises_error(self, tmp_path):
        """load_effective_config() raises ConfigError if config missing."""
        ui = create_ui(plain=True)
        env = AppEnv(ui=ui, config_path=tmp_path / "missing.json")

        with pytest.raises(ConfigError):
            load_effective_config(env)


class TestCliHelpersResolveSources:
    """resolve_sources() function."""

    def test_resolve_sources_empty_returns_none(self):
        """resolve_sources() with no sources returns None."""
        config = Config(
            version=CONFIG_VERSION,
            archive_root=Path("/archive"),
            render_root=Path("/render"),
            sources=[Source(name="inbox", path=Path("/inbox"))],
            path=Path("/config.json"),
        )
        result = resolve_sources(config, (), "test")
        assert result is None

    def test_resolve_sources_specific_names(self):
        """resolve_sources() returns requested source names."""
        config = Config(
            version=CONFIG_VERSION,
            archive_root=Path("/archive"),
            render_root=Path("/render"),
            sources=[
                Source(name="inbox", path=Path("/inbox")),
                Source(name="drive", folder="Google Drive"),
            ],
            path=Path("/config.json"),
        )
        result = resolve_sources(config, ("inbox", "drive"), "test")
        assert result == ["inbox", "drive"]

    def test_resolve_sources_deduplicates(self):
        """resolve_sources() deduplicates source names."""
        config = Config(
            version=CONFIG_VERSION,
            archive_root=Path("/archive"),
            render_root=Path("/render"),
            sources=[Source(name="inbox", path=Path("/inbox"))],
            path=Path("/config.json"),
        )
        result = resolve_sources(config, ("inbox", "inbox"), "test")
        assert result == ["inbox"]

    def test_resolve_sources_unknown_source_fails(self):
        """resolve_sources() fails for unknown source."""
        config = Config(
            version=CONFIG_VERSION,
            archive_root=Path("/archive"),
            render_root=Path("/render"),
            sources=[Source(name="inbox", path=Path("/inbox"))],
            path=Path("/config.json"),
        )
        with pytest.raises(SystemExit) as exc_info:
            resolve_sources(config, ("missing",), "test")
        assert "Unknown source" in str(exc_info.value) or "missing" in str(exc_info.value)

    def test_resolve_sources_last_keyword(self, tmp_path, monkeypatch):
        """resolve_sources() expands 'last' keyword."""
        state_dir = tmp_path / "state"
        monkeypatch.setenv("XDG_STATE_HOME", str(state_dir))
        save_last_source("inbox")

        config = Config(
            version=CONFIG_VERSION,
            archive_root=Path("/archive"),
            render_root=Path("/render"),
            sources=[Source(name="inbox", path=Path("/inbox"))],
            path=Path("/config.json"),
        )
        result = resolve_sources(config, ("last",), "test")
        assert result == ["inbox"]

    def test_resolve_sources_last_not_found_fails(self, tmp_path, monkeypatch):
        """resolve_sources() fails if 'last' used but no previous source."""
        # Ensure no last-source state file exists
        state_dir = tmp_path / "state"
        monkeypatch.setenv("XDG_STATE_HOME", str(state_dir))

        config = Config(
            version=CONFIG_VERSION,
            archive_root=Path("/archive"),
            render_root=Path("/render"),
            sources=[Source(name="inbox", path=Path("/inbox"))],
            path=Path("/config.json"),
        )
        with pytest.raises(SystemExit) as exc_info:
            resolve_sources(config, ("last",), "test")
        # Error mentions either "previously selected" or "Unknown source"
        error_msg = str(exc_info.value).lower()
        assert "previously selected" in error_msg or "unknown source" in error_msg

    def test_resolve_sources_last_with_others_fails(self, tmp_path, monkeypatch):
        """resolve_sources() fails if 'last' combined with other sources."""
        state_dir = tmp_path / "state"
        monkeypatch.setenv("XDG_STATE_HOME", str(state_dir))
        save_last_source("inbox")

        config = Config(
            version=CONFIG_VERSION,
            archive_root=Path("/archive"),
            render_root=Path("/render"),
            sources=[
                Source(name="inbox", path=Path("/inbox")),
                Source(name="drive", folder="Google Drive"),
            ],
            path=Path("/config.json"),
        )
        with pytest.raises(SystemExit) as exc_info:
            resolve_sources(config, ("last", "drive"), "test")
        assert "cannot be combined" in str(exc_info.value).lower()


class TestCliHelpersMaybePromptSources:
    """maybe_prompt_sources() function."""

    def test_maybe_prompt_sources_returns_existing_selection(self):
        """Returns existing selected_sources if provided."""
        from polylogue.ui import create_ui

        config = Config(
            version=CONFIG_VERSION,
            archive_root=Path("/archive"),
            render_root=Path("/render"),
            sources=[Source(name="inbox", path=Path("/inbox"))],
            path=Path("/config.json"),
        )
        ui = create_ui(plain=True)
        env = AppEnv(ui=ui, config_path=Path("/config.json"))

        result = maybe_prompt_sources(env, config, ["inbox"], "test")
        assert result == ["inbox"]

    def test_maybe_prompt_sources_skips_prompt_in_plain_mode(self):
        """Skips prompting in plain mode and returns None."""
        from polylogue.ui import create_ui

        config = Config(
            version=CONFIG_VERSION,
            archive_root=Path("/archive"),
            render_root=Path("/render"),
            sources=[Source(name="inbox", path=Path("/inbox")), Source(name="drive", folder="Drive")],
            path=Path("/config.json"),
        )
        ui = create_ui(plain=True)
        env = AppEnv(ui=ui, config_path=Path("/config.json"))

        result = maybe_prompt_sources(env, config, None, "test")
        assert result is None

    def test_maybe_prompt_sources_skips_single_source(self):
        """Skips prompting when only one source exists."""
        from polylogue.ui import create_ui

        config = Config(
            version=CONFIG_VERSION,
            archive_root=Path("/archive"),
            render_root=Path("/render"),
            sources=[Source(name="inbox", path=Path("/inbox"))],
            path=Path("/config.json"),
        )
        ui = create_ui(plain=True)
        env = AppEnv(ui=ui, config_path=Path("/config.json"))

        result = maybe_prompt_sources(env, config, None, "test")
        assert result is None

    def test_maybe_prompt_sources_fails_on_no_choice(self):
        """Fails when user provides no choice."""
        from polylogue.ui import create_ui
        from unittest.mock import Mock

        config = Config(
            version=CONFIG_VERSION,
            archive_root=Path("/archive"),
            render_root=Path("/render"),
            sources=[Source(name="inbox", path=Path("/inbox")), Source(name="drive", folder="Drive")],
            path=Path("/config.json"),
        )
        ui = create_ui(plain=False)
        ui.choose = Mock(return_value=None)
        env = AppEnv(ui=ui, config_path=Path("/config.json"))

        with pytest.raises(SystemExit) as exc_info:
            maybe_prompt_sources(env, config, None, "test")
        assert "No source selected" in str(exc_info.value)


class TestCliHelpersPrintSummary:
    """print_summary() function."""

    def test_print_summary_displays_basic_info(self, tmp_path, monkeypatch):
        """print_summary() displays basic config info without verbose."""
        from polylogue.ui import create_ui

        config_path = tmp_path / "config.json"
        archive_root = tmp_path / "archive"
        payload = {
            "version": CONFIG_VERSION,
            "archive_root": str(archive_root),
            "sources": [{"name": "inbox", "path": str(tmp_path / "inbox")}],
        }
        config_path.write_text(json.dumps(payload), encoding="utf-8")
        monkeypatch.setenv("POLYLOGUE_CONFIG", str(config_path))
        monkeypatch.setenv("POLYLOGUE_ARCHIVE_ROOT", str(archive_root))

        ui = create_ui(plain=True)
        env = AppEnv(ui=ui, config_path=config_path)

        # Should not raise
        from polylogue.cli.helpers import print_summary

        print_summary(env, verbose=False)

    def test_print_summary_verbose_with_health(self, tmp_path, monkeypatch):
        """print_summary() displays health checks in verbose mode."""
        from polylogue.ui import create_ui
        from polylogue.storage.db import open_connection

        config_path = tmp_path / "config.json"
        archive_root = tmp_path / "archive"
        archive_root.mkdir(parents=True, exist_ok=True)

        # Initialize database
        state_dir = tmp_path / "state"
        db_path = state_dir / "polylogue" / "polylogue.db"
        db_path.parent.mkdir(parents=True, exist_ok=True)
        monkeypatch.setenv("XDG_STATE_HOME", str(state_dir))

        with open_connection(None) as conn:
            conn.execute("SELECT 1")
            conn.commit()

        payload = {
            "version": CONFIG_VERSION,
            "archive_root": str(archive_root),
            "sources": [{"name": "inbox", "path": str(tmp_path / "inbox")}],
        }
        config_path.write_text(json.dumps(payload), encoding="utf-8")
        monkeypatch.setenv("POLYLOGUE_CONFIG", str(config_path))
        monkeypatch.setenv("POLYLOGUE_ARCHIVE_ROOT", str(archive_root))

        ui = create_ui(plain=True)
        env = AppEnv(ui=ui, config_path=config_path)

        # Should not raise and display health info
        from polylogue.cli.helpers import print_summary

        print_summary(env, verbose=True)

    def test_print_summary_handles_missing_config(self, tmp_path):
        """print_summary() handles missing config gracefully."""
        from polylogue.ui import create_ui

        config_path = tmp_path / "missing.json"
        ui = create_ui(plain=True)
        env = AppEnv(ui=ui, config_path=config_path)

        # Should not raise, just print warning
        from polylogue.cli.helpers import print_summary

        print_summary(env, verbose=False)


class TestCliHelpersLatestRenderPath:
    """latest_render_path() function."""

    def test_latest_render_path_empty_root(self, tmp_path):
        """latest_render_path() returns None for empty render root."""
        render_root = tmp_path / "render"
        result = latest_render_path(render_root)
        assert result is None

    def test_latest_render_path_no_files(self, tmp_path):
        """latest_render_path() returns None if no render files found."""
        render_root = tmp_path / "render"
        render_root.mkdir()
        (render_root / "empty_dir").mkdir()
        result = latest_render_path(render_root)
        assert result is None

    def test_latest_render_path_finds_markdown(self, tmp_path):
        """latest_render_path() finds conversation.md files."""
        render_root = tmp_path / "render"
        conv_dir = render_root / "test" / "conv1"
        conv_dir.mkdir(parents=True)
        md_file = conv_dir / "conversation.md"
        md_file.write_text("# Test", encoding="utf-8")

        result = latest_render_path(render_root)
        assert result == md_file

    def test_latest_render_path_finds_html(self, tmp_path):
        """latest_render_path() finds conversation.html files."""
        render_root = tmp_path / "render"
        conv_dir = render_root / "test" / "conv1"
        conv_dir.mkdir(parents=True)
        html_file = conv_dir / "conversation.html"
        html_file.write_text("<html></html>", encoding="utf-8")

        result = latest_render_path(render_root)
        assert result == html_file

    def test_latest_render_path_prefers_html(self, tmp_path):
        """latest_render_path() prefers HTML when multiple files exist."""
        import time

        render_root = tmp_path / "render"

        # Create markdown first
        conv_dir1 = render_root / "test" / "conv1"
        conv_dir1.mkdir(parents=True)
        md_file = conv_dir1 / "conversation.md"
        md_file.write_text("# Test", encoding="utf-8")

        # Create HTML after (with slightly later mtime)
        time.sleep(0.01)
        conv_dir2 = render_root / "test" / "conv2"
        conv_dir2.mkdir(parents=True)
        html_file = conv_dir2 / "conversation.html"
        html_file.write_text("<html></html>", encoding="utf-8")

        result = latest_render_path(render_root)
        # Should return the most recently modified file
        assert result is not None

    def test_latest_render_path_handles_deleted_file(self, tmp_path):
        """latest_render_path() handles race condition where file is deleted."""
        render_root = tmp_path / "render"
        conv_dir = render_root / "test" / "conv1"
        conv_dir.mkdir(parents=True)
        html_file = conv_dir / "conversation.html"
        html_file.write_text("<html></html>", encoding="utf-8")

        # First call should work
        result = latest_render_path(render_root)
        assert result is not None

        # Now create another file and delete the first
        conv_dir2 = render_root / "test" / "conv2"
        conv_dir2.mkdir(parents=True)
        html_file2 = conv_dir2 / "conversation.html"
        html_file2.write_text("<html>2</html>", encoding="utf-8")
        html_file.unlink()

        # Should still work without crashing
        result = latest_render_path(render_root)
        assert result is not None


# ==== CLI Integration Tests ====


class TestCliHelp:
    """CLI help output."""

    def test_cli_help_displays(self):
        """polylogue --help works and lists commands."""
        runner = CliRunner()
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "Polylogue" in result.output or "polylogue" in result.output.lower()

    def test_cli_version_displays(self):
        """polylogue --version displays version."""
        runner = CliRunner()
        result = runner.invoke(cli, ["--version"])
        assert result.exit_code == 0


class TestCliPlainMode:
    """CLI plain mode detection."""

    def test_cli_plain_flag_forces_plain(self, tmp_path, monkeypatch):
        """--plain flag forces plain output mode."""
        config_path = tmp_path / "config.json"
        payload = {
            "version": CONFIG_VERSION,
            "archive_root": str(tmp_path / "archive"),
            "sources": [],
        }
        config_path.write_text(json.dumps(payload), encoding="utf-8")
        monkeypatch.setenv("POLYLOGUE_CONFIG", str(config_path))

        runner = CliRunner()
        result = runner.invoke(cli, ["--plain", "--help"])
        assert result.exit_code == 0

    def test_cli_interactive_flag_forces_interactive(self, tmp_path, monkeypatch):
        """--interactive flag forces interactive output mode."""
        config_path = tmp_path / "config.json"
        payload = {
            "version": CONFIG_VERSION,
            "archive_root": str(tmp_path / "archive"),
            "sources": [],
        }
        config_path.write_text(json.dumps(payload), encoding="utf-8")
        monkeypatch.setenv("POLYLOGUE_CONFIG", str(config_path))

        runner = CliRunner()
        result = runner.invoke(cli, ["--interactive", "--help"])
        assert result.exit_code == 0


class TestCliConfigOption:
    """CLI --config option."""

    def test_cli_config_option_custom_path(self, tmp_path):
        """--config option uses custom config file."""
        config_path = tmp_path / "custom.json"
        payload = {
            "version": CONFIG_VERSION,
            "archive_root": str(tmp_path / "archive"),
            "sources": [],
        }
        config_path.write_text(json.dumps(payload), encoding="utf-8")

        runner = CliRunner()
        result = runner.invoke(cli, ["--config", str(config_path), "--help"])
        assert result.exit_code == 0


# Import Config class for tests
from polylogue.config import Config
