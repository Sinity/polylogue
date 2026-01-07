from __future__ import annotations

import json
from pathlib import Path

import pytest

from polylogue.config import (
    ConfigError,
    Config,
    Source,
    default_config,
    load_config,
    write_config,
    update_source,
)


def test_config_roundtrip(workspace_env):
    config = default_config()
    write_config(config)

    loaded = load_config()
    assert loaded.archive_root == workspace_env["archive_root"]
    assert loaded.render_root == workspace_env["archive_root"] / "render"
    assert loaded.version == config.version
    assert loaded.sources
    assert loaded.sources[0].path is not None


def test_config_rejects_unknown_keys(workspace_env):
    payload = {
        "version": 2,
        "archive_root": str(workspace_env["archive_root"]),
        "sources": [],
        "extra": "nope",
    }
    workspace_env["config_path"].parent.mkdir(parents=True, exist_ok=True)
    workspace_env["config_path"].write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ConfigError):
        load_config()


def test_config_rejects_duplicate_sources(workspace_env):
    payload = {
        "version": 2,
        "archive_root": str(workspace_env["archive_root"]),
        "sources": [
            {"name": "inbox", "path": str(workspace_env["archive_root"])},
            {"name": "inbox", "path": str(workspace_env["archive_root"])},
        ],
    }
    workspace_env["config_path"].parent.mkdir(parents=True, exist_ok=True)
    workspace_env["config_path"].write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ConfigError):
        load_config()


def test_load_config_malformed_json_includes_path(workspace_env):
    """Malformed config.json error message includes file path."""
    config_path = workspace_env["config_path"]
    config_path.parent.mkdir(parents=True, exist_ok=True)
    # Write invalid JSON
    config_path.write_text("{ this is not valid json", encoding="utf-8")

    with pytest.raises(ConfigError) as exc_info:
        load_config()

    error_msg = str(exc_info.value)
    # Error should mention the file path and indicate invalid JSON
    assert str(config_path) in error_msg or "config" in error_msg.lower()
    assert "JSON" in error_msg or "json" in error_msg.lower()


class TestSourceValidation:
    """Tests for Source dataclass validation."""

    def test_source_with_path_valid(self, tmp_path: Path):
        """Source with valid path should work."""
        source = Source(name="test", path=tmp_path)
        assert source.name == "test"
        assert source.path == tmp_path
        assert source.folder is None

    def test_source_with_folder_valid(self):
        """Source with valid folder ID should work."""
        source = Source(name="drive", folder="abc123")
        assert source.name == "drive"
        assert source.folder == "abc123"
        assert source.path is None

    def test_source_is_drive_property_with_folder(self):
        """Source.is_drive should return True when folder is set."""
        source = Source(name="gdrive", folder="folder123")
        assert source.is_drive is True

    def test_source_is_drive_property_without_folder(self, tmp_path: Path):
        """Source.is_drive should return False when only path is set."""
        source = Source(name="local", path=tmp_path)
        assert source.is_drive is False

    def test_source_as_dict_with_path(self, tmp_path: Path):
        """Source.as_dict() should include path as string."""
        source = Source(name="mydata", path=tmp_path)
        result = source.as_dict()
        assert result["name"] == "mydata"
        assert result["path"] == str(tmp_path)
        assert "folder" not in result

    def test_source_as_dict_with_folder(self):
        """Source.as_dict() should include folder."""
        source = Source(name="gdrive", folder="xyz789")
        result = source.as_dict()
        assert result["name"] == "gdrive"
        assert result["folder"] == "xyz789"
        assert "path" not in result

    def test_source_rejects_empty_name_during_parse(self, workspace_env):
        """_parse_source rejects empty name."""
        payload = {
            "version": 2,
            "archive_root": str(workspace_env["archive_root"]),
            "sources": [
                {"name": "", "path": str(workspace_env["archive_root"])},
            ],
        }
        workspace_env["config_path"].parent.mkdir(parents=True, exist_ok=True)
        workspace_env["config_path"].write_text(json.dumps(payload), encoding="utf-8")

        with pytest.raises(ConfigError, match="must be a non-empty string"):
            load_config()

    def test_source_rejects_whitespace_only_name_during_parse(self, workspace_env):
        """_parse_source rejects name with only whitespace."""
        payload = {
            "version": 2,
            "archive_root": str(workspace_env["archive_root"]),
            "sources": [
                {"name": "   ", "path": str(workspace_env["archive_root"])},
            ],
        }
        workspace_env["config_path"].parent.mkdir(parents=True, exist_ok=True)
        workspace_env["config_path"].write_text(json.dumps(payload), encoding="utf-8")

        with pytest.raises(ConfigError, match="must be a non-empty string"):
            load_config()

    def test_source_rejects_missing_path_and_folder_during_parse(self, workspace_env):
        """_parse_source rejects source with neither path nor folder."""
        payload = {
            "version": 2,
            "archive_root": str(workspace_env["archive_root"]),
            "sources": [
                {"name": "invalid"},  # No path or folder
            ],
        }
        workspace_env["config_path"].parent.mkdir(parents=True, exist_ok=True)
        workspace_env["config_path"].write_text(json.dumps(payload), encoding="utf-8")

        with pytest.raises(ConfigError, match="requires non-empty"):
            load_config()

    def test_source_rejects_both_path_and_folder_during_parse(self, workspace_env):
        """_parse_source rejects source with both path and folder."""
        payload = {
            "version": 2,
            "archive_root": str(workspace_env["archive_root"]),
            "sources": [
                {
                    "name": "ambiguous",
                    "path": str(workspace_env["archive_root"]),
                    "folder": "xyz789",
                },
            ],
        }
        workspace_env["config_path"].parent.mkdir(parents=True, exist_ok=True)
        workspace_env["config_path"].write_text(json.dumps(payload), encoding="utf-8")

        with pytest.raises(ConfigError, match="must not set both"):
            load_config()

    def test_source_rejects_empty_path_during_parse(self, workspace_env):
        """_parse_source rejects empty path string."""
        payload = {
            "version": 2,
            "archive_root": str(workspace_env["archive_root"]),
            "sources": [
                {"name": "mydata", "path": ""},
            ],
        }
        workspace_env["config_path"].parent.mkdir(parents=True, exist_ok=True)
        workspace_env["config_path"].write_text(json.dumps(payload), encoding="utf-8")

        with pytest.raises(ConfigError, match="requires non-empty"):
            load_config()

    def test_source_rejects_empty_folder_during_parse(self, workspace_env):
        """_parse_source rejects empty folder string."""
        payload = {
            "version": 2,
            "archive_root": str(workspace_env["archive_root"]),
            "sources": [
                {"name": "gdrive", "folder": ""},
            ],
        }
        workspace_env["config_path"].parent.mkdir(parents=True, exist_ok=True)
        workspace_env["config_path"].write_text(json.dumps(payload), encoding="utf-8")

        with pytest.raises(ConfigError, match="requires non-empty"):
            load_config()

    def test_source_name_is_stripped_during_parse(self, workspace_env):
        """_parse_source strips whitespace from name."""
        payload = {
            "version": 2,
            "archive_root": str(workspace_env["archive_root"]),
            "sources": [
                {"name": "  test  ", "path": str(workspace_env["archive_root"])},
            ],
        }
        workspace_env["config_path"].parent.mkdir(parents=True, exist_ok=True)
        workspace_env["config_path"].write_text(json.dumps(payload), encoding="utf-8")

        config = load_config()
        assert config.sources[0].name == "test"

    def test_source_folder_is_stripped_during_parse(self, workspace_env):
        """_parse_source strips whitespace from folder."""
        payload = {
            "version": 2,
            "archive_root": str(workspace_env["archive_root"]),
            "sources": [
                {"name": "gdrive", "folder": "  abc123  "},
            ],
        }
        workspace_env["config_path"].parent.mkdir(parents=True, exist_ok=True)
        workspace_env["config_path"].write_text(json.dumps(payload), encoding="utf-8")

        config = load_config()
        assert config.sources[0].folder == "abc123"

    def test_source_rejects_neither_path_nor_folder(self):
        """Source with neither path nor folder MUST be rejected.

        This test SHOULD FAIL until Source validation is added.
        """
        with pytest.raises((ValueError, ConfigError), match="path|folder"):
            Source(name="invalid")  # No path, no folder

    def test_source_rejects_both_path_and_folder(self, tmp_path: Path):
        """Source with BOTH path AND folder MUST be rejected (ambiguous).

        This test SHOULD FAIL until Source validation is added.
        """
        with pytest.raises((ValueError, ConfigError), match="both|ambiguous"):
            Source(name="ambiguous", path=tmp_path, folder="abc123")

    def test_source_name_required(self, tmp_path: Path):
        """Source MUST have non-empty name.

        This test SHOULD FAIL until name validation is added.
        """
        with pytest.raises((ValueError, ConfigError), match="name"):
            Source(name="", path=tmp_path)


class TestConfigValidation:
    """Tests for Config validation."""

    def test_config_requires_archive_root(self, workspace_env):
        """Config must have archive_root."""
        # archive_root is a required field in the dataclass
        payload = {
            "version": 2,
            "sources": [],
            # Missing archive_root
        }
        workspace_env["config_path"].parent.mkdir(parents=True, exist_ok=True)
        workspace_env["config_path"].write_text(json.dumps(payload), encoding="utf-8")

        with pytest.raises(ConfigError, match="archive_root"):
            load_config()

    def test_config_requires_archive_root_nonempty(self, workspace_env):
        """Config archive_root must be non-empty string."""
        payload = {
            "version": 2,
            "archive_root": "",
            "sources": [],
        }
        workspace_env["config_path"].parent.mkdir(parents=True, exist_ok=True)
        workspace_env["config_path"].write_text(json.dumps(payload), encoding="utf-8")

        with pytest.raises(ConfigError, match="archive_root"):
            load_config()

    def test_config_sources_must_be_list(self, workspace_env):
        """Config sources must be a list."""
        payload = {
            "version": 2,
            "archive_root": str(workspace_env["archive_root"]),
            "sources": "not a list",
        }
        workspace_env["config_path"].parent.mkdir(parents=True, exist_ok=True)
        workspace_env["config_path"].write_text(json.dumps(payload), encoding="utf-8")

        with pytest.raises(ConfigError, match="sources.*list"):
            load_config()

    def test_config_empty_sources_valid(self, workspace_env):
        """Config with no sources should be valid."""
        payload = {
            "version": 2,
            "archive_root": str(workspace_env["archive_root"]),
            "sources": [],
        }
        workspace_env["config_path"].parent.mkdir(parents=True, exist_ok=True)
        workspace_env["config_path"].write_text(json.dumps(payload), encoding="utf-8")

        config = load_config()
        assert config.sources == []

    def test_config_render_root_optional(self, workspace_env):
        """Config render_root is optional and defaults to archive_root/render."""
        payload = {
            "version": 2,
            "archive_root": str(workspace_env["archive_root"]),
            "sources": [],
            # No render_root specified
        }
        workspace_env["config_path"].parent.mkdir(parents=True, exist_ok=True)
        workspace_env["config_path"].write_text(json.dumps(payload), encoding="utf-8")

        config = load_config()
        assert config.render_root == workspace_env["archive_root"] / "render"

    def test_config_render_root_explicit(self, workspace_env, tmp_path, monkeypatch):
        """Config render_root can be explicitly set."""
        # Unset POLYLOGUE_RENDER_ROOT to test config file value is respected
        monkeypatch.delenv("POLYLOGUE_RENDER_ROOT", raising=False)

        render_root = tmp_path / "custom_render"
        payload = {
            "version": 2,
            "archive_root": str(workspace_env["archive_root"]),
            "render_root": str(render_root),
            "sources": [],
        }
        workspace_env["config_path"].parent.mkdir(parents=True, exist_ok=True)
        workspace_env["config_path"].write_text(json.dumps(payload), encoding="utf-8")

        config = load_config()
        assert config.render_root == render_root

    def test_config_render_root_empty_string_rejected(self, workspace_env):
        """Config render_root must be non-empty if provided."""
        payload = {
            "version": 2,
            "archive_root": str(workspace_env["archive_root"]),
            "render_root": "",
            "sources": [],
        }
        workspace_env["config_path"].parent.mkdir(parents=True, exist_ok=True)
        workspace_env["config_path"].write_text(json.dumps(payload), encoding="utf-8")

        with pytest.raises(ConfigError, match="render_root"):
            load_config()

    def test_config_template_path_optional(self, workspace_env):
        """Config template_path is optional."""
        payload = {
            "version": 2,
            "archive_root": str(workspace_env["archive_root"]),
            "sources": [],
            # No template_path specified
        }
        workspace_env["config_path"].parent.mkdir(parents=True, exist_ok=True)
        workspace_env["config_path"].write_text(json.dumps(payload), encoding="utf-8")

        config = load_config()
        assert config.template_path is None

    def test_config_template_path_explicit(self, workspace_env, tmp_path):
        """Config template_path can be explicitly set."""
        template_file = tmp_path / "template.html"
        payload = {
            "version": 2,
            "archive_root": str(workspace_env["archive_root"]),
            "template_path": str(template_file),
            "sources": [],
        }
        workspace_env["config_path"].parent.mkdir(parents=True, exist_ok=True)
        workspace_env["config_path"].write_text(json.dumps(payload), encoding="utf-8")

        config = load_config()
        assert config.template_path == template_file

    def test_config_template_path_empty_string_rejected(self, workspace_env):
        """Config template_path must be non-empty if provided."""
        payload = {
            "version": 2,
            "archive_root": str(workspace_env["archive_root"]),
            "template_path": "",
            "sources": [],
        }
        workspace_env["config_path"].parent.mkdir(parents=True, exist_ok=True)
        workspace_env["config_path"].write_text(json.dumps(payload), encoding="utf-8")

        with pytest.raises(ConfigError, match="template_path"):
            load_config()

    def test_config_version_required(self, workspace_env):
        """Config must have correct version."""
        payload = {
            # Missing version
            "archive_root": str(workspace_env["archive_root"]),
            "sources": [],
        }
        workspace_env["config_path"].parent.mkdir(parents=True, exist_ok=True)
        workspace_env["config_path"].write_text(json.dumps(payload), encoding="utf-8")

        with pytest.raises(ConfigError, match="Unsupported config version"):
            load_config()

    def test_config_version_must_match(self, workspace_env):
        """Config version must match expected version."""
        payload = {
            "version": 999,
            "archive_root": str(workspace_env["archive_root"]),
            "sources": [],
        }
        workspace_env["config_path"].parent.mkdir(parents=True, exist_ok=True)
        workspace_env["config_path"].write_text(json.dumps(payload), encoding="utf-8")

        with pytest.raises(ConfigError, match="Unsupported config version"):
            load_config()

    def test_config_as_dict_roundtrip(self, workspace_env):
        """Config.as_dict() produces valid serializable output."""
        config = default_config()
        result = config.as_dict()

        assert isinstance(result, dict)
        assert "version" in result
        assert "archive_root" in result
        assert "render_root" in result
        assert "sources" in result
        assert isinstance(result["sources"], list)

    def test_config_as_dict_includes_template_path_when_set(self, tmp_path):
        """Config.as_dict() includes template_path when set."""
        template_file = tmp_path / "template.html"
        config = Config(
            version=2,
            archive_root=tmp_path / "archive",
            render_root=tmp_path / "render",
            sources=[],
            path=tmp_path / "config.json",
            template_path=template_file,
        )
        result = config.as_dict()
        assert "template_path" in result
        assert result["template_path"] == str(template_file)

    def test_config_as_dict_excludes_template_path_when_none(self, tmp_path):
        """Config.as_dict() excludes template_path when None."""
        config = Config(
            version=2,
            archive_root=tmp_path / "archive",
            render_root=tmp_path / "render",
            sources=[],
            path=tmp_path / "config.json",
            template_path=None,
        )
        result = config.as_dict()
        assert "template_path" not in result

    def test_rejects_duplicate_source_names(self, tmp_path: Path):
        """Config with duplicate source names MUST be rejected.

        This test SHOULD FAIL until duplicate detection is added.
        """
        sources = [
            Source(name="mydata", path=tmp_path / "a"),
            Source(name="mydata", path=tmp_path / "b"),
        ]

        with pytest.raises((ValueError, ConfigError), match="(?i)duplicate"):
            Config(
                version=2,
                archive_root=tmp_path / "archive",
                render_root=tmp_path / "render",
                sources=sources,
                path=tmp_path / "config.json",
            )


class TestUpdateSource:
    """Tests for update_source function."""

    def test_update_source_path_field(self, workspace_env, tmp_path):
        """update_source can update path field."""
        new_path = tmp_path / "newdata"
        config = default_config()
        config.sources = [Source(name="inbox", path=tmp_path / "olddata")]

        updated = update_source(config, "inbox", "path", str(new_path))

        assert updated.sources[0].path == new_path
        assert updated.sources[0].folder is None

    def test_update_source_folder_field(self, workspace_env, tmp_path):
        """update_source can update folder field."""
        config = default_config()
        config.sources = [Source(name="gdrive", folder="old123")]

        updated = update_source(config, "gdrive", "folder", "new456")

        assert updated.sources[0].folder == "new456"
        assert updated.sources[0].path is None

    def test_update_source_path_to_folder_switch(self, workspace_env, tmp_path):
        """update_source can switch from path to folder."""
        config = default_config()
        config.sources = [Source(name="data", path=tmp_path / "local")]

        updated = update_source(config, "data", "folder", "gdrive789")

        assert updated.sources[0].folder == "gdrive789"
        assert updated.sources[0].path is None

    def test_update_source_folder_to_path_switch(self, workspace_env, tmp_path):
        """update_source can switch from folder to path."""
        config = default_config()
        config.sources = [Source(name="data", folder="old123")]

        updated = update_source(config, "data", "path", str(tmp_path / "local"))

        assert updated.sources[0].path == tmp_path / "local"
        assert updated.sources[0].folder is None

    def test_update_source_not_found(self, workspace_env):
        """update_source raises if source name not found."""
        config = default_config()

        with pytest.raises(ConfigError, match="not found"):
            update_source(config, "nonexistent", "path", "/tmp/data")

    def test_update_source_unknown_field(self, workspace_env):
        """update_source raises if field is unknown."""
        config = default_config()

        with pytest.raises(ConfigError, match="Unknown source field"):
            update_source(config, "inbox", "unknown", "value")
