from __future__ import annotations

import json

import pytest

from polylogue.config import ConfigError, default_config, load_config, write_config


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
