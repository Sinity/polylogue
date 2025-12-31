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
