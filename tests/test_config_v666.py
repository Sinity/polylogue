from __future__ import annotations

import json

import pytest

from polylogue.config import ConfigError, default_config, load_config, resolve_profile, write_config


def test_config_roundtrip(workspace_env):
    config = default_config()
    write_config(config)

    loaded = load_config()
    assert loaded.archive_root == workspace_env["archive_root"]
    assert loaded.version == config.version
    profile_name, profile = resolve_profile(loaded, None)
    assert profile_name == "default"
    assert profile.attachments == "download"


def test_config_rejects_unknown_keys(workspace_env):
    payload = {
        "version": 1,
        "archive_root": str(workspace_env["archive_root"]),
        "sources": [],
        "profiles": {"default": {"attachments": "download", "html": "auto", "index": True, "sanitize_html": False}},
        "extra": "nope",
    }
    workspace_env["config_path"].parent.mkdir(parents=True, exist_ok=True)
    workspace_env["config_path"].write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ConfigError):
        load_config()
