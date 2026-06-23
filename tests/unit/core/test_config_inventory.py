"""Executable inventory tests for Polylogue runtime configuration."""

from __future__ import annotations

from pathlib import Path

import pytest


def test_inventory_covers_loaded_defaults_and_public_polylogue_config_properties(
    monkeypatch: pytest.MonkeyPatch,
    workspace_env: dict[str, Path],
) -> None:
    from polylogue.config import PolylogueConfig, config_inventory_by_key, load_polylogue_config

    monkeypatch.setenv("POLYLOGUE_SITE_CONFIG", "")
    cfg = load_polylogue_config()
    inventory = config_inventory_by_key()

    missing_defaults = sorted(set(cfg.raw) - set(inventory))
    assert not missing_defaults

    public_properties = {
        name
        for name, value in vars(PolylogueConfig).items()
        if isinstance(value, property) and name not in {"raw", "layers"}
    }
    assert public_properties <= set(inventory)


def test_inventory_toml_paths_drive_loader_and_formatter(tmp_path: Path, workspace_env: dict[str, Path]) -> None:
    from polylogue.config import config_inventory_by_key, format_config_toml, load_polylogue_config

    cfg_path = tmp_path / "polylogue.toml"
    cfg_path.write_text(
        """
[daemon.browser_capture]
spool_path = "/tmp/polylogue-browser-spool"
auth_token = "bc-token"

[embedding]
voyage_api_key = "voyage-token"

[health]
blob_integrity_sample_size = 7
""".strip(),
        encoding="utf-8",
    )

    cfg = load_polylogue_config(config_path=cfg_path, site_config_path=tmp_path / "absent.toml")

    inventory = config_inventory_by_key()
    assert inventory["browser_capture_spool_path"].toml_path == "daemon.browser_capture.spool_path"
    assert inventory["voyage_api_key"].toml_path == "embedding.voyage_api_key"
    assert cfg.browser_capture_spool_path == "/tmp/polylogue-browser-spool"
    assert cfg.browser_capture_auth_token == "bc-token"
    assert cfg.voyage_api_key == "voyage-token"
    assert cfg.health_blob_integrity_sample_size == 7

    rendered = format_config_toml(cfg.raw)
    assert "spool_path" in rendered
    assert "blob_integrity_sample_size" in rendered
    assert "bc-token" not in rendered
    assert "voyage-token" not in rendered
    assert "<set>" in rendered


def test_inventory_env_mapping_is_executable(monkeypatch: pytest.MonkeyPatch, workspace_env: dict[str, Path]) -> None:
    from polylogue.config import config_inventory_by_key, load_polylogue_config

    monkeypatch.setenv("POLYLOGUE_SITE_CONFIG", "")
    monkeypatch.setenv("POLYLOGUE_DAEMON_URL", "http://127.0.0.1:9999")
    monkeypatch.setenv("POLYLOGUE_BROWSER_CAPTURE_ALLOW_REMOTE", "true")
    monkeypatch.setenv("POLYLOGUE_BROWSER_CAPTURE_AUTH_TOKEN", "receiver-token")
    monkeypatch.setenv("POLYLOGUE_BROWSER_CAPTURE_SPOOL_PATH", "/tmp/polylogue-spool")
    monkeypatch.setenv("POLYLOGUE_HEALTH_BLOB_INTEGRITY_SAMPLE_SIZE", "3")
    monkeypatch.setenv("NO_COLOR", "1")

    cfg = load_polylogue_config()
    inventory = config_inventory_by_key()

    assert inventory["daemon_url"].env_var == "POLYLOGUE_DAEMON_URL"
    assert inventory["no_color"].env_var == "NO_COLOR"
    assert cfg.daemon_url == "http://127.0.0.1:9999"
    assert cfg.layer_of("daemon_url") == "env"
    assert cfg.browser_capture_allow_remote is True
    assert cfg.browser_capture_auth_token == "receiver-token"
    assert cfg.browser_capture_spool_path == "/tmp/polylogue-spool"
    assert cfg.health_blob_integrity_sample_size == 3
    assert cfg.no_color is True


def test_effective_config_payload_redacts_secret_presence_and_exposes_source_layer(
    monkeypatch: pytest.MonkeyPatch,
    workspace_env: dict[str, Path],
) -> None:
    from polylogue.config import effective_config_payload, load_polylogue_config

    monkeypatch.setenv("POLYLOGUE_SITE_CONFIG", "")
    monkeypatch.setenv("POLYLOGUE_API_AUTH_TOKEN", "do-not-leak")
    monkeypatch.setenv("POLYLOGUE_API_PORT", "9991")

    payload = effective_config_payload(load_polylogue_config())
    values = payload["values"]
    assert isinstance(values, dict)

    token = values["api_auth_token"]
    assert isinstance(token, dict)
    assert token["value"] == "<set>"
    assert token["secret_present"] is True
    assert token["source_layer"] == "env"

    api_port = values["api_port"]
    assert isinstance(api_port, dict)
    assert api_port["value"] == 9991
    assert api_port["source_layer"] == "env"
    assert api_port["effective_path"] == "polylogue config --format json values.api_port"

    assert "do-not-leak" not in str(payload)
