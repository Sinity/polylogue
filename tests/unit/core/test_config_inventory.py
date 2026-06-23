from __future__ import annotations

from pathlib import Path

import pytest


def test_config_inventory_covers_default_config_keys() -> None:
    from polylogue.config import _default_config_values, config_inventory_by_key

    inventory = config_inventory_by_key()

    assert set(_default_config_values()).issubset(inventory)


def test_toml_paths_in_inventory_are_loadable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from polylogue.config import config_inventory_by_key, load_polylogue_config

    cfg = tmp_path / "polylogue.toml"
    cfg.write_text(
        """
        [daemon.browser_capture]
        spool_path = "/tmp/polylogue-capture"

        [embedding]
        voyage_api_key = "sk-test"
        """,
        encoding="utf-8",
    )
    monkeypatch.setenv("POLYLOGUE_CONFIG", str(cfg))
    monkeypatch.setenv("POLYLOGUE_SITE_CONFIG", str(tmp_path / "absent-site.toml"))

    resolved = load_polylogue_config()
    inventory = config_inventory_by_key()

    assert inventory["browser_capture_spool_path"].toml_path == "daemon.browser_capture.spool_path"
    assert resolved.raw["browser_capture_spool_path"] == "/tmp/polylogue-capture"
    assert resolved.layer_of("browser_capture_spool_path") == "user"
    assert inventory["voyage_api_key"].toml_path == "embedding.voyage_api_key"
    assert resolved.raw["voyage_api_key"] == "sk-test"
    assert resolved.layer_of("voyage_api_key") == "user"


def test_env_vars_in_inventory_are_loadable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from polylogue.config import config_inventory_by_key, load_polylogue_config

    monkeypatch.setenv("POLYLOGUE_CONFIG", str(tmp_path / "absent-user.toml"))
    monkeypatch.setenv("POLYLOGUE_SITE_CONFIG", str(tmp_path / "absent-site.toml"))
    monkeypatch.setenv("POLYLOGUE_BROWSER_CAPTURE_PORT", "9876")

    resolved = load_polylogue_config()
    inventory = config_inventory_by_key()

    assert inventory["browser_capture_port"].env_var == "POLYLOGUE_BROWSER_CAPTURE_PORT"
    assert resolved.raw["browser_capture_port"] == 9876
    assert resolved.layer_of("browser_capture_port") == "env"
