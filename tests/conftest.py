from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _clear_polylogue_env(monkeypatch):
    for key in (
        "POLYLOGUE_CONFIG",
        "POLYLOGUE_ARCHIVE_ROOT",
        "POLYLOGUE_RENDER_ROOT",
        "POLYLOGUE_TEMPLATE_PATH",
        "POLYLOGUE_DECLARATIVE",
    ):
        monkeypatch.delenv(key, raising=False)


@pytest.fixture
def workspace_env(tmp_path, monkeypatch):
    config_dir = tmp_path / "config"
    state_dir = tmp_path / "state"
    archive_root = tmp_path / "archive"

    monkeypatch.setenv("POLYLOGUE_CONFIG", str(config_dir / "config.json"))
    monkeypatch.setenv("XDG_STATE_HOME", str(state_dir))
    monkeypatch.setenv("POLYLOGUE_ARCHIVE_ROOT", str(archive_root))
    monkeypatch.setenv("POLYLOGUE_RENDER_ROOT", str(archive_root / "render"))
    monkeypatch.delenv("QDRANT_URL", raising=False)
    monkeypatch.delenv("QDRANT_API_KEY", raising=False)

    return {
        "config_path": config_dir / "config.json",
        "archive_root": archive_root,
        "state_root": state_dir,
    }
