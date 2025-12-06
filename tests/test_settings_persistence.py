from pathlib import Path

from polylogue.settings import Settings, persist_settings_to, _load_settings_file, reset_settings


def test_settings_persist_and_load(tmp_path: Path):
    path = tmp_path / "settings.json"
    s = Settings(html_previews=True, html_theme="dark", collapse_threshold=42, preferred_providers=["codex"])
    persist_settings_to(path, s)
    loaded = _load_settings_file(path)  # intentionally using private helper for direct load
    assert loaded is not None
    assert loaded.html_previews is True
    assert loaded.html_theme == "dark"
    assert loaded.collapse_threshold == 42
    assert loaded.preferred_providers == ["codex"]


def test_reset_settings_sets_defaults():
    s = Settings(html_previews=True, html_theme="dark", collapse_threshold=10, preferred_providers=["x"])
    reset_settings(s)
    assert s.html_previews is False
    assert s.html_theme == "light"
    assert s.collapse_threshold == 25
    assert s.preferred_providers == []
