from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .config import CONFIG
from .util import CONFIG_HOME


@dataclass
class Settings:
    html_previews: bool = False
    html_theme: str = "light"


SETTINGS = Settings()
SETTINGS_PATH = CONFIG_HOME / "settings.json"


def reset_settings(settings: Optional[Settings] = None) -> Settings:
    target = settings or SETTINGS
    target.html_previews = False
    target.html_theme = "light"
    return target


def _load_settings_file(path: Path) -> Optional[Settings]:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    html = bool(data.get("html_previews", data.get("html", False)))
    theme = data.get("html_theme") or data.get("theme") or "light"
    if theme not in {"light", "dark"}:
        theme = "light"
    return Settings(html_previews=html, html_theme=theme)


def load_persisted_settings() -> Optional[Settings]:
    return _load_settings_file(SETTINGS_PATH)


def persist_settings(settings: Settings) -> None:
    SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {"html_previews": settings.html_previews, "html_theme": settings.html_theme}
    SETTINGS_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def clear_persisted_settings() -> None:
    try:
        SETTINGS_PATH.unlink()
    except FileNotFoundError:
        pass


def ensure_settings_defaults(settings: Optional[Settings] = None) -> Settings:
    target = settings or SETTINGS
    target.html_previews = CONFIG.defaults.html_previews
    target.html_theme = CONFIG.defaults.html_theme
    stored = load_persisted_settings()
    if stored:
        target.html_previews = stored.html_previews
        target.html_theme = stored.html_theme
    return target
