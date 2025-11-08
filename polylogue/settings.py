from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class Settings:
    html_previews: bool = False
    html_theme: str = "light"


SETTINGS = Settings()


def reset_settings(settings: Optional[Settings] = None) -> Settings:
    target = settings or SETTINGS
    target.html_previews = False
    target.html_theme = "light"
    return target

