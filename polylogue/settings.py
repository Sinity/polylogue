from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class Settings:
    html_previews: bool = False
    html_theme: str = "light"


SETTINGS = Settings()


def reset_settings() -> None:
    SETTINGS.html_previews = False
    SETTINGS.html_theme = "light"


