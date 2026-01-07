"""CLI types."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from polylogue.ui import UI


@dataclass
class AppEnv:
    ui: UI
    config_path: Path | None = None
