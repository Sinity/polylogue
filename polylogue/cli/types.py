"""CLI types."""

from __future__ import annotations

from dataclasses import dataclass

from polylogue.ui import UI


@dataclass
class AppEnv:
    """CLI application environment."""

    ui: UI
