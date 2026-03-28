"""Minimal configuration for polylogue (zero-config).

This module provides configuration objects with hardcoded XDG paths.
There is no config file, no load_config(), no write_config().
All paths come from polylogue.paths.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .errors import PolylogueError
from .paths import (
    DriveConfig,
    IndexConfig,
    Source,
    archive_root,
    db_path,
    get_drive_config,
    get_index_config,
    get_sources,
    render_root,
)


class ConfigError(PolylogueError):
    """Configuration error."""


@dataclass
class Config:
    """Application configuration with hardcoded paths.

    All values are derived from polylogue.paths - there is no config file.
    """

    archive_root: Path
    render_root: Path
    sources: list[Source]
    drive_config: DriveConfig | None = None
    index_config: IndexConfig | None = None

    @property
    def db_path(self) -> Path:
        """Database path (always XDG default)."""
        return db_path()


def get_config() -> Config:
    """Return the hardcoded configuration.

    This replaces load_config(). There is no config file to load.
    All values come from XDG paths in polylogue.paths.
    """
    return Config(
        archive_root=archive_root(),
        render_root=render_root(),
        sources=get_sources(),
        drive_config=get_drive_config(),
        index_config=get_index_config(),
    )


__all__ = [
    "Config",
    "ConfigError",
    "DriveConfig",
    "IndexConfig",
    "Source",
    "get_config",
]
