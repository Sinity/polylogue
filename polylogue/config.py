"""Minimal runtime configuration derived from filesystem defaults and env overrides."""

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
    """Application configuration derived from paths and source discovery."""

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
    """Return the effective runtime configuration."""
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
