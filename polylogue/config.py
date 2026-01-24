"""Minimal configuration for polylogue (zero-config).

This module provides configuration objects with hardcoded XDG paths.
There is no config file, no load_config(), no write_config().
All paths come from polylogue.paths.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .paths import (
    ARCHIVE_ROOT,
    DB_PATH,
    RENDER_ROOT,
    DriveConfig,
    IndexConfig,
    Source,
    get_drive_config,
    get_index_config,
    get_sources,
)


class ConfigError(RuntimeError):
    """Configuration error (kept for backward compatibility with error handling)."""

    pass


@dataclass
class Config:
    """Application configuration with hardcoded paths.

    This class exists for backward compatibility with code that expects a Config object.
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
        return DB_PATH


def get_config() -> Config:
    """Return the hardcoded configuration.

    This replaces load_config(). There is no config file to load.
    All values come from XDG paths in polylogue.paths.
    """
    return Config(
        archive_root=ARCHIVE_ROOT,
        render_root=RENDER_ROOT,
        sources=get_sources(),
        drive_config=get_drive_config(),
        index_config=get_index_config(),
    )


# Backward compatibility aliases
def load_config(path: Path | None = None) -> Config:
    """Load configuration (ignores path argument - zero-config).

    This function exists for backward compatibility. The path argument is ignored.
    All configuration comes from hardcoded XDG paths.
    """
    return get_config()


def default_config() -> Config:
    """Backward compatibility alias for get_config()."""
    return get_config()


def write_config(config: Config, path: Path | None = None) -> None:
    """Backward compatibility stub - no-op since zero-config."""
    pass


def update_config(config: Config, **kwargs: object) -> Config:
    """Backward compatibility stub - returns config unchanged."""
    return config


# Re-export from paths for backward compatibility
__all__ = [
    "Config",
    "ConfigError",
    "DriveConfig",
    "IndexConfig",
    "Source",
    "get_config",
    "load_config",
    "default_config",
    "write_config",
    "update_config",
]
