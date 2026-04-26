"""Runtime configuration derived from filesystem defaults and env overrides."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from .errors import PolylogueError
from .paths import (
    GEMINI_DRIVE_FOLDER,
    archive_root,
    claude_code_path,
    codex_path,
    drive_cache_path,
    drive_credentials_path,
    drive_token_path,
    inbox_root,
    render_root,
)
from .paths import (
    db_path as default_db_path,
)


class ConfigError(PolylogueError):
    """Configuration error."""


@dataclass
class Source:
    """A conversation source (local path, Drive folder, or both)."""

    name: str
    path: Path | None = None
    folder: str | None = None

    def __post_init__(self) -> None:
        if not self.name or not self.name.strip():
            raise ValueError("Source name cannot be empty")
        self.name = self.name.strip()
        has_path = self.path is not None
        has_folder = self.folder is not None and self.folder.strip()
        if not has_path and not has_folder:
            raise ValueError(f"Source '{self.name}' must have either 'path' or 'folder'")
        if self.folder:
            self.folder = self.folder.strip()

    @property
    def is_drive(self) -> bool:
        return self.folder is not None


@dataclass
class DriveConfig:
    """Google Drive OAuth configuration."""

    credentials_path: Path = field(default_factory=drive_credentials_path)
    token_path: Path = field(default_factory=drive_token_path)
    retry_count: int = 3
    timeout: int = 30


@dataclass
class IndexConfig:
    """Search indexing configuration."""

    voyage_api_key: str | None = None

    @classmethod
    def from_env(cls) -> IndexConfig:
        """Load IndexConfig from environment variables."""
        return cls(
            voyage_api_key=os.environ.get("VOYAGE_API_KEY"),
        )


@dataclass
class Config:
    """Application configuration derived from paths and source discovery."""

    archive_root: Path
    render_root: Path
    sources: list[Source]
    db_path: Path = field(default_factory=default_db_path)
    drive_config: DriveConfig | None = None
    index_config: IndexConfig | None = None


def get_sources() -> list[Source]:
    """Return the configured conversation sources."""
    sources = [Source(name="inbox", path=inbox_root())]

    if claude_code_path().exists():
        sources.append(Source(name="claude-code", path=claude_code_path()))

    if codex_path().exists():
        sources.append(Source(name="codex", path=codex_path()))

    gemini_cache = drive_cache_path() / "gemini"
    if gemini_cache.exists() or drive_credentials_path().exists() or drive_token_path().exists():
        sources.append(
            Source(
                name="gemini",
                folder=GEMINI_DRIVE_FOLDER,
                path=gemini_cache,
            )
        )
    return sources


def get_drive_config() -> DriveConfig:
    """Return Drive configuration with default paths."""
    return DriveConfig()


def get_index_config() -> IndexConfig:
    """Return index configuration from environment."""
    return IndexConfig.from_env()


def get_config() -> Config:
    """Return the effective runtime configuration."""
    return Config(
        archive_root=archive_root(),
        render_root=render_root(),
        sources=get_sources(),
        db_path=default_db_path(),
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
    "get_drive_config",
    "get_index_config",
    "get_sources",
]
