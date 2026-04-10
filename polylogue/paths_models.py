"""Path-related data models."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from polylogue.paths_roots import drive_credentials_path, drive_token_path


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
    """Search and vector indexing configuration (from env vars)."""

    fts_enabled: bool = True
    voyage_api_key: str | None = None
    voyage_model: str = "voyage-4"
    voyage_dimension: int | None = None
    auto_embed: bool = False

    @classmethod
    def from_env(cls) -> IndexConfig:
        """Load IndexConfig from environment variables."""
        dimension_str = os.environ.get("POLYLOGUE_VOYAGE_DIMENSION")
        dimension: int | None = None
        if dimension_str:
            try:
                dimension = int(dimension_str)
            except ValueError:
                from polylogue.logging import get_logger

                get_logger(__name__).warning(
                    "Invalid POLYLOGUE_VOYAGE_DIMENSION=%r, using model default", dimension_str
                )

        return cls(
            fts_enabled=True,
            voyage_api_key=os.environ.get("POLYLOGUE_VOYAGE_API_KEY") or os.environ.get("VOYAGE_API_KEY"),
            voyage_model=os.environ.get("POLYLOGUE_VOYAGE_MODEL", "voyage-4"),
            voyage_dimension=dimension,
            auto_embed=os.environ.get("POLYLOGUE_AUTO_EMBED", "").lower() in ("1", "true"),
        )
