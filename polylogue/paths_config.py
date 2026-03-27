"""Path-backed configuration accessors."""

from __future__ import annotations

from polylogue.paths_models import DriveConfig, IndexConfig, Source
from polylogue.paths_roots import (
    GEMINI_DRIVE_FOLDER,
    claude_code_path,
    codex_path,
    drive_cache_path,
    inbox_root,
)


def get_sources() -> list[Source]:
    """Return the hardcoded list of sources."""
    sources = [Source(name="inbox", path=inbox_root())]

    if claude_code_path().exists():
        sources.append(Source(name="claude-code", path=claude_code_path()))

    if codex_path().exists():
        sources.append(Source(name="codex", path=codex_path()))

    sources.append(
        Source(
            name="gemini",
            folder=GEMINI_DRIVE_FOLDER,
            path=drive_cache_path() / "gemini",
        )
    )
    return sources


def get_drive_config() -> DriveConfig:
    """Return Drive configuration with default paths."""
    return DriveConfig()


def get_index_config() -> IndexConfig:
    """Return index configuration from environment."""
    return IndexConfig.from_env()
