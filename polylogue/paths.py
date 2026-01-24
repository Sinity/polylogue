"""Shared filesystem paths and helpers for Polylogue.

This module defines all paths used by polylogue. There is no configuration file.
All paths follow XDG Base Directory specification with sensible defaults.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path


def _xdg_path(env_var: str, fallback: Path) -> Path:
    raw = os.environ.get(env_var)
    if raw:
        return Path(raw).expanduser()
    return fallback


# XDG root directories
CONFIG_ROOT = _xdg_path("XDG_CONFIG_HOME", Path.home() / ".config")
DATA_ROOT = _xdg_path("XDG_DATA_HOME", Path.home() / ".local/share")
CACHE_ROOT = _xdg_path("XDG_CACHE_HOME", Path.home() / ".cache")
STATE_ROOT = _xdg_path("XDG_STATE_HOME", Path.home() / ".local/state")

# Polylogue-specific directories
CONFIG_HOME = CONFIG_ROOT / "polylogue"
DATA_HOME = DATA_ROOT / "polylogue"
CACHE_HOME = CACHE_ROOT / "polylogue"
STATE_HOME = STATE_ROOT / "polylogue"

# Core paths (hardcoded, no configuration)
DB_PATH = DATA_HOME / "polylogue.db"
INBOX_ROOT = DATA_HOME / "inbox"
RENDER_ROOT = DATA_HOME / "render"
ARCHIVE_ROOT = DATA_HOME  # DB and renders live here

# Drive OAuth paths
DRIVE_CREDENTIALS_PATH = CONFIG_HOME / "polylogue-credentials.json"
DRIVE_TOKEN_PATH = STATE_HOME / "token.json"

# Standard source paths (auto-discovered)
CLAUDE_CODE_PATH = Path.home() / ".claude" / "projects"
CODEX_PATH = Path.home() / ".codex" / "sessions"
GEMINI_DRIVE_FOLDER = "Google AI Studio"


@dataclass
class Source:
    """A conversation source (local path or Drive folder)."""

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
        if has_path and has_folder:
            raise ValueError(f"Source '{self.name}' cannot have both 'path' and 'folder'")
        if self.folder:
            self.folder = self.folder.strip()

    @property
    def is_drive(self) -> bool:
        return self.folder is not None


@dataclass
class DriveConfig:
    """Google Drive OAuth configuration."""

    credentials_path: Path = DRIVE_CREDENTIALS_PATH
    token_path: Path = DRIVE_TOKEN_PATH
    retry_count: int = 3
    timeout: int = 30


@dataclass
class IndexConfig:
    """Search and vector indexing configuration (from env vars)."""

    fts_enabled: bool = True
    qdrant_url: str | None = None
    qdrant_api_key: str | None = None
    voyage_api_key: str | None = None

    @classmethod
    def from_env(cls) -> "IndexConfig":
        """Load IndexConfig from environment variables."""
        return cls(
            fts_enabled=True,
            qdrant_url=os.environ.get("POLYLOGUE_QDRANT_URL") or os.environ.get("QDRANT_URL"),
            qdrant_api_key=os.environ.get("POLYLOGUE_QDRANT_API_KEY")
            or os.environ.get("QDRANT_API_KEY"),
            voyage_api_key=os.environ.get("POLYLOGUE_VOYAGE_API_KEY")
            or os.environ.get("VOYAGE_API_KEY"),
        )


def get_sources() -> list[Source]:
    """Return the hardcoded list of sources.

    Sources are discovered from standard locations:
    - inbox: ~/.local/share/polylogue/inbox (drop exports here)
    - claude-code: ~/.claude/projects (auto-discovered)
    - codex: ~/.codex/sessions (auto-discovered)
    - gemini: Google Drive "Google AI Studio" folder
    """
    sources = [Source(name="inbox", path=INBOX_ROOT)]

    # Auto-discover Claude Code sessions
    if CLAUDE_CODE_PATH.exists():
        sources.append(Source(name="claude-code", path=CLAUDE_CODE_PATH))

    # Auto-discover Codex sessions
    if CODEX_PATH.exists():
        sources.append(Source(name="codex", path=CODEX_PATH))

    # Gemini via Google Drive (always included, requires auth)
    sources.append(Source(name="gemini", folder=GEMINI_DRIVE_FOLDER))

    return sources


def get_drive_config() -> DriveConfig:
    """Return Drive configuration with default paths."""
    return DriveConfig()


def get_index_config() -> IndexConfig:
    """Return index configuration from environment."""
    return IndexConfig.from_env()


_SAFE_PATH_COMPONENT_RE = re.compile(r"[^A-Za-z0-9._-]")


def safe_path_component(raw: str, *, fallback: str = "item") -> str:
    """Return a filesystem-safe path component derived from raw input."""
    if raw is None:
        raw = ""
    value = str(raw).strip()
    if not value:
        value = fallback
    has_sep = any(sep in value for sep in (os.sep, os.altsep) if sep)
    safe = _SAFE_PATH_COMPONENT_RE.sub("_", value)
    if safe in {"", ".", ".."}:
        safe = fallback
    if has_sep or safe != value:
        digest = sha256(value.encode("utf-8")).hexdigest()[:32]
        prefix = safe.strip("._-") or fallback
        prefix = prefix[:12]
        return f"{prefix}-{digest}"
    return safe


def is_within_root(path: Path, root: Path) -> bool:
    """Return True if path resolves within root."""
    try:
        path.resolve(strict=False).relative_to(root.resolve(strict=False))
    except ValueError:
        return False
    return True


__all__ = [
    # XDG directories
    "CONFIG_HOME",
    "DATA_HOME",
    "CACHE_HOME",
    "STATE_HOME",
    "CONFIG_ROOT",
    "DATA_ROOT",
    "CACHE_ROOT",
    "STATE_ROOT",
    # Core paths
    "DB_PATH",
    "INBOX_ROOT",
    "RENDER_ROOT",
    "ARCHIVE_ROOT",
    "DRIVE_CREDENTIALS_PATH",
    "DRIVE_TOKEN_PATH",
    # Source paths
    "CLAUDE_CODE_PATH",
    "CODEX_PATH",
    "GEMINI_DRIVE_FOLDER",
    # Classes
    "Source",
    "DriveConfig",
    "IndexConfig",
    # Functions
    "get_sources",
    "get_drive_config",
    "get_index_config",
    "safe_path_component",
    "is_within_root",
]
