"""Shared filesystem paths and helpers for Polylogue.

This module defines all paths used by polylogue. There is no configuration file.
All paths follow XDG Base Directory specification with sensible defaults.

Path resolution is lazy — functions read environment variables at call time,
not import time. This eliminates the need for importlib.reload() in tests:
just set the env var and call the function again.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from hashlib import sha256
from pathlib import Path


def _xdg_path(env_var: str, fallback: Path) -> Path:
    raw = os.environ.get(env_var, "").strip()
    if raw:
        return Path(raw).expanduser()
    return fallback


# ---------------------------------------------------------------------------
# Lazy path accessors — read env vars at call time, cached per call site
# ---------------------------------------------------------------------------


def config_root() -> Path:
    """XDG_CONFIG_HOME (default: ~/.config)."""
    return _xdg_path("XDG_CONFIG_HOME", Path.home() / ".config")


def data_root() -> Path:
    """XDG_DATA_HOME (default: ~/.local/share)."""
    return _xdg_path("XDG_DATA_HOME", Path.home() / ".local/share")


def cache_root() -> Path:
    """XDG_CACHE_HOME (default: ~/.cache)."""
    return _xdg_path("XDG_CACHE_HOME", Path.home() / ".cache")


def state_root() -> Path:
    """XDG_STATE_HOME (default: ~/.local/state)."""
    return _xdg_path("XDG_STATE_HOME", Path.home() / ".local/state")


def config_home() -> Path:
    """Polylogue config directory."""
    return config_root() / "polylogue"


def data_home() -> Path:
    """Polylogue data directory."""
    return data_root() / "polylogue"


def cache_home() -> Path:
    """Polylogue cache directory."""
    return cache_root() / "polylogue"


def state_home() -> Path:
    """Polylogue state directory."""
    return state_root() / "polylogue"


def db_path() -> Path:
    """Default database path."""
    return data_home() / "polylogue.db"


def inbox_root() -> Path:
    """Default inbox directory."""
    return data_home() / "inbox"


def archive_root() -> Path:
    """Archive root (overridable via POLYLOGUE_ARCHIVE_ROOT)."""
    return _xdg_path("POLYLOGUE_ARCHIVE_ROOT", data_home())


def render_root() -> Path:
    """Render output root (overridable via POLYLOGUE_RENDER_ROOT)."""
    return _xdg_path("POLYLOGUE_RENDER_ROOT", data_home() / "render")


def drive_credentials_path() -> Path:
    """Drive OAuth credentials path."""
    return config_home() / "polylogue-credentials.json"


def drive_token_path() -> Path:
    """Drive OAuth token path."""
    return state_home() / "token.json"


def claude_code_path() -> Path:
    """Claude Code sessions directory."""
    return Path.home() / ".claude" / "projects"


def codex_path() -> Path:
    """Codex sessions directory."""
    return Path.home() / ".codex" / "sessions"


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
    voyage_dimension: int | None = None  # None = use model default (1024 for voyage-4)
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
                from polylogue.lib.log import get_logger

                get_logger(__name__).warning(
                    "Invalid POLYLOGUE_VOYAGE_DIMENSION=%r, using model default", dimension_str
                )

        return cls(
            fts_enabled=True,
            voyage_api_key=os.environ.get("POLYLOGUE_VOYAGE_API_KEY")
            or os.environ.get("VOYAGE_API_KEY"),
            voyage_model=os.environ.get("POLYLOGUE_VOYAGE_MODEL", "voyage-4"),
            voyage_dimension=dimension,
            auto_embed=os.environ.get("POLYLOGUE_AUTO_EMBED", "").lower() in ("1", "true"),
        )


def get_sources() -> list[Source]:
    """Return the hardcoded list of sources.

    Sources are discovered from standard locations:
    - inbox: ~/.local/share/polylogue/inbox (drop exports here)
    - claude-code: ~/.claude/projects (auto-discovered)
    - codex: ~/.codex/sessions (auto-discovered)
    - gemini: Google Drive "Google AI Studio" folder
    """
    sources = [Source(name="inbox", path=inbox_root())]

    # Auto-discover Claude Code sessions
    if claude_code_path().exists():
        sources.append(Source(name="claude-code", path=claude_code_path()))

    # Auto-discover Codex sessions
    if codex_path().exists():
        sources.append(Source(name="codex", path=codex_path()))

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
    # XDG directory accessors
    "config_home",
    "data_home",
    "cache_home",
    "state_home",
    "config_root",
    "data_root",
    "cache_root",
    "state_root",
    # Core path accessors
    "db_path",
    "inbox_root",
    "render_root",
    "archive_root",
    "drive_credentials_path",
    "drive_token_path",
    # Source path accessors
    "claude_code_path",
    "codex_path",
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
