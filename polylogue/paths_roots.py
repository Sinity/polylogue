"""Filesystem root and path accessors."""

from __future__ import annotations

import os
from pathlib import Path


def _xdg_path(env_var: str, fallback: Path) -> Path:
    raw = os.environ.get(env_var, "").strip()
    if raw:
        return Path(raw).expanduser()
    return fallback


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


def blob_store_root() -> Path:
    """Content-addressed blob store root directory."""
    return data_home() / "blob"


def drive_cache_path() -> Path:
    """Local cache directory for Drive-sourced files."""
    return data_home() / "drive-cache"
