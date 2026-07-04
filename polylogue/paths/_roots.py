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
    """Default archive index database path."""
    return index_db_path()


def source_db_path() -> Path:
    """source-log database path."""
    return archive_root() / "source.db"


def index_db_path() -> Path:
    """projection/index database path."""
    return archive_root() / "index.db"


def embeddings_db_path() -> Path:
    """semantic-index database path."""
    return archive_root() / "embeddings.db"


def resolve_active_index_db_path(*, db_anchor: Path, index_db: Path) -> Path:
    """Resolve the active query/index database path."""
    if db_anchor.name == "index.db":
        return db_anchor
    return index_db


def archive_file_set_index_available_for_paths(*, archive_root_path: Path, db_anchor: Path) -> bool:
    """Return whether routing is active."""
    del archive_root_path
    del db_anchor
    return True


def archive_file_set_root_for_paths(*, archive_root_path: Path, db_anchor: Path) -> Path:
    """Return the configured archive root."""
    if db_anchor.name == "index.db":
        return db_anchor.parent
    return archive_root_path


def active_index_db_path() -> Path:
    """Currently active query/index database path."""
    return index_db_path()


def browser_capture_spool_root() -> Path:
    """Default browser-capture source artifact spool."""
    return data_home() / "browser-capture"


def archive_root() -> Path:
    """Archive root (overridable via POLYLOGUE_ARCHIVE_ROOT)."""
    return _xdg_path("POLYLOGUE_ARCHIVE_ROOT", data_home())


def render_root() -> Path:
    """Render output root under the archive root."""
    return archive_root() / "render"


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


def gemini_cli_path() -> Path:
    """Gemini CLI local session workspace directory."""
    return Path.home() / ".gemini" / "tmp"


def hermes_sessions_path() -> Path:
    """Hermes agent state directory."""
    return Path.home() / ".hermes"


def antigravity_path() -> Path:
    """Antigravity local state directory."""
    return Path.home() / ".gemini" / "antigravity"


GEMINI_DRIVE_FOLDER = "Google AI Studio"


def blob_store_root() -> Path:
    """Content-addressed blob store root directory."""
    return data_home() / "blob"


def hooks_sidecar_dir() -> Path:
    """Directory where hook scripts drop session event data.

    AI coding agent hooks (Claude Code, Codex) write structured JSONL
    event records here. The daemon watcher picks them up for ingestion.
    """
    return data_home() / "hooks"


def drive_cache_path() -> Path:
    """Local cache directory for Drive-sourced files."""
    return data_home() / "drive-cache"
