"""Shared filesystem paths and helpers for Polylogue.

This module defines all paths used by polylogue. There is no configuration file.
All paths follow XDG Base Directory specification with sensible defaults.

Path resolution is lazy — functions read environment variables at call time,
not import time. This eliminates the need for importlib.reload() in tests:
just set the env var and call the function again.
"""

from __future__ import annotations

from polylogue.paths_config import get_drive_config, get_index_config, get_sources
from polylogue.paths_models import DriveConfig, IndexConfig, Source
from polylogue.paths_roots import (
    GEMINI_DRIVE_FOLDER,
    archive_root,
    cache_home,
    cache_root,
    claude_code_path,
    codex_path,
    config_home,
    config_root,
    data_home,
    data_root,
    db_path,
    drive_cache_path,
    drive_credentials_path,
    drive_token_path,
    inbox_root,
    render_root,
    state_home,
    state_root,
)
from polylogue.paths_sanitize import conversation_render_root, is_within_root, safe_path_component

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
    "drive_cache_path",
    # Classes
    "Source",
    "DriveConfig",
    "IndexConfig",
    # Functions
    "get_sources",
    "get_drive_config",
    "get_index_config",
    "safe_path_component",
    "conversation_render_root",
    "is_within_root",
]
