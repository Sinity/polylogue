"""Canonical filesystem paths for Polylogue.

The package root owns only directory layout and concrete file path
resolution. Configuration models/loaders live in :mod:`polylogue.config`;
path-safety helpers live in :mod:`polylogue.paths.sanitize`.
"""

from __future__ import annotations

from polylogue.paths._roots import (
    GEMINI_DRIVE_FOLDER,
    archive_root,
    blob_store_root,
    browser_capture_spool_root,
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
    render_root,
    state_home,
    state_root,
)

__all__ = [
    "GEMINI_DRIVE_FOLDER",
    "archive_root",
    "blob_store_root",
    "browser_capture_spool_root",
    "cache_home",
    "cache_root",
    "claude_code_path",
    "codex_path",
    "config_home",
    "config_root",
    "data_home",
    "data_root",
    "db_path",
    "drive_cache_path",
    "drive_credentials_path",
    "drive_token_path",
    "render_root",
    "state_home",
    "state_root",
]
