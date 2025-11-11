"""Shared filesystem paths and helpers for Polylogue."""

from __future__ import annotations

import os
from pathlib import Path


def _xdg_path(env_var: str, fallback: Path) -> Path:
    raw = os.environ.get(env_var)
    if raw:
        return Path(raw).expanduser()
    return fallback


CONFIG_ROOT = _xdg_path("XDG_CONFIG_HOME", Path.home() / ".config")
DATA_ROOT = _xdg_path("XDG_DATA_HOME", Path.home() / ".local/share")
CACHE_ROOT = _xdg_path("XDG_CACHE_HOME", Path.home() / ".cache")
STATE_ROOT = _xdg_path("XDG_STATE_HOME", Path.home() / ".local/state")


CONFIG_HOME = CONFIG_ROOT / "polylogue"
DATA_HOME = DATA_ROOT / "polylogue"
CACHE_HOME = CACHE_ROOT / "polylogue"
STATE_HOME = STATE_ROOT / "polylogue"


__all__ = [
    "CONFIG_HOME",
    "DATA_HOME",
    "CACHE_HOME",
    "STATE_HOME",
    "CONFIG_ROOT",
    "DATA_ROOT",
    "CACHE_ROOT",
    "STATE_ROOT",
]
