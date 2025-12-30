"""Shared filesystem paths and helpers for Polylogue."""

from __future__ import annotations

import os
import re
from hashlib import sha256
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
    "CONFIG_HOME",
    "DATA_HOME",
    "CACHE_HOME",
    "STATE_HOME",
    "CONFIG_ROOT",
    "DATA_ROOT",
    "CACHE_ROOT",
    "STATE_ROOT",
    "safe_path_component",
    "is_within_root",
]
