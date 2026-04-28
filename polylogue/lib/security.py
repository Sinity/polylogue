"""Path sanitization utilities for preventing traversal attacks."""

from __future__ import annotations

import hashlib
from pathlib import Path

from polylogue.logging import get_logger

logger = get_logger(__name__)


def sanitize_path(v: str | None) -> str | None:
    """Sanitize path to prevent traversal attacks and other security issues."""
    if v is None:
        return v

    original_v = v

    # Remove null bytes
    v = v.replace("\x00", "")

    # Remove control characters (ASCII < 32 and 127)
    v = "".join(c for c in v if ord(c) >= 32 and ord(c) != 127)

    # Detect threats:
    # 1. Traversal attempts (..)
    # 2. Symlinks in path (potential traversal bypass)
    has_traversal = ".." in original_v

    # Check for symlinks in the path by checking path components.
    # On any filesystem error (PermissionError, OSError) treat the path as
    # suspicious — this guard sits in front of traversal protection and a
    # silent skip would let unreadable directories mask a real attack.
    has_symlink = False
    try:
        p = Path(v)
        for parent in [p] + list(p.parents):
            if parent.is_symlink():
                has_symlink = True
                break
    except OSError as exc:
        logger.warning("symlink check failed for path %r: %s; treating as suspicious", v, exc)
        has_symlink = True

    # If traversal or symlinks were detected, hash to prevent re-assembly
    if has_traversal or has_symlink:
        original_hash = hashlib.sha256(original_v.encode()).hexdigest()[:12]
        return f"_blocked_{original_hash}"

    # Safe path: clean up components
    parts = [c.strip() for c in v.split("/") if c.strip() and c.strip() not in (".", "..")]
    joined = "/".join(parts)

    # Absolute paths are rejected — no allowlist for safe directories.
    # An attacker-controlled input like "/etc/passwd" would otherwise pass
    # through to the filesystem. Callers with a known root should resolve
    # against it themselves.
    if v.startswith("/"):
        original_hash = hashlib.sha256(original_v.encode()).hexdigest()[:12]
        return f"_blocked_{original_hash}"

    return joined or v or None
