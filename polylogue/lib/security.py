"""Path sanitization utilities for preventing traversal attacks."""

from __future__ import annotations

from pathlib import Path

from polylogue.lib.log import get_logger

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

    # Check for symlinks in the path by checking path components
    has_symlink = False
    try:
        p = Path(v)
        for parent in [p] + list(p.parents):
            if parent.is_symlink():
                has_symlink = True
                break
    except Exception:
        logger.warning("Error checking symlinks in path: %s", v)

    # If traversal or symlinks were detected, hash to prevent re-assembly
    if has_traversal or has_symlink:
        import hashlib

        original_hash = hashlib.sha256(original_v.encode()).hexdigest()[:12]
        v = f"_blocked_{original_hash}"
    # For safe paths, clean up components but preserve absolute/relative structure
    else:
        try:
            parts = []
            for component in v.split("/"):
                component = component.strip()
                if component and component not in (".", ".."):
                    parts.append(component)
            if original_v.startswith("/"):
                v = "/" + "/".join(parts) if parts else "/"
            else:
                v = "/".join(parts) if parts else v
        except Exception:
            logger.warning("Error cleaning path components: %s", v)

    return v if v else None
