"""Path sanitization utilities for preventing traversal attacks."""

from __future__ import annotations

import hashlib

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

    # Detect traversal attempts (..). A prior symlink check here
    # (Path(v).is_symlink() walked against the process's own CWD) was
    # removed: this value is provider-reported attachment path metadata that
    # is never opened relative to CWD, so the check tested an unrelated
    # filesystem location and could never catch a real traversal-via-symlink
    # (jsy). Traversal and control-character stripping remain the real guard.
    has_traversal = ".." in original_v

    # If traversal was detected, hash to prevent re-assembly
    if has_traversal:
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
