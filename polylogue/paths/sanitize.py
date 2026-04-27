"""Filesystem-safe path helpers."""

from __future__ import annotations

import os
import re
import unicodedata
from hashlib import sha256
from pathlib import Path

_SAFE_PATH_COMPONENT_RE = re.compile(r"[^A-Za-z0-9._-]")


def safe_path_component(raw: object | None, *, fallback: str = "item") -> str:
    """Return a filesystem-safe path component derived from raw input.

    Input is NFC-normalized before sanitization so that visually-confusable
    Unicode codepoints (e.g., NFKC-equivalent or compatibility-decomposed
    forms) collapse to a stable canonical form before the ASCII allowlist
    rewrites everything else to underscore-plus-digest.
    """
    if raw is None:
        raw = ""
    value = unicodedata.normalize("NFC", str(raw)).strip()
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


def conversation_render_root(base_render_root: Path, provider: str, conversation_id: str) -> Path:
    """Return the sanitized render directory for a conversation."""
    safe_provider = safe_path_component(provider, fallback="provider")
    safe_conversation = safe_path_component(conversation_id, fallback="conversation")
    return base_render_root / safe_provider / safe_conversation


def is_within_root(path: Path, root: Path) -> bool:
    """Return True if path resolves within root."""
    try:
        path.resolve(strict=False).relative_to(root.resolve(strict=False))
    except ValueError:
        return False
    return True


__all__ = [
    "conversation_render_root",
    "is_within_root",
    "safe_path_component",
]
