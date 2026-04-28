"""Unified hashing utilities for Polylogue.

This module consolidates all SHA-256 hashing operations to avoid duplicate
implementations scattered across the codebase.
"""

from __future__ import annotations

import hashlib
import unicodedata
from pathlib import Path


def hash_text(text: str) -> str:
    """Hash UTF-8 text to full SHA-256 hex digest (64 chars).

    Applies NFC Unicode normalization to ensure visually identical
    strings produce identical hashes regardless of normalization form.
    """
    # Normalize to NFC (Canonical Decomposition, followed by Canonical Composition)
    # This ensures "café" hashes the same whether é is precomposed or decomposed
    normalized = unicodedata.normalize("NFC", text)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def hash_text_short(text: str, length: int = 16) -> str:
    """Hash UTF-8 text to truncated SHA-256 hex digest.

    Applies NFC Unicode normalization before hashing.
    """
    return hash_text(text)[:length]


def hash_payload(payload: object) -> str:
    """Hash a JSON-serializable object to full SHA-256 hex digest.

    Uses stdlib json for deterministic output across environments.
    orjson would be faster but can produce different byte output for
    non-ASCII content between versions, breaking content-addressed storage.

    String values within the payload are NOT NFC-normalized here.
    Callers should normalize strings before including in payload if
    normalization-invariant hashing is required.
    """
    import json

    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def hash_file(path: Path) -> str:
    """Hash file contents to full SHA-256 hex digest (streams 1MB chunks)."""
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


__all__ = ["hash_text", "hash_text_short", "hash_payload", "hash_file"]
