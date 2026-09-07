"""Unified hashing utilities for Polylogue.

This module consolidates all SHA-256 hashing operations to avoid duplicate
implementations scattered across the codebase.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from polylogue.core.digest import QUERY, digest, nfc


def hash_text(text: str) -> str:
    """Hash UTF-8 text to full SHA-256 hex digest (64 chars).

    Applies NFC Unicode normalization to ensure visually identical
    strings produce identical hashes regardless of normalization form.
    """
    return hashlib.sha256(nfc(text).encode("utf-8")).hexdigest()


def hash_text_short(text: str, length: int = 16) -> str:
    """Hash UTF-8 text to truncated SHA-256 hex digest.

    Applies NFC Unicode normalization before hashing.
    """
    return hash_text(text)[:length]


def hash_payload(payload: object) -> str:
    """Hash a JSON-serializable object under the ``query`` digest profile.

    ASCII-escaped and not NFC-normalized: a caller needing
    normalization-invariant hashing normalizes the payload first, where it can
    tell a field token from a literal, or names a profile that normalizes.
    """
    return digest(payload, QUERY)


def hash_bytes(payload: bytes) -> str:
    """Hash bytes to a full SHA-256 hex digest."""
    return hashlib.sha256(payload).hexdigest()


def hash_file(path: Path) -> str:
    """Hash file contents to full SHA-256 hex digest (streams 1MB chunks)."""
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


__all__ = ["hash_bytes", "hash_file", "hash_payload", "hash_text", "hash_text_short"]
