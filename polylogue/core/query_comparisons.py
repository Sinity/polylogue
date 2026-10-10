"""Shared literal path and Unicode prose comparisons for query filters."""

from __future__ import annotations


def normalize_path_prefix(value: object) -> str:
    """Normalize a user path prefix for component-bounded comparison."""
    raw = str(value).strip().replace("\\", "/")
    while "//" in raw:
        raw = raw.replace("//", "/")
    if raw != "/":
        raw = raw.rstrip("/")
    return raw


def path_matches_prefix(path: object, prefix: object) -> bool:
    """Return true when ``path`` equals ``prefix`` or is under it."""
    normalized_path = normalize_path_prefix(path)
    normalized_prefix = normalize_path_prefix(prefix)
    if not normalized_prefix:
        return False
    if normalized_path == normalized_prefix:
        return True
    if normalized_prefix == "/":
        return normalized_path.startswith("/")
    return normalized_path.startswith(f"{normalized_prefix}/")


def prose_contains(value: object, term: object) -> bool:
    """Compare prose with Python Unicode lower semantics and literal substrings."""
    return bool(value) and str(term).lower() in str(value).lower()


__all__ = ["normalize_path_prefix", "path_matches_prefix", "prose_contains"]
