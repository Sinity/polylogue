"""Path-component bounded matching for query cwd filters."""

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


def escaped_sql_path_prefix_patterns(prefix: object) -> tuple[str, str]:
    """Return exact and LIKE-prefix params for path-component bounded SQL."""
    normalized = normalize_path_prefix(prefix)
    escaped = normalized.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    if escaped == "/":
        return normalized, "/%"
    return normalized, f"{escaped}/%"


__all__ = ["escaped_sql_path_prefix_patterns", "normalize_path_prefix", "path_matches_prefix"]
