"""FTS5 escaping and normalization helpers."""

from __future__ import annotations

import re
from datetime import datetime, timezone

_FTS5_SPECIAL = re.compile(r'''['":*^(){}\[\]|&!+\-\\;%=$,<>@#`~]''')
_FTS5_OPERATORS = {"AND", "OR", "NOT", "NEAR"}
_ASTERISK_ONLY = re.compile(r"^\*+$")


def sort_key_to_iso(sort_key: object) -> str | None:
    """Convert a sort_key (epoch float) to ISO 8601, or None."""
    if sort_key is None:
        return None
    try:
        return datetime.fromtimestamp(float(sort_key), tz=timezone.utc).isoformat()
    except (TypeError, ValueError, OSError):
        return None


def normalize_fts5_query(query: str) -> str | None:
    """Normalize a raw query into a safe FTS5 MATCH expression."""
    if not query or not query.strip():
        return None
    fts_query = escape_fts5_query(query)
    return None if fts_query == '""' else fts_query


def escape_fts5_query(query: str) -> str:
    """Escape a query string for safe use in FTS5 MATCH clauses."""
    def _quoted(value: str) -> str:
        escaped = value.replace('"', '""')
        return f'"{escaped}"'

    if not query or not query.strip():
        return '""'

    query = re.sub(r'[\x00-\x1f\x7f]', '', query.strip())
    if not query:
        return '""'
    if _ASTERISK_ONLY.match(query):
        return '""'
    if query.upper() in _FTS5_OPERATORS:
        return f'"{query}"'
    if _FTS5_SPECIAL.search(query):
        return _quoted(query)

    words = query.split()
    if len(words) > 1:
        if words[0].upper() in _FTS5_OPERATORS or words[-1].upper() in _FTS5_OPERATORS:
            return _quoted(query)
        for index in range(len(words) - 1):
            if words[index].upper() in _FTS5_OPERATORS and words[index + 1].upper() in _FTS5_OPERATORS:
                return _quoted(query)
    return query


__all__ = ["escape_fts5_query", "normalize_fts5_query", "sort_key_to_iso"]
