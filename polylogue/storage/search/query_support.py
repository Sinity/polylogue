"""FTS5 escaping and normalization helpers."""

from __future__ import annotations

import re
from datetime import datetime, timezone

_FTS5_SPECIAL = re.compile(r"""['":*^(){}\[\]|&!+\-\\;%=$,<>@#`~./?]""")
_FTS5_OPERATORS = {"AND", "OR", "NOT", "NEAR"}
_ASTERISK_ONLY = re.compile(r"^\*+$")
_TERM_TOKEN = re.compile(r"[\w*]+", re.UNICODE)


def extract_match_terms(query: str) -> tuple[str, ...]:
    """Extract user-facing match terms from a raw FTS query string.

    Strips FTS5 boolean operators (``AND``/``OR``/``NOT``/``NEAR``),
    quote/colon/paren punctuation, and prefix asterisks so the returned
    tuple represents the literal tokens a reader should expect to see
    highlighted in a hit. Preserves order, deduplicates case-insensitively,
    and lowercases for stable consumer comparisons.

    Used by ``SessionSearchHit.matched_terms`` to populate per-hit
    why-this-matched evidence on lexical (FTS5) search paths (#1267).
    """
    if not query or not query.strip():
        return ()
    tokens = _TERM_TOKEN.findall(query)
    seen: set[str] = set()
    out: list[str] = []
    for raw in tokens:
        token = raw.lower().rstrip("*")
        if not token or token.upper() in _FTS5_OPERATORS:
            continue
        if token in seen:
            continue
        seen.add(token)
        out.append(token)
    return tuple(out)


def sort_key_to_iso(sort_key: object) -> str | None:
    """Convert a sort_key (epoch float) to ISO 8601, or None."""
    if sort_key is None:
        return None
    if not isinstance(sort_key, (int, float, str)):
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

    query = re.sub(r"[\x00-\x1f\x7f]", "", query.strip())
    if not query:
        return '""'
    if _ASTERISK_ONLY.match(query):
        return '""'
    if query.upper() in _FTS5_OPERATORS:
        return f'"{query}"'
    if _FTS5_SPECIAL.search(query):
        # * at the end of a word token is valid FTS5 prefix syntax.
        # Only quote if other special characters are present, or if *
        # appears in a non-prefix position (e.g. *word, w*rd).
        if "*" in query:
            # A ``*`` is only a valid FTS5 prefix when it suffixes a word token
            # (``word*``). Bare asterisks (``*``, ``* *``) are not valid prefix
            # syntax and must be quoted, so only strip word-suffixed ``*`` here.
            without_prefix = re.sub(r"(\w)\*(\s|$)", r"\1\2", query).rstrip("*")
            if _FTS5_SPECIAL.search(without_prefix):
                return _quoted(query)
            # Only special char was * in prefix position — don't quote
            # for that, but still check operator edge cases below.
        else:
            return _quoted(query)

    def _is_op(word: str) -> bool:
        return word.rstrip("*").upper() in _FTS5_OPERATORS

    words = query.split()
    if len(words) >= 1:
        if _is_op(words[0]) or (len(words) > 1 and _is_op(words[-1])):
            return _quoted(query)
        if len(words) > 1:
            for index in range(len(words) - 1):
                if _is_op(words[index]) and _is_op(words[index + 1]):
                    return _quoted(query)
    return query


__all__ = [
    "escape_fts5_query",
    "extract_match_terms",
    "normalize_fts5_query",
    "sort_key_to_iso",
]
