"""Query verb names -- lightweight so startup never imports query_verbs."""

from __future__ import annotations

VERB_NAMES: frozenset[str] = frozenset(
    {
        "analyze",
        "continue",
        "delete",
        "mark",
        "read",
        "select",
    }
)
QUERY_VERB_NAMES = VERB_NAMES
QUERY_VERBS = VERB_NAMES

__all__ = ["QUERY_VERBS", "QUERY_VERB_NAMES", "VERB_NAMES"]
