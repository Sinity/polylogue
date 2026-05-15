"""Query verb names -- lightweight so startup never imports query_verbs."""

from __future__ import annotations

VERB_NAMES: frozenset[str] = frozenset(
    {
        "list",
        "count",
        "stats",
        "open",
        "show",
        "bulk-export",
        "delete",
        "messages",
        "raw",
        "select",
    }
)
QUERY_VERB_NAMES = VERB_NAMES
QUERY_VERBS = VERB_NAMES

__all__ = ["QUERY_VERBS", "QUERY_VERB_NAMES", "VERB_NAMES"]
