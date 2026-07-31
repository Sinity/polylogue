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
__all__ = ["VERB_NAMES"]
