"""Public origin-filter projections for archive surfaces.

The source-admission registry owns the vocabulary.  This small operations
adapter keeps surface packages from importing parser/source declarations
 directly while giving CLI, MCP, and other adapters the same projection.
"""

from __future__ import annotations

from polylogue.sources.origin_specs import public_origin_tokens

__all__ = ["public_origin_filter_tokens"]


def public_origin_filter_tokens() -> tuple[str, ...]:
    """Return origin tokens accepted by public filter arguments."""
    return public_origin_tokens()
