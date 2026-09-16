"""Public origin-filter projections for archive surfaces.

The source-admission registry owns the vocabulary.  This small operations
adapter keeps surface packages from importing parser/source declarations
 directly while giving CLI, MCP, and other adapters the same projection.
"""

from __future__ import annotations

from collections.abc import Iterable

from polylogue.sources.origin_specs import public_origin_tokens

__all__ = ["public_origin_filter_tokens", "unknown_origin_filter_tokens"]


def public_origin_filter_tokens() -> tuple[str, ...]:
    """Return origin tokens accepted by public filter arguments."""
    return public_origin_tokens()


def unknown_origin_filter_tokens(values: Iterable[str]) -> tuple[str, ...]:
    """Return the requested filter tokens that are not public origins.

    polylogue-01fe: the same bad ``origin`` had three answers -- the CLI
    rejected it, MCP accepted it and returned the unfiltered aggregate, and the
    daemon's ``?origin=`` went straight onto the lenient wire-token normalizer
    and answered HTTP 200 with ``total: 0``.  A caller who mistypes an origin
    was told "no data".

    This is deliberately NOT the lenient ``Origin`` string constructor.  That
    constructor's job is normalizing untrusted provider-export tokens, mapping
    anything unrecognized to ``unknown-export``; gating *user* input is a
    different contract, and conflating the two is what produced the silent
    empty.  Every public surface calls this one function so the three
    validators become one.
    """
    choices = set(public_origin_filter_tokens())
    return tuple(token for token in (str(value).strip() for value in values) if token and token not in choices)
