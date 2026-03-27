"""Workflow helpers for archive-product CLI commands.

All per-product-type workflow functions are now handled by the generic
``fetch_products()`` in the product registry. This module is retained
as a thin backwards-compatibility surface for any transitive callers.
"""

from __future__ import annotations

from polylogue.products.registry import fetch_products

__all__ = ["fetch_products"]
