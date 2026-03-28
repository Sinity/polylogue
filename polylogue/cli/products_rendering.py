"""Rendering helpers for archive-product CLI commands.

All per-product-type renderers are replaced by the generic
``render_product_items()`` in the product registry. This module retains
the ``summarize_archive_debt`` helper from support.
"""

from __future__ import annotations

from polylogue.cli.products_rendering_support import summarize_archive_debt
from polylogue.products.registry import render_product_items

__all__ = [
    "render_product_items",
    "summarize_archive_debt",
]
