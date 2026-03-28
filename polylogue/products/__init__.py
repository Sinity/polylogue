"""Data-driven derived product system.

Product types are registered descriptors. CLI rendering, MCP exposure,
and library API are generic — they take a product type name and produce
appropriate output. No per-product-type rendering/workflow/command files.
"""

from polylogue.products.registry import (
    PRODUCT_REGISTRY,
    ProductField,
    ProductType,
    get_product_type,
    list_product_types,
    render_product_items,
)

__all__ = [
    "PRODUCT_REGISTRY",
    "ProductField",
    "ProductType",
    "get_product_type",
    "list_product_types",
    "render_product_items",
]
