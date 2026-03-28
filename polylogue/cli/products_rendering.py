"""Rendering helpers for archive-product CLI commands.

All per-product-type renderers are replaced by the generic
``render_product_items()`` in the product registry. This module retains
the ``summarize_archive_debt`` helper.
"""

from __future__ import annotations

from polylogue.cli.machine_errors import emit_success
from polylogue.products.registry import render_product_items


def model_payload(item: object) -> object:
    return item.model_dump(mode="json") if hasattr(item, "model_dump") else item


def emit_product_list(*, key: str, items: list[object]) -> None:
    emit_success({"count": len(items), key: [model_payload(item) for item in items]})


def summarize_archive_debt(items: list[object]) -> dict[str, int]:
    actionable = [item for item in items if getattr(item, "healthy", True) is False]
    return {
        "tracked_items": len(items),
        "actionable_items": len(actionable),
        "issue_rows": sum(int(getattr(item, "issue_count", 0) or 0) for item in items),
    }


__all__ = [
    "emit_product_list",
    "model_payload",
    "render_product_items",
    "summarize_archive_debt",
]
