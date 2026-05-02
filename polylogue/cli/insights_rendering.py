"""Rendering helpers for archive-insight CLI commands.

All per-insight-type renderers are replaced by the generic
``render_insight_items()`` in the insight registry. This module retains
the ``summarize_archive_debt`` helper.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from polylogue.cli.shared.machine_errors import emit_success
from polylogue.insights.registry import render_insight_items


@runtime_checkable
class SupportsModelDump(Protocol):
    def model_dump(self, *, mode: str) -> object: ...


def model_payload(item: object) -> object:
    return item.model_dump(mode="json") if isinstance(item, SupportsModelDump) else item


def model_payloads(items: Sequence[object]) -> list[object]:
    return [model_payload(item) for item in items]


def emit_product_list(*, key: str, items: list[object]) -> None:
    emit_success({"count": len(items), key: model_payloads(items)})


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
    "model_payloads",
    "render_insight_items",
    "summarize_archive_debt",
]
