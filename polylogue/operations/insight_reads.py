"""Resident adaptation of canonical insight pages on one selected snapshot."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING

from polylogue.analysis.insight_reads import read_insight_page
from polylogue.operations.insight_contracts import InsightListRequest
from polylogue.surfaces.outcome import decide_outcome

if TYPE_CHECKING:
    from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore


def execute_insight_read(
    payload: Mapping[str, object], *, archive: ArchiveStore, checkpoint: Callable[[], None]
) -> dict[str, object]:
    request = InsightListRequest.model_validate(payload)
    checkpoint()
    items = read_insight_page(archive, request.page.query)
    checkpoint()
    return {
        "page": {
            "insight": request.page.insight,
            "items": [item.model_dump(mode="json") for item in items],
            "total": len(items),
        },
        "outcome": decide_outcome(matched=len(items)).to_dict(),
    }
