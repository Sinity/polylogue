"""Resident filesystem bundle composition on the original pinned insight reader."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING

from polylogue.analysis.export_bundles import export_insight_bundle
from polylogue.operations.insight_export_contracts import InsightExportResult, decode_insight_export_request

if TYPE_CHECKING:
    from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore


def execute_insight_export(
    payload: Mapping[str, object], *, archive: ArchiveStore, checkpoint: Callable[[], None]
) -> dict[str, object]:
    request = decode_insight_export_request(payload)
    bundle = export_insight_bundle(archive, request.request, checkpoint=checkpoint)
    return InsightExportResult(bundle=bundle, outcome=bundle.outcome).model_dump(mode="json")
