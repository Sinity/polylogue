"""Degraded-aware archive open for live ingest writers (polylogue-gbs02).

The acquire-only degraded mode (``DegradedReason.derived_only``) skips all
parse/materialize work — but the ordinary ``ArchiveStore.open_existing``
writer open still validates EVERY tier and hard-refuses when the index tier
sits at a semantic-reparse distance (exactly the pre-rebuild state the mode
exists for; the live archive's index.db 46 vs code 57 reproduces the refusal
deterministically). Both live write paths therefore route their open through
this helper: in acquire-only mode it opens the source-tier-only writer,
which validates the durable tiers it will write and never opens a derived
tier at all.
"""

from __future__ import annotations

from pathlib import Path

from polylogue.core.degraded import degraded_reason
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore


def _open_archive_for_live_write(archive_root: Path) -> ArchiveStore:
    """Open the archive for a live ingest write pass.

    In acquire-only degraded mode (a degraded reason flagged
    ``derived_only``), returns the source-tier acquisition writer: durable
    tiers validated, derived tiers never opened, only raw admission usable —
    the per-record acquire-then-skip-parse branches in the callers guarantee
    nothing beyond raw admission is reached. Otherwise returns the ordinary
    full writer, preserving its all-tier validation exactly.
    """
    reason = degraded_reason()
    if reason is not None and reason.derived_only:
        return ArchiveStore.open_source_tier_acquisition(archive_root)
    return ArchiveStore.open_existing(archive_root, read_only=False)
