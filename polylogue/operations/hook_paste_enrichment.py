"""Product operation for replaying retained hook paste evidence by session."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from polylogue.sources.live.hook_paste_enrichment import enrich_paste_from_hooks


def retry_recorded_hook_paste(db_path: Path, session_ids: Sequence[str]) -> int:
    """Reapply durable hook evidence for the sessions named by convergence debt."""
    return enrich_paste_from_hooks(db_path, session_ids=session_ids)
