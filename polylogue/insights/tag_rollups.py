"""Origin tag-rollup synthesis.

The archive tag table stores explicit and automatic tags. Source identity
already lives on ``sessions.origin``, so the archive tag-rollup read does not
store duplicate ``origin:<name>`` rows in ``session_tags``.

To preserve the public contract without duplicating origin identity into the
tag table, this helper synthesizes one ``origin:<origin>``
:class:`SessionTagRollupInsight` per origin directly from the session counts,
honouring the same origin/date filters the archive read applies.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from polylogue.core.enums import Provider
from polylogue.core.sources import origin_from_provider
from polylogue.insights.archive import SessionTagRollupInsight
from polylogue.insights.archive_models import ArchiveInsightProvenance

if TYPE_CHECKING:
    from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore

_ORIGIN_TAG_PREFIX = "origin:"


def synthesize_provider_tag_rollups(
    archive: ArchiveStore,
    *,
    provider: str | None = None,
    query: str | None = None,
    since_ms: int | None = None,
    until_ms: int | None = None,
    materialized_at: str,
) -> list[SessionTagRollupInsight]:
    """Build ``origin:<name>`` rollups from archive session origin counts.

    Returns an empty list when a ``query`` filter is present that no
    synthesized origin tag can satisfy, so the substring filter behaves the
    same as it does against real tag rows.
    """
    counts = archive.stats_by(
        "origin",
        since_ms=since_ms,
        until_ms=until_ms,
    )
    needle = query.strip().lower() if query else None
    # Provider -> Origin, not the reverse: this direction is total and
    # well-defined even for the non-injective Origin.AISTUDIO_DRIVE fiber --
    # both "gemini" and "drive" forward-map to the same "aistudio-drive"
    # origin_filter (polylogue-9e5.8 Step 5 investigation / polylogue-4rrv).
    # There is nothing here for a Source-family disambiguator to recover:
    # disambiguation only matters for the *reverse* Origin -> Provider
    # lookup (see core/sources.py's provider_from_origin family_hint), which
    # this function never performs. What *is* coarse is the resulting
    # rollup: origin_filter can only select the whole "aistudio-drive"
    # bucket, so provider="gemini" and provider="drive" return identical
    # counts today (both fold into one origin-tag rollup, and per-origin
    # ArchiveStore.stats_by("origin", ...) has no finer axis to group on --
    # see polylogue-4rrv's follow-up bead for the durable capture-mode field
    # that would be needed to split them).
    origin_filter = origin_from_provider(Provider.from_string(provider)).value if provider is not None else None
    rollups: list[SessionTagRollupInsight] = []
    for origin_value, count in counts.items():
        if origin_filter is not None and origin_value != origin_filter:
            continue
        if count <= 0:
            continue
        tag = f"{_ORIGIN_TAG_PREFIX}{origin_value}"
        if needle is not None and needle not in tag.lower():
            continue
        rollups.append(
            SessionTagRollupInsight(
                tag=tag,
                session_count=count,
                logical_session_count=count,
                explicit_count=0,
                auto_count=0,
                origin_breakdown={origin_value: count},
                repo_breakdown={},
                provenance=ArchiveInsightProvenance(
                    materializer_version=1,
                    materialized_at=materialized_at,
                ),
            )
        )
    return rollups
