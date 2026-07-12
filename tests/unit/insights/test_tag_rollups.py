"""Regression tests for ``synthesize_provider_tag_rollups`` (polylogue-4rrv).

polylogue-9e5.8's Step 5 investigation into the non-injective
``Origin.AISTUDIO_DRIVE`` collapse (``Provider.GEMINI`` and ``Provider.DRIVE``
both produce it) named ``insights/tag_rollups.py:49`` as a candidate leak
site. Reading it to ground truth shows it performs the *forward*
``Provider -> Origin`` conversion (via ``origin_from_provider``), which is
total and well-defined even for this fiber -- there is no reverse lookup here
for a disambiguator to fix. These tests pin that: both ``provider="gemini"``
and ``provider="drive"`` correctly resolve to the ``aistudio-drive`` origin
filter (not silently falling back to ``unknown-export`` or raising), proving
the site was already correct and is not blocked by the collapse.
"""

from __future__ import annotations

from polylogue.insights.tag_rollups import synthesize_provider_tag_rollups


class _FakeArchive:
    """Minimal stand-in for ``ArchiveStore.stats_by`` used by tag rollups."""

    def __init__(self, counts: dict[str, int]) -> None:
        self._counts = counts

    def stats_by(
        self,
        group_by: str,
        *,
        since_ms: int | None = None,
        until_ms: int | None = None,
    ) -> dict[str, int]:
        assert group_by == "origin"
        return dict(self._counts)


def _counts() -> dict[str, int]:
    return {
        "aistudio-drive": 7,
        "codex-session": 3,
        "claude-code-session": 5,
    }


def test_provider_gemini_filters_to_aistudio_drive_origin() -> None:
    rollups = synthesize_provider_tag_rollups(
        _FakeArchive(_counts()),  # type: ignore[arg-type]
        provider="gemini",
        materialized_at="2026-07-12T00:00:00+00:00",
    )
    assert [r.tag for r in rollups] == ["origin:aistudio-drive"]
    assert rollups[0].session_count == 7


def test_provider_drive_filters_to_the_same_aistudio_drive_origin() -> None:
    """Provider.GEMINI and Provider.DRIVE both forward-map to
    Origin.AISTUDIO_DRIVE (origin_from_provider is total, non-injective only
    in the reverse direction) -- "drive" must resolve identically to
    "gemini", not fall back to unknown-export or raise."""
    rollups = synthesize_provider_tag_rollups(
        _FakeArchive(_counts()),  # type: ignore[arg-type]
        provider="drive",
        materialized_at="2026-07-12T00:00:00+00:00",
    )
    assert [r.tag for r in rollups] == ["origin:aistudio-drive"]
    assert rollups[0].session_count == 7


def test_gemini_and_drive_provider_filters_agree() -> None:
    """Documents the known coarseness (not a bug this bead fixes): since no
    storage tier persists which acquisition mechanism produced an
    aistudio-drive session, both provider filters necessarily return the
    identical rollup. Splitting them needs a durable capture-mode field
    (tracked as this bead's follow-up), not a smarter reverse lookup here."""
    gemini_rollups = synthesize_provider_tag_rollups(
        _FakeArchive(_counts()),  # type: ignore[arg-type]
        provider="gemini",
        materialized_at="2026-07-12T00:00:00+00:00",
    )
    drive_rollups = synthesize_provider_tag_rollups(
        _FakeArchive(_counts()),  # type: ignore[arg-type]
        provider="drive",
        materialized_at="2026-07-12T00:00:00+00:00",
    )
    assert [r.tag for r in gemini_rollups] == [r.tag for r in drive_rollups]
    assert [r.session_count for r in gemini_rollups] == [r.session_count for r in drive_rollups]


def test_no_provider_filter_returns_every_origin() -> None:
    rollups = synthesize_provider_tag_rollups(
        _FakeArchive(_counts()),  # type: ignore[arg-type]
        materialized_at="2026-07-12T00:00:00+00:00",
    )
    assert {r.tag for r in rollups} == {
        "origin:aistudio-drive",
        "origin:codex-session",
        "origin:claude-code-session",
    }
