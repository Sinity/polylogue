"""Regression tests for origin-filtered session-tag rollups (polylogue-4rrv)."""

from __future__ import annotations

from polylogue.analysis.tag_rollups import synthesize_origin_tag_rollups


class _FakeArchive:
    """Minimal stand-in for ``ArchiveStore.origin_session_counts`` used by tag rollups."""

    def __init__(self, counts: dict[str, int]) -> None:
        self._counts = counts

    def origin_session_counts(
        self,
        *,
        origin: str | None = None,
        since_ms: int | None = None,
        until_ms: int | None = None,
    ) -> dict[str, tuple[int, int]]:
        return {key: (value, value) for key, value in self._counts.items() if origin is None or key == origin}


def _counts() -> dict[str, int]:
    return {
        "aistudio-drive": 7,
        "codex-session": 3,
        "claude-code-session": 5,
    }


def test_origin_filter_selects_one_rollup() -> None:
    """A canonical origin filter selects its public origin-tag payload."""
    rollups = synthesize_origin_tag_rollups(
        _FakeArchive(_counts()),  # type: ignore[arg-type]
        origin="aistudio-drive",
        materialized_at="2026-07-12T00:00:00+00:00",
    )
    assert [(rollup.tag, rollup.session_count) for rollup in rollups] == [("origin:aistudio-drive", 7)]


def test_no_origin_filter_returns_every_origin() -> None:
    rollups = synthesize_origin_tag_rollups(
        _FakeArchive(_counts()),  # type: ignore[arg-type]
        materialized_at="2026-07-12T00:00:00+00:00",
    )
    assert {r.tag for r in rollups} == {
        "origin:aistudio-drive",
        "origin:codex-session",
        "origin:claude-code-session",
    }
