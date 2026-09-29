"""TimestampBounds streams the bound selection the list-based sort made."""

from __future__ import annotations

from polylogue.sources.parsers.drive_support_text import TimestampBounds


def _bounds(values: list[str | None]) -> tuple[str | None, str | None]:
    bounds = TimestampBounds()
    for value in values:
        bounds.observe(value)
    earliest = bounds.earliest[1] if bounds.earliest else None
    latest = bounds.latest[1] if bounds.latest else None
    return earliest, latest


def test_repeated_spelling_at_the_latest_instant_keeps_the_last_distinct_spelling() -> None:
    """Anti-vacuity: a plain ``>=`` update returns the final repeated ``Z``."""
    assert _bounds(["2026-01-01T00:00:00Z", "2026-01-01T00:00:00+00:00", "2026-01-01T00:00:00Z"]) == (
        "2026-01-01T00:00:00Z",
        "2026-01-01T00:00:00+00:00",
    )


def test_a_later_instant_resets_the_tie_memory() -> None:
    assert _bounds(
        [
            "2026-01-01T00:00:00Z",
            "2026-01-02T00:00:00+00:00",
            "2026-01-02T00:00:00Z",
            "2026-01-02T00:00:00+00:00",
            None,
            "not a timestamp",
        ]
    ) == ("2026-01-01T00:00:00Z", "2026-01-02T00:00:00Z")
