"""Unit tests for the shared confidence vocabulary (#1277)."""

from __future__ import annotations

import pytest

from polylogue.insights.confidence import (
    ConfidenceBand,
    confidence_band_from_stored,
    from_score,
    from_signals,
)

# ---------------------------------------------------------------------------
# Vocabulary stays str-compatible on the wire
# ---------------------------------------------------------------------------


def test_band_values_match_legacy_string_literals() -> None:
    """The enum members must serialize as the exact strings already on disk."""

    assert ConfidenceBand.STRONG.value == "strong"
    assert ConfidenceBand.MODERATE.value == "moderate"
    assert ConfidenceBand.WEAK.value == "weak"
    assert ConfidenceBand.NONE.value == "none"


def test_band_is_str_subclass_for_json_compat() -> None:
    """JSON / SQLite TEXT columns must see the same string as before."""

    assert isinstance(ConfidenceBand.STRONG, str)
    assert str(ConfidenceBand.MODERATE) == "ConfidenceBand.MODERATE"
    assert ConfidenceBand.MODERATE.value == "moderate"


# ---------------------------------------------------------------------------
# from_signals — support-level decision rule
# ---------------------------------------------------------------------------


def test_from_signals_fallback_forces_weak() -> None:
    band = from_signals(0.99, support_signals=("a", "b", "c"), fallback=True)
    assert band is ConfidenceBand.WEAK


def test_from_signals_empty_signals_is_weak() -> None:
    band = from_signals(0.99, support_signals=())
    assert band is ConfidenceBand.WEAK


def test_from_signals_low_confidence_is_weak() -> None:
    band = from_signals(0.40, support_signals=("a", "b"))
    assert band is ConfidenceBand.WEAK


def test_from_signals_strong_requires_two_signals_and_high_confidence() -> None:
    assert from_signals(0.80, support_signals=("a", "b")) is ConfidenceBand.STRONG
    # one signal alone — moderate at best
    assert from_signals(0.95, support_signals=("a",)) is ConfidenceBand.MODERATE
    # confidence just below strong floor
    assert from_signals(0.77, support_signals=("a", "b")) is ConfidenceBand.MODERATE


def test_from_signals_moderate_band_above_floor_below_strong() -> None:
    assert from_signals(0.60, support_signals=("a",)) is ConfidenceBand.MODERATE
    assert from_signals(0.70, support_signals=("a", "b")) is ConfidenceBand.MODERATE


# ---------------------------------------------------------------------------
# from_score — three-band split for raw float confidences
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("score", "expected"),
    [
        (0.0, ConfidenceBand.WEAK),
        (0.33, ConfidenceBand.WEAK),
        (0.34, ConfidenceBand.MODERATE),
        (0.5, ConfidenceBand.MODERATE),
        (0.66, ConfidenceBand.MODERATE),
        (0.67, ConfidenceBand.STRONG),
        (1.0, ConfidenceBand.STRONG),
    ],
)
def test_from_score_thirds(score: float, expected: ConfidenceBand) -> None:
    assert from_score(score) is expected


# ---------------------------------------------------------------------------
# confidence_band_from_stored — coerce stored string into typed band
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("strong", ConfidenceBand.STRONG),
        ("MODERATE", ConfidenceBand.MODERATE),
        ("  weak ", ConfidenceBand.WEAK),
        ("none", ConfidenceBand.NONE),
    ],
)
def test_confidence_band_from_stored_accepts_known_spellings(value: str, expected: ConfidenceBand) -> None:
    assert confidence_band_from_stored(value) is expected


def test_confidence_band_from_stored_passes_through_enum_instances() -> None:
    assert confidence_band_from_stored(ConfidenceBand.STRONG) is ConfidenceBand.STRONG


@pytest.mark.parametrize("value", [None, "", "unknown", "medium", "high"])
def test_confidence_band_from_stored_unknown_spellings_collapse_to_weak(value: str | None) -> None:
    assert confidence_band_from_stored(value) is ConfidenceBand.WEAK


# ---------------------------------------------------------------------------
# Integration: the converted payload helpers emit typed bands
# ---------------------------------------------------------------------------


def test_support_level_helper_returns_typed_band() -> None:
    from polylogue.storage.insights.session.profiles import support_level

    band = support_level(0.85, support_signals=("a", "b"))
    assert band is ConfidenceBand.STRONG
    # str-subclass invariant — the persisted SQLite TEXT value is the enum value
    assert band.value == "strong"

    band = support_level(0.30, support_signals=("a",))
    assert band is ConfidenceBand.WEAK


def test_repo_inference_strength_returns_typed_band() -> None:
    """The repo-strength helper uses the same vocabulary plus NONE."""

    from typing import cast

    from polylogue.archive.session.session_profile import SessionProfile
    from polylogue.storage.insights.session.profiles import repo_inference_strength

    class _Profile:
        repo_paths: tuple[str, ...] = ()
        repo_names: tuple[str, ...] = ()
        file_paths_touched: tuple[str, ...] = ()
        cwd_paths: tuple[str, ...] = ()

    empty = cast(SessionProfile, _Profile())
    assert repo_inference_strength(empty) is ConfidenceBand.NONE

    class _WithRepos(_Profile):
        repo_paths: tuple[str, ...] = ("/a",)
        repo_names: tuple[str, ...] = ("a",)

    assert repo_inference_strength(cast(SessionProfile, _WithRepos())) is ConfidenceBand.STRONG
