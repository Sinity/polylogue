"""Unit tests for the closed RawAuthorityVerdict projection (polylogue-w6hql, Phase 2)."""

from __future__ import annotations

from polylogue.archive.raw_authority_verdict import derive_raw_authority_verdict
from polylogue.archive.revision_authority import (
    HistoricalRawRevision,
    classify_historical_full_revisions,
)
from polylogue.core.enums import RawAuthorityVerdict


def test_sole_revision_is_sole_copy() -> None:
    decisions = classify_historical_full_revisions([HistoricalRawRevision("only", b"payload")])
    assert derive_raw_authority_verdict(decisions) == {"only": RawAuthorityVerdict.SOLE_COPY}


def test_proven_chain_head_is_verified_and_ancestors_are_superseded() -> None:
    decisions = classify_historical_full_revisions(
        [
            HistoricalRawRevision("oldest", b"one\n"),
            HistoricalRawRevision("middle", b"one\ntwo\n"),
            HistoricalRawRevision("newest", b"one\ntwo\nthree\n"),
        ]
    )
    verdicts = derive_raw_authority_verdict(decisions)
    assert verdicts == {
        "oldest": RawAuthorityVerdict.SUPERSEDED,
        "middle": RawAuthorityVerdict.SUPERSEDED,
        "newest": RawAuthorityVerdict.VERIFIED,
    }


def test_byte_equal_duplicate_of_the_head_is_superseded_not_verified_twice() -> None:
    """Anti-vacuity: a naive mapping that only looked at ``authority`` (both
    BYTE_PROVEN) would wrongly call both copies VERIFIED. ``relation ==
    'duplicate'`` must dominate that check."""
    decisions = classify_historical_full_revisions(
        [
            HistoricalRawRevision("raw-b", b"same-bytes"),
            HistoricalRawRevision("raw-a", b"same-bytes"),
        ]
    )
    verdicts = derive_raw_authority_verdict(decisions)
    # raw-a is the lexicographically-smallest representative (baseline, no
    # successor); with 2 total decisions this is not a sole-copy cohort.
    assert verdicts["raw-a"] == RawAuthorityVerdict.VERIFIED
    assert verdicts["raw-b"] == RawAuthorityVerdict.SUPERSEDED


def test_unprovable_fork_is_diverged_not_verified() -> None:
    """Anti-vacuity: a naive mapping keyed only on ``relation`` (both
    'ambiguous') without checking ``authority is QUARANTINED`` would need this
    case to prove the DIVERGED branch is actually reachable and correct."""
    decisions = classify_historical_full_revisions(
        [HistoricalRawRevision("left", b"left"), HistoricalRawRevision("right", b"right")]
    )
    verdicts = derive_raw_authority_verdict(decisions)
    assert verdicts == {
        "left": RawAuthorityVerdict.DIVERGED,
        "right": RawAuthorityVerdict.DIVERGED,
    }


def test_empty_cohort_derives_no_verdicts() -> None:
    assert derive_raw_authority_verdict([]) == {}
