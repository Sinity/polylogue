"""Hand-written archive-tier enumerations agree with the canonical tier set.

polylogue-b1vr1: ``ARCHIVE_TIER_SPECS`` owns tier membership and
``archive_layout.ARCHIVE_TIER_ORDER`` derives it, but several surfaces still
spell the tier set out by hand. They agree today and nothing asserted it, so a
seventh tier would leave each of them silently short.

Anti-vacuity: adding a tier to ``ARCHIVE_TIER_SPECS`` (or dropping one from any
Literal / filename table below) makes this test red at one place instead of
degrading those surfaces quietly.
"""

from __future__ import annotations

from typing import get_args

from polylogue.daemon import status as daemon_status
from polylogue.storage import archive_identity
from polylogue.storage.archive_layout import ARCHIVE_TIER_ORDER


def test_archive_identity_tier_names_match_canonical_order() -> None:
    assert get_args(archive_identity.ArchiveTierName) == ARCHIVE_TIER_ORDER
    assert tuple(name for name, _ in archive_identity.TIER_FILENAMES) == ARCHIVE_TIER_ORDER
    assert tuple(filename for _, filename in archive_identity.TIER_FILENAMES) == tuple(
        f"{name}.db" for name in ARCHIVE_TIER_ORDER
    )


def test_daemon_status_tier_names_match_canonical_membership() -> None:
    # Display order is a surface concern; membership is not.
    assert set(get_args(daemon_status.ArchiveTierName)) == set(ARCHIVE_TIER_ORDER)
