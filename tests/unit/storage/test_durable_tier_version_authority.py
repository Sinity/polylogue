"""One version per durable tier: what it stamps is what its readers compare against.

Every durable tier used to carry two numbers. ``ARCHIVE_VERSION_BY_TIER``
stamped the archive format floor while ``source.py`` still declared
``SOURCE_SCHEMA_VERSION = 47``, ``user.py`` declared ``11`` and ``audit.py``
declared ``3`` -- the pre-reset lineage numbers whose migration chains
``#5275``/``#5290`` deleted. Nothing wrote those numbers any more, but
``blob_integrity`` still read one of them as an authority threshold, so a fresh
source tier that *was* current compared as not-current forever.

A test that reads only one of the two numbers cannot see that split: it agrees
with itself. These tests deliberately read the writer and the reader and
compare them.
"""

from __future__ import annotations

import importlib
import json
import sqlite3
from pathlib import Path

import pytest

from polylogue.storage import blob_integrity
from polylogue.storage.sqlite.archive_tiers import (
    ARCHIVE_FORMAT_FLOOR_VERSION,
    ARCHIVE_VERSION_BY_TIER,
    archive_plan,
)
from polylogue.storage.sqlite.archive_tiers import bootstrap as tier_bootstrap
from polylogue.storage.sqlite.archive_tiers.bootstrap import (
    ARCHIVE_TIER_SPECS,
    initialize_active_archive_root,
    initialize_archive_database,
)
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from polylogue.storage.sqlite.migration_runner import MigrationError, migrate_archive_tier

DURABLE_TIERS = (ArchiveTier.SOURCE, ArchiveTier.USER, ArchiveTier.AUDIT)

#: The modules that declare each durable tier's DDL. A version constant living
#: here is a second answer to a question ``ARCHIVE_VERSION_BY_TIER`` already
#: answers.
DURABLE_TIER_MODULES = {
    ArchiveTier.SOURCE: "polylogue.storage.sqlite.archive_tiers.source",
    ArchiveTier.USER: "polylogue.storage.sqlite.archive_tiers.user",
    ArchiveTier.AUDIT: "polylogue.storage.sqlite.archive_tiers.audit",
}


@pytest.mark.parametrize("tier", DURABLE_TIERS, ids=lambda tier: tier.value)
def test_durable_tier_stamps_the_version_its_readers_compare_against(tmp_path: Path, tier: ArchiveTier) -> None:
    """Bootstrap writes ``ARCHIVE_VERSION_BY_TIER``; every reader is given the same map.

    Anti-vacuity: stamping the tier from any other number -- for instance the
    retired per-module constant -- leaves ``PRAGMA user_version`` disagreeing
    with the expectation the tier spec and the readiness probe both publish.
    """
    path = tmp_path / ARCHIVE_TIER_SPECS[tier].filename
    initialize_archive_database(path, tier)

    with sqlite3.connect(path) as conn:
        stamped = int(conn.execute("PRAGMA user_version").fetchone()[0])

    assert stamped == ARCHIVE_VERSION_BY_TIER[tier]
    assert ARCHIVE_TIER_SPECS[tier].version == ARCHIVE_VERSION_BY_TIER[tier]
    assert stamped >= ARCHIVE_FORMAT_FLOOR_VERSION


@pytest.mark.parametrize("tier", DURABLE_TIERS, ids=lambda tier: tier.value)
def test_durable_tier_module_declares_no_second_version(tier: ArchiveTier) -> None:
    """A durable tier module must not publish a version of its own.

    Anti-vacuity: re-adding ``SOURCE_SCHEMA_VERSION = 47`` to ``source.py``
    (or ``USER_SCHEMA_VERSION = 11``/``AUDIT_SCHEMA_VERSION = 3``) turns this
    red, naming the constant and both numbers. A declaration that merely
    repeats ``ARCHIVE_VERSION_BY_TIER`` stays green, because a duplicate that
    cannot disagree is not the failure this guards.
    """
    module = importlib.import_module(DURABLE_TIER_MODULES[tier])
    authority = ARCHIVE_VERSION_BY_TIER[tier]

    declared = {
        name: getattr(module, name)
        for name in dir(module)
        if name.endswith("_SCHEMA_VERSION") and isinstance(getattr(module, name), int)
    }

    disagreeing = {name: value for name, value in declared.items() if value != authority}
    assert not disagreeing, (
        f"{DURABLE_TIER_MODULES[tier]} declares {disagreeing}, but the {tier.value} tier "
        f"stamps and compares {authority}; two numbers for one tier is the split "
        f"that made blob_integrity read a threshold nothing writes"
    )


def test_blob_reference_authority_reads_the_stamped_source_version(tmp_path: Path) -> None:
    """The version clause alone must accept a source tier at the stamped version.

    The fixture deliberately carries *no* current blob-ref catalog, so
    ``current_capabilities`` is False and the version comparison is the only
    thing that can grant authority. That isolates the reader's threshold from
    the catalog probe beside it.

    Anti-vacuity: restoring the comparison against a retired per-module
    constant makes ``current_authority`` False and the kind ``legacy`` here,
    because no source tier this runtime writes ever reaches that number.
    """
    path = tmp_path / "source.db"
    stamped = ARCHIVE_VERSION_BY_TIER[ArchiveTier.SOURCE]
    with sqlite3.connect(path) as conn:
        conn.execute(f"PRAGMA user_version = {stamped}")

    with sqlite3.connect(path) as conn:
        capabilities = blob_integrity._source_schema_capabilities(conn)

    assert capabilities.user_version == stamped
    # No current blob-ref catalog and no legacy carrier: the integer is the
    # only evidence in play.
    assert capabilities.current_blob_refs is False
    assert capabilities.legacy_carriers == ()
    assert capabilities.current_authority is True
    assert capabilities.kind == "current_versioned"


def test_blob_reference_authority_refuses_a_foreign_lineage_source_version(tmp_path: Path) -> None:
    """A version this runtime does not stamp earns nothing on its integer alone.

    ``user_version`` was renumbered from one by the archive format floor, so a
    pre-floor ``47`` is not "newer" than the current stamp -- it is a different
    lineage. Its legacy blob carriers must still be scanned.

    Anti-vacuity: comparing with ``>=`` instead of equality readmits every
    pre-floor stamp above the floor and drops the legacy carrier below.
    """
    path = tmp_path / "source.db"
    foreign = ARCHIVE_VERSION_BY_TIER[ArchiveTier.SOURCE] + 46
    with sqlite3.connect(path) as conn:
        conn.execute("CREATE TABLE raw_sessions (raw_id TEXT PRIMARY KEY, blob_hash BLOB NOT NULL)")
        conn.execute(f"PRAGMA user_version = {foreign}")

    with sqlite3.connect(path) as conn:
        capabilities = blob_integrity._source_schema_capabilities(conn)

    assert capabilities.current_authority is False
    assert "raw_sessions" in capabilities.legacy_carriers


def test_fresh_archive_born_above_the_floor_admits_its_own_format_marker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A durable tier target above the floor must not refuse a freshly written marker.

    The floor is the lower bound of a lineage that is meant to evolve forward by
    numbered migrations -- ``DURABLE_MIGRATION_ADOPTION_FLOORS`` starts a future
    train directly above it. The format marker records the durable version each
    tier was *born* at, so once a migration raises the runtime's durable target
    every newly bootstrapped archive is born above the floor.

    Anti-vacuity: pinning the marker's durable ``tier_versions`` to
    ``ARCHIVE_FORMAT_FLOOR_VERSION`` exactly, instead of treating it as a lower
    bound, makes this fresh archive refuse the marker it just wrote with
    "archive format marker has an incomplete six-tier floor".
    """
    advanced = dict(ARCHIVE_VERSION_BY_TIER)
    advanced[ArchiveTier.SOURCE] = ARCHIVE_VERSION_BY_TIER[ArchiveTier.SOURCE] + 1
    advanced[ArchiveTier.USER] = ARCHIVE_VERSION_BY_TIER[ArchiveTier.USER] + 1
    monkeypatch.setattr(tier_bootstrap, "ARCHIVE_VERSION_BY_TIER", advanced)
    monkeypatch.setattr(archive_plan, "ARCHIVE_VERSION_BY_TIER", advanced)

    initialize_active_archive_root(tmp_path)

    marker = json.loads((tmp_path / ".polylogue-format.json").read_text(encoding="utf-8"))
    assert marker["floor_version"] == ARCHIVE_FORMAT_FLOOR_VERSION
    assert marker["tier_versions"]["source"] == advanced[ArchiveTier.SOURCE]
    assert marker["tier_versions"]["user"] == advanced[ArchiveTier.USER]
    for tier in DURABLE_TIERS:
        with sqlite3.connect(tmp_path / ARCHIVE_TIER_SPECS[tier].filename) as conn:
            assert int(conn.execute("PRAGMA user_version").fetchone()[0]) == advanced[tier]

    archive_plan.assert_archive_format_lineage(tmp_path)


def test_a_transplanted_tier_at_the_birth_version_is_still_refused_above_the_floor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Birth-version fingerprint evidence must survive the lineage moving forward.

    The marker's ``durable_schema_fingerprints`` describe each tier at the
    version it was born at. Binding that check to the floor rather than to the
    recorded birth version would silently switch it off for every archive born
    above the floor, readmitting exactly the foreign-lineage file the marker
    exists to reject.

    Anti-vacuity: restoring ``version == ARCHIVE_FORMAT_FLOOR_VERSION`` as the
    fingerprint condition makes this transplanted tier pass admission.
    """
    advanced = dict(ARCHIVE_VERSION_BY_TIER)
    advanced[ArchiveTier.SOURCE] = ARCHIVE_VERSION_BY_TIER[ArchiveTier.SOURCE] + 1
    monkeypatch.setattr(tier_bootstrap, "ARCHIVE_VERSION_BY_TIER", advanced)
    monkeypatch.setattr(archive_plan, "ARCHIVE_VERSION_BY_TIER", advanced)

    initialize_active_archive_root(tmp_path)
    archive_plan.assert_archive_format_lineage(tmp_path)

    source_path = tmp_path / "source.db"
    source_path.unlink()
    with sqlite3.connect(source_path) as foreign:
        foreign.execute("CREATE TABLE foreign_lineage (id INTEGER PRIMARY KEY) STRICT")
        foreign.execute(f"PRAGMA user_version = {advanced[ArchiveTier.SOURCE]}")

    with pytest.raises(RuntimeError, match="is not part of polylogue.archive-format.v1"):
        archive_plan.assert_archive_format_lineage(tmp_path)


def test_audit_schema_change_is_rejected_until_its_numbered_route_is_authorized(tmp_path: Path) -> None:
    """A simulated audit change cannot bypass the missing v2 authority.

    Audit is durable, but its fresh schema still stamps the format floor.  A
    caller asking for the first numbered migration must therefore be rejected
    before any SQL or backup state is touched.  This is the anti-vacuity guard
    for the route record in ``migrations/audit``: adding an audit migration
    without first advancing the tier authority would otherwise create a
    migration file that production can never safely admit.
    """
    path = tmp_path / "audit.db"
    initialize_archive_database(path, ArchiveTier.AUDIT)
    with sqlite3.connect(path) as conn:
        with pytest.raises(MigrationError, match="newer than this runtime expects"):
            migrate_archive_tier(conn, ArchiveTier.AUDIT, backup_manifest=None, target_version=2)
        assert int(conn.execute("PRAGMA user_version").fetchone()[0]) == ARCHIVE_VERSION_BY_TIER[ArchiveTier.AUDIT]
