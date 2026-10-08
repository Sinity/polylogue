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

from polylogue.storage.sqlite.archive_tiers import (
    ARCHIVE_FORMAT_FLOOR_VERSION,
    ARCHIVE_VERSION_BY_TIER,
    archive_plan,
)
from polylogue.storage.sqlite.archive_tiers import bootstrap as tier_bootstrap
from polylogue.storage.sqlite.archive_tiers.bootstrap import (
    ARCHIVE_TIER_SPECS,
    initialize_active_archive_root,
)
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier

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
    initialize_active_archive_root(tmp_path)

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


def test_runtime_target_without_a_declared_train_refuses_after_baseline_birth(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An untrained runtime target cannot invent that version's fresh schema."""
    from polylogue.core.errors import SchemaSkew

    advanced = dict(ARCHIVE_VERSION_BY_TIER)
    advanced[ArchiveTier.USER] += 1
    monkeypatch.setattr(tier_bootstrap, "ARCHIVE_VERSION_BY_TIER", advanced)
    with pytest.raises(SchemaSkew) as refused:
        initialize_active_archive_root(tmp_path)
    assert refused.value.tier == "user"
    marker = json.loads((tmp_path / ".polylogue-format.json").read_text())
    assert set(marker["tier_versions"].values()) == {1}
    with sqlite3.connect(tmp_path / "source.db") as conn:
        assert conn.execute("PRAGMA user_version").fetchone()[0] == ARCHIVE_VERSION_BY_TIER[ArchiveTier.SOURCE]


def test_transplanted_baseline_tier_is_refused_by_its_immutable_birth_fingerprint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tests.infra.durable_tier_fixtures import bootstrap_baseline_archive

    bootstrap_baseline_archive(tmp_path, monkeypatch)
    archive_plan.assert_archive_format_lineage(tmp_path)
    source_path = tmp_path / "source.db"
    source_path.unlink()
    with sqlite3.connect(source_path) as foreign:
        foreign.execute("CREATE TABLE foreign_lineage (id INTEGER PRIMARY KEY) STRICT")
        foreign.execute("PRAGMA user_version = 1")
    with pytest.raises(RuntimeError, match="is not part of polylogue.archive-format.v6"):
        archive_plan.assert_archive_format_lineage(tmp_path)
