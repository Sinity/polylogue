from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from polylogue.storage.sqlite.archive_tiers import ARCHIVE_FORMAT_FLOOR_VERSION, ARCHIVE_VERSION_BY_TIER
from polylogue.storage.sqlite.archive_tiers.archive_plan import (
    ARCHIVE_FORMAT_LINEAGE,
    ArchiveInitAction,
    build_archive_init_plan,
)
from polylogue.storage.sqlite.archive_tiers.bootstrap import ARCHIVE_TIER_SPECS, initialize_active_archive_root
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier


def _planted_db(path: Path, *, user_version: int) -> None:
    conn = sqlite3.connect(path)
    try:
        conn.execute("CREATE TABLE planted (id INTEGER PRIMARY KEY) STRICT")
        conn.execute(f"PRAGMA user_version = {user_version}")
        conn.commit()
    finally:
        conn.close()


def test_archive_plan_creates_absent_tier_targets(tmp_path: Path) -> None:
    (tmp_path / "stray.sqlite").write_text("unrelated file", encoding="utf-8")

    plan = build_archive_init_plan(archive_root=tmp_path)

    assert plan.ready is True
    assert plan.blockers == ()
    assert {tier_plan.tier: tier_plan.action for tier_plan in plan.tiers} == dict.fromkeys(
        ARCHIVE_TIER_SPECS, ArchiveInitAction.CREATE
    )


def test_archive_plan_blocks_existing_targets_by_default(tmp_path: Path) -> None:
    _planted_db(tmp_path / "source.db", user_version=1)

    plan = build_archive_init_plan(archive_root=tmp_path)

    source_plan = next(tier_plan for tier_plan in plan.tiers if tier_plan.tier is ArchiveTier.SOURCE)
    assert plan.ready is False
    assert source_plan.action is ArchiveInitAction.BLOCKED
    assert source_plan.backup_path == tmp_path / "source.db.pre-archive-init.bak"
    assert any("source target already exists" in blocker for blocker in plan.blockers)


def test_archive_plan_classifies_replace_existing_by_durability(tmp_path: Path) -> None:
    for spec in ARCHIVE_TIER_SPECS.values():
        _planted_db(tmp_path / spec.filename, user_version=1)

    plan = build_archive_init_plan(
        archive_root=tmp_path,
        replace_existing=True,
    )

    assert plan.ready is False
    assert {tier_plan.tier: tier_plan.action for tier_plan in plan.tiers} == {
        tier: (
            ArchiveInitAction.BLOCKED
            if tier in {ArchiveTier.SOURCE, ArchiveTier.USER, ArchiveTier.AUDIT}
            else ArchiveInitAction.REPLACE_WITH_BACKUP
            if spec.backup_required
            else ArchiveInitAction.RECREATE_DISPOSABLE
        )
        for tier, spec in ARCHIVE_TIER_SPECS.items()
    }
    assert all("fresh format floor cannot replace durable evidence" in blocker for blocker in plan.blockers)


def test_archive_plan_creates_targets_when_targets_are_absent(tmp_path: Path) -> None:
    plan = build_archive_init_plan(archive_root=tmp_path)

    assert plan.ready is True
    assert plan.blockers == ()
    assert {tier_plan.action for tier_plan in plan.tiers} == {ArchiveInitAction.CREATE}


def test_marker_refuses_a_foreign_lineage_tier(tmp_path: Path) -> None:
    """A stamped version needs format evidence, not a coincidentally equal integer.

    The transplant carries the *birth* version this archive's marker recorded,
    because that is the only integer a foreign file can borrow to look current.
    A tier standing at any other version is already refused by the ordinary
    version comparison, so pinning this to the literal ``1`` would stop
    exercising the marker the moment a numbered migration raised the durable
    target above the floor.

    Anti-vacuity: binding the fingerprint check to ``ARCHIVE_FORMAT_FLOOR_VERSION``
    instead of the recorded birth version admits this file, and ``user.db``
    is then rewritten by a bootstrap that should never have started.
    """
    initialize_active_archive_root(tmp_path)

    marker = json.loads((tmp_path / ".polylogue-format.json").read_text(encoding="utf-8"))
    assert marker["format"] == ARCHIVE_FORMAT_LINEAGE
    assert marker["floor_version"] == ARCHIVE_FORMAT_FLOOR_VERSION
    assert marker["tier_versions"] == {tier.value: ARCHIVE_VERSION_BY_TIER[tier] for tier in ArchiveTier}

    birth_version = ARCHIVE_VERSION_BY_TIER[ArchiveTier.SOURCE]
    user_path = tmp_path / "user.db"
    user_before = user_path.read_bytes()
    source_path = tmp_path / "source.db"
    source_path.unlink()
    with sqlite3.connect(source_path) as historical:
        historical.execute("CREATE TABLE historical_lineage (id INTEGER PRIMARY KEY) STRICT")
        historical.execute(f"PRAGMA user_version = {birth_version}")

    with pytest.raises(RuntimeError, match=f"historical version-{birth_version} schema"):
        initialize_active_archive_root(tmp_path)

    assert user_path.read_bytes() == user_before
