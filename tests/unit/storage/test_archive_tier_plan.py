from __future__ import annotations

import sqlite3
from pathlib import Path

from polylogue.storage.sqlite.archive_tiers.archive_plan import ArchiveInitAction, build_archive_init_plan
from polylogue.storage.sqlite.archive_tiers.bootstrap import ARCHIVE_TIER_SPECS
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

    assert plan.ready is True
    assert {tier_plan.tier: tier_plan.action for tier_plan in plan.tiers} == {
        tier: (ArchiveInitAction.REPLACE_WITH_BACKUP if spec.backup_required else ArchiveInitAction.RECREATE_DISPOSABLE)
        for tier, spec in ARCHIVE_TIER_SPECS.items()
    }


def test_archive_plan_creates_targets_when_targets_are_absent(tmp_path: Path) -> None:
    plan = build_archive_init_plan(archive_root=tmp_path)

    assert plan.ready is True
    assert plan.blockers == ()
    assert {tier_plan.action for tier_plan in plan.tiers} == {ArchiveInitAction.CREATE}
