from __future__ import annotations

import sqlite3
from pathlib import Path

from polylogue.storage.sqlite.archive_tiers.archive_plan import ArchiveInitAction, build_archive_init_plan
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
    assert {tier_plan.tier: tier_plan.action for tier_plan in plan.tiers} == {
        ArchiveTier.SOURCE: ArchiveInitAction.CREATE,
        ArchiveTier.INDEX: ArchiveInitAction.CREATE,
        ArchiveTier.EMBEDDINGS: ArchiveInitAction.CREATE,
        ArchiveTier.USER: ArchiveInitAction.CREATE,
        ArchiveTier.OPS: ArchiveInitAction.CREATE,
    }


def test_archive_plan_blocks_existing_targets_by_default(tmp_path: Path) -> None:
    _planted_db(tmp_path / "source.db", user_version=1)

    plan = build_archive_init_plan(archive_root=tmp_path)

    source_plan = next(tier_plan for tier_plan in plan.tiers if tier_plan.tier is ArchiveTier.SOURCE)
    assert plan.ready is False
    assert source_plan.action is ArchiveInitAction.BLOCKED
    assert source_plan.backup_path == tmp_path / "source.db.pre-archive-init.bak"
    assert any("source target already exists" in blocker for blocker in plan.blockers)


def test_archive_plan_classifies_replace_existing_by_durability(tmp_path: Path) -> None:
    _planted_db(tmp_path / "source.db", user_version=1)
    _planted_db(tmp_path / "index.db", user_version=1)
    _planted_db(tmp_path / "embeddings.db", user_version=1)
    _planted_db(tmp_path / "user.db", user_version=1)
    _planted_db(tmp_path / "ops.db", user_version=1)

    plan = build_archive_init_plan(
        archive_root=tmp_path,
        replace_existing=True,
    )

    assert plan.ready is True
    assert {tier_plan.tier: tier_plan.action for tier_plan in plan.tiers} == {
        ArchiveTier.SOURCE: ArchiveInitAction.REPLACE_WITH_BACKUP,
        ArchiveTier.INDEX: ArchiveInitAction.RECREATE_DISPOSABLE,
        ArchiveTier.EMBEDDINGS: ArchiveInitAction.REPLACE_WITH_BACKUP,
        ArchiveTier.USER: ArchiveInitAction.REPLACE_WITH_BACKUP,
        ArchiveTier.OPS: ArchiveInitAction.RECREATE_DISPOSABLE,
    }


def test_archive_plan_creates_targets_when_targets_are_absent(tmp_path: Path) -> None:
    plan = build_archive_init_plan(archive_root=tmp_path)

    assert plan.ready is True
    assert plan.blockers == ()
    assert {tier_plan.action for tier_plan in plan.tiers} == {ArchiveInitAction.CREATE}
