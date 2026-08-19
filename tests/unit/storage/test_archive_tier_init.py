from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from polylogue.storage.sqlite.archive_tiers import archive_init
from polylogue.storage.sqlite.archive_tiers.archive_init import (
    ArchiveInitBlockedError,
    initialize_archive_tier_files,
)
from polylogue.storage.sqlite.archive_tiers.bootstrap import ARCHIVE_TIER_SPECS
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier


def _fake_initialize_archive_database(path: Path, tier: object) -> None:
    conn = sqlite3.connect(path)
    try:
        conn.execute("CREATE TABLE initialized (tier TEXT PRIMARY KEY) STRICT")
        conn.execute("INSERT INTO initialized VALUES (?)", (str(tier),))
        conn.execute("PRAGMA user_version = 1")
        conn.commit()
    finally:
        conn.close()


def test_initialize_archive_tier_files_creates_all_tiers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "stray.sqlite").write_text("unrelated file", encoding="utf-8")
    monkeypatch.setattr(archive_init, "initialize_archive_database", _fake_initialize_archive_database)

    result = initialize_archive_tier_files(archive_root=tmp_path)

    assert not (tmp_path / "stray.sqlite.retired.bak").exists()
    assert {tier.path.name for tier in result.tier_results} == {spec.filename for spec in ARCHIVE_TIER_SPECS.values()}
    for name in (spec.filename for spec in ARCHIVE_TIER_SPECS.values()):
        conn = sqlite3.connect(tmp_path / name)
        try:
            assert conn.execute("SELECT COUNT(*) FROM initialized").fetchone()[0] == 1
        finally:
            conn.close()


def test_initialize_archive_tier_files_backs_up_replaceable_targets(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for spec in ARCHIVE_TIER_SPECS.values():
        (tmp_path / spec.filename).write_text(f"existing {spec.tier.value} target", encoding="utf-8")
    monkeypatch.setattr(archive_init, "initialize_archive_database", _fake_initialize_archive_database)

    result = initialize_archive_tier_files(
        archive_root=tmp_path,
        replace_existing=True,
    )

    assert (tmp_path / "source.db.pre-archive-init.bak").read_text(encoding="utf-8") == "existing source target"
    assert (tmp_path / "embeddings.db.pre-archive-init.bak").read_text(encoding="utf-8") == "existing embeddings target"
    assert (tmp_path / "user.db.pre-archive-init.bak").read_text(encoding="utf-8") == "existing user target"
    assert (tmp_path / "audit.db.pre-archive-init.bak").read_text(encoding="utf-8") == "existing audit target"
    assert not (tmp_path / "index.db.pre-archive-init.bak").exists()
    assert not (tmp_path / "ops.db.pre-archive-init.bak").exists()
    assert {tier.initialized for tier in result.tier_results} == {True}


def test_initialize_archive_tier_files_refuses_blocked_plan(tmp_path: Path) -> None:
    (tmp_path / "source.db").write_text("existing source target", encoding="utf-8")

    with pytest.raises(ArchiveInitBlockedError, match="source target already exists"):
        initialize_archive_tier_files(archive_root=tmp_path)


# ---------------------------------------------------------------------------
# Tier-initialization telemetry (polylogue-l218h / WS-A bootstrap sizing)
# ---------------------------------------------------------------------------


def test_tier_init_counts_separate_page_copy_from_fresh_ddl(tmp_path: Path) -> None:
    """A page-copy restore and a real DDL execution must not share a counter.

    Both look identical to a caller -- "initialize a tier" -- while the
    restore is microseconds and the DDL is an fsync-bound ``executescript``.
    Anti-vacuity: collapsing the two outcomes into one counter makes this
    fail, and collapsing them is exactly what makes the cost invisible in the
    receipt.
    """
    from polylogue.storage.sqlite.archive_tiers import bootstrap

    bootstrap.reset_archive_tier_init_counts()
    bootstrap._TIER_PROTOTYPES.clear()

    # 1st: empty database, no prototype yet -> real DDL, and it seeds the cache.
    bootstrap.initialize_archive_database(tmp_path / "a.db", ArchiveTier.INDEX)
    # 2nd: a different empty database -> restored by page copy.
    bootstrap.initialize_archive_database(tmp_path / "b.db", ArchiveTier.INDEX)

    counts = bootstrap.archive_tier_init_counts()

    assert counts["index.ddl_fresh"] == 1
    assert counts["index.prototype_hit"] == 1


def test_tier_init_counts_expose_the_ops_only_whole_schema_reapply(tmp_path: Path) -> None:
    """Only ops.db re-executes its WHOLE schema on a same-version open.

    ``initialize_archive_database`` treats an existing same-version tier three
    different ways: INDEX gets its targeted benign-DDL convergence, USER gets
    its annotation-schema ensure, and OPS alone routes back through
    ``initialize_archive_tier`` -- which, finding a non-empty database, runs
    the entire tier DDL again for its ``IF NOT EXISTS`` idempotence. Every one
    of those is a redundant executescript plus its commit fsync, and this
    counter is what makes the asymmetry measurable rather than a code-reading
    exercise.
    """
    from polylogue.storage.sqlite.archive_tiers import bootstrap

    bootstrap.reset_archive_tier_init_counts()

    for _ in range(3):
        bootstrap.initialize_archive_database(tmp_path / "ops.db", ArchiveTier.OPS)
        bootstrap.initialize_archive_database(tmp_path / "index.db", ArchiveTier.INDEX)

    counts = bootstrap.archive_tier_init_counts()

    # First call creates each tier; the two that follow are same-version opens.
    assert counts["ops.ddl_reapply"] == 2
    assert "index.ddl_reapply" not in counts


def test_tier_init_counts_classify_the_uncached_embeddings_tier(tmp_path: Path) -> None:
    """EMBEDDINGS is excluded from the prototype cache, so a fresh archive pays DDL.

    Each new archive gets a real DDL execution rather than a page copy; a
    same-version reopen returns without work, like every tier except OPS.
    """
    from polylogue.storage.sqlite.archive_tiers import bootstrap

    bootstrap.reset_archive_tier_init_counts()

    bootstrap.initialize_archive_database(tmp_path / "e1.db", ArchiveTier.EMBEDDINGS)
    bootstrap.initialize_archive_database(tmp_path / "e2.db", ArchiveTier.EMBEDDINGS)
    bootstrap.initialize_archive_database(tmp_path / "e1.db", ArchiveTier.EMBEDDINGS)

    counts = bootstrap.archive_tier_init_counts()

    assert counts["embeddings.ddl_fresh"] == 2
    assert not any(key.startswith("embeddings.prototype_hit") for key in counts)
