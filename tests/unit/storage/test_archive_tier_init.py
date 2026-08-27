from __future__ import annotations

import os
import sqlite3
import stat
from pathlib import Path
from unittest.mock import Mock

import pytest

from polylogue.storage.sqlite.archive_tiers import ARCHIVE_DDL_BY_TIER, archive_init
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


def test_all_six_tiers_reuse_immutable_prototypes(tmp_path: Path) -> None:
    """Every cache-safe tier pays fresh DDL once, including OPS and embeddings."""
    from polylogue.storage.sqlite.archive_tiers import bootstrap
    from polylogue.storage.sqlite.sqlite_vec_extension import try_load_sqlite_vec

    bootstrap.reset_archive_tier_init_counts()
    bootstrap._TIER_PROTOTYPES.clear()
    tiers = tuple(ArchiveTier)
    try:
        for index, tier in enumerate(tiers):
            first = sqlite3.connect(tmp_path / f"first-{index}.db")
            try:
                if tier is ArchiveTier.EMBEDDINGS:
                    loaded, error = try_load_sqlite_vec(first)
                    if not loaded:
                        pytest.skip(f"sqlite-vec extension is unavailable: {error}")
                bootstrap.initialize_archive_tier(first, tier)
            finally:
                first.close()

            second = sqlite3.connect(tmp_path / f"second-{index}.db")
            try:
                if tier is ArchiveTier.EMBEDDINGS:
                    loaded, error = try_load_sqlite_vec(second)
                    if not loaded:
                        pytest.skip(f"sqlite-vec extension is unavailable: {error}")
                bootstrap.initialize_archive_tier(second, tier)
            finally:
                second.close()
    finally:
        bootstrap._TIER_PROTOTYPES.clear()

    counts = bootstrap.archive_tier_init_counts()
    for tier in tiers:
        assert counts[f"{tier.value}.ddl_fresh"] == 1
        assert counts[f"{tier.value}.prototype_hit"] == 1


def test_tier_prototype_is_atomically_published_and_restored_read_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Prototype publication never exposes a writable cache file.

    Anti-vacuity: direct prototype copies or a normal SQLite open make this
    fail by omitting the same-directory replacement or the read-only URI.
    """
    from polylogue.storage.sqlite.archive_tiers import bootstrap

    bootstrap._TIER_PROTOTYPES.clear()
    try:
        replacements: list[tuple[Path, Path]] = []
        real_replace = os.replace

        def record_replace(source: str | Path, destination: str | Path) -> None:
            source_path = Path(source)
            destination_path = Path(destination)
            replacements.append((source_path, destination_path))
            real_replace(source_path, destination_path)

        monkeypatch.setattr(os, "replace", record_replace)
        bootstrap.initialize_archive_database(tmp_path / "first.db", ArchiveTier.INDEX)

        prototype = next(iter(bootstrap._TIER_PROTOTYPES.values()))
        assert prototype.stat().st_mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH) == 0
        assert len(replacements) == 1
        staging, published = replacements[0]
        assert published == prototype
        assert staging.parent == prototype.parent
        assert not staging.exists()

        connect_spy = Mock(wraps=sqlite3.connect)
        monkeypatch.setattr(sqlite3, "connect", connect_spy)
        bootstrap.initialize_archive_database(tmp_path / "second.db", ArchiveTier.INDEX)

        connect_spy.assert_any_call(prototype.resolve().as_uri() + "?mode=ro", uri=True)
    finally:
        bootstrap._TIER_PROTOTYPES.clear()


def test_tier_prototype_key_includes_rendered_ddl(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A same-version DDL variant must not receive a prototype from older DDL.

    Anti-vacuity: returning the cache key to ``(tier, version)`` makes the
    second initialization restore the first database by page copy, so the
    variant-only table is absent even though the requested DDL declares it.
    """
    from polylogue.storage.sqlite.archive_tiers import bootstrap

    bootstrap._TIER_PROTOTYPES.clear()
    tier = ArchiveTier.INDEX
    original_ddl = ARCHIVE_DDL_BY_TIER[tier]
    try:
        bootstrap.initialize_archive_database(tmp_path / "baseline.db", tier)
        monkeypatch.setitem(
            ARCHIVE_DDL_BY_TIER,
            tier,
            f"{original_ddl}\nCREATE TABLE prototype_variant (id INTEGER PRIMARY KEY) STRICT;",
        )

        variant_path = tmp_path / "variant.db"
        bootstrap.initialize_archive_database(variant_path, tier)

        with sqlite3.connect(variant_path) as conn:
            assert conn.execute("SELECT name FROM sqlite_master WHERE name = 'prototype_variant'").fetchone() == (
                "prototype_variant",
            )

        monkeypatch.setitem(ARCHIVE_DDL_BY_TIER, tier, original_ddl)
        original_path = tmp_path / "original.db"
        bootstrap.initialize_archive_database(original_path, tier)
        with sqlite3.connect(original_path) as conn:
            assert conn.execute("SELECT name FROM sqlite_master WHERE name = 'prototype_variant'").fetchone() is None
    finally:
        bootstrap._TIER_PROTOTYPES.clear()


def test_cached_index_prototype_replays_same_version_convergence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A page-copy hit must still apply newly registered benign DDL."""
    from polylogue.storage.sqlite.archive_tiers import bootstrap, index_convergence

    bootstrap.reset_archive_tier_init_counts()
    bootstrap._TIER_PROTOTYPES.clear()
    try:
        with sqlite3.connect(tmp_path / "first.db") as first:
            bootstrap.initialize_archive_tier(first, ArchiveTier.INDEX)

        entry = index_convergence.BenignDDLEntry(
            name="cached_prototype_test_table",
            sql="CREATE TABLE IF NOT EXISTS cached_prototype_test_table (id TEXT PRIMARY KEY) STRICT",
            reason="test-only same-version additive convergence",
        )
        monkeypatch.setattr(
            index_convergence,
            "INDEX_BENIGN_DDL_REGISTRY",
            (*index_convergence.INDEX_BENIGN_DDL_REGISTRY, entry),
        )
        with sqlite3.connect(tmp_path / "second.db") as second:
            bootstrap.initialize_archive_tier(second, ArchiveTier.INDEX)
            assert second.execute(
                "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'cached_prototype_test_table'"
            ).fetchone() == (1,)
    finally:
        bootstrap._TIER_PROTOTYPES.clear()

    counts = bootstrap.archive_tier_init_counts()
    assert counts["index.ddl_fresh"] == 1
    assert counts["index.prototype_hit"] == 1


def test_tier_init_counts_use_ops_convergence_after_schema_is_known(tmp_path: Path) -> None:
    """OPS performs one compatibility reapply, then uses cheap convergence.

    A database created before the sentinel exists still gets one full,
    idempotent DDL pass so additive same-version tables are not lost.  That
    pass records the applied DDL digest; later opens run only the existing
    convergence repairs. Anti-vacuity: removing the fallback would make a
    legacy OPS database silently miss additive tables, while removing the
    sentinel check would return to one whole DDL pass per open.
    """
    from polylogue.storage.sqlite.archive_tiers import bootstrap

    bootstrap.reset_archive_tier_init_counts()

    for _ in range(3):
        bootstrap.initialize_archive_database(tmp_path / "ops.db", ArchiveTier.OPS)
        bootstrap.initialize_archive_database(tmp_path / "index.db", ArchiveTier.INDEX)

    counts = bootstrap.archive_tier_init_counts()

    # The first call creates and fingerprints the tier; both subsequent opens
    # are convergence-only. Legacy files without the sentinel still use the
    # guarded fallback described above.
    assert "ops.ddl_reapply" not in counts
    assert counts["ops.schema_convergence"] == 2
    assert "index.ddl_reapply" not in counts


def test_embeddings_prototype_restore_keeps_sqlite_vec_ready(tmp_path: Path) -> None:
    """A cached embeddings prototype restores an operational vec0 schema."""
    from polylogue.storage.sqlite.archive_tiers import bootstrap
    from polylogue.storage.sqlite.sqlite_vec_extension import try_load_sqlite_vec

    bootstrap.reset_archive_tier_init_counts()
    bootstrap._TIER_PROTOTYPES.clear()
    try:
        first_path = tmp_path / "e1.db"
        second_path = tmp_path / "e2.db"
        first = sqlite3.connect(first_path)
        try:
            loaded, error = try_load_sqlite_vec(first)
            if not loaded:
                pytest.skip(f"sqlite-vec extension is unavailable: {error}")
            bootstrap.initialize_archive_tier(first, ArchiveTier.EMBEDDINGS)
        finally:
            first.close()

        second = sqlite3.connect(second_path)
        try:
            bootstrap.initialize_archive_tier(second, ArchiveTier.EMBEDDINGS)
            table = second.execute(
                "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'message_embeddings'"
            ).fetchone()
            assert table is not None
            assert "vec0" in str(table[0])
            assert second.execute("SELECT COUNT(*) FROM message_embeddings").fetchone() == (0,)
        finally:
            second.close()
    finally:
        bootstrap._TIER_PROTOTYPES.clear()

    counts = bootstrap.archive_tier_init_counts()
    assert counts["embeddings.ddl_fresh"] == 1
    assert counts["embeddings.prototype_hit"] == 1


def test_embeddings_init_fails_before_vec_ddl_when_extension_is_unavailable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Missing sqlite-vec must not be masked by vec0 DDL parsing.

    Anti-vacuity: removing the guard in ``_initialize_archive_tier_ddl`` makes
    initialization execute the ``+model`` vec0 declaration without the
    extension and raises SQLite's misleading ``near \"+\"`` syntax error.
    """
    from polylogue.storage.sqlite.archive_tiers import bootstrap

    bootstrap._TIER_PROTOTYPES.clear()

    def _fail_vec_load(conn: sqlite3.Connection) -> tuple[bool, Exception | None]:
        del conn
        return False, RuntimeError("simulated sqlite-vec load failure")

    monkeypatch.setattr(bootstrap, "try_load_sqlite_vec", _fail_vec_load)

    with sqlite3.connect(tmp_path / "embeddings.db") as conn:
        with pytest.raises(RuntimeError, match="archive embeddings initialization requires sqlite-vec"):
            bootstrap.initialize_archive_tier(conn, ArchiveTier.EMBEDDINGS)
        assert conn.execute("SELECT COUNT(*) FROM sqlite_master").fetchone() == (0,)


# ---------------------------------------------------------------------------
# Same-version reapply invariants (polylogue-c1jgh)
# ---------------------------------------------------------------------------


def _traced(conn: sqlite3.Connection) -> list[str]:
    """Record every statement SQLite actually executes on this connection."""
    statements: list[str] = []
    conn.set_trace_callback(statements.append)
    return statements


def test_same_version_reapply_does_not_rewrite_user_version(tmp_path: Path) -> None:
    """The header write, not the DDL, was the whole cost of a reapply.

    ``PRAGMA user_version = N`` rewrites the database header even when N is
    already the stored value, dirtying a page and turning an otherwise no-op
    schema pass into a full commit fsync. Measured on NVMe: 148.80ms with the
    unconditional write, 0.34ms without it, of which the ``executescript``
    itself is 0.33ms.

    Anti-vacuity: restoring the unconditional
    ``conn.execute(f"PRAGMA user_version = {spec.version}")`` fails this
    immediately, and that is the line that made every same-version reapply
    fsync.
    """
    from polylogue.storage.sqlite.archive_tiers import bootstrap

    db = tmp_path / "ops.db"
    bootstrap.initialize_archive_database(db, ArchiveTier.OPS)

    conn = sqlite3.connect(db)
    try:
        statements = _traced(conn)
        bootstrap.initialize_archive_tier(conn, ArchiveTier.OPS)
    finally:
        conn.close()

    assignments = [s for s in statements if "user_version" in s.lower() and "=" in s]
    assert assignments == [], f"same-version reapply still wrote the header: {assignments}"
    # The read that decides it must still happen, or nothing is doing the deciding.
    assert any("user_version" in s.lower() for s in statements)


def test_fresh_database_still_records_its_schema_version(tmp_path: Path) -> None:
    """The fresh path is unchanged: a new tier must end at its declared version."""
    from polylogue.storage.sqlite.archive_tiers import bootstrap

    db = tmp_path / "ops.db"
    bootstrap.initialize_archive_database(db, ArchiveTier.OPS)

    with sqlite3.connect(db) as conn:
        stored = int(conn.execute("PRAGMA user_version").fetchone()[0])
    assert stored == ARCHIVE_TIER_SPECS[ArchiveTier.OPS].version
    assert stored != 0


def test_same_version_reapply_still_repairs_a_stale_drift_samples_check(tmp_path: Path) -> None:
    """LOAD-BEARING RED TWIN: the same-version convergence must still run.

    ``_ensure_schema_drift_samples_check`` repairs an ops.db bootstrapped with
    a CHECK naming only three of ``DriftClassification``'s four values, and it
    does so at an UNCHANGED ``user_version`` (#3451 / polylogue-u6tl). Any fix
    for reapply cost that short-circuits on version equality -- skipping the
    convergence helpers rather than only the redundant header write -- silently
    disables this repair, and an archive keeps raising ``IntegrityError`` on
    every ``known_field_unread`` insert forever.

    This test fails under that mistaken fix and passes under the header-write
    guard, which is precisely the distinction that makes the guard safe.
    """
    from polylogue.storage.sqlite.archive_tiers import bootstrap

    db = tmp_path / "ops.db"
    bootstrap.initialize_archive_database(db, ArchiveTier.OPS)

    stale = (
        "CREATE TABLE schema_drift_samples ("
        " sample_id TEXT PRIMARY KEY,"
        " origin TEXT NOT NULL,"
        " element_kind TEXT NOT NULL,"
        " classification TEXT NOT NULL CHECK (classification IN "
        "('unknown_field','type_mismatch','missing_field')),"
        " unseen_key_signature TEXT NOT NULL DEFAULT '',"
        " native_id_example TEXT NOT NULL,"
        " raw_id TEXT NOT NULL,"
        " observed_at_ms INTEGER NOT NULL"
        ") STRICT"
    )
    with sqlite3.connect(db) as conn:
        conn.execute("DROP TABLE schema_drift_samples")
        conn.execute(stale)
        conn.commit()
        assert "known_field_unread" not in _drift_samples_sql(conn)

    bootstrap.initialize_archive_database(db, ArchiveTier.OPS)

    with sqlite3.connect(db) as conn:
        assert "known_field_unread" in _drift_samples_sql(conn)
        # Prove the repair is real, not cosmetic: the value now inserts.
        conn.execute(
            "INSERT INTO schema_drift_samples VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            ("s1", "codex-session", "field", "known_field_unread", "", "n1", "r1", 0),
        )
        conn.commit()


def _drift_samples_sql(conn: sqlite3.Connection) -> str:
    row = conn.execute(
        "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'schema_drift_samples'"
    ).fetchone()
    return str(row[0]) if row and row[0] else ""
