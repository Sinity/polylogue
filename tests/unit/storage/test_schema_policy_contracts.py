"""Contract suite pinning the current index-tier schema policy.

These tests pin the **fresh-first** schema policy:

- ``INDEX_SCHEMA_VERSION`` is the index-tier authority exposed through
  ``storage.sqlite.schema`` for sync/async bootstrap.
- On open, the on-disk ``PRAGMA user_version`` is compared against the
  constant.
- Version match → normal operation.
- Version mismatch → the database is *rejected*. There is no automatic
  in-place upgrade. The operator moves the mismatched tier aside and
  re-ingests/rebuilds from source.

The corresponding doc section is
``docs/internals.md`` § "Schema Versioning Model".

We also pin the FTS-trigger canonical set that fresh index init must produce:
``messages_fts_a{i,d,u}``. There is no separate actions FTS table.
"""

from __future__ import annotations

import asyncio
import sqlite3
from pathlib import Path

import pytest

from polylogue.errors import SchemaVersionMismatchError
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root
from polylogue.storage.sqlite.schema import (
    SCHEMA_VERSION,
    _ensure_schema,
    assert_readable_archive_layout,
    ensure_schema_async,
)
from polylogue.storage.sqlite.schema_bootstrap import (
    capture_schema_snapshot,
    decide_schema_bootstrap,
    schema_version_mismatch_message,
)

# ---------------------------------------------------------------------------
# Canonical FTS triggers — see docs/internals.md
# § "Daemon Convergence Evidence" (fts_trigger_state)
# ---------------------------------------------------------------------------

_CANONICAL_FTS_TRIGGERS = frozenset(
    {
        "messages_fts_ai",
        "messages_fts_ad",
        "messages_fts_au",
        "session_work_events_fts_ai",
        "session_work_events_fts_ad",
        "session_work_events_fts_au",
        "threads_fts_ai",
        "threads_fts_ad",
        "threads_fts_au",
        # ohbx: blocks_command_trigram is a narrower-purpose substring-lookup
        # index (not a message-search freshness surface tracked by
        # fts_trigger_state), but it's still a real FTS-backing trigger set
        # that belongs in this exhaustive schema inventory.
        "blocks_command_trigram_ai",
        "blocks_command_trigram_ad",
        "blocks_command_trigram_au",
    }
)


def test_fts_freshness_state_has_one_production_ddl_owner() -> None:
    """The index tier owns the freshness ledger shape; lifecycle code reuses it."""

    storage_root = Path(__file__).parents[3] / "polylogue" / "storage"
    create_sites: list[str] = []
    needle = "CREATE TABLE IF NOT EXISTS fts_freshness_state"
    for path in storage_root.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        if needle in text:
            create_sites.append(str(path.relative_to(storage_root)))

    assert create_sites == ["sqlite/archive_tiers/index.py"]


# ---------------------------------------------------------------------------
# § Schema Versioning Model — fresh-first; mismatch is rejected.
# ---------------------------------------------------------------------------


def _planted_db(tmp_path: Path, *, planted_version: int) -> Path:
    """Plant a SQLite file whose ``user_version`` is non-zero and not the
    canonical :data:`SCHEMA_VERSION`. ``decide_schema_bootstrap`` must
    classify it as ``version_mismatch`` rather than ``create_fresh``.
    """
    db_path = tmp_path / f"planted-v{planted_version}.db"
    conn = sqlite3.connect(db_path)
    conn.executescript(
        """
        CREATE TABLE raw_sessions (
            raw_id TEXT PRIMARY KEY,
            source_name TEXT NOT NULL DEFAULT '',
            source_path TEXT NOT NULL DEFAULT '',
            blob_size INTEGER NOT NULL DEFAULT 0,
            acquired_at TEXT NOT NULL DEFAULT ''
        );
        CREATE TABLE sessions (
            session_id TEXT PRIMARY KEY
        );
        CREATE TABLE messages (
            message_id TEXT PRIMARY KEY,
            session_id TEXT NOT NULL
        );
        """
    )
    conn.execute(f"PRAGMA user_version = {planted_version}")
    conn.commit()
    conn.close()
    return db_path


def test_fresh_database_initialises_to_current_version(tmp_path: Path) -> None:
    """docs/internals.md § Schema Versioning Model:
    ``SCHEMA_VERSION`` is the authority. A brand-new empty database
    bootstraps to that exact version.
    """
    db_path = tmp_path / "fresh.db"
    conn = sqlite3.connect(db_path)
    _ensure_schema(conn)
    version = conn.execute("PRAGMA user_version").fetchone()[0]
    freshness_table = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='fts_freshness_state'"
    ).fetchone()
    block_indexes = {row[1] for row in conn.execute("PRAGMA index_list(blocks)")}
    conn.close()
    assert version == SCHEMA_VERSION
    assert freshness_table is not None
    assert "idx_blocks_search_text_populated" in block_indexes


def test_matching_version_database_opens_cleanly(tmp_path: Path) -> None:
    """docs/internals.md § Schema Versioning Model: a database whose
    ``user_version`` already equals ``SCHEMA_VERSION`` must open
    without raising — version match is normal operation.
    """
    db_path = tmp_path / "matching.db"
    conn = sqlite3.connect(db_path)
    _ensure_schema(conn)
    # Re-open the same DB through the bootstrap path; this must be a
    # no-op rather than an error.
    _ensure_schema(conn)
    assert conn.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION
    conn.close()


def test_matching_version_database_ensures_runtime_indexes(tmp_path: Path) -> None:
    """Runtime index extensions are safe on existing same-version archives.

    These indexes are performance guards, not schema-version migrations: a
    current archive can gain them without rebuilding from source.
    """
    db_path = tmp_path / "runtime-indexes.db"
    conn = sqlite3.connect(db_path)
    try:
        _ensure_schema(conn)
        for index_name in (
            "idx_session_events_source_message",
            "idx_session_agent_policies_source_message",
            "idx_session_provider_usage_events_source_message",
            "idx_messages_message_type",
            "idx_messages_material_origin",
        ):
            conn.execute(f"DROP INDEX {index_name}")
        _ensure_schema(conn)
        for table, index_name in (
            ("session_events", "idx_session_events_source_message"),
            ("session_agent_policies", "idx_session_agent_policies_source_message"),
            ("session_provider_usage_events", "idx_session_provider_usage_events_source_message"),
            ("messages", "idx_messages_message_type"),
            ("messages", "idx_messages_material_origin"),
        ):
            assert any(row[1] == index_name for row in conn.execute(f"PRAGMA index_list({table})"))
    finally:
        conn.close()


def test_read_only_archive_open_ensures_runtime_indexes(tmp_path: Path) -> None:
    """Read surfaces should not wait for a later write to gain runtime indexes."""
    initialize_active_archive_root(tmp_path)
    index_db = tmp_path / "index.db"
    conn = sqlite3.connect(index_db)
    try:
        conn.execute("DROP INDEX idx_messages_message_type")
        conn.execute("DROP INDEX idx_messages_material_origin")
        conn.commit()
    finally:
        conn.close()

    with ArchiveStore.open_existing(tmp_path) as archive:
        assert archive._read_only is True

    conn = sqlite3.connect(index_db)
    try:
        for table, index_name in (
            ("messages", "idx_messages_message_type"),
            ("messages", "idx_messages_material_origin"),
        ):
            assert any(row[1] == index_name for row in conn.execute(f"PRAGMA index_list({table})"))
    finally:
        conn.close()


def test_read_only_archive_open_does_not_bootstrap_missing_tiers(tmp_path: Path) -> None:
    """Read/status surfaces must not create an empty archive as a side effect."""
    with pytest.raises(sqlite3.OperationalError):
        ArchiveStore.open_existing(tmp_path, read_only=True).close()

    assert not any(tmp_path.glob("*.db"))


def test_future_schema_version_is_rejected(tmp_path: Path) -> None:
    """docs/internals.md § Schema Versioning Model: a DB whose version
    is *newer* than this runtime understands must be rejected.
    Half-running against a forward-versioned DB risks data loss; the
    operator is expected to upgrade the runtime instead.
    """
    db_path = _planted_db(tmp_path, planted_version=SCHEMA_VERSION + 1)
    conn = sqlite3.connect(db_path)
    try:
        with pytest.raises(SchemaVersionMismatchError) as excinfo:
            _ensure_schema(conn)
        assert excinfo.value.current_version == SCHEMA_VERSION + 1
        assert excinfo.value.expected_version == SCHEMA_VERSION
    finally:
        conn.close()


def test_unknown_older_schema_version_is_rejected(tmp_path: Path) -> None:
    """docs/internals.md § Schema Versioning Model: a DB whose version
    is not the canonical version must be rejected. Polylogue has no
    in-place upgrade path — the operator re-ingests from source.
    """
    # Use any non-canonical, non-zero version. Version 17 is just an
    # non-canonical shape for this policy check.
    db_path = _planted_db(tmp_path, planted_version=17)
    conn = sqlite3.connect(db_path)
    try:
        with pytest.raises(SchemaVersionMismatchError) as excinfo:
            _ensure_schema(conn)
        assert excinfo.value.current_version == 17
        assert excinfo.value.expected_version == SCHEMA_VERSION
    finally:
        conn.close()


def test_version_mismatch_message_distinguishes_newer_and_older() -> None:
    """docs/internals.md § Schema Versioning Model: the rejection
    diagnostic must be specific enough that the operator can act on
    it — a newer DB needs a runtime upgrade, an older one needs a
    rebuild from source.
    """
    newer = schema_version_mismatch_message(SCHEMA_VERSION + 1)
    older = schema_version_mismatch_message(SCHEMA_VERSION - 1)
    assert str(SCHEMA_VERSION) in newer
    assert str(SCHEMA_VERSION) in older
    assert newer != older
    assert "newer" in newer.lower()


def test_decision_for_unknown_version_is_explicit_mismatch(tmp_path: Path) -> None:
    """docs/internals.md § Schema Versioning Model: the bootstrap
    decision for an unknown version is the ``version_mismatch``
    action — never silently falling through to apply current
    extensions.

    This pins the policy decision rather than the side effect.
    """
    db_path = _planted_db(tmp_path, planted_version=SCHEMA_VERSION + 5)
    conn = sqlite3.connect(db_path)
    try:
        snapshot = capture_schema_snapshot(conn)
        decision = decide_schema_bootstrap(snapshot)
        assert decision.action == "version_mismatch"
        assert decision.current_version == SCHEMA_VERSION + 5
    finally:
        conn.close()


def test_assert_readable_archive_layout_also_rejects_mismatch(tmp_path: Path) -> None:
    """docs/internals.md § Schema Versioning Model: the read-only
    open path applies the same fresh-first rejection. A read tool
    must not silently operate against a archive-version archive.
    """
    db_path = _planted_db(tmp_path, planted_version=SCHEMA_VERSION + 2)
    conn = sqlite3.connect(db_path)
    try:
        with pytest.raises(SchemaVersionMismatchError):
            assert_readable_archive_layout(conn)
    finally:
        conn.close()


def test_async_path_rejects_unknown_version(tmp_path: Path) -> None:
    """docs/internals.md § Schema Versioning Model: the async
    bootstrap path enforces the same policy as the sync path.
    Polylogue's primary runtime is async; a policy that only fires
    on the sync path would be a hole.
    """
    db_path = _planted_db(tmp_path, planted_version=17)

    async def _run() -> None:
        import aiosqlite

        async with aiosqlite.connect(str(db_path)) as conn:
            with pytest.raises(SchemaVersionMismatchError):
                await ensure_schema_async(conn)

    asyncio.run(_run())


# ---------------------------------------------------------------------------
# § FTS trigger canonical set — docs/internals.md
# § "Daemon Convergence Evidence" (fts_trigger_state)
# ---------------------------------------------------------------------------


def test_fresh_init_creates_canonical_fts_trigger_set(tmp_path: Path) -> None:
    """Fresh index initialization creates the current message FTS triggers."""
    db_path = tmp_path / "fts.db"
    conn = sqlite3.connect(db_path)
    _ensure_schema(conn)

    rows = conn.execute("SELECT name FROM sqlite_master WHERE type='trigger'").fetchall()
    conn.close()

    triggers = {row[0] for row in rows}
    missing = _CANONICAL_FTS_TRIGGERS - triggers
    assert not missing, f"Fresh init is missing canonical FTS triggers: {sorted(missing)}"
    assert triggers == _CANONICAL_FTS_TRIGGERS
