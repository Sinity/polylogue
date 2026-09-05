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

from polylogue.core.enums import BlockType, Provider, Role
from polylogue.core.errors import DatabaseError, SchemaVersionMismatchError
from polylogue.sources.parsers.base_models import ParsedContentBlock, ParsedMessage, ParsedSession
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
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
from polylogue.storage.sqlite.schema_manifest import (
    MESSAGE_FTS_DEGRADABLE_OBJECTS,
    canonical_schema_manifest,
    schema_manifest_diff,
    schema_manifest_diff_is_message_fts_only,
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


def test_existing_archive_repairs_runtime_indexes_before_manifest_validation(tmp_path: Path) -> None:
    """A same-version archive with a dropped runtime index remains restartable.

    The dropped index is the anti-vacuity mutation: without the writable
    bootstrap repair, canonical manifest validation fails before startup can
    finish.
    """
    initialize_active_archive_root(tmp_path)
    index_db = tmp_path / "index.db"
    with sqlite3.connect(index_db) as conn:
        conn.execute("DROP INDEX idx_messages_message_type")
        conn.commit()

    initialize_active_archive_root(tmp_path)

    with sqlite3.connect(index_db) as conn:
        assert any(row[1] == "idx_messages_message_type" for row in conn.execute("PRAGMA index_list(messages)"))


def test_same_version_trigram_trigger_variants_are_accepted() -> None:
    """The v63 lifecycle rule preserves ungated triggers until rebuild."""
    expected = canonical_schema_manifest(ArchiveTier.INDEX)
    variant_names = {
        "blocks_command_trigram_ai",
        "blocks_command_trigram_ad",
        "blocks_command_trigram_au",
    }
    guards = (
        " and not exists (select 1 from derived_refresh_guard where guard_name = 'fts_bulk_session_write')",
        " when not exists (select 1 from derived_refresh_guard where guard_name = 'fts_bulk_session_write')",
    )
    variant_objects = tuple(
        (kind, name, sql.replace(guards[0], "").replace(guards[1], "")) if name in variant_names else (kind, name, sql)
        for kind, name, sql in expected.objects
    )
    actual = expected.__class__(expected.tier, expected.version, variant_objects, "")
    assert schema_manifest_diff(expected, actual)["wrong_definition"] == []

    unrelated_objects = tuple(
        (kind, name, sql + " changed") if name == "blocks_command_trigram_ai" else (kind, name, sql)
        for kind, name, sql in expected.objects
    )
    unrelated = expected.__class__(expected.tier, expected.version, unrelated_objects, "")
    assert schema_manifest_diff(expected, unrelated)["wrong_definition"]


def test_read_only_archive_open_rejects_runtime_index_drift(tmp_path: Path) -> None:
    """Read surfaces reject an existing index with missing runtime indexes."""
    initialize_active_archive_root(tmp_path)
    index_db = tmp_path / "index.db"
    conn = sqlite3.connect(index_db)
    try:
        conn.execute("DROP INDEX idx_messages_message_type")
        conn.execute("DROP INDEX idx_messages_material_origin")
        conn.commit()
    finally:
        conn.close()

    with pytest.raises(SchemaVersionMismatchError):
        ArchiveStore.open_existing(tmp_path).close()

    conn = sqlite3.connect(index_db)
    try:
        for table, index_name in (
            ("messages", "idx_messages_message_type"),
            ("messages", "idx_messages_material_origin"),
        ):
            assert not any(row[1] == index_name for row in conn.execute(f"PRAGMA index_list({table})"))
    finally:
        conn.close()


def test_pinned_read_only_archive_open_rejects_drifted_physical_index_after_promotion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A promoted active pointer cannot make a pinned read repair old evidence.

    The caller resolves the old physical index, then an ownership transition
    promotes a different generation before the production ``open_existing``
    initialization path runs. The path must not reach the write-time runtime
    index helper or mutate the now-inactive physical generation.
    """
    old_root = tmp_path / "old-generation"
    new_root = tmp_path / "new-generation"
    archive_root = tmp_path / "archive"
    initialize_active_archive_root(old_root)
    initialize_active_archive_root(new_root)
    archive_root.mkdir()
    old_index = (old_root / "index.db").resolve()
    new_index = (new_root / "index.db").resolve()
    with sqlite3.connect(old_index) as conn:
        conn.execute("CREATE TABLE pinned_evidence (value TEXT NOT NULL)")
        conn.execute("INSERT INTO pinned_evidence VALUES ('old physical index')")
        conn.execute("DROP INDEX idx_messages_message_type")
        conn.commit()
        assert not any(row[1] == "idx_messages_message_type" for row in conn.execute("PRAGMA index_list(messages)"))

    active_index = archive_root / "index.db"
    active_index.symlink_to(old_index)
    pinned_index = active_index.resolve(strict=True)
    active_index.unlink()
    active_index.symlink_to(new_index)

    def fail_if_write_time_indexes_run(_conn: sqlite3.Connection) -> None:
        pytest.fail("pinned read entered the write-time runtime-index DDL path")

    monkeypatch.setattr(
        "polylogue.storage.sqlite.archive_tiers.archive.ensure_runtime_indexes_sync",
        fail_if_write_time_indexes_run,
    )
    with pytest.raises(SchemaVersionMismatchError):
        ArchiveStore.open_existing(archive_root, index_path=pinned_index).close()

    with sqlite3.connect(old_index) as conn:
        assert not any(row[1] == "idx_messages_message_type" for row in conn.execute("PRAGMA index_list(messages)"))


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


def test_every_prior_index_schema_version_is_rejected_not_silently_reopened(tmp_path: Path) -> None:
    """polylogue-f2qv.5 regression: every already-deployed archive right now
    is stamped at ``SCHEMA_VERSION - 1`` (the immediately prior version).
    ``CREATE TABLE IF NOT EXISTS`` is a no-op against an already-existing
    table, so a DDL edit alone (e.g. widening a CHECK constraint) does
    *not* retroactively apply to those archives — only the version bump
    forces them through ``version_mismatch`` rejection and the documented
    fresh-first rebuild (``polylogue ops reset --index && polylogued run``)
    instead of being silently reopened with stale DDL that a subsequent
    write could violate (see commit that added
    ``INDEX_SCHEMA_VERSION`` specifically so this scenario is caught here,
    not as a runtime CHECK-constraint failure deep in convergence).
    """
    db_path = _planted_db(tmp_path, planted_version=SCHEMA_VERSION - 1)
    conn = sqlite3.connect(db_path)
    try:
        with pytest.raises(SchemaVersionMismatchError) as excinfo:
            _ensure_schema(conn)
        assert excinfo.value.current_version == SCHEMA_VERSION - 1
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
    assert "Reset the derived index and let `polylogued run` rebuild it from source." in older


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


def test_readable_layout_rejects_shape_drift_with_typed_refusal(tmp_path: Path) -> None:
    """A matching user_version does not authorize a generation with an older shape."""
    db_path = _planted_db(tmp_path, planted_version=SCHEMA_VERSION)
    conn = sqlite3.connect(db_path)
    try:
        with pytest.raises(SchemaVersionMismatchError) as caught:
            assert_readable_archive_layout(conn, generation_id="gen-old")
        error = caught.value
        assert error.current_version == SCHEMA_VERSION
        assert error.expected_version == SCHEMA_VERSION
        assert error.generation_id == "gen-old"
        assert error.lifecycle_action == "rebuild_index"
        assert "gen-old" in str(error)
        assert "rebuild" in str(error).lower()
    finally:
        conn.close()


def test_readable_layout_rejects_extra_object_with_typed_refusal(tmp_path: Path) -> None:
    """A matching user_version does not authorize undeclared schema objects."""
    db_path = tmp_path / "extra-object.db"
    conn = sqlite3.connect(db_path)
    _ensure_schema(conn)
    conn.execute("CREATE TABLE undeclared_read_only_drift (value TEXT)")
    conn.commit()
    try:
        with pytest.raises(SchemaVersionMismatchError) as caught:
            assert_readable_archive_layout(conn, generation_id="gen-extra")
        error = caught.value
        assert error.current_version == SCHEMA_VERSION
        assert error.expected_version == SCHEMA_VERSION
        assert error.generation_id == "gen-extra"
        assert error.lifecycle_action == "rebuild_index"
        assert "undeclared_read_only_drift" in str(error)
        assert "rebuild" in str(error).lower()
    finally:
        conn.close()


def test_read_only_archive_open_rejects_semantic_manifest_drift(tmp_path: Path) -> None:
    """The production read-only route validates the complete manifest.

    An extra table is the anti-vacuity mutation: the legacy sessions-column
    sentinel still passes, while the archive contains an object outside the
    current schema contract.
    """
    initialize_active_archive_root(tmp_path)
    with sqlite3.connect(tmp_path / "index.db") as conn:
        conn.execute("CREATE TABLE unexpected_read_object (value TEXT NOT NULL)")
        conn.commit()

    with pytest.raises(SchemaVersionMismatchError) as caught:
        ArchiveStore.open_existing(tmp_path, read_only=True)
    error = caught.value
    assert error.current_version == SCHEMA_VERSION
    assert error.expected_version == SCHEMA_VERSION
    assert error.lifecycle_action == "rebuild_index"
    assert "schema semantic manifest mismatch" in str(error)


def test_readable_layout_admits_a_missing_message_fts_surface(tmp_path: Path) -> None:
    """A degraded search index is a reported read state, not a closed archive.

    ``messages_fts`` is contentless, trigger-maintained, and rebuildable from
    ``blocks``. Losing it degrades search; every other read of the index stays
    valid, and the search route reports the degradation as route state.

    Anti-vacuity: widening the admission to any manifest diff makes
    ``test_readable_layout_rejects_extra_object_with_typed_refusal`` red, and
    dropping a non-FTS table here still raises.
    """
    db_path = tmp_path / "degraded-fts.db"
    conn = sqlite3.connect(db_path)
    _ensure_schema(conn)
    for trigger in ("messages_fts_ai", "messages_fts_ad", "messages_fts_au"):
        conn.execute(f"DROP TRIGGER {trigger}")
    conn.execute("DROP TABLE messages_fts")
    conn.commit()
    try:
        assert_readable_archive_layout(conn, generation_id="gen-degraded-fts")

        conn.execute("DROP TABLE session_links")
        conn.commit()
        with pytest.raises(SchemaVersionMismatchError, match="session_links"):
            assert_readable_archive_layout(conn, generation_id="gen-degraded-fts")
    finally:
        conn.close()


def test_readable_layout_rejects_current_version_with_cost_usage_shape_drift(tmp_path: Path) -> None:
    """The read guard checks the complete canonical index schema.

    Dropping both cost columns is the anti-vacuity mutation: the old
    ``sessions.reported_cost_usd`` sentinel remains present, while coverage
    reads still fail when they select the usage columns.
    """
    db_path = tmp_path / "cost-usage-drift.db"
    conn = sqlite3.connect(db_path)
    _ensure_schema(conn)
    conn.executescript(
        """
        ALTER TABLE session_model_usage RENAME TO session_model_usage_current;
        CREATE TABLE session_model_usage (
            session_id TEXT NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,
            model_name TEXT NOT NULL,
            input_tokens INTEGER NOT NULL DEFAULT 0 CHECK(input_tokens >= 0),
            output_tokens INTEGER NOT NULL DEFAULT 0 CHECK(output_tokens >= 0),
            cache_read_tokens INTEGER NOT NULL DEFAULT 0 CHECK(cache_read_tokens >= 0),
            cache_write_tokens INTEGER NOT NULL DEFAULT 0 CHECK(cache_write_tokens >= 0),
            message_count INTEGER NOT NULL DEFAULT 0 CHECK(message_count >= 0),
            cost_credits REAL,
            PRIMARY KEY(session_id, model_name)
        ) STRICT;
        DROP TABLE session_model_usage_current;
        """
    )
    conn.commit()
    try:
        with pytest.raises(SchemaVersionMismatchError) as caught:
            assert_readable_archive_layout(conn, generation_id="gen-cost-drift")
        error = caught.value
        assert error.current_version == SCHEMA_VERSION
        assert error.expected_version == SCHEMA_VERSION
        assert error.generation_id == "gen-cost-drift"
        assert error.lifecycle_action == "rebuild_index"
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
    """Fresh index initialization creates exactly the current FTS-backing triggers.

    Fresh init also creates a larger family of *non*-FTS triggers —
    ``query_unit_frame_*`` (cache-invalidation epoch bump on
    ``query_unit_frame_state``) and the ``blocks_action_pairs_*`` /
    ``session_links_delegation_facts_*`` / ``session_profiles_delegation_facts_*``
    family (materializing the plain ``action_pairs``/``delegation_facts``
    tables). Neither writes into an FTS5 virtual table, so per the ohbx
    precedent above (a trigger belongs in this set only if it is genuinely
    *FTS-backing*, not merely "any trigger fresh init happens to create"),
    they are excluded here by construction rather than by an ever-growing
    name list.
    """
    db_path = tmp_path / "fts.db"
    conn = sqlite3.connect(db_path)
    _ensure_schema(conn)

    fts5_tables = {
        row[0]
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND sql LIKE '%USING fts5%'"
        ).fetchall()
    }
    assert fts5_tables, "expected at least one FTS5 virtual table in fresh init"

    trigger_rows = conn.execute("SELECT name, sql FROM sqlite_master WHERE type='trigger'").fetchall()
    conn.close()

    triggers = {
        name
        for name, sql in trigger_rows
        if any(f"INSERT INTO {fts_table}" in sql or f"DELETE FROM {fts_table}" in sql for fts_table in fts5_tables)
    }
    missing = _CANONICAL_FTS_TRIGGERS - triggers
    assert not missing, f"Fresh init is missing canonical FTS triggers: {sorted(missing)}"
    assert triggers == _CANONICAL_FTS_TRIGGERS


def test_message_fts_admission_is_a_declared_object_set() -> None:
    """The degradable surface is declared, not inferred from an object-name prefix.

    ``messages_fts_identity`` is a plain declared table of ours that happens to
    share the ``messages_fts`` prefix; a future surface could too. Admission
    names the objects it admits so a new one is refused until it is declared.

    Anti-vacuity: matching on the ``messages_fts`` name prefix admits the
    undeclared object below and turns this red.
    """
    diff = {
        "missing": [("table", "messages_fts_speculative_surface")],
        "extra": [],
        "wrong_definition": [],
    }
    assert not schema_manifest_diff_is_message_fts_only(diff)

    declared = {
        ("table", "messages_fts"),
        ("table", "messages_fts_identity"),
        ("trigger", "messages_fts_ai"),
        ("trigger", "messages_fts_ad"),
        ("trigger", "messages_fts_au"),
    }
    assert set(MESSAGE_FTS_DEGRADABLE_OBJECTS) == declared
    canonical = {(kind, name) for kind, name, _ in canonical_schema_manifest(ArchiveTier.INDEX).objects}
    assert declared <= canonical


def test_admitted_missing_message_fts_refuses_block_search_with_a_typed_error(tmp_path: Path) -> None:
    """Every reader of the admitted-absent surface degrades with a typed error.

    ``assert_readable_archive_layout`` admits an index whose message FTS
    surface is gone on the promise that reads report it as search-route state.
    That promise holds only if no reader of ``messages_fts`` reaches SQL.

    Anti-vacuity: removing the existence check from ``search_archive_blocks``
    raises ``sqlite3.OperationalError: no such table: messages_fts`` instead,
    which is not a ``DatabaseError``.
    """
    root = tmp_path / "archive"
    session = ParsedSession(
        source_name=Provider.CODEX,
        provider_session_id="fts-degraded-1",
        messages=[
            ParsedMessage(
                provider_message_id="m1",
                role=Role.USER,
                blocks=[ParsedContentBlock(type=BlockType.TEXT, text="one searchable needle")],
            )
        ],
    )
    with ArchiveStore(root) as writer:
        writer.write_parsed(session)

    with sqlite3.connect(root / "index.db") as conn:
        for trigger in ("messages_fts_ai", "messages_fts_ad", "messages_fts_au"):
            conn.execute(f"DROP TRIGGER {trigger}")
        conn.execute("DROP TABLE messages_fts")
        conn.commit()

    with ArchiveStore.open_existing(root, read_only=True) as store:
        with pytest.raises(DatabaseError, match="Search index"):
            store.search_blocks("needle")
