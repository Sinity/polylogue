"""Derived-tier identity stamps refuse stale rebuildable state."""

import importlib.util
import shutil
import sqlite3
import sys
from pathlib import Path
from types import ModuleType

import aiosqlite
import pytest

from polylogue.core.errors import SchemaVersionMismatchError
from polylogue.storage.sqlite import schema_manifest
from polylogue.storage.sqlite.archive_tiers import ARCHIVE_VERSION_BY_TIER
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from polylogue.storage.sqlite.archive_tiers.bootstrap import (
    initialize_active_archive_root,
    initialize_archive_database,
    initialize_archive_tier,
)
from polylogue.storage.sqlite.archive_tiers.index import INDEX_DDL
from polylogue.storage.sqlite.archive_tiers.ops import OPS_DDL
from polylogue.storage.sqlite.archive_tiers.schema_identity import (
    DERIVED_SCHEMA_META_DDL,
    DerivedTier,
    derived_schema_identity,
    read_schema_identity,
)
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from polylogue.storage.sqlite.schema import _ensure_schema, ensure_schema_async
from polylogue.storage.sqlite.schema_bootstrap import SchemaSkew, stamp_derived_schema_identity
from polylogue.storage.sqlite.schema_manifest import SchemaManifest, assert_schema_manifest


def _row_and_public_identities() -> tuple[str, str, tuple[str, ...]]:
    """Return stable row and public identities for the unchanged input fixture."""
    from polylogue.archive.message.roles import Role
    from polylogue.core.enums import Provider
    from polylogue.pipeline.ids import message_content_identity, session_id
    from polylogue.sources.origin_specs import public_origin_tokens
    from polylogue.sources.parsers.base_models import ParsedMessage

    return (
        str(session_id(Provider.CODEX, "schema-identity-fixture")),
        message_content_identity(ParsedMessage(provider_message_id="message-1", role=Role.USER, text="hello")),
        public_origin_tokens(),
    )


def _load_ids_source(path: Path, module_name: str) -> ModuleType:
    """Load an isolated pipeline.ids source copy while keeping normal imports."""
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _manifest_for_ddl(ddl: str) -> SchemaManifest:
    """Render a small derived-tier schema through the production manifest path."""
    with sqlite3.connect(":memory:") as conn:
        conn.executescript(ddl)
        conn.executescript(DERIVED_SCHEMA_META_DDL)
        conn.execute(f"PRAGMA user_version = {ARCHIVE_VERSION_BY_TIER[ArchiveTier.OPS]}")
        return SchemaManifest.from_connection(conn, ArchiveTier.OPS)


def test_schema_manifest_preserves_sql_literal_case_and_whitespace() -> None:
    """Literal values are part of schema semantics, not formatting noise."""
    baseline = _manifest_for_ddl(
        """
        CREATE TABLE sample (
            value TEXT DEFAULT 'user' CHECK (value = 'a b')
        ) STRICT;
        """
    )
    different_case = _manifest_for_ddl(
        """
        CREATE TABLE sample (
            value TEXT DEFAULT 'USER' CHECK (value = 'a b')
        ) STRICT;
        """
    )
    different_literal_spacing = _manifest_for_ddl(
        """
        CREATE TABLE sample (
            value TEXT DEFAULT 'user' CHECK (value = 'a  b')
        ) STRICT;
        """
    )

    assert baseline.fingerprint != different_case.fingerprint
    assert baseline.fingerprint != different_literal_spacing.fingerprint


def test_schema_manifest_admission_rejects_literal_drift(monkeypatch: pytest.MonkeyPatch) -> None:
    """A derived file with only a quoted literal change is stale, not compatible."""
    import polylogue.storage.sqlite.archive_tiers as archive_tiers

    declared = "CREATE TABLE sample (value TEXT DEFAULT 'user') STRICT;"
    actual = "CREATE TABLE sample (value TEXT DEFAULT 'USER') STRICT;"
    monkeypatch.setitem(archive_tiers.ARCHIVE_DDL_BY_TIER, ArchiveTier.OPS, declared)
    schema_manifest._canonical_schema_manifest.cache_clear()
    try:
        with sqlite3.connect(":memory:") as conn:
            conn.executescript(actual)
            conn.executescript(DERIVED_SCHEMA_META_DDL)
            conn.execute(f"PRAGMA user_version = {ARCHIVE_VERSION_BY_TIER[ArchiveTier.OPS]}")
            with pytest.raises(SchemaVersionMismatchError, match="semantic manifest mismatch"):
                assert_schema_manifest(conn, ArchiveTier.OPS)
    finally:
        schema_manifest._canonical_schema_manifest.cache_clear()


def test_index_identity_changes_when_a_fingerprint_input_changes(monkeypatch: pytest.MonkeyPatch) -> None:
    before = derived_schema_identity(DerivedTier.INDEX)
    monkeypatch.setattr(
        "polylogue.sources.origin_specs.lowering_fingerprint",
        lambda: "mutated-lowering-input",
    )
    after = derived_schema_identity(DerivedTier.INDEX)
    assert after != before


def test_performance_only_closure_edit_moves_combined_identity_but_not_row_ids(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A behavior-neutral closure edit still requires derived-index reconvergence."""
    import polylogue.sources.origin_specs as origin_specs

    isolated_root = tmp_path / "source-tree"
    shutil.copytree("polylogue", isolated_root / "polylogue", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    isolated_ids = isolated_root / "polylogue/pipeline/ids.py"
    declared_paths = origin_specs._LOWERING_FINGERPRINT_PATHS
    assert "polylogue/pipeline/ids.py" in declared_paths
    monkeypatch.setattr(origin_specs, "_SOURCE_ROOT", isolated_root)
    assert declared_paths == origin_specs._LOWERING_FINGERPRINT_PATHS
    origin_specs._semantic_source_closure.cache_clear()
    origin_specs._local_import_paths.cache_clear()
    origin_specs._fingerprint_sources_cached.cache_clear()
    origin_specs._invalidate_source_signatures()

    original_ids = _load_ids_source(isolated_ids, "polylogue.pipeline._identity_fixture_ids_before")
    row_and_public_ids_before = _row_and_public_identities()
    from polylogue.archive.message.roles import Role
    from polylogue.sources.parsers.base_models import ParsedMessage

    message = ParsedMessage(provider_message_id="message-1", role=Role.USER, text="hello")
    session_id_before = original_ids.session_id("codex", "schema-identity-fixture")
    message_identity_before = original_ids.message_content_identity(message)
    before = derived_schema_identity(DerivedTier.INDEX)

    original_source = isolated_ids.read_text(encoding="utf-8")
    old_check = 'if source_text == "":'
    new_check = "if not source_text:"
    assert original_source.count(old_check) == 1
    isolated_ids.write_text(original_source.replace(old_check, new_check), encoding="utf-8")
    origin_specs._invalidate_source_signatures()

    optimized_ids = _load_ids_source(isolated_ids, "polylogue.pipeline._identity_fixture_ids_after")
    message_identity_after = optimized_ids.message_content_identity(message)
    row_and_public_ids_after = _row_and_public_identities()
    after = derived_schema_identity(DerivedTier.INDEX)

    assert optimized_ids.session_id("codex", "schema-identity-fixture") == session_id_before
    assert message_identity_after == message_identity_before
    assert row_and_public_ids_after == row_and_public_ids_before
    assert after != before

    path = tmp_path / "index.db"
    initialize_archive_database(path, ArchiveTier.INDEX)
    with sqlite3.connect(path) as conn:
        conn.execute("UPDATE schema_identity SET identity = ? WHERE tier = 'index'", (before,))
    with pytest.raises(SchemaSkew, match="stale derived tier"):
        initialize_archive_database(path, ArchiveTier.INDEX)


def test_semantic_recipe_input_edit_moves_combined_identity_without_row_id_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A real production-declared semantic input edit rejects a stamped index."""
    import polylogue.sources.origin_specs as origin_specs

    isolated_root = tmp_path / "source-tree"
    shutil.copytree("polylogue", isolated_root / "polylogue", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    recipe_path = isolated_root / "polylogue/sources/dispatch.py"
    declared_paths = origin_specs._LOWERING_FINGERPRINT_PATHS
    monkeypatch.setattr(origin_specs, "_SOURCE_ROOT", isolated_root)
    assert declared_paths == origin_specs._LOWERING_FINGERPRINT_PATHS
    origin_specs._semantic_source_closure.cache_clear()
    origin_specs._local_import_paths.cache_clear()
    origin_specs._fingerprint_sources_cached.cache_clear()
    origin_specs._invalidate_source_signatures()

    before = derived_schema_identity(DerivedTier.INDEX)
    path = tmp_path / "index.db"
    with sqlite3.connect(path) as conn:
        initialize_archive_tier(conn, ArchiveTier.INDEX)
        conn.execute(
            "INSERT INTO sessions (native_id, origin, title, content_hash) VALUES (?, ?, ?, ?)",
            ("seed-session", "codex-session", "Seed", b"s" * 32),
        )
        session_id = str(conn.execute("SELECT session_id FROM sessions").fetchone()[0])
        conn.execute(
            """INSERT INTO messages (
                session_id, native_id, position, role, material_origin, content_hash
            ) VALUES (?, ?, ?, ?, ?, ?)""",
            (session_id, "seed-message", 0, "user", "human_authored", b"m" * 32),
        )
        message_id = str(conn.execute("SELECT message_id FROM messages").fetchone()[0])
        conn.execute(
            """INSERT INTO blocks (
                session_id, message_id, position, block_type, text, content_hash
            ) VALUES (?, ?, ?, ?, ?, ?)""",
            (session_id, message_id, 0, "text", "seed content", b"b" * 32),
        )
        stamp_derived_schema_identity(conn, "index")
        conn.commit()
    with sqlite3.connect(path) as conn:
        before_rows = conn.execute("SELECT rowid, tier, identity FROM schema_identity").fetchall()
        before_content = (
            conn.execute("SELECT session_id, native_id, title, content_hash FROM sessions").fetchall(),
            conn.execute("SELECT session_id, message_id, native_id, content_hash FROM messages").fetchall(),
            conn.execute("SELECT block_id, message_id, position, text, content_hash FROM blocks").fetchall(),
        )

    source = recipe_path.read_text(encoding="utf-8")
    assert source.count("_MAX_PARSE_DEPTH = 10") == 1
    recipe_path.write_text(source.replace("_MAX_PARSE_DEPTH = 10", "_MAX_PARSE_DEPTH = 11"), encoding="utf-8")
    origin_specs._invalidate_source_signatures()
    after = derived_schema_identity(DerivedTier.INDEX)

    assert after != before
    with pytest.raises(SchemaSkew, match="stale derived tier"):
        initialize_archive_database(path, ArchiveTier.INDEX)
    with sqlite3.connect(path) as conn:
        assert conn.execute("SELECT rowid, tier, identity FROM schema_identity").fetchall() == before_rows
        assert (
            conn.execute("SELECT session_id, native_id, title, content_hash FROM sessions").fetchall(),
            conn.execute("SELECT session_id, message_id, native_id, content_hash FROM messages").fetchall(),
            conn.execute("SELECT block_id, message_id, position, text, content_hash FROM blocks").fetchall(),
        ) == before_content


def test_index_identity_uses_semantic_manifest_not_ddl_comments(monkeypatch: pytest.MonkeyPatch) -> None:
    """Maintenance comments in the declared script do not force a new identity.

    Anti-vacuity: hashing ``INDEX_DDL`` directly makes the prefixed comment
    alter the identity.  The canonical manifest still renders the full
    production schema, so this exercises the same route used by bootstrap.
    """
    import polylogue.storage.sqlite.archive_tiers as archive_tiers

    original = archive_tiers.ARCHIVE_DDL_BY_TIER[ArchiveTier.INDEX]
    before = derived_schema_identity(DerivedTier.INDEX)
    monkeypatch.setitem(archive_tiers.ARCHIVE_DDL_BY_TIER, ArchiveTier.INDEX, f"-- packaging note\n{original}")
    schema_manifest._canonical_schema_manifest.cache_clear()
    try:
        assert derived_schema_identity(DerivedTier.INDEX) == before
    finally:
        schema_manifest._canonical_schema_manifest.cache_clear()


def test_stamped_wrong_identity_is_refused_before_index_use(tmp_path: Path) -> None:
    path = tmp_path / "index.db"
    initialize_archive_database(path, ArchiveTier.INDEX)
    with sqlite3.connect(path) as conn:
        conn.execute("UPDATE schema_identity SET identity = 'wrong' WHERE tier = 'index'")

    with pytest.raises(SchemaSkew, match="stale derived tier"):
        initialize_archive_database(path, ArchiveTier.INDEX)


def test_fresh_index_carries_current_identity(tmp_path: Path) -> None:
    path = tmp_path / "index.db"
    initialize_archive_database(path, ArchiveTier.INDEX)
    with sqlite3.connect(path) as conn:
        assert read_schema_identity(conn, DerivedTier.INDEX) == derived_schema_identity(DerivedTier.INDEX)


def test_derived_tier_ddl_declares_schema_identity() -> None:
    """Canonical tier scripts must include metadata used by bootstrap."""
    assert DERIVED_SCHEMA_META_DDL.strip() in INDEX_DDL
    assert DERIVED_SCHEMA_META_DDL.strip() in OPS_DDL


@pytest.mark.asyncio
async def test_async_fresh_index_carries_current_identity(tmp_path: Path) -> None:
    """Async fresh bootstrap must create the identity table before stamping."""
    path = tmp_path / "index.db"
    async with aiosqlite.connect(path) as conn:
        await ensure_schema_async(conn)
        cursor = await conn.execute("SELECT identity FROM schema_identity WHERE tier = ?", (DerivedTier.INDEX.value,))
        row = await cursor.fetchone()
    assert row is not None
    assert row[0] == derived_schema_identity(DerivedTier.INDEX)


def test_current_unstamped_derived_tier_is_adopted_before_identity_validation(tmp_path: Path) -> None:
    """Legacy current-version derived files gain identity metadata on reopen."""
    path = tmp_path / "index.db"
    with sqlite3.connect(path) as conn:
        initialize_archive_tier(conn, ArchiveTier.INDEX)
        conn.execute("DROP TABLE schema_identity")
        assert (
            conn.execute("SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'schema_identity'").fetchone()
            is None
        )

    initialize_archive_database(path, ArchiveTier.INDEX)

    with sqlite3.connect(path) as conn:
        assert read_schema_identity(conn, DerivedTier.INDEX) == derived_schema_identity(DerivedTier.INDEX)


def test_current_unstamped_ops_tier_is_adopted_before_identity_validation(tmp_path: Path) -> None:
    """The legacy adoption route applies to both rebuildable tiers."""
    path = tmp_path / "ops.db"
    with sqlite3.connect(path) as conn:
        initialize_archive_tier(conn, ArchiveTier.OPS)
        conn.execute("DROP TABLE schema_identity")
        assert (
            conn.execute("SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'schema_identity'").fetchone()
            is None
        )

    initialize_archive_database(path, ArchiveTier.OPS)

    with sqlite3.connect(path) as conn:
        assert read_schema_identity(conn, DerivedTier.OPS) == derived_schema_identity(DerivedTier.OPS)


def test_superseded_ops_identity_converges_to_the_current_schema(tmp_path: Path) -> None:
    """Disposable ops state is rebuilt in place instead of blocking startup."""
    path = tmp_path / "ops.db"
    with sqlite3.connect(path) as conn:
        initialize_archive_tier(conn, ArchiveTier.OPS)
        conn.execute("UPDATE schema_identity SET identity = 'from-another-runtime' WHERE tier = 'ops'")
        conn.execute("PRAGMA user_version = 1")
        conn.commit()

    initialize_archive_database(path, ArchiveTier.OPS)

    with sqlite3.connect(path) as conn:
        assert int(conn.execute("PRAGMA user_version").fetchone()[0]) == ARCHIVE_VERSION_BY_TIER[ArchiveTier.OPS]
        assert read_schema_identity(conn, DerivedTier.OPS) == derived_schema_identity(DerivedTier.OPS)


def test_canonical_sync_bootstrap_adopts_current_unstamped_index(tmp_path: Path) -> None:
    path = tmp_path / "index.db"
    with sqlite3.connect(path) as conn:
        initialize_archive_tier(conn, ArchiveTier.INDEX)
        conn.execute("DROP TABLE schema_identity")

    with sqlite3.connect(path) as conn:
        _ensure_schema(conn)
        assert read_schema_identity(conn, DerivedTier.INDEX) == derived_schema_identity(DerivedTier.INDEX)


@pytest.mark.asyncio
async def test_canonical_async_bootstrap_adopts_current_unstamped_index(tmp_path: Path) -> None:
    path = tmp_path / "index.db"
    with sqlite3.connect(path) as conn:
        initialize_archive_tier(conn, ArchiveTier.INDEX)
        conn.execute("DROP TABLE schema_identity")

    async with aiosqlite.connect(path) as conn:
        await ensure_schema_async(conn)
        cursor = await conn.execute("SELECT identity FROM schema_identity WHERE tier = ?", (DerivedTier.INDEX.value,))
        row = await cursor.fetchone()
    assert row is not None
    assert row[0] == derived_schema_identity(DerivedTier.INDEX)


def test_read_only_archive_open_refuses_stale_index_identity(tmp_path: Path) -> None:
    """A read-only archive must reject stale derived results before serving them."""
    archive_root = tmp_path / "archive"
    initialize_active_archive_root(archive_root)
    with sqlite3.connect(archive_root / "index.db") as conn:
        conn.execute("UPDATE schema_identity SET identity = 'wrong' WHERE tier = 'index'")

    with pytest.raises(SchemaSkew, match="stale derived tier"):
        ArchiveStore.open_existing(archive_root, read_only=True)
