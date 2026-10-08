from __future__ import annotations

import sqlite3
from builtins import BaseExceptionGroup
from collections.abc import Callable
from contextlib import closing
from pathlib import Path
from typing import Any

import pytest

from polylogue.core.errors import SchemaSkew
from polylogue.storage.sqlite import connection_profile
from polylogue.storage.sqlite.archive_tiers import ARCHIVE_VERSION_BY_TIER
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from polylogue.storage.sqlite.schema_bootstrap import stamp_derived_schema_identity


def _declared_profile(name: str) -> connection_profile.SQLiteConnectionProfile:
    """Resolve a named class the way a production caller does: read, then write."""
    return connection_profile.READ_PROFILES.get(name) or connection_profile.WRITE_PROFILES[name]


def test_declared_timeout_classes_cover_read_and_write_profiles() -> None:
    assert set(connection_profile.TIMEOUT_CLASSES) == {
        "interactive-read",
        "background-read",
        "publication",
        "offline-bulk",
        "active-cold-build",
    }
    assert all(profile.role == "read" and profile.query_only for profile in connection_profile.READ_PROFILES.values())


def test_writers_retain_bounded_autocheckpoint() -> None:
    assert connection_profile.WAL_AUTOCHECKPOINT_PAGES == 10000
    assert "PRAGMA wal_autocheckpoint = 10000" in connection_profile.WRITE_CONNECTION_PROFILE.pragma_statements
    assert "PRAGMA wal_autocheckpoint = 10000" in connection_profile.DAEMON_WRITE_CONNECTION_PROFILE.pragma_statements


def test_open_readonly_connection_uses_descriptor_bound_database(tmp_path: Path) -> None:
    # A tier-named file makes the factory assert that tier's declared version;
    # this test's subject is descriptor binding.
    db_path = tmp_path / "descriptor-bound.db"
    with sqlite3.connect(db_path) as connection:
        connection.execute(f"PRAGMA user_version = {ARCHIVE_VERSION_BY_TIER[ArchiveTier.INDEX]}")
        connection.execute("CREATE TABLE evidence (value TEXT)")
        connection.execute("INSERT INTO evidence VALUES ('selected')")

    descriptor_handle = db_path.open("rb")
    try:
        reader = connection_profile.open_readonly_connection(db_path, opened_main_fd=descriptor_handle.fileno())
        try:
            assert reader.execute("SELECT value FROM evidence").fetchone() == ("selected",)
        finally:
            reader.close()
    finally:
        descriptor_handle.close()


def test_open_readonly_connection_refuses_without_descriptor_bound_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db_path = tmp_path / "index.db"
    with sqlite3.connect(db_path) as connection:
        connection.execute("CREATE TABLE evidence (value TEXT)")

    descriptor_handle = db_path.open("rb")
    try:
        monkeypatch.setattr(connection_profile, "_descriptor_database_uri", lambda _fd, _suffix: None)
        with pytest.raises(RuntimeError, match="descriptor-bound path"):
            connection_profile.open_readonly_connection(db_path, opened_main_fd=descriptor_handle.fileno())
    finally:
        descriptor_handle.close()


def test_open_readonly_connection_rejects_immutable_with_descriptor(tmp_path: Path) -> None:
    db_path = tmp_path / "index.db"
    with sqlite3.connect(db_path) as connection:
        connection.execute("CREATE TABLE evidence (value TEXT)")

    descriptor_handle = db_path.open("rb")
    try:
        with pytest.raises(ValueError, match="immutable mode"):
            connection_profile.open_readonly_connection(
                db_path,
                immutable=True,
                opened_main_fd=descriptor_handle.fileno(),
            )
    finally:
        descriptor_handle.close()


@pytest.mark.parametrize(
    "statement",
    [
        "INSERT INTO evidence VALUES ('wrong')",
        "UPDATE evidence SET value = 'wrong'",
        "DELETE FROM evidence",
        "CREATE TABLE unwanted (value TEXT)",
        "PRAGMA query_only = OFF",
        "PRAGMA journal_mode = DELETE",
        "PRAGMA wal_checkpoint",
        "ATTACH DATABASE ':memory:' AS writable",
    ],
)
def test_profiled_reader_rejects_write_mutants_at_sqlite_boundary(tmp_path: Path, statement: str) -> None:
    db_path = tmp_path / "index.db"
    with sqlite3.connect(db_path) as writer:
        writer.execute("CREATE TABLE evidence (value TEXT)")
        writer.execute("INSERT INTO evidence VALUES ('original')")

    reader = connection_profile.open_readonly_connection(db_path, validate_schema=False)
    try:
        assert reader.execute("SELECT value FROM evidence").fetchone() == ("original",)
        assert reader.execute("PRAGMA table_info(evidence)").fetchall()
        with pytest.raises(sqlite3.DatabaseError):
            reader.execute(statement)
    finally:
        reader.close()


def test_readonly_temp_staging_cannot_write_persistent_or_attached_database(tmp_path: Path) -> None:
    db_path = tmp_path / "index.db"
    with sqlite3.connect(db_path) as writer:
        writer.execute("CREATE TABLE evidence (value TEXT)")
    reader = connection_profile.open_readonly_connection(db_path, validate_schema=False)
    try:
        with connection_profile.readonly_temp_staging(reader):
            reader.execute("CREATE TEMP TABLE projection (value TEXT)")
            reader.execute("INSERT INTO projection VALUES ('derived')")
            with pytest.raises(sqlite3.DatabaseError):
                reader.execute("INSERT INTO evidence VALUES ('wrong')")
            with pytest.raises(sqlite3.DatabaseError):
                reader.execute("ATTACH DATABASE ':memory:' AS writable")
        assert reader.execute("SELECT value FROM projection").fetchone() == ("derived",)
        assert reader.execute("PRAGMA query_only").fetchone() == (1,)
        with pytest.raises(sqlite3.DatabaseError):
            reader.execute("INSERT INTO projection VALUES ('wrong')")
    finally:
        reader.close()


def test_readonly_temp_staging_can_select_file_backing_before_projection(tmp_path: Path) -> None:
    db_path = tmp_path / "index.db"
    with sqlite3.connect(db_path) as writer:
        writer.execute("CREATE TABLE evidence (value TEXT)")
    reader = connection_profile.open_readonly_connection(db_path, validate_schema=False)
    try:
        with connection_profile.readonly_temp_staging(reader, temp_store="FILE"):
            assert reader.execute("PRAGMA temp_store").fetchone() == (1,)
            reader.execute("CREATE TEMP TABLE projection (value TEXT)")
            reader.execute("INSERT INTO projection VALUES ('derived')")
        assert reader.execute("SELECT value FROM projection").fetchone() == ("derived",)
        with pytest.raises(ValueError, match="before TEMP objects exist"):
            with connection_profile.readonly_temp_staging(reader, temp_store="FILE"):
                pass
    finally:
        reader.close()


def test_attach_database_on_a_profiled_reader_is_read_only(tmp_path: Path) -> None:
    """A reader's sibling attachment succeeds and cannot write the sibling.

    Anti-vacuity: route ``attach_database`` through a plain parameterized
    ATTACH and the read authorizer denies it; open the sibling without
    ``mode=ro`` and the INSERT below succeeds.
    """
    db_path = tmp_path / "index.db"
    sibling_path = tmp_path / "source.db"
    with sqlite3.connect(db_path) as writer:
        writer.execute("CREATE TABLE evidence (value TEXT)")
    with sqlite3.connect(sibling_path) as writer:
        writer.execute("CREATE TABLE raw (value TEXT)")
        writer.execute("INSERT INTO raw VALUES ('source')")

    reader = connection_profile.open_readonly_connection(db_path, validate_schema=False)
    try:
        connection_profile.attach_database(reader, sibling_path, alias="source_tier")
        assert reader.execute("SELECT value FROM source_tier.raw").fetchone() == ("source",)
        with pytest.raises(sqlite3.DatabaseError):
            reader.execute("INSERT INTO source_tier.raw VALUES ('wrong')")
        with pytest.raises(sqlite3.DatabaseError):
            reader.execute("PRAGMA busy_timeout = 1")
    finally:
        reader.close()


@pytest.mark.parametrize(
    ("profile_name", "expected_busy_timeout_ms", "expected_query_only"),
    [
        ("background-read", connection_profile.DB_TIMEOUT * 1000, 1),
        ("interactive-read", connection_profile.READ_DB_TIMEOUT * 1000, 1),
        ("publication", connection_profile.DB_TIMEOUT * 1000, 0),
    ],
)
def test_open_profiled_connection_applies_the_selected_profile(
    tmp_path: Path,
    profile_name: str,
    expected_busy_timeout_ms: int,
    expected_query_only: int,
) -> None:
    db_path = tmp_path / "generation.sqlite"
    with sqlite3.connect(db_path) as connection:
        connection.execute("CREATE TABLE evidence (value TEXT)")

    connection = connection_profile.open_profiled_connection(
        db_path,
        profile=_declared_profile(profile_name),
    )
    try:
        assert connection.execute("PRAGMA busy_timeout").fetchone() == (expected_busy_timeout_ms,)
        assert connection.execute("PRAGMA query_only").fetchone() == (expected_query_only,)
        if profile_name == "publication":
            assert connection.execute("PRAGMA journal_mode").fetchone() == ("wal",)
        else:
            with pytest.raises(sqlite3.DatabaseError, match="not authorized|attempt to write a readonly database"):
                connection.execute("INSERT INTO evidence VALUES ('blocked')")
    finally:
        connection.close()


def test_index_schema_guard_distinguishes_uninitialized_from_stale(tmp_path: Path) -> None:
    db_path = tmp_path / "index.db"
    with sqlite3.connect(db_path):
        pass

    with connection_profile.open_readonly_connection(db_path) as connection:
        assert connection.execute("PRAGMA user_version").fetchone() == (0,)

    # An unstamped file with a schema is partial derived state, not absence.
    with sqlite3.connect(db_path) as connection:
        connection.execute("CREATE TABLE evidence (value TEXT NOT NULL)")
    with pytest.raises(SchemaSkew) as partial:
        connection_profile.open_readonly_connection(db_path)
    assert partial.value.tier == ArchiveTier.INDEX.value
    assert partial.value.found == 0

    # A version this runtime cannot serve. The index sits at the format
    # floor, so stepping one below collapses onto 0 -- the uninitialized
    # sentinel this test distinguishes from skew.
    stale_version = ARCHIVE_VERSION_BY_TIER[ArchiveTier.INDEX] + 1
    with sqlite3.connect(db_path) as connection:
        connection.execute(f"PRAGMA user_version = {stale_version}")

    with pytest.raises(SchemaSkew) as excinfo:
        connection_profile.open_readonly_connection(db_path)

    assert excinfo.value.tier == ArchiveTier.INDEX.value
    assert excinfo.value.found == stale_version


@pytest.mark.parametrize("factory", [connection_profile.open_connection, connection_profile.open_daemon_connection])
def test_schema_skew_write_profiles_refuse_stale_archive_tier_before_returning_connection(
    tmp_path: Path, factory: Callable[..., sqlite3.Connection]
) -> None:
    # A version this runtime cannot serve in either direction. The durable
    # tiers sit at the format floor, so stepping one below collapses onto 0 --
    # the never-provisioned sentinel, not a skewed schema.
    skewed_version = ARCHIVE_VERSION_BY_TIER[ArchiveTier.USER] + 1
    db_path = tmp_path / "user.db"
    with sqlite3.connect(db_path) as connection:
        connection.execute(f"PRAGMA user_version = {skewed_version}")

    with pytest.raises(SchemaSkew) as excinfo:
        factory(db_path)

    assert excinfo.value.tier == ArchiveTier.USER.value
    assert excinfo.value.expected == ARCHIVE_VERSION_BY_TIER[ArchiveTier.USER]
    assert excinfo.value.found == skewed_version
    assert "durable state" in excinfo.value.remedy
    assert "do not rebuild" in excinfo.value.remedy
    assert "runtime that wrote it" in excinfo.value.remedy


@pytest.mark.parametrize(
    ("tier", "expected_terms"),
    [
        (ArchiveTier.SOURCE, ("durable state", "do not rebuild", "runtime that wrote it")),
        (ArchiveTier.AUDIT, ("durable state", "do not rebuild", "runtime that wrote it")),
        (ArchiveTier.INDEX, ("rebuildable derived state", "rebuild or recreate")),
        (ArchiveTier.EMBEDDINGS, ("expensive_rebuild derived state", "rebuild or recreate")),
        (ArchiveTier.OPS, ("disposable derived state", "rebuild or recreate")),
    ],
)
def test_schema_skew_remedy_matches_tier_durability(tier: ArchiveTier, expected_terms: tuple[str, ...]) -> None:
    remedy = connection_profile._schema_skew_remedy(tier)

    for term in expected_terms:
        assert term in remedy


def test_schema_skew_read_profile_refuses_stale_archive_tier(tmp_path: Path) -> None:
    db_path = tmp_path / "index.db"
    with sqlite3.connect(db_path) as connection:
        connection.execute(f"PRAGMA user_version = {ARCHIVE_VERSION_BY_TIER[ArchiveTier.INDEX] + 1}")

    with pytest.raises(SchemaSkew, match="index schema skew"):
        connection_profile.open_readonly_connection(db_path)


def test_schema_skew_diagnostic_read_profile_opens_stale_archive_tier(tmp_path: Path) -> None:
    db_path = tmp_path / "index.db"
    # A version this runtime cannot serve. The index sits at the format
    # floor, so stepping one below collapses onto 0 -- the uninitialized
    # sentinel this test distinguishes from skew.
    stale_version = ARCHIVE_VERSION_BY_TIER[ArchiveTier.INDEX] + 1
    with sqlite3.connect(db_path) as connection:
        connection.execute(f"PRAGMA user_version = {stale_version}")

    reader = connection_profile.open_readonly_connection(db_path, validate_schema=False)
    try:
        assert reader.execute("PRAGMA user_version").fetchone() == (stale_version,)
    finally:
        reader.close()


def test_schema_skew_explicit_tier_checks_noncanonical_generation_path(tmp_path: Path) -> None:
    db_path = tmp_path / "generation.sqlite"
    with sqlite3.connect(db_path) as connection:
        # Above, not below: the source tier sits at the format floor, and one
        # below it is the never-provisioned sentinel rather than a skew.
        connection.execute(f"PRAGMA user_version = {ARCHIVE_VERSION_BY_TIER[ArchiveTier.SOURCE] + 1}")

    with pytest.raises(SchemaSkew) as excinfo:
        connection_profile.open_readonly_connection(db_path, tier=ArchiveTier.SOURCE)

    assert excinfo.value.tier == ArchiveTier.SOURCE.value


def test_scratch_synchronous_override_only_honours_off(monkeypatch: pytest.MonkeyPatch) -> None:
    """Anti-vacuity: production profiles keep NORMAL unless the harness asks for OFF."""
    profile = connection_profile.WRITE_CONNECTION_PROFILE
    monkeypatch.delenv(connection_profile.SCRATCH_SYNCHRONOUS_ENV, raising=False)
    assert "PRAGMA synchronous = NORMAL" in profile.pragma_statements
    monkeypatch.setenv(connection_profile.SCRATCH_SYNCHRONOUS_ENV, "FULL")
    assert "PRAGMA synchronous = NORMAL" in profile.pragma_statements
    monkeypatch.setenv(connection_profile.SCRATCH_SYNCHRONOUS_ENV, "off")
    assert "PRAGMA synchronous = OFF" in profile.pragma_statements
    assert "PRAGMA synchronous = NORMAL" not in profile.pragma_statements


@pytest.mark.parametrize("factory", [connection_profile.open_connection, connection_profile.open_daemon_connection])
@pytest.mark.parametrize(
    "sibling_tier", [ArchiveTier.SOURCE, ArchiveTier.USER, ArchiveTier.EMBEDDINGS, ArchiveTier.OPS]
)
def test_index_write_profiles_refuse_stale_sibling_before_attach(
    tmp_path: Path,
    factory: Callable[..., sqlite3.Connection],
    sibling_tier: ArchiveTier,
) -> None:
    root = tmp_path
    index_path = root / "index.db"
    with closing(sqlite3.connect(index_path)) as connection:
        connection.execute(f"PRAGMA user_version = {ARCHIVE_VERSION_BY_TIER[ArchiveTier.INDEX]}")
        # The index itself must be current, identity included, or its own
        # derived-identity check refuses first and no sibling is consulted.
        stamp_derived_schema_identity(connection, ArchiveTier.INDEX.value)
        connection.commit()
    sibling_path = root / f"{sibling_tier.value}.db"
    # A version this runtime cannot serve in either direction. Stepping one
    # below the expected version collapses onto 0 for a version-1 tier, which
    # is the never-provisioned sentinel rather than a skewed schema.
    skewed_version = ARCHIVE_VERSION_BY_TIER[sibling_tier] + 1
    with sqlite3.connect(sibling_path) as connection:
        connection.execute(f"PRAGMA user_version = {skewed_version}")

    with pytest.raises(SchemaSkew) as excinfo:
        factory(index_path)

    assert excinfo.value.tier == sibling_tier.value
    assert excinfo.value.expected == ARCHIVE_VERSION_BY_TIER[sibling_tier]
    assert excinfo.value.found == skewed_version


def test_unprovisioned_durable_tier_opens_instead_of_reporting_skew(tmp_path: Path) -> None:
    """An empty tier file carries no schema, so it cannot be skewed against one.

    Anti-vacuity: restoring the version-only comparison makes the open raise
    ``SchemaSkew`` for ``found == 0``.
    """
    db_path = tmp_path / "source.db"
    sqlite3.connect(db_path).close()

    with connection_profile.open_readonly_connection(db_path) as connection:
        assert connection.execute("PRAGMA user_version").fetchone() == (0,)


def test_populated_durable_tier_at_version_zero_still_reports_skew(tmp_path: Path) -> None:
    """A durable tier holding tables without a version stamp is skew, not a fresh file.

    Anti-vacuity: exempting every ``found == 0`` tier regardless of its schema
    makes this open succeed.
    """
    db_path = tmp_path / "source.db"
    with sqlite3.connect(db_path) as connection:
        connection.execute("CREATE TABLE raw_sessions (raw_id TEXT PRIMARY KEY)")

    with pytest.raises(SchemaSkew, match="source schema skew"):
        connection_profile.open_readonly_connection(db_path)


def test_sealed_staging_connection_allows_only_in_memory_temp_writes(tmp_path: Path) -> None:
    """The historical exception stages rows without relaxing the main image."""
    db_path = tmp_path / "source.db"
    with sqlite3.connect(db_path) as connection:
        connection.execute("CREATE TABLE evidence (value TEXT)")
        connection.execute("INSERT INTO evidence VALUES ('selected')")
    before = db_path.read_bytes()
    before_files = {path.name for path in tmp_path.iterdir()}

    connection = connection_profile.open_sealed_staging_connection(db_path, validate_schema=False)
    try:
        assert connection.execute("PRAGMA query_only").fetchone() == (0,)
        assert connection.execute("PRAGMA temp_store").fetchone() == (2,)
        assert connection.execute("SELECT value FROM evidence").fetchone() == ("selected",)
        connection.execute("CREATE TEMP TABLE stage (value TEXT)")
        connection.execute("INSERT INTO stage VALUES ('staged')")
        connection.execute("CREATE INDEX stage_value ON stage(value)")
        assert connection.execute("SELECT value FROM temp.stage").fetchone() == ("staged",)
        connection.execute("REINDEX stage_value")

        denied = (
            "INSERT INTO main.evidence VALUES ('blocked')",
            "ATTACH DATABASE 'other.db' AS other",
            "DETACH DATABASE main",
            "PRAGMA query_only = ON",
            "PRAGMA temp_store = FILE",
            "VACUUM",
            "CREATE VIRTUAL TABLE virtual_stage USING fts5(value)",
            "CREATE TEMP TRIGGER stage_trigger AFTER INSERT ON stage BEGIN SELECT 1; END",
            "SELECT random()",
        )
        for statement in denied:
            with pytest.raises(sqlite3.DatabaseError):
                connection.execute(statement)
    finally:
        connection.close()

    assert db_path.read_bytes() == before
    assert {path.name for path in tmp_path.iterdir()} == before_files


def test_sealed_staging_profile_is_not_an_ordinary_read_profile() -> None:
    profile = connection_profile.SEALED_STAGING_CONNECTION_PROFILE
    assert profile.immutable is True
    assert profile.generation_identity == "sealed"
    assert profile.temp_store == "MEMORY"
    assert profile.query_only is False
    assert profile not in connection_profile.READ_PROFILES.values()


def test_ordinary_immutable_reader_still_rejects_temp_staging(tmp_path: Path) -> None:
    db_path = tmp_path / "source.db"
    sqlite3.connect(db_path).close()

    with connection_profile.open_readonly_connection(db_path, immutable=True, validate_schema=False) as connection:
        assert connection.execute("PRAGMA query_only").fetchone() == (1,)
        with pytest.raises(sqlite3.DatabaseError, match="not authorized|readonly database"):
            connection.execute("CREATE TEMP TABLE forbidden (value TEXT)")


def test_explicit_read_timeout_bounds_the_lock_wait_even_at_the_default_value(tmp_path: Path) -> None:
    """An explicit bound equal to the old default sentinel used to be discarded.

    Anti-vacuity: comparing ``timeout`` against ``READ_DB_TIMEOUT`` (5 s)
    instead of ``None`` makes the background profile's 30 s busy_timeout win.
    """
    db_path = tmp_path / "evidence.db"
    with closing(sqlite3.connect(db_path)) as connection:
        connection.execute("CREATE TABLE evidence (value TEXT)")

    def busy_ms(**kwargs: Any) -> int:
        with closing(connection_profile.open_readonly_connection(db_path, **kwargs)) as connection:
            return int(connection.execute("PRAGMA busy_timeout").fetchone()[0])

    assert busy_ms(timeout_class="background-read") == 30_000
    assert busy_ms(timeout_class="background-read", timeout=5.0) == 5_000
    assert busy_ms(timeout=0.2) == 200


@pytest.mark.parametrize("factory", [connection_profile.open_connection, connection_profile.open_daemon_connection])
@pytest.mark.parametrize("tier", [ArchiveTier.USER, ArchiveTier.SOURCE, ArchiveTier.EMBEDDINGS])
def test_writers_refuse_an_uninitialized_durable_tier(
    tmp_path: Path, factory: Callable[..., sqlite3.Connection], tier: ArchiveTier
) -> None:
    """The former shared read exemption returned a writable zero-schema handle."""
    path = tmp_path / f"{tier.value}.db"
    path.touch()
    with pytest.raises(SchemaSkew) as raised:
        with factory(path):
            pass
    assert raised.value.tier == tier.value
    assert raised.value.found == 0


def test_pending_population_blocks_literal_bytes_and_uri_connection_paths(tmp_path: Path) -> None:
    import os

    from polylogue.storage.sqlite.managed_connection import sqlite_connection
    from polylogue.storage.sqlite.population_admission import POPULATION_PENDING, ArchivePopulationPendingError

    root = tmp_path / "pending archive"
    root.mkdir()
    (root / POPULATION_PENDING).write_text('{"fixture":"unfinished"}')
    database = root / "source.db"
    for path, uri in ((os.fsencode(database), False), (database.as_uri() + "?mode=rwc", True)):
        with pytest.raises(ArchivePopulationPendingError):
            with sqlite_connection(path, uri=uri):
                pytest.fail("a literal filesystem or URI path bypassed pending admission")
    assert not database.exists()


def test_writer_source_attachment_retains_reads_and_refuses_native_cursor_mutations(tmp_path: Path) -> None:
    from polylogue.storage.sqlite.write_lease import write_lease

    source = tmp_path / "source.db"
    with closing(sqlite3.connect(source)) as connection:
        connection.execute("CREATE TABLE evidence (value TEXT)")
        connection.execute("INSERT INTO evidence VALUES ('retained')")
        connection.commit()
    with write_lease("test.attached-source", archive_root=tmp_path):
        with closing(
            connection_profile.open_isolated_write_connection(
                tmp_path / "index.db", purpose="test.attached-source", archive_root=tmp_path
            )
        ) as index:
            connection_profile.attach_database(index, source, alias="source_tier")
            with closing(index.execute("SELECT value FROM source_tier.evidence")) as cursor:
                assert cursor.fetchone()[0] == "retained"
            with closing(index.cursor(factory=sqlite3.Cursor)) as cursor:
                with pytest.raises(sqlite3.OperationalError):
                    cursor.execute("UPDATE source_tier.evidence SET value = 'wrong'")
            index.rollback()
            with closing(index.execute("CREATE TABLE local_evidence (value TEXT)")):
                pass
            with closing(index.execute("INSERT INTO local_evidence VALUES ('index remains writable')")):
                pass
            index.commit()
            with closing(index.execute("SELECT value FROM local_evidence")) as cursor:
                assert cursor.fetchone()[0] == "index remains writable"


@pytest.mark.parametrize("factory", ["ordinary", "daemon", "existing-only", "cached"])
def test_archive_writer_factories_preserve_readonly_source_uri_attachments(tmp_path: Path, factory: str) -> None:
    from polylogue.storage.io_phase_metrics import live_connection_cursors, native_connection_physically_closed
    from polylogue.storage.sqlite.connection import _get_cached_connection
    from polylogue.storage.sqlite.connection_profile import retained_native_sql_owners_on_current_thread
    from polylogue.storage.sqlite.write_lease import write_lease
    from tests.infra.archive_templates import bootstrap_archive_root

    with write_lease("test.writer-source-uri", archive_root=tmp_path):
        bootstrap_archive_root(tmp_path)
        path = tmp_path / "index.db"
        if factory == "cached":
            connection = _get_cached_connection(path, archive_root=tmp_path)
        elif factory == "existing-only":
            connection = connection_profile._connect_archive_writer(
                path, profile=connection_profile.WRITE_CONNECTION_PROFILE, archive_root=tmp_path, existing_only=True
            )
            connection_profile._attach_sibling_tiers(connection, archive_root=tmp_path)
        else:
            selected = (
                connection_profile.open_connection
                if factory == "ordinary"
                else connection_profile.open_daemon_connection
            )
            connection = selected(path, archive_root=tmp_path)
        try:
            with closing(connection.execute("SELECT count(*) FROM source_tier.raw_sessions")) as cursor:
                assert cursor.fetchone()[0] == 0
            with closing(connection.cursor(factory=sqlite3.Cursor)) as cursor:
                with pytest.raises(sqlite3.OperationalError):
                    cursor.execute("DELETE FROM source_tier.raw_sessions")
            connection.rollback()
            with closing(connection.execute("CREATE TABLE local_uri_evidence(value TEXT)")):
                pass
            with closing(connection.execute("INSERT INTO local_uri_evidence VALUES ('writable')")):
                pass
            connection.commit()
        finally:
            if factory == "cached":
                owner = next(
                    owner for owner in retained_native_sql_owners_on_current_thread() if owner.connection is connection
                )
                owner.close()
            else:
                connection.close()
        assert native_connection_physically_closed(connection)
        assert live_connection_cursors(connection) == ()


def test_existing_archive_writer_refuses_missing_file_without_creating_it(tmp_path: Path) -> None:
    from polylogue.storage.sqlite.write_lease import write_lease

    path = tmp_path / "missing.db"
    with write_lease("test.existing-uri-refusal", archive_root=tmp_path):
        with pytest.raises(FileNotFoundError):
            connection_profile._connect_archive_writer(
                path, profile=connection_profile.WRITE_CONNECTION_PROFILE, archive_root=tmp_path, existing_only=True
            )
    assert not path.exists()


@pytest.mark.parametrize("failed_close", [False, True])
def test_native_owner_settles_original_supported_connection_after_external_close(
    tmp_path: Path, failed_close: bool
) -> None:
    import os

    from polylogue.storage.io_phase_metrics import connect_measured, native_connection_physically_closed
    from tests.infra.native_sql_descriptor_probe import selected_file_descriptors
    from tests.infra.sqlite_cursor_settlement import ControlledCursor

    path = tmp_path / "externally-closed.db"
    connection = connect_measured(path)
    owner = connection_profile.NativeSQLCustodyOwner(connection)
    completions: list[str] = []
    owner.retain_settlement_callback(lambda: completions.append("settled"))
    cursor = connection.cursor(factory=ControlledCursor)
    assert isinstance(cursor, ControlledCursor)
    cursor.execute("SELECT 1 UNION ALL SELECT 2")
    cursor.fetchone()
    identity = path.stat().st_dev, path.stat().st_ino
    observe_descriptors = os.path.isdir("/proc/self/fd")
    if observe_descriptors:
        assert selected_file_descriptors(identity)
    try:
        if failed_close:
            cursor.allow_cleanup.clear()
            with pytest.raises(BaseExceptionGroup):
                connection.close()
            assert not native_connection_physically_closed(connection)
            with pytest.raises(connection_profile.NativeConnectionSettlementError):
                owner.close()
            assert owner.connection is connection and completions == []
            if observe_descriptors:
                assert selected_file_descriptors(identity)
            cursor.allow_cleanup.set()
            owner.close()
        else:
            connection.close()
            assert native_connection_physically_closed(connection)
            assert owner.connection is connection and completions == []
            owner.close()
        assert owner.connection is None and completions == ["settled"]
        assert native_connection_physically_closed(connection)
        assert cursor.close_attempts == (3 if failed_close else 1)
        if observe_descriptors:
            assert selected_file_descriptors(identity) == ()
        owner.close()
        assert completions == ["settled"]
    finally:
        cursor.allow_cleanup.set()
        owner.close()


def test_source_factory_initializes_local_profile_before_authorization_and_retains_write_refusal(
    tmp_path: Path,
) -> None:
    from polylogue.storage.sqlite.write_lease import write_lease
    from tests.infra.archive_templates import bootstrap_archive_root

    with write_lease("test.source-profile-bootstrap", archive_root=tmp_path):
        bootstrap_archive_root(tmp_path)
    owner = connection_profile.NativeSQLCustodyOwner(
        connection_profile.open_source_tier_write_connection(tmp_path / "source.db", archive_root=tmp_path)
    )
    try:
        connection = owner.require_connection()
        with closing(connection.execute("PRAGMA foreign_keys")) as rows:
            assert rows.fetchone()[0] == 1
        for statement in ("PRAGMA foreign_keys = OFF", "PRAGMA journal_mode = DELETE", "BEGIN IMMEDIATE"):
            with pytest.raises(sqlite3.DatabaseError):
                connection.execute(statement)
        assert not connection.in_transaction
    finally:
        owner.close()
