"""Source002 changes only index partitions, with live rollback and row parity."""

from __future__ import annotations

import sqlite3
import tracemalloc
from contextlib import closing
from dataclasses import replace
from pathlib import Path

import pytest

from polylogue.core.raw_failure_evidence import RawFailureEvidenceKind
from polylogue.storage.sqlite import migration_runner
from polylogue.storage.sqlite.archive_tiers import ARCHIVE_VERSION_BY_TIER
from polylogue.storage.sqlite.archive_tiers.source_write import upsert_raw_artifact
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from polylogue.storage.sqlite.migration_runner import MigrationError, migrate_archive_tier
from tests.infra.durable_tier_fixtures import bootstrap_baseline_archive
from tests.infra.index_replacement import missing_coordinates_artifact, source_baseline


@pytest.mark.parametrize(
    "before,after",
    (
        ("prefix\x00before", "prefix\x00after"),
        (None, ""),
        (b"", ""),
        (1, 1.0),
        ("1", 1),
    ),
)
def test_literal_row_proof_preserves_exact_sqlite_types_and_nul_suffixes(
    before: object,
    after: object,
) -> None:
    with closing(sqlite3.connect(":memory:")) as conn:
        conn.execute("CREATE TABLE values_proof (id INTEGER PRIMARY KEY, value)")
        conn.execute("INSERT INTO values_proof VALUES (1, ?)", (before,))
        original = migration_runner._durable_literal_rows_digest(conn)
        conn.execute("UPDATE values_proof SET value=? WHERE id=1", (after,))
        assert migration_runner._durable_literal_rows_digest(conn) != original


@pytest.mark.parametrize("without_rowid", (False, True))
def test_literal_row_proof_uses_actual_row_identity_across_index_replacement(without_rowid: bool) -> None:
    with closing(sqlite3.connect(":memory:")) as conn:
        conn.execute(
            "CREATE TABLE items (a TEXT, b INTEGER, value BLOB, PRIMARY KEY(a,b))"
            + (" WITHOUT ROWID" if without_rowid else "")
        )
        conn.executemany("INSERT INTO items VALUES (?, ?, ?)", (("z", 1, b"one"), ("a", 2, b"two")))
        original = migration_runner._durable_literal_rows_digest(conn)
        conn.execute("CREATE INDEX reordered ON items(a DESC, b)")
        assert migration_runner._durable_literal_rows_digest(conn) == original
        conn.execute("DROP INDEX reordered")
        conn.execute("CREATE INDEX reordered ON items(b DESC, a)")
        assert migration_runner._durable_literal_rows_digest(conn) == original


def test_literal_row_proof_preserves_non_utf8_sqlite_text_bytes() -> None:
    with closing(sqlite3.connect(":memory:")) as conn:
        conn.execute("CREATE TABLE values_proof (id INTEGER PRIMARY KEY, value TEXT)")
        conn.execute("INSERT INTO values_proof VALUES (1, CAST(X'80FF' AS TEXT))")
        original = migration_runner._durable_literal_rows_digest(conn)
        conn.execute("UPDATE values_proof SET value=CAST(X'80FE' AS TEXT)")
        assert migration_runner._durable_literal_rows_digest(conn) != original


def test_source002_rehearses_on_connection_local_schema_replica(tmp_path: Path) -> None:
    """Rehearsal must not demand a physical archive identity for its memory clone."""
    conn, _raw_ids = source_baseline(tmp_path / "source.db")
    try:
        proof = migration_runner.rehearse_durable_migration_chain(
            conn,
            ArchiveTier.SOURCE,
            target_version=ARCHIVE_VERSION_BY_TIER[ArchiveTier.SOURCE],
            evidence_ref="proof:source002-memory-rehearsal",
        )
        assert proof.matches
        assert tuple(step.version for step in proof.steps) == tuple(
            range(2, ARCHIVE_VERSION_BY_TIER[ArchiveTier.SOURCE] + 1)
        )
        assert conn.execute("PRAGMA user_version").fetchone()[0] == 1
    finally:
        conn.close()


def test_runtime_probe_installs_numbered_source_effect_without_train_release(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_runtime_tier_probe

    def refuse_recursive_release(*args: object, **kwargs: object) -> None:
        pytest.fail("an isolated schema probe must not release the train it is proving")

    monkeypatch.setattr(
        "polylogue.storage.sqlite.durable_change_train.execute_durable_change_train", refuse_recursive_release
    )
    with closing(sqlite3.connect(":memory:")) as conn:
        initialize_runtime_tier_probe(conn, ArchiveTier.SOURCE)
        assert conn.execute("PRAGMA user_version").fetchone()[0] == ARCHIVE_VERSION_BY_TIER[ArchiveTier.SOURCE]
        partitions = conn.execute(
            "SELECT sql FROM sqlite_schema WHERE name IN "
            "('idx_raw_artifacts_source_identity', 'idx_raw_artifacts_failure_identity')"
        ).fetchall()
        assert len(partitions) == 2
        assert all("'terminal_missing_source_coordinates'" in row[0] for row in partitions)


def test_runtime_probe_refuses_populated_or_undeclared_file_connections(tmp_path: Path) -> None:
    from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_runtime_tier_probe

    with closing(sqlite3.connect(":memory:")) as conn:
        conn.execute("CREATE TABLE custody (value TEXT)")
        conn.execute("INSERT INTO custody VALUES ('retained')")
        conn.commit()
        with pytest.raises(RuntimeError):
            initialize_runtime_tier_probe(conn, ArchiveTier.SOURCE)
        assert conn.execute("SELECT value FROM custody").fetchall() == [("retained",)]
        assert conn.execute("PRAGMA user_version").fetchone()[0] == 0
    path = tmp_path / "isolated-source.db"
    with closing(sqlite3.connect(path)) as conn:
        with pytest.raises(RuntimeError):
            initialize_runtime_tier_probe(conn, ArchiveTier.SOURCE)
        assert conn.execute("SELECT name FROM sqlite_schema").fetchall() == []
        initialize_runtime_tier_probe(conn, ArchiveTier.SOURCE, probe_path=path)
        assert conn.execute("PRAGMA user_version").fetchone()[0] == ARCHIVE_VERSION_BY_TIER[ArchiveTier.SOURCE]


def test_canonical_birth_marker_stays_baseline_after_source_train_and_reopen(tmp_path: Path) -> None:
    import json

    from polylogue.storage.sqlite.archive_tiers.bootstrap import (
        initialize_active_archive_root,
        invalidate_active_archive_bootstrap,
    )

    initialize_active_archive_root(tmp_path)
    marker_path = tmp_path / ".polylogue-format.json"
    original = marker_path.read_bytes()
    assert set(json.loads(original)["tier_versions"].values()) == {1}
    with closing(sqlite3.connect(tmp_path / "source.db")) as conn:
        assert conn.execute("PRAGMA user_version").fetchone()[0] == ARCHIVE_VERSION_BY_TIER[ArchiveTier.SOURCE]
    invalidate_active_archive_bootstrap(tmp_path)
    initialize_active_archive_root(tmp_path)
    assert marker_path.read_bytes() == original


def test_standalone_runtime_source_constructor_refuses_before_creating_baseline(tmp_path: Path) -> None:
    from polylogue.core.errors import SchemaSkew
    from polylogue.storage.sqlite.archive_tiers.bootstrap import (
        initialize_archive_database,
        open_initialized_tier_connection,
    )

    path = tmp_path / "source.db"
    for open_tier in (initialize_archive_database, open_initialized_tier_connection):
        with pytest.raises(SchemaSkew) as refusal:
            open_tier(path, ArchiveTier.SOURCE)
        assert refusal.value.found == 0
        assert refusal.value.expected == ARCHIVE_VERSION_BY_TIER[ArchiveTier.SOURCE]
        assert not path.exists()
    conn, _raw_ids = source_baseline(path)
    try:
        original = migration_runner._durable_literal_rows_digest(conn)
    finally:
        conn.close()
    with pytest.raises(SchemaSkew) as refusal:
        initialize_archive_database(path, ArchiveTier.SOURCE)
    assert refusal.value.found == 1
    with closing(sqlite3.connect(path)) as conn:
        assert conn.execute("PRAGMA user_version").fetchone()[0] == 1
        assert migration_runner._durable_literal_rows_digest(conn) == original


def test_source002_preserves_each_windowless_raw_refusal_after_restart(tmp_path: Path) -> None:
    """Keeping the v1 coordinate index would roll back the second raw's refusal."""
    path = tmp_path / "source.db"
    conn, raw_ids = source_baseline(path)
    try:
        upsert_raw_artifact(conn, raw_ids[0], missing_coordinates_artifact(raw_ids[0]))
        with pytest.raises(sqlite3.IntegrityError):
            upsert_raw_artifact(conn, raw_ids[1], missing_coordinates_artifact(raw_ids[1]))
        literal_rows = tuple(tuple(row) for row in conn.execute("SELECT * FROM raw_sessions ORDER BY raw_id"))
        result = migrate_archive_tier(conn, ArchiveTier.SOURCE, target_version=2, backup_manifest=None)
        assert result.applied_versions == (2,)
        assert tuple(tuple(row) for row in conn.execute("SELECT * FROM raw_sessions ORDER BY raw_id")) == literal_rows
        upsert_raw_artifact(conn, raw_ids[1], missing_coordinates_artifact(raw_ids[1]))
        assert (
            migrate_archive_tier(conn, ArchiveTier.SOURCE, target_version=2, backup_manifest=None).applied_versions
            == ()
        )
    finally:
        conn.close()
    with closing(sqlite3.connect(path)) as restarted:
        assert restarted.execute("PRAGMA user_version").fetchone()[0] == 2
        assert {row[0] for row in restarted.execute("SELECT raw_id FROM raw_artifacts")} == set(raw_ids)
        assert restarted.execute("PRAGMA foreign_key_check").fetchall() == []


@pytest.mark.parametrize(
    "sql",
    (
        "DROP INDEX a;",
        "DROP INDEX a; CREATE UNIQUE INDEX b ON items(id) WHERE kind IN ('old','new');",
        "DROP TABLE items; CREATE INDEX a ON items(id) WHERE kind IN ('old','new');",
        "DELETE FROM items;",
        "DROP INDEX a; CREATE INDEX a ON items(id) WHERE kind IN ('old','new'); DELETE FROM items;",
        "DROP INDEX a; CREATE INDEX a ON items(id) WHERE kind IN ('old','new'); ALTER TABLE items ADD COLUMN extra TEXT;",
        "DROP INDEX a; CREATE INDEX a ON items(id);",
        "DROP INDEX a; CREATE INDEX a ON items(id) WHERE kind IN ('old','new'); COMMIT;",
    ),
)
def test_index_safety_claim_rejects_nonpaired_or_row_mutating_sql(sql: str) -> None:
    with pytest.raises(MigrationError):
        migration_runner._requires_migration_backup(
            Path("002_invalid.sql"),
            f"{migration_runner._INDEX_REPLACEMENT_MARKER}\n{sql}\n",
        )


def test_schema_rehearsal_refuses_a_false_index_effect_before_live_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    conn, _raw_ids = source_baseline(tmp_path / "source.db")
    execute = migration_runner._execute_migration_sql
    reached = False

    def wrong_keys(replica: sqlite3.Connection, sql: str) -> None:
        nonlocal reached
        if not replica.execute("PRAGMA database_list").fetchone()[2] and sql.startswith(
            migration_runner._INDEX_REPLACEMENT_MARKER
        ):
            reached = True
            sql = sql.replace(
                "ON raw_artifacts(origin, source_path, source_index)", "ON raw_artifacts(origin, source_path)"
            )
        execute(replica, sql)

    monkeypatch.setattr(migration_runner, "_execute_migration_sql", wrong_keys)
    try:
        before = tuple(tuple(row) for row in conn.execute("SELECT name, sql FROM sqlite_schema ORDER BY name"))
        with pytest.raises(MigrationError):
            migration_runner.rehearse_durable_migration_chain(
                conn,
                ArchiveTier.SOURCE,
                target_version=ARCHIVE_VERSION_BY_TIER[ArchiveTier.SOURCE],
                evidence_ref="proof:false-index-claim",
            )
        assert reached
        assert tuple(tuple(row) for row in conn.execute("SELECT name, sql FROM sqlite_schema ORDER BY name")) == before
        assert conn.execute("PRAGMA user_version").fetchone()[0] == 1
    finally:
        conn.close()


@pytest.mark.parametrize(
    "mutation",
    (
        "keys",
        "unique",
        "members",
        "partition",
        "expression",
        "collation",
        "order",
        "predicate",
    ),
)
def test_live_index_effect_proof_rolls_back_false_safe_claim(
    tmp_path: Path,
    mutation: str,
) -> None:
    conn, _raw_ids = source_baseline(tmp_path / "source.db")
    try:
        step = migration_runner._load_migrations(ArchiveTier.SOURCE)[0]
        sql = step.sql
        if mutation == "keys":
            sql = sql.replace(
                "ON raw_artifacts(origin, source_path, source_index)", "ON raw_artifacts(origin, source_path)"
            )
        elif mutation == "unique":
            sql = sql.replace("CREATE UNIQUE INDEX", "CREATE INDEX", 1)
        elif mutation == "members":
            sql = sql.replace("    'terminal_corrupt_input',\n", "", 1)
        elif mutation == "partition":
            sql = sql.replace("'terminal_missing_source_coordinates'", "'different_partition'", 1)
        elif mutation == "expression":
            sql = sql.replace(
                "ON raw_artifacts(origin, source_path, source_index)",
                "ON raw_artifacts(lower(origin), source_path, source_index)",
            )
        elif mutation == "collation":
            sql = sql.replace(
                "ON raw_artifacts(origin, source_path, source_index)",
                "ON raw_artifacts(origin COLLATE NOCASE, source_path, source_index)",
            )
        elif mutation == "order":
            sql = sql.replace(
                "ON raw_artifacts(origin, source_path, source_index)",
                "ON raw_artifacts(origin DESC, source_path, source_index)",
            )
        else:
            sql = sql.replace("artifact_kind NOT IN", "artifact_kind IN", 1)
        original = tuple(
            tuple(row) for row in conn.execute("SELECT name, sql FROM sqlite_schema WHERE type='index' ORDER BY name")
        )
        conn.execute("BEGIN IMMEDIATE")
        with pytest.raises(MigrationError), conn:
            migration_runner._execute_proved_migration_sql(conn, replace(step, sql=sql))
        assert (
            tuple(
                tuple(row)
                for row in conn.execute("SELECT name, sql FROM sqlite_schema WHERE type='index' ORDER BY name")
            )
            == original
        )
        assert conn.execute("PRAGMA user_version").fetchone()[0] == 1
    finally:
        conn.close()


@pytest.mark.parametrize("failure", (MigrationError, KeyboardInterrupt))
def test_failure_between_index_drop_and_create_preserves_old_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: type[BaseException],
) -> None:
    conn, raw_ids = source_baseline(tmp_path / "source.db")
    original_execute = migration_runner._execute_migration_sql
    reached = False
    injected = failure("injected_index_replacement_interruption")
    try:
        original = tuple(
            tuple(row) for row in conn.execute("SELECT name, sql FROM sqlite_schema WHERE type='index' ORDER BY name")
        )

        def interrupted(connection: sqlite3.Connection, sql: str) -> None:
            nonlocal reached
            if connection is conn and sql.startswith(migration_runner._INDEX_REPLACEMENT_MARKER):
                reached = True
                connection.execute("DROP INDEX idx_raw_artifacts_source_identity")
                raise injected
            original_execute(connection, sql)

        monkeypatch.setattr(migration_runner, "_execute_migration_sql", interrupted)
        with pytest.raises(failure) as outcome:
            migrate_archive_tier(conn, ArchiveTier.SOURCE, target_version=2, backup_manifest=None)
        assert reached and outcome.value is injected
        assert conn.execute("PRAGMA user_version").fetchone()[0] == 1
        assert {row[0] for row in conn.execute("SELECT raw_id FROM raw_sessions")} == set(raw_ids)
        assert (
            tuple(
                tuple(row)
                for row in conn.execute("SELECT name, sql FROM sqlite_schema WHERE type='index' ORDER BY name")
            )
            == original
        )
    finally:
        conn.close()


def test_second_unique_index_failure_rolls_back_first_replacement(tmp_path: Path) -> None:
    """Two existing refusal raws make a narrowed second unique key fail CREATE."""
    conn, raw_ids = source_baseline(tmp_path / "source.db")
    try:
        for raw_id in raw_ids:
            kind = RawFailureEvidenceKind.TERMINAL_UNKNOWN_EXPORT_NO_SESSION
            artifact = replace(
                missing_coordinates_artifact(raw_id), artifact_kind=kind.value, support_status=kind.support_status
            )
            upsert_raw_artifact(conn, raw_id, artifact)
        step = migration_runner._load_migrations(ArchiveTier.SOURCE)[0]
        sql = step.sql.replace(
            "ON raw_artifacts(raw_id, origin, source_path, source_index)",
            "ON raw_artifacts(origin, source_path, source_index)",
        )
        before = tuple(tuple(row) for row in conn.execute("SELECT name, sql FROM sqlite_schema ORDER BY name"))
        digest = migration_runner._durable_literal_rows_digest(conn)
        conn.execute("BEGIN IMMEDIATE")
        with pytest.raises(sqlite3.IntegrityError), conn:
            migration_runner._execute_proved_migration_sql(conn, replace(step, sql=sql))
        assert tuple(tuple(row) for row in conn.execute("SELECT name, sql FROM sqlite_schema ORDER BY name")) == before
        assert migration_runner._durable_literal_rows_digest(conn) == digest
        assert conn.execute("PRAGMA user_version").fetchone()[0] == 1
        assert conn.execute("PRAGMA foreign_key_check").fetchall() == []
        assert conn.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    finally:
        conn.close()


def test_fresh_constructor_records_baseline_before_declared_source002_and_restarts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failure at train admission retains the six baseline tiers for restart."""
    from polylogue.storage.sqlite import durable_change_train
    from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root

    execute = durable_change_train.execute_durable_change_train
    entered = False

    def fail_before_train(*args: object, **kwargs: object) -> object:
        nonlocal entered
        entered = True
        for tier in ArchiveTier:
            with closing(sqlite3.connect(tmp_path / f"{tier.value}.db")) as conn:
                assert conn.execute("PRAGMA user_version").fetchone()[0] == 1
        markers = tmp_path / ".maintenance-state" / "durable-change-trains"
        assert (markers / ".bootstrap").is_file()
        raise RuntimeError("injected_before_source_train")

    monkeypatch.setattr(durable_change_train, "execute_durable_change_train", fail_before_train)
    with pytest.raises(RuntimeError, match="injected_before_source_train"):
        initialize_active_archive_root(tmp_path)
    assert entered
    monkeypatch.setattr(durable_change_train, "execute_durable_change_train", execute)
    initialize_active_archive_root(tmp_path)
    with closing(sqlite3.connect(tmp_path / "source.db")) as conn:
        assert conn.execute("PRAGMA user_version").fetchone()[0] == ARCHIVE_VERSION_BY_TIER[ArchiveTier.SOURCE]
    initialize_active_archive_root(tmp_path)


def test_source002_constructor_preserves_populated_baseline_on_restart(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from polylogue.storage.sqlite import durable_change_train
    from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root

    execute = durable_change_train.execute_durable_change_train

    def pending(*args: object, **kwargs: object) -> object:
        raise RuntimeError("injected_train_pending")

    monkeypatch.setattr(durable_change_train, "execute_durable_change_train", pending)
    with pytest.raises(RuntimeError, match="injected_train_pending"):
        initialize_active_archive_root(tmp_path)
    conn, raw_ids = source_baseline(tmp_path / "source.db")
    try:
        rows = tuple(tuple(row) for row in conn.execute("SELECT * FROM raw_sessions ORDER BY raw_id"))
    finally:
        conn.close()
    monkeypatch.setattr(durable_change_train, "execute_durable_change_train", execute)
    # The restarted bootstrap resumes the pending source train over the
    # populated baseline and completes every step with its rows intact
    # (50831b9048 made a restarted train complete instead of refusing).
    initialize_active_archive_root(tmp_path)

    with closing(sqlite3.connect(tmp_path / "source.db")) as conn:
        assert conn.execute("PRAGMA user_version").fetchone()[0] == ARCHIVE_VERSION_BY_TIER[ArchiveTier.SOURCE]
        assert tuple(tuple(row) for row in conn.execute("SELECT * FROM raw_sessions ORDER BY raw_id")) == rows
        assert {row[0] for row in conn.execute("SELECT raw_id FROM raw_sessions")} == set(raw_ids)
        upsert_raw_artifact(conn, raw_ids[0], missing_coordinates_artifact(raw_ids[0]))
        upsert_raw_artifact(conn, raw_ids[1], missing_coordinates_artifact(raw_ids[1]))
    initialize_active_archive_root(tmp_path)
    with closing(sqlite3.connect(tmp_path / "source.db")) as conn:
        assert tuple(tuple(row) for row in conn.execute("SELECT * FROM raw_sessions ORDER BY raw_id")) == rows
        assert conn.execute(
            "SELECT COUNT(*) FROM raw_artifacts WHERE artifact_id LIKE 'missing-coordinates:%'"
        ).fetchone() == (2,)


def test_populated_baseline_train_cancellation_rolls_back_and_restarts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root

    bootstrap_baseline_archive(tmp_path, monkeypatch)
    conn, raw_ids = source_baseline(tmp_path / "source.db")
    try:
        before = migration_runner._durable_literal_rows_digest(conn)
        raw_rows = tuple(tuple(row) for row in conn.execute("SELECT * FROM raw_sessions ORDER BY raw_id"))
        blob_rows = tuple(
            tuple(row) for row in conn.execute("SELECT * FROM blob_refs ORDER BY blob_hash,ref_type,ref_id")
        )
    finally:
        conn.close()
    execute = migration_runner._execute_migration_sql
    reached = False
    injected = KeyboardInterrupt("injected_source_train_cancel")

    def cancelled(connection: sqlite3.Connection, sql: str) -> None:
        nonlocal reached
        database = connection.execute("PRAGMA database_list").fetchone()[2]
        if database == str(tmp_path / "source.db") and sql.startswith(migration_runner._INDEX_REPLACEMENT_MARKER):
            reached = True
            connection.execute("DROP INDEX idx_raw_artifacts_source_identity")
            raise injected
        execute(connection, sql)

    monkeypatch.setattr(migration_runner, "_execute_migration_sql", cancelled)
    with pytest.raises(KeyboardInterrupt) as failure:
        initialize_active_archive_root(tmp_path)
    assert reached and failure.value is injected
    with closing(sqlite3.connect(tmp_path / "source.db")) as conn:
        assert conn.execute("PRAGMA user_version").fetchone()[0] == 1
        assert migration_runner._durable_literal_rows_digest(conn) == before
        assert conn.execute("SELECT 1 FROM sqlite_schema WHERE name='idx_raw_artifacts_source_identity'").fetchone()
    monkeypatch.setattr(migration_runner, "_execute_migration_sql", execute)
    # The restarted train resumes from the rolled-back baseline and completes
    # every source step with the original rows intact.
    initialize_active_archive_root(tmp_path)
    with closing(sqlite3.connect(tmp_path / "source.db")) as conn:
        assert conn.execute("PRAGMA user_version").fetchone()[0] == ARCHIVE_VERSION_BY_TIER[ArchiveTier.SOURCE]
        assert tuple(tuple(row) for row in conn.execute("SELECT * FROM raw_sessions ORDER BY raw_id")) == raw_rows
        assert (
            tuple(tuple(row) for row in conn.execute("SELECT * FROM blob_refs ORDER BY blob_hash,ref_type,ref_id"))
            == blob_rows
        )
        assert {row[0] for row in conn.execute("SELECT raw_id FROM raw_sessions")} == set(raw_ids)


def test_baseline_readonly_and_acquisition_opens_refuse_without_migrating(
    one_shot_workspace_env: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from polylogue.core.errors import SchemaSkew
    from polylogue.storage.sqlite import durable_change_train
    from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
    from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root
    from polylogue.storage.sqlite.connection_profile import open_readonly_connection

    root = one_shot_workspace_env["archive_root"]

    def pending(*args: object, **kwargs: object) -> object:
        raise RuntimeError("injected_train_pending")

    monkeypatch.setattr(durable_change_train, "execute_durable_change_train", pending)
    with pytest.raises(RuntimeError, match="injected_train_pending"):
        initialize_active_archive_root(root)
    with pytest.raises(SchemaSkew):
        open_readonly_connection(root / "source.db", tier=ArchiveTier.SOURCE)
    with pytest.raises(SchemaSkew):
        ArchiveStore.open_source_tier_acquisition(root)
    with closing(sqlite3.connect(root / "source.db")) as conn:
        assert conn.execute("PRAGMA user_version").fetchone()[0] == 1


def test_released_source_train_admits_ordinary_acquisition_without_row_rescan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from polylogue.core.enums import Origin, Provider
    from polylogue.storage.sqlite import durable_change_train
    from polylogue.storage.sqlite.archive_tiers.bootstrap import (
        initialize_active_archive_root,
        invalidate_active_archive_bootstrap,
    )
    from polylogue.storage.sqlite.archive_tiers.source_write import write_source_raw_session

    initialize_active_archive_root(tmp_path)
    with closing(sqlite3.connect(tmp_path / "source.db")) as conn:
        raw_id = write_source_raw_session(
            conn,
            origin=Origin.CODEX_SESSION,
            capture_mode=Provider.CODEX,
            source_path="/synthetic/after-migration.jsonl",
            canonical_source_path="/synthetic/after-migration.jsonl",
            source_index=0,
            payload=b'{"synthetic_record":"after release"}\n',
            acquired_at_ms=2,
        )

    def refuse_row_scan(*args: object, **kwargs: object) -> object:
        raise AssertionError("released admission scanned mutable row evidence")

    monkeypatch.setattr(durable_change_train, "capture_durable_database_evidence", refuse_row_scan)
    monkeypatch.setattr(migration_runner, "_durable_table_counts", refuse_row_scan)
    monkeypatch.setattr(migration_runner, "_durable_literal_rows_digest", refuse_row_scan)
    invalidate_active_archive_bootstrap(tmp_path)
    initialize_active_archive_root(tmp_path)
    with closing(sqlite3.connect(tmp_path / "source.db")) as conn:
        assert conn.execute("SELECT raw_id FROM raw_sessions").fetchall() == [(raw_id,)]


@pytest.mark.parametrize("change", ("identity", "schema"))
def test_released_source_train_refuses_changed_physical_or_schema_identity(tmp_path: Path, change: str) -> None:
    from polylogue.storage.sqlite.archive_tiers.bootstrap import (
        initialize_active_archive_root,
        invalidate_active_archive_bootstrap,
    )
    from polylogue.storage.sqlite.migration_runner import DurableChangeTrainError

    initialize_active_archive_root(tmp_path)
    source = tmp_path / "source.db"
    if change == "identity":
        replacement = tmp_path / "replacement.db"
        with closing(sqlite3.connect(source)) as original, closing(sqlite3.connect(replacement)) as target:
            original.backup(target)
        source.unlink()
        replacement.replace(source)
    else:
        with closing(sqlite3.connect(source)) as conn:
            conn.execute("CREATE TABLE unproved_schema (value TEXT)")
            conn.commit()
    invalidate_active_archive_bootstrap(tmp_path)
    with pytest.raises(DurableChangeTrainError):
        initialize_active_archive_root(tmp_path)


def test_proven_source_train_recovery_rejects_changed_nul_suffix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import json

    from polylogue.storage.sqlite import durable_change_train
    from polylogue.storage.sqlite.archive_tiers.bootstrap import (
        initialize_active_archive_root,
        invalidate_active_archive_bootstrap,
    )
    from polylogue.storage.sqlite.migration_runner import DurableChangeTrainError, DurableChangeTrainState

    bootstrap_baseline_archive(tmp_path, monkeypatch)
    conn, raw_ids = source_baseline(tmp_path / "source.db")
    try:
        conn.execute("UPDATE raw_sessions SET native_id=? WHERE raw_id=?", ("native\x00before", raw_ids[0]))
        conn.commit()
    finally:
        conn.close()
    initialize_active_archive_root(tmp_path)
    manifest = (
        tmp_path
        / f".maintenance-state/durable-change-trains/source-{ARCHIVE_VERSION_BY_TIER[ArchiveTier.SOURCE]:03}.json"
    )
    train = durable_change_train.load_durable_change_train_manifest(manifest)
    assert train.proof is not None
    pending = replace(
        train,
        state=DurableChangeTrainState.PROVEN,
        released_at_ms=None,
        release_evidence_ref=None,
        proof_refs=train.proof.proof_refs,
    )
    manifest.write_text(json.dumps(migration_runner.durable_change_train_to_payload(pending)))
    with closing(sqlite3.connect(tmp_path / "source.db")) as conn:
        conn.execute("UPDATE raw_sessions SET native_id=? WHERE raw_id=?", ("native\x00after", raw_ids[0]))
        conn.commit()
    invalidate_active_archive_bootstrap(tmp_path)
    with pytest.raises(DurableChangeTrainError):
        initialize_active_archive_root(tmp_path)
    assert durable_change_train.load_durable_change_train_manifest(manifest).state is DurableChangeTrainState.PROVEN


def test_memory_probe_cancel_settles_actual_inventory_and_leaves_destination_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from polylogue.storage.io_phase_metrics import connection_cursor, native_connection_physically_closed
    from polylogue.storage.sqlite import durable_change_train
    from polylogue.storage.sqlite.archive_tiers import bootstrap
    from polylogue.storage.sqlite.connection_profile import retained_native_sql_owners_on_current_thread

    # The current probe computes canonical inventory before touching the
    # supplied destination. Force that real owner construction, not the
    # retired backup-to-TemporaryDirectory path or an unrelated cached answer.
    durable_change_train._canonical_schema_inventory_for_ddl.cache_clear()
    inventories: list[sqlite3.Connection] = []
    injected = KeyboardInterrupt("synthetic canonical inventory cancellation")

    def cancelled(connection: sqlite3.Connection, step: migration_runner.MigrationStep) -> None:
        inventories.append(connection)
        raise injected

    monkeypatch.setattr(migration_runner, "_execute_proved_migration_sql", cancelled)
    with closing(sqlite3.connect(":memory:")) as conn:
        with pytest.raises(KeyboardInterrupt) as failure:
            bootstrap.initialize_runtime_tier_probe(conn, ArchiveTier.SOURCE)
        assert failure.value is injected and inventories
        with connection_cursor(conn, "PRAGMA user_version") as cursor:
            assert cursor.fetchone()[0] == 0
        with connection_cursor(conn, "SELECT name FROM sqlite_schema") as cursor:
            assert cursor.fetchall() == []
    assert all(native_connection_physically_closed(connection) for connection in inventories)
    assert not any(owner.connection in inventories for owner in retained_native_sql_owners_on_current_thread())


@pytest.mark.parametrize("storage_class", ("text", "blob"))
def test_literal_row_proof_streams_large_cells_and_primary_keys_inside_write_transaction(storage_class: str) -> None:
    """Whole-value projection exceeds this page-bound allocation proof."""
    size = 8 * 1024 * 1024
    with closing(sqlite3.connect(":memory:")) as conn:
        conn.execute("CREATE TABLE large_cells (key TEXT PRIMARY KEY, value)")
        expression = "CAST(zeroblob(? - 2) || X'80FF' AS TEXT)" if storage_class == "text" else "zeroblob(?)"
        conn.execute(
            f"INSERT INTO large_cells VALUES (CAST(zeroblob(? - 2) || X'80FF' AS TEXT), {expression})",
            (size, size),
        )
        assert conn.in_transaction
        tracemalloc.start()
        try:
            original = migration_runner._durable_literal_rows_digest(conn)
            _retained, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
        assert peak < size // 4, (peak, size)
        # The original row is still uncommitted. A different connection or a
        # stale on-disk snapshot cannot prove its last literal byte.
        target_type = "TEXT" if storage_class == "text" else "BLOB"
        conn.execute(
            f"UPDATE large_cells SET value=CAST(substr(CAST(value AS BLOB), 1, ?) || X'01' AS {target_type})",
            (size - 1,),
        )
        assert migration_runner._durable_literal_rows_digest(conn) != original


def test_literal_row_proof_preserves_large_without_rowid_composite_keys() -> None:
    with closing(sqlite3.connect(":memory:")) as conn:
        conn.execute(
            "CREATE TABLE large_keys (a TEXT COLLATE NOCASE, b INTEGER, value BLOB, PRIMARY KEY(a,b)) WITHOUT ROWID"
        )
        conn.execute("INSERT INTO large_keys VALUES (CAST(zeroblob(131070) || X'80FF' AS TEXT), 2, zeroblob(196608))")
        original = migration_runner._durable_literal_rows_digest(conn)
        conn.execute("UPDATE large_keys SET value=CAST(substr(value,1,196607) || X'01' AS BLOB)")
        changed_payload = migration_runner._durable_literal_rows_digest(conn)
        assert changed_payload != original
        conn.execute("UPDATE large_keys SET a=CAST(zeroblob(131070) || X'81FF' AS TEXT)")
        assert migration_runner._durable_literal_rows_digest(conn) != changed_payload


def test_without_rowid_literal_proof_visits_each_large_key_once() -> None:
    """Doubling history must not repeat an ordered skipped-prefix scan per cell."""
    import sqlite3

    def work(rows: int) -> tuple[str, int]:
        with sqlite3.connect(":memory:") as conn:
            conn.execute("CREATE TABLE evidence (key TEXT PRIMARY KEY, payload BLOB) WITHOUT ROWID")
            for number in range(rows):
                conn.execute(
                    "INSERT INTO evidence VALUES(CAST(? || CAST(zeroblob(131072) AS TEXT) AS TEXT), zeroblob(131073))",
                    (f"{number:08d}",),
                )
            steps = 0

            def count() -> int:
                nonlocal steps
                steps += 1
                return 0

            conn.set_progress_handler(count, 1)
            try:
                result = migration_runner._durable_literal_rows_digest(conn)
            finally:
                conn.set_progress_handler(None, 0)
            assert conn.in_transaction
            assert not tuple(conn.execute("SELECT name FROM sqlite_temp_schema WHERE type='table'"))
            return result, steps

    first, small_work = work(16)
    second, large_work = work(32)
    assert first != second
    assert large_work < small_work * 3


def test_schema_rehearsal_reuses_only_bound_pure_inputs_across_physical_archives(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = 0
    actual = migration_runner._schema_only_replica

    def replica(source: sqlite3.Connection) -> sqlite3.Connection:
        nonlocal calls
        calls += 1
        return actual(source)

    monkeypatch.setattr(migration_runner, "_schema_only_replica", replica)
    proofs = []
    for number in (1, 2):
        conn, _raw_ids = source_baseline(tmp_path / f"source-{number}.db")
        try:
            proofs.append(
                migration_runner.rehearse_durable_migration_chain(
                    conn,
                    ArchiveTier.SOURCE,
                    target_version=ARCHIVE_VERSION_BY_TIER[ArchiveTier.SOURCE],
                    evidence_ref=f"proof:archive-{number}",
                )
            )
            assert conn.execute("PRAGMA user_version").fetchone()[0] == 1
        finally:
            conn.close()
    assert calls == 1
    assert proofs[0].chain_sha256 == proofs[1].chain_sha256
    assert proofs[0].evidence_ref != proofs[1].evidence_ref


def test_manifest_schema_reuse_preserves_every_payload_validation(monkeypatch: pytest.MonkeyPatch) -> None:
    import json
    from importlib import resources
    from typing import get_type_hints

    payload = json.loads(
        resources.files("polylogue.storage.sqlite.migrations.source").joinpath("002.train.json").read_text()
    )
    original = get_type_hints
    resolutions: list[type] = []

    def resolve(annotation: type) -> dict[str, object]:
        resolutions.append(annotation)
        return original(annotation)

    migration_runner._manifest_dataclass_hints.cache_clear()
    monkeypatch.setattr(migration_runner, "get_type_hints", resolve)
    try:
        expected = migration_runner.durable_change_train_from_payload(payload)
        assert migration_runner.durable_change_train_from_payload(payload) == expected
        assert resolutions
        assert len(resolutions) == len(set(resolutions))

        malformed = dict(payload)
        malformed["owner_ref"] = 7
        unsigned = dict(malformed)
        unsigned.pop("manifest_sha256")
        malformed["manifest_sha256"] = migration_runner._canonical_json_sha256(unsigned)
        with pytest.raises(migration_runner.DurableChangeTrainError, match="must be a string"):
            migration_runner.durable_change_train_from_payload(malformed)

        malformed = dict(payload)
        malformed["unrecognized"] = "field"
        unsigned = dict(malformed)
        unsigned.pop("manifest_sha256")
        malformed["manifest_sha256"] = migration_runner._canonical_json_sha256(unsigned)
        with pytest.raises(migration_runner.DurableChangeTrainError, match="fields differ"):
            migration_runner.durable_change_train_from_payload(malformed)
        assert len(resolutions) == len(set(resolutions))
    finally:
        migration_runner._manifest_dataclass_hints.cache_clear()


def test_source002_retains_large_marker_payloads_with_incremental_row_evidence(tmp_path: Path) -> None:
    from polylogue.storage.accepted_marker_inputs import (
        persist_pending_marker_input_sync,
        prepare_accepted_marker_input,
    )

    path = tmp_path / "source.db"
    conn, raw_ids = source_baseline(path)
    size = 8 * 1024 * 1024
    try:
        pending = prepare_accepted_marker_input(raw_ids[0], [], request_facts={"synthetic_metadata": "x" * size})
        accepted = prepare_accepted_marker_input(raw_ids[1], [], request_facts={"synthetic_metadata": "x" * size})
        persist_pending_marker_input_sync(
            conn,
            pending,
            expected_incarnation_id="00000000-0000-0000-0000-000000000001",
        )
        conn.execute("INSERT INTO accepted_marker_stream VALUES(1, ?)", ("synthetic-source-stream",))
        conn.execute(
            "INSERT INTO accepted_marker_inputs(identity, raw_id, payload, payload_sha256) VALUES(?, ?, ?, ?)",
            (accepted.identity, accepted.raw_id, accepted.payload, accepted.payload_sha256),
        )
        lengths = (len(accepted.payload), len(pending.payload))
        del accepted, pending
        conn.commit()
        before = migration_runner._durable_literal_rows_digest(conn)
        # This oracle measures incremental row evidence, independently of test
        # order. Resolve the fixed train schema/import closure before tracing;
        # cold API imports and native SQLite allocations are separate contracts.
        migration_runner.durable_preparation_fingerprint(conn, ArchiveTier.SOURCE)
        tracemalloc.start()
        try:
            result = migrate_archive_tier(conn, ArchiveTier.SOURCE, target_version=2, backup_manifest=None)
            _retained, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
        assert result.applied_versions == (2,)
        assert peak < size // 4, (peak, size)
        assert migration_runner._durable_literal_rows_digest(conn) == before
    finally:
        conn.close()
    with closing(sqlite3.connect(path)) as reopened:
        assert reopened.execute("PRAGMA user_version").fetchone()[0] == 2
        assert reopened.execute("SELECT length(payload) FROM accepted_marker_inputs").fetchall() == [(lengths[0],)]
        assert reopened.execute("SELECT length(payload) FROM pending_accepted_marker_inputs").fetchall() == [
            (lengths[1],)
        ]


@pytest.mark.parametrize("rows_as_mapping", (False, True))
@pytest.mark.parametrize("text_as_bytes", (False, True))
def test_source_rehearsal_configuration_ignores_caller_result_factories(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, rows_as_mapping: bool, text_as_bytes: bool
) -> None:
    monkeypatch.setattr(migration_runner, "_SCHEMA_REHEARSAL_CACHE", {})
    conn, _raw_ids = source_baseline(tmp_path / "source.db")
    try:
        expected = migration_runner.durable_preparation_fingerprint(conn, ArchiveTier.SOURCE)
        conn.row_factory = sqlite3.Row if rows_as_mapping else None
        conn.text_factory = bytes if text_as_bytes else str
        assert migration_runner.durable_preparation_fingerprint(conn, ArchiveTier.SOURCE) == expected
        proof = migration_runner.rehearse_durable_migration_chain(
            conn,
            ArchiveTier.SOURCE,
            target_version=ARCHIVE_VERSION_BY_TIER[ArchiveTier.SOURCE],
            evidence_ref="proof:configured-source",
        )
        assert proof.target_version == ARCHIVE_VERSION_BY_TIER[ArchiveTier.SOURCE]
        assert conn.row_factory is (sqlite3.Row if rows_as_mapping else None)
        assert conn.text_factory is (bytes if text_as_bytes else str)
    finally:
        conn.close()


@pytest.mark.parametrize("generated", ["VIRTUAL", "STORED"])
@pytest.mark.parametrize(
    "literal", ["NULL", "''", "X''", "CAST(zeroblob(196606) || X'80FF' AS TEXT)", "zeroblob(196608)"]
)
def test_generated_table_literal_proof_preserves_exact_cells(generated: str, literal: str) -> None:
    """A generated column prevents Blob.open even on ordinary stored fields."""
    from polylogue.storage.sqlite.managed_connection import sqlite_connection

    with sqlite_connection(":memory:") as computed, sqlite_connection(":memory:") as explicit:
        computed.execute(
            f"CREATE TABLE evidence(key TEXT PRIMARY KEY, payload, derived GENERATED ALWAYS AS (payload) {generated})"
        ).close()
        explicit.execute("CREATE TABLE evidence(key TEXT PRIMARY KEY, payload, derived)").close()
        computed.execute(f"INSERT INTO evidence(key,payload) VALUES('synthetic-key', {literal})").close()
        explicit.execute(f"INSERT INTO evidence VALUES('synthetic-key', {literal}, {literal})").close()
        original = migration_runner._durable_literal_rows_digest(computed)
        assert original == migration_runner._durable_literal_rows_digest(explicit)
        assert computed.in_transaction and explicit.in_transaction
        computed.execute("UPDATE evidence SET payload=CAST(zeroblob(196607) || X'01' AS BLOB)").close()
        assert migration_runner._durable_literal_rows_digest(computed) != original
        from polylogue.storage.io_phase_metrics import _MeasuredConnection

        assert isinstance(computed, _MeasuredConnection)
        assert not computed.live_cursors()


@pytest.mark.parametrize("execute_failure", [False, True])
def test_generated_migration_literal_fault_keeps_actual_cursor_until_creator_retry(
    monkeypatch: pytest.MonkeyPatch, execute_failure: bool
) -> None:
    from builtins import BaseExceptionGroup
    from typing import Any

    from polylogue.storage.io_phase_metrics import _MeasuredConnection
    from polylogue.storage.sqlite.connection_profile import (
        NativeConnectionSettlementError,
        retained_native_sql_owners_on_current_thread,
    )
    from polylogue.storage.sqlite.managed_connection import sqlite_connection
    from tests.infra.sqlite_cursor_settlement import ControlledCursor

    primary = OSError("synthetic migration literal read failure")
    blocked: list[ControlledCursor] = []
    connection: sqlite3.Connection | None = None

    class FaultCursor(ControlledCursor):
        def execute(self, sql: str, parameters: Any = ()) -> FaultCursor:
            result = super().execute(sql, parameters)
            if sql.startswith('SELECT substr(CAST("payload" AS BLOB)'):
                self.allow_cleanup.clear()
                blocked.append(self)
                if execute_failure:
                    raise primary
            return result

    try:
        with pytest.raises((NativeConnectionSettlementError, BaseExceptionGroup)) as failure:
            with sqlite_connection(":memory:") as actual:
                connection = actual
                actual.execute("CREATE TABLE evidence(payload TEXT, derived AS (payload))").close()
                actual.execute("INSERT INTO evidence(payload) VALUES (?)", ("λ" * 100000,)).close()
                original_cursor = actual.cursor
                monkeypatch.setattr(actual, "cursor", lambda: original_cursor(factory=FaultCursor))
                migration_runner._durable_literal_rows_digest(actual)
        owners = tuple(
            owner for owner in retained_native_sql_owners_on_current_thread() if owner.connection is connection
        )
        assert len(owners) == 1
        owner = owners[0]
        assert isinstance(connection, _MeasuredConnection)
        assert blocked and all(id(cursor) in connection._unsettled_native_cursors for cursor in blocked)
        if execute_failure:
            pending: list[BaseException] = [failure.value]
            found = False
            visited: set[int] = set()
            while pending:
                error = pending.pop()
                if id(error) in visited:
                    continue
                visited.add(id(error))
                found |= error is primary
                if isinstance(error, BaseExceptionGroup):
                    pending.extend(error.exceptions)
                for nested in (error.__cause__, error.__context__, getattr(error, "failure", None)):
                    if isinstance(nested, BaseException):
                        pending.append(nested)
            assert found
        with pytest.raises(NativeConnectionSettlementError):
            owner.close()
        for cursor in blocked:
            cursor.allow_cleanup.set()
        owner.close()
        assert owner.connection is None
    finally:
        for cursor in blocked:
            cursor.allow_cleanup.set()
        for owner in retained_native_sql_owners_on_current_thread():
            if owner.connection is connection:
                owner.close()
