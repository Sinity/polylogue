"""Row-preserving index replacement proofs, with live rollback and row parity.

The archive currently ships no numbered Source migration, so the replacement
under test is a synthetic step that widens the Source baseline's raw-artifact
failure partition by one kind, the shape a future partition change would take.
"""

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
from polylogue.storage.sqlite.migration_runner import MigrationError
from tests.infra.index_replacement import missing_coordinates_artifact, source_baseline


def _synthetic_partition_replacement() -> migration_runner.MigrationStep:
    """Replace both baseline raw-artifact partitions, moving one new kind to failures."""
    from polylogue.storage.sqlite.archive_tiers.source import SOURCE_DDL

    statements = []
    for name in ("idx_raw_artifacts_source_identity", "idx_raw_artifacts_failure_identity"):
        start = SOURCE_DDL.index(f"CREATE UNIQUE INDEX IF NOT EXISTS {name}")
        create = SOURCE_DDL[start : SOURCE_DDL.index(";", start) + 1]
        widened = create.replace(
            "    'terminal_missing_source_coordinates',\n",
            "    'terminal_missing_source_coordinates',\n    'terminal_synthetic_partition_probe',\n",
        )
        assert widened != create
        statements.append(f"DROP INDEX {name};\n{widened.replace(' IF NOT EXISTS', '')}")
    return migration_runner.MigrationStep(
        tier=ArchiveTier.SOURCE,
        version=2,
        name="002_synthetic_partition_probe.sql",
        sql=f"{migration_runner._INDEX_REPLACEMENT_MARKER}\n" + "\n\n".join(statements) + "\n",
        requires_backup=False,
    )


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


def test_runtime_probe_builds_source_partitions_without_train_release(
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


def test_canonical_birth_marker_stays_baseline_after_reopen(tmp_path: Path) -> None:
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


def test_baseline_keeps_each_windowless_raw_refusal_after_restart(tmp_path: Path) -> None:
    """A coordinate-keyed failure partition would refuse the second raw's refusal."""
    path = tmp_path / "source.db"
    conn, raw_ids = source_baseline(path)
    try:
        for raw_id in raw_ids:
            upsert_raw_artifact(conn, raw_id, missing_coordinates_artifact(raw_id))
        conn.commit()
    finally:
        conn.close()
    with closing(sqlite3.connect(path)) as restarted:
        assert restarted.execute("PRAGMA user_version").fetchone()[0] == ARCHIVE_VERSION_BY_TIER[ArchiveTier.SOURCE]
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
        step = _synthetic_partition_replacement()
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


def test_true_partition_replacement_is_proved_and_keeps_rows(tmp_path: Path) -> None:
    """Anti-vacuity for the rollbacks above: the unmutated step itself is admitted."""
    conn, raw_ids = source_baseline(tmp_path / "source.db")
    try:
        for raw_id in raw_ids:
            upsert_raw_artifact(conn, raw_id, missing_coordinates_artifact(raw_id))
        digest = migration_runner._durable_literal_rows_digest(conn)
        conn.execute("BEGIN IMMEDIATE")
        with conn:
            migration_runner._execute_proved_migration_sql(conn, _synthetic_partition_replacement())
        assert migration_runner._durable_literal_rows_digest(conn) == digest
        assert all(
            "'terminal_synthetic_partition_probe'" in row[0]
            for row in conn.execute(
                "SELECT sql FROM sqlite_schema WHERE name IN "
                "('idx_raw_artifacts_source_identity', 'idx_raw_artifacts_failure_identity')"
            )
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
        step = _synthetic_partition_replacement()
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
