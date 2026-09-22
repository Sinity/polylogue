"""``assertions.status`` is NOT NULL in canonical DDL and reached by copy-forward.

``status TEXT DEFAULT 'active'`` was nullable. No production writer could put a
NULL there -- ``upsert_assertion`` and ``mark_assertion_status`` both route
through ``_normalize_assertion_status``, which resolves an absent status to
``ASSERTION_DEFAULT_STATUS`` -- so the nullability was reachable only by a raw
SQL write, and readers compensated for it with ``COALESCE(status, 'active')``.
``user.db`` is durable and irreplaceable and is never rebuilt from source
evidence, so fresh DDL alone would leave every existing archive nullable. Slot
002 carries the shape forward.

The vocabulary half stays where repo policy puts it. ``devtools gate
durable-enum-checks`` refuses any durable-tier membership list whose member set
equals a reachable enum's values, so an ``AssertionStatus``-generated CHECK
cannot live in ``USER_DDL``; ``test_durable_ddl_pins_no_status_vocabulary``
pins that refutation so a later lane does not re-add one, and the migration's
retention of an out-of-vocabulary legacy status is what that buys.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import closing
from pathlib import Path

import pytest

from devtools import verify_durable_enum_checks
from polylogue.core.enums import AssertionStatus
from polylogue.storage.sqlite import migration_runner
from polylogue.storage.sqlite.archive_tiers import ARCHIVE_VERSION_BY_TIER
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_database
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from polylogue.storage.sqlite.archive_tiers.user import USER_DDL
from polylogue.storage.sqlite.archive_tiers.user_write import (
    mark_assertion_status,
    read_assertion_envelope,
    upsert_assertion,
)
from polylogue.storage.sqlite.durable_change_train import (
    _runtime_consumer_results,
    validate_durable_migration_sidecars,
)

_ASSERTION_COLUMN_COUNT = 18

#: The nullable shape every pre-002 user tier carries, reconstructed from
#: canonical DDL rather than hand-copied, so a later column addition cannot
#: leave this fixture silently describing a table that never existed.
_NULLABLE_STATUS_COLUMN = "    status              TEXT DEFAULT 'active',"
_NOT_NULL_STATUS_COLUMN = "    status              TEXT NOT NULL DEFAULT 'active',"


def _historical_user_ddl() -> str:
    assert USER_DDL.count(_NOT_NULL_STATUS_COLUMN) == 1, "canonical DDL no longer declares status NOT NULL"
    return USER_DDL.replace(_NOT_NULL_STATUS_COLUMN, _NULLABLE_STATUS_COLUMN)


def _seed_rows(conn: sqlite3.Connection) -> None:
    """Three rows spanning every disposition the copy-forward must answer."""
    conn.executemany(
        "INSERT INTO assertions VALUES (" + ",".join("?" * _ASSERTION_COLUMN_COUNT) + ")",
        [
            # An absent status: what the declared DEFAULT and every reader
            # already resolve to 'active'.
            (
                "null-status",
                None,
                "session:1",
                "k1",
                "note",
                None,
                "b1",
                "user:local",
                "user",
                "[]",
                None,
                "private",
                None,
                None,
                '{"inject":false}',
                "[]",
                10,
                11,
            ),
            # A status outside AssertionStatus: retained verbatim, never coerced
            # and never dropped.
            (
                "legacy-status",
                None,
                "session:2",
                "k2",
                "note",
                None,
                "b2",
                "user:local",
                "user",
                "[]",
                "archived",
                "private",
                0.5,
                None,
                '{"inject":false}',
                "[]",
                20,
                21,
            ),
            # An ordinary in-vocabulary status.
            (
                "candidate-status",
                None,
                "session:3",
                "k3",
                "note",
                None,
                "b3",
                "user:local",
                "user",
                "[]",
                AssertionStatus.CANDIDATE.value,
                "private",
                None,
                None,
                '{"inject":false}',
                "[]",
                30,
                31,
            ),
        ],
    )


def _build_pre_migration_user_tier(path: Path) -> None:
    """Write a v1 user tier carrying the nullable status column and its rows."""
    with closing(sqlite3.connect(path)) as conn:
        conn.executescript(_historical_user_ddl())
        _seed_rows(conn)
        conn.execute("PRAGMA user_version = 1")
        conn.commit()


@pytest.fixture
def fresh_user(tmp_path: Path) -> Iterator[sqlite3.Connection]:
    """A user tier built by the production bootstrap route and nothing else."""
    path = tmp_path / "user.db"
    initialize_archive_database(path, ArchiveTier.USER)
    connection = sqlite3.connect(path)
    try:
        yield connection
    finally:
        connection.close()


def test_fresh_schema_refuses_a_null_status(fresh_user: sqlite3.Connection) -> None:
    """Canonical DDL makes an absent status unrepresentable, not merely unusual.

    Anti-vacuity: restoring ``status TEXT DEFAULT 'active'`` in ``USER_DDL``
    makes this INSERT succeed and the ``pytest.raises`` block fail. A test that
    only read the column back through the ordinary writer would stay green
    under that mutation, because the writer never produces the NULL in the
    first place -- which is why this goes through raw SQL.
    """
    with pytest.raises(sqlite3.IntegrityError, match="NOT NULL constraint failed: assertions.status"):
        fresh_user.execute(
            "INSERT INTO assertions VALUES (" + ",".join("?" * _ASSERTION_COLUMN_COUNT) + ")",
            (
                "raw-null",
                None,
                "session:x",
                None,
                "note",
                None,
                None,
                "user:local",
                "user",
                "[]",
                None,
                "private",
                None,
                None,
                '{"inject":false}',
                "[]",
                1,
                2,
            ),
        )
    assert ARCHIVE_VERSION_BY_TIER[ArchiveTier.USER] > 1


def test_ordinary_write_resolves_an_absent_status(fresh_user: sqlite3.Connection) -> None:
    """The opposite direction: NOT NULL must not turn an ordinary write into a refusal.

    Anti-vacuity: a blanket rejection of the column -- a CHECK that admits
    nothing, or a writer that stopped resolving ``status=None`` -- fails here
    while leaving :func:`test_fresh_schema_refuses_a_null_status` green, so the
    pair pins the constraint rather than a refusal.
    """
    envelope = upsert_assertion(
        fresh_user,
        assertion_id="ordinary-write",
        target_ref="session:ordinary",
        kind="note",
        body_text="no status supplied",
        author_ref="user:test",
        author_kind="user",
        now_ms=1_780_000_000_000,
    )
    assert envelope.status is AssertionStatus.ACTIVE
    assert fresh_user.execute(
        "SELECT status FROM assertions WHERE assertion_id = ?", ("ordinary-write",)
    ).fetchone() == (AssertionStatus.ACTIVE.value,)

    assert mark_assertion_status(fresh_user, "ordinary-write", AssertionStatus.SUPERSEDED) is True
    reread = read_assertion_envelope(fresh_user, "ordinary-write")
    assert reread is not None
    assert reread.status is AssertionStatus.SUPERSEDED


def test_durable_ddl_pins_no_status_vocabulary() -> None:
    """``assertions.status`` carries no membership CHECK, generated or otherwise.

    ``polylogue-lbk1`` was scoped to generate the CHECK from ``AssertionStatus``
    via ``check()``/``nullable_check()``. At this head that is refused by
    ``devtools gate durable-enum-checks``: a durable membership list whose
    member set equals a reachable enum's values pins that enum's evolution into
    durable DDL, so every later token would become a durable migration. The
    vocabulary is owned at the write boundary instead.

    Anti-vacuity: adding ``CHECK ({check('status', AssertionStatus)})`` to the
    ``assertions`` DDL makes the second assertion report an
    ``EnumCheckViolation`` for ``status``/``AssertionStatus`` and turns this
    red -- the same finding ``gate durable-enum-checks`` reports.
    """
    status_declaration = next(line for line in USER_DDL.splitlines() if line.strip().startswith("status "))
    assert "CHECK" not in status_declaration.upper()
    assert verify_durable_enum_checks.scan_ddl_for_enum_membership_checks(USER_DDL, tier="user") == []


def test_migration_is_refused_without_a_verified_backup(tmp_path: Path) -> None:
    """A table rebuild is not additive; slot 002 may not run unbacked.

    Anti-vacuity: adding ``-- migration-safety: additive-no-backup`` to
    ``002_assertions_status_not_null.sql`` makes the migration run here instead
    of refusing, and the sidecar's ``requires_backup`` binding then fails its
    own validation.
    """
    path = tmp_path / "user.db"
    _build_pre_migration_user_tier(path)
    with closing(sqlite3.connect(path)) as conn:
        with pytest.raises(migration_runner.MigrationError, match="requires a verified backup manifest"):
            migration_runner.migrate_archive_tier(conn, ArchiveTier.USER, backup_manifest=None)
        assert conn.execute("PRAGMA user_version").fetchone() == (1,)


def test_existing_archive_is_refused_before_the_migration(tmp_path: Path) -> None:
    """A v1 user tier is not silently opened by a runtime that expects v2.

    Anti-vacuity: leaving ``ARCHIVE_VERSION_BY_TIER[USER]`` at the floor makes
    this open succeed, and every existing archive would then keep the nullable
    column with nothing reporting it.
    """
    path = tmp_path / "user.db"
    _build_pre_migration_user_tier(path)
    with pytest.raises(Exception, match="older than the current user tier version"):
        initialize_archive_database(path, ArchiveTier.USER, allow_create=False)


def _apply_numbered_user_migrations(conn: sqlite3.Connection) -> tuple[int, ...]:
    """Apply every checked-in user slot through the production loader.

    ``migrate_archive_tier`` is the ordinary route and is exercised for its
    refusal by :func:`test_migration_is_refused_without_a_verified_backup`, but
    it cannot be driven to completion here: slot 002 requires a verified backup
    manifest, and ``backup_archive(profile="full_evidence", verify=True)`` fails
    on any archive whose ``source.db`` is in WAL mode -- the state
    ``initialize_active_archive_root`` leaves it in. The verifier's own restore
    opens the restored ``source.db``, SQLite writes ``restore/source.db-shm``,
    and the backup's artifact inventory then refuses it as "an unbound SQLite
    sidecar". That reproduces on a pristine archive at the base commit with no
    part of this change present, so it is not worked around here; forcing the
    source tier out of WAL to make the bundle verify would be a harness that is
    more permissive than production and would hide exactly that defect.

    ``_load_migrations`` is the same discovery the runner uses, including
    sidecar validation, so the SQL executed below is the shipped slot and not a
    copy of it.
    """
    steps = migration_runner._load_migrations(ArchiveTier.USER)
    applied: list[int] = []
    for step in steps:
        conn.executescript(step.sql)
        conn.execute(f"PRAGMA user_version = {step.version}")
        applied.append(step.version)
    conn.commit()
    return tuple(applied)


def test_slot_002_is_the_only_user_migration_and_requires_a_backup() -> None:
    """The shipped chain is one contiguous backup-gated slot above the floor.

    Anti-vacuity: adding ``-- migration-safety: additive-no-backup`` to the SQL
    flips ``requires_backup`` and makes both this and the sidecar's own binding
    validation fail; renaming or renumbering the file breaks the sidecar slot
    check inside ``validate_durable_migration_sidecars``.
    """
    steps = migration_runner._load_migrations(ArchiveTier.USER)
    assert [(step.version, step.name, step.requires_backup) for step in steps] == [
        (2, "002_assertions_status_not_null.sql", True)
    ]
    sidecars = validate_durable_migration_sidecars(ArchiveTier.USER, tuple((step.name, step.sql) for step in steps))
    assert [sidecar.slot for sidecar in sidecars] == [2]
    train = sidecars[0].train
    assert train.tier is ArchiveTier.USER
    assert (train.current_version, train.target_version) == (1, 2)
    assert train.migration.requires_backup is True
    assert train.backup_plan_ref
    assert {constraint.object_ref for constraint in train.drop_constraints} == {
        "table:assertions",
        "index:idx_assertions_target_kind",
        "index:idx_assertions_kind_status_updated",
        "index:idx_assertions_target_kind_status_visibility",
        "index:idx_assertions_scope_kind_status",
        "trigger:query_unit_frame_assertions_insert",
        "trigger:query_unit_frame_assertions_update",
        "trigger:query_unit_frame_assertions_delete",
    }


def test_every_rider_consumer_has_a_working_probe(tmp_path: Path) -> None:
    """The train's declared runtime consumers resolve and behave.

    ``_runtime_consumer_results`` is the production prove-step dispatch. Without
    a probe adapter for each ``production_ref`` it refuses with "no durable
    probe adapter", so this is what binds the sidecar's behavior proofs to real
    writer behavior rather than to a name.

    Anti-vacuity: deleting the ``:upsert_assertion`` branch from
    ``durable_change_train._runtime_consumer_results`` turns this red with that
    exact refusal; making ``upsert_assertion`` pass the caller's ``None`` status
    through turns it red on the probe's own NULL check.
    """
    steps = migration_runner._load_migrations(ArchiveTier.USER)
    sidecars = validate_durable_migration_sidecars(ArchiveTier.USER, tuple((step.name, step.sql) for step in steps))
    results = _runtime_consumer_results(sidecars[0].train, tmp_path)
    assert {result.consumer_id for result in results} == {
        "assertion-upsert-writer",
        "assertion-status-marker",
    }
    assert all(result.passed for result in results)


def test_copy_forward_preserves_every_row(tmp_path: Path) -> None:
    """Slot 002 migrates an existing archive forward with its rows intact.

    Three dispositions, all proven on the same archive: an absent status
    becomes the default it already meant, an out-of-vocabulary legacy status
    survives byte-identically, and an ordinary status is untouched. Row counts,
    the query-unit frame epoch, and every index and trigger survive, and the
    migrated schema equals canonical fresh DDL after normalization.

    Anti-vacuity: replacing the migration's ``COALESCE(status, 'active')`` with
    a bare ``status`` makes the copy fail on the NOT NULL column instead of
    carrying the row forward; dropping the index/trigger recreation makes the
    fresh-DDL parity proof report ``missing_objects``; letting the epoch
    triggers survive the rebuild makes the epoch assertion fail because the
    row copy would bump it once per row.
    """
    user_path = tmp_path / "user.db"
    _build_pre_migration_user_tier(user_path)

    with closing(sqlite3.connect(user_path)) as conn:
        before_rows = conn.execute("SELECT count(*) FROM assertions").fetchone()
        before_epoch = conn.execute("SELECT epoch FROM query_unit_frame_state").fetchone()

        assert _apply_numbered_user_migrations(conn) == (2,)

        assert conn.execute("SELECT count(*) FROM assertions").fetchone() == before_rows
        assert conn.execute("SELECT epoch FROM query_unit_frame_state").fetchone() == before_epoch
        assert conn.execute("SELECT assertion_id, status FROM assertions ORDER BY assertion_id").fetchall() == [
            ("candidate-status", AssertionStatus.CANDIDATE.value),
            ("legacy-status", "archived"),
            ("null-status", AssertionStatus.ACTIVE.value),
        ]
        # Everything else about the carried-forward rows is untouched.
        assert conn.execute(
            "SELECT target_ref, key, body_text, confidence, created_at_ms, updated_at_ms "
            "FROM assertions WHERE assertion_id = ?",
            ("legacy-status",),
        ).fetchone() == ("session:2", "k2", "b2", 0.5, 20, 21)

        assert conn.execute("PRAGMA user_version").fetchone() == (ARCHIVE_VERSION_BY_TIER[ArchiveTier.USER],)
        with pytest.raises(sqlite3.IntegrityError, match="NOT NULL constraint failed: assertions.status"):
            conn.execute("UPDATE assertions SET status = NULL WHERE assertion_id = ?", ("null-status",))

        surviving = {
            (str(row[0]), str(row[1]))
            for row in conn.execute(
                "SELECT type, name FROM sqlite_schema "
                "WHERE tbl_name = 'assertions' AND type IN ('index', 'trigger') "
                "AND name NOT LIKE 'sqlite_%'"
            ).fetchall()
        }
        assert surviving == {
            ("index", "idx_assertions_target_kind"),
            ("index", "idx_assertions_kind_status_updated"),
            ("index", "idx_assertions_target_kind_status_visibility"),
            ("index", "idx_assertions_scope_kind_status"),
            ("trigger", "query_unit_frame_assertions_insert"),
            ("trigger", "query_unit_frame_assertions_update"),
            ("trigger", "query_unit_frame_assertions_delete"),
        }
        # The implicit PRIMARY KEY index came back with the rebuilt table.
        with pytest.raises(sqlite3.IntegrityError, match="UNIQUE constraint failed: assertions.assertion_id"):
            conn.execute(
                "INSERT INTO assertions (assertion_id, target_ref, kind, created_at_ms, updated_at_ms) "
                "VALUES (?, ?, ?, ?, ?)",
                ("legacy-status", "session:dup", "note", 1, 1),
            )

        with closing(sqlite3.connect(":memory:")) as fresh:
            fresh.executescript(USER_DDL)
            fresh.execute(f"PRAGMA user_version = {ARCHIVE_VERSION_BY_TIER[ArchiveTier.USER]}")
            parity = migration_runner.prove_durable_fresh_ddl_parity(
                ArchiveTier.USER,
                ARCHIVE_VERSION_BY_TIER[ArchiveTier.USER],
                migrated_connection=conn,
                fresh_connection=fresh,
                evidence_ref="proof:assertion-status-not-null:fresh-ddl-parity",
            )
        assert parity.matches, (parity.missing_objects, parity.unexpected_objects, parity.changed_objects)

        # The ordinary writer still works on the migrated tier, and the epoch
        # trigger the rebuild recreated still fires.
        upsert_assertion(
            conn,
            assertion_id="post-migration",
            target_ref="session:post",
            kind="note",
            body_text="written after the copy-forward",
            author_ref="user:test",
            author_kind="user",
            now_ms=1_780_000_000_000,
        )
        assert conn.execute("SELECT epoch FROM query_unit_frame_state").fetchone() != before_epoch

    initialize_archive_database(user_path, ArchiveTier.USER, allow_create=False)
