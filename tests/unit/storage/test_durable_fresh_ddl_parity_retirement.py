"""Fresh-vs-migrated parity exempts declared retirements at every grain.

``RETIRED_SOURCE_SCHEMA_OBJECTS`` names objects a fresh source generation no
longer declares while a migrated historical tier still carries them. The
parity proof must report each one as retired -- not as an unexpected or
changed object -- or the first pre-floor source tier that reaches the runtime
is refused for carrying exactly what the retirement declared it would carry.

The set is hand-maintained beside the DDL, so both grains are pinned here:

* table/index/trigger/view members are compared by plain set intersection
  against ``DurableSchemaObjectEvidence.object_ref``, which is unqualified
  (``table:x``, not ``source:table:x``). Qualifying either operand empties the
  intersection silently.
* ``column:<table>.<name>`` members have no object of their own in the
  inventory -- a retained column instead moves the owning *table*'s digest --
  so they are projected out of the migrated table before the digests are
  compared.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import closing, contextmanager
from pathlib import Path

import pytest

from polylogue.storage.sqlite.archive_tiers import ARCHIVE_VERSION_BY_TIER
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_database
from polylogue.storage.sqlite.archive_tiers.source import (
    RETIRED_SOURCE_SCHEMA_OBJECTS,
    SOURCE_DDL,
)
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier

SOURCE_TIER_VERSION = ARCHIVE_VERSION_BY_TIER[ArchiveTier.SOURCE]
from polylogue.storage.sqlite.migration_runner import (
    DurableFreshDDLParityProof,
    MigrationError,
    capture_durable_database_evidence,
    capture_durable_schema_inventory,
    prove_durable_fresh_ddl_parity,
)

#: A real member of the declared set at the column grain: the raw-authority
#: parser census kept ``censused_at_ms`` as a wall clock that both writers
#: bound to the literal 0 (polylogue-48bos). This is the first entry that made
#: the exemption load-bearing -- a fixture with an empty retirement set proves
#: nothing, because an empty set is exactly the state in which the mismatch
#: was invisible.
RETIRED_COLUMN_TABLE = "raw_authority_parser_census"
RETIRED_COLUMN_NAME = "censused_at_ms"

#: A real member at the object grain.
RETIRED_TABLE_NAME = "raw_authority_censuses"


@contextmanager
def _fresh_source() -> Iterator[sqlite3.Connection]:
    """A source tier exactly as current canonical DDL builds it."""
    with closing(sqlite3.connect(":memory:")) as conn:
        conn.executescript(SOURCE_DDL)
        conn.execute(f"PRAGMA user_version = {SOURCE_TIER_VERSION}")
        conn.commit()
        yield conn


def _prove(migrated: sqlite3.Connection, fresh: sqlite3.Connection) -> DurableFreshDDLParityProof:
    return prove_durable_fresh_ddl_parity(
        ArchiveTier.SOURCE,
        SOURCE_TIER_VERSION,
        migrated_connection=migrated,
        fresh_connection=fresh,
        evidence_ref="proof:retirement-parity-test",
    )


def test_declared_retirements_use_the_inventory_ref_shape() -> None:
    """Every hand-written member matches the shape parity actually compares.

    Anti-vacuity: qualifying the declaration with its tier (``source:table:x``)
    or moving to a grain the inventory never emits turns this red.
    """
    assert RETIRED_SOURCE_SCHEMA_OBJECTS, "the exemption is untestable while the declared set is empty"
    with _fresh_source() as fresh:
        emitted_kinds = {item.object_type for item in capture_durable_schema_inventory(fresh).objects}
    assert emitted_kinds <= {"table", "index", "trigger", "view"}
    assert "column" not in emitted_kinds

    for ref in RETIRED_SOURCE_SCHEMA_OBJECTS:
        kind, _, remainder = ref.partition(":")
        assert remainder, f"{ref} is not a <kind>:<name> ref"
        assert kind in {"table", "index", "trigger", "view", "column"}, f"{ref} names an unknown grain"
        assert kind != ArchiveTier.SOURCE.value, f"{ref} carries a tier prefix the inventory never emits"
        if kind == "column":
            table_name, dot, column_name = remainder.partition(".")
            assert dot and table_name and column_name, f"{ref} is not a column:<table>.<name> ref"


def test_a_migrated_tier_keeping_a_retired_table_is_not_an_unexpected_object() -> None:
    """A retired table left behind by migration is exempt, and parity holds.

    Anti-vacuity: prefixing either operand of the exemption intersection with
    the tier -- ``source:table:raw_authority_censuses`` on the declaration side,
    or a tier-qualified ``object_ref`` on the inventory side -- empties the
    intersection and drops the table into ``unexpected_objects``.
    """
    assert f"table:{RETIRED_TABLE_NAME}" in RETIRED_SOURCE_SCHEMA_OBJECTS
    with _fresh_source() as fresh, _fresh_source() as migrated:
        migrated.execute(f"CREATE TABLE {RETIRED_TABLE_NAME} (census_id TEXT PRIMARY KEY) STRICT")
        migrated.commit()
        proof = _prove(migrated, fresh)

    assert proof.unexpected_objects == ()
    assert proof.changed_objects == ()
    assert proof.missing_objects == ()
    assert proof.matches is True


def test_a_migrated_tier_keeping_a_retired_column_has_fresh_ddl_parity() -> None:
    """A retained retired column does not make its table a changed object.

    A column is not its own inventory object, so the retirement surfaces as a
    moved digest on ``raw_authority_parser_census`` itself.

    Anti-vacuity: dropping the ``retired_columns=`` projection from
    ``prove_durable_fresh_ddl_parity`` puts ``table:raw_authority_parser_census``
    back into ``changed_objects`` and makes ``matches`` false.
    """
    assert f"column:{RETIRED_COLUMN_TABLE}.{RETIRED_COLUMN_NAME}" in RETIRED_SOURCE_SCHEMA_OBJECTS
    with _fresh_source() as fresh, _fresh_source() as migrated:
        migrated.execute(
            f"ALTER TABLE {RETIRED_COLUMN_TABLE} "
            f"ADD COLUMN {RETIRED_COLUMN_NAME} INTEGER NOT NULL DEFAULT 0 CHECK({RETIRED_COLUMN_NAME} >= 0)"
        )
        migrated.commit()
        retained = {str(row[1]) for row in migrated.execute(f"PRAGMA table_xinfo({RETIRED_COLUMN_TABLE})")}
        assert RETIRED_COLUMN_NAME in retained, "the fixture must actually carry the retired column"
        proof = _prove(migrated, fresh)

    assert proof.changed_objects == ()
    assert proof.unexpected_objects == ()
    assert proof.missing_objects == ()
    assert proof.matches is True
    assert proof.parity_inventory_sha256 == proof.fresh_inventory_sha256
    assert proof.migrated_inventory_sha256 != proof.parity_inventory_sha256


def test_the_retired_column_projection_is_not_a_blanket_table_exemption() -> None:
    """An undeclared extra column on the same table still fails parity.

    This is the mutation that would make the test above vacuous: an exemption
    that ignored the owning table wholesale would pass here too.
    """
    with _fresh_source() as fresh, _fresh_source() as migrated:
        migrated.execute(f"ALTER TABLE {RETIRED_COLUMN_TABLE} ADD COLUMN not_retired_at_all TEXT")
        migrated.commit()
        proof = _prove(migrated, fresh)

    assert proof.changed_objects == (f"table:{RETIRED_COLUMN_TABLE}",)
    assert proof.matches is False


def test_an_undeclared_extra_table_is_still_unexpected() -> None:
    """The object-grain exemption covers only the declared set."""
    with _fresh_source() as fresh, _fresh_source() as migrated:
        migrated.execute("CREATE TABLE raw_not_retired_at_all (raw_id TEXT PRIMARY KEY) STRICT")
        migrated.commit()
        proof = _prove(migrated, fresh)

    assert proof.unexpected_objects == ("table:raw_not_retired_at_all",)
    assert proof.matches is False


def test_a_retirement_sqlite_cannot_perform_is_a_typed_refusal() -> None:
    """A column another column's CHECK reads cannot be retired, and it says so.

    SQLite refuses the drop, and the capture turns that into a named refusal.
    The alternative -- silently leaving the table unprojected -- would report a
    declaration defect as an ordinary ``changed`` schema object.

    Anti-vacuity: the second half drops the dependent constraint and the same
    declared retirement then projects, so this pins SQLite's own verdict rather
    than a blanket refusal.
    """
    with closing(sqlite3.connect(":memory:")) as conn:
        conn.executescript("CREATE TABLE t (id TEXT PRIMARY KEY, doomed INTEGER, other INTEGER CHECK(other > doomed))")
        with pytest.raises(MigrationError, match="cannot be removed from t"):
            capture_durable_schema_inventory(conn, retired_columns={"t": frozenset({"doomed"})})

        conn.executescript("DROP TABLE t;CREATE TABLE t (id TEXT PRIMARY KEY, doomed INTEGER, other INTEGER)")
        inventory = capture_durable_schema_inventory(conn, retired_columns={"t": frozenset({"doomed"})})
    assert [item.object_ref for item in inventory.objects] == ["table:t"]


@pytest.mark.parametrize(
    "declaration",
    [
        pytest.param(f"ALTER TABLE {RETIRED_COLUMN_TABLE} ADD COLUMN {RETIRED_COLUMN_NAME} INTEGER", id="plain"),
        pytest.param(
            f"ALTER TABLE {RETIRED_COLUMN_TABLE} ADD COLUMN {RETIRED_COLUMN_NAME} INTEGER NOT NULL DEFAULT 0",
            id="not-null-default",
        ),
    ],
)
def test_the_projection_does_not_depend_on_how_the_retired_column_was_declared(declaration: str) -> None:
    """SQLite performs the removal, so the projected text is engine-canonical."""
    with _fresh_source() as fresh, _fresh_source() as migrated:
        migrated.execute(declaration)
        migrated.commit()
        proof = _prove(migrated, fresh)

    assert proof.matches is True


def test_the_unprojected_digest_is_the_one_the_apply_evidence_records(tmp_path: Path) -> None:
    """polylogue-jkoah: the train gates compare parity against unprojected bytes.

    ``prove_durable_change_train``, ``_validate_train_proof`` and
    ``_historical_schema_evidence`` all require
    ``parity.migrated_inventory_sha256 == apply_evidence.post.schema_inventory_sha256``.
    ``apply_durable_change_train`` builds that ``post`` with
    ``capture_durable_database_evidence``, which captures the inventory with no
    retirement projection at all. Pointing the parity side at the projected
    digest makes the two equal only on a tier that retires nothing -- so the
    proof would refuse exactly the tier the retirement exemption exists for.

    The evidence below is the same production pair those gates compare: a
    ``capture_durable_database_evidence`` capture of a real source tier, and a
    ``prove_durable_fresh_ddl_parity`` proof over the same connection.

    Anti-vacuity: the fixture retains a declared retirement at both grains and
    the test *requires* the projected and unprojected digests to differ. On a
    tier retiring nothing they are equal and every assertion here holds both
    before and after the fix. Restoring the projected digest as
    ``migrated_inventory_sha256`` turns the ``post`` comparison red.
    """
    source_path = tmp_path / "source.db"
    # A real file, because ``capture_durable_database_evidence`` binds its
    # evidence to the tier's own archive identity on disk.
    initialize_archive_database(source_path, ArchiveTier.SOURCE)
    with closing(sqlite3.connect(source_path)) as migrated:
        migrated.execute(f"PRAGMA user_version = {SOURCE_TIER_VERSION}")
        # A historical tier that migration carried forward carrying exactly
        # what the retirement declaration says it would carry.
        migrated.execute(f"CREATE TABLE {RETIRED_TABLE_NAME} (census_id TEXT PRIMARY KEY) STRICT")
        migrated.execute(
            f"ALTER TABLE {RETIRED_COLUMN_TABLE} "
            f"ADD COLUMN {RETIRED_COLUMN_NAME} INTEGER NOT NULL DEFAULT 0 CHECK({RETIRED_COLUMN_NAME} >= 0)"
        )
        migrated.commit()
        post = capture_durable_database_evidence(migrated, ArchiveTier.SOURCE)
        with _fresh_source() as fresh:
            proof = _prove(migrated, fresh)

    assert proof.matches is True
    assert proof.unexpected_objects == ()
    assert proof.changed_objects == ()
    # The unprojected digest is the operand the applied-bytes gates compare.
    assert proof.migrated_inventory_sha256 == post.schema_inventory_sha256
    # The projected digest is the operand fresh DDL can equal.
    assert proof.parity_inventory_sha256 == proof.fresh_inventory_sha256
    # The fixture must actually exercise the exemption: on a tier retiring
    # nothing these two are equal and neither assertion above can fail.
    assert proof.parity_inventory_sha256 != proof.migrated_inventory_sha256


def test_a_parity_proof_over_different_bytes_still_fails_the_applied_binding(tmp_path: Path) -> None:
    """The binding the equality provides is not lost by splitting the digests.

    A parity proof captured from a database whose schema is not the one the
    apply recorded must still be unequal to that ``post`` evidence, or the fix
    would have traded a false refusal for a false acceptance.

    Anti-vacuity: making ``migrated_inventory_sha256`` anything other than the
    migrated database's own unprojected inventory -- a constant, the fresh
    digest, the projected digest -- turns this green while the test above stays
    green too, so the pair pins both directions.
    """
    source_path = tmp_path / "source.db"
    # A real file, because ``capture_durable_database_evidence`` binds its
    # evidence to the tier's own archive identity on disk.
    initialize_archive_database(source_path, ArchiveTier.SOURCE)
    with closing(sqlite3.connect(source_path)) as migrated:
        migrated.execute(f"PRAGMA user_version = {SOURCE_TIER_VERSION}")
        migrated.execute(f"CREATE TABLE {RETIRED_TABLE_NAME} (census_id TEXT PRIMARY KEY) STRICT")
        migrated.commit()
        post = capture_durable_database_evidence(migrated, ArchiveTier.SOURCE)
        # The applied file then gains an object the recorded evidence never saw.
        migrated.execute("CREATE TABLE raw_undeclared_drift (id TEXT PRIMARY KEY) STRICT")
        migrated.commit()
        with _fresh_source() as fresh:
            drifted = _prove(migrated, fresh)

    assert drifted.unexpected_objects == ("table:raw_undeclared_drift",)
    assert drifted.matches is False
    assert drifted.migrated_inventory_sha256 != post.schema_inventory_sha256
